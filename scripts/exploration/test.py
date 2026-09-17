#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import ROOT
import SndlhcGeo

# Ensure sndMuTri package root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from snd import load_trident_libraries
    load_trident_libraries(REPO_ROOT)
except Exception as e:
    print(f"Notice: could not load trident C++ libraries ({e})")

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Configuration: Datasets, Geometries, and Trees
FILES = {
    "Data": {
        "path": "/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-1_1?_*.root",
        "geo": "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root",
        "tree": "rawConv",
        "color": ROOT.kBlack,
        "is_mc": False,
    },
    "SingleMuMC": {
        "path": "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root",
        "geo": "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/geofile_full.Ntuple-TGeant4.root",
        "tree": "cbmsim",
        "color": ROOT.kBlue,
        "is_mc": True,
    },
    "ThreeMuMC": {
        "path": "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-2??_trks.root",
        "geo": "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/geofile_full.Ntuple-TGeant4_boost100.0.root",
        "tree": "cbmsim",
        "color": ROOT.kRed,
        "is_mc": True,
    },
}

DEFAULT_OUTPUT_PATH = "/eos/user/i/idioniso/sndMuTri/out/digi_validation_comparison.root"


def extrapolate_track_to_z(track, z):
    pos = track.getStart()
    mom = track.getTrackMom()

    pz = mom.Z()
    if abs(pz) < 1e-10:
        pz = 1e-10 if pz >= 0 else -1e-10

    t = (z - pos.Z()) / pz
    return ROOT.TVector3(
        pos.X() + t * mom.X(),
        pos.Y() + t * mom.Y(),
        z
    )


def extract_event_weight(chain, sample_name, cfg, weight_scale, truth_processor=None):
    """
    Extract the physical event weight consistent with summarize_trident_yields.py:
    - Data: weight = 1.0
    - Precomputed mc_weight: mc_weight * weight_scale
    - truth_info struct: scaledWeight
    - TridentTruthProcessor on MCTrack: scaledWeight (or mcWeight * weight_scale)
    - MCEventHeader / MCTrack[0]: generator weight * weight_scale
    """
    if not cfg.get("is_mc", False) or sample_name == "Data":
        return 1.0

    # 1. Check precomputed mc_weight branch
    if hasattr(chain, "mc_weight"):
        try:
            return float(chain.mc_weight) * weight_scale
        except Exception:
            pass

    # 2. Check truth_info branch
    if hasattr(chain, "truth_info"):
        try:
            return float(chain.truth_info.scaledWeight)
        except Exception:
            pass

    # 3. For ThreeMuMC, evaluate truth processor on MCTrack if candidate exists
    if sample_name == "ThreeMuMC" and truth_processor is not None and hasattr(chain, "MCTrack"):
        try:
            truth = truth_processor.process(chain.MCTrack)
            if truth.hasCandidate:
                return float(truth.scaledWeight)
            elif truth.mcWeight > 0:
                return float(truth.mcWeight) * weight_scale
        except Exception:
            pass

    # 4. Fallback to MCEventHeader weight
    if hasattr(chain, "MCEventHeader"):
        try:
            w = float(chain.MCEventHeader.GetWeight())
            if w > 0:
                return w * weight_scale
        except Exception:
            pass

    # 5. Fallback to primary track generator weight (standard for PassingMuon MC)
    if hasattr(chain, "MCTrack") and chain.MCTrack.GetEntries() > 0:
        try:
            w = float(chain.MCTrack[0].GetWeight())
            if w > 0:
                return w * weight_scale
        except Exception:
            pass

    return 1.0 * weight_scale


def analyze_sample(sample_name, cfg, max_events=50000, weight_scale=40.0, target_z_range=(260.0, 355.0)):
    print(f"--> Processing {sample_name}...")

    # Load geometry
    snd_geo = SndlhcGeo.GeoInterface(cfg["geo"])
    scifi = snd_geo.modules["Scifi"]

    # Select tree name from cfg, default to 'cbmsim' if omitted
    tree_name = cfg.get("tree", "cbmsim")
    chain = ROOT.TChain(tree_name)
    n_files = chain.Add(cfg["path"])
    print(f"    Loaded tree '{tree_name}' from {n_files} file(s) matching '{cfg['path']}'")

    # Set up TridentTruthProcessor for ThreeMuMC if MCTrack is used
    truth_processor = None
    if sample_name == "ThreeMuMC":
        try:
            config = ROOT.snd.trident.TridentTruthConfig()
            config.targetZMin = target_z_range[0]
            config.targetZMax = target_z_range[1]
            config.weightScale = weight_scale
            config.zMin = -99999.0
            config.zMax = 99999.0
            config.useFiducial = False
            config.processMask = 7
            truth_processor = ROOT.snd.trident.TridentTruthProcessor(config)
        except Exception as e:
            print(f"    Note: TridentTruthProcessor not available ({e}), using branch weights.")

    # Book Histograms
    hists = {
        # --- SciFi Hits & Clusters ---
        "qdc_single_hit": ROOT.TH1F(
            f"qdc_single_{sample_name}",
            f"Single Hit QDC ({sample_name});QDC [a.u.];Normalized Entries",
            150, -10, 140,
        ),
        "cluster_qdc": ROOT.TH1F(
            f"cluster_qdc_{sample_name}",
            f"Total Cluster QDC ({sample_name});Cluster QDC [a.u.];Normalized Entries",
            150, -10, 200,
        ),
        "cluster_size": ROOT.TH1F(
            f"cls_size_{sample_name}",
            f"Cluster Size ({sample_name});# SiPM channels / cluster;Normalized Fraction",
            12, 0.5, 12.5,
        ),
        "nhits_st1": ROOT.TH1F(
            f"nhits_st1_{sample_name}",
            f"SciFi Hits in Station 1 ({sample_name});# Hits in Station 1;Normalized Entries",
            40, -0.5, 39.5,
        ),
        "nhits_total": ROOT.TH1F(
            f"nhits_tot_{sample_name}",
            f"Total SciFi Hits ({sample_name});# SciFi Hits;Normalized Entries",
            100, -0.5, 99.5,
        ),
        "qdc_vs_dist": ROOT.TProfile(
            f"qdc_dist_{sample_name}",
            f"QDC vs Distance to SiPM ({sample_name});Distance to SiPM [cm];<QDC>",
            40, 0, 40,
        ),
        "nhits_per_station": ROOT.TProfile(
            f"nhits_per_station_{sample_name}",
            f"SciFi Hit Profile by Station ({sample_name});SciFi Station Number;<Hits / Event>",
            5, 0.5, 5.5,
        ),
        # --- Tracking & Multiplicity ---
        "ntracks": ROOT.TH1F(
            f"ntracks_{sample_name}",
            f"Reconstructed Muon Tracks ({sample_name});# Reco Muon Tracks;Normalized Entries",
            6, -0.5, 5.5,
        ),
        "track_chi2_ndf": ROOT.TH1F(
            f"track_chi2_ndf_{sample_name}",
            f"Track Fit #chi^{{2}} / NDF ({sample_name});#chi^{{2}} / NDF;Normalized Entries",
            50, 0.0, 10.0,
        ),
        "track_slope_xz": ROOT.TH1F(
            f"track_slope_xz_{sample_name}",
            f"Track Slope dx/dz ({sample_name});Slope dx/dz;Normalized Entries",
            100, -0.08, 0.08,
        ),
        "track_slope_yz": ROOT.TH1F(
            f"track_slope_yz_{sample_name}",
            f"Track Slope dy/dz ({sample_name});Slope dy/dz;Normalized Entries",
            100, -0.08, 0.08,
        ),
        "nhits_mufilter": ROOT.TH1F(
            f"nhits_mufilter_{sample_name}",
            f"Total MuFilter Hits ({sample_name});# MuFilter Hits;Normalized Entries",
            60, -0.5, 59.5,
        ),
    }

    # Enable Sumw2 for accurate weighted uncertainties and detach from gDirectory
    for h in hists.values():
        if hasattr(h, "Sumw2"):
            h.Sumw2()
        h.SetDirectory(0)

    A, B = ROOT.TVector3(), ROOT.TVector3()
    total_entries = chain.GetEntries()
    n_ev = min(max_events, total_entries)
    print(f"    Processing {n_ev} / {total_entries} events with mc_weight...")

    sum_weights = 0.0
    for i in range(n_ev):
        chain.GetEntry(i)

        w = extract_event_weight(chain, sample_name, cfg, weight_scale, truth_processor)
        sum_weights += w

        # 1. Total and Per-Station SciFi Hits
        station_hits = [0] * 6  # 1-indexed for stations 1..5
        hit_dict = {}
        if hasattr(chain, "Digi_ScifiHits"):
            nHitsTotal = 0
            for idx, hit in enumerate(chain.Digi_ScifiHits):
                if not hit.isValid():
                    continue
                nHitsTotal += 1
                st = hit.GetStation()
                if 1 <= st <= 5:
                    station_hits[st] += 1
                hit_dict[hit.GetDetectorID()] = idx

            hists["nhits_total"].Fill(nHitsTotal, w)
            hists["nhits_st1"].Fill(station_hits[1], w)
            for st in range(1, 6):
                hists["nhits_per_station"].Fill(st, station_hits[st], w)

        # 2. Clusters: Size and Cluster QDC
        if hasattr(chain, "Cluster_Scifi"):
            for cl in chain.Cluster_Scifi:
                hists["cluster_size"].Fill(cl.GetN(), w)

                cl_qdc = 0.0
                first = cl.GetFirst()
                for c_id in range(first, first + cl.GetN()):
                    if c_id in hit_dict:
                        cl_qdc += chain.Digi_ScifiHits[hit_dict[c_id]].GetSignal(0)
                hists["cluster_qdc"].Fill(cl_qdc, w)

        # 3. MuFilter Hit Multiplicity
        if hasattr(chain, "Digi_MuFilterHits"):
            nMufiHits = sum(1 for mf_hit in chain.Digi_MuFilterHits if mf_hit.isValid())
            hists["nhits_mufilter"].Fill(nMufiHits, w)

        # 4. Tracking Diagnostics
        if hasattr(chain, "Reco_MuonTracks"):
            n_tracks = chain.Reco_MuonTracks.GetEntries()
            hists["ntracks"].Fill(n_tracks, w)

            for trk in chain.Reco_MuonTracks:
                if trk.getTrackFlag():
                    hists["track_chi2_ndf"].Fill(trk.getChi2Ndf(), w)
                    hists["track_slope_xz"].Fill(trk.getSlopeXZ(), w)
                    hists["track_slope_yz"].Fill(trk.getSlopeYZ(), w)

            # Clean Single Track Selection for single-hit response and attenuation
            if n_tracks == 1 and hasattr(chain, "Digi_ScifiHits"):
                track = chain.Reco_MuonTracks[0]
                if track.getTrackFlag() and track.getChi2Ndf() < 2.0:
                    for hit in chain.Digi_ScifiHits:
                        if not hit.isValid():
                            continue
                        qdc = hit.GetSignal(0)
                        hists["qdc_single_hit"].Fill(qdc, w)

                        # Distance along fiber to readout SiPM
                        scifi.GetSiPMPosition(hit.GetDetectorID(), A, B)
                        z_hit = (A.Z() + B.Z()) / 2.0
                        track_pos = extrapolate_track_to_z(track, z_hit)

                        if hit.isVertical():
                            dist = abs(B.Y() - track_pos.Y())
                        else:
                            dist = abs(A.X() - track_pos.X())

                        hists["qdc_vs_dist"].Fill(dist, qdc, w)

    print(f"    Completed {sample_name}. Total effective event weight: {sum_weights:.3e}")
    return hists


def draw_pad(pad_obj, var, var_label, logy, norm, results, files_cfg):
    pad_obj.cd()
    leg = ROOT.TLegend(0.62, 0.65, 0.89, 0.89)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.04)

    is_profile = "dist" in var or "per_station" in var
    opt = "E1" if is_profile else "HIST"
    first = True
    drawn_count = 0

    for name in ["Data", "SingleMuMC", "ThreeMuMC"]:
        if name not in results or var not in results[name]:
            continue
        h = results[name][var]
        if h.GetEntries() == 0 or (norm and h.Integral() <= 0):
            continue

        h.SetLineColor(files_cfg[name]["color"])
        h.SetLineWidth(2)
        if is_profile:
            h.SetMarkerColor(files_cfg[name]["color"])
            h.SetMarkerStyle(20)
            h.SetMarkerSize(0.8)

        if norm and not is_profile and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        if logy:
            h.SetMinimum(1e-4)

        draw_opt = opt if first else (opt + " SAME" if opt else "SAME")
        h.Draw(draw_opt)
        leg.AddEntry(h, name, "lep" if is_profile else "l")
        first = False
        drawn_count += 1

    if logy and drawn_count > 0:
        pad_obj.SetLogy(1)

    if drawn_count > 0:
        leg.Draw()

    return leg


def main():
    parser = argparse.ArgumentParser(
        description="Digitization validation comparing Data, SingleMuMC, and ThreeMuMC using physical event weights."
    )
    parser.add_argument("-n", "--max-events", type=int, default=30000, help="Max events per sample (default: 30000)")
    parser.add_argument("--lumi", "--L-lhc", dest="lumi_target", type=float, default=1.0, help="Target luminosity in fb^-1 (default: 1.0)")
    parser.add_argument("--L-mc", dest="lumi_mc", type=float, default=0.025, help="MC integrated luminosity in fb^-1 (default: 0.025 = 1/40 fb^-1)")
    parser.add_argument("--z-range", dest="target_z_range", nargs=2, type=float, default=[260.0, 355.0], help="Target Z range in cm (default: 260.0 355.0)")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_PATH, help="Output ROOT file path")

    args = parser.parse_args()

    weight_scale = args.lumi_target / args.lumi_mc if args.lumi_mc > 0 else 1.0
    print(f"Weight scale (L_target / L_mc): {args.lumi_target:.3f} / {args.lumi_mc:.3f} = {weight_scale:.2f}")

    # Run analysis for all datasets
    results = {}
    for name, cfg in FILES.items():
        results[name] = analyze_sample(
            name, cfg,
            max_events=args.max_events,
            weight_scale=weight_scale,
            target_z_range=args.target_z_range,
        )

    # --- Canvas 1: SciFi Hit & Cluster Digitization ---
    c1 = ROOT.TCanvas("c_scifi_digi", "SciFi Hit and Cluster Digitization", 1500, 950)
    ROOT.SetOwnership(c1, False)
    c1.Divide(3, 2)

    plots_c1 = [
        ("qdc_single_hit", "Single Hit QDC", True, True),
        ("cluster_qdc", "Total Cluster QDC", True, True),
        ("cluster_size", "Cluster Size", True, True),
        ("qdc_vs_dist", "QDC vs Distance to SiPM", False, False),
        ("nhits_st1", "Hits in Station 1", True, True),
        ("nhits_total", "Total SciFi Hits", True, True),
    ]

    legends_c1 = []
    for pad_id, (var, label, logy, norm) in enumerate(plots_c1, start=1):
        pad_obj = c1.cd(pad_id)
        pad_obj.SetRightMargin(0.06)
        pad_obj.SetLeftMargin(0.12)
        leg = draw_pad(pad_obj, var, label, logy, norm, results, FILES)
        legends_c1.append(leg)

    c1.Update()

    # --- Canvas 2: Tracking, Occupancy & MuFilter ---
    c2 = ROOT.TCanvas("c_tracking_multiplicity", "Tracking and Detector Multiplicity", 1500, 950)
    ROOT.SetOwnership(c2, False)
    c2.Divide(3, 2)

    plots_c2 = [
        ("nhits_per_station", "Hit Profile by Station", False, False),
        ("ntracks", "Reconstructed Tracks", True, True),
        ("track_chi2_ndf", "Track Fit Chi2 / NDF", True, True),
        ("track_slope_xz", "Track Slope dx/dz", True, True),
        ("track_slope_yz", "Track Slope dy/dz", True, True),
        ("nhits_mufilter", "Total MuFilter Hits", True, True),
    ]

    legends_c2 = []
    for pad_id, (var, label, logy, norm) in enumerate(plots_c2, start=1):
        pad_obj = c2.cd(pad_id)
        pad_obj.SetRightMargin(0.06)
        pad_obj.SetLeftMargin(0.12)
        leg = draw_pad(pad_obj, var, label, logy, norm, results, FILES)
        legends_c2.append(leg)

    c2.Update()

    # Save output to ROOT file
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    f_out = ROOT.TFile.Open(args.output, "RECREATE")
    f_out.cd()
    c1.Write()
    c2.Write()

    # Also store individual histograms organized by sample directory
    for name in results:
        sub_dir = f_out.mkdir(name)
        sub_dir.cd()
        for h_name, h in results[name].items():
            h.Write()
        f_out.cd()

    f_out.Close()
    print(f"\nCanvases and histograms successfully written to: {args.output}")

    # Export PNG images for quick visual inspection
    png_path_c1 = args.output.replace(".root", "_scifi_digi.png")
    png_path_c2 = args.output.replace(".root", "_tracking_multiplicity.png")
    c1.SaveAs(png_path_c1)
    c2.SaveAs(png_path_c2)
    print(f"Exported PNG plots to:\n  - {png_path_c1}\n  - {png_path_c2}")


if __name__ == "__main__":
    main()
