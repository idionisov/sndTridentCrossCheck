#!/usr/bin/env python3
import os
import sys
import time
import argparse
import ROOT

# Ensure sndMuTri package root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from snd import DataManager, load_trident_libraries

# Load compiled C++ analysis libraries
load_trident_libraries(REPO_ROOT)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Declare thread-safe C++ processor for RDataFrame
ROOT.gInterpreter.Declare("""
#include "TClonesArray.h"
#include "sndCluster.h"
#include "sndScifiHit.h"
#include "sndRecoTrack.h"
#include "MuFilterHit.h"
#include "ShipMCTrack.h"
#include "TVector3.h"
#include <vector>
#include <unordered_map>
#include <cmath>

namespace DigiRdf {

struct EventSummary {
    double n_scifi_hits{0.0};
    double n_scifi_st1{0.0};
    double n_mufi_hits{0.0};
    double n_tracks{0.0};
    std::vector<double> cluster_sizes;
    std::vector<double> cluster_qdcs;
    std::vector<double> track_chi2;
    std::vector<double> track_slope_xz;
    std::vector<double> track_slope_yz;
    std::vector<double> clean_hit_qdc;
    std::vector<double> clean_hit_dist;
    std::vector<double> clean_hit_weights;
    std::vector<double> station_numbers{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> station_hits{0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> station_weights{1.0, 1.0, 1.0, 1.0, 1.0};
};

EventSummary process_event(
    const TClonesArray* scifiHits,
    const TClonesArray* clusters,
    const TClonesArray* mufiHits,
    const TClonesArray* tracks,
    double event_weight = 1.0
) {
    EventSummary s;
    s.station_weights = {event_weight, event_weight, event_weight, event_weight, event_weight};

    // 1. SciFi Hits
    std::unordered_map<int, double> hit_qdc_map;
    if (scifiHits) {
        int n_sf = scifiHits->GetEntries();
        for (int i = 0; i < n_sf; ++i) {
            auto* h = static_cast<sndScifiHit*>(scifiHits->At(i));
            if (!h || !h->isValid()) continue;
            s.n_scifi_hits += 1.0;
            int st = h->GetStation();
            if (st == 1) s.n_scifi_st1 += 1.0;
            if (st >= 1 && st <= 5) s.station_hits[st - 1] += 1.0;
            hit_qdc_map[h->GetDetectorID()] = h->GetSignal(0);
        }
    }

    // 2. Clusters
    if (clusters) {
        int n_cl = clusters->GetEntries();
        s.cluster_sizes.reserve(n_cl);
        s.cluster_qdcs.reserve(n_cl);
        for (int i = 0; i < n_cl; ++i) {
            auto* cl = static_cast<sndCluster*>(clusters->At(i));
            if (!cl) continue;
            s.cluster_sizes.push_back(static_cast<double>(cl->GetN()));
            double qdc_sum = 0.0;
            int first = cl->GetFirst();
            int n = cl->GetN();
            for (int ch = first; ch < first + n; ++ch) {
                auto it = hit_qdc_map.find(ch);
                if (it != hit_qdc_map.end()) qdc_sum += it->second;
            }
            s.cluster_qdcs.push_back(qdc_sum);
        }
    }

    // 3. MuFilter Hits
    if (mufiHits) {
        int n_mf = mufiHits->GetEntries();
        for (int i = 0; i < n_mf; ++i) {
            auto* h = static_cast<MuFilterHit*>(mufiHits->At(i));
            if (h && h->isValid()) s.n_mufi_hits += 1.0;
        }
    }

    // 4. Tracks
    if (tracks) {
        int n_tr = tracks->GetEntries();
        s.n_tracks = static_cast<double>(n_tr);
        for (int i = 0; i < n_tr; ++i) {
            auto* trk = static_cast<sndRecoTrack*>(tracks->At(i));
            if (!trk || !trk->getTrackFlag()) continue;
            float c2 = trk->getChi2Ndf();
            if (c2 >= 0.0f) {
                s.track_chi2.push_back(static_cast<double>(c2));
                s.track_slope_xz.push_back(static_cast<double>(trk->getSlopeXZ()));
                s.track_slope_yz.push_back(static_cast<double>(trk->getSlopeYZ()));
            }
        }

        // Clean single track response
        if (n_tr == 1 && scifiHits) {
            auto* trk = static_cast<sndRecoTrack*>(tracks->At(0));
            if (trk && trk->getTrackFlag() && trk->getChi2Ndf() >= 0.0f && trk->getChi2Ndf() < 2.0f) {
                TVector3 start = trk->getStart();
                TVector3 mom   = trk->getTrackMom();
                double pz = (std::abs(mom.Z()) > 1e-10) ? mom.Z() : 1e-10;

                int n_sf = scifiHits->GetEntries();
                for (int i = 0; i < n_sf; ++i) {
                    auto* h = static_cast<sndScifiHit*>(scifiHits->At(i));
                    if (!h || !h->isValid()) continue;
                    double qdc = h->GetSignal(0);
                    s.clean_hit_qdc.push_back(qdc);
                    s.clean_hit_weights.push_back(event_weight);

                    int st = h->GetStation();
                    if (st < 1) st = 1;
                    if (st > 5) st = 5;
                    double z_st = 300.0 + (st - 1) * 8.5;
                    double t = (z_st - start.Z()) / pz;
                    double x_pos = start.X() + t * mom.X();
                    double y_pos = start.Y() + t * mom.Y();

                    double dist = h->isVertical() ? std::abs(54.5 - y_pos) : std::abs(-47.5 - x_pos);
                    s.clean_hit_dist.push_back(dist);
                }
            }
        }
    }

    return s;
}

} // namespace DigiRdf
""")

# Configuration: Datasets, Geometries, and Trees
FILES = {
    "Data": {
        "path": "/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-*_muonReco.root",
        "geo": "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root",
        "color": ROOT.kBlack,
        "is_mc": False,
    },
    "SingleMuMC": {
        "path": "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root",
        "geo": "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/geofile_full.Ntuple-TGeant4.root",
        "color": ROOT.kBlue,
        "is_mc": True,
    },
    "ThreeMuMC": {
        "path": "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-2??_trks.root",
        "geo": "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/geofile_full.Ntuple-TGeant4_boost100.0.root",
        "color": ROOT.kRed,
        "is_mc": True,
    },
}

DEFAULT_OUTPUT_PATH = "/eos/user/i/idioniso/sndMuTri/out/digi_validation_comparison.root"


def analyze_sample_rdf(sample_name, cfg, num_threads=8, max_events=-1, weight_scale=40.0):
    print(f"\n=======================================================")
    print(f"--> Initializing DataManager for {sample_name}...")
    print(f"=======================================================")

    t0 = time.time()
    data = DataManager(
        source=cfg["path"],
        geo_path=cfg["geo"],
        num_threads=num_threads,
        init_geo=False,
    )

    print(f"    Loaded {data.num_files} file(s) with tree '{data.tree_name}'. Total entries: {data.entries:,}")

    # Create multi-threaded RDataFrame with automatic progress monitor
    df = data.rdf(progress=True, every_seconds=10.0, progress_label=sample_name)

    # Optional range filtering
    if max_events > 0 and max_events < data.entries:
        if ROOT.IsImplicitMTEnabled():
            df = df.Filter(f"rdfentry_ < {max_events}")
        else:
            df = df.Range(max_events)
        print(f"    Filtered up to {max_events:,} events.")

    # 1. Define event weight
    if not cfg.get("is_mc", False) or sample_name == "Data":
        df = df.Define("weight", "1.0")
    elif data.has_branch("mc_weight"):
        df = df.Define("weight", f"static_cast<double>(mc_weight) * {weight_scale}")
    elif data.has_branch("MCEventHeader.") or data.has_branch("MCEventHeader"):
        header_name = "MCEventHeader." if data.has_branch("MCEventHeader.") else "MCEventHeader"
        df = df.Define("weight", f"static_cast<double>({header_name}.GetWeight()) * {weight_scale}")
    elif data.has_branch("MCTrack"):
        df = df.Define(
            "weight",
            f"(MCTrack.GetEntries() > 0 ? static_cast<double>(static_cast<ShipMCTrack*>(MCTrack.At(0))->GetWeight()) : 1.0) * {weight_scale}"
        )
    else:
        df = df.Define("weight", f"{weight_scale}")

    # 2. Check branch presence and bind C++ processor
    sf_col = "Digi_ScifiHits" if data.has_branch("Digi_ScifiHits") else "nullptr"
    cl_col = "Cluster_Scifi" if data.has_branch("Cluster_Scifi") else "nullptr"
    mf_col = "Digi_MuFilterHits" if data.has_branch("Digi_MuFilterHits") else ("Digi_MuFilterHit" if data.has_branch("Digi_MuFilterHit") else "nullptr")
    tr_col = "Reco_MuonTracks" if data.has_branch("Reco_MuonTracks") else "nullptr"

    cols = []
    args_expr = []
    for col_name in [sf_col, cl_col, mf_col, tr_col]:
        if col_name != "nullptr":
            args_expr.append(f"&{col_name}")
            cols.append(col_name)
        else:
            args_expr.append("static_cast<const TClonesArray*>(nullptr)")

    proc_call = f"DigiRdf::process_event({args_expr[0]}, {args_expr[1]}, {args_expr[2]}, {args_expr[3]}, weight)"
    df = df.Define("ev", proc_call)

    # 3. Book Histograms and Profiles
    models = {
        "qdc_single_hit": ROOT.RDF.TH1DModel(f"qdc_single_{sample_name}", f"Single Hit QDC ({sample_name});QDC [a.u.];Normalized Entries", 150, -10, 140),
        "cluster_qdc": ROOT.RDF.TH1DModel(f"cluster_qdc_{sample_name}", f"Total Cluster QDC ({sample_name});Cluster QDC [a.u.];Normalized Entries", 150, -10, 200),
        "cluster_size": ROOT.RDF.TH1DModel(f"cls_size_{sample_name}", f"Cluster Size ({sample_name});# SiPM channels / cluster;Normalized Fraction", 12, 0.5, 12.5),
        "nhits_st1": ROOT.RDF.TH1DModel(f"nhits_st1_{sample_name}", f"SciFi Hits in Station 1 ({sample_name});# Hits in Station 1;Normalized Entries", 40, -0.5, 39.5),
        "nhits_total": ROOT.RDF.TH1DModel(f"nhits_tot_{sample_name}", f"Total SciFi Hits ({sample_name});# SciFi Hits;Normalized Entries", 100, -0.5, 99.5),
        "qdc_vs_dist": ROOT.RDF.TProfile1DModel(f"qdc_dist_{sample_name}", f"QDC vs Distance to SiPM ({sample_name});Distance to SiPM [cm];<QDC>", 40, 0, 40),
        "nhits_per_station": ROOT.RDF.TProfile1DModel(f"nhits_per_station_{sample_name}", f"SciFi Hit Profile by Station ({sample_name});SciFi Station Number;<Hits / Event>", 5, 0.5, 5.5),
        "ntracks": ROOT.RDF.TH1DModel(f"ntracks_{sample_name}", f"Reconstructed Muon Tracks ({sample_name});# Reco Muon Tracks;Normalized Entries", 6, -0.5, 5.5),
        "track_chi2_ndf": ROOT.RDF.TH1DModel(f"track_chi2_ndf_{sample_name}", f"Track Fit #chi^{{2}} / NDF ({sample_name});#chi^{{2}} / NDF;Normalized Entries", 50, 0.0, 10.0),
        "track_slope_xz": ROOT.RDF.TH1DModel(f"track_slope_xz_{sample_name}", f"Track Slope dx/dz ({sample_name});Slope dx/dz;Normalized Entries", 100, -0.08, 0.08),
        "track_slope_yz": ROOT.RDF.TH1DModel(f"track_slope_yz_{sample_name}", f"Track Slope dy/dz ({sample_name});Slope dy/dz;Normalized Entries", 100, -0.08, 0.08),
        "nhits_mufilter": ROOT.RDF.TH1DModel(f"nhits_mufilter_{sample_name}", f"Total MuFilter Hits ({sample_name});# MuFilter Hits;Normalized Entries", 60, -0.5, 59.5),
    }

    rdf_ptrs = {
        "qdc_single_hit":    df.Histo1D(models["qdc_single_hit"], "ev.clean_hit_qdc", "ev.clean_hit_weights"),
        "cluster_qdc":       df.Histo1D(models["cluster_qdc"], "ev.cluster_qdcs", "weight"),
        "cluster_size":      df.Histo1D(models["cluster_size"], "ev.cluster_sizes", "weight"),
        "nhits_st1":         df.Histo1D(models["nhits_st1"], "ev.n_scifi_st1", "weight"),
        "nhits_total":       df.Histo1D(models["nhits_total"], "ev.n_scifi_hits", "weight"),
        "qdc_vs_dist":       df.Profile1D(models["qdc_vs_dist"], "ev.clean_hit_dist", "ev.clean_hit_qdc", "ev.clean_hit_weights"),
        "nhits_per_station": df.Profile1D(models["nhits_per_station"], "ev.station_numbers", "ev.station_hits", "ev.station_weights"),
        "ntracks":           df.Histo1D(models["ntracks"], "ev.n_tracks", "weight"),
        "track_chi2_ndf":    df.Histo1D(models["track_chi2_ndf"], "ev.track_chi2", "weight"),
        "track_slope_xz":    df.Histo1D(models["track_slope_xz"], "ev.track_slope_xz", "weight"),
        "track_slope_yz":    df.Histo1D(models["track_slope_yz"], "ev.track_slope_yz", "weight"),
        "nhits_mufilter":    df.Histo1D(models["nhits_mufilter"], "ev.n_mufi_hits", "weight"),
    }

    sum_w_ptr = df.Sum("weight")

    # Trigger single-pass multi-threaded computation
    print(f"    Executing parallel event loop across {num_threads} threads...")
    hists = {}
    for key, ptr in rdf_ptrs.items():
        h = ptr.GetValue().Clone()
        h.SetDirectory(0)
        hists[key] = h

    elapsed = time.time() - t0
    total_w = sum_w_ptr.GetValue()
    print(f"    Completed {sample_name} in {elapsed:.1f}s. Total effective weight: {total_w:.3e}")
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
        description="High-performance multi-threaded digitization validation using DataManager and RDataFrame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-j", "--threads", dest="num_threads", type=int, default=8, help="Number of worker threads")
    parser.add_argument("-n", "--max-events", type=int, default=-1, help="Max events per sample (-1 = all entries)")
    parser.add_argument("--lumi", "--L-lhc", dest="lumi_target", type=float, default=1.0, help="Target luminosity in fb^-1")
    parser.add_argument("--L-mc", dest="lumi_mc", type=float, default=0.025, help="MC integrated luminosity in fb^-1")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_PATH, help="Output ROOT file path")

    args = parser.parse_args()

    weight_scale = args.lumi_target / args.lumi_mc if args.lumi_mc > 0 else 1.0
    print(f"Active Multi-Threading Threads: {args.num_threads}")
    print(f"Weight scale (L_target / L_mc): {args.lumi_target:.3f} / {args.lumi_mc:.3f} = {weight_scale:.2f}")

    # Process all datasets in parallel with RDataFrame
    results = {}
    for name, cfg in FILES.items():
        results[name] = analyze_sample_rdf(
            name, cfg,
            num_threads=args.num_threads,
            max_events=args.max_events,
            weight_scale=weight_scale,
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

    # Store individual histograms by sample
    for name in results:
        sub_dir = f_out.mkdir(name)
        sub_dir.cd()
        for h_name, h in results[name].items():
            h.Write()
        f_out.cd()

    f_out.Close()
    print(f"\nCanvases and histograms successfully written to: {args.output}")

    # Export PNG images
    png_path_c1 = args.output.replace(".root", "_scifi_digi.png")
    png_path_c2 = args.output.replace(".root", "_tracking_multiplicity.png")
    c1.SaveAs(png_path_c1)
    c2.SaveAs(png_path_c2)
    print(f"Exported PNG plots to:\n  - {png_path_c1}\n  - {png_path_c2}")


if __name__ == "__main__":
    main()
