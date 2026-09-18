#!/usr/bin/env python3
"""
Digitization & Reconstruction Validation Suite for SND@LHC

Compares detector response and event observables between:
  1. Collision / Muon Beam Data (run 8329 / 6640)
  2. Single Passing Muon MC (SingleMuMC / FLUKA)
  3. Trident Signal MC (ThreeMuMC / MG5+Pythia8)

Features:
  - Exact SiPM channel position geometry cache using Scifi::GetSiPMPosition(detID, A, B)
  - Track-to-channel distance filtering using sndRecoTrack track parameters & channel geometry
  - Associated hits: SciFi and Veto close to SciFi track; US and DS close to DS track
  - Dedicated distributions of track-to-channel / bar distances in a separate TDirectory
  - Accurate Monte Carlo event weighting (FLUKA generator weight & trident scaled weight)
  - Comprehensive 1D histograms, TProfiles, and 2D correlation plots
  - Publication-quality overlays and standalone figures
"""

import os
import sys

# Ensure grid / CA certificates are discoverable by XRootD
if "X509_CERT_DIR" not in os.environ:
    for cert_dir in ["/cvmfs/grid.cern.ch/etc/grid-security/certificates", "/etc/pki/tls/certs"]:
        if os.path.isdir(cert_dir):
            os.environ["X509_CERT_DIR"] = cert_dir
            break

import math
import glob
import time
import argparse
import ROOT

# Force unbuffered / line-buffered stdout
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from snd import load_trident_libraries
import SndlhcGeo

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kBird)

# Default dataset configurations
DEFAULT_DATA_PATTERN = "/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-1_1?_*.root"
DEFAULT_DATA_GEO = "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root"

DEFAULT_PMU_PATH = "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root"
DEFAULT_PMU_GEO = "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/geofile_full.Ntuple-TGeant4.root"

DEFAULT_TRI_PATTERN = "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-2??_trks.root"
DEFAULT_TRI_GEO = "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/geofile_full.Ntuple-TGeant4_boost100.0.root"


def resolve_files(pattern: str, max_files: int = 50) -> list:
    """Resolve file paths matching a pattern with fast validation."""
    if not pattern:
        return []
    if "*" in pattern or "?" in pattern or "[" in pattern:
        matched = sorted(glob.glob(pattern))
    elif os.path.exists(pattern):
        matched = [pattern]
    else:
        matched = []

    valid = []
    for p in matched:
        try:
            if os.path.getsize(p) > 1024:
                valid.append(p)
        except OSError:
            pass
        if len(valid) >= max_files:
            break
    return valid


def analyze_sample(sample_name, config, args):
    """Process a single dataset using ROOT RDataFrame and DigiValidationProcessor."""
    t0 = time.time()
    print(f"\n{'='*75}")
    print(f"[*] Processing Dataset: {sample_name}")
    print(f"{'='*75}")

    # Estimate required files based on max_events
    avg_per_file = config.get("avg_events_per_file", 25000)
    if args.max_events and args.max_events > 0:
        needed_files = max(1, int(math.ceil(args.max_events / avg_per_file)) + 1)
        max_f = min(config.get("max_files", 50), needed_files)
    else:
        max_f = config.get("max_files", 50)

    files = resolve_files(config["path"], max_files=max_f)
    if not files and "alt_path" in config:
        files = resolve_files(config["alt_path"], max_files=max_f)

    if not files:
        print(f"[-] ERROR: No input files found for {sample_name} at '{config['path']}'")
        return None

    print(f"[*] Selected {len(files)} file(s) for processing.")
    print(f"[*] Loading geometry: {config['geo']}")
    snd_geo = SndlhcGeo.GeoInterface(config["geo"])
    scifi_det = snd_geo.modules["Scifi"]
    mufi_det  = snd_geo.modules["MuFilter"]

    print("[*] Initializing DigiValidationProcessor with track-hit distance cuts...")
    val_cfg = ROOT.snd.trident.DigiValidationConfig()
    val_cfg.scifi_max_dist = args.scifi_max_dist
    val_cfg.veto_max_dist  = args.veto_max_dist
    val_cfg.us_max_dist    = args.us_max_dist
    val_cfg.ds_max_dist    = args.ds_max_dist
    print(f"[*] Configured cuts: SciFi <= {args.scifi_max_dist*10:.2f} mm | Veto <= {args.veto_max_dist:.1f} cm | US <= {args.us_max_dist:.1f} cm | DS <= {args.ds_max_dist*10:.2f} mm")

    proc = ROOT.snd.trident.DigiValidationProcessor(scifi_det, mufi_det, val_cfg)
    n_cached = proc.getCachedChannelCount()
    n_mufi_cached = proc.getCachedMufiChannelCount()
    print(f"[*] Pre-cached {n_cached:,} SciFi channels and {n_mufi_cached:,} MuFilter bars.")

    tree_name = config["tree"]
    chain = ROOT.TChain(tree_name)
    for f in files:
        chain.Add(f)

    df = ROOT.RDataFrame(chain)
    if args.max_events and args.max_events > 0:
        df = df.Range(args.max_events)
        print(f"[*] Processing limited to {args.max_events:,} events.")

    # -------------------------------------------------------------
    # Monte Carlo Event Weighting
    # -------------------------------------------------------------
    if not config.get("is_mc", False):
        df = df.Define("weight", "1.0")
        print("[*] Weight model: Unweighted Data (weight = 1.0)")
    elif sample_name == "SingleMuMC":
        scale_pmu = args.scale_pmu if args.scale_pmu else (args.lumi_target * 8e5 / 1213000.0)
        truth_cfg = ROOT.snd.trident.PassingMuonTruthConfig()
        pmu_proc = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)
        df = (
            df
            .Define("truth", pmu_proc, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
            .Define("weight", f"truth.mc_weight * {scale_pmu}")
        )
        print(f"[*] Weight model: FLUKA primary muon weight * scale ({scale_pmu:.4e})")
    elif sample_name == "ThreeMuMC":
        weight_scale = args.lumi_target / args.lumi_mc_tri if args.lumi_mc_tri > 0 else 1.0
        tri_cfg = ROOT.snd.trident.TridentTruthConfig()
        tri_cfg.weightScale = weight_scale
        tri_proc = ROOT.snd.trident.TridentTruthProcessor(tri_cfg)
        df = (
            df
            .Define("truth", tri_proc, ["MCTrack"])
            .Define("weight", "truth.scaledWeight")
        )
        print(f"[*] Weight model: Trident truth weight (scale = {weight_scale:.4f})")
    else:
        df = df.Define("weight", "1.0")

    # -------------------------------------------------------------
    # Apply DigiValidationProcessor
    # -------------------------------------------------------------
    df = (
        df
        .Define("ev", proc, ["Digi_ScifiHits", "Cluster_Scifi", "Digi_MuFilterHits", "Reco_MuonTracks"])
        .Define("n_scifi_hits", "ev.n_scifi_hits")
        .Define("n_scifi_hits_all", "ev.n_scifi_hits_all")
        .Define("n_scifi_st1", "ev.n_scifi_st1")
        .Define("scifi_sum_qdc", "ev.scifi_sum_qdc")
        .Define("scifi_mean_qdc", "ev.scifi_mean_qdc")
        .Define("n_clusters", "ev.n_scifi_clusters")
        .Define("cluster_size", "ev.cluster_size")
        .Define("cluster_qdc", "ev.cluster_qdc")
        .Define("n_mufi_hits", "ev.n_mufi_hits")
        .Define("n_mufi_hits_all", "ev.n_mufi_hits_all")
        .Define("n_mufi_veto_hits", "ev.n_mufi_veto_hits")
        .Define("n_mufi_us_hits", "ev.n_mufi_us_hits")
        .Define("n_mufi_ds_hits", "ev.n_mufi_ds_hits")
        .Define("mufi_sum_qdc", "ev.mufi_sum_qdc")
        .Define("n_tracks", "ev.n_tracks")
        .Define("has_scifi_track", "ev.has_scifi_track")
        .Define("has_ds_track", "ev.has_ds_track")
        .Define("has_both_tracks", "ev.has_both_tracks")
        .Define("track_chi2", "ev.track_chi2")
        .Define("track_slope_xz", "ev.track_slope_xz")
        .Define("track_slope_yz", "ev.track_slope_yz")
        .Define("has_clean_track", "ev.has_clean_track")
        .Define("track_chi2_clean", "ev.track_chi2_ndf_clean")
        .Define("track_slope_xz_clean", "ev.track_slope_xz_clean")
        .Define("track_slope_yz_clean", "ev.track_slope_yz_clean")
        .Define("track_x_300", "ev.track_x_300")
        .Define("track_y_300", "ev.track_y_300")
        .Define("clean_hit_qdc", "ev.hit_qdc")
        .Define("clean_hit_dist", "ev.hit_distance")
        .Define("clean_hit_qdc_horiz", "ev.hit_qdc_horiz")
        .Define("clean_hit_dist_horiz", "ev.hit_dist_horiz")
        .Define("clean_hit_qdc_vert", "ev.hit_qdc_vert")
        .Define("clean_hit_dist_vert", "ev.hit_dist_vert")
        .Define("station_numbers", "ev.station_numbers")
        .Define("station_hits", "ev.station_hits")
        .Define("station_qdc", "ev.station_qdc")
        .Define("dist_scifi", "ev.dist_scifi")
        .Define("dist_scifi_horiz", "ev.dist_scifi_horiz")
        .Define("dist_scifi_vert", "ev.dist_scifi_vert")
        .Define("doca_scifi", "ev.doca_scifi")
        .Define("dist_veto", "ev.dist_veto")
        .Define("doca_veto", "ev.doca_veto")
        .Define("dist_us", "ev.dist_us")
        .Define("doca_us", "ev.doca_us")
        .Define("dist_ds", "ev.dist_ds")
        .Define("doca_ds", "ev.doca_ds")
    )

    df_clean = df.Filter("has_clean_track")

    # -------------------------------------------------------------
    # Book 1D Histograms & Profiles
    # -------------------------------------------------------------
    histograms = {}

    # Hit QDC along clean tracks (only hits passing distance cut)
    histograms["h_qdc_single_hit"] = df_clean.Histo1D(
        (f"h_qdc_{sample_name}", f"{sample_name} Single Hit QDC;Hit QDC [a.u.];Normalized Entries", 150, -10.0, 140.0),
        "clean_hit_qdc", "weight"
    )
    histograms["h_qdc_horiz"] = df_clean.Histo1D(
        (f"h_qdc_h_{sample_name}", f"{sample_name} Horizontal Hit QDC;Hit QDC [a.u.];Normalized Entries", 150, -10.0, 140.0),
        "clean_hit_qdc_horiz", "weight"
    )
    histograms["h_qdc_vert"] = df_clean.Histo1D(
        (f"h_qdc_v_{sample_name}", f"{sample_name} Vertical Hit QDC;Hit QDC [a.u.];Normalized Entries", 150, -10.0, 140.0),
        "clean_hit_qdc_vert", "weight"
    )

    # Clusters (only track-associated)
    histograms["h_cluster_qdc"] = df.Histo1D(
        (f"h_cl_qdc_{sample_name}", f"{sample_name} Cluster Integrated QDC;Cluster QDC [a.u.];Normalized Entries", 105, -10.0, 200.0),
        "cluster_qdc", "weight"
    )
    histograms["h_cluster_size"] = df.Histo1D(
        (f"h_cl_sz_{sample_name}", f"{sample_name} Cluster Size;Cluster Size [channels];Normalized Entries", 12, 0.5, 12.5),
        "cluster_size", "weight"
    )

    # SciFi Tracker Multiplicities (track-associated hits)
    histograms["h_nhits_st1"] = df.Histo1D(
        (f"h_st1_{sample_name}", f"{sample_name} Station 1 Hits;SciFi Station 1 Hits;Normalized Entries", 40, -0.5, 39.5),
        "n_scifi_st1", "weight"
    )
    histograms["h_nhits_total"] = df.Histo1D(
        (f"h_sf_tot_{sample_name}", f"{sample_name} Total SciFi Hits (track-associated);Track SciFi Hits;Normalized Entries", 50, -0.5, 99.5),
        "n_scifi_hits", "weight"
    )
    histograms["h_nhits_all_total"] = df.Histo1D(
        (f"h_sf_all_tot_{sample_name}", f"{sample_name} Total Raw SciFi Hits;Raw SciFi Hits;Normalized Entries", 50, -0.5, 99.5),
        "n_scifi_hits_all", "weight"
    )
    histograms["h_scifi_mean_qdc"] = df.Histo1D(
        (f"h_sf_mqdc_{sample_name}", f"{sample_name} SciFi Mean QDC;Mean SciFi Hit QDC [a.u.];Normalized Entries", 50, 0.0, 100.0),
        "scifi_mean_qdc", "weight"
    )

    # MuFilter Multiplicities (track-associated hits)
    histograms["h_nhits_mufi"] = df.Histo1D(
        (f"h_mf_tot_{sample_name}", f"{sample_name} Total MuFilter Hits (track-associated);Track MuFilter Hits;Normalized Entries", 60, -0.5, 59.5),
        "n_mufi_hits", "weight"
    )
    histograms["h_nhits_veto"] = df.Histo1D(
        (f"h_mf_veto_{sample_name}", f"{sample_name} Veto MuFilter Hits (track-associated);Veto Hits;Normalized Entries", 20, -0.5, 19.5),
        "n_mufi_veto_hits", "weight"
    )
    histograms["h_nhits_us"] = df.Histo1D(
        (f"h_mf_us_{sample_name}", f"{sample_name} US MuFilter Hits (track-associated);US MuFilter Hits;Normalized Entries", 40, -0.5, 39.5),
        "n_mufi_us_hits", "weight"
    )
    histograms["h_nhits_ds"] = df.Histo1D(
        (f"h_mf_ds_{sample_name}", f"{sample_name} DS MuFilter Hits (track-associated);DS MuFilter Hits;Normalized Entries", 40, -0.5, 39.5),
        "n_mufi_ds_hits", "weight"
    )

    # Tracking Observables
    histograms["h_ntracks"] = df.Histo1D(
        (f"h_ntrk_{sample_name}", f"{sample_name} Reconstructed Tracks;Reconstructed Tracks;Normalized Entries", 6, -0.5, 5.5),
        "n_tracks", "weight"
    )
    histograms["h_track_chi2_ndf"] = df.Histo1D(
        (f"h_chi2_{sample_name}", f"{sample_name} Track #chi^{{2}}/NDF;Track #chi^{{2}}/NDF;Normalized Entries", 50, 0.0, 10.0),
        "track_chi2", "weight"
    )
    histograms["h_track_slope_xz"] = df.Histo1D(
        (f"h_slpxz_{sample_name}", f"{sample_name} Track Slope dx/dz;Track Slope dx/dz;Normalized Entries", 50, -0.08, 0.08),
        "track_slope_xz", "weight"
    )
    histograms["h_track_slope_yz"] = df.Histo1D(
        (f"h_slpyz_{sample_name}", f"{sample_name} Track Slope dy/dz;Track Slope dy/dz;Normalized Entries", 50, -0.08, 0.08),
        "track_slope_yz", "weight"
    )

    # TProfiles: Hit Multiplicity & QDC vs Station
    histograms["p_nhits_per_station"] = df.Profile1D(
        (f"p_sthits_{sample_name}", f"{sample_name} Hits / Station;SciFi Station;Mean Hits", 5, 0.5, 5.5),
        "station_numbers", "station_hits", "weight"
    )
    histograms["p_qdc_per_station"] = df.Profile1D(
        (f"p_stqdc_{sample_name}", f"{sample_name} QDC / Station;SciFi Station;Mean Integrated QDC [a.u.]", 5, 0.5, 5.5),
        "station_numbers", "station_qdc", "weight"
    )
    histograms["p_qdc_vs_dist"] = df_clean.Profile1D(
        (f"p_qdcdist_{sample_name}", f"{sample_name} #LTQDC#GT vs Dist to SiPM;Distance to SiPM [cm];#LTQDC#GT [a.u.]", 45, 0.0, 45.0),
        "clean_hit_dist", "clean_hit_qdc", "weight"
    )
    histograms["p_qdc_dist_horiz"] = df_clean.Profile1D(
        (f"p_qdcdist_h_{sample_name}", f"{sample_name} Horizontal #LTQDC#GT vs Dist;Distance to SiPM [cm];#LTQDC#GT [a.u.]", 45, 0.0, 45.0),
        "clean_hit_dist_horiz", "clean_hit_qdc_horiz", "weight"
    )
    histograms["p_qdc_dist_vert"] = df_clean.Profile1D(
        (f"p_qdcdist_v_{sample_name}", f"{sample_name} Vertical #LTQDC#GT vs Dist;Distance to SiPM [cm];#LTQDC#GT [a.u.]", 45, 0.0, 45.0),
        "clean_hit_dist_vert", "clean_hit_qdc_vert", "weight"
    )

    # -------------------------------------------------------------
    # Track to Channel / Bar Distance Histograms
    # -------------------------------------------------------------
    # SciFi Track -> SciFi Channels
    histograms["h_dist_scifi"] = df.Histo1D(
        (f"h_dist_sf_{sample_name}", f"{sample_name} SciFi Track-Channel Distance;1D Distance to SciFi Track [cm];Normalized Entries", 100, 0.0, 1.0),
        "dist_scifi", "weight"
    )
    histograms["h_dist_scifi_zoom"] = df.Histo1D(
        (f"h_dist_sf_zoom_{sample_name}", f"{sample_name} SciFi Track-Channel Distance (Zoom);1D Distance to SciFi Track [cm];Normalized Entries", 100, 0.0, 0.2),
        "dist_scifi", "weight"
    )
    histograms["h_dist_scifi_h"] = df.Histo1D(
        (f"h_dist_sf_h_{sample_name}", f"{sample_name} SciFi Track-Channel Dist (Horiz);1D Distance to SciFi Track [cm];Normalized Entries", 100, 0.0, 0.2),
        "dist_scifi_horiz", "weight"
    )
    histograms["h_dist_scifi_v"] = df.Histo1D(
        (f"h_dist_sf_v_{sample_name}", f"{sample_name} SciFi Track-Channel Dist (Vert);1D Distance to SciFi Track [cm];Normalized Entries", 100, 0.0, 0.2),
        "dist_scifi_vert", "weight"
    )
    histograms["h_doca_scifi"] = df.Histo1D(
        (f"h_doca_sf_{sample_name}", f"{sample_name} SciFi Track-Channel 3D DOCA;DOCA to SciFi Track [cm];Normalized Entries", 100, 0.0, 1.0),
        "doca_scifi", "weight"
    )

    # SciFi Track -> Veto Bars
    histograms["h_dist_veto"] = df.Histo1D(
        (f"h_dist_veto_{sample_name}", f"{sample_name} Veto Bar Distance to SciFi Track;1D Distance to SciFi Track [cm];Normalized Entries", 60, 0.0, 30.0),
        "dist_veto", "weight"
    )
    histograms["h_doca_veto"] = df.Histo1D(
        (f"h_doca_veto_{sample_name}", f"{sample_name} Veto Bar 3D DOCA to SciFi Track;DOCA to SciFi Track [cm];Normalized Entries", 60, 0.0, 30.0),
        "doca_veto", "weight"
    )

    # DS Track -> US Bars
    histograms["h_dist_us"] = df.Histo1D(
        (f"h_dist_us_{sample_name}", f"{sample_name} US Bar Distance to DS Track;1D Distance to DS Track [cm];Normalized Entries", 60, 0.0, 30.0),
        "dist_us", "weight"
    )
    histograms["h_doca_us"] = df.Histo1D(
        (f"h_doca_us_{sample_name}", f"{sample_name} US Bar 3D DOCA to DS Track;DOCA to DS Track [cm];Normalized Entries", 60, 0.0, 30.0),
        "doca_us", "weight"
    )

    # DS Track -> DS Bars
    histograms["h_dist_ds"] = df.Histo1D(
        (f"h_dist_ds_{sample_name}", f"{sample_name} DS Bar Distance to DS Track;1D Distance to DS Track [cm];Normalized Entries", 60, 0.0, 3.0),
        "dist_ds", "weight"
    )
    histograms["h_dist_ds_zoom"] = df.Histo1D(
        (f"h_dist_ds_zoom_{sample_name}", f"{sample_name} DS Bar Distance to DS Track (Zoom);1D Distance to DS Track [cm];Normalized Entries", 60, 0.0, 0.6),
        "dist_ds", "weight"
    )
    histograms["h_doca_ds"] = df.Histo1D(
        (f"h_doca_ds_{sample_name}", f"{sample_name} DS Bar 3D DOCA to DS Track;DOCA to DS Track [cm];Normalized Entries", 60, 0.0, 3.0),
        "doca_ds", "weight"
    )

    # -------------------------------------------------------------
    # Book 2D Histograms
    # -------------------------------------------------------------
    histograms["h2_qdc_vs_dist"] = df_clean.Histo2D(
        (f"h2_qdcdist_{sample_name}", f"{sample_name}: Hit QDC vs Distance to SiPM;Distance to SiPM [cm];Hit QDC [a.u.];Entries", 180, 0.0, 45.0, 300, 0.0, 150.0),
        "clean_hit_dist", "clean_hit_qdc", "weight"
    )
    histograms["h2_cls_qdc_vs_size"] = df.Histo2D(
        (f"h2_clqdcsz_{sample_name}", f"{sample_name}: Cluster QDC vs Size;Cluster Size [channels];Cluster QDC [a.u.];Entries", 10, 0.5, 10.5, 50, 0.0, 200.0),
        "cluster_size", "cluster_qdc", "weight"
    )
    histograms["h2_scifi_vs_mufi"] = df.Histo2D(
        (f"h2_sf_mf_{sample_name}", f"{sample_name}: SciFi vs MuFilter Hits;Track SciFi Hits;Track MuFilter Hits;Entries", 50, 0.0, 100.0, 30, 0.0, 60.0),
        "n_scifi_hits", "n_mufi_hits", "weight"
    )
    histograms["h2_qdc_vs_hits"] = df.Histo2D(
        (f"h2_qdc_hits_{sample_name}", f"{sample_name}: SciFi Total QDC vs Hits;Track SciFi Hits;Track SciFi QDC [a.u.];Entries", 50, 0.0, 100.0, 50, 0.0, 2500.0),
        "n_scifi_hits", "scifi_sum_qdc", "weight"
    )
    histograms["h2_track_slopes"] = df_clean.Histo2D(
        (f"h2_slopes_{sample_name}", f"{sample_name}: Clean Track Slopes;Slope dx/dz;Slope dy/dz;Entries", 40, -0.06, 0.06, 40, -0.06, 0.06),
        "track_slope_xz_clean", "track_slope_yz_clean", "weight"
    )
    histograms["h2_track_xy"] = df_clean.Histo2D(
        (f"h2_xy300_{sample_name}", f"{sample_name}: Beam Spot at z = 300 cm;Track X [cm];Track Y [cm];Entries", 50, -50.0, 0.0, 50, 10.0, 60.0),
        "track_x_300", "track_y_300", "weight"
    )

    # Book event counts
    node_n_proc = df.Count()
    node_n_clean = df_clean.Count()

    # Materialize all results via single event loop
    print(f"[*] Executing computation graph for {sample_name}...")
    results = {k: v.GetValue() for k, v in histograms.items()}
    n_processed = node_n_proc.GetValue()
    n_clean_ev = node_n_clean.GetValue()
    print(f"[+] Completed {n_processed:,} events ({n_clean_ev:,} with clean muon tracks) in {time.time()-t0:.2f} s")

    results["color"] = config["color"]
    results["n_events"] = n_processed
    results["n_clean"] = n_clean_ev
    return results


def draw_pad_1d(pad, hist_dict, hist_key, is_profile=False, log_y=False, cut_line=None, cut_label=None):
    """Draw overlaid 1D histograms or TProfiles across samples with normalization."""
    pad.cd()
    if log_y:
        pad.SetLogy(1)
    else:
        pad.SetLogy(0)

    leg = ROOT.TLegend(0.58, 0.68, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.04)

    drawn = []
    max_val = 0.0

    first = True
    first_h = None
    for sample_name, res in hist_dict.items():
        if hist_key not in res:
            continue
        orig = res[hist_key]
        h = orig.Clone(f"{orig.GetName()}_draw_{hist_key}")
        drawn.append(h)

        if not is_profile:
            integral = h.Integral()
            if integral > 0:
                h.Scale(1.0 / integral)

        h.SetLineColor(res["color"])
        h.SetLineWidth(2)
        h.SetMarkerColor(res["color"])
        h.SetMarkerStyle(20)
        h.SetMarkerSize(0.7)

        max_val = max(max_val, h.GetMaximum())
        d_opt = "E1" if first else "E1 SAME"
        h.Draw(d_opt)
        if first:
            first = False
            first_h = h

        l_opt = "lep" if (is_profile or sample_name == "Data") else "lep"
        leg.AddEntry(h, sample_name, l_opt)

    if first_h:
        if log_y:
            first_h.SetMaximum(max_val * 10.0 if max_val > 0 else 1.0)
            first_h.SetMinimum(1e-4)
        else:
            first_h.SetMaximum(max_val * 1.35 if max_val > 0 else 1.0)
            first_h.SetMinimum(0.0)

        if cut_line is not None:
            y_top = first_h.GetMaximum() * 0.95
            line = ROOT.TLine(cut_line, 0, cut_line, y_top)
            line.SetLineColor(ROOT.kRed + 1)
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw("SAME")
            drawn.append(line)
            lbl = cut_label if cut_label else f"Cut: {cut_line} cm"
            leg.AddEntry(line, lbl, "l")

    leg.Draw()
    pad.Update()
    pad._drawn = drawn
    pad._leg = leg


def main():
    parser = argparse.ArgumentParser(description="Digitization & Reconstruction Validation for SND@LHC")
    parser.add_argument("-id", "--input-data", default=DEFAULT_DATA_PATTERN, help="Collision/Data file pattern")
    parser.add_argument("-ipmu", "--input-pmu", default=DEFAULT_PMU_PATH, help="Passing muon MC file")
    parser.add_argument("-itri", "--input-tri", default=DEFAULT_TRI_PATTERN, help="ThreeMu MC file pattern")
    parser.add_argument("--geo-data", default=DEFAULT_DATA_GEO, help="Geometry file for data")
    parser.add_argument("--geo-pmu", default=DEFAULT_PMU_GEO, help="Geometry file for PMU MC")
    parser.add_argument("--geo-tri", default=DEFAULT_TRI_GEO, help="Geometry file for ThreeMu MC")
    parser.add_argument("-n", "--max-events", type=int, default=None, help="Max events per sample")
    parser.add_argument("-j", "--num-threads", type=int, default=4, help="Number of RDF worker threads")
    parser.add_argument("--lumi-target", type=float, default=28.0, help="Target integrated luminosity [fb^-1]")
    parser.add_argument("--lumi-mc-tri", type=float, default=160.0, help="ThreeMu MC integrated luminosity [fb^-1]")
    parser.add_argument("--scale-pmu", type=float, default=None, help="Manual PMU MC scale factor")
    parser.add_argument("--scifi-max-dist", type=float, default=0.1, help="Max distance between SciFi hit & SciFi track [cm] (default: 0.1 cm = 1 mm)")
    parser.add_argument("--veto-max-dist", type=float, default=3.0, help="Max distance between Veto hit & SciFi track [cm] (default: 3.0 cm)")
    parser.add_argument("--us-max-dist", type=float, default=3.0, help="Max distance between US hit & DS track [cm] (default: 3.0 cm)")
    parser.add_argument("--ds-max-dist", type=float, default=0.3, help="Max distance between DS hit & DS track [cm] (default: 0.3 cm = 3 mm)")
    parser.add_argument("-o", "--output", default="out/digi_validation.root", help="Output ROOT file")
    parser.add_argument("--out-dir", default="plots/digi_validation", help="Output directory for plots")
    parser.add_argument("--no-data", action="store_true", help="Skip data sample")
    parser.add_argument("--no-pmu", action="store_true", help="Skip passing muon MC")
    parser.add_argument("--no-tri", action="store_true", help="Skip trident MC")
    args = parser.parse_args()

    # Load libraries and configure threads
    load_trident_libraries(REPO_ROOT)
    if args.num_threads > 1 and not args.max_events:
        ROOT.EnableImplicitMT(args.num_threads)
        print(f"[*] Enabled ROOT implicit multi-threading with {args.num_threads} threads.")
    else:
        ROOT.DisableImplicitMT()
        if args.max_events:
            print(f"[*] Running single-threaded for event Range({args.max_events}).")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    samples = {}
    if not args.no_data:
        samples["Data"] = {
            "path": args.input_data,
            "alt_path": "/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-0_0_8329_muonReco.root",
            "geo": args.geo_data,
            "tree": "rawConv",
            "color": ROOT.kBlack,
            "is_mc": False,
            "max_files": 40,
            "avg_events_per_file": 27000,
        }
    if not args.no_pmu:
        samples["SingleMuMC"] = {
            "path": args.input_pmu,
            "geo": args.geo_pmu,
            "tree": "cbmsim",
            "color": ROOT.kBlue,
            "is_mc": True,
            "max_files": 1,
            "avg_events_per_file": 1213000,
        }
    if not args.no_tri:
        samples["ThreeMuMC"] = {
            "path": args.input_tri,
            "geo": args.geo_tri,
            "tree": "cbmsim",
            "color": ROOT.kRed,
            "is_mc": True,
            "max_files": 40,
            "avg_events_per_file": 18600,
        }

    results = {}
    for name, cfg in samples.items():
        res = analyze_sample(name, cfg, args)
        if res:
            results[name] = res

    if not results:
        print("[-] No samples successfully processed. Exiting.")
        return

    # -------------------------------------------------------------
    # Canvas 1: SciFi Digitization Distributions (3 x 2)
    # -------------------------------------------------------------
    print("\n[*] Generating Canvas 1: SciFi Digitization Observables...")
    c1 = ROOT.TCanvas("c_scifi_digi", "SciFi Digitization Validation", 1600, 1000)
    c1.Divide(3, 2)

    draw_pad_1d(c1.cd(1), results, "h_qdc_single_hit", is_profile=False, log_y=True)
    draw_pad_1d(c1.cd(2), results, "h_cluster_qdc", is_profile=False, log_y=True)
    draw_pad_1d(c1.cd(3), results, "h_cluster_size", is_profile=False, log_y=True)
    draw_pad_1d(c1.cd(4), results, "p_qdc_vs_dist", is_profile=True, log_y=False)
    draw_pad_1d(c1.cd(5), results, "h_nhits_st1", is_profile=False, log_y=True)
    draw_pad_1d(c1.cd(6), results, "h_nhits_total", is_profile=False, log_y=True)

    c1.SaveAs(os.path.join(args.out_dir, "scifi_digitization_validation.png"))
    c1.SaveAs(os.path.join(args.out_dir, "scifi_digitization_validation.pdf"))

    # -------------------------------------------------------------
    # Canvas 2: Tracking & Detector Multiplicities (3 x 2)
    # -------------------------------------------------------------
    print("[*] Generating Canvas 2: Tracking & Detector Multiplicities...")
    c2 = ROOT.TCanvas("c_tracking_multiplicity", "Tracking & Multiplicities", 1600, 1000)
    c2.Divide(3, 2)

    draw_pad_1d(c2.cd(1), results, "p_nhits_per_station", is_profile=True, log_y=False)
    draw_pad_1d(c2.cd(2), results, "h_ntracks", is_profile=False, log_y=True)
    draw_pad_1d(c2.cd(3), results, "h_track_chi2_ndf", is_profile=False, log_y=True)
    draw_pad_1d(c2.cd(4), results, "h_track_slope_xz", is_profile=False, log_y=False)
    draw_pad_1d(c2.cd(5), results, "h_track_slope_yz", is_profile=False, log_y=False)
    draw_pad_1d(c2.cd(6), results, "h_nhits_mufi", is_profile=False, log_y=True)

    c2.SaveAs(os.path.join(args.out_dir, "tracking_multiplicity_validation.png"))
    c2.SaveAs(os.path.join(args.out_dir, "tracking_multiplicity_validation.pdf"))

    # -------------------------------------------------------------
    # Canvas 3: Track to Activated Channel Distances (3 x 2)
    # -------------------------------------------------------------
    print("[*] Generating Canvas 3: Track-to-Channel Distance Distributions...")
    c_dist = ROOT.TCanvas("c_track_distances", "Track to Channel Distances", 1600, 1000)
    c_dist.Divide(3, 2)

    draw_pad_1d(c_dist.cd(1), results, "h_dist_scifi_zoom", is_profile=False, log_y=True,
                cut_line=args.scifi_max_dist, cut_label=f"Cut: {args.scifi_max_dist*10:.1f} mm")
    draw_pad_1d(c_dist.cd(2), results, "h_dist_veto", is_profile=False, log_y=True,
                cut_line=args.veto_max_dist, cut_label=f"Cut: {args.veto_max_dist:.1f} cm")
    draw_pad_1d(c_dist.cd(3), results, "h_dist_us", is_profile=False, log_y=True,
                cut_line=args.us_max_dist, cut_label=f"Cut: {args.us_max_dist:.1f} cm")
    draw_pad_1d(c_dist.cd(4), results, "h_dist_ds_zoom", is_profile=False, log_y=True,
                cut_line=args.ds_max_dist, cut_label=f"Cut: {args.ds_max_dist*10:.1f} mm")
    draw_pad_1d(c_dist.cd(5), results, "h_doca_scifi", is_profile=False, log_y=True)
    draw_pad_1d(c_dist.cd(6), results, "h_doca_ds", is_profile=False, log_y=True)

    c_dist.SaveAs(os.path.join(args.out_dir, "track_channel_distances.png"))
    c_dist.SaveAs(os.path.join(args.out_dir, "track_channel_distances.pdf"))

    # Standalone distance figures
    dist_single_configs = [
        ("dist_scifi", "h_dist_scifi_zoom", args.scifi_max_dist, f"Cut: {args.scifi_max_dist*10:.1f} mm"),
        ("dist_veto", "h_dist_veto", args.veto_max_dist, f"Cut: {args.veto_max_dist:.1f} cm"),
        ("dist_us", "h_dist_us", args.us_max_dist, f"Cut: {args.us_max_dist:.1f} cm"),
        ("dist_ds", "h_dist_ds_zoom", args.ds_max_dist, f"Cut: {args.ds_max_dist*10:.1f} mm"),
    ]
    for fig_name, key, cut, lbl in dist_single_configs:
        c_single = ROOT.TCanvas(f"c_{fig_name}", fig_name, 800, 600)
        draw_pad_1d(c_single, results, key, is_profile=False, log_y=True, cut_line=cut, cut_label=lbl)
        c_single.SaveAs(os.path.join(args.out_dir, f"{fig_name}.png"))

    # -------------------------------------------------------------
    # Canvas 4: 2D Correlation Matrix (N_samples x 4)
    # -------------------------------------------------------------
    print("[*] Generating 2D Correlation Plots...")
    n_samp = len(results)
    c4 = ROOT.TCanvas("c_2d_correlations", "2D Correlation Matrix", 450 * n_samp, 1600)
    c4.Divide(n_samp, 4)

    row_keys = ["h2_qdc_vs_dist", "h2_cls_qdc_vs_size", "h2_scifi_vs_mufi", "h2_track_slopes"]
    for row_idx, key in enumerate(row_keys):
        for col_idx, (sample_name, res) in enumerate(results.items()):
            pad_num = row_idx * n_samp + col_idx + 1
            pad = c4.cd(pad_num)
            pad.SetRightMargin(0.14)
            pad.SetLogz(1)
            h2 = res[key]
            h2.Draw("COLZ")

    c4.SaveAs(os.path.join(args.out_dir, "2d_correlation_matrix.png"))
    c4.SaveAs(os.path.join(args.out_dir, "2d_correlation_matrix.pdf"))

    # Also save standalone high-resolution 2D figures
    for sample_name, res in results.items():
        for key in ["h2_qdc_vs_dist", "h2_cls_qdc_vs_size", "h2_scifi_vs_mufi", "h2_qdc_vs_hits", "h2_track_slopes", "h2_track_xy"]:
            c_single = ROOT.TCanvas(f"c_{key}_{sample_name}", key, 800, 700)
            c_single.SetRightMargin(0.14)
            c_single.SetLogz(1)
            res[key].Draw("COLZ")
            c_single.SaveAs(os.path.join(args.out_dir, f"{key}_{sample_name}.png"))

    # -------------------------------------------------------------
    # Write Everything to ROOT Output File with dedicated TDirectory
    # -------------------------------------------------------------
    print(f"\n[*] Writing ROOT file: {args.output}")
    out_file = ROOT.TFile(args.output, "RECREATE")
    c1.Write()
    c2.Write()
    c_dist.Write()
    c4.Write()

    # Create dedicated top-level directory for Track-to-Channel Distances
    dir_distances = out_file.mkdir("TrackHitDistances")

    dist_keys = [
        "h_dist_scifi", "h_dist_scifi_zoom", "h_dist_scifi_h", "h_dist_scifi_v", "h_doca_scifi",
        "h_dist_veto", "h_doca_veto",
        "h_dist_us", "h_doca_us",
        "h_dist_ds", "h_dist_ds_zoom", "h_doca_ds"
    ]

    for sample_name, res in results.items():
        # Main sample directory
        dir_sample = out_file.mkdir(sample_name)
        dir_sample.cd()
        for k, obj in res.items():
            if isinstance(obj, (ROOT.TH1, ROOT.TH2, ROOT.TProfile)):
                obj.Write()

        # Write distance plots into dedicated TrackHitDistances/<sample_name>
        dir_distances.cd()
        dir_sample_dist = dir_distances.mkdir(sample_name)
        dir_sample_dist.cd()
        for k in dist_keys:
            if k in res and isinstance(res[k], (ROOT.TH1, ROOT.TH2, ROOT.TProfile)):
                res[k].Write()

    out_file.Close()
    print(f"[+] Done! All figures saved in '{args.out_dir}' and ROOT histograms in '{args.output}'.\n")


if __name__ == "__main__":
    main()
