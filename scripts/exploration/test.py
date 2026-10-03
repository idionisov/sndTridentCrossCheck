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

# Declare numerical Landau convolved with Gaussian function in ROOT
ROOT.gInterpreter.Declare("""
#include "TMath.h"
#include "TF1.h"

Double_t langaufun(Double_t *x, Double_t *par) {
   // Fit parameters:
   // par[0] = Landau scale parameter (width)
   // par[1] = Most Probable (MP, location) of Landau
   // par[2] = Total area (normalization constant)
   // par[3] = Gaussian sigma (resolution broadening)

   if (par[0] <= 0.0 || par[3] <= 0.0) return 0.0;
   Double_t invsq2pi = 0.3989422804014;
   Double_t mpshift  = -0.22278298;

   Double_t np = 80.0;
   Double_t sc =  5.0;

   Double_t xx, fland, sum = 0.0;
   Double_t xlow = x[0] - sc * par[3];
   Double_t xupp = x[0] + sc * par[3];
   Double_t step = (xupp - xlow) / np;

   for(Double_t i = 1.0; i <= np/2; i++) {
      xx = xlow + (i - 0.5) * step;
      fland = TMath::Landau(xx, par[1] - mpshift * par[0], par[0]) / par[0];
      sum += fland * TMath::Gaus(x[0], xx, par[3]);

      xx = xupp - (i - 0.5) * step;
      fland = TMath::Landau(xx, par[1] - mpshift * par[0], par[0]) / par[0];
      sum += fland * TMath::Gaus(x[0], xx, par[3]);
   }

   return (par[2] * step * sum * invsq2pi / par[3]);
}
""")

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
    geo_file = config["geo"]
    if not os.path.exists(geo_file) and "alt_geo" in config and os.path.exists(config["alt_geo"]):
        geo_file = config["alt_geo"]
    print(f"[*] Loading geometry: {geo_file}")
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    scifi_det = snd_geo.modules["Scifi"]
    mufi_det  = snd_geo.modules["MuFilter"]

    print("[*] Initializing DigiValidationProcessor with track-hit distance cuts...")
    val_cfg = ROOT.snd.trident.DigiValidationConfig()
    val_cfg.scifi_max_dist = args.scifi_max_dist
    val_cfg.cluster_max_dist = args.cluster_max_dist
    val_cfg.veto_max_dist  = args.veto_max_dist
    val_cfg.us_max_dist    = args.us_max_dist
    val_cfg.ds_max_dist    = args.ds_max_dist
    print(f"[*] Configured cuts: SciFi Hit <= {args.scifi_max_dist*10:.2f} mm | SciFi Cluster <= {args.cluster_max_dist*10:.2f} mm | Veto <= {args.veto_max_dist:.1f} cm | US <= {args.us_max_dist:.1f} cm | DS <= {args.ds_max_dist*10:.2f} mm")

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
        .Define("n_scifi_st2", "ev.n_scifi_st2")
        .Define("n_scifi_st3", "ev.n_scifi_st3")
        .Define("n_scifi_st4", "ev.n_scifi_st4")
        .Define("n_scifi_st5", "ev.n_scifi_st5")
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
        .Define("cluster_distance", "ev.cluster_distance")
        .Define("cluster_dist_to_track", "ev.cluster_dist_to_track")
        .Define("cluster_station", "ev.cluster_station")
        .Define("cluster_is_vertical", "ev.cluster_is_vertical")
        .Define("cluster_qdc_horiz", "ev.cluster_qdc_horiz")
        .Define("cluster_dist_horiz", "ev.cluster_dist_horiz")
        .Define("cluster_qdc_vert", "ev.cluster_qdc_vert")
        .Define("cluster_dist_vert", "ev.cluster_dist_vert")
        .Define("cluster_seed_qdc", "ev.cluster_seed_qdc")
        .Define("cluster_neighbor_qdc", "ev.cluster_neighbor_qdc")
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

    # Clusters (only track-associated within tolerable distance)
    histograms["h_cluster_qdc"] = df.Histo1D(
        (f"h_cl_qdc_{sample_name}", f"{sample_name} Cluster Integrated QDC;Cluster QDC [a.u.];Normalized Entries", 100, 0.0, 25.0),
        "cluster_qdc", "weight"
    )
    histograms["h_cluster_seed_qdc"] = df_clean.Histo1D(
        (f"h_cl_seed_{sample_name}", f"{sample_name} Cluster Seed QDC;Hit QDC [a.u.];Normalized Entries", 80, -5.0, 15.0),
        "cluster_seed_qdc", "weight"
    )
    histograms["h_cluster_neighbor_qdc"] = df_clean.Histo1D(
        (f"h_cl_neigh_{sample_name}", f"{sample_name} Cluster Neighbor QDC;Hit QDC [a.u.];Normalized Entries", 80, -5.0, 15.0),
        "cluster_neighbor_qdc", "weight"
    )
    histograms["h_cluster_size"] = df.Histo1D(
        (f"h_cl_sz_{sample_name}", f"{sample_name} Cluster Size;Cluster Size [channels];Normalized Entries", 12, 0.5, 12.5),
        "cluster_size", "weight"
    )
    histograms["h_cluster_dist_to_track"] = df.Histo1D(
        (f"h_cl_dist_trk_{sample_name}", f"{sample_name} SciFi Cluster Distance to SciFi Track;Residual Distance [cm];Normalized Entries", 100, 0.0, 0.1),
        "cluster_dist_to_track", "weight"
    )

    # SciFi Tracker Multiplicities (track-associated hits on clean tracks)
    histograms["h_nhits_st1"] = df_clean.Histo1D(
        (f"h_st1_{sample_name}", f"{sample_name} Station 1 Hits;SciFi Station 1 Hits;Normalized Entries", 40, -0.5, 39.5),
        "n_scifi_st1", "weight"
    )
    histograms["h_nhits_st2"] = df_clean.Histo1D(
        (f"h_st2_{sample_name}", f"{sample_name} Station 2 Hits;SciFi Station 2 Hits;Normalized Entries", 40, -0.5, 39.5),
        "n_scifi_st2", "weight"
    )
    histograms["h_nhits_st3"] = df_clean.Histo1D(
        (f"h_st3_{sample_name}", f"{sample_name} Station 3 Hits;SciFi Station 3 Hits;Normalized Entries", 40, -0.5, 39.5),
        "n_scifi_st3", "weight"
    )
    histograms["h_nhits_st4"] = df_clean.Histo1D(
        (f"h_st4_{sample_name}", f"{sample_name} Station 4 Hits;SciFi Station 4 Hits;Normalized Entries", 40, -0.5, 39.5),
        "n_scifi_st4", "weight"
    )
    histograms["h_nhits_st5"] = df_clean.Histo1D(
        (f"h_st5_{sample_name}", f"{sample_name} Station 5 Hits;SciFi Station 5 Hits;Normalized Entries", 40, -0.5, 39.5),
        "n_scifi_st5", "weight"
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
        (f"h2_qdcdist_{sample_name}", f"{sample_name}: Hit QDC vs Distance to SiPM;Distance to SiPM [cm];Hit QDC [a.u.];Entries", 180, 0.0, 45.0, 320, -10.0, 150.0),
        "clean_hit_dist", "clean_hit_qdc", "weight"
    )
    histograms["h2_qdc_vs_dist_h"] = df_clean.Histo2D(
        (f"h2_qdcdist_h_{sample_name}", f"{sample_name}: Horizontal Hit QDC vs Distance;Distance to SiPM [cm];Hit QDC [a.u.];Entries", 180, 0.0, 45.0, 320, -10.0, 150.0),
        "clean_hit_dist_horiz", "clean_hit_qdc_horiz", "weight"
    )
    histograms["h2_qdc_vs_dist_v"] = df_clean.Histo2D(
        (f"h2_qdcdist_v_{sample_name}", f"{sample_name}: Vertical Hit QDC vs Distance;Distance to SiPM [cm];Hit QDC [a.u.];Entries", 180, 0.0, 45.0, 320, -10.0, 150.0),
        "clean_hit_dist_vert", "clean_hit_qdc_vert", "weight"
    )
    histograms["h2_cl_qdc_vs_dist"] = df.Histo2D(
        (f"h2_cl_qdcdist_{sample_name}", f"{sample_name}: Cluster QDC vs SiPM Distance;Distance to SiPM [cm];Cluster QDC [a.u.];Entries", 40, 0.0, 40.0, 100, 0.0, 25.0),
        "cluster_distance", "cluster_qdc", "weight"
    )
    histograms["h2_cl_qdc_vs_dist_h"] = df.Histo2D(
        (f"h2_cl_qdcdist_h_{sample_name}", f"{sample_name}: Horizontal Cluster QDC vs Distance;Distance to SiPM [cm];Cluster QDC [a.u.];Entries", 40, 0.0, 40.0, 100, 0.0, 25.0),
        "cluster_dist_horiz", "cluster_qdc_horiz", "weight"
    )
    histograms["h2_cl_qdc_vs_dist_v"] = df.Histo2D(
        (f"h2_cl_qdcdist_v_{sample_name}", f"{sample_name}: Vertical Cluster QDC vs Distance;Distance to SiPM [cm];Cluster QDC [a.u.];Entries", 40, 0.0, 40.0, 100, 0.0, 25.0),
        "cluster_dist_vert", "cluster_qdc_vert", "weight"
    )
    histograms["p_cl_size_vs_dist"] = df.Profile1D(
        (f"p_cl_sizedist_{sample_name}", f"{sample_name}: Mean Cluster Size vs SiPM Distance;Distance to SiPM [cm];Mean Cluster Size [channels]", 20, 0.0, 40.0),
        "cluster_distance", "cluster_size", "weight"
    )
    histograms["p_cl_qdc_vs_dist"] = df.Profile1D(
        (f"p_cl_qdcdist_{sample_name}", f"{sample_name}: Cluster Mean QDC vs SiPM Distance;Distance to SiPM [cm];Cluster Mean QDC [a.u.]", 40, 0.0, 40.0),
        "cluster_distance", "cluster_qdc", "weight"
    )
    histograms["h2_cls_qdc_vs_size"] = df.Histo2D(
        (f"h2_clqdcsz_{sample_name}", f"{sample_name}: Cluster QDC vs Size;Cluster Size [channels];Cluster QDC [a.u.];Entries", 10, 0.5, 10.5, 50, 0.0, 25.0),
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


def analyze_landau_vavilov(results, out_dir, out_file):
    """
    Fits two-component (Gaussian pedestal noise + Landau MIP) distributions to single hit QDC,
    fits Landau-Vavilov to cluster QDC, slices 2D QDC vs distance distributions into distance bins,
    fits each slice with two-component models to extract the pure MIP Most Probable Value (MPV),
    and models the optical attenuation length. Fits are extracted solely on Data and compared
    directly against nominal/default simulation parameters.
    """
    # Ensure implicit multi-threading is disabled during fitting to avoid PyROOT GIL deadlocks with TBB
    ROOT.DisableImplicitMT()
    print("\n[*] Performing Two-Component (Pedestal Noise + Landau MIP) Energy Loss & Attenuation Analysis on Data...")
    dir_landau = out_file.mkdir("LandauVavilov")

    c_landau = ROOT.TCanvas("c_landau_vavilov", "Landau-Vavilov Energy Loss & MPV Attenuation", 1600, 1200)
    c_landau.Divide(2, 2)

    target_sample = "Data" if "Data" in results else list(results.keys())[0]
    res_target = results[target_sample]

    # Pad 1: Single Hit QDC Two-Component (Gaus Noise + Landau MIP) Fit (Data Only)
    p1 = c_landau.cd(1)
    p1.SetLogy(0)
    leg1 = ROOT.TLegend(0.40, 0.58, 0.88, 0.88)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetTextSize(0.027)

    drawn1 = []
    if "h_qdc_single_hit" in res_target:
        h1 = res_target["h_qdc_single_hit"].Clone(f"h_qdc_fit_{target_sample}")
        h1.Scale(1.0 / max(1.0, h1.Integral()))
        h1.GetXaxis().SetRangeUser(-3.0, 15.0)
        h1.GetYaxis().SetTitle("Normalized Entries")
        h1.SetLineColor(res_target["color"])
        h1.SetMarkerColor(res_target["color"])
        h1.SetMarkerStyle(20)
        h1.SetMarkerSize(0.6)

        # Fit Gaus(0) + Landau(3):
        f_tot = ROOT.TF1(f"f_tot_{target_sample}", "gaus(0) + landau(3)", -3.0, 12.0)
        f_tot.SetParameters(h1.GetMaximum() * 0.7, 0.0, 0.9, h1.GetMaximum() * 0.5, 2.2, 0.6)
        f_tot.SetParLimits(1, -0.4, 0.4)
        f_tot.SetParLimits(2, 0.4, 1.4)
        f_tot.SetParLimits(4, 1.2, 4.0)
        f_tot.SetParLimits(5, 0.3, 1.5)
        f_tot.SetLineColor(res_target["color"])
        f_tot.SetLineWidth(2)
        h1.Fit(f_tot, "RQS0")

        f_noise = ROOT.TF1(f"f_noise_{target_sample}", "gaus", -3.0, 12.0)
        f_noise.SetParameters(f_tot.GetParameter(0), f_tot.GetParameter(1), f_tot.GetParameter(2))
        f_noise.SetLineColor(ROOT.kRed + 1)
        f_noise.SetLineStyle(2)
        f_noise.SetLineWidth(2)

        f_mip = ROOT.TF1(f"f_mip_{target_sample}", "landau", -3.0, 12.0)
        f_mip.SetParameters(f_tot.GetParameter(3), f_tot.GetParameter(4), f_tot.GetParameter(5))
        f_mip.SetLineColor(ROOT.kBlue + 1)
        f_mip.SetLineStyle(2)
        f_mip.SetLineWidth(2)

        h1.Draw("E1")
        f_tot.Draw("SAME")
        f_noise.Draw("SAME")
        f_mip.Draw("SAME")
        drawn1.extend([h1, f_tot, f_noise, f_mip])

        mpv = f_tot.GetParameter(4)
        sigma = f_tot.GetParameter(5)
        noise_sig = f_tot.GetParameter(2)
        leg1.AddEntry(h1, f"{target_sample} Single Hits", "lep")
        leg1.AddEntry(f_tot, f"Total Fit (Noise + MIP)", "l")
        leg1.AddEntry(f_noise, f"Pedestal Noise (#sigma={noise_sig:.2f})", "l")
        leg1.AddEntry(f_mip, f"MIP Landau (MPV={mpv:.2f}, #sigma={sigma:.2f})", "l")
        res_target["h_qdc_fit"] = h1
        res_target["f_noise_hit"] = f_noise
        res_target["f_landau_hit"] = f_mip
        res_target["f_tot_hit"] = f_tot

    leg1.Draw()
    p1.Update()
    p1._drawn = drawn1

    # Pad 2: Cluster QDC LanGaus Fit (Area-normalized, Data Only)
    p2 = c_landau.cd(2)
    p2.SetLogy(0)
    leg2 = ROOT.TLegend(0.38, 0.65, 0.88, 0.88)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    leg2.SetTextSize(0.027)

    drawn2 = []
    if "h_cluster_qdc" in res_target:
        h2_cl = res_target["h_cluster_qdc"].Clone(f"h_cl_fit_{target_sample}")
        h2_cl.Scale(1.0 / max(1.0, h2_cl.Integral()))
        h2_cl.GetXaxis().SetRangeUser(0.0, 25.0)
        h2_cl.GetYaxis().SetTitle("Normalized Entries")
        h2_cl.SetLineColor(res_target["color"])
        h2_cl.SetMarkerColor(res_target["color"])
        h2_cl.SetMarkerStyle(20)
        h2_cl.SetMarkerSize(0.6)

        x_peak = h2_cl.GetXaxis().GetBinCenter(h2_cl.GetMaximumBin())
        f_cl = ROOT.TF1(f"f_langau_cl_{target_sample}", ROOT.langaufun, 0.4, 7.5, 4)
        f_cl.SetParNames("Width_L", "MPV_L", "Area", "Sigma_G")
        f_cl.SetParameters(0.4, max(0.4, x_peak), 0.25, 0.6)
        f_cl.SetParLimits(0, 0.05, 1.5)
        f_cl.SetParLimits(1, 0.3, 3.5)
        f_cl.SetParLimits(2, 0.01, 2.0)
        f_cl.SetParLimits(3, 0.05, 1.8)
        f_cl.SetLineColor(res_target["color"])
        f_cl.SetLineWidth(2)
        h2_cl.Fit(f_cl, "RQS0")

        h2_cl.Draw("E1")
        f_cl.Draw("SAME")
        drawn2.extend([h2_cl, f_cl])

        peak_pos = f_cl.GetMaximumX(0.4, 4.0)
        wl = f_cl.GetParameter(0)
        wg = f_cl.GetParameter(3)
        leg2.AddEntry(h2_cl, f"{target_sample} Track Clusters", "lep")
        leg2.AddEntry(f_cl, f"LanGaus Fit (Peak={peak_pos:.2f}, #sigma_{{L}}={wl:.2f}, #sigma_{{G}}={wg:.2f})", "l")
        res_target["f_langau_cl"] = f_cl
        res_target["f_landau_cl"] = f_cl  # alias
        res_target["h_cl_fit"] = h2_cl

    leg2.Draw()
    p2.Update()
    p2._drawn = drawn2

    # Pad 3: Multi-Slice Cluster LanGaus Peak Shift (Data Only)
    p3 = c_landau.cd(3)
    p3.SetLogy(0)
    leg3 = ROOT.TLegend(0.40, 0.60, 0.88, 0.88)
    leg3.SetBorderSize(0)
    leg3.SetFillStyle(0)
    leg3.SetTextSize(0.027)

    drawn3 = []
    h2_target = res_target.get("h2_cl_qdc_vs_dist", None)
    slice_specs = [
        (2.0, 6.0, ROOT.kBlue + 1),
        (12.0, 16.0, ROOT.kTeal + 2),
        (22.0, 26.0, ROOT.kOrange + 7),
        (32.0, 36.0, ROOT.kRed + 1),
    ]

    res_target["slice_histos"] = []
    res_target["slice_fits"] = []

    if h2_target:
        first = True
        for x_lo, x_hi, col in slice_specs:
            b1 = h2_target.GetXaxis().FindBin(x_lo + 0.01)
            b2 = h2_target.GetXaxis().FindBin(x_hi - 0.01)
            py = h2_target.ProjectionY(f"py_cl_slice_{int(x_lo)}_{int(x_hi)}", b1, b2)
            if py.Integral() < 10.0:
                continue
            py.Scale(1.0 / max(1.0, py.Integral()))
            py.GetXaxis().SetRangeUser(0.0, 12.0)
            py.SetLineColor(col)
            py.SetLineWidth(2)
            py.SetTitle(f"{target_sample}: Cluster QDC Slices along Fiber Length;Cluster QDC [a.u.];Normalized Entries")

            x_peak = py.GetXaxis().GetBinCenter(py.GetMaximumBin())
            fn = ROOT.TF1(f"fn_cl_sl_{int(x_lo)}_{int(x_hi)}", ROOT.langaufun, 0.4, 7.5, 4)
            fn.SetParNames("Width_L", "MPV_L", "Area", "Sigma_G")
            fn.SetParameters(0.4, max(0.4, x_peak * 0.8), 0.25, 0.6)
            fn.SetParLimits(0, 0.05, 1.5)
            fn.SetParLimits(1, 0.3, 3.5)
            fn.SetParLimits(2, 0.01, 2.0)
            fn.SetParLimits(3, 0.1, 1.5)
            fn.SetLineColor(col)
            fn.SetLineStyle(1)
            fn.SetLineWidth(2)
            py.Fit(fn, "RQS0")

            d_opt = "HIST" if first else "HIST SAME"
            py.Draw(d_opt)
            fn.Draw("SAME")
            first = False
            drawn3.extend([py, fn])
            res_target["slice_histos"].append(py)
            res_target["slice_fits"].append(fn)

            peak_pos = fn.GetMaximumX(0.4, 4.0)
            leg3.AddEntry(py, f"d #in [{x_lo:.0f}, {x_hi:.0f}] cm (Peak={peak_pos:.2f})", "l")

        leg3.Draw()
        p3.Update()
        p3._drawn = drawn3

    # Pad 4: Cluster Peak vs Distance & Attenuation Fit over 2D Background (Data Only)
    p4 = c_landau.cd(4)
    p4.SetLogy(0)
    p4.SetRightMargin(0.14)
    leg4 = ROOT.TLegend(0.35, 0.65, 0.84, 0.88)
    leg4.SetBorderSize(0)
    leg4.SetFillStyle(1001)
    leg4.SetFillColorAlpha(ROOT.kWhite, 0.85)
    leg4.SetTextSize(0.028)

    drawn4 = []
    if h2_target:
        h2_bg = h2_target.Clone("h2_cl_att_bg_Data")
        h2_bg.SetTitle("Data Cluster QDC vs SiPM Distance & Attenuation Fit;Distance to SiPM [cm];Cluster QDC [a.u.]")
        h2_bg.GetXaxis().SetRangeUser(0.0, 42.0)
        h2_bg.GetYaxis().SetRangeUser(0.0, 8.0)
        h2_bg.Draw("COLZ")
        drawn4.append(h2_bg)

        gr = ROOT.TGraphErrors()
        gr.SetName(f"g_cl_peak_vs_dist_{target_sample}")
        gr.SetTitle("Cluster Peak vs Distance to SiPM;Distance along Fiber [cm];Cluster Peak QDC [a.u.]")
        gr.SetMarkerColor(ROOT.kBlack)
        gr.SetLineColor(ROOT.kBlack)
        gr.SetMarkerStyle(20)
        gr.SetMarkerSize(0.9)

        idx = 0
        step = 4.0
        for x_low in range(0, 36, int(step)):
            x_high = x_low + step
            b1 = h2_target.GetXaxis().FindBin(x_low + 0.01)
            b2 = h2_target.GetXaxis().FindBin(x_high - 0.01)
            py = h2_target.ProjectionY(f"py_cl_{target_sample}_{int(x_low)}_{int(x_high)}", b1, b2)
            if py.Integral() < 10.0:
                continue
            py.Scale(1.0 / max(1.0, py.Integral()))
            x_peak = py.GetXaxis().GetBinCenter(py.GetMaximumBin())

            fn = ROOT.TF1(f"fn_cl_{target_sample}_{int(x_low)}", ROOT.langaufun, 0.4, 7.5, 4)
            fn.SetParameters(0.4, max(0.4, x_peak * 0.8), 0.25, 0.6)
            fn.SetParLimits(0, 0.05, 1.5)
            fn.SetParLimits(1, 0.3, 3.5)
            fn.SetParLimits(2, 0.01, 2.0)
            fn.SetParLimits(3, 0.1, 1.5)
            py.Fit(fn, "RQS0")

            peak_val = fn.GetMaximumX(0.4, 4.0)
            err = fn.GetParError(1)
            if err > 0.50 or peak_val < 0.4 or peak_val > 5.0:
                continue
            gr.SetPoint(idx, 0.5 * (x_low + x_high), peak_val)
            gr.SetPointError(idx, 0.5 * step, err)
            idx += 1

        f_att = None
        if gr.GetN() > 3:
            f_att = ROOT.TF1(f"f_cl_att_{target_sample}", "[0]*exp(-x/[1])", 2.0, 36.0)
            f_att.SetParameters(2.2, 80.0)
            f_att.SetParLimits(0, 0.5, 5.0)
            f_att.SetParLimits(1, 20.0, 500.0)
            f_att.SetLineColor(ROOT.kRed + 1)
            f_att.SetLineWidth(3)
            f_att.SetParNames("A_0", "lambda_att")
            gr.Fit(f_att, "RQS0")
            drawn4.append(f_att)
            q0_val = f_att.GetParameter(0)
            q0_err = f_att.GetParError(0)
            l_att = f_att.GetParameter(1)
            l_err = f_att.GetParError(1)
            leg4.AddEntry(gr, f"Data LanGaus MPV Points", "lep")
            leg4.AddEntry(f_att, f"Data Fit (#lambda = {l_att:.1f}#pm{l_err:.1f} cm)", "l")
            res_target["f_expo_att"] = f_att
            res_target["f_expo_mpv"] = f_att  # alias

            # Nominal/Default simulation attenuation curve (lambda = 300 cm)
            f_default = ROOT.TF1(f"f_cl_att_default_{target_sample}", "[0]*exp(-x/300.0)", 0.0, 42.0)
            f_default.SetParameter(0, f_att.GetParameter(0))
            f_default.SetLineColor(ROOT.kBlue + 2)
            f_default.SetLineStyle(7)
            f_default.SetLineWidth(3)
            leg4.AddEntry(f_default, "sndsw default (#lambda = 300.0 cm)", "l")
            drawn4.append(f_default)

            # Extraction of mip_light_yield at reference distance d = 20 cm
            # In sndsw: ly(20 cm) = Delta_E * Y_MIP
            # Q(20 cm) = A * ly(20 cm) + B  ->  N_pe(20 cm) = (Q(20 cm) - B) / A
            # With A = 0.172, B = -1.31, and <Delta_E> = 260.0 keV (6-layer SciFi mat MIP deposit)
            A_gain = 0.172
            B_ped = -1.31
            delta_E_mip = 260.0  # keV
            q20 = q0_val * ROOT.TMath.Exp(-20.0 / l_att)
            # Propagate error from q0 and l_att
            # dq20/dq0 = exp(-20/l_att), dq20/dl_att = q0 * (20/l_att^2) * exp(-20/l_att)
            dq_dq0 = ROOT.TMath.Exp(-20.0 / l_att)
            dq_dlam = q0_val * (20.0 / (l_att * l_att)) * ROOT.TMath.Exp(-20.0 / l_att)
            q20_err = ROOT.TMath.Sqrt((dq_dq0 * q0_err)**2 + (dq_dlam * l_err)**2)

            n_pe_20 = (q20 - B_ped) / A_gain
            n_pe_err = q20_err / A_gain
            y_mip = n_pe_20 / delta_E_mip
            y_mip_err = n_pe_err / delta_E_mip

            print(f"\n[*] SciFi Light Yield Extraction:")
            print(f"    - Attenuation Length: lambda_att = {l_att:.1f} +/- {l_err:.1f} cm")
            print(f"    - MPV QDC at d = 20 cm: Q(20) = {q20:.3f} +/- {q20_err:.3f} a.u.")
            print(f"    - Equivalent Light Yield at 20 cm: N_pe = {n_pe_20:.2f} +/- {n_pe_err:.2f} p.e.")
            print(f"    - Extracted mip_light_yield: Y_MIP = {y_mip:.4f} +/- {y_mip_err:.4f} p.e./keV (nominal sndsw: 0.1600)")
            leg4.AddEntry(ROOT.nullptr, f"Extracted Y_{{MIP}} = {y_mip:.3f}#pm{y_mip_err:.3f} p.e./keV", "")
            leg4.AddEntry(ROOT.nullptr, f"(sndsw default Y_{{MIP}} = 0.160 p.e./keV)", "")

            res_target["y_mip"] = y_mip
            res_target["y_mip_err"] = y_mip_err
            res_target["q20"] = q20

        gr.Draw("P SAME")
        drawn4.append(gr)
        res_target["g_peak_vs_dist"] = gr
        res_target["g_mpv_vs_dist"] = gr  # alias

        if f_att:
            f_att.Draw("SAME")
            f_default.Draw("SAME")

    leg4.Draw()
    p4.Update()
    p4._drawn = drawn4

    # Save canvas to disk
    c_landau.SaveAs(os.path.join(out_dir, "landau_vavilov_energy_loss.png"))
    c_landau.SaveAs(os.path.join(out_dir, "landau_vavilov_energy_loss.pdf"))

    # Write to ROOT file in LandauVavilov directory (Real Data Only)
    dir_landau.cd()
    c_landau.Write()
    if "h_qdc_fit" in res_target:
        res_target["h_qdc_fit"].Write()
    if "f_noise_hit" in res_target:
        res_target["f_noise_hit"].Write()
    if "f_landau_hit" in res_target:
        res_target["f_landau_hit"].Write()
    if "f_tot_hit" in res_target:
        res_target["f_tot_hit"].Write()
    if "h_cl_fit" in res_target:
        res_target["h_cl_fit"].Write()
    if "f_langau_cl" in res_target:
        res_target["f_langau_cl"].Write()
        f_cl_alias = res_target["f_langau_cl"].Clone(f"f_landau_cl_{target_sample}")
        f_cl_alias.Write()
    if "slice_histos" in res_target:
        for sh in res_target["slice_histos"]:
            sh.Write()
    if "slice_fits" in res_target:
        for sf in res_target["slice_fits"]:
            sf.Write()
    if "g_peak_vs_dist" in res_target:
        res_target["g_peak_vs_dist"].Write()
        gr_alias = res_target["g_peak_vs_dist"].Clone(f"g_cl_mpv_vs_dist_{target_sample}")
        gr_alias.Write()
    if "f_expo_att" in res_target:
        res_target["f_expo_att"].Write()
        f_att_alias = res_target["f_expo_att"].Clone(f"f_cl_mpv_att_{target_sample}")
        f_att_alias.Write()
    if "f_cl_att_default_" + target_sample in [d.GetName() for d in drawn4 if hasattr(d, "GetName")]:
        f_default.Write("f_cl_mpv_att_default")
    if "y_mip" in res_target:
        p_ymip = ROOT.TParameter("double")("mip_light_yield", res_target["y_mip"])
        p_ymip.Write()
        p_ymip_err = ROOT.TParameter("double")("mip_light_yield_err", res_target["y_mip_err"])
        p_ymip_err.Write()
        t_summary = ROOT.TNamed("CalibrationSummary",
            f"lambda_att={l_att:.2f}+/-{l_err:.2f} cm; Y_MIP={res_target['y_mip']:.4f}+/-{res_target['y_mip_err']:.4f} p.e./keV; Q(20cm)={res_target['q20']:.3f} a.u.")
        t_summary.Write()



def analyze_scifi_hit_calibration(results, out_dir, out_file):
    """
    SciFi Hit & Cluster Multiplicity and Effective Threshold Calibration.
    Analyzes cluster size distribution, hit multiplicity per station,
    low-QDC threshold turn-on, and mean cluster size vs distance along the fiber.
    Fits are extracted solely on Data and stored in the dedicated SciFiCalibration directory.
    """
    ROOT.DisableImplicitMT()
    print("\n[*] Performing SciFi Hit Multiplicity & Threshold Calibration on Data...")
    dir_calib = out_file.mkdir("SciFiCalibration")

    target_sample = "Data" if "Data" in results else list(results.keys())[0]
    res = results[target_sample]

    c_calib = ROOT.TCanvas("c_scifi_calibration", "SciFi Hit Multiplicity & Threshold Calibration", 1600, 1200)
    c_calib.Divide(2, 2)

    # Pad 1: Cluster Size Distribution (Fibers per Cluster)
    p1 = c_calib.cd(1)
    p1.SetLogy(1)
    leg1 = ROOT.TLegend(0.48, 0.65, 0.88, 0.88)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetTextSize(0.028)

    drawn1 = []
    if "h_cluster_size" in res:
        h_sz = res["h_cluster_size"].Clone(f"h_sz_calib_{target_sample}")
        h_sz.Scale(1.0 / max(1.0, h_sz.Integral()))
        h_sz.GetXaxis().SetRangeUser(0.5, 8.5)
        h_sz.GetYaxis().SetTitle("Cluster Fraction")
        h_sz.GetYaxis().SetRangeUser(1e-4, 1.0)
        h_sz.SetLineColor(res["color"])
        h_sz.SetLineWidth(2)
        h_sz.SetMarkerColor(res["color"])
        h_sz.SetMarkerStyle(20)
        h_sz.SetMarkerSize(0.8)
        h_sz.Draw("E1")
        drawn1.append(h_sz)

        n1 = h_sz.GetBinContent(1) * 100.0
        n2 = h_sz.GetBinContent(2) * 100.0
        n3 = h_sz.GetBinContent(3) * 100.0
        leg1.AddEntry(h_sz, f"{target_sample} Cluster Size", "lep")
        leg1.AddEntry(ROOT.nullptr, f"1-hit: {n1:.1f}% | 2-hit: {n2:.1f}% | 3-hit: {n3:.1f}%", "")
        leg1.Draw()
        p1.Update()
        p1._drawn = drawn1

    # Pad 2: SciFi Track Hit Multiplicity per Station (Conditioned on Clean Tracks)
    p2 = c_calib.cd(2)
    p2.SetLogy(0)
    leg2 = ROOT.TLegend(0.45, 0.65, 0.88, 0.88)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    leg2.SetTextSize(0.028)

    drawn2 = []
    st_colors = [ROOT.kBlue + 1, ROOT.kAzure + 2, ROOT.kTeal + 2, ROOT.kOrange + 7, ROOT.kRed + 1]
    first_st = True
    for st in range(1, 6):
        key = f"h_nhits_st{st}"
        if key in res:
            h_st = res[key].Clone(f"h_st{st}_calib_{target_sample}")
            h_st.SetTitle(f"{target_sample}: SciFi Hits per Station (Clean Tracks);SciFi Station Hits;Normalized Entries")
            # Exclude zero-hit bin if normalizing over detected tracks in that station
            h_st.Scale(1.0 / max(1.0, h_st.Integral()))
            h_st.GetXaxis().SetRangeUser(0.5, 8.5)
            h_st.GetYaxis().SetTitle("Normalized Entries")
            h_st.GetYaxis().SetRangeUser(0.0, 0.65)
            h_st.SetLineColor(st_colors[st - 1])
            h_st.SetLineWidth(2)
            d_opt = "HIST" if first_st else "HIST SAME"
            h_st.Draw(d_opt)
            first_st = False
            drawn2.append(h_st)
            leg2.AddEntry(h_st, f"Station {st} (Mean: {h_st.GetMean():.2f})", "l")

    leg2.Draw()
    p2.Update()
    p2._drawn = drawn2

    # Pad 3: Hit Detection Threshold Turn-On (Seed vs. Neighbor Hits)
    p3 = c_calib.cd(3)
    p3.SetLogy(0)
    leg3 = ROOT.TLegend(0.35, 0.62, 0.88, 0.88)
    leg3.SetBorderSize(0)
    leg3.SetFillStyle(0)
    leg3.SetTextSize(0.027)

    drawn3 = []
    h_seed = None
    h_neigh = None
    line_nom = None
    if "h_cluster_seed_qdc" in res and "h_cluster_neighbor_qdc" in res:
        h_seed = res["h_cluster_seed_qdc"].Clone(f"h_seed_qdc_{target_sample}")
        h_neigh = res["h_cluster_neighbor_qdc"].Clone(f"h_neigh_qdc_{target_sample}")
        h_seed.Scale(1.0 / max(1.0, h_seed.Integral()))
        h_neigh.Scale(1.0 / max(1.0, h_neigh.Integral()))

        h_seed.SetTitle(f"{target_sample}: Seed vs Neighbor Hit QDC;Hit QDC [a.u.];Normalized Entries")
        h_seed.GetXaxis().SetRangeUser(-5.0, 12.0)
        h_seed.GetYaxis().SetTitle("Normalized Entries")
        max_y = max(h_seed.GetMaximum(), h_neigh.GetMaximum()) * 1.25
        h_seed.GetYaxis().SetRangeUser(0.0, max_y)
        h_seed.SetLineColor(ROOT.kBlack)
        h_seed.SetLineWidth(2)
        h_seed.Draw("HIST")
        drawn3.append(h_seed)

        h_neigh.SetLineColor(ROOT.kRed + 1)
        h_neigh.SetFillColorAlpha(ROOT.kRed + 1, 0.25)
        h_neigh.SetLineWidth(2)
        h_neigh.Draw("HIST SAME")
        drawn3.append(h_neigh)

        # Nominal simulation threshold nphe_min = 3.5 p.e. -> QDC = 0.172 * 3.5 - 1.31 = -0.708 a.u.
        qdc_nom_th = 0.172 * 3.5 - 1.31
        line_nom = ROOT.TLine(qdc_nom_th, 0.0, qdc_nom_th, max_y * 0.9)
        line_nom.SetLineColor(ROOT.kBlue + 2)
        line_nom.SetLineStyle(2)
        line_nom.SetLineWidth(2)
        line_nom.Draw("SAME")
        drawn3.append(line_nom)

        # Extract empirical 5th percentile cutoff of neighbor hits
        prob = [0.05]
        q_val = [0.0]
        import array
        a_prob = array.array('d', prob)
        a_q = array.array('d', q_val)
        h_neigh.GetQuantiles(1, a_q, a_prob)
        cutoff_5pct = a_q[0]

        leg3.AddEntry(h_seed, f"Cluster Seed Hits (Max QDC)", "l")
        leg3.AddEntry(h_neigh, f"Neighbor Hits (Probe, 5%: {cutoff_5pct:.2f} a.u.)", "f")
        leg3.AddEntry(line_nom, f"Nominal nphe_min=3.5 (Q={qdc_nom_th:.2f} a.u.)", "l")

    leg3.Draw()
    p3.Update()
    p3._drawn = drawn3

    # Pad 4: Cluster Width / Multiplicity vs Distance to SiPM
    p4 = c_calib.cd(4)
    p4.SetLogy(0)
    leg4 = ROOT.TLegend(0.40, 0.68, 0.88, 0.88)
    leg4.SetBorderSize(0)
    leg4.SetFillStyle(0)
    leg4.SetTextSize(0.028)

    drawn4 = []
    p_sz_dist = None
    f_sz_trend = None
    if "p_cl_size_vs_dist" in res:
        p_sz_dist = res["p_cl_size_vs_dist"].Clone(f"p_sz_vs_dist_{target_sample}")
        p_sz_dist.SetLineColor(res["color"])
        p_sz_dist.SetMarkerColor(res["color"])
        p_sz_dist.SetMarkerStyle(20)
        p_sz_dist.SetMarkerSize(0.8)
        p_sz_dist.SetLineWidth(2)
        p_sz_dist.GetXaxis().SetRangeUser(0.0, 40.0)
        p_sz_dist.GetYaxis().SetRangeUser(1.0, 3.0)
        p_sz_dist.Draw("E1")
        drawn4.append(p_sz_dist)

        # Fit linear decrease due to attenuation
        f_sz_trend = ROOT.TF1(f"f_sz_trend_{target_sample}", "pol1", 2.0, 36.0)
        f_sz_trend.SetLineColor(ROOT.kRed + 1)
        f_sz_trend.SetLineWidth(2)
        p_sz_dist.Fit(f_sz_trend, "RQS0")
        f_sz_trend.Draw("SAME")
        drawn4.append(f_sz_trend)

        leg4.AddEntry(p_sz_dist, f"{target_sample} Mean Size (Profile)", "lep")
        leg4.AddEntry(f_sz_trend, f"Slope: {f_sz_trend.GetParameter(1)*100:.2f} 10^{{-2}}/cm", "l")

    leg4.Draw()
    p4.Update()
    p4._drawn = drawn4

    # Save canvas to disk
    c_calib.SaveAs(os.path.join(out_dir, "scifi_hit_calibration.png"))
    c_calib.SaveAs(os.path.join(out_dir, "scifi_hit_calibration.pdf"))

    # Write to ROOT file in SciFiCalibration directory
    dir_calib.cd()
    c_calib.Write()
    if "h_sz" in locals() and h_sz:
        h_sz.Write()
    for h_st in drawn2:
        if isinstance(h_st, ROOT.TH1):
            h_st.Write()
    if "h_seed" in locals() and h_seed:
        h_seed.Write()
    if "h_neigh" in locals() and h_neigh:
        h_neigh.Write()
    if "line_nom" in locals() and line_nom:
        line_nom.Write("line_nominal_threshold")
    if p_sz_dist:
        p_sz_dist.Write()
    if f_sz_trend:
        f_sz_trend.Write()


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
    parser.add_argument("--cluster-max-dist", type=float, default=0.1, help="Max distance between SciFi cluster & SciFi track [cm] (default: 0.1 cm = 1 mm)")
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
            "alt_path": "data/run8329/sndsw_raw-1_1?_8329_muonReco.root",
            "geo": args.geo_data,
            "alt_geo": "data/geofile_sndlhc_TI18_V3_2023.root",
            "tree": "rawConv",
            "color": ROOT.kBlack,
            "is_mc": False,
            "max_files": 40,
            "avg_events_per_file": 27000,
        }
    if not args.no_pmu:
        samples["SingleMuMC"] = {
            "path": args.input_pmu,
            "alt_path": "data/pmu/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root",
            "geo": args.geo_pmu,
            "alt_geo": "data/geofile_pmu.root",
            "tree": "cbmsim",
            "color": ROOT.kBlue,
            "is_mc": True,
            "max_files": 1,
            "avg_events_per_file": 1213000,
        }
    if not args.no_tri:
        samples["ThreeMuMC"] = {
            "path": args.input_tri,
            "alt_path": "data/3mu/trimuon_digCPP-2??_hough_*.root",
            "geo": args.geo_tri,
            "alt_geo": "data/geofile_full.Ntuple-TGeant4_boost100.0.root",
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

    # Disable implicit multi-threading once event processing is complete.
    # This drains TBB thread pool cleanup tasks and prevents PyROOT GIL deadlocks
    # when ROOT's TH1::Fit attempts parallel evaluation via TBB.
    ROOT.DisableImplicitMT()

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
        ("dist_cluster", "h_cluster_dist_to_track", args.cluster_max_dist, f"Cut: {args.cluster_max_dist*10:.1f} mm"),
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

    row_keys = ["h2_cl_qdc_vs_dist", "h2_cls_qdc_vs_size", "h2_scifi_vs_mufi", "h2_track_slopes"]
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
        for key in ["h2_qdc_vs_dist", "h2_qdc_vs_dist_h", "h2_qdc_vs_dist_v", "h2_cl_qdc_vs_dist", "h2_cl_qdc_vs_dist_h", "h2_cl_qdc_vs_dist_v", "h2_cls_qdc_vs_size", "h2_scifi_vs_mufi", "h2_qdc_vs_hits", "h2_track_slopes", "h2_track_xy"]:
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

    # Landau-Vavilov Energy Loss & MPV Attenuation
    analyze_landau_vavilov(results, args.out_dir, out_file)
    # SciFi Hit Multiplicity & Threshold Calibration
    analyze_scifi_hit_calibration(results, args.out_dir, out_file)

    # Create dedicated top-level directory for Track-to-Channel Distances
    dir_distances = out_file.mkdir("TrackHitDistances")

    dist_keys = [
        "h_dist_scifi", "h_dist_scifi_zoom", "h_dist_scifi_h", "h_dist_scifi_v", "h_doca_scifi",
        "h_cluster_dist_to_track",
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
