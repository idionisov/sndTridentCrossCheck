#!/usr/bin/env python3
"""
================================================================================
SND@LHC SciFi Muon Calibration Histogram Extraction Engine
================================================================================
High-throughput calibration distribution extractor supporting 1D, 2D, and
TProfile observables for clean single through-going muons in data and MC.

Features:
- Multi-threaded event processing via ROOT RDataFrame and DataManager.
- Selection of clean single muons using compiled C++ MuonCalibrationProcessor.
- Books and fills all SciFi hit charge, plane sum, attenuation, and track metrics.
- Saves raw histograms to ROOT file (Histograms/1D, Histograms/2D, Histograms/Profiles)
  for downstream fitting and calibration studies in Python.
- Safe hierarchical directory creation and atomic file writing (safe on EOS).

Author: SND@LHC Collaboration
================================================================================
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
from typing import Dict, Any, List, Tuple

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Add project root to sys.path
_curr = os.path.abspath(os.path.dirname(__file__))
_repo_root = _curr
while _curr != "/" and _curr != os.path.dirname(_curr):
    if os.path.exists(os.path.join(_curr, "analysis")) and os.path.exists(os.path.join(_curr, "snd")):
        _repo_root = _curr
        break
    _curr = os.path.dirname(_curr)

if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import DataManager, load_trident_libraries
from config.calibration_histograms_config import CALIB_CONFIGS_1D, CALIB_CONFIGS_2D, CALIB_PROFILE_CONFIGS

# Load compiled C++ analysis libraries (libtrident_analysis.so)
load_trident_libraries(_repo_root)


def get_or_create_dir(parent_tfile_or_dir, path_str: str):
    """Safely navigates or creates nested directory hierarchy in ROOT."""
    parts = [p for p in path_str.split("/") if p]
    curr = parent_tfile_or_dir
    for part in parts:
        sub = curr.GetDirectory(part)
        if not sub:
            sub = curr.mkdir(part)
        curr = sub
    return curr


def setup_calibration_dataframe(
    data: DataManager,
    config: Any,
    max_events: int = 0,
    skip_scifi_hits_cut: bool = True,
    ip1_only: bool = False,
) -> Tuple[ROOT.RDataFrame, ROOT.snd.trident.MuonCalibrationProcessor]:
    """
    Configures multi-threaded RDataFrame with compiled C++ MuonCalibrationProcessor.
    Dynamically identifies hit and track branches across sndsw production formats.
    """
    df = data.df()
    if max_events > 0:
        if ROOT.IsImplicitMTEnabled():
            df = df.Filter(f"rdfentry_ < {max_events}")
        else:
            df = df.Range(max_events)

    # Optional IP1 bunch timing selection for collision data
    if ip1_only and data.has_event_header():
        df = df.Filter("EventHeader.isIP1()", "Stage 0: IP1 Collision Bunch Crossing")

    # 1. Resolve Track and Hit Branch Names dynamically
    track_branch = "Reco_MuonTracks" if data.has_branch("Reco_MuonTracks") else "fittedTracks"
    sf_branch = "Digi_ScifiHits" if data.has_branch("Digi_ScifiHits") else "Digits_Scifi"
    if data.has_branch("Digi_MuFilterHits"):
        mf_branch = "Digi_MuFilterHits"
    elif data.has_branch("Digi_MuFilterHit"):
        mf_branch = "Digi_MuFilterHit"
    else:
        mf_branch = "Digits_MuFilter"

    # 2. Extract Calibration Metrics using compiled C++ processor
    processor = ROOT.snd.trident.MuonCalibrationProcessor(config)
    df = df.Define("calib", processor, [track_branch, sf_branch, mf_branch])

    # 3. Filter for Clean Single Through-Going Muon Tracks sequentially
    df = df.Filter("calib.pass_cut_single_scifi_track", "Stage 1: Exactly 1 Reconstructed SciFi Track")
    df = df.Filter("calib.pass_cut_chi2", f"Stage 2: SciFi Track Chi2/ndf <= {config.chi2_max}")
    df = df.Filter("calib.pass_cut_slope", f"Stage 3: Max Angular Slope <= {config.max_slope} rad")
    df = df.Filter("calib.pass_cut_fiducial", f"Stage 4: Fiducial Boundary Margin >= {config.fiducial_margin} cm")
    df = df.Filter("calib.pass_cut_ds_match", f"Stage 5: DS Track Slope Match <= {config.ds_match_slope_max} rad")
    if not skip_scifi_hits_cut:
        df = df.Filter("calib.pass_cut_scifi_hits", f"Stage 6: SciFi Total Hits [{config.scifi_hits_min}, {config.scifi_hits_max}]")

    df_clean = df

    # 4. Define vector observables for unrolled hit/plane histograms & profiles
    df_clean = (
        df_clean
        .Define("hit_qdc",         "calib.get_hit_qdc()")
        .Define("hit_distance",    "calib.get_hit_distance()")
        .Define("hit_station",     "calib.get_hit_station()")
        .Define("hit_orientation", "calib.get_hit_orientation()")
        .Define("plane_qdc",       "calib.get_plane_qdc()")
        .Define("plane_nhits",     "calib.get_plane_nhits()")
        .Define("plane_station",   "calib.get_plane_station()")
    )

    # 5. Define flat scalar observables from calib struct
    vector_cols = {"hit_qdc", "hit_distance", "hit_station", "hit_orientation", "plane_qdc", "plane_nhits", "plane_station"}
    for v_name, _, _, _, _, _ in CALIB_CONFIGS_1D:
        if v_name not in vector_cols:
            if data.has_branch(v_name):
                df_clean = df_clean.Redefine(v_name, f"calib.{v_name}")
            else:
                df_clean = df_clean.Define(v_name, f"calib.{v_name}")

    return df_clean, processor


def book_calibration_histograms(df_node: ROOT.RDataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Books 1D, 2D, and Profile histograms for calibration."""
    h1_ptrs: Dict[str, Any] = {}
    h2_ptrs: Dict[str, Any] = {}
    prof_ptrs: Dict[str, Any] = {}

    # 1. Book 1D Histograms
    for v_name, title, nbins, xmin, xmax, _ in CALIB_CONFIGS_1D:
        model = ROOT.RDF.TH1DModel(f"h1_{v_name}", title, nbins, xmin, xmax)
        h1_ptrs[v_name] = df_node.Histo1D(model, v_name)

    # 2. Book 2D Histograms
    for h2_name, title, nx, xmin, xmax, ny, ymin, ymax, x_var, y_var, _ in CALIB_CONFIGS_2D:
        model = ROOT.RDF.TH2DModel(h2_name, title, nx, xmin, xmax, ny, ymin, ymax)
        h2_ptrs[h2_name] = df_node.Histo2D(model, x_var, y_var)

    # 3. Book Profiles
    for prof_name, title, nbins_x, xmin, xmax, x_var, y_var, _ in CALIB_PROFILE_CONFIGS:
        model = ROOT.RDF.TProfile1DModel(prof_name, title, nbins_x, xmin, xmax)
        prof_ptrs[prof_name] = df_node.Profile1D(model, x_var, y_var)

    return h1_ptrs, h2_ptrs, prof_ptrs


def main():
    parser = argparse.ArgumentParser(
        description="Multi-threaded SciFi QDC & Attenuation Histogram Extractor (RDataFrame)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--input", dest="input_data",
                        default="/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-*.root",
                        help="Input reconstructed track ROOT files pattern (default: run 8329)")
    parser.add_argument("-t", "--tree", dest="tree_name", default="",
                        help="TTree name: 'cbmsim' or 'rawConv' (default: auto-detect)")
    parser.add_argument("-o", "--output", dest="output_file",
                        default="plots/scifi_calibration_histograms.root",
                        help="Output calibration histograms ROOT file (default: plots/scifi_calibration_histograms.root)")
    parser.add_argument("-n", "--max-events", dest="max_events", type=int, default=0,
                        help="Max events limit (0 = all)")
    parser.add_argument("-j", "--threads", "--workers", dest="num_threads", type=int, default=8,
                        help="Number of RDataFrame worker threads (default: 8)")

    # Configurable clean muon selection cuts
    parser.add_argument("--chi2-max", type=float, default=10.0, help="Max track chi2/ndf (default: 10.0)")
    parser.add_argument("--max-slope", type=float, default=0.05, help="Max angular slope dx/dz, dy/dz (default: 0.05)")
    parser.add_argument("--scifi-min", type=int, default=10, help="Min SciFi hits (default: 10)")
    parser.add_argument("--scifi-max", type=int, default=35, help="Max SciFi hits (default: 35)")
    parser.add_argument("--scifi-threshold", "--scifi-qdc-min", dest="scifi_threshold", type=float, default=0.0,
                        help="Offline SciFi hit threshold [p.e. / QDC] (default: 0.0, uses default hit.isValid())")
    parser.add_argument("--ds-min", type=int, default=2, help="Min Downstream MuFilter hits (default: 2)")
    parser.add_argument("--ds-match-slope", type=float, default=0.04, help="Max slope diff between SF and DS tracks (default: 0.04)")
    parser.add_argument("--fiducial-margin", type=float, default=1.5, help="Fiducial margin in cm (default: 1.5)")

    # Event saving and cut bypass options
    parser.add_argument("--save-events", "--copy-events", "--save-tree", dest="save_events", action="store_true", default=True,
                        help="Copy and store passing events with original branches into output ROOT file (default: True)")
    parser.add_argument("--no-save-events", dest="save_events", action="store_false",
                        help="Do not save event tree, only extract histograms")
    parser.add_argument("--skip-scifi-hits-cut", "--no-scifi-hits-cut", dest="skip_scifi_hits_cut", action="store_true", default=True,
                        help="Skip the final SciFi hits [10, 35] cut (Stage 6) to preserve full hit spectrum (default: True)")
    parser.add_argument("--apply-scifi-hits-cut", dest="skip_scifi_hits_cut", action="store_false",
                        help="Apply the final SciFi hits [10, 35] cut")
    parser.add_argument("--ip1-only", dest="ip1_only", action="store_true", default=False,
                        help="Filter data for IP1 collision bunches (default: False)")
    parser.add_argument("--save-calib-branches", dest="save_calib_branches", action="store_true", default=False,
                        help="Also save calib.* scalar columns in the output TTree (default: False, only original branches)")
    parser.add_argument("--tree-output-name", dest="tree_output_name", default="",
                        help="Output tree name (default: same as input tree, e.g. rawConv or cbmsim)")

    args = parser.parse_args()

    t0_all = time.time()
    print("=" * 80)
    print("SND@LHC SCIFI CALIBRATION HISTOGRAM & EVENT EXTRACTION ENGINE (RDataFrame)")
    print("=" * 80)
    print(f"Input Pattern      : {args.input_data}")
    print(f"TTree Name         : {args.tree_name if args.tree_name else 'Auto-detect (rawConv / cbmsim)'}")
    print(f"Max Events         : {args.max_events if args.max_events > 0 else 'All'}")
    print(f"Worker Threads     : {args.num_threads}")
    print(f"Output File        : {args.output_file}")
    print("-" * 80)
    print("Selection Criteria:")
    if args.ip1_only:
        print("  Bunch Crossing Timing    : IP1 Collision Bunches only (EventHeader.isIP1())")
    print(f"  SciFi Track Multiplicity : Exactly 1 Reconstructed SciFi Track")
    print(f"  SciFi Track #chi2/ndf    : <= {args.chi2_max}")
    print(f"  Max Angular Slope        : <= {args.max_slope} rad (~3 deg)")
    print(f"  DS Track Slope Match     : |#Delta slope| <= {args.ds_match_slope}")
    print(f"  Fiducial Margin          : >= {args.fiducial_margin} cm from borders")
    if args.skip_scifi_hits_cut:
        print(f"  SciFi Total Hits         : [SKIPPED - Preserving full hit multiplicity for threshold studies]")
    else:
        print(f"  SciFi Total Hits         : [{args.scifi_min}, {args.scifi_max}]")
    print(f"  DS System Hits           : >= {args.ds_min}")
    print(f"  Save Filtered Events     : {args.save_events} (tree: '{args.tree_output_name if args.tree_output_name else 'auto-detect'}')")
    print("=" * 80)

    # 1. Enable Multi-Threading
    if args.num_threads > 1:
        ROOT.EnableImplicitMT(args.num_threads)
        print(f"[*] Enabled ROOT Implicit Multi-Threading with {args.num_threads} worker threads.")

    # 2. Setup DataManager
    data = DataManager(args.input_data, tree_name=args.tree_name, num_threads=args.num_threads)
    print(f"[*] Resolved {data.num_files:,} files | Active tree: '{data.tree_name}'")

    # 3. Setup Calibration Config & RDataFrame
    calib_cfg = ROOT.snd.trident.MuonCalibrationConfig()
    calib_cfg.chi2_max = args.chi2_max
    calib_cfg.max_slope = args.max_slope
    calib_cfg.fiducial_margin = args.fiducial_margin
    calib_cfg.scifi_hits_min = args.scifi_min
    calib_cfg.scifi_hits_max = args.scifi_max
    calib_cfg.ds_hits_min = args.ds_min
    calib_cfg.ds_match_slope_max = args.ds_match_slope
    calib_cfg.scifi_qdc_min = args.scifi_threshold

    df_clean, processor = setup_calibration_dataframe(
        data, calib_cfg, args.max_events,
        skip_scifi_hits_cut=args.skip_scifi_hits_cut,
        ip1_only=args.ip1_only
    )

    # 4. Prepare Atomic Temporary Container
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
        tmp_path = tmp.name

    out_tree_name = args.tree_output_name if args.tree_output_name else data.tree_name

    # Optional Lazy Snapshot to write events during the same event loop pass
    snap_book = None
    if args.save_events:
        branches_to_save = sorted(list(set(data.branch_names)))
        if args.save_calib_branches:
            vector_cols = {"hit_qdc", "hit_distance", "hit_station", "hit_orientation", "plane_qdc", "plane_nhits", "plane_station"}
            for v_name, _, _, _, _, _ in CALIB_CONFIGS_1D:
                if v_name not in vector_cols and v_name not in branches_to_save:
                    branches_to_save.append(v_name)

        opts = ROOT.RDF.RSnapshotOptions()
        opts.fLazy = True
        opts.fMode = "RECREATE"
        print(f"[*] Booking lazy snapshot of tree '{out_tree_name}' ({len(branches_to_save)} branches) to {tmp_path}...")
        snap_book = df_clean.Snapshot(out_tree_name, tmp_path, branches_to_save, opts)

    # 5. Book Histograms and Profiles
    h1_books, h2_books, prof_books = book_calibration_histograms(df_clean)
    count_book = df_clean.Count()

    # 6. Execute Event Loop across Worker Threads
    print("[*] RDataFrame computational graph booked. Executing multi-threaded event loop...")
    t_loop = time.time()
    n_clean_muons = count_book.GetValue()
    print(f"[✓] Event loop completed in {time.time() - t_loop:.2f} s")
    print(f"[✓] Clean single through-going muons selected: {n_clean_muons:,}")

    # Print Cutflow Report
    print("\n--- RDataFrame Cutflow Report ---")
    cutflow_rep = df_clean.Report()
    cutflow_rep.Print()

    # 7. Materialize ROOT objects into local memory
    hist_1d: Dict[str, ROOT.TH1D] = {}
    for name, ptr in h1_books.items():
        h = ptr.GetValue()
        h.SetDirectory(0)
        hist_1d[name] = h

    hist_2d: Dict[str, ROOT.TH2D] = {}
    for name, ptr in h2_books.items():
        h2 = ptr.GetValue()
        h2.SetDirectory(0)
        hist_2d[name] = h2

    profiles: Dict[str, ROOT.TProfile] = {}
    for name, ptr in prof_books.items():
        p = ptr.GetValue()
        p.SetDirectory(0)
        profiles[name] = p

    # Build cutflow summary histogram
    cuts = list(cutflow_rep)
    h_cutflow = ROOT.TH1D("h_cutflow", "Clean Muon Selection Cutflow;Cut Step;Events", len(cuts) + 1, 0.5, len(cuts) + 1.5)
    h_cutflow.SetDirectory(0)
    if len(cuts) > 0:
        h_cutflow.GetXaxis().SetBinLabel(1, "All Processed")
        h_cutflow.SetBinContent(1, cuts[0].GetAll())
        for idx, c in enumerate(cuts, start=2):
            h_cutflow.GetXaxis().SetBinLabel(idx, c.GetName())
            h_cutflow.SetBinContent(idx, c.GetPass())

    # ==============================================================================
    # 8. Safe Hierarchical Atomic File Writing (EOS Safe)
    # ==============================================================================
    print(f"\n[*] Serializing histograms to atomic container: {tmp_path}")
    if args.save_events:
        tfile = ROOT.TFile(tmp_path, "UPDATE")
    else:
        tfile = ROOT.TFile(tmp_path, "RECREATE")

    # A. Save Cutflow
    h_cutflow.Write()

    # B. Save 1D Histograms
    dir_1d = get_or_create_dir(tfile, "Histograms/1D")
    dir_1d.cd()
    for h in hist_1d.values():
        h.Write()

    # C. Save 2D Histograms
    dir_2d = get_or_create_dir(tfile, "Histograms/2D")
    dir_2d.cd()
    for h2 in hist_2d.values():
        h2.Write()

    # D. Save Profiles
    dir_prof = get_or_create_dir(tfile, "Histograms/Profiles")
    dir_prof.cd()
    for p in profiles.values():
        p.Write()

    tfile.Flush()
    tfile.Close()

    # Verify size and move atomically
    fsize = os.path.getsize(tmp_path)
    if fsize < 1000:
        os.remove(tmp_path)
        raise RuntimeError(f"Output file size is suspiciously small ({fsize} bytes). Aborting commit!")

    shutil.move(tmp_path, args.output_file)
    print(f"[✓] Successfully committed output ROOT file ({fsize / (1024*1024):.2f} MB):")
    print(f"    --> {args.output_file}")
    if args.save_events:
        print(f"[✓] Saved TTree '{out_tree_name}' with {n_clean_muons:,} events.")
    print(f"[✓] Extracted {len(hist_1d)} 1D histograms, {len(hist_2d)} 2D histograms, {len(profiles)} Profiles.")
    print(f"[✓] Total elapsed extraction time: {time.time() - t0_all:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
