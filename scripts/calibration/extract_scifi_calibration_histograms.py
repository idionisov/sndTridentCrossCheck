#!/usr/bin/env python3
import os
import sys
import time
import argparse
import tempfile
import shutil
from typing import Dict, Any, List, Tuple, Optional

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

from snd import DataManager, load_trident_libraries, add_progress_printer, ProgressPrinter
from config.calibration_histograms_config import (
    load_scifi_calibration_config,
    DEFAULT_YAML_PATH,
    CALIB_CONFIGS_1D,
    CALIB_CONFIGS_2D,
    CALIB_PROFILE_CONFIGS
)

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
    calib_cfg: Any,
    cuts_pipeline: List[Dict[str, Any]],
    hist_configs_1d: List[Tuple],
    hist_configs_2d: List[Tuple],
    profile_configs: List[Tuple],
    max_events: int = 0,
    progress_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[ROOT.RDataFrame, ROOT.snd.trident.MuonCalibrationProcessor, List[str], Any]:
    """
    Configures multi-threaded RDataFrame with compiled C++ MuonCalibrationProcessor.
    Applies the ordered cut pipeline dynamically as configured in YAML.
    """
    df = data.rdf()
    if max_events > 0:
        if ROOT.IsImplicitMTEnabled():
            print(f"[!] Warning: ROOT Implicit Multi-Threading is active ({ROOT.GetThreadPoolSize()} threads).")
            print(f"[!] ROOT RDataFrame does not support Range() in multi-thread mode.")
            print(f"[!] Applying software filter (rdfentry_ < {max_events}). Note: To stop processing immediately at {max_events:,} events, run with -j 1.")
            df = df.Filter(f"rdfentry_ < {max_events}")
        else:
            df = df.Range(max_events)

    progress_printer = None
    if progress_cfg and progress_cfg.get("enabled", True):
        total_ev = max_events if max_events > 0 else data.entries
        every_sec = float(progress_cfg.get("every_seconds", 30.0))
        every_ev = int(progress_cfg.get("every_events", 0))
        label = str(progress_cfg.get("label", "SciFi Calib"))
        df, progress_printer = add_progress_printer(
            df,
            total_events=total_ev,
            every_seconds=every_sec,
            every_events=every_ev,
            label=label,
        )

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
    processor = ROOT.snd.trident.MuonCalibrationProcessor(calib_cfg)
    df = df.Define("calib", processor, [track_branch, sf_branch, mf_branch])

    # 3. Setup IP1 bunch filter if EventHeader is present (for real collision data)
    has_header = data.has_branch("EventHeader.") or data.has_branch("EventHeader")
    header_col = "EventHeader." if data.has_branch("EventHeader.") else ("EventHeader" if data.has_branch("EventHeader") else None)
    if has_header and header_col:
        df = df.Define("is_ip1", ROOT.snd.trident.IP1Filter(), [header_col])

    # 4. Dynamically apply active cuts in the exact sequence specified by YAML
    applied_cut_labels = []
    for cut in cuts_pipeline:
        if not cut.get("enabled", False):
            continue

        # Check if cut requires event header (e.g. isIP1 for collision data)
        if cut.get("requires_header", False):
            if not has_header:
                continue

        filter_expr = cut["filter"]
        label = cut["label"]
        df = df.Filter(filter_expr, label)
        applied_cut_labels.append(label)

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

    # 5. Define flat scalar observables from calib struct required by booked histograms
    vector_cols = {"hit_qdc", "hit_distance", "hit_station", "hit_orientation", "plane_qdc", "plane_nhits", "plane_station"}
    needed_cols = set()
    for item in hist_configs_1d:
        needed_cols.add(item[0])
    for item in hist_configs_2d:
        needed_cols.add(item[8])  # x_var
        needed_cols.add(item[9])  # y_var
    for item in profile_configs:
        needed_cols.add(item[5])  # x_var
        needed_cols.add(item[6])  # y_var

    for v_name in sorted(list(needed_cols)):
        if v_name not in vector_cols:
            if data.has_branch(v_name):
                df_clean = df_clean.Redefine(v_name, f"calib.{v_name}")
            else:
                df_clean = df_clean.Define(v_name, f"calib.{v_name}")

    return df_clean, processor, applied_cut_labels, progress_printer


def book_calibration_histograms(
    df_node: ROOT.RDataFrame,
    hist_configs_1d: List[Tuple],
    hist_configs_2d: List[Tuple],
    profile_configs: List[Tuple],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Books 1D, 2D, and Profile histograms for calibration."""
    h1_ptrs: Dict[str, Any] = {}
    h2_ptrs: Dict[str, Any] = {}
    prof_ptrs: Dict[str, Any] = {}

    # 1. Book 1D Histograms
    for v_name, title, nbins, xmin, xmax, _ in hist_configs_1d:
        model = ROOT.RDF.TH1DModel(f"h1_{v_name}", title, nbins, xmin, xmax)
        h1_ptrs[v_name] = df_node.Histo1D(model, v_name)

    # 2. Book 2D Histograms
    for h2_name, title, nx, xmin, xmax, ny, ymin, ymax, x_var, y_var, _ in hist_configs_2d:
        model = ROOT.RDF.TH2DModel(h2_name, title, nx, xmin, xmax, ny, ymin, ymax)
        h2_ptrs[h2_name] = df_node.Histo2D(model, x_var, y_var)

    # 3. Book Profiles
    for prof_name, title, nbins_x, xmin, xmax, x_var, y_var, _ in profile_configs:
        model = ROOT.RDF.TProfile1DModel(prof_name, title, nbins_x, xmin, xmax)
        prof_ptrs[prof_name] = df_node.Profile1D(model, x_var, y_var)

    return h1_ptrs, h2_ptrs, prof_ptrs


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration File Argument Parser
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="SND@LHC SciFi Muon Calibration Histogram & Event Extraction Engine (RDataFrame)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", dest="config_file", default=DEFAULT_YAML_PATH,
        help=f"Master YAML configuration file (default: {DEFAULT_YAML_PATH})"
    )
    parser.add_argument(
        "-i", "--input", dest="input_data", default=None,
        help="Input reconstructed track ROOT files pattern (overrides YAML)"
    )
    parser.add_argument(
        "-t", "--tree", dest="tree_name", default=None,
        help="TTree name: 'cbmsim' or 'rawConv' (overrides YAML)"
    )
    parser.add_argument(
        "-o", "--output", dest="output_file", default=None,
        help="Output calibration histograms ROOT file (overrides YAML)"
    )
    parser.add_argument(
        "-n", "--max-events", dest="max_events", type=int, default=None,
        help="Max events limit, 0 = all (overrides YAML)"
    )
    parser.add_argument(
        "-f", "--n-files", dest="n_files", type=int, default=None,
        help="Max number of input files to process, <= 0 = all (overrides YAML)"
    )
    parser.add_argument(
        "-j", "--threads", "--workers", dest="num_threads", type=int, default=None,
        help="Number of RDataFrame worker threads (overrides YAML)"
    )
    parser.add_argument(
        "--scifi-threshold", "--scifi-qdc-min", dest="scifi_threshold", type=float, default=None,
        help="Offline SciFi hit threshold in p.e./QDC (overrides YAML)"
    )
    parser.add_argument(
        "--save-events", dest="save_events", action="store_true", default=None,
        help="Force saving passing events to output ROOT file (overrides YAML)"
    )
    parser.add_argument(
        "--no-save-events", dest="save_events", action="store_false", default=None,
        help="Do not save event tree, only extract histograms (overrides YAML)"
    )
    parser.add_argument(
        "--skip-scifi-hits-cut", dest="skip_scifi_hits_cut", action="store_true", default=None,
        help="Bypass legacy SciFi hits cut [10, 35] in the cut pipeline"
    )
    parser.add_argument(
        "--save-calib-branches", dest="save_calib_branches", action="store_true", default=None,
        help="Also save calib.* scalar columns in output TTree (overrides YAML)"
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # 2. Load Configuration from YAML
    # --------------------------------------------------------------------------
    cfg = load_scifi_calibration_config(args.config_file)
    job_cfg = cfg.get("job", {})
    params_cfg = cfg.get("parameters", {})
    pipeline_cfg = cfg.get("cuts", {}).get("pipeline", [])
    h1_cfg = [tuple(x) for x in cfg.get("histograms_1d", [])]
    h2_cfg = [tuple(x) for x in cfg.get("histograms_2d", [])]
    prof_cfg = [tuple(x) for x in cfg.get("profiles", [])]
    progress_cfg = job_cfg.get("progress", {})

    # Job & IO parameters (YAML defaults overridden by CLI arguments if specified)
    input_data = args.input_data if args.input_data is not None else job_cfg.get("input_data", "/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-*.root")
    tree_name = args.tree_name if args.tree_name is not None else job_cfg.get("tree_name", "")
    output_file = args.output_file if args.output_file is not None else job_cfg.get("output_file", "plots/scifi_calibration_histograms.root")
    max_events = args.max_events if args.max_events is not None else int(job_cfg.get("max_events", 0))
    n_files = args.n_files if args.n_files is not None else int(job_cfg.get("n_files", -1))
    num_threads = args.num_threads if args.num_threads is not None else int(job_cfg.get("num_threads", 8))
    save_events = args.save_events if args.save_events is not None else bool(job_cfg.get("save_events", True))
    save_calib_branches = args.save_calib_branches if args.save_calib_branches is not None else bool(job_cfg.get("save_calib_branches", False))
    tree_output_name = job_cfg.get("tree_output_name", "")

    # Physics parameters
    calib_cfg = ROOT.snd.trident.MuonCalibrationConfig()
    for k, v in params_cfg.items():
        if hasattr(calib_cfg, k):
            setattr(calib_cfg, k, v)
        elif k == "scifi_threshold" and hasattr(calib_cfg, "scifi_qdc_min"):
            calib_cfg.scifi_qdc_min = float(v)

    if args.scifi_threshold is not None:
        calib_cfg.scifi_qdc_min = float(args.scifi_threshold)

    if args.skip_scifi_hits_cut is True:
        for cut in pipeline_cfg:
            if cut.get("id") == "scifi_hits_range":
                cut["enabled"] = False

    # --------------------------------------------------------------------------
    # 3. Execution Summary Banner
    # --------------------------------------------------------------------------
    t0_all = time.time()
    print("=" * 80)
    print("SND@LHC SCIFI CALIBRATION HISTOGRAM & EVENT EXTRACTION ENGINE (RDataFrame)")
    print("=" * 80)
    print(f"Master Configuration: {args.config_file}")
    print(f"Input Pattern       : {input_data}")
    print(f"TTree Name          : {tree_name if tree_name else 'Auto-detect (rawConv / cbmsim)'}")
    print(f"Max Events          : {max_events if max_events > 0 else 'All'}")
    print(f"Max Files           : {n_files if n_files > 0 else 'All'}")
    print(f"Worker Threads      : {num_threads}")
    print(f"Output File         : {output_file}")
    print("-" * 80)
    print("Physics Parameters:")
    print(f"  SciFi Track Type  : {calib_cfg.scifi_track_type} (11 = Hough)")
    print(f"  DS Track Type     : {calib_cfg.ds_track_type} (13 = Hough)")
    print(f"  Track #chi2/ndf   : SciFi <= {calib_cfg.chi2_max_scifi}, DS <= {calib_cfg.chi2_max_ds}")
    print(f"  Max Angular Slope : < {calib_cfg.max_slope} rad")
    print(f"  Fiducial Plane    : z = {calib_cfg.z_match:.1f} cm (margin >= {calib_cfg.fiducial_margin} cm)")
    print(f"  DS Matching       : dR <= {calib_cfg.pos_match_max} cm, dTheta <= {calib_cfg.angle_match_max} rad")
    print("-" * 80)
    print("Active Cut Sequence Pipeline (in execution order):")
    active_cuts_count = 0
    for idx, cut in enumerate(pipeline_cfg, start=1):
        status_str = "[ACTIVE]" if cut.get("enabled", False) else "[BYPASSED]"
        print(f"  {idx:2d}. {status_str:10s} {cut['id']:25s} : {cut['label']}")
        if cut.get("enabled", False):
            active_cuts_count += 1
    print(f"Total Active Cuts: {active_cuts_count} / {len(pipeline_cfg)}")
    print("=" * 80)

    # 4. Multi-Threading Control
    if num_threads > 1:
        ROOT.EnableImplicitMT(num_threads)
        print(f"[*] Enabled ROOT Implicit Multi-Threading with {num_threads} worker threads.")
    else:
        ROOT.DisableImplicitMT()
        print(f"[*] Implicit Multi-Threading disabled (single thread execution).")

    # 5. Setup DataManager
    data = DataManager(input_data, tree_name=tree_name, num_threads=num_threads, n_files=n_files)
    print(f"[*] Resolved {data.num_files:,} files | Active tree: '{data.tree_name}'")

    # 6. Setup Calibration DataFrame and Dynamic Filter Pipeline
    df_clean, processor, applied_cut_labels, progress_printer = setup_calibration_dataframe(
        data, calib_cfg, pipeline_cfg,
        h1_cfg, h2_cfg, prof_cfg,
        max_events=max_events,
        progress_cfg=progress_cfg
    )

    # 7. Prepare Atomic Temporary Container
    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
        tmp_path = tmp.name

    out_tree_name = tree_output_name if tree_output_name else data.tree_name

    # 8. Book Histograms and Profiles
    h1_books, h2_books, prof_books = book_calibration_histograms(df_clean, h1_cfg, h2_cfg, prof_cfg)
    count_book = df_clean.Count()

    # 9. Execute Event Loop across Worker Threads
    print("[*] RDataFrame computational graph booked. Executing multi-threaded event loop...")
    t_loop = time.time()
    n_clean_muons = count_book.GetValue()
    if progress_printer:
        progress_printer.PrintSummary()
    print(f"[✓] Event loop completed in {time.time() - t_loop:.2f} s")
    print(f"[✓] Clean single through-going muons selected: {n_clean_muons:,}")

    # Print Cutflow Report
    print("\n--- RDataFrame Cutflow Report ---")
    cutflow_rep = df_clean.Report()
    cutflow_rep.Print()

    # Optional Snapshot to write events (executed only if passing events exist to prevent MT empty-tree crashes)
    events_saved = False
    if save_events:
        if n_clean_muons > 0:
            branches_to_save = sorted(list(set(data.branch_names)))
            if save_calib_branches:
                vector_cols = {"hit_qdc", "hit_distance", "hit_station", "hit_orientation", "plane_qdc", "plane_nhits", "plane_station"}
                for item in h1_cfg:
                    v_name = item[0]
                    if v_name not in vector_cols and v_name not in branches_to_save:
                        branches_to_save.append(v_name)

            opts = ROOT.RDF.RSnapshotOptions()
            opts.fLazy = False
            opts.fMode = "RECREATE"
            print(f"[*] Snapshotting {n_clean_muons:,} passing events ({len(branches_to_save)} branches) to {tmp_path}...")
            t_snap = time.time()
            df_clean.Snapshot(out_tree_name, tmp_path, branches_to_save, opts)
            events_saved = True
            print(f"[✓] Event snapshot completed in {time.time() - t_snap:.2f} s")
        else:
            print("[*] No events survived the selection pipeline. Skipping event snapshot.")

    # 10. Materialize ROOT objects into local memory
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
    # 11. Safe Hierarchical Atomic File Writing (EOS Safe)
    # ==============================================================================
    print(f"\n[*] Serializing histograms to atomic container: {tmp_path}")
    if events_saved:
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

    shutil.move(tmp_path, output_file)
    print(f"[✓] Successfully committed output ROOT file ({fsize / (1024*1024):.2f} MB):")
    print(f"    --> {output_file}")
    if save_events:
        print(f"[✓] Saved TTree '{out_tree_name}' with {n_clean_muons:,} events.")
    print(f"[✓] Extracted {len(hist_1d)} 1D histograms, {len(hist_2d)} 2D histograms, {len(profiles)} Profiles.")
    print(f"[✓] Total elapsed extraction time: {time.time() - t0_all:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
