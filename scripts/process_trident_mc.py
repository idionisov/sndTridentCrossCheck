#!/usr/bin/env python3
"""
filter_tridents_mc.py
---------------------
Extracts and filters Monte Carlo trident signal events into skimmed ROOT files
with truth observables and optional fiducial containment cuts.
Utilizes snd::trident::TridentTruthProcessor and snd.DataManager.
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
import ROOT

# Add repository root to python search path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from snd import (
    DataManager,
    create_truth_config,
    attach_truth_branches,
)


def extract_file_tag(filename: str) -> str:
    """
    Extracts the numeric identifier or clean tag from filename.
    Examples:
      - '...filteredAtScoringPlane_digCPP-200.root' -> '200'
      - '...filteredAtScoringPlane_digCPP-255.root' -> '255'
      - 'sndsw_raw-0000_looseCuts7.root' -> '0000'
    """
    import re
    base = os.path.basename(filename)
    match_dig = re.search(r"digCPP-(\d+)", base)
    if match_dig:
        return match_dig.group(1)
    match_num = re.search(r"(?:raw|run)[-_](\d+)", base, re.IGNORECASE)
    if match_num:
        return match_num.group(1)
    match_trailing = re.search(r"[-_](\d+)\.root$", base)
    if match_trailing:
        return match_trailing.group(1)
    return os.path.splitext(base)[0]


def process_single_file(
    input_file: str,
    output_file: str,
    processor: ROOT.snd.trident.TridentTruthProcessor,
    store_all_events: bool = False,
    use_fiducial: bool = False,
    num_threads: int = 6,
    max_events: Optional[int] = None,
):
    """
    Processes a single Monte Carlo ROOT file using RDataFrame and TridentTruthProcessor.
    Snapshots filtered events and copies non-tree metadata (ShipGeo).
    """
    t0 = time.time()

    data = DataManager(input_file, num_threads=num_threads)
    total_events = data.entries
    tree_name = data.tree_name
    df = data.df()
    if max_events is not None and max_events > 0:
        df = df.Filter(f"rdfentry_ < {max_events}")
        total_events = min(total_events, max_events)

    # Define structured truth analysis
    df = df.Define("truth", processor, ["MCTrack"])

    if not store_all_events:
        if use_fiducial:
            df_base = df.Filter("truth.hasCandidate && truth.isFiducial")
        else:
            df_base = df.Filter("truth.hasCandidate")
    else:
        df_base = df

    # Attach flattened legacy scalar branches for 100% backward compatibility
    df_filtered = attach_truth_branches(df_base, truth_col="truth")

    # Book summary counters
    c_total = df_filtered.Count()
    df_sig = df_filtered.Filter("is_signal == true")
    c_sig = df_sig.Count()
    c_fid = df_sig.Filter("is_fiducial == true").Count()
    c_genuine = df_sig.Filter("proc_type == 0").Count()
    c_gamma = df_sig.Filter("proc_type == 1").Count()
    c_annihil = df_sig.Filter("proc_type == 2").Count()
    c_sec = df_sig.Filter("(trident_flags & 0x10) != 0").Count()
    c_rock = df_sig.Filter("region_type == 1").Count()
    c_target = df_sig.Filter("region_type == 2").Count()
    c_muon = df_sig.Filter("region_type == 3").Count()

    # Snapshot to output ROOT file
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fMode = "RECREATE"
    df_filtered.Snapshot(tree_name, output_file, "", opts)

    stored_count = c_total.GetValue()
    sig_count = c_sig.GetValue()
    fid_count = c_fid.GetValue()
    gen_count = c_genuine.GetValue()
    gam_count = c_gamma.GetValue()
    ann_count = c_annihil.GetValue()
    sec_count = c_sec.GetValue()
    r_count = c_rock.GetValue()
    t_count = c_target.GetValue()
    m_count = c_muon.GetValue()

    # Copy metadata objects (ShipGeo, etc.)
    DataManager.copy_metadata(input_file, output_file, exclude_trees=[tree_name])

    elapsed = time.time() - t0
    rate = total_events / elapsed if elapsed > 0 else 0
    print(f"  Processed {total_events:,} events in {elapsed:.1f}s ({rate:.0f} ev/s)")
    print(f"  Output Tree Entries: {stored_count:,} (Signal: {sig_count:,} | Fiducial: {fid_count:,})")
    print(f"    - Processes : Genuine: {gen_count} | Gamma: {gam_count} | Annihil: {ann_count} | Sec: {sec_count}")
    print(f"    - Regions   : Rock: {r_count} | Target: {t_count} | MuonSystem: {m_count}")

    return (
        total_events, stored_count, sig_count, fid_count,
        gen_count, gam_count, ann_count, sec_count,
        r_count, t_count, m_count
    )


def main():
    default_input = "/eos/experiment/sndlhc/MonteCarlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-2*.root"
    default_output = "trimuon_filtered_%s.root"

    parser = argparse.ArgumentParser(
        description="Extract and save trident signal events into ROOT files with comprehensive physics observables and fiducial area constraints."
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_pattern",
        nargs="?",
        default=default_input,
        help="Input ROOT file path or wildcard pattern (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_pattern",
        default=default_output,
        help="Output ROOT file path or pattern with %%s for tag (default: %(default)s)"
    )
    parser.add_argument(
        "--store-all-events", "--keep-all",
        dest="store_all_events",
        action="store_true",
        default=False,
        help="If set, stores all events in the input files (both signal and non-signal) with extra observables without filtering (default: False)"
    )
    parser.add_argument(
        "--fiducial", "--fiducial-area",
        dest="fiducial_mode",
        choices=["unconstrained", "default", "custom"],
        default="unconstrained",
        help="Fiducial area constraint for 3-muon containment: 'unconstrained' (default), 'default' (z=320, x in [-42, -11], y in [18, 49]), or 'custom' (default: %(default)s)"
    )
    parser.add_argument(
        "--fiducial-box",
        dest="fiducial_box",
        nargs=5,
        type=float,
        default=[320.0, -42.0, -11.0, 18.0, 49.0],
        metavar=("Z", "X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Custom fiducial plane and rectangular boundaries [Z, X_MIN, X_MAX, Y_MIN, Y_MAX] in cm (default: 320.0 -42.0 -11.0 18.0 49.0)"
    )
    parser.add_argument(
        "--z-range",
        dest="target_z_range",
        nargs=2,
        type=float,
        default=[260.0, 355.0],
        metavar=("Z_MIN", "Z_MAX"),
        help="Target region Z boundaries [Z_MIN, Z_MAX] in cm (default: 260.0 355.0)"
    )
    parser.add_argument(
        "--region",
        dest="region",
        choices=["all", "rock", "target", "muon_system", "custom"],
        default="all",
        help="Volume region filter: 'all' (all Z), 'rock' (Z < target_z_min), 'target' (target_z_min <= Z < target_z_max), 'muon_system' (Z >= target_z_max), or 'custom' (default: %(default)s)"
    )
    parser.add_argument(
        "--z-min",
        dest="z_min",
        type=float,
        default=None,
        help="Minimum Z vertex coordinate in cm (overrides region preset if set)"
    )
    parser.add_argument(
        "-z", "--z-max", "--rock_boundary",
        dest="z_max",
        type=float,
        default=None,
        help="Maximum Z vertex coordinate in cm (overrides region preset if set)"
    )
    parser.add_argument(
        "--processes",
        dest="processes",
        choices=["all", "genuine", "gamma", "annihil"],
        default="all",
        help="Filter process: 'all', 'genuine' (MuonToMuonPair), 'gamma' (GammaToMuPair), 'annihil' (Positron Annihilation) (default: %(default)s)"
    )
    parser.add_argument(
        "-n", "--events", "--max-events",
        dest="max_events",
        type=int,
        default=None,
        help="Maximum number of events to process per file (for testing) (default: all)"
    )
    parser.add_argument(
        "-j", "--threads",
        dest="num_threads",
        type=int,
        default=6,
        help="Number of worker threads for RDataFrame (default: %(default)s)"
    )

    args = parser.parse_args()

    # Unpack target Z boundaries
    target_z_min, target_z_max = args.target_z_range

    # Determine Z boundaries for filtering
    if args.region == "rock":
        z_min = -1e9
        z_max = target_z_min
    elif args.region == "target":
        z_min = target_z_min
        z_max = target_z_max
    elif args.region == "muon_system":
        z_min = target_z_max
        z_max = 1e9
    elif args.region == "all":
        z_min = -1e9
        z_max = 1e9
    else:  # custom
        z_min = -1e9
        z_max = target_z_min

    if args.z_min is not None:
        z_min = args.z_min
    if args.z_max is not None:
        z_max = args.z_max

    # Determine process bitmask
    if args.processes == "genuine":
        process_mask = 1 << 0
    elif args.processes == "gamma":
        process_mask = 1 << 1
    elif args.processes == "annihil":
        process_mask = 1 << 2
    else:
        process_mask = (1 << 0) | (1 << 1) | (1 << 2)

    use_fiducial = (args.fiducial_mode != "unconstrained")
    if args.fiducial_mode == "custom" or args.fiducial_box != [320.0, -42.0, -11.0, 18.0, 49.0]:
        fid_box = args.fiducial_box
        use_fiducial = True
    else:
        fid_box = [320.0, -42.0, -11.0, 18.0, 49.0]

    matched_files = sorted(glob.glob(args.input_pattern))
    if not matched_files:
        if os.path.exists(args.input_pattern):
            matched_files = [args.input_pattern]
        else:
            print(f"Error: No files found matching pattern: {args.input_pattern}")
            sys.exit(1)

    print("=" * 65)
    print("SND@LHC MC Trident Signal Event Extractor")
    print("=" * 65)
    print(f"Input Pattern     : {args.input_pattern}")
    print(f"Files Found       : {len(matched_files)}")
    print(f"Store All Events  : {args.store_all_events}")
    print(f"Target Z Region   : [{target_z_min:.1f}, {target_z_max:.1f}] cm")
    print(f"Region Filter     : {args.region} ({z_min:.1f} cm <= Z < {z_max:.1f} cm)")
    print(f"Process Filter    : {args.processes}")
    print(f"Fiducial Area     : {'ENABLED' if use_fiducial else 'UNCONSTRAINED'}")
    if use_fiducial:
        print(f"  * Fiducial Plane: Z = {fid_box[0]:.1f} cm")
        print(f"  * X Boundary    : [{fid_box[1]:.1f}, {fid_box[2]:.1f}] cm")
        print(f"  * Y Boundary    : [{fid_box[3]:.1f}, {fid_box[4]:.1f}] cm")
    print(f"Worker Threads    : {args.num_threads}")
    print("=" * 65)

    truth_config = create_truth_config(
        z_min=z_min,
        z_max=z_max,
        target_z_min=target_z_min,
        target_z_max=target_z_max,
        process_mask=process_mask,
        weight_scale=1.0,
        use_fiducial=use_fiducial,
        fid_box=fid_box,
    )
    processor = ROOT.snd.trident.TridentTruthProcessor(truth_config)

    overall_t0 = time.time()
    grand_total_events = 0
    grand_total_stored = 0
    grand_total_signal = 0
    grand_total_fiducial = 0

    grand_total_genuine = 0
    grand_total_gamma = 0
    grand_total_annihil = 0
    grand_total_sec = 0

    grand_total_rock = 0
    grand_total_target = 0
    grand_total_muon = 0

    created_files = []

    for idx, input_file in enumerate(matched_files, 1):
        tag = extract_file_tag(input_file)
        if "%s" in args.output_pattern:
            out_file = args.output_pattern % tag
        elif len(matched_files) > 1:
            base_name, ext = os.path.splitext(args.output_pattern)
            out_file = f"{base_name}_{tag}{ext}"
        else:
            out_file = args.output_pattern

        if not out_file.endswith(".root"):
            out_file += ".root"

        out_dir = os.path.dirname(os.path.abspath(out_file))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{idx}/{len(matched_files)}] Filtering: '{input_file}' -> '{out_file}'")
        (t_evts, stored_evts, sig_evts, fid_evts,
         gen_evts, gam_evts, ann_evts, sec_evts,
         r_evts, tgt_evts, mu_evts) = process_single_file(
             input_file, out_file, processor,
             store_all_events=args.store_all_events,
             use_fiducial=use_fiducial,
             num_threads=args.num_threads,
             max_events=args.max_events
         )

        grand_total_events += t_evts
        grand_total_stored += stored_evts
        grand_total_signal += sig_evts
        grand_total_fiducial += fid_evts

        grand_total_genuine += gen_evts
        grand_total_gamma += gam_evts
        grand_total_annihil += ann_evts
        grand_total_sec += sec_evts

        grand_total_rock += r_evts
        grand_total_target += tgt_evts
        grand_total_muon += mu_evts

        created_files.append(out_file)

    overall_elapsed = time.time() - overall_t0

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE SUMMARY")
    print("=" * 65)
    print(f"Total Input Files Processed : {len(matched_files)}")
    print(f"Total Output Files Created  : {len(created_files)}")
    print(f"Total Events Scanned        : {grand_total_events:,}")
    print(f"Total Events Stored         : {grand_total_stored:,}")
    print(f"Total Signal Events Found   : {grand_total_signal:,}")
    print(f"Total Fiducial 3-Muon Signal: {grand_total_fiducial:,} ({grand_total_fiducial/max(1,grand_total_signal)*100:.1f}% acceptance)")

    print(f"\n--- Physical Process Breakdown (Signal Events) ---")
    print(f"  * Genuine Tridents (MuonToMuonPair)  : {grand_total_genuine:,} events")
    print(f"  * Photon Conversions (GammaToMuPair) : {grand_total_gamma:,} events")
    print(f"  * Positron Annihilations             : {grand_total_annihil:,} events")
    print(f"  * Secondary Muon Induced             : {grand_total_sec:,} events")

    print(f"\n--- Spatial Region Breakdown (Signal Events) ---")

    print(f"  * Upstream Rock (< {target_z_min:.1f} cm)         : {grand_total_rock:,} events")
    print(f"  * Target Region [{target_z_min:.1f}, {target_z_max:.1f}] cm : {grand_total_target:,} events")
    print(f"  * Muon System (>= {target_z_max:.1f} cm)         : {grand_total_muon:,} events")

    print(f"\n============================================================")
    print(f"Total Processing Time        : {overall_elapsed:.1f} s")
    print("=" * 65)


if __name__ == "__main__":
    main()
