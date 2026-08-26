#!/usr/bin/env python3
"""
================================================================================
SND@LHC HTCondor Job Argument Generator
================================================================================
Scans an input directory (or glob pattern) for ROOT files, determines the event
count for each file, and splits them into subtask chunks such that no job exceeds
N events. Writes the resulting arguments line-by-line into an arguments file.

Usage Examples:
  1) Generate arguments with column format (input geo par start n_ev out):
     python3 generate_job_arguments.py \
         -i "/eos/user/i/idioniso/sndMuTri/data/run_006640" \
         -g "/eos/user/i/idioniso/sndMuTri/data/geofile_data_2023.root" \
         -p "TrackingParams_sf4.xml" \
         -N 10000 \
         -o args_run6640.txt

  2) Generate command-line flag format (-i ... -g ... -s ... -n ... -o ...):
     python3 generate_job_arguments.py \
         -i "/eos/user/i/idioniso/sndMuTri/data/run_006640/*.root" \
         -N 10000 \
         --format flags \
         -o args_flags.txt
================================================================================
"""

import os
import sys
import glob
import math
import argparse
from typing import List, Tuple, Optional

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


def get_tree_entries(file_path: str) -> Tuple[int, str]:
    """
    Opens a ROOT file and returns (total_entries, tree_name).
    Checks 'rawConv' and 'cbmsim' trees. Returns (0, '') if not found or corrupted.
    """
    f = ROOT.TFile.Open(file_path, "READ")
    if not f or f.IsZombie():
        return 0, ""

    tree_name = ""
    entries = 0
    for cand in ["rawConv", "cbmsim"]:
        t = f.Get(cand)
        if t:
            entries = int(t.GetEntries())
            tree_name = cand
            break

    f.Close()
    return entries, tree_name


def main():
    parser = argparse.ArgumentParser(
        description="Split ROOT files in an input directory into HTCondor job argument chunks (<= N events per job)."
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_paths",
        nargs="+",
        required=True,
        help="Input directory, file(s), or glob pattern(s) (e.g. '/path/to/data' or '/path/to/*.root')"
    )
    parser.add_argument(
        "-g", "--geo",
        dest="geo_file",
        default="/eos/user/i/idioniso/sndMuTri/data/geofile_data_2023.root",
        help="Geometry ROOT file path (default: %(default)s)"
    )
    parser.add_argument(
        "-p", "--par",
        dest="par_file",
        default="TrackingParams_sf4.xml",
        help="TrackingParams XML file path (default: %(default)s)"
    )
    parser.add_argument(
        "-N", "--max-events",
        dest="max_events_per_job",
        type=int,
        default=10000,
        help="Maximum number of events per job (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--outdir",
        dest="output_data_dir",
        default="",
        help="Optional destination directory prefix for output ROOT/Parquet files (default: none / filename only)"
    )
    parser.add_argument(
        "-o", "--out-file",
        dest="args_file",
        default="job_args.txt",
        help="Output text file to write job arguments into (default: %(default)s)"
    )
    parser.add_argument(
        "--format",
        dest="format",
        choices=["columns", "flags", "csv"],
        default="columns",
        help="Format of each line: 'columns' (space-separated), 'flags' (-i ... -s ...), or 'csv' (default: %(default)s)"
    )
    parser.add_argument(
        "--target",
        dest="target_type",
        choices=["hough", "features"],
        default="hough",
        help="Target pipeline: 'hough' (generates .root outputs) or 'features' (generates .parquet outputs) (default: %(default)s)"
    )
    parser.add_argument(
        "--pattern",
        dest="file_pattern",
        default="*.root",
        help="Glob pattern when input is a directory (default: %(default)s)"
    )

    args = parser.parse_args()

    # Find input ROOT files across all provided paths/patterns
    input_files = []
    for p in args.input_paths:
        if os.path.isdir(p):
            search_pattern = os.path.join(p, args.file_pattern)
            input_files.extend(glob.glob(search_pattern))
        elif "*" in p or "?" in p or "[" in p:
            input_files.extend(glob.glob(p))
        elif os.path.isfile(p):
            input_files.append(p)
        else:
            matched = glob.glob(p)
            if matched:
                input_files.extend(matched)
            else:
                print(f"Warning: Path '{p}' not found, skipping.")

    input_files = sorted(list(dict.fromkeys(input_files)))

    if not input_files:
        print(f"Error: No ROOT files found matching: {args.input_paths}")
        sys.exit(1)

    print("=" * 72)
    print("SND@LHC HTCondor Job Arguments Generator")
    print("=" * 72)
    print(f"Input Search Paths : {args.input_paths if len(args.input_paths) <= 3 else f'{len(args.input_paths)} paths/patterns'}")
    print(f"Files Found        : {len(input_files)}")
    print(f"Geometry File      : {args.geo_file}")
    print(f"Tracking Params    : {args.par_file}")
    print(f"Max Events / Job   : {args.max_events_per_job:,}")
    print(f"Output Format      : {args.format}")
    print(f"Target Output Type : {args.target_type}")
    print(f"Arguments Output   : {args.args_file}")
    print("=" * 72)

    job_lines = []
    total_events_all_files = 0
    out_ext = ".root" if args.target_type == "hough" else ".parquet"
    suffix_tag = "_hough" if args.target_type == "hough" else "_features"

    for idx, f_path in enumerate(input_files, 1):
        f_abs = os.path.abspath(f_path)
        base_name = os.path.splitext(os.path.basename(f_path))[0]

        entries, tree_name = get_tree_entries(f_abs)
        if entries == 0:
            print(f"[{idx}/{len(input_files)}] Skipping empty or invalid file: {os.path.basename(f_path)}")
            continue

        total_events_all_files += entries
        n_chunks = math.ceil(entries / args.max_events_per_job)

        print(f"[{idx}/{len(input_files)}] {os.path.basename(f_path)}: {entries:,} events -> {n_chunks} job(s)")

        for chunk_idx in range(n_chunks):
            start_ev = chunk_idx * args.max_events_per_job
            rem = entries - start_ev
            n_ev = min(args.max_events_per_job, rem)
            end_ev = start_ev + n_ev

            # Format output filename with chunk index (0, 1, 2, ...)
            out_filename = f"{base_name}{suffix_tag}_{chunk_idx}{out_ext}"
            if args.output_data_dir and args.output_data_dir not in [".", ""]:
                out_filepath = os.path.join(args.output_data_dir, out_filename)
            else:
                out_filepath = out_filename

            if args.format == "columns":
                # Space-separated columns:
                # <input_file> <geo_file> <par_file> <start_event> <n_events> <out_filepath>
                line = f"{f_abs} {args.geo_file} {args.par_file} {start_ev} {n_ev} {out_filepath}"
            elif args.format == "csv":
                line = f"{f_abs},{args.geo_file},{args.par_file},{start_ev},{n_ev},{out_filepath}"
            elif args.format == "flags":
                # Direct CLI flag string ready for executable arguments:
                line = f"-i {f_abs} -g {args.geo_file} -p {args.par_file} -s {start_ev} -n {n_ev} -o {out_filepath}"

            job_lines.append(line)

    # Write output file
    if args.args_file == "-":
        for line in job_lines:
            print(line)
    else:
        out_dir = os.path.dirname(os.path.abspath(args.args_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.args_file, "w") as f:
            for line in job_lines:
                f.write(line + "\n")

    print("=" * 72)
    print(f"GENERATION COMPLETE")
    print(f"Total ROOT Files Processed : {len(input_files)}")
    print(f"Total Events Found         : {total_events_all_files:,}")
    print(f"Total Job Lines Generated  : {len(job_lines):,}")
    print(f"Saved To File              : {args.args_file}")
    print("=" * 72)


if __name__ == "__main__":
    main()
