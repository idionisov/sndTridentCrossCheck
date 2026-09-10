#!/usr/bin/env python3
"""
scripts/helpers/run_parallel_displays.py
----------------------------------------
Runs filter_and_display_tridents.py across multiple input ROOT files in parallel
using a multiprocessing pool on the local machine.
Stores the resulting canvases in separate ROOT files prefixed with 'displays_'.
"""

import os
import sys
import glob
import time
import shutil
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ROOT

def process_file_task(args_tuple):
    input_file, final_output_file, script_path, extra_args = args_tuple
    base_name = os.path.basename(input_file)
    tmp_out = f"/tmp/{os.path.basename(final_output_file)}"

    cmd = [
        sys.executable,
        script_path,
        "-i", input_file,
        "-o", tmp_out
    ] + extra_args

    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        # Copy from tmp to EOS final location
        shutil.copyfile(tmp_out, final_output_file)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        elapsed = time.time() - t0
        return True, base_name, os.path.basename(final_output_file), elapsed, res.stdout
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass
        elapsed = time.time() - t0
        return False, base_name, os.path.basename(final_output_file), elapsed, e.stdout
    except Exception as ex:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass
        elapsed = time.time() - t0
        return False, base_name, os.path.basename(final_output_file), elapsed, str(ex)

def main():
    parser = argparse.ArgumentParser(description="Run filter_and_display_tridents.py in parallel on local machine.")
    parser.add_argument(
        "-i", "--input-dir",
        default="/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100_signal",
        help="Input directory containing ROOT files (default: %(default)s)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="hough_trimuon_filtered_digCPP-*.root",
        help="Pattern to match input files (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100_signal",
        help="Output directory for generated display ROOT files (default: %(default)s)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=12,
        help="Number of parallel worker processes (default: %(default)s)"
    )
    parser.add_argument(
        "--prefix",
        default="displays_",
        help="Prefix for output filenames (default: %(default)s)"
    )
    args, unknown = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    display_script = os.path.join(script_dir, "scripts", "filter_and_display_tridents.py")
    if not os.path.exists(display_script):
        display_script = os.path.join(script_dir, "filter_and_display_tridents.py")
    if not os.path.exists(display_script):
        print(f"Error: Could not find '{display_script}'")
        sys.exit(1)

    raw_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    target_files = [f for f in raw_files if "_displays" not in f and not os.path.basename(f).startswith(args.prefix)]

    if not target_files:
        print(f"Error: No target files found matching pattern '{args.pattern}' in '{args.input_dir}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    for f_in in target_files:
        in_basename = os.path.basename(f_in)
        out_basename = f"{args.prefix}{in_basename}"
        f_out = os.path.join(args.output_dir, out_basename)
        tasks.append((f_in, f_out, display_script, unknown))

    print("=" * 70)
    print("Parallel Event Display Generator")
    print(f"Input Directory  : {args.input_dir}")
    print(f"Output Directory : {args.output_dir}")
    print(f"Files Found      : {len(tasks)}")
    print(f"Workers (Cores)  : {args.workers}")
    print(f"Output Prefix    : {args.prefix}")
    print("=" * 70)

    start_time = time.time()
    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_task, task): task for task in tasks}
        total_tasks = len(futures)

        for i, future in enumerate(as_completed(futures), 1):
            success, in_name, out_name, elapsed, log_out = future.result()
            if success:
                successful += 1
                print(f"[{i:3d}/{total_tasks:3d}] SUCCESS ({elapsed:5.1f}s): {in_name} -> {out_name}", flush=True)
            else:
                failed += 1
                print(f"[{i:3d}/{total_tasks:3d}] FAILED  ({elapsed:5.1f}s): {in_name} -> {out_name}", flush=True)
                print(f"--- Error log for {in_name} ---")
                print(log_out[-500:])
                print("-" * 50, flush=True)

    total_elapsed = time.time() - start_time
    print("=" * 70)
    print("PARALLEL PROCESSING COMPLETED")
    print(f"Total Files Processed : {total_tasks}")
    print(f"Successful            : {successful}")
    print(f"Failed                : {failed}")
    print(f"Total Execution Time  : {total_elapsed:.1f}s ({total_elapsed/60.0:.2f} mins)")
    print("=" * 70)

    # Verification step
    print("\nVerifying generated ROOT files...")
    corrupted = 0
    for task in tasks:
        out_path = task[1]
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            print(f"  [MISSING/EMPTY] {out_path}")
            corrupted += 1
            continue
        try:
            f = ROOT.TFile.Open(out_path, "READ")
            if not f or f.IsZombie():
                print(f"  [CORRUPTED] {out_path}")
                corrupted += 1
            else:
                f.Close()
        except Exception as e:
            print(f"  [ERROR OPENING] {out_path}: {e}")
            corrupted += 1

    if corrupted == 0:
        print(f"All {successful} output ROOT files verified successfully!")
    else:
        print(f"Warning: {corrupted} files failed verification.")

if __name__ == "__main__":
    main()
