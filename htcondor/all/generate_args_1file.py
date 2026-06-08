#!/usr/bin/env python3
import os
import argparse
import ROOT

def main():
    parser = argparse.ArgumentParser(description="Generate HTCondor arguments file for find_tridents.")
    parser.add_argument('-i', '--input-file', required=True, help="Input ROOT file path")
    parser.add_argument('-j', '--jobs', type=int, default=100, help="Number of jobs to split into")
    parser.add_argument('-od', '--out-dir', required=True, help="Directory to save the chunked ROOT and CSV output files")
    parser.add_argument('-b', '--basename', default="trident_output", help="Base name for the output files")
    parser.add_argument('-f', '--fraction', type=float, default=1.0, help="Fraction of events (e.g., 1.0)")
    parser.add_argument('-par', '--par-file', required=True, help="Path to the tracking parameter par.xml file")
    parser.add_argument('-out', '--args-file', default="job_args.txt", help="Output text file containing HTCondor arguments")
    args = parser.parse_args()

    # 1. Open the ROOT file and get the total number of events
    ROOT.gErrorIgnoreLevel = ROOT.kError # Suppress warnings
    f_in = ROOT.TFile.Open(args.input_file, "READ")
    if not f_in or f_in.IsZombie():
        print(f"Error: Could not open {args.input_file}")
        return

    tree = f_in.Get("rawConv")
    if not tree:
        print("Error: Tree 'rawConv' not found in the input file.")
        f_in.Close()
        return

    total_events = tree.GetEntries()
    f_in.Close()
    
    print(f"Total events found: {total_events}")

    # 2. Calculate chunk sizes
    if total_events == 0:
        print("No events to process.")
        return

    num_jobs = min(args.jobs, total_events) # Don't create more jobs than events
    base_chunk = total_events // num_jobs
    remainder = total_events % num_jobs

    # Ensure output directory exists (for the batch jobs to write to later)
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. Generate the arguments file
    current_start = 0
    with open(args.args_file, 'w') as f_out:
        for i in range(num_jobs):
            # Distribute the remainder across the first few jobs
            n_events = base_chunk + (1 if i < remainder else 0)
            
            if n_events == 0:
                continue

            # Construct unique output filenames for this job chunk
            out_root = os.path.join(args.out_dir, f"{args.basename}_{i}.root")
            out_csv = os.path.join(args.out_dir, f"{args.basename}_{i}.csv")

            # Format: INPUT_FILE START_EVENT N_EVENTS OUTPUT_ROOT OUTPUT_CSV FRACTION PAR_FILE
            args_line = f"{args.input_file} {current_start} {n_events} {out_root} {out_csv} {args.fraction} {args.par_file}\n"
            f_out.write(args_line)

            current_start += n_events

    print(f"Successfully generated {num_jobs} jobs in '{args.args_file}'.")

if __name__ == '__main__':
    main()
