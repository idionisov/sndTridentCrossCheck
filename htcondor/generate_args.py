import os, re
from typing import Optional
import ROOT
import argparse
import glob

sndsw_path = os.environ["SNDSW_ROOT"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-files', type=str, default="/eos/user/i/idioniso/tridents/initial_cuts/tridents_sndsw_raw-*.root", help="Input files.")
    parser.add_argument('-o', '--output-dir', type=str, default="/eos/user/i/idioniso/tridents", help="Output directory")
    parser.add_argument('-a', '--args-file', type=str, default="args.txt", help="Name of args file.")
    parser.add_argument('-n', '--n-events', type=int, default=1000000, help="Max events to be processed per job.")
    parser.add_argument('-j', '--jobs-per-file', type=int, default=1, help="Number of jobs per file.")
    parser.add_argument('-fr', '--fraction', type=float, default=1, help="Fraction of events to process.")
    parser.add_argument('-par', '--parFile', type=str, default=os.path.join(sndsw_path, "python/TrackingParams.xml"), help="Tracking parameter file")
    parser.add_argument('-gal', '--gallery', type=str, help='Only process events listed in the provided gallery json file')
    return parser.parse_args()

def main():
    args = get_args()

    files = sorted(glob.glob(args.input_files))

    with open(args.args_file, "w") as f_out:
        for file_path in files:
            filename = os.path.basename(file_path)
            
            match = re.search(r'(\d+)', filename)
            if match:
                file_num_str = match.group(1)
            else:
                file_num_str = "unknown"
            
            for chunk in range(args.jobs_per_file):
                start_event = chunk * args.n_events
                
                out_root = f"{args.output_dir}/recoTri_6640_f{file_num_str}.{chunk}.root"
                out_parquet = f"{args.output_dir}/recoTri_6640_f{file_num_str}"
                
                arg_line = f"{file_path} {start_event} {args.n_events} {out_root} {out_parquet} {args.fraction} {args.parFile} {args.gallery}"
                
                f_out.write(arg_line + "\n")
    print(f"Successfully generated {len(files) * args.jobs_per_file} job arguments in {args.args_file}")


if __name__=="__main__":
    main()
