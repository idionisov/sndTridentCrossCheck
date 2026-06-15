import os, re, json
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
    default_input = "/eos/user/i/idioniso/tridents/initial_cuts/tridents_sndsw_raw-*.root"
    
    with open(args.args_file, "w") as f_out:
        count = 0
        
        if args.gallery and os.path.exists(args.gallery):
            print(f"Generating jobs from gallery: {args.gallery}")
            with open(args.gallery, 'r') as f:
                gallery_data = json.load(f)
                
            job_map = {}
            for run, events in gallery_data.items():
                run_int = int(run)
                job_map[run_int] = {}
                for e in events:
                    f_num = e // args.n_events
                    if f_num not in job_map[run_int]:
                        job_map[run_int][f_num] = []
                    job_map[run_int][f_num].append(e)

            for run_num, files_dict in job_map.items():
                if 4361 <= run_num <= 5421: year=2022
                elif 5483 <= run_num <= 7357: year=2023
                elif 7649 <= run_num <= 8317: year=2024
                else: raise ValueError(f"Invalid run_number! {run_num}")
        
                for file_num in files_dict.keys():
                    file_path = f"/eos/experiment/sndlhc/convertedData/physics/{year}/run_{run_num:06d}/sndsw_raw-{file_num:04d}.root"
                    if os.path.exists(file_path):
                        out_root = f"{args.output_dir}/recoTri_{run_num}_f{file_num:04d}.root"
                        out_parquet = f"{args.output_dir}/recoTri_{run_num}_f{file_num:04d}"
                        f_out.write(f"{file_path} 0 {args.n_events} {out_root} {out_parquet} {args.fraction} {args.parFile} {args.gallery}\n")
                        count += 1

        else:
            files = sorted(glob.glob(args.input_files))
            for file_path in files:
                filename = os.path.basename(file_path)
                file_match = re.search(r'raw-(\d+)', filename)
                
                run_num = 6640
                file_num = int(file_match.group(1))
                
                out_root = f"{args.output_dir}/recoTri_{run_num}_f{file_num:04d}.root"
                out_parquet = f"{args.output_dir}/recoTri_{run_num}_f{file_num:04d}"
                f_out.write(f"{file_path} 0 {args.n_events} {out_root} {out_parquet} {args.fraction} {args.parFile} {args.gallery}\n")
                count += 1
            
        print(f"Successfully generated {count} job arguments in {args.args_file}")



if __name__=="__main__":
    main()
