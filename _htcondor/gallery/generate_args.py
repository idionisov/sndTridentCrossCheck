import os
import csv
import glob
import argparse
import re





multi_muon_search_dir = "/eos/experiment/sndlhc/users/odurhan/multi_muon_search"
SND_DATA = "/eos/experiment/sndlhc/convertedData/physics"
eos = "/eos/user/i/idioniso"
SND_HOME = "/afs/cern.ch/user/i/idioniso/snd_master"
gallery_csv_path = "/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/gallery/gallery_events.csv"

parser = argparse.ArgumentParser()
parser.add_argument('-par', '--parFile', type=str, default="/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/gallery/par_files/par.xml", help='Path to Hough parameter file.')
parser.add_argument('-o', '--args-file', type=str, default=f"args.txt")
parser.add_argument('--suffix', type=str, default="")
args = parser.parse_args()

output_args_file = args.args_file
par_file = args.parFile


required_jobs = set()

with open(gallery_csv_path, mode='r') as f_csv:
    reader = csv.DictReader(f_csv)
    for row in reader:
        run_val = int(row['run'])
        event_val = int(row['event_number'])
        file_idx = event_val // 1000000
        required_jobs.add((run_val, file_idx))

print(f"Loaded {len(required_jobs)} unique run-file combinations from CSV.")

job_count = 0
part = "sf3"
with open(output_args_file, "w") as f_out:
    if not os.path.exists(multi_muon_search_dir):
        print(f"Error: Base directory {multi_muon_search_dir} not found.")
        exit(1)

    for y in os.listdir(multi_muon_search_dir):
        year_path = os.path.join(multi_muon_search_dir, y)
        if not os.path.isdir(year_path):
            continue

        for r in os.listdir(year_path):
            r = int(r)
            data_run_dir = os.path.join(SND_DATA, y, f"run_{r:06d}")
            
            if not os.path.isdir(data_run_dir):
                continue


            search_pattern = os.path.join(data_run_dir, "*sndsw_raw*.root")
            
            for filepath in glob.glob(search_pattern):
                filename = os.path.basename(filepath)

                file_num = int(re.search(r'-(\d+)\.', filename).group(1))
                if (r, file_num) not in required_jobs:
                    continue

                input_path = f"{SND_DATA}/{y}/run_{r:06d}/{filename}"
                if not args.suffix:
                    output_csv = f"{eos}/tridents/gallery/{part}/tridents_in_gallery_run{r}_f{file_num}.csv"
                    output_root = f"{eos}/tridents/gallery/{part}/tridents_in_gallery_run{r}_f{file_num}.root"
                else:
                    output_csv = f"{eos}/tridents/gallery/{part}/tridents_in_gallery_run{r}_f{file_num}_{args.suffix}.csv"
                    output_root = f"{eos}/tridents/gallery/{part}/tridents_in_gallery_run{r}_f{file_num}_{args.suffix}.root"

                # INPUT OUT_CSV OUT_ROOT START_EVENT N_EVENTS FRACTION PAR_FILE
                args_out = f"{input_path} {output_csv} {output_root} 0 1000000 1 {par_file}"
                
                f_out.write(args_out + "\n")
                job_count += 1

print(f"Successfully generated {job_count} job arguments in '{output_args_file}'.")
