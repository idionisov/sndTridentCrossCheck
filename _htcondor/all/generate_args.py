import os
import glob

input_dir = "/eos/user/i/idioniso/tridents/initial_cuts"
output_dir = "/eos/user/i/idioniso/tridents/all/sf4"
output_args_file = "args_sf4.txt"
events_per_job = 1000000
jobs_per_file = 1
fraction = 1
parFile = "/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/all/par_files/par_sf4.xml"

files = sorted(glob.glob(os.path.join(input_dir, "tridents_sndsw_raw-*.root")))

with open(output_args_file, "w") as f_out:
    for file_path in files:
        filename = os.path.basename(file_path)
        
        file_num_str = filename.replace("tridents_sndsw_raw-", "").replace(".root", "")
        
        for chunk in range(jobs_per_file):
            start_event = chunk * events_per_job
            
            out_root = f"{output_dir}/recoTri_6640_f{file_num_str}.{chunk}.root"
            out_csv = f"{output_dir}/recoTri_6640_f{file_num_str}.{chunk}.csv"
            print(out_csv)
            
            arg_line = f"{file_path} {start_event} {events_per_job} {out_root} {out_csv} {fraction} {parFile}"
            
            f_out.write(arg_line + "\n")

print(f"Successfully generated {len(files) * jobs_per_file} job arguments in {output_args_file}")
