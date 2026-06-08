import os
import pandas as pd

df = pd.read_csv("/eos/user/i/idioniso/tridents/trident_files.csv")
base_dir = "/eos/experiment/sndlhc/convertedData/physics"
eos_tridents = "/eos/user/i/idioniso/tridents/gallery_tridents"

year = -1
for index, row in df.iterrows():
    run = int(row['run'])
    file_num = int(row['file_num'])

    if run < 5421:
        year = 2022
    elif run > 5421 and run <= 7357:
        year = 2023
    else:
        continue

    input_file = f"{base_dir}/{year}/run_{run:06d}/sndsw_raw-{file_num:04d}.root"
    output_file_root = f"{eos_tridents}/mu3_run{run:06d}_f{file_num:04d}.root"
    output_file_csv = f"{eos_tridents}/mu3_run{run:06d}_f{file_num:04d}.csv"

    cmd = f'python tridentTracking.py -s 0 -n 1000000 -f 1 -i {input_file} -or {output_file_root} -o {output_file_csv}'
    
    print(f"Executing: {cmd}") # Good for debugging
    os.system(cmd)
