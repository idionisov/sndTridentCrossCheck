#!/bin/bash
# ==============================================================================
# Generate HTCondor arguments for BDT Feature Extraction (Step 2)
# ==============================================================================
# Maps each Hough ROOT file in <input_dir> to an identical-length Parquet file
# located in <input_dir>/parquet/<filename>.parquet

INPUT_DIR="${1:-/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100/_hough}"
GEO_FILE="${2:-/eos/user/i/idioniso/sndMuTri/data/geofile_trimuon_boost100.root}"
PAR_FILE="TrackingParams_sf4.xml"
OUT_PARQUET_DIR="${INPUT_DIR}/parquet"
ARGS_FILE="${3:-args_trimuon_boost100_features.txt}"

mkdir -p "${OUT_PARQUET_DIR}"

python3 -c "
import glob, os, sys

in_dir = '${INPUT_DIR}'
geo = '${GEO_FILE}'
par = '${PAR_FILE}'
out_dir = '${OUT_PARQUET_DIR}'
args_out = '${ARGS_FILE}'

root_files = sorted(glob.glob(os.path.join(in_dir, '*.root')))
if not root_files:
    # If jobs are not finished yet, try matching from Hough args file if available
    hough_args = 'args_trimuon_boost100_hough.txt'
    if os.path.exists(hough_args):
        with open(hough_args) as f:
            for l in f:
                parts = l.strip().split()
                if len(parts) >= 6:
                    root_files.append(parts[5])

lines = []
for f in root_files:
    bname = os.path.splitext(os.path.basename(f))[0]
    out_pq = os.path.join(out_dir, bname + '.parquet')
    lines.append(f'{f} {geo} {par} 0 0 {out_pq}')

with open(args_out, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f'Generated {len(lines)} feature extraction jobs in {args_out}')
print(f'Output Parquet files directory: {out_dir}')
"
