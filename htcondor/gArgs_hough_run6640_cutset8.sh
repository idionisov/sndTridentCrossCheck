#!/bin/bash
# ==============================================================================
# Generate HTCondor arguments for Hough line reconstruction on Run 6640 Cutset 8
# ==============================================================================

INPUT_DIR="/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640"
GEO_FILE="/eos/user/i/idioniso/sndMuTri/data/geofile_data_2023.root"
PAR_FILE="TrackingParams_sf4.xml"
OUT_DIR="/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/hough"
ARGS_FILE="args_run6640_cutset8_hough.txt"
CHUNK_SIZE=10000

mkdir -p "$OUT_DIR"

echo "Scanning input files in ${INPUT_DIR}..."
python3 generate_job_arguments.py \
    -i "${INPUT_DIR}/sndsw_raw-*.root" \
    -g "${GEO_FILE}" \
    -p "${PAR_FILE}" \
    -N ${CHUNK_SIZE} \
    -d "${OUT_DIR}" \
    -o "${ARGS_FILE}"

echo "Done! Generated $(wc -l < "${ARGS_FILE}" 2>/dev/null || echo 0) jobs in ${ARGS_FILE}"
