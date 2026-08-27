#!/bin/bash
# ==============================================================================
# Generate HTCondor arguments for 1_houghLines.sub on trimuon_boost100 cutset8
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN_DIR="/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100_cutset8"
GEO_FILE="/eos/user/i/idioniso/sndMuTri/data/geofile_trimuon_boost100.root"
PAR_FILE="TrackingParams_sf4.xml"
ARGS_FILE="args_trimuon_boost100_cutset8_hough.txt"
N_EVENTS=10000

echo "Scanning input files in ${IN_DIR}..."
python3 "${SCRIPT_DIR}/generate_job_arguments.py" \
    -i "${IN_DIR}/trimuon_digCPP-*_cutset8.root" \
    -g "${GEO_FILE}" \
    -p "${PAR_FILE}" \
    -N "${N_EVENTS}" \
    -t hough \
    -o "${ARGS_FILE}"
