#!/usr/bin/env bash
set -euo pipefail

SND_CROSS_CHECK_DIR="/afs/cern.ch/work/i/idioniso/sndTridentCrossCheck"
SND_MUTRI_DIR="/afs/cern.ch/work/i/idioniso/sndMuTri"
RAW_DIR="/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100_signal_fiducial"
CUTS_DIR="/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100_signal_fiducial_bdtCuts"

HOUGH_DIR="${CUTS_DIR}/hough"
MASTER_PARQUET="${CUTS_DIR}/master_trimuon_bdt_features.parquet"

TRIDENT_SAMPLE_BIN="${SND_CROSS_CHECK_DIR}/build/bin/tridentSample"
HOUGH_RECO_SCRIPT="${SND_MUTRI_DIR}/hough_line_reco.py"
EXTRACTOR_SCRIPT="${SND_MUTRI_DIR}/extract_trident_bdt_features.py"

PAR_FILE="${SND_MUTRI_DIR}/TrackingParams_sf4.xml"
GEO_FILE="/eos/user/i/idioniso/sndMuTri/data/geofile_trimuon_boost100.root"

# Cut set selection:
#   6: tridentSelection (veto, SciFi planes, 9/3 or 3/9 hits, density & fiducial)
#   7: bdtPreselection  (DS QDC >= 400, DS hits >= 8, last DS plane >= 2, SciFi hits >= 10)
CUT_SET=7

N_CORES=$(nproc)

# ==============================================================================
# 2. Setup & Directory Creation
# ==============================================================================
echo "========================================================================"
echo "SND@LHC Trident End-to-End Processing Pipeline"
echo "========================================================================"
echo "Raw Input Dir    : ${RAW_DIR}"
echo "Cuts Output Dir  : ${CUTS_DIR}"
echo "Hough Output Dir : ${HOUGH_DIR}"
echo "Master Parquet   : ${MASTER_PARQUET}"
echo "Cut Set          : ${CUT_SET}"
echo "Parallel Cores   : ${N_CORES}"
echo "========================================================================"

mkdir -p "${CUTS_DIR}"
mkdir -p "${HOUGH_DIR}"

if [[ ! -x "${TRIDENT_SAMPLE_BIN}" ]]; then
    echo "[-] Error: ${TRIDENT_SAMPLE_BIN} not found or not executable."
    exit 1
fi

# ==============================================================================
# Stage 1: Apply C++ tridentSample Cuts (Parallelized)
# ==============================================================================
echo ""
echo "[Stage 1/3] Filtering raw events with Cut Set ${CUT_SET}..."

export TRIDENT_SAMPLE_BIN CUTS_DIR CUT_SET

filter_single_file() {
    local infile="$1"
    local fname
    fname=$(basename "$infile")
    local outfile="${CUTS_DIR}/${fname}"
    
    echo "  [Cuts] -> ${fname}"
    "${TRIDENT_SAMPLE_BIN}" "${infile}" "${outfile}" "${CUT_SET}" > /dev/null 2>&1
}
export -f filter_single_file

# Filter only raw digCPP files (excluding any preexisting *_hough.root)
find "${RAW_DIR}" -maxdepth 1 -name "trimuon_digCPP-2*.root" ! -name "*_hough.root" | sort | \
    xargs -P "${N_CORES}" -n 1 -I {} bash -c 'filter_single_file "$@"' _ {}

echo "[+] Stage 1 Complete: Filtered ROOT files ready in ${CUTS_DIR}"

# ==============================================================================
# Stage 2: Hough Line Reconstruction (SciFi + DS Tracking in Parallel)
# ==============================================================================
echo ""
echo "[Stage 2/3] Running multi-track Hough Line Reconstruction (SciFi + DS)..."

export HOUGH_RECO_SCRIPT HOUGH_DIR PAR_FILE GEO_FILE

reconstruct_single_file() {
    local infile="$1"
    local base
    base=$(basename "$infile" .root)
    local outfile="${HOUGH_DIR}/${base}_hough.root"

    echo "  [Hough] -> ${base}_hough.root"
    python3 "${HOUGH_RECO_SCRIPT}" \
        -i "${infile}" \
        -o "${outfile}" \
        -p "${PAR_FILE}" \
        -g "${GEO_FILE}" \
        --system both \
        --tracking-case-sf muon_trident_Sf \
        --tracking-case-ds muon_trident_DS > /dev/null 2>&1
}
export -f reconstruct_single_file

find "${CUTS_DIR}" -maxdepth 1 -name "trimuon_digCPP-2*.root" | sort | \
    xargs -P "${N_CORES}" -n 1 -I {} bash -c 'reconstruct_single_file "$@"' _ {}

echo "[+] Stage 2 Complete: Hough-reconstructed files ready in ${HOUGH_DIR}"

# ==============================================================================
# Stage 3: Extract BDT Features & Merge into Unified Master Parquet
# ==============================================================================
echo ""
echo "[Stage 3/3] Extracting BDT Features into Master Parquet..."

python3 "${EXTRACTOR_SCRIPT}" \
    -i "${HOUGH_DIR}/trimuon_digCPP-2*_hough.root" \
    -o "${HOUGH_DIR}/features_%s.parquet" \
    -g "${GEO_FILE}" \
    -p "${PAR_FILE}" \
    -j "${N_CORES}" \
    --merge "${MASTER_PARQUET}"

echo ""
echo "========================================================================"
echo "[+] PIPELINE COMPLETE!"
echo "Master Dataset: ${MASTER_PARQUET}"
echo "========================================================================"
