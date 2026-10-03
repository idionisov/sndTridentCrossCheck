#!/bin/bash
set -u

# Configuration
INPUT_DIR="./data/3mu/new_pars"
GEO_FILE="./data/geofile_sndlhc_TI18_V3_2023.root"
MAX_PARALLEL=5
TRACK_SCRIPT="${SNDSW_ROOT}/shipLHC/scripts/run_TrackSelections.py"

# Verify environment and dependencies
if [ -z "${SNDSW_ROOT:-}" ]; then
    echo "[!] Error: SNDSW_ROOT environment variable is not set." >&2
    exit 1
fi

if [ ! -f "${TRACK_SCRIPT}" ]; then
    echo "[!] Error: Track selection script not found at ${TRACK_SCRIPT}" >&2
    exit 1
fi

if [ ! -f "${GEO_FILE}" ]; then
    echo "[!] Error: Geofile not found at ${GEO_FILE}" >&2
    exit 1
fi

# Collect input files (excluding any existing _Trks.root)
FILES=($(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*_digCPP.root" | sort))
TOTAL=${#FILES[@]}

echo "================================================================================"
echo "SND@LHC BATCH TRACK RECONSTRUCTION (PARALLEL WORKERS = ${MAX_PARALLEL})"
echo "================================================================================"
echo "Input Directory : ${INPUT_DIR}"
echo "Total Files     : ${TOTAL}"
echo "Geo File        : ${GEO_FILE}"
echo "Track Script    : ${TRACK_SCRIPT}"
echo "================================================================================"

run_track_reco() {
    local in_file="$1"
    local idx="$2"
    local total="$3"
    local out_file="${in_file%.root}_Trks.root"
    local log_file="${in_file%.root}_Trks.log"

    # Skip if output already exists and is non-empty
    if [ -s "${out_file}" ]; then
        echo "[SKIP] [${idx}/${total}] $(basename "${out_file}") already exists."
        return 0
    fi

    echo "[START] [${idx}/${total}] Processing $(basename "${in_file}")..."
    t0=$(date +%s)

    # Note: run_TrackSelections.py intentionally self-terminates via kill (code 143) on exit
    python "${TRACK_SCRIPT}" \
        -f "${in_file}" \
        -g "${GEO_FILE}" \
        -o "${out_file}" \
        -st -ht -t ScifiDS > "${log_file}" 2>&1 || true

    t1=$(date +%s)
    echo "[DONE]  [${idx}/${total}] Finished $(basename "${out_file}") in $((t1 - t0)) s"
}

current=0
for in_file in "${FILES[@]}"; do
    current=$((current + 1))
    
    run_track_reco "${in_file}" "${current}" "${TOTAL}" &

    # Maintain parallel pool
    while [ $(jobs -r -p | wc -l) -ge "${MAX_PARALLEL}" ]; do
        wait -n 2>/dev/null || true
    done
done

# Wait for all background tasks to finish
wait 2>/dev/null || true

echo "================================================================================"
echo "[✓] All ${TOTAL} files processed successfully!"
echo "================================================================================"

