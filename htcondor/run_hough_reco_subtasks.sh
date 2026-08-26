#!/usr/bin/env bash
# ==============================================================================
# SND@LHC Multi-Track Hough Reconstruction - Subtask Runner
# ==============================================================================
# Performs multi-track Hough line finding (SciFi + DS) in subtasks or chunks.
#
# Usage examples:
#   1) Run only the first 10,000 events:
#      ./run_hough_reco_subtasks.sh -s 0 -n 10000
#
#   2) Run a specific subtask chunk (e.g. chunk 0 -> events 0 to 10,000):
#      ./run_hough_reco_subtasks.sh --task-id 0 --chunk-size 10000
#
#   3) Run all chunks in parallel and merge into one ROOT file via hadd:
#      ./run_hough_reco_subtasks.sh --task-id all --chunk-size 10000 -j 4 --merge output/merged_hough.root
#
#   4) Custom input file and geofile:
#      ./run_hough_reco_subtasks.sh \
#          -i "/eos/user/i/idioniso/sndMuTri/data/run_006640/sndsw_raw-0010_looseCuts7.root" \
#          -g "/eos/user/i/idioniso/sndMuTri/data/geofile_data_2023.root" \
#          -s 0 -n 10000
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Source sndswEnv.sh (or fallback to CVMFS setUp.sh) ---
set +eu  # Environment scripts may contain unset variables
LOADED_ENV=0
for env_cand in \
    "${SCRIPT_DIR}/sndswEnv.sh" \
    "${SCRIPT_DIR}/../sndswEnv.sh" \
    "$(pwd)/sndswEnv.sh" \
    "/afs/cern.ch/user/i/idioniso/sndswEnv.sh" \
    "/afs/cern.ch/work/i/idioniso/sndTridentCrossCheck/htcondor/sndswEnv.sh"; do
    if [[ -f "${env_cand}" ]]; then
        # shellcheck source=/dev/null
        source "${env_cand}" > /dev/null 2>&1 || true
        LOADED_ENV=1
        break
    fi
done

if [[ ${LOADED_ENV} -eq 0 && -f "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Jan30/setUp.sh" ]]; then
    # shellcheck source=/dev/null
    source "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Jan30/setUp.sh" > /dev/null 2>&1 || true
fi
set -euo pipefail

# --- Automatic Path Resolution ---
HOUGH_SCRIPT=""
for cand in \
    "${SCRIPT_DIR}/hough_line_reco.py" \
    "${SCRIPT_DIR}/../hough_line_reco.py" \
    "/afs/cern.ch/work/i/idioniso/sndMuTri/hough_line_reco.py" \
    "$(pwd)/hough_line_reco.py"; do
    if [[ -f "${cand}" ]]; then
        HOUGH_SCRIPT="$(cd "$(dirname "${cand}")" && pwd)/$(basename "${cand}")"
        break
    fi
done

DEFAULT_PAR=""
for cand in \
    "${SCRIPT_DIR}/TrackingParams_sf4.xml" \
    "${SCRIPT_DIR}/../TrackingParams_sf4.xml" \
    "/afs/cern.ch/work/i/idioniso/sndMuTri/TrackingParams_sf4.xml"; do
    if [[ -f "${cand}" ]]; then
        DEFAULT_PAR="$(cd "$(dirname "${cand}")" && pwd)/$(basename "${cand}")"
        break
    fi
done

# --- Default Parameters ---
DEFAULT_INPUT="/eos/user/i/idioniso/sndMuTri/data/run_006640/sndsw_raw-0010_looseCuts7.root"
DEFAULT_GEO="/eos/user/i/idioniso/sndMuTri/data/geofile_data_2023.root"
DEFAULT_OUTDIR="${SCRIPT_DIR}/hough_output"

INPUT_FILE="${DEFAULT_INPUT}"
GEO_FILE="${DEFAULT_GEO}"
PAR_FILE="${DEFAULT_PAR}"
OUT_DIR="${DEFAULT_OUTDIR}"
SYSTEM="both"
CASE_SF="muon_trident_Sf"
CASE_DS="muon_trident_DS"
CHUNK_SIZE=10000
TASK_ID=""             # empty = single run with START_EVENT / N_EVENTS, integer = specific chunk, "all" = all chunks
START_EVENT=0
N_EVENTS=0             # 0 = entire file or full chunk
PARALLEL_JOBS=1
MERGE_OUTPUT=""
SAVE_DISPLAYS=""

# --- Help function ---
print_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -i, --input PATH             Input ROOT file path or pattern (Default: ${DEFAULT_INPUT})
  -g, --geo PATH               Geometry ROOT file (Default: ${DEFAULT_GEO})
  -p, --par PATH               TrackingParams XML file (Default: ${DEFAULT_PAR})
  -H, --hough-script PATH      Path to hough_line_reco.py (Default: ${HOUGH_SCRIPT})
  --system SYS                 Detector system: 'both', 'scifi', 'ds' (Default: ${SYSTEM})
  --tracking-case-sf NAME      Tracking case name for SciFi (Default: ${CASE_SF})
  --tracking-case-ds NAME      Tracking case name for DS (Default: ${CASE_DS})
  -o, --outdir DIR             Output directory for ROOT files (Default: ${DEFAULT_OUTDIR})
  -c, --chunk-size N           Number of events per subtask / chunk (Default: 10000)
  -t, --task-id ID             Subtask index (0, 1, 2...) or "all" to process all chunks (Default: none)
  -s, --start N                Explicit start event index (Default: 0)
  -n, --n-events N             Number of events to process (Default: 0 for all)
  -j, --parallel N             Number of parallel worker processes when running 'all' chunks (Default: 1)
  -m, --merge PATH             Optional destination path to hadd all chunk ROOT files into one
  --save-displays              Save 2D event display canvases in output ROOT files
  -h, --help                   Display this help message and exit

Examples:
  # Process the first 10,000 events:
  ./$(basename "$0") -s 0 -n 10000

  # Process subtask chunk 2 (events 20,000 to 30,000):
  ./$(basename "$0") -t 2 -c 10000

  # Run all 10k event chunks in parallel and hadd merge:
  ./$(basename "$0") -t all -c 10000 -j 4 --merge "${OUT_DIR}/master_hough.root"
EOF
    exit 0
}

# --- Parse Arguments (supports both positional and flagged options) ---
EXPLICIT_OUTFILE=""

if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    # Positional mode: <input_file> [geo_file] [par_file] [start_event] [n_events] [output_file_or_dir]
    INPUT_FILE="$1"; shift
    [[ $# -gt 0 ]] && { GEO_FILE="$1"; shift; }
    [[ $# -gt 0 ]] && { PAR_FILE="$1"; shift; }
    [[ $# -gt 0 ]] && { START_EVENT="$1"; shift; }
    [[ $# -gt 0 ]] && { N_EVENTS="$1"; shift; }
    if [[ $# -gt 0 ]]; then
        if [[ "$1" == *.root || "$1" == *.parquet ]]; then
            EXPLICIT_OUTFILE="$1"
        else
            OUT_DIR="$1"
        fi
        shift
    fi
else
    # Flagged mode
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -i|--input) INPUT_FILE="$2"; shift 2 ;;
            -g|--geo) GEO_FILE="$2"; shift 2 ;;
            -p|--par) PAR_FILE="$2"; shift 2 ;;
            -H|--hough-script) HOUGH_SCRIPT="$2"; shift 2 ;;
            --system) SYSTEM="$2"; shift 2 ;;
            --tracking-case-sf) CASE_SF="$2"; shift 2 ;;
            --tracking-case-ds) CASE_DS="$2"; shift 2 ;;
            -o|--outdir|--output)
                if [[ "$2" == *.root || "$2" == *.parquet ]]; then
                    EXPLICIT_OUTFILE="$2"
                else
                    OUT_DIR="$2"
                fi
                shift 2 ;;
            -c|--chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
            -t|--task-id) TASK_ID="$2"; shift 2 ;;
            -s|--start) START_EVENT="$2"; shift 2 ;;
            -n|--n-events) N_EVENTS="$2"; shift 2 ;;
            -j|--parallel) PARALLEL_JOBS="$2"; shift 2 ;;
            -m|--merge) MERGE_OUTPUT="$2"; shift 2 ;;
            --save-displays) SAVE_DISPLAYS="--save-displays"; shift 1 ;;
            -h|--help) print_help ;;
            *) echo "[-] Unknown argument: $1"; print_help ;;
        esac
    done
fi

# --- Validation & Path Resolution ---
if [[ -z "${HOUGH_SCRIPT}" || ! -f "${HOUGH_SCRIPT}" ]]; then
    echo "[-] Error: Hough reco script not found at '${HOUGH_SCRIPT}'"
    exit 1
fi

if [[ ! -f "${PAR_FILE}" ]]; then
    for cand in "${SCRIPT_DIR}/${PAR_FILE}" "${SCRIPT_DIR}/../${PAR_FILE}" "/afs/cern.ch/work/i/idioniso/sndMuTri/${PAR_FILE}"; do
        if [[ -f "${cand}" ]]; then
            PAR_FILE="$(cd "$(dirname "${cand}")" && pwd)/$(basename "${cand}")"
            break
        fi
    done
fi

if [[ ! -f "${GEO_FILE}" ]]; then
    echo "[!] Warning: Geofile not found at '${GEO_FILE}'. Trying fallback or default."
fi

if [[ -n "${EXPLICIT_OUTFILE}" ]]; then
    OUT_DIR="$(dirname "${EXPLICIT_OUTFILE}")"
fi
mkdir -p "${OUT_DIR}"

# Helper function to get total entries from a ROOT file
get_total_entries() {
    local root_file="$1"
    python3 -c "
import ROOT, sys
f = ROOT.TFile.Open('${root_file}', 'READ')
if not f or f.IsZombie():
    sys.exit(1)
t = f.Get('rawConv') or f.Get('cbmsim')
if t:
    print(t.GetEntries())
else:
    print(0)
f.Close()
" 2>/dev/null || echo "0"
}

# Helper function to execute reconstruction on a single chunk
run_hough_chunk_job() {
    local in_f="$1"
    local start_ev="$2"
    local n_ev="$3"
    local out_root="$4"

    echo "  [Hough Subtask] Start: ${start_ev} | Events: ${n_ev} -> $(basename "${out_root}")"
    
    local extra_args=()
    if [[ -n "${SAVE_DISPLAYS}" ]]; then
        extra_args+=("${SAVE_DISPLAYS}")
    fi

    python3 "${HOUGH_SCRIPT}" \
        -i "${in_f}" \
        -o "${out_root}" \
        -g "${GEO_FILE}" \
        -p "${PAR_FILE}" \
        --system "${SYSTEM}" \
        --tracking-case-sf "${CASE_SF}" \
        --tracking-case-ds "${CASE_DS}" \
        -s "${start_ev}" \
        -n "${n_ev}" \
        "${extra_args[@]}"
}

BASE_NAME="$(basename "${INPUT_FILE}" .root)"

echo "========================================================================"
echo "SND@LHC Hough Reconstruction Subtask Launcher"
echo "========================================================================"
echo "Input File   : ${INPUT_FILE}"
echo "Geometry     : ${GEO_FILE}"
echo "Params       : ${PAR_FILE}"
echo "System       : ${SYSTEM} (SF: ${CASE_SF}, DS: ${CASE_DS})"
echo "Output Dir   : ${OUT_DIR}"
echo "Chunk Size   : ${CHUNK_SIZE}"
echo "========================================================================"

# --- Execution Branching ---

# Mode A: Specific Task ID chunk (e.g. task-id 0 -> 0..10000, task-id 1 -> 10000..20000)
if [[ -n "${TASK_ID}" && "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    START=$(( TASK_ID * CHUNK_SIZE ))
    N_EV="${CHUNK_SIZE}"
    OUT_ROOT="${OUT_DIR}/${BASE_NAME}_hough_chunk$(printf "%04d" "${TASK_ID}").root"
    
    echo "[+] Running Subtask #${TASK_ID} (Events ${START} to $(( START + N_EV )))..."
    run_hough_chunk_job "${INPUT_FILE}" "${START}" "${N_EV}" "${OUT_ROOT}"

# Mode B: Run all chunks
elif [[ "${TASK_ID}" == "all" ]]; then
    echo "[*] Querying total events in '${INPUT_FILE}'..."
    TOTAL_ENTRIES=$(get_total_entries "${INPUT_FILE}")
    
    if [[ "${TOTAL_ENTRIES}" -le 0 ]]; then
        echo "[-] Error: Could not determine event count for '${INPUT_FILE}'"
        exit 1
    fi

    TOTAL_CHUNKS=$(( (TOTAL_ENTRIES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    echo "[+] Total Events: ${TOTAL_ENTRIES} across ${TOTAL_CHUNKS} chunks (Size: ${CHUNK_SIZE})"

    CHUNK_FILES=()
    PIDS=()
    RUNNING=0

    for (( i=0; i<TOTAL_CHUNKS; i++ )); do
        START=$(( i * CHUNK_SIZE ))
        REMAINING=$(( TOTAL_ENTRIES - START ))
        N_EV=$(( REMAINING < CHUNK_SIZE ? REMAINING : CHUNK_SIZE ))
        OUT_ROOT="${OUT_DIR}/${BASE_NAME}_hough_chunk$(printf "%04d" "${i}").root"
        CHUNK_FILES+=("${OUT_ROOT}")

        run_hough_chunk_job "${INPUT_FILE}" "${START}" "${N_EV}" "${OUT_ROOT}" &
        PIDS+=($!)
        ((RUNNING++))

        # Limit concurrency
        if [[ ${RUNNING} -ge ${PARALLEL_JOBS} ]]; then
            wait -n 2>/dev/null || true
            ((RUNNING--))
        fi
    done

    # Wait for remaining background jobs
    wait

    echo "[+] All ${TOTAL_CHUNKS} subtasks completed successfully."

    # Merge via hadd if requested
    if [[ -n "${MERGE_OUTPUT}" ]]; then
        echo "[+] Merging chunk ROOT files with hadd into '${MERGE_OUTPUT}'..."
        hadd -f "${MERGE_OUTPUT}" "${CHUNK_FILES[@]}"
        echo "[+] Successfully created merged ROOT file: ${MERGE_OUTPUT}"
    fi

# Mode C: Direct Start Event / N Events (e.g. first 10,000 events)
else
    START="${START_EVENT}"
    N_EV="${N_EVENTS}"
    if [[ -n "${EXPLICIT_OUTFILE}" ]]; then
        OUT_ROOT="${EXPLICIT_OUTFILE}"
    elif [[ "${N_EV}" -eq 0 ]]; then
        OUT_ROOT="${OUT_DIR}/${BASE_NAME}_hough.root"
    else
        OUT_ROOT="${OUT_DIR}/${BASE_NAME}_hough_ev${START}_to_$(( START + N_EV )).root"
    fi

    echo "[+] Running Hough reconstruction for Events ${START} to $(( START + N_EV ))..."
    run_hough_chunk_job "${INPUT_FILE}" "${START}" "${N_EV}" "${OUT_ROOT}"
fi

echo "========================================================================"
echo "[+] HOUGH RECONSTRUCTION COMPLETE!"
echo "========================================================================"
