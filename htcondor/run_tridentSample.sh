#!/bin/bash

# Arguments
INPUT_FILE="$1"
OUTPUT_FILE="$2"
CUT_SET="$3"

unset PYTHONPATH
unset LD_LIBRARY_PATH
unset ROOTSYS

# Load CERN sndsw environment stack
export snd_stack=/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7
source ${snd_stack}/setUp.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/sndswEnv.sh"

# Ensure library path includes libtrident_cuts.so
export LD_LIBRARY_PATH="${SCRIPT_DIR}:${SCRIPT_DIR}/lib:/afs/cern.ch/work/i/idioniso/sndMuTri/build/lib:${LD_LIBRARY_PATH}"

# Executable location
if [ -f "./tridentSample" ]; then
    EXEC="./tridentSample"
else
    EXEC="/afs/cern.ch/work/i/idioniso/sndMuTri/build/bin/tridentSample"
fi

echo "=========================================="
echo "Job started on: $(date)"
echo "Host: $(hostname)"
echo "Executable: $EXEC"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Cut set: $CUT_SET"
echo "=========================================="

"$EXEC" "$INPUT_FILE" "$OUTPUT_FILE" "$CUT_SET"
EXIT_CODE=$?

echo "=========================================="
echo "Job finished at $(date) with exit code $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
