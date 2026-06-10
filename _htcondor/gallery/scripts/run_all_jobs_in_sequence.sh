#!/bin/bash

ARGS_FILE="${1:-/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/gallery/args.txt}"
RUN_DIR="${2:-/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/gallery}"

if [[ ! -f "$ARGS_FILE" ]]; then
    echo "Error: Cannot find $ARGS_FILE"
    exit 1
fi

echo "Starting sequential processing from $ARGS_FILE..."
echo "---------------------------------------------------"

# Read the file line by line
while read -r IN_FILE OUT_CSV OUT_ROOT START_EV N_EV FRAC PAR_FILE; do
    # Skip empty lines to avoid errors
    if [[ -z "$IN_FILE" ]]; then
        continue
    fi

    echo "Running ./find_tridents for input:"
    echo "  -> $IN_FILE"

    ${RUN_DIR}/find_tridents "$IN_FILE" "$OUT_CSV" "$OUT_ROOT" "$START_EV" "$N_EV" "$FRAC" "$PAR_FILE"
    
    if [[ $? -ne 0 ]]; then
        echo "Warning: ./find_tridents encountered an error processing $IN_FILE."
    fi

    echo "---------------------------------------------------"

done < "$ARGS_FILE"

echo "All jobs completed!"
