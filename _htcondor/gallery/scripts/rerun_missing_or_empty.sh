#!/bin/bash

ARGS_FILE="$1"
RUN_DIR="${2:-/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/gallery/}"

if [ -z "$ARGS_FILE" ] || [ ! -f "$ARGS_FILE" ]; then
    echo "Usage: $0 <path_to_args.txt> [run_directory]"
    exit 1
fi

ARGS_FILE=$(realpath "$ARGS_FILE")

echo "Moving to execution directory: $RUN_DIR"
cd "$RUN_DIR" || { echo "Failed to enter $RUN_DIR"; exit 1; }

JOBS_RERUN=0

echo "Scanning $ARGS_FILE for missing or empty outputs..."
echo "==================================================="

while read -r line; do
    [ -z "$line" ] && continue

    read -r in_root out_csv out_root rest <<< "$line"

    NEEDS_RERUN=false
    REASON=""

    if [ ! -f "$out_root" ]; then
        NEEDS_RERUN=true
        REASON="Missing ROOT file"
    elif [ ! -f "$out_csv" ]; then
        NEEDS_RERUN=true
        REASON="Missing CSV file"
    else
        # Both files exist, now check the CSV content length
        lines=$(wc -l < "$out_csv")
        if [ "$lines" -le 1 ]; then
            NEEDS_RERUN=true
            REASON="CSV has no data (lines: $lines)"
        fi
    fi

    if [ "$NEEDS_RERUN" = true ]; then
        echo "-------------------------------------------"
        echo "Flagged: $REASON"
        echo "Cleaning up existing files for this job..."
        
        rm -f "$out_csv" "$out_root"

        echo "Rerunning: ${RUN_DIR}/find_tridents $line"
        
        ${RUN_DIR}/find_tridents $line
        
        ((JOBS_RERUN++))
    fi

done < "$ARGS_FILE"

echo "==================================================="
echo "Done. Total jobs rerun: $JOBS_RERUN"
