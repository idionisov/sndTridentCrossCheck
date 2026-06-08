#!/bin/bash

# 1. Check if the user provided an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_args.txt>"
    exit 1
fi

ARGS_FILE="$1"
MISSING_ROOT_COUNT=0
MISSING_CSV_COUNT=0

echo "Scanning for missing outputs defined in: $ARGS_FILE"
echo "---------------------------------------------------"

# 2. Read the file line by line
# Using '_' as a dummy variable to skip columns we don't need
while read -r _ _ _ out_root out_csv _; do
    
    # Skip empty lines
    if [ -z "$out_root" ]; then
        continue
    fi

    # 3. Check if the expected .root file is missing
    if [ ! -f "$out_root" ]; then
        echo "Missing: $out_root"
        ((MISSING_ROOT_COUNT++))
    fi

    # 4. Check if the expected .csv file is missing
    if [ ! -f "$out_csv" ]; then
        echo "Missing: $out_csv"
        ((MISSING_CSV_COUNT++))
    fi

done < "$ARGS_FILE"

# 5. Print out the final summary
echo "---------------------------------------------------"
echo "Summary:"
echo "Missing .root files: $MISSING_ROOT_COUNT"
echo "Missing .csv files:  $MISSING_CSV_COUNT"
