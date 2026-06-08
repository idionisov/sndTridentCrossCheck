#!/bin/bash

# 1. Check if the user provided an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_args.txt>"
    exit 1
fi

ARGS_FILE="$1"
EMPTY_OR_HEADER_ONLY=0
TOTAL_CHECKED=0

echo "Scanning for empty/header-only CSV files defined in: $ARGS_FILE"
echo "---------------------------------------------------"

# 2. Read the file line by line
while read -r _ _ _ out_root out_csv _; do
    
    # Skip empty lines
    if [ -z "$out_csv" ]; then
        continue
    fi

    # 3. Only process the file if it actually exists
    if [ -f "$out_csv" ]; then
        
        # Count the number of lines in the file
        # The '<' feeds the file content to wc, so wc only outputs the number, not the filename
        lines=$(wc -l < "$out_csv")
        
        # 4. Check if the line count is 1 (header only) or 0 (completely empty)
        if [ "$lines" -le 1 ]; then
            echo "No data: $out_csv (Lines: $lines)"
            ((EMPTY_OR_HEADER_ONLY++))
        fi
        
        ((TOTAL_CHECKED++))
    fi

done < "$ARGS_FILE"

# 5. Print out the final summary
echo "---------------------------------------------------"
echo "Summary:"
echo "Existing CSV files checked:  $TOTAL_CHECKED"
echo "Files with no data:          $EMPTY_OR_HEADER_ONLY"
