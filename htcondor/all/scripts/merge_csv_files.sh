#!/bin/bash

# Default variables
INPUT_DIR="${1:-/eos/user/i/idioniso/tridents/all/sf3}"
INPUT_CSV_FILES="${2:-recoTri_*.csv}"
OUTPUT_CSV_FILE="${3:-recoTri.csv}"

OUTPUT_PATH="${INPUT_DIR}/${OUTPUT_CSV_FILE}"

if [[ "$OUTPUT_CSV_FILE" == $INPUT_CSV_FILES ]]; then
    echo "Error: Output filename matches the input pattern. This will cause an infinite loop."
    exit 1
fi

if ! ls "${INPUT_DIR}"/${INPUT_CSV_FILES} >/dev/null 2>&1; then
    echo "Error: No files found matching ${INPUT_DIR}/${INPUT_CSV_FILES}"
    exit 1
fi

echo "Merging files into $OUTPUT_PATH..."

awk 'FNR==1 && NR!=1{next;}{print}' ${INPUT_DIR}/${INPUT_CSV_FILES} > "$OUTPUT_PATH"

echo "Done! Merged file contains $(wc -l < "$OUTPUT_PATH") lines."
