#!/bin/bash

ARGS_FILE="$1"

while IFS= read -r line || [[ -n "$line" ]]; do
    echo "Processing arguments: $line"
    ./find_tridents $line
done < ${ARGS_FILE}
