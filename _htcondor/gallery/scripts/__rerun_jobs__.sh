#!/bin/bash

OUT_DIR="${1:-/eos/user/i/idioniso/tridents/gallery/sf3_new_2}"
RUN_DIR="${2:-/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/gallery/}"

RERUN_LIST=$(mktemp)

echo "Checking for empty files in $OUT_DIR..."
cd "$OUT_DIR" || { echo "Failed to enter $OUT_DIR"; exit 1; }

for f in *.csv; do 
    [ -e "$f" ] || continue 

    if [ $(wc -l < "$f") -le 1 ]; then
        base="${f%.csv}"
        echo "$base" >> "$RERUN_LIST"
        
        echo "Deleting empty: $f and ${base}.root"
        rm -f "$f" "${base}.root"
    fi
done

if [ ! -s "$RERUN_LIST" ]; then
    echo "No empty files found. Exiting."
    rm "$RERUN_LIST"
    exit 0
fi

echo "Found $(wc -l < "$RERUN_LIST") jobs to rerun."
cd "$RUN_DIR" || { echo "Failed to enter $RUN_DIR"; exit 1; }

grep -Fwf "$RERUN_LIST" args.txt | while read -r line; do
    echo "-------------------------------------------"
    echo "Rerunning: $line"
    ./find_tridents $line
done

rm "$RERUN_LIST"
echo "Done."
