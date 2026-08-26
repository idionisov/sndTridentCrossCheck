#!/bin/bash

RUN_DIR="/eos/experiment/sndlhc/convertedData/physics/2023/run_006640"
OUT_DIR="/eos/user/i/idioniso/sndMuTri/data/run_006640_cutset8"
CUT_SET=8
ARGS_FILE="args_run6640_tridentSample.txt"

mkdir -p "$OUT_DIR"

> "$ARGS_FILE"

for f in $(ls ${RUN_DIR}/sndsw_raw-*.root | sort -V); do
    fname=$(basename "$f" .root)
    echo "$f ${OUT_DIR}/${fname}_cutset${CUT_SET}.root ${CUT_SET}" >> "$ARGS_FILE"
done

echo "Generated $(wc -l < "$ARGS_FILE") argument lines in $ARGS_FILE"
