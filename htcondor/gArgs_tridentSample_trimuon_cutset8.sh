#!/bin/bash
# ==============================================================================
# Generate HTCondor arguments for tridentSample (Cutset 8) directly on trimuon_boost100
# ==============================================================================

IN_DIR="/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100"
OUT_DIR="/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100/cutset8"
CUT_SET=8
ARGS_FILE="args_trimuon_boost100_cutset8_tridentSample.txt"

mkdir -p "$OUT_DIR"
mkdir -p out err log

> "$ARGS_FILE"

for f in $(ls ${IN_DIR}/trimuon_digCPP-*.root 2>/dev/null | sort -V); do
    fname=$(basename "$f" .root)
    echo "$f ${OUT_DIR}/${fname}_cutset${CUT_SET}.root ${CUT_SET}" >> "$ARGS_FILE"
done

echo "========================================================================"
echo "SND@LHC TridentSample Arguments Generator (Cutset ${CUT_SET})"
echo "========================================================================"
echo "Input Directory : $IN_DIR"
echo "Output Directory: $OUT_DIR"
echo "Arguments File  : $ARGS_FILE"
echo "Total Jobs      : $(wc -l < "$ARGS_FILE")"
echo "========================================================================"
