#!/bin/bash

y=2023
r=6663

for f in $(ls /eos/experiment/sndlhc/convertedData/physics/${y}/run_00${r} | grep sndsw_raw | grep .root); do
    python ./trident_tracking.py -i $SND_DATA/${y}/run_00${r}/${f} -o tridents_run6663.csv -s 0 -n 1000000 -f 0.01
done;
