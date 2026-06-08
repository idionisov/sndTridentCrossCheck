#!/bin/bash

for y in $(ls /eos/experiment/sndlhc/users/odurhan/multi_muon_search); do
    for r in $(ls /eos/experiment/sndlhc/users/odurhan/multi_muon_search/${y}); do
        for f in $(ls /eos/experiment/sndlhc/convertedData/physics/${y}/run_00${r} | grep sndsw_raw | grep .root); do
            python ${SHIPLHC}/scripts/tridentTracking.py -i $SND_DATA/${y}/run_00${r}/${f} -o ${eos}/tridents/tridents_found_in_gallery.csv -s 0 -n 1000000 -par ${SND_HOME}/sndsw/trackingParams.xml
        done;
    done;
done
