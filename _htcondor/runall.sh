#!/bin/bash

for y in $(ls /eos/experiment/sndlhc/users/odurhan/multi_muon_search); do
    for r in $(ls /eos/experiment/sndlhc/users/odurhan/multi_muon_search/${y}); do
        for f in $(ls /eos/experiment/sndlhc/convertedData/physics/${y}/run_00${r} | grep sndsw_raw | grep .root); do
#            python gallery_tracking.py -i $SND_DATA/${y}/run_00${r}/${f} -o tridents_found_in_gallery.csv -s 0 -n 1000000 -par ${SND_HOME}/sndsw/python/TrackingParams3.xml
            python gallery_tracking.py -i $SND_DATA/${y}/run_00${r}/${f} -o tridents_found_in_gallery.csv -s 0 -n 1000000 -par ${SND_HOME}/sndsw/trackingParams.xml
        done;
    done;
done
