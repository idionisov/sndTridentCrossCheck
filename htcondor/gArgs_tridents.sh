#!/bin/bash

source sndswEnv.sh

#python generate_args.py -i /eos/user/i/idioniso/tridents/initial_cuts/\*.root -a args_tridents.txt -n 1000000 -j 1 -fr 1 -par /afs/cern.ch/work/i/idioniso/sndTridentCrossCheck/parFiles/TrackingParams.xml -o /eos/user/i/idioniso/tridents/run_006640-15Jun26

python generate_args.py -i /eos/experiment/sndlhc/convertedData/physics/2023/run_006640/\*.root -a args_tridents.txt -n 100000 -j 10 -fr 1 -par /afs/cern.ch/work/i/idioniso/sndTridentCrossCheck/parFiles/TrackingParams.xml -o /eos/user/i/idioniso/tridents/run_006640-15Jun26
