#!/bin/bash

source sndswEnv.sh

mkdir -p /eos/user/i/idioniso/tridents/ThreeMuons_MC_reco

python generate_args.py \
  --mc \
  -a args_mc_only.txt \
  -n 1000000 \
  -j 1 \
  -fr 1 \
  -par /afs/cern.ch/work/i/idioniso/sndTridentCrossCheck/parFiles/TrackingParams.xml \
  -o /eos/user/i/idioniso/tridents/ThreeMuons_MC_reco
