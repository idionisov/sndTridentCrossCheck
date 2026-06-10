import os
import random
import ROOT
import argparse
import csv
import json
import pyarrow as pa
import pyarrow.parquet as pq
import datetime
import time
from array import array
import numpy as np
import pandas as pd
import rootUtils as ut
import pythonHelpers.general
import pythonHelpers.hough
import pythonHelpers.geometry
from typing import Optional

ROOT.gROOT.SetBatch(True)
for library in ["libBase", "libShipData", "libshipLHC", "libsnd_analysis_tools"]:
    ROOT.gSystem.Load(library)

import SndlhcGeo
import SndlhcMuonReco
import SndlhcTracking

sndsw_path = os.environ['SNDSW_ROOT']
ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndSciFiTools.h"')
ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndGeometryGetter.h"')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file',  type=str, required=True, help='Path to input file.')
    parser.add_argument('-parquet', '--output-parquet', type=str, required=True, help='Output Parquet filename')
    parser.add_argument('-o', '--output-root', type=str, help='Output ROOT filename for selected events')
    parser.add_argument('-s', '--start-event', type=int, help='Start event number.', default=0)
    parser.add_argument('-n', '--n-events', type=int, help='Number of events.', default=1000000)
    parser.add_argument('-f', '--fraction', type=float, default=1.0, help='Fraction of events to process')
    parser.add_argument('-par', '--parFile', type=str, default="TrackingParams.xml", help='Tracking parameter file')
    parser.add_argument('-zmin', '--z-vtx-min', type=float, default=None, help='Minimum z-vertex intersection')
    parser.add_argument('-zmax', '--z-vtx-max', type=float, default=None, help='Maximum z-vertex intersection')
    parser.add_argument('-gal', '--gallery', type=str, help='Only process events listed in the provided gallery json file')
    return parser.parse_args()





def main():
    event_buffer = []
    batch_size = 1000

    histograms = {}
    projections = {1: 'xz', 2: 'yz'}

    args = get_arguments()
    start_time_process = time.time()

    gallery = {}
    if args.gallery is not None:
        try:
            with open(args.gallery, "r") as f:
                gallery = json.load(f)
        except FileNotFoundError:
            print(f"Error: {args.gallery} not found!")
            return

    input_file = ROOT.TFile.Open(args.input_file)
    input_tree = input_file.Get("rawConv")
    total_entries = input_tree.GetEntries()

    input_tree.GetEntry(0)
    run_number = input_tree.EventHeader.GetRunId()
    
    fair_run = ROOT.FairRunAna()
    io_manager = ROOT.FairRootManager.Instance()
    io_manager.SetTreeName("rawConv") 
    
    fair_run.SetSource(ROOT.FairFileSource(input_file))
    fair_run.SetSink(ROOT.FairRootFileSink(ROOT.TMemFile('dummy','CREATE')))

    rtdb = fair_run.GetRuntimeDb()
    if os.path.exists(args.parFile):
        par_source = ROOT.FairParAsciiFileIo()
        
        if not par_source.open(args.parFile, 'in'):
            print(f"Error: Failed to open parameter file {args.parFile}")
            return
            
        rtdb.setFirstInput(par_source)
        
        print(f"Setting RunId: {run_number} (type: {type(run_number)})")
        rtdb.addRun(int(run_number))
        
        rtdb.setOutput(par_source)
    else:
        print(f"Warning: Parameter file {args.parFile} not found!")

    
    geo = SndlhcGeo.GeoInterface(ROOT.snd.analysis_tools.GetGeoPath(run_number))
    pythonHelpers.geometry.initialize_event_display(geo)

    muon_reco_task = SndlhcMuonReco.MuonReco()
    muon_reco_task.SetParFile(args.parFile)
    fair_run.AddTask(muon_reco_task)
    muon_reco_task.SetHoughSpaceFormat("linearSlopeIntercept")
    muon_reco_task.SetTrackingCase('muon_trident_Sf')
    
    fair_run.Init()
    
    input_tree = io_manager.GetInTree()
    if input_tree.GetBranch('Digi_MuFilterHit'):
        input_tree.Digi_MuFilterHits = input_tree.Digi_MuFilterHit

    output_root_file, output_tree = None, None
    if args.output_root:
        output_root_file = ROOT.TFile(args.output_root if args.output_root.endswith(".root") else f"{args.output_root}.root", "RECREATE")
        output_tree = input_tree.CloneTree(0)

    buffer = []
    BUFFER_SIZE = 5000
    batch_counter = 0
    columns = [
        'run', 'event_number', 'n_lines',
        'xz_m1', 'xz_c1', 'xz_m2', 'xz_c2', 'xz_m3', 'xz_c3',
        'yz_m1', 'yz_c1', 'yz_m2', 'yz_c2', 'yz_m3', 'yz_c3',
        'sum_hit_weight_density', 'sum_qdc'
    ]

    processed_count = 0
    progress_step = 0
    num_events_to_process = min(args.n_events, total_entries - args.start_event)

    for entry_index in range(args.start_event, total_entries):
        if processed_count >= args.n_events:
            break
    
        input_tree.GetEntry(entry_index)
        event_number = input_tree.EventHeader.GetEventNumber() 

        if args.gallery:
            run_str = str(run_number)
            if run_str not in gallery or event_number not in gallery[run_str]:
                continue
        else:
            if random.random() >= args.fraction:
                continue

        if num_events_to_process > 0 and ((entry_index - args.start_event) * 100 / num_events_to_process) >= progress_step:
            elapsed = int(time.time() - start_time_process)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            print(f"[{progress_step}%] \t {entry_index-args.start_event:,}/{num_events_to_process:,} \t {h:02d}:{m:02d}:{s:02d}")
            progress_step += 1

        if input_tree.EventHeader.ClassName() == 'SNDLHCEventHeader':
            geo.modules['Scifi'].InitEvent(input_tree.EventHeader)

        n_lines, track_lines = run_hough_transform(
            muon_reco_task, input_tree, geo,
            z_vtx_min=args.z_vtx_min, z_vtx_max=args.z_vtx_max
        )

        if n_lines >= 1:
            row = {
                'run': run_number, 
                'event_number': event_number, 
                'n_lines': n_lines,
                'xz_m1': get_line_params('XZ', 0, track_lines)[0], 
                'xz_c1': get_line_params('XZ', 0, track_lines)[1],
                'xz_m2': get_line_params('XZ', 1, track_lines)[0], 
                'xz_c2': get_line_params('XZ', 1, track_lines)[1],
                'xz_m3': get_line_params('XZ', 2, track_lines)[0], 
                'xz_c3': get_line_params('XZ', 2, track_lines)[1],
                'yz_m1': get_line_params('YZ', 0, track_lines)[0], 
                'yz_c1': get_line_params('YZ', 0, track_lines)[1],
                'yz_m2': get_line_params('YZ', 1, track_lines)[0], 
                'yz_c2': get_line_params('YZ', 1, track_lines)[1],
                'yz_m3': get_line_params('YZ', 2, track_lines)[0], 
                'yz_c3': get_line_params('YZ', 2, track_lines)[1],
                'sum_hit_weight_density': get_scifi_hit_density(input_tree.Digi_ScifiHits),
                'sum_qdc': get_scifi_total_qdc(input_tree.Digi_ScifiHits)
            }
            buffer.append(row)

            if len(buffer) >= BUFFER_SIZE:
                flush_to_parquet(buffer, batch_counter, columns, args.output_parquet)
                buffer = []
                batch_counter += 1


        if n_lines >= 2 and output_tree:
            output_tree.Fill()
            draw_event_hits_and_tracks(input_tree, geo, track_lines)
            canvas = histograms['simpleDisplay']
            canvas.SetName(f"c_Run{run_number}_{event_number}")
            canvas.Write()

        processed_count += 1



    if buffer:
        flush_to_parquet(buffer, batch_counter, columns, args.output_parquet)

    if output_root_file:
        output_root_file.cd()
        output_tree.Write()
        output_root_file.Close()
    print(f"Finished in {time.time() - start_time_process:.2f}s.")

if __name__ == "__main__":
    main()
