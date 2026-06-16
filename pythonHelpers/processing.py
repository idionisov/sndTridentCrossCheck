import os
import random
import ROOT
import json
import time
import numpy as np
import pandas as pd
import rootUtils as ut

from pythonHelpers.general import get_scifi_hit_density, get_scifi_total_qdc, flush_to_parquet
from pythonHelpers.hough import run_hough_transform, get_line_params
from pythonHelpers.geometry import initialize_event_display, draw_event_hits_and_tracks

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

def run_hough_selection_data(
    input_file_path: str,
    output_parquet: str,
    output_root: Optional[str] = None,
    start_event: int = 0,
    n_events: int = 1000000,
    fraction: float = 1.0,
    par_file: str = os.path.join(os.path.dirname(__file__), '..', 'parFiles', 'TrackingParams.xml'),
    z_vtx_min: Optional[float] = None,
    z_vtx_max: Optional[float] = None,
    gallery_file: Optional[str] = None
):
    event_buffer = []
    batch_size = 1000

    histograms = {}
    projections = {1: 'xz', 2: 'yz'}

    start_time_process = time.time()

    gallery = {}
    if gallery_file is not None:
        try:
            with open(gallery_file, "r") as f:
                gallery = json.load(f)
            # Convert lists to sets for O(1) lookup
            for r in gallery:
                gallery[r] = set(gallery[r])
            print(f"Loaded gallery from {gallery_file}")
        except Exception as e:
            print(f"Error loading gallery {gallery_file}: {e}")
            return

    input_file = ROOT.TFile.Open(input_file_path)
    if not input_file:
        print(f"Error: Could not open input file {input_file_path}")
        return

    input_tree = input_file.Get("rawConv")
    tree_name = "rawConv"
    if not input_tree:
        input_tree = input_file.Get("cbmsim")
        tree_name = "cbmsim"

    if not input_tree:
        print(f"Error: Could not find rawConv or cbmsim tree in {input_file_path}")
        return

    total_entries = input_tree.GetEntries()
    print(f"Total entries in tree: {total_entries}")

    input_tree.GetEntry(0)
    run_number = input_tree.EventHeader.GetRunId()
    print(f"Run number: {run_number}")

    fair_run = ROOT.FairRunAna()
    io_manager = ROOT.FairRootManager.Instance()
    io_manager.SetTreeName(tree_name)

    fair_run.SetSource(ROOT.FairFileSource(input_file))
    fair_run.SetSink(ROOT.FairRootFileSink(ROOT.TMemFile('dummy','CREATE')))

    rtdb = fair_run.GetRuntimeDb()
    if os.path.exists(par_file):
        print(f"Found parameter file: {par_file}")
        par_source = ROOT.FairParAsciiFileIo()

        if not par_source.open(par_file, 'in'):
            print(f"Error: Failed to open parameter file {par_file}")
            return

        rtdb.setFirstInput(par_source)
        rtdb.addRun(int(run_number))

        rtdb.setOutput(par_source)
    else:
        print(f"Warning: Parameter file {par_file} not found!")


    geo = SndlhcGeo.GeoInterface(ROOT.snd.analysis_tools.GetGeoPath(run_number))
    initialize_event_display(geo, histograms)

    muon_reco_task = SndlhcMuonReco.MuonReco()
    muon_reco_task.SetParFile(par_file)
    fair_run.AddTask(muon_reco_task)
    muon_reco_task.SetHoughSpaceFormat("linearSlopeIntercept")
    muon_reco_task.SetTrackingCase('muon_trident_Sf')

    fair_run.Init()

    input_tree = io_manager.GetInTree()
    if input_tree.GetBranch('Digi_MuFilterHit'):
        input_tree.Digi_MuFilterHits = input_tree.Digi_MuFilterHit

    output_root_file, output_tree = None, None
    if output_root:
        output_root_file = ROOT.TFile(output_root if output_root.endswith(".root") else f"{output_root}.root", "RECREATE")
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

    gallery_match_count = 0
    progress_step = 0

    end_event = min(start_event + n_events, total_entries)
    num_to_scan = end_event - start_event

    print(f"Scanning entries {start_event} to {end_event} (max {n_events} entries)")

    for entry_index in range(start_event, end_event):
        current_progress_pct = int((entry_index - start_event) * 100 / num_to_scan) if num_to_scan > 0 else 100
        if current_progress_pct >= progress_step:
            elapsed = int(time.time() - start_time_process)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            print(f"[{current_progress_pct}%] \t {entry_index-start_event:,}/{num_to_scan:,} \t {h:02d}:{m:02d}:{s:02d}")
            progress_step = ((current_progress_pct // 10) + 1) * 10

        input_tree.GetEntry(entry_index)
        event_number = input_tree.EventHeader.GetEventNumber()
        current_run = input_tree.EventHeader.GetRunId()

        if gallery_file:
            run_str = str(current_run)
            if run_str not in gallery or event_number not in gallery[run_str]:
                continue
            gallery_match_count += 1
        else:
            if fraction < 1.0 and random.random() >= fraction:
                continue

        if input_tree.EventHeader.ClassName() == 'SNDLHCEventHeader':
            geo.modules['Scifi'].InitEvent(input_tree.EventHeader)

        n_lines, track_lines = run_hough_transform(
            muon_reco_task, input_tree, geo,
            z_vtx_min=z_vtx_min, z_vtx_max=z_vtx_max
        )

        if n_lines >= 1:
            row = {
                'run': current_run,
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
                flush_to_parquet(buffer, batch_counter, columns, output_parquet)
                buffer = []
                batch_counter += 1


        if n_lines >= 2 and output_tree:
            print(f"Event {event_number}\tnLines: {n_lines}")
            output_tree.Fill()
            draw_event_hits_and_tracks(input_tree, geo, track_lines, histograms, projections)
            canvas = histograms['simpleDisplay']
            canvas.SetName(f"c_Run{current_run}_{event_number}")
            canvas.Write()

    if gallery_file:
        print(f"Gallery matches found: {gallery_match_count} out of {num_to_scan} entries scanned.")

    if buffer:
        flush_to_parquet(buffer, batch_counter, columns, output_parquet)

    if output_root_file:
        output_root_file.cd()
        output_tree.Write()
        output_root_file.Close()
    print(f"Finished in {time.time() - start_time_process:.2f}s.")
