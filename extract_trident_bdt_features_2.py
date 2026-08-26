#!/usr/bin/env python3
"""
================================================================================
SND@LHC Trident ML / BDT Feature Extractor
================================================================================
Extracts physics observables, Hough tracking parameters, SciFi ionization / 
densities, and MuFilter spectrometer signatures from processed MC/Data ROOT 
files (including 'filter_tridents_mc.py --keep-all' files) into Apache Parquet 
format ready for BDT (XGBoost / LightGBM / scikit-learn) training.

Features Extracted:
- Target Labels & Weights: is_signal, proc_type, region_type, mc_weight, truth kinematics
- SciFi Ionization / Densities: total QDC, max QDC, hit/QDC densities, station profiles, asymmetry
- Hough Kinematics: n_lines, slopes, intercepts, opening angles, hits/QDC per line
- MuFilter Spectrometer: US/DS hits, QDC, plane multiplicities, penetration depth

Memory & Large Dataset Handling:
- Isolated multiprocessing with 'spawn' context and worker recycling ('maxtasksperchild')
- Streaming PyArrow Parquet merge (zero-RAM footprint for arbitrarily large datasets)
- Resumable file skipping ('--skip-existing')
- In-worker memory cleanup and defensive ROOT I/O checks

Author: SND@LHC Collaboration
================================================================================
"""

import os
import sys
import re
import glob
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ROOT

# Set ROOT batch mode
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

# Load SND@LHC C++ Libraries
for lib in ["libBase", "libShipData", "libshipLHC", "libsnd_analysis_tools"]:
    ROOT.gSystem.Load(lib)

sndsw_path = os.environ.get("SNDSW_ROOT", "")
if sndsw_path:
    scifi_tools_h = os.path.join(sndsw_path, "analysis/tools/sndSciFiTools.h")
    if os.path.exists(scifi_tools_h):
        ROOT.gInterpreter.ProcessLine(f'#include "{scifi_tools_h}"')
    geo_getter_h = os.path.join(sndsw_path, "analysis/tools/sndGeometryGetter.h")
    if os.path.exists(geo_getter_h):
        ROOT.gInterpreter.ProcessLine(f'#include "{geo_getter_h}"')

# Declare Fast C++ SciFi and Density Metric Computations
if not hasattr(ROOT, "computeSciFiMLMetrics"):
    ROOT.gInterpreter.Declare("""
    #include "sndScifiHit.h"
    #include "TClonesArray.h"
    #include "sndSciFiTools.h"
    #include <vector>
    #include <algorithm>
    #include <cmath>

    struct SciFiMLMetrics {
        int sf_nhits;
        double sum_qdc;
        double max_qdc;
        double mean_qdc;
        double sum_hit_w_density;
        double max_hit_w_density;
        double sum_qdc_w_density;
        double max_qdc_w_density;
        int max_sf_nhits_per_plane;
        double max_sf_qdc_per_plane;
        
        // Station by station profiles
        int nhits_st[5];
        double qdc_st[5];
        
        // Per-hit arrays
        std::vector<double> hit_qdc;
        std::vector<double> hit_w_density;
        std::vector<double> qdc_w_density;
    };

    bool validateHitLocal(sndScifiHit *aHit, int ref_station, bool ref_orientation) {
        if (!aHit || !aHit->isValid()) return false;
        if (aHit->GetStation() != ref_station) return false;
        if (aHit->isVertical() != ref_orientation) return false;
        return true;
    }

    double qdcDensityLocal(int reference_SiPM, TClonesArray* scifi_hits, int radius) {
        if (!scifi_hits) return 0.0;
        double qdc_density = 0.0;
        bool orientation = (int(reference_SiPM / 100000) % 10 == 1);
        int ref_station = reference_SiPM / 1000000;
        int referenceChannel = snd::analysis_tools::calculateSiPMNumber(reference_SiPM);
        
        int entries = scifi_hits->GetEntries();
        for (int i = 0; i < entries; ++i) {
            sndScifiHit* hit = (sndScifiHit*)scifi_hits->At(i);
            if (!hit || !validateHitLocal(hit, ref_station, orientation)) continue;
            int hitChannel = snd::analysis_tools::calculateSiPMNumber(hit->GetChannelID());
            if (radius == -1) {
                qdc_density += hit->GetSignal();
            } else {
                if (hitChannel > referenceChannel + radius) break;
                if (std::abs(referenceChannel - hitChannel) <= radius) {
                    qdc_density += hit->GetSignal();
                }
            }
        }
        return qdc_density;
    }

    SciFiMLMetrics computeSciFiMLMetrics(TClonesArray* scifi_hits, double radius, bool min_check, double min_hit_density) {
        SciFiMLMetrics m = {0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, {0,0,0,0,0}, {0.0,0.0,0.0,0.0,0.0}, {}, {}, {}};
        if (!scifi_hits) return m;

        int entries = scifi_hits->GetEntries();
        m.sf_nhits = entries;
        m.hit_qdc.resize(entries, 0.0);
        m.hit_w_density.resize(entries, 0.0);
        m.qdc_w_density.resize(entries, 0.0);

        int nhits_per_plane[2][5] = {{0}};
        double qdc_per_plane[2][5] = {{0.0}};

        for (int i = 0; i < entries; ++i) {
            sndScifiHit* hit = (sndScifiHit*)scifi_hits->At(i);
            if (!hit) continue;

            double qdc = hit->GetSignal();
            m.hit_qdc[i] = qdc;
            if (!hit->isValid()) continue;

            m.sum_qdc += qdc;
            if (qdc > m.max_qdc) m.max_qdc = qdc;

            int is_vert = hit->isVertical() ? 1 : 0;
            int st = hit->GetStation();
            if (st >= 1 && st <= 5) {
                m.nhits_st[st - 1] += 1;
                m.qdc_st[st - 1] += qdc;
                nhits_per_plane[is_vert][st - 1] += 1;
                qdc_per_plane[is_vert][st - 1] += qdc;
            }

            double hit_w = (double)snd::analysis_tools::densityScifi(hit->GetChannelID(), *scifi_hits, radius, min_hit_density, min_check);
            m.hit_w_density[i] = hit_w;
            m.sum_hit_w_density += hit_w;
            if (hit_w > m.max_hit_w_density) m.max_hit_w_density = hit_w;

            double qdc_w = qdcDensityLocal(hit->GetChannelID(), scifi_hits, radius);
            m.qdc_w_density[i] = qdc_w;
            m.sum_qdc_w_density += qdc_w;
            if (qdc_w > m.max_qdc_w_density) m.max_qdc_w_density = qdc_w;
        }

        if (entries > 0) m.mean_qdc = m.sum_qdc / entries;

        for (int orient = 0; orient < 2; ++orient) {
            for (int p = 0; p < 5; ++p) {
                if (nhits_per_plane[orient][p] > m.max_sf_nhits_per_plane) m.max_sf_nhits_per_plane = nhits_per_plane[orient][p];
                if (qdc_per_plane[orient][p] > m.max_sf_qdc_per_plane) m.max_sf_qdc_per_plane = qdc_per_plane[orient][p];
            }
        }

        return m;
    }
    """)

import SndlhcGeo
import SndlhcMuonReco

def find_candidate_file(candidates: List[str]) -> str:
    """Finds the first existing file from a list of candidate paths."""
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return candidates[0] if candidates else ""

def load_run_list(source: Optional[str]) -> set:
    """Loads a set of integer run IDs from a file or string."""
    if not source:
        return set()
    if os.path.isfile(source):
        with open(source, "r") as f:
            content = f.read()
        return {int(x) for x in re.findall(r"\b\d+\b", content)}
    elif isinstance(source, str):
        return {int(x) for x in re.findall(r"\b\d+\b", source)}
    return set()

def extract_run_number(file_path: str) -> Optional[int]:
    """Extracts run number from filename (e.g. run_005132, run-6250, sndsw_raw-0010)."""
    m = re.search(r"(?:run[_-]|run)0*(\d+)", file_path, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"sndsw_raw-0*(\d+)", file_path, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

DEFAULT_GEOFILE = "python/geofile_full.Ntuple-TGeant4_boost100.0.root"
DEFAULT_GEO_2022 = find_candidate_file([
    "/eos/user/i/idioniso/sndMuTri/data/geofile_data_2022.root",
    "geofile_data_2022.root",
    "data/geofile_data_2022.root",
    "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V0_2022.root"
])
DEFAULT_GEO_2023 = find_candidate_file([
    "/eos/user/i/idioniso/sndMuTri/data/geofile_data_2023.root",
    "geofile_data_2023.root",
    "data/geofile_data_2023.root",
    "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V0_2023.root"
])

cand_par = os.path.join(os.getcwd(), "TrackingParams_sf4.xml")
DEFAULT_PARFILE = cand_par if os.path.exists(cand_par) else (
    os.path.join(sndsw_path, "python", "TrackingParams_V1_28November2022.xml") if sndsw_path else "python/TrackingParams_V1_28November2022.xml"
)

# ==============================================================================
# Helper Extraction Functions
# ==============================================================================

def run_hough_tracking(muon_reco_task, event, geo, max_lines=3):
    """
    Extracts detector hits (with 3D fiber center midpoints) and executes iterative 
    Hough line finding, masking used hits to reconstruct up to max_lines per projection.
    """
    hit_collection = {
        "pos": [[], [], []],
        "d": [[], [], []],
        "vert": [],
        "system": [],
        "detectorID": [],
        "qdc": []
    }

    pos_a, pos_b = ROOT.TVector3(), ROOT.TVector3()

    scifi_hits = getattr(event, "Digi_ScifiHits", None)
    if scifi_hits:
        for hit in scifi_hits:
            if not hit.isValid():
                continue
            geo.modules["Scifi"].GetSiPMPosition(hit.GetDetectorID(), pos_a, pos_b)
            hit_collection["pos"][0].append((pos_a.X() + pos_b.X()) * 0.5)
            hit_collection["pos"][1].append((pos_a.Y() + pos_b.Y()) * 0.5)
            hit_collection["pos"][2].append((pos_a.Z() + pos_b.Z()) * 0.5)
            hit_collection["d"][0].append(muon_reco_task.Scifi_dx)
            hit_collection["d"][1].append(muon_reco_task.Scifi_dy)
            hit_collection["d"][2].append(muon_reco_task.Scifi_dz)
            hit_collection["vert"].append(hit.isVertical())
            hit_collection["system"].append(0)
            hit_collection["detectorID"].append(hit.GetDetectorID())
            hit_collection["qdc"].append(hit.GetSignal())

    if not hit_collection["pos"][0]:
        return 0, 0, 0, {}, {}

    for k in hit_collection:
        if k in ["pos", "d"]:
            hit_collection[k] = [np.array(hit_collection[k][dim], dtype=np.float64) for dim in range(3)]
        elif k == "vert":
            hit_collection[k] = np.array(hit_collection[k], dtype=np.bool_)
        else:
            hit_collection[k] = np.array(hit_collection[k])

    lines = {"XZ": [], "YZ": []}
    track_hit_indices = {"XZ": [], "YZ": []}

    for is_vertical, axis, projection_name in [(True, 0, "XZ"), (False, 1, "YZ")]:
        hough_obj = muon_reco_task.h_ZX if is_vertical else muon_reco_task.h_ZY
        hits_used = np.zeros(len(hit_collection["pos"][0]), dtype=np.bool_)

        valid_lines = 0
        attempts = 0
        while valid_lines < max_lines and attempts < 20:
            attempts += 1
            mask = np.logical_and(hit_collection["vert"] == is_vertical, ~hits_used)
            if not np.any(mask):
                break

            fit_res = hough_obj.fit_randomize(
                np.dstack([hit_collection["pos"][2][mask], hit_collection["pos"][axis][mask]])[0],
                np.dstack([hit_collection["d"][2][mask], hit_collection["d"][axis][mask]])[0],
                muon_reco_task.n_random, False, False
            )

            if fit_res[0] in [-1, -999]:
                break

            slope, intercept = fit_res[0], fit_res[1]
            related_hits = SndlhcMuonReco.hit_finder(
                slope, intercept,
                np.dstack([hit_collection["pos"][2][mask], hit_collection["pos"][axis][mask]]),
                np.dstack([hit_collection["d"][2][mask], hit_collection["d"][axis][mask]]),
                muon_reco_task.tolerance
            )

            if len(related_hits) == 0:
                break

            n_planes = SndlhcMuonReco.numPlanesHit(
                hit_collection["system"][mask][related_hits],
                hit_collection["detectorID"][mask][related_hits]
            )

            if n_planes >= muon_reco_task.min_planes_hit:
                lines[projection_name].append((slope, intercept))
                sel_idx = np.where(mask)[0][related_hits]
                track_hit_indices[projection_name].append(sel_idx.tolist())
                hits_used[sel_idx] = True
                valid_lines += 1
            else:
                break

    n_xz = len(lines["XZ"])
    n_yz = len(lines["YZ"])
    n_total = max(n_xz, n_yz)
    return n_total, n_xz, n_yz, lines, track_hit_indices


def extract_mufilter_features(event):
    """Computes Upstream (US) and Downstream (DS) MuFilter calorimetric and tracking metrics."""
    mufi_hits = getattr(event, "Digi_MuFilterHits", getattr(event, "Digi_MuFilterHit", None))
    if not mufi_hits or mufi_hits.GetEntries() == 0:
        return {
            "mufi_nhits": 0, "mufi_sum_qdc": 0.0,
            "us_nhits": 0, "us_sum_qdc": 0.0, "us_planes_hit": 0,
            "ds_nhits": 0, "ds_sum_qdc": 0.0, "ds_planes_hit": 0,
            "ds_max_hits_per_plane": 0, "ds_max_station_penetrated": 0
        }

    mufi_nhits = mufi_hits.GetEntries()
    mufi_sum_qdc = 0.0
    us_nhits = 0
    us_sum_qdc = 0.0
    ds_nhits = 0
    ds_sum_qdc = 0.0

    us_planes = set()
    ds_planes = set()
    ds_hits_per_plane = [0, 0, 0, 0] # 4 DS stations
    max_ds_st = 0

    for i in range(mufi_nhits):
        hit = mufi_hits.At(i)
        if not hit or not hit.isValid():
            continue
        qdc = hit.GetSignal()
        mufi_sum_qdc += qdc
        sys_id = hit.GetSystem()
        det_id = hit.GetDetectorID()

        if sys_id == 2: # Upstream MuFilter
            us_nhits += 1
            us_sum_qdc += qdc
            plane_id = (det_id % 10000) // 1000
            us_planes.add(plane_id)
        elif sys_id == 3: # Downstream MuFilter
            ds_nhits += 1
            ds_sum_qdc += qdc
            plane_id = (det_id % 10000) // 1000
            ds_planes.add(plane_id)
            if 0 <= plane_id < 4:
                ds_hits_per_plane[plane_id] += 1
                if (plane_id + 1) > max_ds_st:
                    max_ds_st = plane_id + 1

    return {
        "mufi_nhits": mufi_nhits,
        "mufi_sum_qdc": mufi_sum_qdc,
        "us_nhits": us_nhits,
        "us_sum_qdc": us_sum_qdc,
        "us_planes_hit": len(us_planes),
        "ds_nhits": ds_nhits,
        "ds_sum_qdc": ds_sum_qdc,
        "ds_planes_hit": len(ds_planes),
        "ds_max_hits_per_plane": max(ds_hits_per_plane) if ds_hits_per_plane else 0,
        "ds_max_station_penetrated": max_ds_st
    }


def process_single_file_worker(args_tuple):
    """Worker function for processing one ROOT file in a separate process."""
    input_file, output_parquet, geofile, parfile, first_event, max_events, skip_existing = args_tuple
    t0 = time.time()

    # Fast resume check
    if skip_existing and os.path.exists(output_parquet) and os.path.getsize(output_parquet) > 0:
        try:
            meta = pq.read_metadata(output_parquet)
            n_rows = meta.num_rows
            print(f"  [SKIP] '{os.path.basename(input_file)}' already processed -> '{os.path.basename(output_parquet)}' ({n_rows:,} rows)")
            return n_rows, 0
        except Exception:
            pass

    try:
        f_in = ROOT.TFile.Open(input_file, "READ")
    except Exception as e:
        print(f"Worker Error: Exception opening {input_file}: {e}")
        return 0, 0

    if not f_in or f_in.IsZombie():
        print(f"Worker Error: Could not open or zombie file: {input_file}")
        if f_in:
            f_in.Close()
        return 0, 0

    t_name = "rawConv" if f_in.Get("rawConv") else "cbmsim"
    tree = f_in.Get(t_name)
    if not tree:
        print(f"Worker Error: Neither 'cbmsim' nor 'rawConv' tree found in {input_file}")
        f_in.Close()
        return 0, 0

    # Determine MC Geofile
    if not os.path.exists(geofile):
        in_dir = os.path.dirname(input_file)
        cand = os.path.join(in_dir, "geofile_full.Ntuple-TGeant4_boost100.0.root")
        if os.path.exists(cand):
            geofile = cand

    geo = None
    try:
        geo = SndlhcGeo.GeoInterface(geofile)
    except Exception as e:
        print(f"Worker Warning: Failed to load geometry '{geofile}': {e}")

    # Inspect available branches in ROOT tree
    all_branch_objs = list(tree.GetListOfBranches())
    available_branches = {b.GetName() for b in all_branch_objs}

    # Identify all scalar/primitive branches to copy directly (skip large object collections)
    scalar_branches = []
    for b in all_branch_objs:
        bname = b.GetName()
        cname = b.GetClassName()
        if cname in ["TClonesArray", "SNDLHCEventHeader"] or bname.startswith("MCTrack") or bname.startswith("Digi_") or bname.startswith("ScifiPoint") or bname.startswith("MuFilterPoint") or bname.startswith("EmulsionDetPoint"):
            continue
        scalar_branches.append(bname)

    has_precomputed_hough = ("xz_sf_m1" in available_branches) or ("xz_m1" in available_branches)

    # Initialize FairRunAna / MuonReco Task ONLY if Hough tracking is not already present in the ROOT tree
    muon_reco_task = None
    fair_run = None
    if not has_precomputed_hough and geo:
        fair_run = ROOT.FairRunAna()
        io_manager = ROOT.FairRootManager.Instance()
        io_manager.SetTreeName(t_name)
        fair_run.SetSource(ROOT.FairFileSource(f_in))
        fair_run.SetSink(ROOT.FairRootFileSink(ROOT.TMemFile("dummy_sink", "CREATE")))

        rtdb = fair_run.GetRuntimeDb()
        if os.path.exists(parfile):
            par_source = ROOT.FairParAsciiFileIo()
            if par_source.open(parfile, "in"):
                rtdb.setFirstInput(par_source)
                rtdb.addRun(1)
                rtdb.setOutput(par_source)

        muon_reco_task = SndlhcMuonReco.MuonReco()
        muon_reco_task.SetParFile(parfile)
        fair_run.AddTask(muon_reco_task)
        muon_reco_task.SetHoughSpaceFormat("linearSlopeIntercept")
        
        try:
            import xml.etree.ElementTree as ET
            tree_xml = ET.parse(parfile)
            cases = [c.attrib.get("name") for c in tree_xml.getroot().findall("tracking_case")]
            if "muon_trident_Sf" in cases:
                muon_reco_task.SetTrackingCase("muon_trident_Sf")
            else:
                muon_reco_task.SetTrackingCase("passing_mu_Sf")
        except Exception:
            muon_reco_task.SetTrackingCase("passing_mu_Sf")

        fair_run.Init()

    total_entries = tree.GetEntries()
    start_ev = max(0, first_event)
    if start_ev >= total_entries:
        print(f"Worker Warning: start_event ({start_ev}) >= total_entries ({total_entries}) in {input_file}")
        f_in.Close()
        return 0, 0

    end_ev = min(total_entries, start_ev + max_events) if max_events > 0 else total_entries

    proc_id_map = {"non_signal": 0, "genuine": 1, "gamma_conv": 2, "positron_annihil": 3, "secondary_muon": 4, "hadronic": 5}
    region_id_map = {"upstream_rock": 0, "target": 1, "muon_system": 2, "unknown": -1}

    rows = []

    for i_event in range(start_ev, end_ev):
        tree.GetEntry(i_event)
        event = tree

        # Apply SciFi run/event alignment if real data header present
        if geo and hasattr(event, "EventHeader") and hasattr(event.EventHeader, "ClassName") and event.EventHeader.ClassName() == "SNDLHCEventHeader":
            try:
                geo.modules["Scifi"].InitEvent(event.EventHeader)
            except Exception:
                pass

        # Run and Event Identification
        run_id = event.EventHeader.GetRunId() if hasattr(event, "EventHeader") and hasattr(event.EventHeader, "GetRunId") else 1
        event_id = event.EventHeader.GetEventNumber() if hasattr(event, "EventHeader") and hasattr(event.EventHeader, "GetEventNumber") else i_event

        row = {
            "run_id": run_id,
            "event_id": event_id,
            "entry_index": i_event,
        }

        # 1. Directly copy all existing scalar/truth/reconstructed branches from the ROOT tree
        for bname in scalar_branches:
            val = getattr(event, bname, None)
            if val is not None:
                if isinstance(val, (int, float, bool, np.number)):
                    row[bname] = float(val) if not isinstance(val, (int, bool, np.integer, np.bool_)) else int(val)
                else:
                    row[bname] = str(val)

        # Target Labels & Truth ID mapping
        proc_str = str(getattr(event, "proc_type", row.get("proc_type", "non_signal")))
        region_str = str(getattr(event, "region_type", row.get("region_type", "unknown")))
        if "proc_id" not in row:
            row["proc_id"] = proc_id_map.get(proc_str, 0)
        if "region_id" not in row:
            row["region_id"] = region_id_map.get(region_str, -1)
        if "is_signal" not in row:
            row["is_signal"] = int(getattr(event, "is_signal", 0))
        if "mc_weight" not in row:
            row["mc_weight"] = float(getattr(event, "mc_weight", 1.0))

        # 2. SciFi Event-Level Ionization & Station Profiles
        scifi_hits = getattr(event, "Digi_ScifiHits", getattr(event, "Digits_Scifi", None))
        sf_m = None
        if "scifi_nhits_st1" not in row or "scifi_sum_qdc" not in row or not has_precomputed_hough:
            sf_m = ROOT.computeSciFiMLMetrics(scifi_hits, 40.0, False, 1000000.0) if scifi_hits else None
            if sf_m:
                if "scifi_nhits" not in row: row["scifi_nhits"] = sf_m.sf_nhits
                if "scifi_sum_qdc" not in row: row["scifi_sum_qdc"] = sf_m.sum_qdc
                if "scifi_max_qdc" not in row: row["scifi_max_qdc"] = sf_m.max_qdc
                if "scifi_mean_qdc" not in row: row["scifi_mean_qdc"] = sf_m.mean_qdc
                if "scifi_sum_hit_w_density" not in row: row["scifi_sum_hit_w_density"] = sf_m.sum_hit_w_density
                if "scifi_max_hit_w_density" not in row: row["scifi_max_hit_w_density"] = sf_m.max_hit_w_density
                if "scifi_sum_qdc_w_density" not in row: row["scifi_sum_qdc_w_density"] = sf_m.sum_qdc_w_density
                if "scifi_max_qdc_w_density" not in row: row["scifi_max_qdc_w_density"] = sf_m.max_qdc_w_density
                if "scifi_max_nhits_plane" not in row: row["scifi_max_nhits_plane"] = sf_m.max_sf_nhits_per_plane
                if "scifi_max_qdc_plane" not in row: row["scifi_max_qdc_plane"] = sf_m.max_sf_qdc_per_plane

                for st in range(1, 6):
                    row[f"scifi_nhits_st{st}"] = sf_m.nhits_st[st - 1]
                    row[f"scifi_qdc_st{st}"] = sf_m.qdc_st[st - 1]
                row["scifi_qdc_ratio_down_up"] = ((sf_m.qdc_st[3] + sf_m.qdc_st[4]) / (sf_m.qdc_st[0] + sf_m.qdc_st[1] + 1e-3))

        # 3. MuFilter Spectrometer & Penetration Observables
        if "ds_planes_hit" not in row or "us_planes_hit" not in row:
            mufi_feats = extract_mufilter_features(event)
            for k, v in mufi_feats.items():
                if k not in row:
                    row[k] = v

        # 4. On-the-fly Hough Tracking (if missing from input ROOT file)
        if not has_precomputed_hough and muon_reco_task and geo:
            n_lines, n_xz, n_yz, track_lines, track_hit_indices = run_hough_tracking(muon_reco_task, event, geo, max_lines=3)
            row["n_lines"] = n_lines
            row["n_lines_sf"] = n_lines
            row["n_lines_sf_xz"] = n_xz
            row["n_lines_sf_yz"] = n_yz

            xz_lines = track_lines.get("XZ", [])
            yz_lines = track_lines.get("YZ", [])

            row["delta_slope_xz_12"] = abs(xz_lines[0][0] - xz_lines[1][0]) if len(xz_lines) >= 2 else np.nan
            row["delta_slope_xz_13"] = abs(xz_lines[0][0] - xz_lines[2][0]) if len(xz_lines) >= 3 else np.nan
            row["delta_slope_yz_12"] = abs(yz_lines[0][0] - yz_lines[1][0]) if len(yz_lines) >= 2 else np.nan
            row["delta_slope_yz_13"] = abs(yz_lines[0][0] - yz_lines[2][0]) if len(yz_lines) >= 3 else np.nan

            for proj in ["xz", "yz"]:
                proj_upper = proj.upper()
                p_lines = track_lines.get(proj_upper, [])
                p_indices = track_hit_indices.get(proj_upper, [])

                for l_idx in range(1, 4):
                    slot = l_idx - 1
                    if slot < len(p_lines):
                        m, c = p_lines[slot]
                        indices = p_indices[slot] if slot < len(p_indices) else []
                        row[f"{proj}_sf_m{l_idx}"] = float(m)
                        row[f"{proj}_sf_c{l_idx}"] = float(c)
                        row[f"{proj}_m{l_idx}"] = float(m)
                        row[f"{proj}_c{l_idx}"] = float(c)
                        row[f"{proj}_sf_n_hits_{l_idx}"] = len(indices)
                        row[f"{proj}_n_hits_{l_idx}"] = len(indices)

                        if sf_m and len(indices) > 0:
                            l_qdcs = [sf_m.hit_qdc[i] for i in indices if i < len(sf_m.hit_qdc)]
                            l_hit_w = [sf_m.hit_w_density[i] for i in indices if i < len(sf_m.hit_w_density)]
                            l_qdc_w = [sf_m.qdc_w_density[i] for i in indices if i < len(sf_m.qdc_w_density)]
                            row[f"{proj}_sf_sum_qdc_{l_idx}"] = sum(l_qdcs)
                            row[f"{proj}_sf_max_qdc_{l_idx}"] = max(l_qdcs) if l_qdcs else 0.0
                            row[f"{proj}_sum_qdc_{l_idx}"] = sum(l_qdcs)
                            row[f"{proj}_max_qdc_{l_idx}"] = max(l_qdcs) if l_qdcs else 0.0
                            row[f"{proj}_sf_sum_hit_w_density_{l_idx}"] = sum(l_hit_w)
                            row[f"{proj}_sf_max_hit_w_density_{l_idx}"] = max(l_hit_w) if l_hit_w else 0.0
                            row[f"{proj}_sum_hit_w_density_{l_idx}"] = sum(l_hit_w)
                            row[f"{proj}_max_hit_w_density_{l_idx}"] = max(l_hit_w) if l_hit_w else 0.0
                            row[f"{proj}_sf_sum_qdc_w_density_{l_idx}"] = sum(l_qdc_w)
                            row[f"{proj}_sf_max_qdc_w_density_{l_idx}"] = max(l_qdc_w) if l_qdc_w else 0.0
                            row[f"{proj}_sum_qdc_w_density_{l_idx}"] = sum(l_qdc_w)
                            row[f"{proj}_max_qdc_w_density_{l_idx}"] = max(l_qdc_w) if l_qdc_w else 0.0
                    else:
                        row[f"{proj}_sf_m{l_idx}"] = np.nan
                        row[f"{proj}_sf_c{l_idx}"] = np.nan
                        row[f"{proj}_m{l_idx}"] = np.nan
                        row[f"{proj}_c{l_idx}"] = np.nan
                        row[f"{proj}_sf_n_hits_{l_idx}"] = 0
                        row[f"{proj}_n_hits_{l_idx}"] = 0

        # 5. Populate standard aliases and compute delta slopes if missing
        if "xz_sf_m1" in row and "xz_m1" not in row: row["xz_m1"] = row["xz_sf_m1"]
        if "xz_sf_c1" in row and "xz_c1" not in row: row["xz_c1"] = row["xz_sf_c1"]
        if "yz_sf_m1" in row and "yz_m1" not in row: row["yz_m1"] = row["yz_sf_m1"]
        if "yz_sf_c1" in row and "yz_c1" not in row: row["yz_c1"] = row["yz_sf_c1"]
        if "n_lines_sf" in row and "n_lines" not in row: row["n_lines"] = row["n_lines_sf"]

        if "delta_slope_xz_12" not in row and "xz_sf_m1" in row and "xz_sf_m2" in row:
            m1, m2 = row.get("xz_sf_m1", np.nan), row.get("xz_sf_m2", np.nan)
            row["delta_slope_xz_12"] = abs(m1 - m2) if not np.isnan(m1) and not np.isnan(m2) else np.nan
        if "delta_slope_yz_12" not in row and "yz_sf_m1" in row and "yz_sf_m2" in row:
            m1, m2 = row.get("yz_sf_m1", np.nan), row.get("yz_sf_m2", np.nan)
            row["delta_slope_yz_12"] = abs(m1 - m2) if not np.isnan(m1) and not np.isnan(m2) else np.nan

        rows.append(row)

    f_in.Close()

    # Save to Parquet using PyArrow
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_parquet, engine="pyarrow", compression="snappy", index=False)

    elapsed = time.time() - t0
    n_sig = int(df["is_signal"].sum()) if "is_signal" in df else 0
    n_total = len(df)
    n_cols = len(df.columns)
    
    # Prompt memory cleanup
    del rows, df

    print(f"  -> Done '{os.path.basename(input_file)}': {n_total:,} events ({n_sig:,} signal, {n_cols} features) in {elapsed:.2f}s -> '{os.path.basename(output_parquet)}'")
    return n_total, n_sig


# ==============================================================================
# Streaming Parquet Merger (Zero-RAM Overhead)
# ==============================================================================

def stream_merge_parquet_files(input_files: List[str], output_file: str, clean_intermediate: bool = False) -> Tuple[int, float]:
    """
    Streams individual Parquet files into a master Parquet file one by one.
    Memory footprint remains constant (~30-50 MB) regardless of total data size.
    """
    t0 = time.time()
    valid_files = [f for f in input_files if os.path.exists(f) and os.path.getsize(f) > 0]
    
    if not valid_files:
        print(f"  [Merge Warning] No valid non-empty files found to merge into '{output_file}'")
        return 0, 0.0

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    writer = None
    master_schema = None
    total_rows = 0

    print(f"\n[STREAM MERGE] Merging {len(valid_files):,} Parquet files into '{output_file}'...")

    for idx, f in enumerate(valid_files):
        try:
            table = pq.read_table(f)
            if table.num_rows == 0:
                continue

            if writer is None:
                master_schema = table.schema
                writer = pq.ParquetWriter(output_file, master_schema, compression="snappy")

            # Schema alignment
            if table.schema != master_schema:
                try:
                    table = table.cast(master_schema)
                except Exception:
                    # Column reordering to match master schema
                    cols = [table.column(name) if name in table.column_names else pa.nulls(len(table), type=master_schema.field(name).type) for name in master_schema.names]
                    table = pa.Table.from_arrays(cols, schema=master_schema)

            writer.write_table(table)
            total_rows += table.num_rows

            if (idx + 1) % 50 == 0 or (idx + 1) == len(valid_files):
                print(f"  [{idx + 1}/{len(valid_files)}] Merged {total_rows:,} rows so far...")

            # Clean intermediate file if requested
            if clean_intermediate and os.path.abspath(f) != os.path.abspath(output_file):
                try:
                    os.remove(f)
                except OSError:
                    pass

        except Exception as e:
            print(f"  [Merge Warning] Failed to process '{os.path.basename(f)}': {e}")

    if writer:
        writer.close()

    elapsed = time.time() - t0
    out_size_mb = os.path.getsize(output_file) / (1024 * 1024) if os.path.exists(output_file) else 0.0
    return total_rows, out_size_mb


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract Tabular Machine Learning / BDT Feature Dataset from SND@LHC MC/Data into Apache Parquet."
    )
    parser.add_argument("-i", "--input", dest="input_pattern", required=True, help="Input ROOT file or glob pattern (e.g. 'trimuon_filtered_*.root')")
    parser.add_argument("-o", "--output", dest="output_pattern", default="trimuon_features_%s.parquet", help="Output Parquet path pattern with %%s for tag (default: %(default)s)")
    parser.add_argument("-g", "--geoFile", dest="geo_file", default=DEFAULT_GEOFILE, help="Path to default/fallback geometry ROOT file")
    parser.add_argument("--geo-2022", dest="geo_2022", default=DEFAULT_GEO_2022, help="Path to 2022 geometry ROOT file (default: %(default)s)")
    parser.add_argument("--geo-2023", dest="geo_2023", default=DEFAULT_GEO_2023, help="Path to 2023 geometry ROOT file (default: %(default)s)")
    parser.add_argument("--runs-2022", dest="runs_2022_src", default="2022_gal_runs.txt" if os.path.exists("2022_gal_runs.txt") else "", help="2022 run list file or comma-separated numbers (default: %(default)s)")
    parser.add_argument("--runs-2023", dest="runs_2023_src", default="2023_gal_runs.txt" if os.path.exists("2023_gal_runs.txt") else "", help="2023 run list file or comma-separated numbers (default: %(default)s)")
    parser.add_argument("--force-geo", dest="force_geo", action="store_true", help="Force single --geoFile for all files, disabling automatic 2022/2023 geometry routing")
    parser.add_argument("-p", "--parFile", dest="par_file", default=DEFAULT_PARFILE, help="Path to TrackingParams XML file")
    parser.add_argument("-s", "--start", dest="start_event", type=int, default=0, help="First event / start entry to process (default: 0)")
    parser.add_argument("-n", "--n-events", dest="n_events", type=int, default=0, help="Number of events to process per file (0 = all, default: 0)")
    parser.add_argument("-j", "--workers", dest="workers", type=int, default=1, help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--skip-existing", "--resume", dest="skip_existing", action="store_true", help="Skip processing files whose output parquet already exists and is non-empty")
    parser.add_argument("--merge", dest="merge_output", default="", help="Optional unified destination path to stream-merge all output parquet files into one master dataset")
    parser.add_argument("--clean-intermediate", dest="clean_intermediate", action="store_true", help="Delete individual parquet files after successful merge into master dataset")

    args = parser.parse_args()

    matched_files = sorted(glob.glob(args.input_pattern))
    if not matched_files:
        if os.path.exists(args.input_pattern):
            matched_files = [args.input_pattern]
        else:
            print(f"Error: No files matching pattern: '{args.input_pattern}'")
            sys.exit(1)

    # Load run lists for year-aware routing
    runs_2022 = load_run_list(args.runs_2022_src)
    runs_2023 = load_run_list(args.runs_2023_src)

    # Resolve tasks and per-file geofiles
    tasks = []
    output_files = []
    geo_counts = {"2022": 0, "2023": 0, "fallback": 0}

    for f in matched_files:
        base = os.path.splitext(os.path.basename(f))[0]
        # Extract tag if pattern contains %s
        if "%s" in args.output_pattern:
            tag = base.replace("trimuon_filtered_", "").replace("trimuon_", "").replace("sndLHC.Ntuple-", "")
            out_p = args.output_pattern % tag
        elif len(matched_files) == 1:
            out_p = args.output_pattern
        else:
            out_dir = os.path.dirname(args.output_pattern) if os.path.dirname(args.output_pattern) else "."
            out_p = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(args.output_pattern))[0]}_{base}.parquet")

        # Determine geometry file
        if args.force_geo:
            assigned_geo = args.geo_file
            geo_counts["fallback"] += 1
        else:
            rn = extract_run_number(f)
            if rn is not None:
                if (runs_2022 and rn in runs_2022) or (not runs_2022 and rn < 6000):
                    assigned_geo = args.geo_2022 if (args.geo_2022 and os.path.exists(args.geo_2022)) else args.geo_file
                    geo_counts["2022"] += 1
                elif (runs_2023 and rn in runs_2023) or (not runs_2023 and rn >= 6000):
                    assigned_geo = args.geo_2023 if (args.geo_2023 and os.path.exists(args.geo_2023)) else args.geo_file
                    geo_counts["2023"] += 1
                else:
                    assigned_geo = args.geo_file
                    geo_counts["fallback"] += 1
            else:
                assigned_geo = args.geo_file
                geo_counts["fallback"] += 1

        output_files.append(out_p)
        tasks.append((f, out_p, assigned_geo, args.par_file, args.start_event, args.n_events, args.skip_existing))

    print("=" * 72)
    print("SND@LHC Trident ML / BDT Feature Extractor")
    print("=" * 72)
    print(f"Input Pattern     : {args.input_pattern}")
    print(f"Files Found       : {len(matched_files):,}")
    print("Geometry Routing  :")
    if not args.force_geo:
        print(f"  * 2022 Data Geo : {args.geo_2022} ({geo_counts['2022']} files)")
        print(f"  * 2023 Data Geo : {args.geo_2023} ({geo_counts['2023']} files)")
        if geo_counts['fallback'] > 0:
            print(f"  * Fallback Geo  : {args.geo_file} ({geo_counts['fallback']} files)")
    else:
        print(f"  * Forced Single : {args.geo_file} (all {len(matched_files)} files)")
    print(f"Tracking Params   : {args.par_file}")
    print(f"Start Event Index : {args.start_event}")
    print(f"Events / File     : {'All Events' if args.n_events == 0 else args.n_events}")
    print(f"Worker Processes  : {args.workers}")
    print(f"Skip Existing     : {args.skip_existing}")
    print(f"Unified Merge     : {args.merge_output if args.merge_output else 'Separate Parquet Files'}")
    print("=" * 72)

    t0_all = time.time()
    total_events = 0
    total_signal = 0

    if args.workers > 1 and len(tasks) > 1:
        # Use spawn context and recycle workers after 10 tasks to prevent ROOT memory leaks
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(args.workers, len(tasks)), maxtasksperchild=10) as pool:
            results = pool.map(process_single_file_worker, tasks)
        for r in results:
            if r:
                total_events += r[0]
                total_signal += r[1]
    else:
        for t in tasks:
            r = process_single_file_worker(t)
            if r:
                total_events += r[0]
                total_signal += r[1]

    # Optional Memory-Safe Unified Streaming Merge
    if args.merge_output and output_files:
        merged_rows, out_size = stream_merge_parquet_files(
            output_files, args.merge_output, clean_intermediate=args.clean_intermediate
        )
        print(f"\n[MERGE COMPLETE] Master dataset saved: {merged_rows:,} rows | Size: {out_size:.1f} MB -> '{args.merge_output}'")

    total_time = time.time() - t0_all
    print("=" * 72)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 72)
    print(f"Total Files Processed : {len(matched_files):,}")
    print(f"Total Events Scanned  : {total_events:,}")
    print(f"Total Signal Events   : {total_signal:,} ({total_signal/max(1, total_events)*100:.2f}%)")
    print(f"Total Background      : {total_events - total_signal:,}")
    print(f"Total Execution Time  : {total_time:.1f} s ({total_events/max(0.1, total_time):.0f} ev/s)")
    print("=" * 72)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
