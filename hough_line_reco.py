#!/usr/bin/env python3
"""
hough_line_reco.py
------------------
Performs Hough transform track/line finding (up to 3 tracks per projection XZ/YZ)
on SND@LHC event files (MC or Real Data) using core SndlhcMuonReco and geometry tools.

Features:
- Stores all original ROOT branches and objects, augmented with reconstructed
  Hough line parameters and hit metrics for both SciFi and Downstream (DS) MuFilter.
- Dedicated and clear branch names:
  * SciFi: xz_sf_m1..3, xz_sf_c1..3, yz_sf_m1..3, yz_sf_c1..3 (and aliases xz_m1..3, xz_c1..3, etc.)
  * Downstream: xz_ds_m1..3, xz_ds_c1..3, yz_ds_m1..3, yz_ds_c1..3
- Hit metrics per line and global summaries for both SciFi and DS systems.
- Configurable tracking systems: 'both' (default), 'scifi', or 'ds'.
- Accepts custom geometry file (--geoFile) with automatic fallback detection.
- Accepts custom tracking parameters XML file (--parFile).
- Optional vertex Z constraint filtering (--z-vtx-range).
- Placeholder values (NaN / 0) for missing lines (when fewer than 3 lines are found).
- Single file or multi-file wildcard processing with format pattern output (e.g. -o "out_%s.root").
- Optional 2D event display canvas storage (--save-displays).
"""

import os
import sys
import glob
import json
import time
import random
import argparse
from array import array
from typing import Optional, Dict, List, Tuple

import ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)

# Load SND@LHC shared libraries
for lib in ["libBase", "libShipData", "libshipLHC", "libsnd_analysis_tools"]:
    ROOT.gSystem.Load(lib)

import SndlhcGeo
import SndlhcMuonReco


def get_event_scifi_info(scifi_hits) -> Dict:
    """
    Computes global SciFi hit statistics, QDC sums, and plane-level hit/QDC densities.
    """
    hits_per_plane = {}
    qdc_per_plane = {}
    qdcs = []

    # First pass: Count hits and sum QDC per plane
    if scifi_hits:
        for hit in scifi_hits:
            if not hit.isValid():
                continue
            det_id = hit.GetDetectorID()
            station = hit.GetStation() if hasattr(hit, "GetStation") else int(str(det_id)[0])
            is_vert = hit.isVertical()
            plane_key = (station, is_vert)

            qdc = hit.GetSignal()
            hits_per_plane[plane_key] = hits_per_plane.get(plane_key, 0) + 1
            qdc_per_plane[plane_key] = qdc_per_plane.get(plane_key, 0.0) + qdc

    # Second pass: compute densities per individual hit
    hit_w_densities = []
    qdc_w_densities = []

    if scifi_hits:
        for hit in scifi_hits:
            if not hit.isValid():
                continue
            det_id = hit.GetDetectorID()
            station = hit.GetStation() if hasattr(hit, "GetStation") else int(str(det_id)[0])
            is_vert = hit.isVertical()
            plane_key = (station, is_vert)

            qdc = hit.GetSignal()
            qdcs.append(qdc)

            hit_w_densities.append(float(hits_per_plane[plane_key]))
            qdc_w_densities.append(float(qdc_per_plane[plane_key]))

    sf_nhits = len(qdcs)
    sum_qdc = sum(qdcs) if sf_nhits > 0 else 0.0
    max_qdc = max(qdcs) if sf_nhits > 0 else 0.0

    max_scifi_nhits_per_plane = max(hits_per_plane.values()) if hits_per_plane else 0
    max_scifi_qdc_per_plane = max(qdc_per_plane.values()) if qdc_per_plane else 0.0

    sum_hit_weight_density = sum(hit_w_densities) if hit_w_densities else 0.0
    max_hit_weight_density = max(hit_w_densities) if hit_w_densities else 0.0
    sum_qdc_weight_density = sum(qdc_w_densities) if qdc_w_densities else 0.0
    max_qdc_weight_density = max(qdc_w_densities) if qdc_w_densities else 0.0

    return {
        "sf_nhits": sf_nhits,
        "qdcs": qdcs,
        "hit_w_densities": hit_w_densities,
        "qdc_w_densities": qdc_w_densities,
        "sum_qdc": sum_qdc,
        "max_qdc": max_qdc,
        "max_scifi_nhits_per_plane": max_scifi_nhits_per_plane,
        "max_scifi_qdc_per_plane": max_scifi_qdc_per_plane,
        "sum_hit_weight_density": sum_hit_weight_density,
        "max_hit_weight_density": max_hit_weight_density,
        "sum_qdc_weight_density": sum_qdc_weight_density,
        "max_qdc_weight_density": max_qdc_weight_density,
    }

def get_event_ds_info(mufi_hits) -> Dict:
    """
    Computes global Downstream (DS) MuFilter hit statistics, QDC sums, and plane-level densities.
    """
    hits_per_plane = {}
    qdc_per_plane = {}
    qdcs = []
    hit_w_densities = []
    qdc_w_densities = []

    ds_hits = []
    if mufi_hits:
        for hit in mufi_hits:
            if not hit.isValid() or hit.GetSystem() != 3:
                continue
            ds_hits.append(hit)
            det_id = hit.GetDetectorID()
            plane = hit.GetPlane() if hasattr(hit, "GetPlane") else (det_id // 1000)
            is_vert = hit.isVertical()
            plane_key = (plane, is_vert)

            qdc = hit.GetSignal()
            hits_per_plane[plane_key] = hits_per_plane.get(plane_key, 0) + 1
            qdc_per_plane[plane_key] = qdc_per_plane.get(plane_key, 0.0) + qdc

    for hit in ds_hits:
        det_id = hit.GetDetectorID()
        plane = hit.GetPlane() if hasattr(hit, "GetPlane") else (det_id // 1000)
        is_vert = hit.isVertical()
        plane_key = (plane, is_vert)

        qdc = hit.GetSignal()
        qdcs.append(qdc)
        hit_w_densities.append(float(hits_per_plane[plane_key]))
        qdc_w_densities.append(float(qdc_per_plane[plane_key]))

    ds_nhits = len(qdcs)
    sum_qdc = sum(qdcs) if ds_nhits > 0 else 0.0
    max_qdc = max(qdcs) if ds_nhits > 0 else 0.0

    max_ds_nhits_per_plane = max(hits_per_plane.values()) if hits_per_plane else 0
    max_ds_qdc_per_plane = max(qdc_per_plane.values()) if qdc_per_plane else 0.0

    sum_hit_weight_density = sum(hit_w_densities) if hit_w_densities else 0.0
    max_hit_weight_density = max(hit_w_densities) if hit_w_densities else 0.0
    sum_qdc_weight_density = sum(qdc_w_densities) if qdc_w_densities else 0.0
    max_qdc_weight_density = max(qdc_w_densities) if qdc_w_densities else 0.0

    return {
        "ds_nhits": ds_nhits,
        "qdcs": qdcs,
        "hit_w_densities": hit_w_densities,
        "qdc_w_densities": qdc_w_densities,
        "sum_qdc": sum_qdc,
        "max_qdc": max_qdc,
        "max_ds_nhits_per_plane": max_ds_nhits_per_plane,
        "max_ds_qdc_per_plane": max_ds_qdc_per_plane,
        "sum_hit_weight_density": sum_hit_weight_density,
        "max_hit_weight_density": max_hit_weight_density,
        "sum_qdc_weight_density": sum_qdc_weight_density,
        "max_qdc_weight_density": max_qdc_weight_density,
    }

def run_hough_transform(
    muon_reco_task,
    event,
    geo,
    system: str = "scifi",
    z_vtx_min: Optional[float] = None,
    z_vtx_max: Optional[float] = None,
    max_lines: int = 3
) -> Tuple[int, Dict[str, List[Tuple[float, float]]], Dict[str, List[List[int]]]]:
    """
    Extracts detector hits and performs iterative Hough line finding with vertex constraints.
    Returns: (n_lines_max, track_lines, track_hit_indices)
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

    # 1. Collect SciFi hits
    if system in ["scifi", "both", "sfds", "all"]:
        scifi_hits = getattr(event, "Digi_ScifiHits", None)
        if scifi_hits:
            for hit in scifi_hits:
                if not hit.isValid():
                    continue
                geo.modules["Scifi"].GetSiPMPosition(hit.GetDetectorID(), pos_a, pos_b)
                for i in range(3):
                    hit_collection["pos"][i].append(pos_a[i])
                hit_collection["d"][0].append(muon_reco_task.Scifi_dx)
                hit_collection["d"][1].append(muon_reco_task.Scifi_dy)
                hit_collection["d"][2].append(muon_reco_task.Scifi_dz)
                hit_collection["vert"].append(hit.isVertical())
                hit_collection["system"].append(0)
                hit_collection["detectorID"].append(hit.GetDetectorID())
                hit_collection["qdc"].append(hit.GetSignal())

    # 2. Collect Downstream MuFilter hits
    if system in ["ds", "mufilter", "both", "sfds", "all"]:
        mufi_hits = getattr(event, "Digi_MuFilterHits", getattr(event, "Digi_MuFilterHit", None))
        if mufi_hits:
            for hit in mufi_hits:
                if not hit.isValid():
                    continue
                # System 3 is Downstream MuFilter
                if hit.GetSystem() != 3:
                    continue
                geo.modules["MuFilter"].GetPosition(hit.GetDetectorID(), pos_a, pos_b)
                for i in range(3):
                    hit_collection["pos"][i].append(pos_a[i])
                hit_collection["d"][0].append(muon_reco_task.MuFilter_ds_dx)
                hit_collection["d"][1].append(muon_reco_task.MuFilter_ds_dx if hit.GetSystem() == 3 else muon_reco_task.MuFilter_us_dy)
                hit_collection["d"][2].append(muon_reco_task.MuFilter_ds_dz)
                hit_collection["vert"].append(hit.isVertical())
                hit_collection["system"].append(hit.GetSystem())
                hit_collection["detectorID"].append(hit.GetDetectorID())
                hit_collection["qdc"].append(hit.GetSignal())

    if not hit_collection["pos"][0]:
        return 0, {"XZ": [], "YZ": []}, {"XZ": [], "YZ": []}

    for k in ["pos", "d"]:
        hit_collection[k] = np.array(hit_collection[k], dtype=np.float32)
    for k, dt in [("vert", np.bool_), ("system", np.int32), ("detectorID", np.int32), ("qdc", np.float32)]:
        hit_collection[k] = np.array(hit_collection[k], dtype=dt)

    counts = {"XZ": 0, "YZ": 0}
    lines = {"XZ": [], "YZ": []}
    track_hit_indices = {"XZ": [], "YZ": []}

    for projection_name in ["XZ", "YZ"]:
        is_vertical = (projection_name == "XZ")
        axis = 0 if is_vertical else 1
        hough_object = muon_reco_task.h_ZX if is_vertical else muon_reco_task.h_ZY
        hits_used = np.zeros(len(hit_collection["pos"][0]), dtype=np.bool_)

        valid_lines_found = 0
        attempts = 0
        max_attempts = 20

        limit_lines = getattr(muon_reco_task, "max_reco_muons", max_lines)

        while valid_lines_found < limit_lines and attempts < max_attempts:
            attempts += 1
            mask = np.logical_and(hit_collection["vert"] == is_vertical, ~hits_used)
            if not np.any(mask):
                break

            fit_result = hough_object.fit_randomize(
                np.dstack([hit_collection["pos"][2][mask], hit_collection["pos"][axis][mask]])[0],
                np.dstack([hit_collection["d"][2][mask], hit_collection["d"][axis][mask]])[0],
                muon_reco_task.n_random, False, False
            )

            if fit_result[0] in [-1, -999]:
                break

            new_slope, new_intercept = fit_result[0], fit_result[1]
            related_hits = SndlhcMuonReco.hit_finder(
                new_slope, new_intercept,
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
                skip_track = False
                conflict_params = None

                # Check vertex constraints with existing lines only if explicitly requested
                if z_vtx_min is not None or z_vtx_max is not None:
                    for existing_line in lines[projection_name]:
                        ext_m, ext_c = existing_line[0], existing_line[1]
                        if abs(new_slope - ext_m) > 1e-6:
                            z_vertex = (ext_c - new_intercept) / (new_slope - ext_m)
                            if (z_vtx_min is not None and z_vertex < z_vtx_min) or \
                               (z_vtx_max is not None and z_vertex > z_vtx_max):
                                skip_track = True
                                conflict_params = (ext_m, ext_c)
                                break

                    if skip_track:
                        if conflict_params:
                            global_indices = np.where(mask)[0][related_hits]
                            z_bad = hit_collection["pos"][2][global_indices]
                            c_bad = hit_collection["pos"][axis][global_indices]
                            dist = np.abs(c_bad - (conflict_params[0] * z_bad + conflict_params[1]))
                            hits_used[global_indices[np.argmin(dist)]] = True
                        else:
                            hits_used[np.where(mask)[0][related_hits]] = True
                        continue

                counts[projection_name] += 1
                lines[projection_name].append(fit_result)
                selected_global_idx = np.where(mask)[0][related_hits]
                track_hit_indices[projection_name].append(selected_global_idx.tolist())

                projection_idx = np.where(hit_collection["vert"] == is_vertical)[0]

                z_sel = hit_collection["pos"][2][selected_global_idx]
                c_sel = hit_collection["pos"][axis][selected_global_idx]
                z_all = hit_collection["pos"][2][projection_idx]
                c_all = hit_collection["pos"][axis][projection_idx]

                dz = z_all[:, np.newaxis] - z_sel[np.newaxis, :]
                dc = c_all[:, np.newaxis] - c_sel[np.newaxis, :]
                close_hits_mask = np.any((np.abs(dz) < 1e-3) & (np.abs(dc) < muon_reco_task.tolerance), axis=1)
                hits_used[projection_idx[close_hits_mask]] = True

                valid_lines_found += 1
            else:
                break

    n_lines_max = max(counts.values()) if counts else 0
    return n_lines_max, lines, track_hit_indices

def draw_simple_display(input_tree, geo, track_lines_sf, track_lines_ds, run_number, event_number):
    """Generates a 2D event display canvas with reconstructed SciFi and DS Hough lines overlaid."""
    c_name = f"c_Run{run_number}_{event_number}"
    c = ROOT.TCanvas(c_name, f"Event Display Run {run_number} Event {event_number}", 1200, 600)
    c.Divide(2, 1)

    # 1. XZ View
    c.cd(1)
    frame_xz = c.cd(1).DrawFrame(250.0, -85.0, 600.0, 15.0, f"Event {event_number} XZ Projection;Z [cm];X [cm]")
    line_objs = []
    # SciFi lines
    for m, b in track_lines_sf.get("XZ", []):
        z1, z2 = 260.0, 375.0
        x1, x2 = m * z1 + b, m * z2 + b
        l = ROOT.TLine(z1, x1, z2, x2)
        l.SetLineColor(ROOT.kCyan + 2)
        l.SetLineWidth(2)
        l.Draw("SAME")
        line_objs.append(l)
    # DS lines
    for m, b in track_lines_ds.get("XZ", []):
        z1, z2 = 375.0, 560.0
        x1, x2 = m * z1 + b, m * z2 + b
        l = ROOT.TLine(z1, x1, z2, x2)
        l.SetLineColor(ROOT.kGreen + 2)
        l.SetLineWidth(2)
        l.Draw("SAME")
        line_objs.append(l)

    # 2. YZ View
    c.cd(2)
    frame_yz = c.cd(2).DrawFrame(250.0, -15.0, 600.0, 85.0, f"Event {event_number} YZ Projection;Z [cm];Y [cm]")
    # SciFi lines
    for m, b in track_lines_sf.get("YZ", []):
        z1, z2 = 260.0, 375.0
        y1, y2 = m * z1 + b, m * z2 + b
        l = ROOT.TLine(z1, y1, z2, y2)
        l.SetLineColor(ROOT.kCyan + 2)
        l.SetLineWidth(2)
        l.Draw("SAME")
        line_objs.append(l)
    # DS lines
    for m, b in track_lines_ds.get("YZ", []):
        z1, z2 = 375.0, 560.0
        y1, y2 = m * z1 + b, m * z2 + b
        l = ROOT.TLine(z1, y1, z2, y2)
        l.SetLineColor(ROOT.kGreen + 2)
        l.SetLineWidth(2)
        l.Draw("SAME")
        line_objs.append(l)

    c.Update()
    return c, line_objs

def process_single_file(
    input_file_path: str,
    output_file_path: str,
    geo,
    muon_reco_task_sf,
    muon_reco_task_ds,
    args,
    gallery: Dict[str, set]
) -> Tuple[int, int]:
    """Processes a single input ROOT file and creates an augmented output ROOT file."""
    input_file = ROOT.TFile.Open(input_file_path, "READ")
    if not input_file or input_file.IsZombie():
        print(f"Error: Could not open input file '{input_file_path}'")
        return 0, 0

    tree_name = "rawConv" if input_file.Get("rawConv") else "cbmsim"
    input_tree = input_file.Get(tree_name)
    if not input_tree:
        print(f"Error: Neither 'rawConv' nor 'cbmsim' tree found in '{input_file_path}'")
        input_file.Close()
        return 0, 0

    total_entries = input_tree.GetEntries()

    out_dir = os.path.dirname(output_file_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    output_root_file = ROOT.TFile(output_file_path, "RECREATE")
    
    # Check if input tree has branches with missing dictionaries (e.g. Event_Type)
    has_unknown_branch = False
    for b in input_tree.GetListOfBranches():
        cname = b.GetClassName()
        if cname:
            cls = ROOT.gROOT.GetClass(cname)
            if not cls or not cls.HasDictionary():
                has_unknown_branch = True
                break

    if not has_unknown_branch:
        output_tree = input_tree.CloneTree(0)
    else:
        output_tree = ROOT.TTree(tree_name, tree_name)
        obj_holders = {}
        for b in input_tree.GetListOfBranches():
            bname = b.GetName()
            cname = b.GetClassName()
            if cname:
                cls = ROOT.gROOT.GetClass(cname)
                if not cls or not cls.HasDictionary():
                    continue
            if cname == "SNDLHCEventHeader":
                obj_holders[bname] = ROOT.SNDLHCEventHeader()
                input_tree.SetBranchAddress(bname, ROOT.AddressOf(obj_holders[bname]))
                output_tree.Branch(bname, obj_holders[bname])
            elif cname == "TClonesArray":
                elem_cls = "sndScifiHit" if "Scifi" in bname else "MuFilterHit"
                if "MCTrack" in bname:
                    elem_cls = "ShipMCTrack"
                obj_holders[bname] = ROOT.TClonesArray(elem_cls)
                input_tree.SetBranchAddress(bname, ROOT.AddressOf(obj_holders[bname]))
                output_tree.Branch(bname, obj_holders[bname])
            elif cname:
                try:
                    obj_holders[bname] = getattr(ROOT, cname)()
                    input_tree.SetBranchAddress(bname, ROOT.AddressOf(obj_holders[bname]))
                    output_tree.Branch(bname, obj_holders[bname])
                except Exception:
                    pass

    # Declare branch buffer dictionaries
    branches = {
        # Global line counts
        "n_lines": array("i", [0]),          # SciFi max lines (for backward compat)
        "n_lines_sf": array("i", [0]),       # SciFi max lines
        "n_lines_sf_xz": array("i", [0]),    # SciFi XZ lines
        "n_lines_sf_yz": array("i", [0]),    # SciFi YZ lines
        "n_lines_ds": array("i", [0]),       # DS max lines
        "n_lines_ds_xz": array("i", [0]),    # DS XZ lines
        "n_lines_ds_yz": array("i", [0]),    # DS YZ lines
        "n_lines_total": array("i", [0]),    # n_lines_sf + n_lines_ds

        # SciFi global metrics
        "scifi_nhits": array("i", [0]),
        "scifi_sum_qdc": array("f", [0.0]),
        "scifi_max_qdc": array("f", [0.0]),
        "sum_qdc": array("f", [0.0]),        # alias
        "max_qdc": array("f", [0.0]),        # alias
        "sum_hit_weight_density": array("f", [0.0]),
        "max_hit_weight_density": array("f", [0.0]),
        "sum_qdc_weight_density": array("f", [0.0]),
        "max_qdc_weight_density": array("f", [0.0]),
        "max_scifi_nhits_per_plane": array("i", [0]),
        "max_scifi_qdc_per_plane": array("f", [0.0]),

        # DS global metrics
        "ds_nhits": array("i", [0]),
        "ds_sum_qdc": array("f", [0.0]),
        "ds_max_qdc": array("f", [0.0]),
        "max_ds_nhits_per_plane": array("i", [0]),
        "max_ds_qdc_per_plane": array("f", [0.0]),
        "ds_sum_hit_weight_density": array("f", [0.0]),
        "ds_max_hit_weight_density": array("f", [0.0]),
        "ds_sum_qdc_weight_density": array("f", [0.0]),
        "ds_max_qdc_weight_density": array("f", [0.0]),
    }

    # SciFi and DS per-line parameters (1 to 3)
    for idx in range(1, 4):
        # 1. SciFi Hough line slopes and intercepts
        branches[f"xz_sf_m{idx}"] = array("f", [float("nan")])
        branches[f"xz_sf_c{idx}"] = array("f", [float("nan")])
        branches[f"yz_sf_m{idx}"] = array("f", [float("nan")])
        branches[f"yz_sf_c{idx}"] = array("f", [float("nan")])
        # Backward-compat aliases
        branches[f"xz_m{idx}"] = array("f", [float("nan")])
        branches[f"xz_c{idx}"] = array("f", [float("nan")])
        branches[f"yz_m{idx}"] = array("f", [float("nan")])
        branches[f"yz_c{idx}"] = array("f", [float("nan")])

        # SciFi per-line hit counts and metrics
        branches[f"xz_sf_n_hits_{idx}"] = array("i", [0])
        branches[f"yz_sf_n_hits_{idx}"] = array("i", [0])
        branches[f"xz_n_hits_{idx}"] = array("i", [0])
        branches[f"yz_n_hits_{idx}"] = array("i", [0])

        branches[f"xz_sf_max_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"xz_sf_sum_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"xz_sf_max_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_sf_max_qdc_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_sf_sum_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_sf_sum_qdc_weight_density_{idx}"] = array("f", [float("nan")])

        branches[f"yz_sf_max_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"yz_sf_sum_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"yz_sf_max_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_sf_max_qdc_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_sf_sum_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_sf_sum_qdc_weight_density_{idx}"] = array("f", [float("nan")])

        # Backward-compat aliases for SciFi metrics
        branches[f"xz_max_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"xz_sum_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"xz_max_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_max_qdc_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_sum_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_sum_qdc_weight_density_{idx}"] = array("f", [float("nan")])

        branches[f"yz_max_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"yz_sum_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"yz_max_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_max_qdc_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_sum_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_sum_qdc_weight_density_{idx}"] = array("f", [float("nan")])

        # 2. Downstream (DS) Hough line slopes and intercepts
        branches[f"xz_ds_m{idx}"] = array("f", [float("nan")])
        branches[f"xz_ds_c{idx}"] = array("f", [float("nan")])
        branches[f"yz_ds_m{idx}"] = array("f", [float("nan")])
        branches[f"yz_ds_c{idx}"] = array("f", [float("nan")])

        # DS per-line hit counts and metrics
        branches[f"xz_ds_n_hits_{idx}"] = array("i", [0])
        branches[f"yz_ds_n_hits_{idx}"] = array("i", [0])

        branches[f"xz_ds_max_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"xz_ds_sum_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"xz_ds_max_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_ds_max_qdc_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_ds_sum_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"xz_ds_sum_qdc_weight_density_{idx}"] = array("f", [float("nan")])

        branches[f"yz_ds_max_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"yz_ds_sum_qdc_{idx}"] = array("f", [float("nan")])
        branches[f"yz_ds_max_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_ds_max_qdc_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_ds_sum_hit_weight_density_{idx}"] = array("f", [float("nan")])
        branches[f"yz_ds_sum_qdc_weight_density_{idx}"] = array("f", [float("nan")])

    for bname, arr in branches.items():
        type_str = "I" if arr.typecode == "i" else "F"
        output_tree.Branch(bname, arr, f"{bname}/{type_str}")

    end_event = min(args.start_event + args.n_events, total_entries)
    events_processed = 0
    events_saved = 0
    display_objects = []

    z_min = args.z_vtx_range[0] if args.z_vtx_range else None
    z_max = args.z_vtx_range[1] if args.z_vtx_range else None

    run_sf = args.system in ["scifi", "both", "sfds", "all"]
    run_ds = args.system in ["ds", "mufilter", "both", "sfds", "all"]

    for entry_idx in range(args.start_event, end_event):
        events_processed += 1
        input_tree.GetEntry(entry_idx)
        curr_run = input_tree.EventHeader.GetRunId() if hasattr(input_tree, "EventHeader") and hasattr(input_tree.EventHeader, "GetRunId") else 1
        curr_event = input_tree.EventHeader.GetEventNumber() if hasattr(input_tree, "EventHeader") and hasattr(input_tree.EventHeader, "GetEventNumber") else entry_idx

        # Filter by gallery or random fraction
        if gallery:
            run_s = str(curr_run)
            if run_s not in gallery or curr_event not in gallery[run_s]:
                continue
        elif args.fraction < 1.0 and random.random() >= args.fraction:
            continue

        if hasattr(input_tree, "EventHeader") and input_tree.EventHeader.ClassName() == "SNDLHCEventHeader":
            geo.modules["Scifi"].InitEvent(input_tree.EventHeader)

        # 1. Run SciFi Hough reconstruction
        n_lines_sf = 0
        track_lines_sf = {"XZ": [], "YZ": []}
        track_hit_indices_sf = {"XZ": [], "YZ": []}

        if run_sf and muon_reco_task_sf:
            n_lines_sf, track_lines_sf, track_hit_indices_sf = run_hough_transform(
                muon_reco_task_sf, input_tree, geo,
                system="scifi",
                z_vtx_min=z_min,
                z_vtx_max=z_max,
                max_lines=3
            )

        # 2. Run Downstream (DS) Hough reconstruction
        n_lines_ds = 0
        track_lines_ds = {"XZ": [], "YZ": []}
        track_hit_indices_ds = {"XZ": [], "YZ": []}

        if run_ds and muon_reco_task_ds:
            n_lines_ds, track_lines_ds, track_hit_indices_ds = run_hough_transform(
                muon_reco_task_ds, input_tree, geo,
                system="ds",
                z_vtx_min=z_min,
                z_vtx_max=z_max,
                max_lines=3
            )

        scifi_hits = getattr(input_tree, "Digi_ScifiHits", [])
        mufi_hits = getattr(input_tree, "Digi_MuFilterHits", getattr(input_tree, "Digi_MuFilterHit", []))

        info_sf = get_event_scifi_info(scifi_hits)
        info_ds = get_event_ds_info(mufi_hits)

        # Fill global line counts
        branches["n_lines"][0] = n_lines_sf
        branches["n_lines_sf"][0] = n_lines_sf
        branches["n_lines_sf_xz"][0] = len(track_lines_sf.get("XZ", []))
        branches["n_lines_sf_yz"][0] = len(track_lines_sf.get("YZ", []))

        branches["n_lines_ds"][0] = n_lines_ds
        branches["n_lines_ds_xz"][0] = len(track_lines_ds.get("XZ", []))
        branches["n_lines_ds_yz"][0] = len(track_lines_ds.get("YZ", []))
        branches["n_lines_total"][0] = n_lines_sf + n_lines_ds

        # Fill SciFi global metrics
        branches["scifi_nhits"][0] = info_sf["sf_nhits"]
        branches["scifi_sum_qdc"][0] = info_sf["sum_qdc"]
        branches["scifi_max_qdc"][0] = info_sf["max_qdc"]
        branches["sum_qdc"][0] = info_sf["sum_qdc"]
        branches["max_qdc"][0] = info_sf["max_qdc"]
        branches["sum_hit_weight_density"][0] = info_sf["sum_hit_weight_density"]
        branches["max_hit_weight_density"][0] = info_sf["max_hit_weight_density"]
        branches["sum_qdc_weight_density"][0] = info_sf["sum_qdc_weight_density"]
        branches["max_qdc_weight_density"][0] = info_sf["max_qdc_weight_density"]
        branches["max_scifi_nhits_per_plane"][0] = info_sf["max_scifi_nhits_per_plane"]
        branches["max_scifi_qdc_per_plane"][0] = info_sf["max_scifi_qdc_per_plane"]

        # Fill DS global metrics
        branches["ds_nhits"][0] = info_ds["ds_nhits"]
        branches["ds_sum_qdc"][0] = info_ds["sum_qdc"]
        branches["ds_max_qdc"][0] = info_ds["max_qdc"]
        branches["max_ds_nhits_per_plane"][0] = info_ds["max_ds_nhits_per_plane"]
        branches["max_ds_qdc_per_plane"][0] = info_ds["max_ds_qdc_per_plane"]
        branches["ds_sum_hit_weight_density"][0] = info_ds["sum_hit_weight_density"]
        branches["ds_max_hit_weight_density"][0] = info_ds["max_hit_weight_density"]
        branches["ds_sum_qdc_weight_density"][0] = info_ds["sum_qdc_weight_density"]
        branches["ds_max_qdc_weight_density"][0] = info_ds["max_qdc_weight_density"]

        # Reset per-line arrays with NaN / 0 placeholders
        for idx in range(1, 4):
            # SciFi
            branches[f"xz_sf_m{idx}"][0] = float("nan")
            branches[f"xz_sf_c{idx}"][0] = float("nan")
            branches[f"yz_sf_m{idx}"][0] = float("nan")
            branches[f"yz_sf_c{idx}"][0] = float("nan")
            branches[f"xz_m{idx}"][0] = float("nan")
            branches[f"xz_c{idx}"][0] = float("nan")
            branches[f"yz_m{idx}"][0] = float("nan")
            branches[f"yz_c{idx}"][0] = float("nan")

            branches[f"xz_sf_n_hits_{idx}"][0] = 0
            branches[f"yz_sf_n_hits_{idx}"][0] = 0
            branches[f"xz_n_hits_{idx}"][0] = 0
            branches[f"yz_n_hits_{idx}"][0] = 0

            branches[f"xz_sf_max_qdc_{idx}"][0] = float("nan")
            branches[f"xz_sf_sum_qdc_{idx}"][0] = float("nan")
            branches[f"xz_sf_max_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_sf_max_qdc_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_sf_sum_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_sf_sum_qdc_weight_density_{idx}"][0] = float("nan")

            branches[f"yz_sf_max_qdc_{idx}"][0] = float("nan")
            branches[f"yz_sf_sum_qdc_{idx}"][0] = float("nan")
            branches[f"yz_sf_max_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_sf_max_qdc_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_sf_sum_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_sf_sum_qdc_weight_density_{idx}"][0] = float("nan")

            branches[f"xz_max_qdc_{idx}"][0] = float("nan")
            branches[f"xz_sum_qdc_{idx}"][0] = float("nan")
            branches[f"xz_max_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_max_qdc_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_sum_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_sum_qdc_weight_density_{idx}"][0] = float("nan")

            branches[f"yz_max_qdc_{idx}"][0] = float("nan")
            branches[f"yz_sum_qdc_{idx}"][0] = float("nan")
            branches[f"yz_max_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_max_qdc_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_sum_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_sum_qdc_weight_density_{idx}"][0] = float("nan")

            # DS
            branches[f"xz_ds_m{idx}"][0] = float("nan")
            branches[f"xz_ds_c{idx}"][0] = float("nan")
            branches[f"yz_ds_m{idx}"][0] = float("nan")
            branches[f"yz_ds_c{idx}"][0] = float("nan")

            branches[f"xz_ds_n_hits_{idx}"][0] = 0
            branches[f"yz_ds_n_hits_{idx}"][0] = 0

            branches[f"xz_ds_max_qdc_{idx}"][0] = float("nan")
            branches[f"xz_ds_sum_qdc_{idx}"][0] = float("nan")
            branches[f"xz_ds_max_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_ds_max_qdc_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_ds_sum_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"xz_ds_sum_qdc_weight_density_{idx}"][0] = float("nan")

            branches[f"yz_ds_max_qdc_{idx}"][0] = float("nan")
            branches[f"yz_ds_sum_qdc_{idx}"][0] = float("nan")
            branches[f"yz_ds_max_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_ds_max_qdc_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_ds_sum_hit_weight_density_{idx}"][0] = float("nan")
            branches[f"yz_ds_sum_qdc_weight_density_{idx}"][0] = float("nan")

        # 1. Populate SciFi reconstructed lines
        for proj in ["XZ", "YZ"]:
            p_lower = proj.lower()
            p_lines = track_lines_sf.get(proj, [])
            p_indices = track_hit_indices_sf.get(proj, [])

            for l_idx in range(3):
                idx_num = l_idx + 1
                if l_idx < len(p_lines):
                    slope, intercept = p_lines[l_idx]
                    branches[f"{p_lower}_sf_m{idx_num}"][0] = float(slope)
                    branches[f"{p_lower}_sf_c{idx_num}"][0] = float(intercept)
                    branches[f"{p_lower}_m{idx_num}"][0] = float(slope)
                    branches[f"{p_lower}_c{idx_num}"][0] = float(intercept)

                    # Compute line hit statistics
                    if l_idx < len(p_indices) and len(p_indices[l_idx]) > 0:
                        hit_idxs = p_indices[l_idx]
                        valid_idxs = [i for i in hit_idxs if i < len(info_sf["qdcs"])]
                        if valid_idxs:
                            l_qdcs = [info_sf["qdcs"][i] for i in valid_idxs]
                            l_hit_w = [info_sf["hit_w_densities"][i] for i in valid_idxs]
                            l_qdc_w = [info_sf["qdc_w_densities"][i] for i in valid_idxs]

                            branches[f"{p_lower}_sf_n_hits_{idx_num}"][0] = len(valid_idxs)
                            branches[f"{p_lower}_n_hits_{idx_num}"][0] = len(valid_idxs)
                            branches[f"{p_lower}_sf_max_qdc_{idx_num}"][0] = max(l_qdcs)
                            branches[f"{p_lower}_max_qdc_{idx_num}"][0] = max(l_qdcs)
                            branches[f"{p_lower}_sf_sum_qdc_{idx_num}"][0] = sum(l_qdcs)
                            branches[f"{p_lower}_sum_qdc_{idx_num}"][0] = sum(l_qdcs)
                            branches[f"{p_lower}_sf_max_hit_weight_density_{idx_num}"][0] = max(l_hit_w)
                            branches[f"{p_lower}_max_hit_weight_density_{idx_num}"][0] = max(l_hit_w)
                            branches[f"{p_lower}_sf_max_qdc_weight_density_{idx_num}"][0] = max(l_qdc_w)
                            branches[f"{p_lower}_max_qdc_weight_density_{idx_num}"][0] = max(l_qdc_w)
                            branches[f"{p_lower}_sf_sum_hit_weight_density_{idx_num}"][0] = sum(l_hit_w)
                            branches[f"{p_lower}_sum_hit_weight_density_{idx_num}"][0] = sum(l_hit_w)
                            branches[f"{p_lower}_sf_sum_qdc_weight_density_{idx_num}"][0] = sum(l_qdc_w)
                            branches[f"{p_lower}_sum_qdc_weight_density_{idx_num}"][0] = sum(l_qdc_w)

        # 2. Populate Downstream (DS) reconstructed lines
        for proj in ["XZ", "YZ"]:
            p_lower = proj.lower()
            p_lines = track_lines_ds.get(proj, [])
            p_indices = track_hit_indices_ds.get(proj, [])

            for l_idx in range(3):
                idx_num = l_idx + 1
                if l_idx < len(p_lines):
                    slope, intercept = p_lines[l_idx]
                    branches[f"{p_lower}_ds_m{idx_num}"][0] = float(slope)
                    branches[f"{p_lower}_ds_c{idx_num}"][0] = float(intercept)

                    # Compute line hit statistics
                    if l_idx < len(p_indices) and len(p_indices[l_idx]) > 0:
                        hit_idxs = p_indices[l_idx]
                        valid_idxs = [i for i in hit_idxs if i < len(info_ds["qdcs"])]
                        if valid_idxs:
                            l_qdcs = [info_ds["qdcs"][i] for i in valid_idxs]
                            l_hit_w = [info_ds["hit_w_densities"][i] for i in valid_idxs]
                            l_qdc_w = [info_ds["qdc_w_densities"][i] for i in valid_idxs]

                            branches[f"{p_lower}_ds_n_hits_{idx_num}"][0] = len(valid_idxs)
                            branches[f"{p_lower}_ds_max_qdc_{idx_num}"][0] = max(l_qdcs)
                            branches[f"{p_lower}_ds_sum_qdc_{idx_num}"][0] = sum(l_qdcs)
                            branches[f"{p_lower}_ds_max_hit_weight_density_{idx_num}"][0] = max(l_hit_w)
                            branches[f"{p_lower}_ds_max_qdc_weight_density_{idx_num}"][0] = max(l_qdc_w)
                            branches[f"{p_lower}_ds_sum_hit_weight_density_{idx_num}"][0] = sum(l_hit_w)
                            branches[f"{p_lower}_ds_sum_qdc_weight_density_{idx_num}"][0] = sum(l_qdc_w)

        output_tree.Fill()
        events_saved += 1

        if args.save_displays and (n_lines_sf >= 2 or n_lines_ds >= 2):
            output_root_file.cd()
            canvas, line_objs = draw_simple_display(input_tree, geo, track_lines_sf, track_lines_ds, curr_run, curr_event)
            canvas.Write()
            display_objects.append((canvas, line_objs))

    # Preserve metadata keys from input ROOT file
    metadata_keys = []
    for k in input_file.GetListOfKeys():
        kname = k.GetName()
        kclass = k.GetClassName()
        if kclass != "TTree" and kname != tree_name:
            obj = input_file.Get(kname)
            if obj:
                metadata_keys.append((kname, obj))

    output_root_file.cd()
    output_tree.Write(tree_name, ROOT.TObject.kOverwrite)

    if metadata_keys:
        for kname, obj in metadata_keys:
            obj.Write(kname, ROOT.TObject.kSingleKey | ROOT.TObject.kOverwrite)

    output_root_file.Close()
    input_file.Close()

    return events_processed, events_saved

def main():
    sndsw_path = os.environ.get("SNDSW_ROOT", "")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(repo_root, "TrackingParams_sf4.xml"),
        os.path.join(repo_root, "parFiles", "TrackingParams.xml"),
        os.path.join(sndsw_path, "TrackingParams_sf4.xml") if sndsw_path else "",
        os.path.join(sndsw_path, "python", "TrackingParams_V2_28January2023.xml") if sndsw_path else "",
        "TrackingParams_sf4.xml", "TrackingParams.xml"
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            default_par_file = candidate
            break
    else:
        default_par_file = os.path.join(repo_root, "TrackingParams_sf4.xml")

    parser = argparse.ArgumentParser(
        description="Reconstruct up to 3 Hough lines per projection (XZ/YZ) for SciFi and DS MuFilter on SND@LHC data/MC and store all parameters in the output ROOT file."
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_pattern",
        required=True,
        help="Input ROOT file path or wildcard pattern (e.g. '/path/to/files_*.root')"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_pattern",
        default=None,
        help="Output ROOT file path or pattern (e.g. '/path/to/hough_%s.root' or 'out.root')"
    )
    parser.add_argument(
        "-g", "--geoFile", "--geofile",
        dest="geofile_path",
        default=None,
        help="Path to geometry ROOT file. If omitted, automatically determined from run number or MC path."
    )
    parser.add_argument(
        "-p", "--parFile", "--par-file",
        dest="par_file",
        default=default_par_file,
        help="Path to TrackingParams.xml (default: %(default)s)"
    )
    parser.add_argument(
        "--system",
        dest="system",
        choices=["both", "scifi", "ds"],
        default="both",
        help="Detector system to reconstruct Hough lines for: 'both' (default, reconstructs SciFi and DS), 'scifi', or 'ds'"
    )
    parser.add_argument(
        "--tracking-case-sf",
        dest="tracking_case_sf",
        default="muon_trident_Sf",
        help="Tracking case name in TrackingParams.xml for SciFi (default: %(default)s)"
    )
    parser.add_argument(
        "--tracking-case-ds",
        dest="tracking_case_ds",
        default="muon_trident_DS",
        help="Tracking case name in TrackingParams.xml for DS MuFilter (default: %(default)s)"
    )
    parser.add_argument(
        "--z-vtx-range",
        dest="z_vtx_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("Z_MIN", "Z_MAX"),
        help="Optional vertex Z constraint range [Z_MIN, Z_MAX] in cm (default: None = unconstrained)"
    )
    parser.add_argument(
        "-s", "--start-event", "--start",
        dest="start_event",
        type=int,
        default=0,
        help="Starting event index (default: 0)"
    )
    parser.add_argument(
        "-n", "--n-events",
        dest="n_events",
        type=int,
        default=10000000,
        help="Number of events to process per file (default: 10000000)"
    )
    parser.add_argument(
        "--fraction",
        dest="fraction",
        type=float,
        default=1.0,
        help="Fraction of events to randomly process (default: 1.0)"
    )
    parser.add_argument(
        "--gallery",
        dest="gallery_file",
        default=None,
        help="Optional JSON gallery file containing specific {run: [events]} to process"
    )
    parser.add_argument(
        "--save-displays",
        dest="save_displays",
        action="store_true",
        default=False,
        help="If set, writes 2D event display canvases into output ROOT file for events with >= 2 lines (default: False)"
    )

    args = parser.parse_args()

    # Find matching input files
    matched_files = sorted(glob.glob(args.input_pattern))
    if not matched_files:
        if os.path.exists(args.input_pattern):
            matched_files = [args.input_pattern]
        else:
            print(f"Error: No files found matching pattern: {args.input_pattern}")
            sys.exit(1)

    print("=" * 65)
    print("SND@LHC Hough Line Finding & Tracking Processor (SciFi + DS)")
    print("=" * 65)
    print(f"Input Pattern     : {args.input_pattern}")
    print(f"Files Found       : {len(matched_files):,}")
    print(f"Detector System   : {args.system}")
    print(f"Tracking Params   : {args.par_file}")
    print(f"SciFi Tracking    : {args.tracking_case_sf}")
    print(f"DS Tracking       : {args.tracking_case_ds}")
    if args.z_vtx_range:
        print(f"Vertex Z Range    : [{args.z_vtx_range[0]:.1f}, {args.z_vtx_range[1]:.1f}] cm")
    print(f"Max Events / File : {args.n_events:,}")
    print("=" * 65)

    # Resolve geometry file
    first_file = matched_files[0]
    geofile_path = args.geofile_path
    if not geofile_path:
        is_real_data = False
        r_num = 1
        try:
            t_first = ROOT.TFile.Open(first_file, "READ")
            if t_first and not t_first.IsZombie():
                if t_first.Get("rawConv"):
                    is_real_data = True
                t_in = t_first.Get("rawConv") or t_first.Get("cbmsim")
                if t_in and t_in.GetEntries() > 0:
                    t_in.GetEntry(0)
                    if hasattr(t_in, "EventHeader") and hasattr(t_in.EventHeader, "GetRunId"):
                        r_num = t_in.EventHeader.GetRunId()
                        is_real_data = True
                t_first.Close()
        except Exception:
            pass

        if not is_real_data and ("run_" in first_file or "rawConv" in first_file):
            import re
            m = re.search(r'run_0*(\d+)', first_file)
            if m:
                r_num = int(m.group(1))
                is_real_data = True

        if is_real_data:
            try:
                geofile_path = ROOT.snd.analysis_tools.GetGeoPath(int(r_num))
            except Exception:
                geofile_path = None

        if not geofile_path:
            input_dir = os.path.dirname(os.path.abspath(first_file))
            boost_suffix = "boost1000" if "boost1000" in first_file else "boost100"
            candidates = [
                os.path.join(input_dir, f"geofile_full.Ntuple-TGeant4_{boost_suffix}.0.root"),
                os.path.join(input_dir, "geofile_full.Ntuple-TGeant4_boost100.0.root"),
                "python/geofile_full.Ntuple-TGeant4_boost100.0.root",
                os.path.join(sndsw_path, "python", "geofile_full.Ntuple-TGeant4_boost100.0.root") if sndsw_path else None,
                "/eos/experiment/sndlhc/MonteCarlo/ThreeMuons/geofile_full.Ntuple-TGeant4_boost100.0.root"
            ]
            for cand in candidates:
                if cand and os.path.exists(cand):
                    geofile_path = cand
                    break

    geo_valid = False
    if geofile_path:
        if geofile_path.startswith("root://") or os.path.exists(geofile_path):
            geo_valid = True

    if not geo_valid:
        print(f"Error: Could not locate geometry file (tried: {geofile_path})")
        sys.exit(1)

    print(f"Loading Geometry from: {geofile_path}")
    geo = SndlhcGeo.GeoInterface(geofile_path)

    # Initialize FairRunAna environment using a file with valid entries
    init_file = None
    for fpath in matched_files:
        try:
            f_chk = ROOT.TFile.Open(fpath, "READ")
            if f_chk and not f_chk.IsZombie():
                t_chk = f_chk.Get("rawConv") or f_chk.Get("cbmsim")
                if t_chk and t_chk.GetEntries() > 0:
                    init_file = fpath
                    f_chk.Close()
                    break
                f_chk.Close()
        except Exception:
            pass

    if not init_file:
        init_file = first_file

    init_f_obj = ROOT.TFile.Open(init_file, "READ")
    t_name = "rawConv" if init_f_obj.Get("rawConv") else "cbmsim"
    init_tree = init_f_obj.Get(t_name)
    init_run = 1
    if init_tree and init_tree.GetEntries() > 0:
        init_tree.GetEntry(0)
        if hasattr(init_tree, "EventHeader") and hasattr(init_tree.EventHeader, "GetRunId"):
            init_run = init_tree.EventHeader.GetRunId()
    else:
        import re
        m = re.search(r'run_0*(\d+)', init_file)
        if m:
            init_run = int(m.group(1))

    fair_run = ROOT.FairRunAna()
    io_manager = ROOT.FairRootManager.Instance()
    io_manager.SetTreeName(t_name)
    fair_run.SetSource(ROOT.FairFileSource(init_f_obj))
    fair_run.SetSink(ROOT.FairRootFileSink(ROOT.TMemFile("dummy", "CREATE")))

    rtdb = fair_run.GetRuntimeDb()
    if os.path.exists(args.par_file):
        par_source = ROOT.FairParAsciiFileIo()
        if par_source.open(args.par_file, "in"):
            rtdb.setFirstInput(par_source)
            rtdb.addRun(int(init_run))
            rtdb.setOutput(par_source)

    # Configure SciFi MuonReco task
    muon_reco_task_sf = None
    if args.system in ["scifi", "both", "sfds", "all"]:
        muon_reco_task_sf = SndlhcMuonReco.MuonReco()
        muon_reco_task_sf.SetParFile(args.par_file)
        fair_run.AddTask(muon_reco_task_sf)
        muon_reco_task_sf.SetHoughSpaceFormat("linearSlopeIntercept")
        muon_reco_task_sf.SetTrackingCase(args.tracking_case_sf)

    # Configure DS MuonReco task
    muon_reco_task_ds = None
    if args.system in ["ds", "mufilter", "both", "sfds", "all"]:
        muon_reco_task_ds = SndlhcMuonReco.MuonReco()
        muon_reco_task_ds.SetParFile(args.par_file)
        fair_run.AddTask(muon_reco_task_ds)
        muon_reco_task_ds.SetHoughSpaceFormat("linearSlopeIntercept")
        muon_reco_task_ds.SetTrackingCase(args.tracking_case_ds)

    fair_run.Init()

    # Load gallery if specified
    gallery = {}
    if args.gallery_file:
        try:
            with open(args.gallery_file, "r") as f:
                raw_g = json.load(f)
                for r in raw_g:
                    gallery[str(r)] = set(raw_g[r])
            print(f"Loaded {sum(len(v) for v in gallery.values())} events from gallery file.")
        except Exception as e:
            print(f"Error loading gallery file '{args.gallery_file}': {e}")
            sys.exit(1)

    t0_overall = time.time()
    grand_total_scanned = 0
    grand_total_stored = 0
    created_files = []

    for f_idx, in_path in enumerate(matched_files):
        in_basename = os.path.splitext(os.path.basename(in_path))[0]
        if args.output_pattern:
            if "%s" in args.output_pattern:
                out_path = args.output_pattern % in_basename
            elif len(matched_files) == 1:
                out_path = args.output_pattern
            else:
                out_dir = os.path.dirname(args.output_pattern)
                out_path = os.path.join(out_dir, f"{in_basename}_hough.root")
        else:
            in_dir = os.path.dirname(in_path)
            out_path = os.path.join(in_dir, f"{in_basename}_hough.root")

        print(f"\n[{f_idx + 1}/{len(matched_files)}] Processing: '{in_path}' -> '{out_path}'")
        n_scanned, n_stored = process_single_file(
            in_path, out_path, geo, muon_reco_task_sf, muon_reco_task_ds, args, gallery
        )
        grand_total_scanned += n_scanned
        grand_total_stored += n_stored
        created_files.append(out_path)
        print(f"  -> Done: Scanned {n_scanned:,} events | Stored {n_stored:,} events")

    total_elapsed = time.time() - t0_overall
    print("\n" + "=" * 65)
    print("HOUGH LINE FINDING BATCH COMPLETE (SciFi + DS)")
    print("=" * 65)
    print(f"Total Files Processed : {len(matched_files):,}")
    print(f"Total Files Created   : {len(created_files):,}")
    print(f"Total Events Scanned  : {grand_total_scanned:,}")
    print(f"Total Events Stored   : {grand_total_stored:,}")
    print(f"Total Elapsed Time    : {total_elapsed:.1f} s ({grand_total_scanned/total_elapsed:.0f} ev/s)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
