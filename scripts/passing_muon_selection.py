#!/usr/bin/env python3
"""
================================================================================
SND@LHC Passing Muon Selection & Truth Assessment Engine
================================================================================
Assesses the efficiency, background rejection, and sample purity of the clean
single-muon calibration selection criteria on Geant4 Monte Carlo:

Selection Criteria:
  1. SciFi Track Multiplicity : Exactly 1 Reconstructed SciFi Track
  2. SciFi Track chi2/ndf    : <= 10.0 (Kalman fit quality and flag)
  3. Max Angular Slope        : <= 0.05 rad (~3 deg near-normal incidence)
  4. DS Track Slope Match     : |Delta slope| <= 0.04 rad
  5. Fiducial Margin          : >= 1.5 cm from active plane borders
  6. SciFi Total Hits         : [10, 35] (Clean MIP band)
  7. DS System Hits           : >= 2 (Downstream penetration requirement)

Weights:
  Always uses FLUKA importance-sampling generator weights:
    double weight = ((ShipMCTrack*)MCTrack->At(0))->GetWeight();

Author: SND@LHC Collaboration
================================================================================
"""

import os
import sys
import time
import math
import argparse
import tempfile
import shutil
import glob
import re
from typing import Dict, Any, List, Tuple, Optional


import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning


# Add project root to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == "scripts" else _script_dir
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import DataManager, load_trident_libraries

# Load compiled C++ analysis libraries (libtrident_analysis.so)
load_trident_libraries(_repo_root)

# PMU Simulation Normalization Constants:
# 100e6 collisions/s with inelastic cross section sigma = 80 mb:
# L_MC = 100e6 / 80 mb = 1.25e6 s^-1 mb^-1 = 1 / (8e5) [s^-1 fb^-1]
L_MC: float = 1.0 / 800000.0           # [s^-1 fb^-1]
PMU_LUMI_SCALE_COEFF: float = 800000.0  # 1 / L_MC = 8e5 [s fb]

CATEGORY_NAMES = [
    "NoDetectorHit",
    "Clean_Passing_Muon",
    "Catastrophic_EM_Shower",
    "Hadronic_Shower",
    "MultiMuon",
    "Halo_Shower_NoMuon",
    "Stopping_Scattered_Muon"
]

CATEGORY_LABELS = {
    0: "No Detector Hit",
    1: "Clean Passing Muon (Signal)",
    2: "Catastrophic EM Shower",
    3: "Hadronic Shower",
    4: "Multi-Muon Bundle",
    5: "Halo / Shower Only (No Muon)",
    6: "Stopping / Scattered Muon"
}

CUTFLOW_STYLES = {
    "total": {
        "key": "Total_Active",
        "legend_label": "Total Active",
        "title": "Total Detector Active",
        "color": ROOT.kBlack,
        "line_style": 2,      # dashed
        "line_width": 2,
        "marker_style": 29,   # filled star
        "marker_size": 1.4,
    },
    1: {
        "key": "Clean_Passing_Muon",
        "legend_label": "Signal (Clean #mu)",
        "title": "Clean Passing Muon (Signal)",
        "color": ROOT.kAzure + 2,
        "line_style": 1,      # solid
        "line_width": 4,
        "marker_style": 20,   # filled circle
        "marker_size": 1.2,
    },
    2: {
        "key": "Catastrophic_EM_Shower",
        "legend_label": "EM Shower",
        "title": "Catastrophic EM Shower",
        "color": ROOT.kRed + 1,
        "line_style": 1,      # solid
        "line_width": 2,
        "marker_style": 21,   # filled square
        "marker_size": 1.0,
    },
    3: {
        "key": "Hadronic_Shower",
        "legend_label": "Hadronic Shower",
        "title": "Hadronic Shower",
        "color": ROOT.kOrange + 7,
        "line_style": 7,      # dashed
        "line_width": 2,
        "marker_style": 22,   # filled triangle up
        "marker_size": 1.1,
    },
    4: {
        "key": "MultiMuon",
        "legend_label": "Multi-Muon",
        "title": "Multi-Muon Bundle",
        "color": ROOT.kMagenta + 2,
        "line_style": 3,      # dotted
        "line_width": 2,
        "marker_style": 23,   # filled triangle down
        "marker_size": 1.1,
    },
    5: {
        "key": "Halo_Shower_NoMuon",
        "legend_label": "Halo / No Muon",
        "title": "Halo / Shower Only (No Muon)",
        "color": ROOT.kGreen + 2,
        "line_style": 4,      # dash-dot
        "line_width": 2,
        "marker_style": 24,   # open circle
        "marker_size": 1.0,
    },
    6: {
        "key": "Stopping_Scattered_Muon",
        "legend_label": "Stopping / Scatter",
        "title": "Stopping / Scattered Muon",
        "color": ROOT.kGray + 2,
        "line_style": 5,      # long dash-dot
        "line_width": 2,
        "marker_style": 25,   # open square
        "marker_size": 1.0,
    },
    "data": {
        "key": "Collision_Data",
        "legend_label": "Collision Data",
        "title": "Collision Data",
        "color": ROOT.kBlack,
        "line_style": 1,
        "line_width": 2,
        "marker_style": 20,   # filled circle
        "marker_size": 1.2,
    },
}


def extract_duration_and_lumi(
    dm_data: Any,
    dur_override: Optional[float] = None,
    lumi_override: Optional[float] = None
) -> Tuple[float, float, str, str]:
    """
    Extracts duration (seconds) and integrated luminosity (fb^-1) from DataManager,
    supporting short single chunks (via atlas_lumi tree interpolation) as well as full runs/fills.
    """
    # 1. Duration (fast boundary estimation first)
    if dur_override is not None and dur_override > 0:
        duration = float(dur_override)
    else:
        dur = dm_data.get_duration_fast(cached=True)
        if dur is None or dur <= 0:
            dur = dm_data.get_duration(cached=True)
        if dur is None or dur <= 0:
            dur = dm_data.duration
        duration = float(dur) if (dur is not None and dur > 0) else 0.0

    if duration < 60:
        dur_str = f"{duration:.1f} s"
    elif duration < 3600:
        dur_str = f"{duration / 60.0:.1f} min"
    else:
        dur_str = f"{duration / 3600.0:.2f} h"

    # 2. Luminosity
    if lumi_override is not None and lumi_override > 0:
        lumi = float(lumi_override)
    else:
        lumi = dm_data.get_lumi_fast()
        if lumi <= 0.0:
            lumi = dm_data.get_lumi_exact()
        if lumi <= 0.0 and duration > 0:
            # Check if we can interpolate instantaneous luminosity from atlas_lumi
            tr = dm_data.get_time_range(cached=True)
            fn = dm_data.fill_number
            if tr and tr[0] > 0 and tr[1] > 0 and fn and fn > 0:
                lumi_file = f"/eos/experiment/sndlhc/atlas_lumi/fill_{fn:06d}.root"
                if os.path.exists(lumi_file):
                    try:
                        f = ROOT.TFile.Open(lumi_file, "READ")
                        t = f.Get("atlas_lumi") if f and not f.IsZombie() else None
                        if t:
                            t_mid = 0.5 * (tr[0] + tr[1])
                            best_dt = 1e9
                            inst_lumi = 0.0
                            for i in range(t.GetEntries()):
                                t.GetEntry(i)
                                dt = abs(t.unix_timestamp - t_mid)
                                if dt < best_dt:
                                    best_dt = dt
                                    inst_lumi = t.var
                            f.Close()
                            if best_dt < 600.0:
                                # inst_lumi is in ub^-1 s^-1. 1 ub^-1 = 1e-9 fb^-1
                                lumi = duration * (inst_lumi * 1e-9)
                    except Exception:
                        pass
        if lumi <= 0.0:
            lumi = dm_data.lumi if dm_data.lumi > 0 else 0.0

    if lumi >= 1e-2:
        lumi_str = f"{lumi:.4f} fb^{{-1}}"
    elif lumi > 0:
        lumi_str = f"{lumi:.3e} fb^{{-1}}"
    else:
        lumi_str = "0 fb^{-1}"

    return duration, lumi, dur_str, lumi_str


def build_cutflow_histograms(
    cutflow_stages: List[Tuple[str, str]],
    stage_labels: List[str],
    cutflow_matrix: Dict[str, Dict[int, float]],
    cutflow_matrix_err: Dict[str, Dict[int, float]],
    is_weighted: bool = True,
    y_title: Optional[str] = None,
    name_prefix: str = ""
) -> Dict[Any, ROOT.TH1D]:
    """
    Constructs 1D cutflow histograms for each event category and for Total Active.
    """
    hists = {}
    mode = "weighted" if is_weighted else "raw"
    if y_title:
        y_unit = y_title
    else:
        y_unit = "FLUKA Weighted Yield / 10^{8} p-p" if is_weighted else "Unweighted Simulated Events"
    n_stages = len(cutflow_stages)

    keys = ["total", 1, 2, 3, 4, 5, 6]
    for k in keys:
        cfg = CUTFLOW_STYLES[k]
        h_name = f"h_{name_prefix}cutflow_{mode}_{cfg['key']}"
        h_title = f"{cfg['title']} Sequential Cutflow ({mode.capitalize()});;{y_unit}"
        h = ROOT.TH1D(h_name, h_title, n_stages, 0.5, n_stages + 0.5)
        ROOT.SetOwnership(h, False)
        h.SetStats(0)

        for s_idx, (stage_name, _) in enumerate(cutflow_stages, start=1):
            h.GetXaxis().SetBinLabel(s_idx, stage_labels[s_idx - 1])
            if k == "total":
                val = sum(cutflow_matrix[stage_name].get(c, 0.0) for c in range(1, 7))
                err = math.sqrt(sum(cutflow_matrix_err[stage_name].get(c, 0.0)**2 for c in range(1, 7)))
            else:
                val = cutflow_matrix[stage_name].get(k, 0.0)
                err = cutflow_matrix_err[stage_name].get(k, 0.0)

            h.SetBinContent(s_idx, val)
            h.SetBinError(s_idx, err)

        h.SetLineColor(cfg["color"])
        h.SetLineStyle(cfg["line_style"])
        h.SetLineWidth(cfg["line_width"])
        h.SetMarkerColor(cfg["color"])
        h.SetMarkerStyle(cfg["marker_style"])
        h.SetMarkerSize(cfg["marker_size"])
        hists[k] = h

    return hists


def scale_data_th1(h: Optional[ROOT.TH1], scale_factor: float) -> Optional[ROOT.TH1]:
    """
    Scales a collision data histogram by scale_factor (e.g. if reconstruction ran on a 1/N subsample).
    Ensures ROOT Sumw2 is initialized so bin errors scale properly as S * sqrt(N_raw).
    """
    if h is None or abs(scale_factor - 1.0) < 1e-9:
        return h
    if not h.GetSumw2N():
        h.Sumw2()
    h.Scale(scale_factor)
    return h


def build_superimposed_cutflow_canvas(
    hists_dict: Dict[Any, ROOT.TH1D],
    last_stage_name: str,
    cutflow_matrix: Dict[str, Dict[int, float]],
    is_weighted: bool = True,
    canvas_name: str = "c_cutflow_superimposed_weighted",
    h_data: Optional[ROOT.TH1D] = None,
    data_final_count: Optional[float] = None,
    header_right: Optional[str] = None,
    header_left: Optional[str] = "#bf{SND@LHC} #font[52]{Internal}",
    legend_header: Optional[str] = None,
    mc_total_label: str = "PMU MC Total"
) -> ROOT.TCanvas:
    """
    Renders publication-quality superimposed cutflow TCanvas with log-y scale,
    two-column legend, and standard experiment header. If collision data is provided,
    plots real data alongside the scaled simulation.
    """
    has_data = h_data is not None
    mode_str = "FLUKA Weighted" if is_weighted else "Raw Counts"
    if has_data:
        mode_str += " & Collision Data"

    c = ROOT.TCanvas(canvas_name, f"Superimposed Cutflow ({mode_str})", 1150, 750)
    ROOT.SetOwnership(c, False)

    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.04)
    c.SetTopMargin(0.085)
    c.SetBottomMargin(0.14)
    c.SetLogy(True)
    c.SetGrid(1, 1)

    y_unit = "Events" if has_data else ("FLUKA Weighted Yield / 10^{8} p-p" if is_weighted else "Unweighted Simulated Events")

    # Dynamic log-scale boundaries
    all_pos_vals = []
    for h in hists_dict.values():
        for b in range(1, h.GetNbinsX() + 1):
            val = h.GetBinContent(b)
            if val > 0:
                all_pos_vals.append(val)
    if h_data:
        for b in range(1, h_data.GetNbinsX() + 1):
            val = h_data.GetBinContent(b)
            if val > 0:
                all_pos_vals.append(val)

    min_pos = min(all_pos_vals) if all_pos_vals else (1e-3 if is_weighted else 0.5)
    max_pos = max(all_pos_vals) if all_pos_vals else 100.0

    log_min = math.floor(math.log10(min_pos))
    log_max = math.ceil(math.log10(max_pos))
    ymin = max(1e-7, 10.0 ** (log_min - 0.5))
    ymax = 10.0 ** (log_max + 2.5)

    # Clean axis frame
    frame = hists_dict["total"].Clone(f"frame_{canvas_name}")
    ROOT.SetOwnership(frame, False)
    frame.Reset()
    frame.SetTitle(f";;{y_unit}")
    frame.GetYaxis().SetRangeUser(ymin, ymax)
    frame.GetXaxis().LabelsOption("h")
    frame.GetXaxis().SetLabelSize(0.033)
    frame.GetXaxis().SetLabelOffset(0.015)
    frame.GetYaxis().SetTitleSize(0.044)
    frame.GetYaxis().SetLabelSize(0.038)
    frame.GetYaxis().SetTitleOffset(1.25)
    frame.Draw("AXIS")

    # Draw order: reference curves first, signal on top
    draw_order = ["total", 5, 6, 4, 3, 2, 1]
    for k in draw_order:
        h = hists_dict[k]
        h.Draw("HIST SAME")
        h.Draw("P SAME")

    # Draw Collision Data on top if present
    if has_data:
        h_data.SetLineColor(ROOT.kBlack)
        h_data.SetLineWidth(2)
        h_data.SetMarkerStyle(20)
        h_data.SetMarkerSize(1.2)
        h_data.SetMarkerColor(ROOT.kBlack)
        h_data.Draw("E1 P SAME")

    # Two-column Legend
    leg_y1 = 0.71 if legend_header else 0.73
    leg = ROOT.TLegend(0.36, leg_y1, 0.95, 0.91)
    ROOT.SetOwnership(leg, False)
    leg.SetNColumns(2)
    leg.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg.SetBorderSize(1)
    leg.SetLineColor(ROOT.kGray + 1)
    leg.SetTextFont(42)
    leg.SetTextSize(0.026)

    if legend_header:
        leg.SetHeader(legend_header)

    if has_data:
        # Row 1: Collision Data | MC Total
        data_cnt_str = f"{int(round(data_final_count)):,}" if data_final_count is not None else ""
        data_lbl = f"Collision Data ({data_cnt_str})" if data_final_count is not None else "Collision Data"
        leg.AddEntry(h_data, data_lbl, "EP")
        fin_tot = cutflow_matrix[last_stage_name].get("total", sum(cutflow_matrix[last_stage_name].get(c, 0.0) for c in range(1, 7)))
        fin_tot_str = fmt_w(fin_tot) if is_weighted else f"{int(fin_tot):,}"
        leg.AddEntry(hists_dict["total"], f"{mc_total_label} ({fin_tot_str})", "LP")

        # Row 2-4: Signal | EM, Hadronic | MultiMu, Halo | Stopping
        cat_order = [1, 2, 3, 4, 5, 6]
        for k in cat_order:
            cfg = CUTFLOW_STYLES[k]
            fin = cutflow_matrix[last_stage_name].get(k, 0.0)
            base_lbl = cfg["legend_label"]
            if is_weighted:
                if fin == 0:
                    val_str = "0.0"
                elif fin >= 0.01:
                    val_str = f"{fin:.2f}"
                else:
                    val_str = f"{fin:.1e}"
            else:
                val_str = f"{int(fin):,}"
            leg.AddEntry(hists_dict[k], f"{base_lbl} ({val_str})", "LP")
    else:
        # Paired order for 2 columns without data
        legend_order = ["total", 1, 2, 3, 4, 5, 6]
        for k in legend_order:
            cfg = CUTFLOW_STYLES[k]
            fin = cutflow_matrix[last_stage_name].get(k, 0.0) if k != "total" else sum(cutflow_matrix[last_stage_name].get(c, 0.0) for c in range(1, 7))
            base_lbl = mc_total_label if k == "total" else cfg["legend_label"]
            if is_weighted:
                if fin == 0:
                    val_str = "0.0"
                elif fin >= 0.01:
                    val_str = f"{fin:.2f}"
                else:
                    val_str = f"{fin:.1e}"
            else:
                val_str = f"{int(fin):,}"
            leg.AddEntry(hists_dict[k], f"{base_lbl} ({val_str})", "LP")

    leg.Draw()

    # Experiment header labels (non-overlapping)
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(62)
    latex.SetTextSize(0.038)
    if header_left:
        latex.DrawLatex(0.12, 0.938, header_left)

    latex.SetTextFont(42)
    latex.SetTextSize(0.028 if has_data else 0.030)
    latex.SetTextAlign(31)
    hdr_right = header_right if header_right else "160 #murad, 100#times10^{6} p-p, FLUKA E_{cut} = 10 GeV"
    latex.DrawLatex(0.96, 0.938, hdr_right)

    ROOT.gPad.RedrawAxis()
    return c


def build_trimuon_cutflow_canvas(
    h_tri: ROOT.TH1D,
    h_data: Optional[ROOT.TH1D] = None,
    data_final_count: Optional[float] = None,
    is_weighted: bool = True,
    canvas_name: str = "c_cutflow_trimuon_superimposed_weighted",
    header_right: Optional[str] = None,
    header_left: Optional[str] = "#bf{SND@LHC} #it{Internal}",
    legend_header: Optional[str] = None
) -> ROOT.TCanvas:
    """
    Renders publication-quality superimposed cutflow TCanvas comparing Trimuon Monte Carlo
    with Collision Data (isolated from PMU MC to prevent visual overcrowding).
    """
    has_data = h_data is not None
    mode_str = "Weighted Yield" if is_weighted else "Raw Counts"
    if has_data:
        mode_str += " & Collision Data"

    c = ROOT.TCanvas(canvas_name, f"Trimuon Cutflow ({mode_str})", 1150, 750)
    ROOT.SetOwnership(c, False)

    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.04)
    c.SetTopMargin(0.085)
    c.SetBottomMargin(0.14)
    c.SetLogy(True)
    c.SetGrid(1, 1)

    y_unit = "Events" if has_data else ("Trimuon MC Yield" if is_weighted else "Raw Trimuon MC Events")

    all_pos = []
    for b in range(1, h_tri.GetNbinsX() + 1):
        v = h_tri.GetBinContent(b)
        if v > 0: all_pos.append(v)
    if h_data:
        for b in range(1, h_data.GetNbinsX() + 1):
            v = h_data.GetBinContent(b)
            if v > 0: all_pos.append(v)

    min_pos = min(all_pos) if all_pos else (1e-4 if is_weighted else 0.5)
    max_pos = max(all_pos) if all_pos else 100.0

    ymin = max(1e-7, 10.0 ** (math.floor(math.log10(min_pos)) - 0.5))
    ymax = 10.0 ** (math.ceil(math.log10(max_pos)) + 2.0)

    frame = h_tri.Clone(f"frame_{canvas_name}")
    ROOT.SetOwnership(frame, False)
    frame.Reset()
    frame.SetTitle(f";;{y_unit}")
    frame.GetYaxis().SetRangeUser(ymin, ymax)
    frame.GetXaxis().LabelsOption("h")
    frame.GetXaxis().SetLabelSize(0.033)
    frame.GetXaxis().SetLabelOffset(0.015)
    frame.GetYaxis().SetTitleSize(0.044)
    frame.GetYaxis().SetLabelSize(0.038)
    frame.GetYaxis().SetTitleOffset(1.25)
    frame.Draw("AXIS")

    h_tri.SetStats(0)
    h_tri.SetLineColor(ROOT.kMagenta + 2)
    h_tri.SetLineWidth(3)
    h_tri.SetMarkerStyle(21)
    h_tri.SetMarkerSize(1.2)
    h_tri.SetMarkerColor(ROOT.kMagenta + 2)
    h_tri.Draw("HIST SAME")
    h_tri.Draw("E1 P SAME")

    if has_data:
        h_data.SetLineColor(ROOT.kBlack)
        h_data.SetLineWidth(2)
        h_data.SetMarkerStyle(20)
        h_data.SetMarkerSize(1.2)
        h_data.SetMarkerColor(ROOT.kBlack)
        h_data.Draw("E1 P SAME")

    leg = ROOT.TLegend(0.48, 0.74, 0.94, 0.89)
    ROOT.SetOwnership(leg, False)
    leg.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg.SetBorderSize(1)
    leg.SetLineColor(ROOT.kGray + 1)
    leg.SetTextFont(42)
    leg.SetTextSize(0.028)
    if legend_header:
        leg.SetHeader(legend_header)

    tri_final = h_tri.GetBinContent(h_tri.GetNbinsX())
    tri_str = fmt_w(tri_final) if is_weighted else f"{int(tri_final):,}"
    leg.AddEntry(h_tri, f"Trimuon MC ({tri_str})", "LP")

    if has_data:
        data_cnt_str = f"{int(round(data_final_count)):,}" if data_final_count is not None else ""
        leg.AddEntry(h_data, f"Collision Data ({data_cnt_str})", "EP")

    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(62)
    latex.SetTextSize(0.036)
    if header_left:
        latex.DrawLatex(0.12, 0.935, header_left)
    latex.SetTextFont(42)
    latex.SetTextSize(0.030)
    latex.SetTextAlign(31)
    hdr_right = header_right if header_right else "Trimuon Monte Carlo Selection Assessment"
    latex.DrawLatex(0.96, 0.938, hdr_right)

    ROOT.gPad.RedrawAxis()
    return c


def build_multi_stage_canvas(
    canvas_name: str,
    canvas_title: str,
    stage_short_titles: List[str],
    hists_mc_tot: List[ROOT.TH1D],
    hists_mc_sig: Optional[List[ROOT.TH1D]] = None,
    hists_data: Optional[List[ROOT.TH1D]] = None,
    cut_lines: Optional[List[float]] = None,
    x_title: str = "Hits",
    log_y: bool = True,
    header_right: Optional[str] = None,
    mc_legend_label: str = "MC Tot",
    mc_color: int = ROOT.kBlack,
    mc_line_style: int = 2,
    sig_legend_label: str = "Signal",
    hists_mc_extra: Optional[List[ROOT.TH1D]] = None,
    extra_legend_label: Optional[str] = None,
    extra_color: int = ROOT.kViolet + 1,
    extra_line_style: int = 7
) -> ROOT.TCanvas:
    """
    Renders an 8-panel (4x2) sequential evolution canvas displaying the hit
    multiplicity distribution at each cut stage (Data vs Scaled MC).
    """
    c = ROOT.TCanvas(canvas_name, canvas_title, 1400, 850)
    ROOT.SetOwnership(c, False)
    c.Divide(4, 2, 0.005, 0.005)

    has_data = hists_data is not None and len(hists_data) == len(hists_mc_tot)

    for i in range(len(stage_short_titles)):
        pad = c.cd(i + 1)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.14)
        pad.SetLogy(log_y)
        pad.SetGrid(1, 1)

        h_tot = hists_mc_tot[i]
        h_sig = hists_mc_sig[i] if (hists_mc_sig is not None and i < len(hists_mc_sig)) else None
        h_dat = hists_data[i] if has_data else None
        h_ext = hists_mc_extra[i] if (hists_mc_extra is not None and i < len(hists_mc_extra)) else None

        pos_vals = []
        for b in range(1, h_tot.GetNbinsX() + 1):
            v = h_tot.GetBinContent(b)
            if v > 0: pos_vals.append(v)
            if h_sig is not None:
                v_sig = h_sig.GetBinContent(b)
                if v_sig > 0: pos_vals.append(v_sig)
            if h_ext is not None:
                v_ext = h_ext.GetBinContent(b)
                if v_ext > 0: pos_vals.append(v_ext)
            if h_dat:
                v_dat = h_dat.GetBinContent(b)
                if v_dat > 0: pos_vals.append(v_dat)

        min_val = min(pos_vals) if pos_vals else 0.1
        max_val = max(pos_vals) if pos_vals else 10.0
        if log_y:
            ymin = max(0.05, 10.0 ** (math.floor(math.log10(min_val)) - 0.2))
            ymax = 10.0 ** (math.ceil(math.log10(max_val)) + 1.2)
        else:
            ymin = 0.0
            ymax = max_val * 1.35

        frame = h_tot.Clone(f"frame_{canvas_name}_p{i}")
        ROOT.SetOwnership(frame, False)
        frame.Reset()
        frame.SetStats(0)
        frame.SetTitle(f";{x_title};Events")
        frame.GetYaxis().SetRangeUser(ymin, ymax)
        frame.GetXaxis().SetTitleSize(0.048)
        frame.GetXaxis().SetLabelSize(0.042)
        frame.GetYaxis().SetTitleSize(0.048)
        frame.GetYaxis().SetLabelSize(0.042)
        frame.GetYaxis().SetTitleOffset(1.25)
        frame.Draw("AXIS")

        # MC Total (Line + Vertical Error Bars)
        h_tot.SetStats(0)
        h_tot.SetLineColor(mc_color)
        h_tot.SetLineStyle(mc_line_style)
        h_tot.SetLineWidth(2)
        h_tot.Draw("HIST SAME")

        h_tot_err = h_tot.Clone(f"{h_tot.GetName()}_err_p{i}")
        ROOT.SetOwnership(h_tot_err, False)
        h_tot_err.SetStats(0)
        h_tot_err.SetLineColor(mc_color)
        h_tot_err.SetMarkerStyle(0)
        h_tot_err.SetLineWidth(1)
        h_tot_err.Draw("E1 SAME")

        # MC Extra component (e.g. Multi-Muon) - optional
        if h_ext is not None:
            h_ext.SetStats(0)
            h_ext.SetLineColor(extra_color)
            h_ext.SetLineStyle(extra_line_style)
            h_ext.SetLineWidth(2)
            h_ext.Draw("HIST SAME")

        # MC Signal (Fill + Outline + Vertical Error Bars) - optional
        if h_sig is not None:
            h_sig.SetStats(0)
            h_sig.SetLineColor(ROOT.kAzure + 2)
            h_sig.SetFillColorAlpha(ROOT.kAzure + 1, 0.25)
            h_sig.SetLineWidth(2)
            h_sig.Draw("HIST SAME")

            h_sig_err = h_sig.Clone(f"{h_sig.GetName()}_err_p{i}")
            ROOT.SetOwnership(h_sig_err, False)
            h_sig_err.SetStats(0)
            h_sig_err.SetLineColor(ROOT.kAzure + 2)
            h_sig_err.SetMarkerStyle(0)
            h_sig_err.SetLineWidth(1)
            h_sig_err.Draw("E1 SAME")

        # Collision Data
        if h_dat:
            h_dat.SetStats(0)
            h_dat.SetLineColor(ROOT.kBlack)
            h_dat.SetMarkerStyle(20)
            h_dat.SetMarkerSize(0.7)
            h_dat.SetMarkerColor(ROOT.kBlack)
            h_dat.Draw("E1 P SAME")

        # Selection cut boundaries
        if cut_lines:
            for x_cut in cut_lines:
                line = ROOT.TLine(x_cut, ymin, x_cut, ymax * 0.4)
                ROOT.SetOwnership(line, False)
                line.SetLineColor(ROOT.kRed + 1)
                line.SetLineStyle(7)
                line.SetLineWidth(2)
                line.Draw("SAME")

        leg = ROOT.TLegend(0.40, 0.60, 0.94, 0.89)
        ROOT.SetOwnership(leg, False)
        leg.SetTextFont(42)
        leg.SetTextSize(0.034)
        leg.SetBorderSize(1)
        leg.SetFillColorAlpha(ROOT.kWhite, 0.85)
        if h_dat:
            leg.AddEntry(h_dat, f"Data ({int(round(h_dat.Integral())):,})", "EP")
        leg.AddEntry(h_tot, f"{mc_legend_label} ({fmt_w(h_tot.Integral())})", "LE")
        if h_sig is not None:
            leg.AddEntry(h_sig, f"{sig_legend_label} ({fmt_w(h_sig.Integral())})", "FLE")
        if h_ext is not None:
            lbl = extra_legend_label if extra_legend_label else "Extra"
            leg.AddEntry(h_ext, f"{lbl} ({fmt_w(h_ext.Integral())})", "L")
        leg.Draw()

        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextFont(62)
        latex.SetTextSize(0.050)
        latex.DrawLatex(0.14, 0.92, stage_short_titles[i])

    return c


def build_muon_energy_stages_canvas(
    canvas_name: str,
    canvas_title: str,
    stage_short_titles: List[str],
    hists_energy_sig: List[ROOT.TH1D],
    hists_energy_all: List[ROOT.TH1D],
    header_right: Optional[str] = None
) -> ROOT.TCanvas:
    """
    Renders an 8-panel (4x2) sequential evolution canvas displaying the passing
    muon energy spectrum at each cut stage in Monte Carlo.
    """
    c = ROOT.TCanvas(canvas_name, canvas_title, 1400, 850)
    ROOT.SetOwnership(c, False)
    c.Divide(4, 2, 0.005, 0.005)

    for i in range(len(stage_short_titles)):
        pad = c.cd(i + 1)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.14)
        pad.SetLogy(True)
        pad.SetGrid(1, 1)

        h_sig = hists_energy_sig[i]
        h_all = hists_energy_all[i]

        pos_vals = []
        for b in range(1, h_all.GetNbinsX() + 1):
            v = h_all.GetBinContent(b)
            if v > 0: pos_vals.append(v)
            v_sig = h_sig.GetBinContent(b)
            if v_sig > 0: pos_vals.append(v_sig)

        min_val = min(pos_vals) if pos_vals else 0.01
        max_val = max(pos_vals) if pos_vals else 10.0
        ymin = max(1e-4, 10.0 ** (math.floor(math.log10(min_val)) - 0.2))
        ymax = 10.0 ** (math.ceil(math.log10(max_val)) + 1.2)

        frame = h_all.Clone(f"frame_e_p{i}")
        ROOT.SetOwnership(frame, False)
        frame.Reset()
        frame.SetStats(0)
        frame.SetTitle(";E_{#mu} [GeV];Events / 20 GeV")
        frame.GetYaxis().SetRangeUser(ymin, ymax)
        frame.GetXaxis().SetTitleSize(0.048)
        frame.GetXaxis().SetLabelSize(0.042)
        frame.GetYaxis().SetTitleSize(0.048)
        frame.GetYaxis().SetLabelSize(0.042)
        frame.GetYaxis().SetTitleOffset(1.25)
        frame.Draw("AXIS")

        h_all.SetStats(0)
        h_all.SetLineColor(ROOT.kBlack)
        h_all.SetLineStyle(2)
        h_all.SetLineWidth(2)
        h_all.Draw("HIST SAME")

        h_sig.SetStats(0)
        h_sig.SetLineColor(ROOT.kAzure + 2)
        h_sig.SetFillColorAlpha(ROOT.kAzure + 1, 0.25)
        h_sig.SetLineWidth(2)
        h_sig.Draw("HIST SAME")

        leg = ROOT.TLegend(0.38, 0.65, 0.94, 0.88)
        ROOT.SetOwnership(leg, False)
        leg.SetTextFont(42)
        leg.SetTextSize(0.036)
        leg.SetBorderSize(1)
        leg.SetFillColorAlpha(ROOT.kWhite, 0.85)
        leg.AddEntry(h_all, f"All Mu (N={fmt_w(h_all.Integral())})", "L")
        leg.AddEntry(h_sig, f"Signal (N={fmt_w(h_sig.Integral())})", "FL")
        leg.AddEntry("", f"#LT E_{{#mu}}#GT = {h_sig.GetMean():.0f} GeV", "")
        leg.Draw()

        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextFont(62)
        latex.SetTextSize(0.050)
        latex.DrawLatex(0.14, 0.92, stage_short_titles[i])

    return c


def build_energy_overlay_canvas(
    canvas_name: str,
    stage_short_titles: List[str],
    hists_energy_sig: List[ROOT.TH1D],
    header_right: Optional[str] = None
) -> ROOT.TCanvas:
    """
    Overlays the signal muon energy spectrum across all 8 successive cut stages
    on a single publication-grade canvas to evaluate spectral bias or shaping.
    """
    c = ROOT.TCanvas(canvas_name, "Passing Muon Energy Spectrum across Cut Stages", 1150, 750)
    ROOT.SetOwnership(c, False)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.04)
    c.SetTopMargin(0.085)
    c.SetBottomMargin(0.14)
    c.SetLogy(True)
    c.SetGrid(1, 1)

    colors = [
        ROOT.kGray + 1,
        ROOT.kCyan + 2,
        ROOT.kTeal + 2,
        ROOT.kGreen + 2,
        ROOT.kOrange + 7,
        ROOT.kRed + 1,
        ROOT.kMagenta + 2,
        ROOT.kBlue + 2
    ]

    all_pos = []
    for h in hists_energy_sig:
        for b in range(1, h.GetNbinsX() + 1):
            v = h.GetBinContent(b)
            if v > 0: all_pos.append(v)

    min_val = min(all_pos) if all_pos else 0.01
    max_val = max(all_pos) if all_pos else 100.0
    ymin = max(1e-4, 10.0 ** (math.floor(math.log10(min_val)) - 0.5))
    ymax = 10.0 ** (math.ceil(math.log10(max_val)) + 1.8)

    frame = hists_energy_sig[0].Clone(f"frame_{canvas_name}")
    ROOT.SetOwnership(frame, False)
    frame.Reset()
    frame.SetStats(0)
    frame.SetTitle(";E_{#mu} [GeV];Scaled Passing Muon Yield / 20 GeV")
    frame.GetYaxis().SetRangeUser(ymin, ymax)
    frame.GetXaxis().SetTitleSize(0.044)
    frame.GetXaxis().SetLabelSize(0.038)
    frame.GetYaxis().SetTitleSize(0.044)
    frame.GetYaxis().SetLabelSize(0.038)
    frame.GetYaxis().SetTitleOffset(1.25)
    frame.Draw("AXIS")

    leg = ROOT.TLegend(0.40, 0.65, 0.95, 0.91)
    ROOT.SetOwnership(leg, False)
    leg.SetNColumns(2)
    leg.SetFillColorAlpha(ROOT.kWhite, 0.92)
    leg.SetBorderSize(1)
    leg.SetLineColor(ROOT.kGray + 1)
    leg.SetTextFont(42)
    leg.SetTextSize(0.024)
    leg.SetHeader("Passing Muon Signal Energy Spectrum by Cut Stage", "C")

    for i, h in enumerate(hists_energy_sig):
        h.SetStats(0)
        col = colors[i % len(colors)]
        h.SetLineColor(col)
        h.SetLineWidth(3 if i == len(hists_energy_sig) - 1 else 2)
        h.SetLineStyle(2 if i == 0 else 1)
        h.Draw("HIST SAME")
        mean_e = h.GetMean()
        integ = h.Integral()
        leg.AddEntry(h, f"{stage_short_titles[i]} (#LT E_{{#mu}}#GT={mean_e:.0f} GeV)", "L")

    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(62)
    latex.SetTextSize(0.038)
    latex.DrawLatex(0.12, 0.938, "#bf{SND@LHC} #font[52]{Simulation}")

    latex.SetTextFont(42)
    latex.SetTextSize(0.028)
    latex.SetTextAlign(31)
    hdr_right = header_right if header_right else "Passing Muon FLUKA Weighted Energy Spectrum"
    latex.DrawLatex(0.96, 0.938, hdr_right)

    ROOT.gPad.RedrawAxis()
    return c


def get_or_create_dir(parent_tfile_or_dir, path_str: str):
    """Safely navigates or creates nested directory hierarchy in ROOT."""
    parts = [p for p in path_str.split("/") if p]
    curr = parent_tfile_or_dir
    for part in parts:
        sub = curr.GetDirectory(part)
        if not sub:
            sub = curr.mkdir(part)
        curr = sub
    return curr


def format_table_row(cols: List[str], widths: List[int]) -> str:
    """Formats a row of a fixed-width table."""
    return " | ".join(f"{col:>{w}}" for col, w in zip(cols, widths))


def fmt_w(val: float) -> str:
    """Formats weighted yields cleanly depending on magnitude."""
    if val == 0:
        return "0.0"
    elif 0.001 <= abs(val) < 1e5:
        return f"{val:.3f}"
    else:
        return f"{val:.3e}"


def print_summary_tables(
    cat_w_all: Dict[int, float],
    cat_r_all: Dict[int, int],
    cat_w_pass: Dict[int, float],
    cat_r_pass: Dict[int, int],
    cutflow_matrix_w: Dict[str, Dict[int, float]],
    cutflow_matrix_r: Dict[str, Dict[int, int]],
    data_counts: Optional[Dict[str, float]] = None,
    raw_data_counts: Optional[Dict[str, int]] = None,
    pmu_scale_factor: float = 1.0,
    data_meta: Optional[Dict[str, Any]] = None,
    section_offset: int = 0,
    sample_tag: str = "PMU MC",
    print_exposure: bool = True
):
    """Prints comprehensive assessment, purity, rejection, and cutflow tables with FLUKA weights and Data comparison."""
    if data_meta and print_exposure:
        print("\n" + "=" * 80)
        print("COLLISION DATA EXPOSURE & PASSING MUON MC SCALING")
        print("=" * 80)
        print(f"  Data Source           : {data_meta.get('source')}")
        print(f"  Run / Fill Number     : Run {data_meta.get('run')} (Fill {data_meta.get('fill')})")
        print(f"  Total Data Events     : {data_meta.get('entries', 0):,}")
        if data_meta.get("scaling", 1.0) != 1.0:
            print(f"  Data Scaling Factor   : {data_meta['scaling']:.2f} (Random subsample inverse fraction)")
        print(f"  Data Duration         : {data_meta.get('duration', 0.0):.2f} s ({data_meta.get('dur_str')})")
        print(f"  Integrated Luminosity : {data_meta.get('lumi', 0.0):.6e} fb^-1 ({data_meta.get('lumi_str')})")
        print(f"  MC Ref Luminosity     : L_MC = 1/8e5 s^-1 fb^-1 (100e6 coll/s at 80 mb)")
        print(f"  Rate Factor (k_s)     : {data_meta.get('rate_factor', 0.0):.4f} s^-1  [L_inst / L_MC]")
        print(f"  PMU MC Scale Factor   : {pmu_scale_factor:.4f}  [k_s * duration = Lumi * 8e5]")
        print("=" * 80)

    widths = [32, 18, 18, 14, 14]
    sep_line = "-" * (sum(widths) + 3 * (len(widths) - 1))

    print("\n" + "=" * len(sep_line))
    scale_label = f" (Scaled to Lumi = {data_meta['lumi_str']})" if data_meta else (f" (Scaled by {pmu_scale_factor:.4f})" if abs(pmu_scale_factor - 1.0) > 1e-9 else "")
    sec_1 = 1 + section_offset
    print(f"{sec_1}. TRUTH CLASSIFICATION & SELECTION PERFORMANCE [{sample_tag}{scale_label}]")
    print("=" * len(sep_line))
    header = ["Truth Category", "All [Weight (Raw)]", "Selected [Wt (Raw)]", "Weighted Eff", "Rejection"]
    print(format_table_row(header, widths))
    print(sep_line)

    tot_w_all_det = sum(cat_w_all.get(i, 0.0) for i in range(1, 7))
    tot_r_all_det = sum(cat_r_all.get(i, 0) for i in range(1, 7))
    tot_w_pass_det = sum(cat_w_pass.get(i, 0.0) for i in range(1, 7))
    tot_r_pass_det = sum(cat_r_pass.get(i, 0) for i in range(1, 7))

    for cat_id in range(1, 7):
        name = CATEGORY_LABELS.get(cat_id, f"Category {cat_id}")
        w_all = cat_w_all.get(cat_id, 0.0)
        r_all = cat_r_all.get(cat_id, 0)
        w_pass = cat_w_pass.get(cat_id, 0.0)
        r_pass = cat_r_pass.get(cat_id, 0)

        pass_rate = (w_pass / w_all * 100.0) if w_all > 0 else 0.0
        rejection = (1.0 - w_pass / w_all) * 100.0 if w_all > 0 else 100.0

        all_str = f"{fmt_w(w_all)} ({r_all:,})"
        pass_str = f"{fmt_w(w_pass)} ({r_pass:,})"

        if cat_id == 1:
            row = [name, all_str, pass_str, f"{pass_rate:6.2f}%", "--"]
        else:
            row = [name, all_str, pass_str, f"{pass_rate:6.2f}%", f"{rejection:6.2f}%"]
        print(format_table_row(row, widths))

    print(sep_line)
    w_sig_all = cat_w_all.get(1, 0.0)
    w_sig_pass = cat_w_pass.get(1, 0.0)
    w_bkg_all = tot_w_all_det - w_sig_all
    w_bkg_pass = tot_w_pass_det - w_sig_pass

    tot_bkg_rej = (1.0 - w_bkg_pass / w_bkg_all) * 100.0 if w_bkg_all > 0 else 100.0
    purity_w = (w_sig_pass / tot_w_pass_det * 100.0) if tot_w_pass_det > 0 else 0.0
    contamination_w = (100.0 - purity_w) if tot_w_pass_det > 0 else 0.0

    tot_all_str = f"{fmt_w(tot_w_all_det)} ({tot_r_all_det:,})"
    tot_pass_str = f"{fmt_w(tot_w_pass_det)} ({tot_r_pass_det:,})"
    bkg_all_str = f"{fmt_w(w_bkg_all)} ({tot_r_all_det - cat_r_all.get(1, 0):,})"
    bkg_pass_str = f"{fmt_w(w_bkg_pass)} ({tot_r_pass_det - cat_r_pass.get(1, 0):,})"

    print(format_table_row(["TOTAL DETECTOR ACTIVE", tot_all_str, tot_pass_str, f"{tot_w_pass_det/tot_w_all_det*100.0 if tot_w_all_det>0 else 0:6.2f}%", "--"], widths))
    print(format_table_row(["TOTAL NON-SIGNAL (BACKGROUND)", bkg_all_str, bkg_pass_str, f"{w_bkg_pass/w_bkg_all*100.0 if w_bkg_all>0 else 0:6.2f}%", f"{tot_bkg_rej:6.2f}%"], widths))
    print("=" * len(sep_line))

    # Purity and Quality Assessment Box
    sec_2 = 2 + section_offset
    print("\n" + "=" * 64)
    print(f"{sec_2}. CALIBRATION SAMPLE PURITY & CONTAMINATION ({sample_tag})")
    print("=" * 64)
    print(f"  Target Signal Efficiency       : {w_sig_pass / w_sig_all * 100.0 if w_sig_all > 0 else 0.0:6.2f}% (Yield: {fmt_w(w_sig_pass)} / {fmt_w(w_sig_all)})")
    print(f"  Total Background Rejection     : {tot_bkg_rej:6.2f}%")
    print(f"  Selected Sample Purity         : {purity_w:6.2f}% (Clean Single Muons)")
    print(f"  Selected Sample Contamination  : {contamination_w:6.2f}%")
    if tot_w_pass_det > 0:
        print("  Residual Contaminant Breakdown (by weighted yield):")
        for cat_id in range(2, 7):
            w_cnt = cat_w_pass.get(cat_id, 0.0)
            r_cnt = cat_r_pass.get(cat_id, 0)
            frac = w_cnt / tot_w_pass_det * 100.0
            print(f"    - {CATEGORY_LABELS[cat_id]:32s}: {fmt_w(w_cnt)} ({r_cnt} raw) [{frac:5.2f}%]")
    print("=" * 64)

    # Cutflow Matrix Table
    sec_3 = 3 + section_offset
    if data_counts is not None:
        data_scaling = data_meta.get("scaling", 1.0) if data_meta else 1.0
        is_scaled = (raw_data_counts is not None and abs(data_scaling - 1.0) > 1e-9)
        d_col_title = "Collision Data [Scaled (Raw)]" if is_scaled else "Collision Data"
        d_col_w = 30 if is_scaled else 16
        c_widths = [24, d_col_w, 14, 14, 14, 14, 14, 14]
        tot_width = sum(c_widths) + 3 * (len(c_widths) - 1)
        print("\n" + "=" * tot_width)
        print(f"{sec_3}. SEQUENTIAL CUTFLOW MATRIX: DATA vs SCALED {sample_tag}")
        print("=" * tot_width)
        c_header = ["Cut Stage", d_col_title, "MC Signal", "EM Cascade", "Hadronic", "Multi-Mu", "Halo/NoMu", f"Total {sample_tag}"]
        print(" | ".join(f"{h:^{w}}" for h, w in zip(c_header, c_widths)))
        print("-" * tot_width)

        for stage_name, counts_w in cutflow_matrix_w.items():
            if is_scaled:
                d_cnt = data_counts.get(stage_name, 0.0)
                r_cnt = raw_data_counts.get(stage_name, 0)
                d_str = f"{d_cnt:,.0f} ({r_cnt:,})"
            else:
                d_cnt = data_counts.get(stage_name, 0.0)
                d_str = f"{int(round(d_cnt)):,}"
            row = [
                stage_name,
                d_str,
                fmt_w(counts_w.get(1, 0.0)),
                fmt_w(counts_w.get(2, 0.0)),
                fmt_w(counts_w.get(3, 0.0)),
                fmt_w(counts_w.get(4, 0.0)),
                fmt_w(counts_w.get(5, 0.0)),
                fmt_w(sum(counts_w.get(c, 0.0) for c in range(1, 7)))
            ]
            print(" | ".join(f"{val:>{w}}" for val, w in zip(row, c_widths)))
        print("=" * tot_width + "\n")
    else:
        print("\n" + "=" * 104)
        print("3. SEQUENTIAL CUTFLOW MATRIX: WEIGHTED YIELD (RAW COUNTS)")
        print("=" * 104)
        c_widths = [26, 14, 14, 14, 14, 14, 14]
        c_header = ["Cut Stage", "Signal", "EM Cascade", "Hadronic", "Multi-Mu", "Halo/NoMu", "Total Active"]
        print(" | ".join(f"{h:^{w}}" for h, w in zip(c_header, c_widths)))
        print("-" * 104)

        for stage_name, counts_w in cutflow_matrix_w.items():
            counts_r = cutflow_matrix_r[stage_name]
            row = [
                stage_name,
                f"{fmt_w(counts_w.get(1, 0.0))} ({counts_r.get(1, 0)})",
                f"{fmt_w(counts_w.get(2, 0.0))} ({counts_r.get(2, 0)})",
                f"{fmt_w(counts_w.get(3, 0.0))} ({counts_r.get(3, 0)})",
                f"{fmt_w(counts_w.get(4, 0.0))} ({counts_r.get(4, 0)})",
                f"{fmt_w(counts_w.get(5, 0.0))} ({counts_r.get(5, 0)})",
                f"{fmt_w(sum(counts_w.get(c, 0.0) for c in range(1, 7)))} ({sum(counts_r.get(c, 0) for c in range(1, 7))})"
            ]
            print(" | ".join(f"{val:>{w}}" for val, w in zip(row, c_widths)))
        print("=" * 104 + "\n")


def print_trimuon_summary_table(
    cutflow_stages: List[Tuple[str, str]],
    tri_counts_w: Dict[str, float],
    tri_counts_r: Dict[str, int],
    data_counts: Optional[Dict[str, float]] = None,
    raw_data_counts: Optional[Dict[str, int]] = None,
    scale_tri: float = 1.0,
    lumi_str: str = "N/A",
    table_number: int = 7
):
    """Prints a dedicated assessment table for Trimuon Monte Carlo sequential selection survival."""
    is_scaled_data = (raw_data_counts is not None and data_counts is not None and any(abs(data_counts.get(s[0], 0.0) - raw_data_counts.get(s[0], 0)) > 1e-3 for s in cutflow_stages))
    d_col_title = "Collision Data [Scaled (Raw)]" if is_scaled_data else "Collision Data"
    d_col_w = 30 if is_scaled_data else 18
    c_widths = [24, 18, 14, 14, 14, d_col_w] if data_counts else [26, 20, 16, 16, 16]
    tot_width = sum(c_widths) + 3 * (len(c_widths) - 1)

    print("\n" + "=" * tot_width)
    print(f"{table_number}. TRIMUON MONTE CARLO CUTFLOW & SELECTION SURVIVAL (Scaled to Lumi = {lumi_str})")
    print("=" * tot_width)
    c_header = ["Cut Stage", "Trimuon Yield", "Raw Events", "Stage Eff (%)", "Step Eff (%)"]
    if data_counts:
        c_header.append(d_col_title)
    print(" | ".join(f"{h:^{w}}" for h, w in zip(c_header, c_widths)))
    print("-" * tot_width)

    s0_w = tri_counts_w.get(cutflow_stages[0][0], 0.0)
    prev_w = s0_w

    for s_name, _ in cutflow_stages:
        w_val = tri_counts_w.get(s_name, 0.0)
        r_val = tri_counts_r.get(s_name, 0)
        cum_eff = (w_val / s0_w * 100.0) if s0_w > 0 else 0.0
        step_eff = (w_val / prev_w * 100.0) if prev_w > 0 else 0.0
        prev_w = w_val

        row = [
            s_name,
            fmt_w(w_val),
            f"{r_val:,}",
            f"{cum_eff:6.2f}%",
            f"{step_eff:6.2f}%"
        ]
        if data_counts:
            if is_scaled_data:
                d_c = data_counts.get(s_name, 0.0)
                r_c = raw_data_counts.get(s_name, 0) if raw_data_counts else 0
                row.append(f"{d_c:,.0f} ({r_c:,})")
            else:
                d_c = data_counts.get(s_name, 0.0)
                row.append(f"{int(round(d_c)):,}")
        print(" | ".join(f"{val:>{w}}" for val, w in zip(row, c_widths)))

    print("=" * tot_width + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SND@LHC Passing Muon Selection & Assessment Engine (FLUKA Weighted & Collision Data)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Input options: Simulation, Trimuon MC, and Collision Data
    parser.add_argument("-ipmu", "--input-pmu", "-i", "--input", "--input-mc", "--mc", dest="input_mc",
                        default="/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root",
                        help="Input passing muon simulation ROOT file")
    parser.add_argument("-itri", "--input-tri", "--trimuon", "--input-trimuon", dest="input_tri",
                        default="/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-*_trks.root",
                        help="Input trimuon Monte Carlo ROOT file(s) or pattern (default: 100 files on EOS)")
    parser.add_argument("-id", "--input-data", "-d", "--data", dest="input_data", default=None,
                        help="Input collision data ROOT file, directory, or run number")
    parser.add_argument("-t", "--tree", "--tree-mc", dest="tree_mc", default="cbmsim",
                        help="MC TTree name (default: cbmsim)")
    parser.add_argument("--tree-tri", dest="tree_tri", default="cbmsim",
                        help="Trimuon MC TTree name (default: cbmsim)")
    parser.add_argument("--tree-data", dest="tree_data", default="",
                        help="Data TTree name (default: auto-detected, e.g. rawConv)")
    parser.add_argument("-o", "--output", dest="output_file",
                        default="plots/passing_muon_selection.root",
                        help="Output ROOT file with validation histograms and cutflow graphs")
    parser.add_argument("-n", "--max-events", dest="max_events", type=int, default=0,
                        help="Max events limit (0 = all)")
    parser.add_argument("-j", "--threads", "--workers", dest="num_threads", type=int, default=8,
                        help="Number of RDataFrame worker threads (default: 8)")

    # Data Duration, Luminosity, PMU and Trimuon Scaling overrides
    parser.add_argument("--data-duration", dest="data_duration", type=float, default=None,
                        help="Explicit duration of collision data in seconds (overrides DataManager extraction)")
    parser.add_argument("--data-lumi", dest="data_lumi", type=float, default=None,
                        help="Explicit integrated luminosity of data in fb^-1 (overrides DataManager extraction)")
    parser.add_argument("--scale-pmu", dest="scale_pmu", type=float, default=None,
                        help="Explicit scale factor for PMU simulation (overrides luminosity scaling: lumi * 8e5)")
    parser.add_argument("--scale-tri", "--scale-trimuon", dest="scale_tri", type=float, default=None,
                        help="Explicit scale factor for trimuon simulation (overrides luminosity scaling: data_lumi / lumi_mc_tri)")
    parser.add_argument("--lumi-mc-tri", "--tri-lumi-mc", dest="lumi_mc_tri", type=float, default=0.025,
                        help="Nominal integrated luminosity of trimuon MC in fb^-1 (default: 0.025 = 1/40 fb^-1)")
    parser.add_argument("--no-tri", "--disable-tri", dest="enable_tri", action="store_false", default=True,
                        help="Disable trimuon Monte Carlo processing")
    parser.add_argument("--data-scaling", "--data-scale", "--scale-data", dest="data_scaling", type=float, default=1.0,
                        help="Scaling factor for collision data (default: 1.0, e.g. 100.0 if reconstruction ran on 1/100 random subsample)")

    # Configurable selection cuts
    parser.add_argument("--chi2-max", type=float, default=10.0, help="General track chi2/ndf upper limit")
    parser.add_argument("--chi2-max-scifi", type=float, default=10.0, help="Max SciFi track chi2/ndf (default: 10.0)")
    parser.add_argument("--chi2-max-ds", type=float, default=10.0, help="Max DS track chi2/ndf (default: 10.0)")
    parser.add_argument("--max-slope", type=float, default=0.05, help="Max angular slope (default: 0.05 rad)")
    parser.add_argument("--z-match", type=float, default=430.0, help="Fiducial plane Z position [cm] (default: 430.0)")
    parser.add_argument("--pos-match-max", type=float, default=3.0, help="Max distance at z_match between SF and DS (default: 3.0 cm)")
    parser.add_argument("--angle-match-max", type=float, default=0.015, help="Max angle difference between SF and DS (default: 0.015 rad)")
    parser.add_argument("--scifi-track-type", type=int, default=11, help="SciFi track type (default: 11 = SciFi Hough)")
    parser.add_argument("--ds-track-type", type=int, default=13, help="DS track type (default: 13 = DS Hough)")
    parser.add_argument("--ds-match-slope", type=float, default=0.04, help="Max slope diff between SF and DS (default: 0.04 rad)")
    parser.add_argument("--fiducial-margin", type=float, default=1.5, help="Fiducial border margin in cm (default: 1.5)")
    parser.add_argument("--scifi-min", type=int, default=10, help="Min SciFi hits (default: 10)")
    parser.add_argument("--scifi-max", type=int, default=35, help="Max SciFi hits (default: 35)")
    parser.add_argument("--ds-min", type=int, default=6, help="Min Downstream MuFilter hits (default: 6)")
    parser.add_argument("--require-ip1", "--ip1", dest="require_ip1", action="store_true", default=True,
                        help="Require EventHeader.isIP1() for collision data (default: True)")
    parser.add_argument("--no-ip1", dest="require_ip1", action="store_false",
                        help="Disable EventHeader.isIP1() requirement for collision data")

    # Truth definition parameters
    parser.add_argument("--shower-threshold", type=float, default=2.0,
                        help="Kinetic energy threshold for catastrophic secondary shower in GeV (default: 2.0 GeV)")

    args = parser.parse_args()

    if args.data_scaling <= 0:
        parser.error(f"--data-scaling must be strictly positive (> 0), got {args.data_scaling}")

    t0 = time.time()
    print("=" * 80)
    print("SND@LHC PASSING MUON SELECTION & ASSESSMENT ENGINE (FLUKA WEIGHTED & DATA)")
    print("=" * 80)
    print(f"Input PMU MC File  : {args.input_mc}")
    print(f"Input Trimuon MC   : {args.input_tri if args.enable_tri and args.input_tri else 'Disabled'}")
    print(f"Input Collision Data: {args.input_data if args.input_data else 'None (MC Only Mode)'}")
    if args.input_data and abs(args.data_scaling - 1.0) > 1e-9:
        print(f"Data Scaling Factor: {args.data_scaling:g} (subsample scaling to full luminosity)")
    print(f"PMU MC TTree Name  : {args.tree_mc}")
    print(f"Weight Formula     : MCTrack[0]->GetWeight() (FLUKA generator weight)")
    print(f"Max Events         : {args.max_events if args.max_events > 0 else 'All'}")
    print(f"Worker Threads     : {args.num_threads}")
    print(f"Output File        : {args.output_file}")
    print("-" * 80)
    print("Selection Criteria to Assess:")
    print(f"  1. EventHeader.isIP1()      : Collision Data Only (MC identical to Stage 0)")
    print(f"  2. SciFi Track Multiplicity : Exactly 1 Reconstructed SciFi Track")
    print(f"  3. DS Track Requirement     : Reconstructed DS Track (Type {args.ds_track_type})")
    print(f"  4. Track Fit Quality        : SciFi #chi2/ndf <= {args.chi2_max_scifi}, DS #chi2/ndf <= {args.chi2_max_ds}")
    print(f"  5. Max Angular Slope        : < {args.max_slope} rad")
    print(f"  6. Fiducial Plane at z=430  : Margin >= {args.fiducial_margin} cm at z = {args.z_match} cm")
    print(f"  7. DS-SciFi Match at z=430  : #Delta R <= {args.pos_match_max} cm, #Delta#theta <= {args.angle_match_max} rad")
    print(f"  [!] No hit multiplicity or QDC cuts applied")
    print("=" * 80)

    # 1. Enable Implicit Multi-Threading
    if args.num_threads > 1:
        ROOT.EnableImplicitMT(args.num_threads)
        print(f"[*] Enabled ROOT Implicit Multi-Threading with {args.num_threads} worker threads.")

    # 2. Extract Collision Data Metadata & PMU Scale Factor (if Data provided)
    data_meta = None
    data_duration = 0.0
    data_lumi = 0.0
    dur_str = "N/A"
    lumi_str = "N/A"
    pmu_scale_factor = 1.0

    if args.input_data:
        print("\n[*] Initializing Collision DataManager for exposure assessment...")
        dm_data = DataManager(args.input_data, tree_name=args.tree_data, num_threads=args.num_threads)
        data_duration, data_lumi, dur_str, lumi_str = extract_duration_and_lumi(
            dm_data, args.data_duration, args.data_lumi
        )
        data_run = dm_data.run_number
        if not data_run:
            _rn_match = re.search(r"run_(\d+)", args.input_data)
            data_run = int(_rn_match.group(1)) if _rn_match else "N/A"
        data_fill = dm_data.fill_number or "N/A"
        if dm_data.num_files <= 20:
            data_entries = dm_data.entries
            entries_str = f"{data_entries:,}"
        else:
            data_entries = -1
            entries_str = "Streamed across files"

        if args.scale_pmu is not None and args.scale_pmu > 0:
            pmu_scale_factor = args.scale_pmu
            rate_factor = pmu_scale_factor / data_duration if data_duration > 0 else 0.0
            print(f"[*] Manual PMU MC scale factor override: {pmu_scale_factor:.4f}")
        elif data_lumi > 0:
            # Scale PMU MC:
            # 100e6 coll/s with sigma = 80 mb gives L_MC = 1 / 8e5 [s^-1 fb^-1]
            # Rate factor: k_s = L_inst / L_MC = (data_lumi / data_duration) * 8e5 [s^-1]
            # Exposure scaling: scale_factor = k_s * data_duration = data_lumi * 8e5
            rate_factor = (data_lumi / data_duration) * PMU_LUMI_SCALE_COEFF if data_duration > 0 else 0.0
            pmu_scale_factor = data_lumi * PMU_LUMI_SCALE_COEFF
        else:
            pmu_scale_factor = 1.0
            rate_factor = 1.0 / data_duration if data_duration > 0 else 0.0
            print("[!] Warning: Integrated luminosity is 0 or unavailable; defaulting PMU MC scale factor to 1.0")

        data_meta = {
            "source": args.input_data,
            "run": data_run,
            "fill": data_fill,
            "entries": data_entries,
            "duration": data_duration,
            "lumi": data_lumi,
            "dur_str": dur_str,
            "lumi_str": lumi_str,
            "rate_factor": rate_factor,
            "scaling": args.data_scaling,
        }
        print(f"  * Run: {data_run} | Fill: {data_fill} | Files: {dm_data.num_files} | Events: {entries_str}")
        if abs(args.data_scaling - 1.0) > 1e-9:
            print(f"  * Collision Data Scaling: Factor = {args.data_scaling:g} (Scaling {1.0/args.data_scaling * 100:.2g}% subsample to full data)")
        print(f"  * Exposure: Duration = {data_duration:.2f} s ({dur_str}) | Integrated Lumi = {data_lumi:.6e} fb^-1 ({lumi_str})")
        print(f"  * MC Ref Luminosity: L_MC = 1/8e5 s^-1 fb^-1 (100e6 coll/s at 80 mb)")
        print(f"  * Rate Factor (k_s) : L_inst / L_MC = {rate_factor:.4f} s^-1")
        print(f"  * Computed PMU MC Scale Factor = k_s * duration = Lumi * {PMU_LUMI_SCALE_COEFF:g} = {pmu_scale_factor:.4f}")
    else:
        if args.scale_pmu is not None and args.scale_pmu > 0:
            pmu_scale_factor = args.scale_pmu
            print(f"[*] Manual PMU MC scale factor: {pmu_scale_factor:.4f}")
        else:
            pmu_scale_factor = 1.0

    # 3. Canonicalize MC input path & auto-detect local cached copy
    resolved_mc_input = os.path.realpath(args.input_mc) if os.path.exists(args.input_mc) else args.input_mc
    local_data_path = os.path.join(_repo_root, "data", os.path.basename(resolved_mc_input))
    if os.path.exists(local_data_path) and os.path.getsize(local_data_path) > 1024 * 1024 * 100:
        if resolved_mc_input != local_data_path:
            print(f"[*] Detected local cached copy of MC at: {local_data_path}")
            print(f"    --> Automatically switching to local MC file to avoid EOS network latency!")
            resolved_mc_input = local_data_path

    # Setup MC RDataFrame
    if "*" in resolved_mc_input or "?" in resolved_mc_input:
        mc_dm = DataManager(resolved_mc_input, tree_name=args.tree_mc, num_threads=args.num_threads)
        df_mc = mc_dm.rdf()
    else:
        df_mc = ROOT.RDataFrame(args.tree_mc, resolved_mc_input)

    if args.max_events > 0:
        if ROOT.IsImplicitMTEnabled():
            df_mc = df_mc.Filter(f"rdfentry_ < {args.max_events}")
        else:
            df_mc = df_mc.Range(args.max_events)

    # 4. Setup Processors
    truth_cfg = ROOT.snd.trident.PassingMuonTruthConfig()
    truth_cfg.shower_energy_threshold = args.shower_threshold
    truth_cfg.fiducial_margin = args.fiducial_margin
    truth_proc = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)

    calib_cfg = ROOT.snd.trident.MuonCalibrationConfig()
    calib_cfg.chi2_max = args.chi2_max
    calib_cfg.chi2_max_scifi = args.chi2_max_scifi
    calib_cfg.chi2_max_ds = args.chi2_max_ds
    calib_cfg.max_slope = args.max_slope
    calib_cfg.fiducial_margin = args.fiducial_margin
    calib_cfg.scifi_track_type = args.scifi_track_type
    calib_cfg.ds_track_type = args.ds_track_type
    calib_cfg.z_match = args.z_match
    calib_cfg.pos_match_max = args.pos_match_max
    calib_cfg.angle_match_max = args.angle_match_max
    calib_cfg.scifi_hits_min = args.scifi_min
    calib_cfg.scifi_hits_max = args.scifi_max
    calib_cfg.ds_hits_min = args.ds_min
    calib_cfg.ds_match_slope_max = args.ds_match_slope
    calib_proc = ROOT.snd.trident.MuonCalibrationProcessor(calib_cfg)

    # 5. Define MC Computational Graph
    cols_mc = set(str(c) for c in df_mc.GetColumnNames())
    track_branch_mc = "Reco_MuonTracks" if "Reco_MuonTracks" in cols_mc else "fittedTracks"
    sf_branch_mc = "Digi_ScifiHits" if "Digi_ScifiHits" in cols_mc else "Digits_Scifi"
    if "Digi_MuFilterHits" in cols_mc:
        mf_branch_mc = "Digi_MuFilterHits"
    elif "Digi_MuFilterHit" in cols_mc:
        mf_branch_mc = "Digi_MuFilterHit"
    else:
        mf_branch_mc = "Digits_MuFilter"

    df_mc = (
        df_mc
        .Define("truth", truth_proc, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
        .Define("reco", calib_proc, [track_branch_mc, sf_branch_mc, mf_branch_mc])
        .Define("weight", "truth.mc_weight")
        .Define("cat_id", "truth.category_id")
        .Define("is_signal", "truth.is_signal")
        .Define("is_clean_reco", "reco.is_clean")
        .Define("reco_chi2_ndf", "reco.track_chi2_ndf")
        .Define("reco_sf_hits", "reco.scifi_nhits")
        .Define("reco_ds_hits", "reco.ds_nhits")
        .Define("reco_veto_hits", "reco.veto_nhits")
        .Define("reco_us_hits", "reco.us_nhits")
        .Define("scaled_weight", f"weight * {pmu_scale_factor:.10e}")
        .Define("mu_p", "truth.primary_muon_p")
        .Define("mu_energy", "truth.primary_muon_p > 0 ? std::sqrt(truth.primary_muon_p*truth.primary_muon_p + 0.105658*0.105658) : 0.0")
        .Define("c1_ip1",           "truth.is_in_acceptance")
        .Define("c2_one_scifi_trk", "c1_ip1 && reco.pass_cut_single_scifi_track")
        .Define("c3_ds_track",      "c2_one_scifi_trk && reco.pass_cut_ds_track")
        .Define("c4_track_quality", "c3_ds_track && reco.pass_cut_chi2")
        .Define("c5_angular_slope", "c4_track_quality && reco.pass_cut_slope")
        .Define("c6_fiducial_430",  "c5_angular_slope && reco.pass_cut_fiducial_430")
        .Define("c7_ds_match",      "c6_fiducial_430 && reco.pass_cut_ds_match")
    )

    # 6. Book MC Histograms
    h_cat_all_w  = df_mc.Histo1D(("h_cat_all_weighted", "All Events by Category (Weighted);Category ID;FLUKA Weighted Yield", 7, -0.5, 6.5), "cat_id", "weight")
    h_cat_all_r  = df_mc.Histo1D(("h_cat_all_raw", "All Events by Category (Raw);Category ID;Raw Events", 7, -0.5, 6.5), "cat_id")

    h_cat_pass_w = df_mc.Filter("is_clean_reco").Histo1D(("h_cat_pass_weighted", "Selected Clean Muons (Weighted);Category ID;FLUKA Weighted Yield", 7, -0.5, 6.5), "cat_id", "weight")
    h_cat_pass_r = df_mc.Filter("is_clean_reco").Histo1D(("h_cat_pass_raw", "Selected Clean Muons (Raw);Category ID;Raw Events", 7, -0.5, 6.5), "cat_id")

    h_chi2_sig  = df_mc.Filter("cat_id == 1 && c2_one_scifi_trk").Histo1D(("h_chi2_signal", "Track #chi^{2}/ndf (Signal);#chi^{2}/ndf;Weighted Tracks", 100, 0, 20), "reco_chi2_ndf", "weight")
    h_chi2_em   = df_mc.Filter("cat_id == 2 && c2_one_scifi_trk").Histo1D(("h_chi2_em_shower", "Track #chi^{2}/ndf (EM Cascade);#chi^{2}/ndf;Weighted Tracks", 100, 0, 20), "reco_chi2_ndf", "weight")
    h_chi2_had  = df_mc.Filter("cat_id == 3 && c2_one_scifi_trk").Histo1D(("h_chi2_hadronic", "Track #chi^{2}/ndf (Hadronic);#chi^{2}/ndf;Weighted Tracks", 100, 0, 20), "reco_chi2_ndf", "weight")

    h_hits_sig  = df_mc.Filter("cat_id == 1 && c2_one_scifi_trk").Histo1D(("h_sf_hits_signal", "SciFi Hits (Signal);SciFi Hits;Weighted Events", 60, 0, 60), "reco_sf_hits", "weight")
    h_hits_em   = df_mc.Filter("cat_id == 2 && c2_one_scifi_trk").Histo1D(("h_sf_hits_em_shower", "SciFi Hits (EM Cascade);SciFi Hits;Weighted Events", 60, 0, 60), "reco_sf_hits", "weight")
    h_hits_had  = df_mc.Filter("cat_id == 3 && c2_one_scifi_trk").Histo1D(("h_sf_hits_hadronic", "SciFi Hits (Hadronic);SciFi Hits;Weighted Events", 60, 0, 60), "reco_sf_hits", "weight")

    h_veto_hits_sig = df_mc.Filter("cat_id == 1 && c2_one_scifi_trk").Histo1D(("h_veto_hits_signal", "Veto Hits (Signal);Veto Hits;Weighted Events", 16, -0.5, 15.5), "reco_veto_hits", "weight")
    h_us_hits_sig   = df_mc.Filter("cat_id == 1 && c2_one_scifi_trk").Histo1D(("h_us_hits_signal", "US Hits (Signal);US System Hits;Weighted Events", 31, -0.5, 30.5), "reco_us_hits", "weight")

    h_p_sig_pass= df_mc.Filter("cat_id == 1 && is_clean_reco").Histo1D(("h_p_signal_selected", "Selected Clean Muon Momentum;p [GeV/c];Weighted Muons", 100, 0, 3000), "mu_p", "weight")

    # Master stage registry: stage_name -> (cut_expr, short_title, axis_line1, axis_line2)
    # axis_line1 / axis_line2 are the two rows shown on the x-axis; ROOT renders "\n" as a
    # line break for rotated TAxis bin labels, avoiding #splitline rendering artifacts.
    _STAGE_DEFS = {
        "0. Detector Active":       ("truth.is_in_acceptance",  "0. Active Detector",              "0. Active",             "Detector Hits"),
        "1. EventHeader.isIP1()":   ("c1_ip1",                  "1. EventHeader.isIP1()",           "1. EventHeader",        ".isIP1()"),
        "2. N(SciFi Track) == 1":   ("c2_one_scifi_trk",        "2. SciFi Track == 1",              "2. SciFi Track",        "Multiplicity == 1"),
        "3. DS Track (Type 13)":    ("c3_ds_track",             f"3. DS Track (Type {args.ds_track_type})", "3. DS Track",   f"(Type {args.ds_track_type})"),

        "4. Track Fit Quality":     ("c4_track_quality",        f"4. chi2/ndf <= {args.chi2_max_scifi:g}", "4. Track Fit",  f"chi2/ndf <= {args.chi2_max_scifi:g}"),
        "5. Angular Slope Cut":     ("c5_angular_slope",        f"5. Slope < {args.max_slope:g} rad", "5. Slope Cut",        f"< {args.max_slope:g} rad"),
        "6. Fiducial Plane (z=430)":("c6_fiducial_430",         f"6. Fiducial (z={args.z_match:g} cm)", "6. Fiducial",       f"(z={args.z_match:g} cm)"),
        "7. DS Match at z=430":     ("c7_ds_match",             f"7. DS Match \u0394R \u2264 {args.pos_match_max:g} cm", "7. DS Match", f"\u0394R \u2264 {args.pos_match_max:g} cm"),
    }

    cutflow_stages = [
        (name, _STAGE_DEFS[name][0])
        for name in _STAGE_DEFS
    ]

    # Derived from the active cutflow_stages — always in sync, no parallel list to maintain
    stage_short_titles = [_STAGE_DEFS[name][1] for name, _ in cutflow_stages]
    # Two-row axis labels: line1 + newline + line2, rendered correctly by ROOT on rotated bins
    stage_axis_labels  = [
        f"{_STAGE_DEFS[name][2]}\n{_STAGE_DEFS[name][3]}"
        for name, _ in cutflow_stages
    ]


    stage_hist_ptrs_w = {}
    stage_hist_ptrs_r = {}
    stage_sf_mc_tot_ptrs = []
    stage_sf_mc_sig_ptrs = []
    stage_sf_mc_em_ptrs  = []
    stage_sf_mc_had_ptrs = []
    stage_sf_mc_raw_sig_ptrs = []

    stage_ds_mc_tot_ptrs = []
    stage_ds_mc_sig_ptrs = []
    stage_ds_mc_em_ptrs  = []
    stage_ds_mc_had_ptrs = []
    stage_ds_mc_raw_sig_ptrs = []

    stage_veto_mc_tot_ptrs = []
    stage_veto_mc_sig_ptrs = []
    stage_veto_mc_em_ptrs  = []
    stage_veto_mc_had_ptrs = []
    stage_veto_mc_raw_sig_ptrs = []

    stage_us_mc_tot_ptrs = []
    stage_us_mc_sig_ptrs = []
    stage_us_mc_em_ptrs  = []
    stage_us_mc_had_ptrs = []
    stage_us_mc_raw_sig_ptrs = []

    stage_energy_mc_sig_ptrs = []
    stage_energy_mc_all_ptrs = []
    stage_energy_mc_raw_sig_ptrs = []

    for s_idx, (stage_name, cut_expr) in enumerate(cutflow_stages):
        df_s_mc = df_mc.Filter(cut_expr)
        s_tag = f"s{s_idx}"
        stage_hist_ptrs_w[stage_name] = df_s_mc.Histo1D((f"h_cut_w_{stage_name[:2]}", f"Stage: {stage_name};Category;Weighted Yield", 7, -0.5, 6.5), "cat_id", "weight")
        stage_hist_ptrs_r[stage_name] = df_s_mc.Histo1D((f"h_cut_r_{stage_name[:2]}", f"Stage: {stage_name};Category;Raw Events", 7, -0.5, 6.5), "cat_id")

        # SciFi Hits (0 to 70)
        stage_sf_mc_tot_ptrs.append(df_s_mc.Histo1D((f"h_sf_hits_mc_tot_{s_tag}", f"SciFi Hits - {stage_name} (MC Total);SciFi Hits;Events", 71, -0.5, 70.5), "reco_sf_hits", "scaled_weight"))
        stage_sf_mc_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_sf_hits_mc_sig_{s_tag}", f"SciFi Hits - {stage_name} (Signal);SciFi Hits;Events", 71, -0.5, 70.5), "reco_sf_hits", "scaled_weight"))
        stage_sf_mc_em_ptrs.append(df_s_mc.Filter("cat_id == 2").Histo1D((f"h_sf_hits_mc_em_{s_tag}", f"SciFi Hits - {stage_name} (EM Cascade);SciFi Hits;Events", 71, -0.5, 70.5), "reco_sf_hits", "scaled_weight"))
        stage_sf_mc_had_ptrs.append(df_s_mc.Filter("cat_id == 3").Histo1D((f"h_sf_hits_mc_had_{s_tag}", f"SciFi Hits - {stage_name} (Hadronic);SciFi Hits;Events", 71, -0.5, 70.5), "reco_sf_hits", "scaled_weight"))
        stage_sf_mc_raw_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_sf_hits_mc_raw_sig_{s_tag}", f"SciFi Hits Raw - {stage_name} (Signal);SciFi Hits;Raw Events", 71, -0.5, 70.5), "reco_sf_hits"))

        # DS Hits (0 to 35)
        stage_ds_mc_tot_ptrs.append(df_s_mc.Histo1D((f"h_ds_hits_mc_tot_{s_tag}", f"DS Hits - {stage_name} (MC Total);DS System Hits;Events", 36, -0.5, 35.5), "reco_ds_hits", "scaled_weight"))
        stage_ds_mc_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_ds_hits_mc_sig_{s_tag}", f"DS Hits - {stage_name} (Signal);DS System Hits;Events", 36, -0.5, 35.5), "reco_ds_hits", "scaled_weight"))
        stage_ds_mc_em_ptrs.append(df_s_mc.Filter("cat_id == 2").Histo1D((f"h_ds_hits_mc_em_{s_tag}", f"DS Hits - {stage_name} (EM Cascade);DS System Hits;Events", 36, -0.5, 35.5), "reco_ds_hits", "scaled_weight"))
        stage_ds_mc_had_ptrs.append(df_s_mc.Filter("cat_id == 3").Histo1D((f"h_ds_hits_mc_had_{s_tag}", f"DS Hits - {stage_name} (Hadronic);DS System Hits;Events", 36, -0.5, 35.5), "reco_ds_hits", "scaled_weight"))
        stage_ds_mc_raw_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_ds_hits_mc_raw_sig_{s_tag}", f"DS Hits Raw - {stage_name} (Signal);DS System Hits;Raw Events", 36, -0.5, 35.5), "reco_ds_hits"))

        # Veto Hits (0 to 15)
        stage_veto_mc_tot_ptrs.append(df_s_mc.Histo1D((f"h_veto_hits_mc_tot_{s_tag}", f"Veto Hits - {stage_name} (MC Total);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits", "scaled_weight"))
        stage_veto_mc_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_veto_hits_mc_sig_{s_tag}", f"Veto Hits - {stage_name} (Signal);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits", "scaled_weight"))
        stage_veto_mc_em_ptrs.append(df_s_mc.Filter("cat_id == 2").Histo1D((f"h_veto_hits_mc_em_{s_tag}", f"Veto Hits - {stage_name} (EM Cascade);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits", "scaled_weight"))
        stage_veto_mc_had_ptrs.append(df_s_mc.Filter("cat_id == 3").Histo1D((f"h_veto_hits_mc_had_{s_tag}", f"Veto Hits - {stage_name} (Hadronic);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits", "scaled_weight"))
        stage_veto_mc_raw_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_veto_hits_mc_raw_sig_{s_tag}", f"Veto Hits Raw - {stage_name} (Signal);Veto Hits;Raw Events", 16, -0.5, 15.5), "reco_veto_hits"))

        # US Hits (0 to 30)
        stage_us_mc_tot_ptrs.append(df_s_mc.Histo1D((f"h_us_hits_mc_tot_{s_tag}", f"US Hits - {stage_name} (MC Total);US System Hits;Events", 31, -0.5, 30.5), "reco_us_hits", "scaled_weight"))
        stage_us_mc_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_us_hits_mc_sig_{s_tag}", f"US Hits - {stage_name} (Signal);US System Hits;Events", 31, -0.5, 30.5), "reco_us_hits", "scaled_weight"))
        stage_us_mc_em_ptrs.append(df_s_mc.Filter("cat_id == 2").Histo1D((f"h_us_hits_mc_em_{s_tag}", f"US Hits - {stage_name} (EM Cascade);US System Hits;Events", 31, -0.5, 30.5), "reco_us_hits", "scaled_weight"))
        stage_us_mc_had_ptrs.append(df_s_mc.Filter("cat_id == 3").Histo1D((f"h_us_hits_mc_had_{s_tag}", f"US Hits - {stage_name} (Hadronic);US System Hits;Events", 31, -0.5, 30.5), "reco_us_hits", "scaled_weight"))
        stage_us_mc_raw_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_us_hits_mc_raw_sig_{s_tag}", f"US Hits Raw - {stage_name} (Signal);US System Hits;Raw Events", 31, -0.5, 30.5), "reco_us_hits"))

        # Muon Energy (0 to 2000 GeV, 100 bins)
        stage_energy_mc_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_muon_energy_sig_{s_tag}", f"Passing Muon Energy - {stage_name} (Signal);E_{{#mu}} [GeV];Events / 20 GeV", 100, 0.0, 2000.0), "mu_energy", "scaled_weight"))
        stage_energy_mc_all_ptrs.append(df_s_mc.Filter("truth.primary_muon_p > 0").Histo1D((f"h_muon_energy_all_{s_tag}", f"Passing Muon Energy - {stage_name} (All Passing Muons);E_{{#mu}} [GeV];Events / 20 GeV", 100, 0.0, 2000.0), "mu_energy", "scaled_weight"))
        stage_energy_mc_raw_sig_ptrs.append(df_s_mc.Filter("cat_id == 1").Histo1D((f"h_muon_energy_raw_sig_{s_tag}", f"Passing Muon Energy Raw - {stage_name} (Signal);E_{{#mu}} [GeV];Raw Events / 20 GeV", 100, 0.0, 2000.0), "mu_energy"))

    count_all_mc = df_mc.Count()

    # 7. Setup Collision Data Graph & Histograms (if Data provided)
    data_count_ptrs = {}
    stage_sf_data_ptrs = []
    stage_ds_data_ptrs = []
    stage_veto_data_ptrs = []
    stage_us_data_ptrs = []
    h_chi2_data_ptr = None
    h_sf_hits_data_ptr = None
    h_ds_hits_data_ptr = None
    h_veto_hits_data_ptr = None
    h_us_hits_data_ptr = None

    if args.input_data:
        df_data = dm_data.rdf()
        if args.max_events > 0:
            if ROOT.IsImplicitMTEnabled():
                df_data = df_data.Filter(f"rdfentry_ < {args.max_events}")
            else:
                df_data = df_data.Range(args.max_events)

        cols_data = set(str(c) for c in df_data.GetColumnNames())
        track_branch_data = "Reco_MuonTracks" if "Reco_MuonTracks" in cols_data else "fittedTracks"
        sf_branch_data = "Digi_ScifiHits" if "Digi_ScifiHits" in cols_data else "Digits_Scifi"
        if "Digi_MuFilterHits" in cols_data:
            mf_branch_data = "Digi_MuFilterHits"
        elif "Digi_MuFilterHit" in cols_data:
            mf_branch_data = "Digi_MuFilterHit"
        else:
            mf_branch_data = "Digits_MuFilter"

        if args.require_ip1:
            header_branch_data = "EventHeader." if "EventHeader." in cols_data else "EventHeader"
            if header_branch_data in cols_data:
                ROOT.gInterpreter.ProcessLine("""
#ifndef SND_IP1_FILTER_DEFINED
#define SND_IP1_FILTER_DEFINED
#include "SNDLHCEventHeader.h"
struct IP1Filter {
    bool operator()(const SNDLHCEventHeader& h) const {
        return const_cast<SNDLHCEventHeader&>(h).isIP1();
    }
};
#endif
""")
                ip1_checker = ROOT.IP1Filter()
                df_data = df_data.Define("pass_is_ip1", ip1_checker, [header_branch_data])
                print(f"  * Configured EventHeader.isIP1() selection for collision data")
            else:
                print("[!] Warning: EventHeader branch not found in collision data; defaulting pass_is_ip1 to true")
                df_data = df_data.Define("pass_is_ip1", "true")
        else:
            df_data = df_data.Define("pass_is_ip1", "true")

        df_data = (
            df_data
            .Define("reco", calib_proc, [track_branch_data, sf_branch_data, mf_branch_data])
            .Define("reco_chi2_ndf", "reco.track_chi2_ndf")
            .Define("reco_sf_hits",  "reco.scifi_nhits")
            .Define("reco_ds_hits",  "reco.ds_nhits")
            .Define("reco_veto_hits", "reco.veto_nhits")
            .Define("reco_us_hits",   "reco.us_nhits")
            .Define("c1_ip1",           "pass_is_ip1")
            .Define("c2_one_scifi_trk", "c1_ip1 && reco.pass_cut_single_scifi_track")
            .Define("c3_ds_track",      "c2_one_scifi_trk && reco.pass_cut_ds_track")
            .Define("c4_track_quality", "c3_ds_track && reco.pass_cut_chi2")
            .Define("c5_angular_slope", "c4_track_quality && reco.pass_cut_slope")
            .Define("c6_fiducial_430",  "c5_angular_slope && reco.pass_cut_fiducial_430")
            .Define("c7_ds_match",      "c6_fiducial_430 && reco.pass_cut_ds_match")
        )

        data_stage_cuts = [
            ("0. Detector Active", ""),
            ("1. EventHeader.isIP1()", "c1_ip1"),
            ("2. N(SciFi Track) == 1", "c2_one_scifi_trk"),
            ("3. DS Track (Type 13)", "c3_ds_track"),
            ("4. Track Fit Quality", "c4_track_quality"),
            ("5. Angular Slope Cut", "c5_angular_slope"),
            ("6. Fiducial Plane (z=430)", "c6_fiducial_430"),
            ("7. DS Match at z=430", "c7_ds_match"),
        ]

        for s_idx, (s_name, c_expr) in enumerate(data_stage_cuts):
            df_s_data = df_data.Filter(c_expr) if c_expr else df_data
            s_tag = f"s{s_idx}"
            data_count_ptrs[s_name] = df_s_data.Count()
            stage_sf_data_ptrs.append(df_s_data.Histo1D((f"h_sf_hits_data_{s_tag}", f"SciFi Hits - {s_name} (Data);SciFi Hits;Events", 71, -0.5, 70.5), "reco_sf_hits"))
            stage_ds_data_ptrs.append(df_s_data.Histo1D((f"h_ds_hits_data_{s_tag}", f"DS Hits - {s_name} (Data);DS System Hits;Events", 36, -0.5, 35.5), "reco_ds_hits"))
            stage_veto_data_ptrs.append(df_s_data.Histo1D((f"h_veto_hits_data_{s_tag}", f"Veto Hits - {s_name} (Data);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits"))
            stage_us_data_ptrs.append(df_s_data.Histo1D((f"h_us_hits_data_{s_tag}", f"US Hits - {s_name} (Data);US System Hits;Events", 31, -0.5, 30.5), "reco_us_hits"))

        h_chi2_data_ptr = df_data.Filter("c2_one_scifi_trk").Histo1D(("h_chi2_data", "Track #chi^{2}/ndf (Data);#chi^{2}/ndf;Events", 100, 0, 20), "reco_chi2_ndf")
        h_sf_hits_data_ptr = df_data.Filter("c2_one_scifi_trk").Histo1D(("h_sf_hits_data", "SciFi Hits (Data);SciFi Hits;Events", 60, 0, 60), "reco_sf_hits")
        h_ds_hits_data_ptr = df_data.Filter("c2_one_scifi_trk").Histo1D(("h_ds_hits_data", "DS Hits (Data);DS Hits;Events", 40, 0, 40), "reco_ds_hits")
        h_veto_hits_data_ptr = df_data.Filter("c2_one_scifi_trk").Histo1D(("h_veto_hits_data", "Veto Hits (Data);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits")
        h_us_hits_data_ptr = df_data.Filter("c2_one_scifi_trk").Histo1D(("h_us_hits_data", "US Hits (Data);US Hits;Events", 31, -0.5, 30.5), "reco_us_hits")

    # 7b. Setup Trimuon Monte Carlo Graph & Histograms (if Trimuon MC enabled & provided)
    tri_files = []
    df_tri = None
    scale_tri = 1.0
    lumi_mc_tri_eff = args.lumi_mc_tri
    stage_tri_w_ptrs = {}
    stage_tri_r_ptrs = {}
    stage_tri_hist_ptrs_w = {}
    stage_tri_hist_ptrs_r = {}
    h_tri_cat_all_w = None
    h_tri_cat_all_r = None
    h_tri_cat_pass_w = None
    h_tri_cat_pass_r = None

    stage_sf_tri_tot_ptrs = []
    stage_sf_tri_sig_ptrs = []
    stage_sf_tri_multimu_ptrs = []
    stage_sf_tri_raw_ptrs = []

    stage_ds_tri_tot_ptrs = []
    stage_ds_tri_sig_ptrs = []
    stage_ds_tri_multimu_ptrs = []
    stage_ds_tri_raw_ptrs = []

    stage_veto_tri_tot_ptrs = []
    stage_veto_tri_sig_ptrs = []
    stage_veto_tri_multimu_ptrs = []
    stage_veto_tri_raw_ptrs = []

    stage_us_tri_tot_ptrs = []
    stage_us_tri_sig_ptrs = []
    stage_us_tri_multimu_ptrs = []
    stage_us_tri_raw_ptrs = []

    h_chi2_tri_ptr = None
    h_sf_hits_tri_ptr = None
    h_ds_hits_tri_ptr = None
    h_veto_hits_tri_ptr = None
    h_us_hits_tri_ptr = None
    count_all_tri = None

    if args.enable_tri and args.input_tri and str(args.input_tri).lower() not in ("none", ""):
        if any(c in args.input_tri for c in ("*", "?", "[", "]")):
            tri_files = sorted(glob.glob(args.input_tri))
        elif os.path.exists(args.input_tri):
            tri_files = [args.input_tri]

        if not tri_files:
            print(f"[!] Warning: No trimuon files found matching '{args.input_tri}'. Skipping Trimuon MC.")
        else:
            print(f"\n[*] Initializing Trimuon Monte Carlo ({len(tri_files)} files found)...")
            lumi_mc_tri_eff = args.lumi_mc_tri * (len(tri_files) / 100.0) if len(tri_files) <= 100 else args.lumi_mc_tri
            if args.scale_tri is not None and args.scale_tri > 0:
                scale_tri = args.scale_tri
                print(f"  * Manual Trimuon scale factor override: {scale_tri:.6e}")
            elif data_lumi > 0:
                scale_tri = data_lumi / lumi_mc_tri_eff if lumi_mc_tri_eff > 0 else 1.0
                print(f"  * Trimuon Scaling to Data Lumi: {scale_tri:.6e}  [L_data ({data_lumi:.6e} fb^-1) / L_MC ({lumi_mc_tri_eff:.6e} fb^-1)]")
            else:
                scale_tri = 1.0
                print(f"  * Trimuon Scale Factor: 1.0 (Dataset L_MC = {lumi_mc_tri_eff:.4f} fb^-1)")

            v_tri = ROOT.std.vector('string')()
            for f in tri_files:
                v_tri.push_back(f)
            df_tri = ROOT.RDataFrame(args.tree_tri, v_tri)
            if args.max_events > 0:
                if ROOT.IsImplicitMTEnabled():
                    df_tri = df_tri.Filter(f"rdfentry_ < {args.max_events}")
                else:
                    df_tri = df_tri.Range(args.max_events)

            cols_tri = set(str(c) for c in df_tri.GetColumnNames())
            track_branch_tri = "Reco_MuonTracks" if "Reco_MuonTracks" in cols_tri else "fittedTracks"
            sf_branch_tri = "Digi_ScifiHits" if "Digi_ScifiHits" in cols_tri else "Digits_Scifi"
            if "Digi_MuFilterHits" in cols_tri:
                mf_branch_tri = "Digi_MuFilterHits"
            elif "Digi_MuFilterHit" in cols_tri:
                mf_branch_tri = "Digi_MuFilterHit"
            else:
                mf_branch_tri = "Digits_MuFilter"

            truth_proc_tri = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)

            df_tri = (
                df_tri
                .Define("truth", truth_proc_tri, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
                .Define("reco", calib_proc, [track_branch_tri, sf_branch_tri, mf_branch_tri])
                .Define("weight", "truth.mc_weight")
                .Define("cat_id", "truth.category_id")
                .Define("is_signal", "truth.is_signal")
                .Define("is_clean_reco", "reco.is_clean")
                .Define("reco_chi2_ndf", "reco.track_chi2_ndf")
                .Define("reco_sf_hits",  "reco.scifi_nhits")
                .Define("reco_ds_hits",  "reco.ds_nhits")
                .Define("reco_veto_hits", "reco.veto_nhits")
                .Define("reco_us_hits",   "reco.us_nhits")
                .Define("scaled_weight", f"weight * {scale_tri:.10e}")
                .Define("mu_p", "truth.primary_muon_p")
                .Define("mu_energy", "truth.primary_muon_p > 0 ? std::sqrt(truth.primary_muon_p*truth.primary_muon_p + 0.105658*0.105658) : 0.0")
                .Define("c1_ip1",           "truth.is_in_acceptance")
                .Define("c2_one_scifi_trk", "c1_ip1 && reco.pass_cut_single_scifi_track")
                .Define("c3_ds_track",      "c2_one_scifi_trk && reco.pass_cut_ds_track")
                .Define("c4_track_quality", "c3_ds_track && reco.pass_cut_chi2")
                .Define("c5_angular_slope", "c4_track_quality && reco.pass_cut_slope")
                .Define("c6_fiducial_430",  "c5_angular_slope && reco.pass_cut_fiducial_430")
                .Define("c7_ds_match",      "c6_fiducial_430 && reco.pass_cut_ds_match")
            )

            h_tri_cat_all_w  = df_tri.Histo1D(("h_tri_cat_all_weighted", "All Trimuon Events by Category (Weighted);Category ID;Weighted Yield", 7, -0.5, 6.5), "cat_id", "weight")
            h_tri_cat_all_r  = df_tri.Histo1D(("h_tri_cat_all_raw", "All Trimuon Events by Category (Raw);Category ID;Raw Events", 7, -0.5, 6.5), "cat_id")
            h_tri_cat_pass_w = df_tri.Filter("c7_ds_match").Histo1D(("h_tri_cat_pass_weighted", "Selected Trimuon Events (Weighted);Category ID;Weighted Yield", 7, -0.5, 6.5), "cat_id", "weight")
            h_tri_cat_pass_r = df_tri.Filter("c7_ds_match").Histo1D(("h_tri_cat_pass_raw", "Selected Trimuon Events (Raw);Category ID;Raw Events", 7, -0.5, 6.5), "cat_id")

            count_all_tri = df_tri.Count()

            for s_idx, (s_name, c_expr) in enumerate(cutflow_stages):
                df_s_tri = df_tri.Filter(c_expr) if c_expr else df_tri
                s_tag = f"s{s_idx}"
                stage_tri_w_ptrs[s_name] = df_s_tri.Sum("scaled_weight")
                stage_tri_r_ptrs[s_name] = df_s_tri.Count()
                stage_tri_hist_ptrs_w[s_name] = df_s_tri.Histo1D((f"h_tri_cut_w_{s_name[:2]}", f"Stage: {s_name} (Trimuon MC);Category;Weighted Yield", 7, -0.5, 6.5), "cat_id", "weight")
                stage_tri_hist_ptrs_r[s_name] = df_s_tri.Histo1D((f"h_tri_cut_r_{s_name[:2]}", f"Stage: {s_name} (Trimuon MC);Category;Raw Events", 7, -0.5, 6.5), "cat_id")

                # SciFi Hits (0 to 70)
                stage_sf_tri_tot_ptrs.append(df_s_tri.Histo1D((f"h_sf_hits_tri_tot_{s_tag}", f"SciFi Hits - {s_name} (Trimuon MC Total);SciFi Hits;Events", 71, -0.5, 70.5), "reco_sf_hits", "scaled_weight"))
                stage_sf_tri_sig_ptrs.append(df_s_tri.Filter("cat_id == 1").Histo1D((f"h_sf_hits_tri_sig_{s_tag}", f"SciFi Hits - {s_name} (Clean Single #mu);SciFi Hits;Events", 71, -0.5, 70.5), "reco_sf_hits", "scaled_weight"))
                stage_sf_tri_multimu_ptrs.append(df_s_tri.Filter("cat_id == 4").Histo1D((f"h_sf_hits_tri_multimu_{s_tag}", f"SciFi Hits - {s_name} (Multi-#mu Bundle);SciFi Hits;Events", 71, -0.5, 70.5), "reco_sf_hits", "scaled_weight"))
                stage_sf_tri_raw_ptrs.append(df_s_tri.Histo1D((f"h_sf_hits_tri_raw_{s_tag}", f"SciFi Hits Raw - {s_name} (Trimuon MC Total);SciFi Hits;Raw Events", 71, -0.5, 70.5), "reco_sf_hits"))

                # DS Hits (0 to 35)
                stage_ds_tri_tot_ptrs.append(df_s_tri.Histo1D((f"h_ds_hits_tri_tot_{s_tag}", f"DS Hits - {s_name} (Trimuon MC Total);DS System Hits;Events", 36, -0.5, 35.5), "reco_ds_hits", "scaled_weight"))
                stage_ds_tri_sig_ptrs.append(df_s_tri.Filter("cat_id == 1").Histo1D((f"h_ds_hits_tri_sig_{s_tag}", f"DS Hits - {s_name} (Clean Single #mu);DS System Hits;Events", 36, -0.5, 35.5), "reco_ds_hits", "scaled_weight"))
                stage_ds_tri_multimu_ptrs.append(df_s_tri.Filter("cat_id == 4").Histo1D((f"h_ds_hits_tri_multimu_{s_tag}", f"DS Hits - {s_name} (Multi-#mu Bundle);DS System Hits;Events", 36, -0.5, 35.5), "reco_ds_hits", "scaled_weight"))
                stage_ds_tri_raw_ptrs.append(df_s_tri.Histo1D((f"h_ds_hits_tri_raw_{s_tag}", f"DS Hits Raw - {s_name} (Trimuon MC Total);DS System Hits;Raw Events", 36, -0.5, 35.5), "reco_ds_hits"))

                # Veto Hits (0 to 15)
                stage_veto_tri_tot_ptrs.append(df_s_tri.Histo1D((f"h_veto_hits_tri_tot_{s_tag}", f"Veto Hits - {s_name} (Trimuon MC Total);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits", "scaled_weight"))
                stage_veto_tri_sig_ptrs.append(df_s_tri.Filter("cat_id == 1").Histo1D((f"h_veto_hits_tri_sig_{s_tag}", f"Veto Hits - {s_name} (Clean Single #mu);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits", "scaled_weight"))
                stage_veto_tri_multimu_ptrs.append(df_s_tri.Filter("cat_id == 4").Histo1D((f"h_veto_hits_tri_multimu_{s_tag}", f"Veto Hits - {s_name} (Multi-#mu Bundle);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits", "scaled_weight"))
                stage_veto_tri_raw_ptrs.append(df_s_tri.Histo1D((f"h_veto_hits_tri_raw_{s_tag}", f"Veto Hits Raw - {s_name} (Trimuon MC Total);Veto Hits;Raw Events", 16, -0.5, 15.5), "reco_veto_hits"))

                # US Hits (0 to 30)
                stage_us_tri_tot_ptrs.append(df_s_tri.Histo1D((f"h_us_hits_tri_tot_{s_tag}", f"US Hits - {s_name} (Trimuon MC Total);US System Hits;Events", 31, -0.5, 30.5), "reco_us_hits", "scaled_weight"))
                stage_us_tri_sig_ptrs.append(df_s_tri.Filter("cat_id == 1").Histo1D((f"h_us_hits_tri_sig_{s_tag}", f"US Hits - {s_name} (Clean Single #mu);US System Hits;Events", 31, -0.5, 30.5), "reco_us_hits", "scaled_weight"))
                stage_us_tri_multimu_ptrs.append(df_s_tri.Filter("cat_id == 4").Histo1D((f"h_us_hits_tri_multimu_{s_tag}", f"US Hits - {s_name} (Multi-#mu Bundle);US System Hits;Events", 31, -0.5, 30.5), "reco_us_hits", "scaled_weight"))
                stage_us_tri_raw_ptrs.append(df_s_tri.Histo1D((f"h_us_hits_tri_raw_{s_tag}", f"US Hits Raw - {s_name} (Trimuon MC Total);US System Hits;Raw Events", 31, -0.5, 30.5), "reco_us_hits"))

            h_chi2_tri_ptr = df_tri.Filter("c2_one_scifi_trk").Histo1D(("h_chi2_trimuon", "Track #chi^{2}/ndf (Trimuon MC);#chi^{2}/ndf;Events", 100, 0, 20), "reco_chi2_ndf", "scaled_weight")
            h_sf_hits_tri_ptr = df_tri.Filter("c2_one_scifi_trk").Histo1D(("h_sf_hits_trimuon", "SciFi Hits (Trimuon MC);SciFi Hits;Events", 60, 0, 60), "reco_sf_hits", "scaled_weight")
            h_ds_hits_tri_ptr = df_tri.Filter("c2_one_scifi_trk").Histo1D(("h_ds_hits_trimuon", "DS Hits (Trimuon MC);DS Hits;Events", 40, 0, 40), "reco_ds_hits", "scaled_weight")
            h_veto_hits_tri_ptr = df_tri.Filter("c2_one_scifi_trk").Histo1D(("h_veto_hits_trimuon", "Veto Hits (Trimuon MC);Veto Hits;Events", 16, -0.5, 15.5), "reco_veto_hits", "scaled_weight")
            h_us_hits_tri_ptr = df_tri.Filter("c2_one_scifi_trk").Histo1D(("h_us_hits_trimuon", "US Hits (Trimuon MC);US Hits;Events", 31, -0.5, 30.5), "reco_us_hits", "scaled_weight")

    # 8. Execute Computational Graph across worker threads
    print("[*] Executing RDataFrame computational graph across worker threads...")
    t_loop = time.time()
    n_processed_mc = count_all_mc.GetValue()
    print(f"[✓] PMU MC Event loop completed in {time.time() - t_loop:.2f} s ({n_processed_mc / (time.time() - t_loop):.0f} ev/s)")

    if count_all_tri is not None:
        t_tri = time.time()
        n_processed_tri = count_all_tri.GetValue()
        print(f"[✓] Trimuon MC Event loop completed in {time.time() - t_tri:.2f} s ({n_processed_tri / (time.time() - t_tri):.0f} ev/s)")

    # 9. Extract MC Results
    h_all_w_val = h_cat_all_w.GetValue()
    h_all_r_val = h_cat_all_r.GetValue()
    h_pass_w_val = h_cat_pass_w.GetValue()
    h_pass_r_val = h_cat_pass_r.GetValue()

    cat_w_all = {i: float(h_all_w_val.GetBinContent(i + 1)) * pmu_scale_factor for i in range(7)}
    cat_r_all = {i: int(h_all_r_val.GetBinContent(i + 1)) for i in range(7)}
    cat_w_pass = {i: float(h_pass_w_val.GetBinContent(i + 1)) * pmu_scale_factor for i in range(7)}
    cat_r_pass = {i: int(h_pass_r_val.GetBinContent(i + 1)) for i in range(7)}

    cutflow_matrix_w = {}
    cutflow_matrix_w_err = {}
    cutflow_matrix_r = {}
    cutflow_matrix_r_err = {}
    for stage_name in stage_hist_ptrs_w.keys():
        hw = stage_hist_ptrs_w[stage_name].GetValue()
        hr = stage_hist_ptrs_r[stage_name].GetValue()
        cutflow_matrix_w[stage_name] = {i: float(hw.GetBinContent(i + 1)) * pmu_scale_factor for i in range(7)}
        cutflow_matrix_w_err[stage_name] = {i: float(hw.GetBinError(i + 1)) * pmu_scale_factor for i in range(7)}
        cutflow_matrix_r[stage_name] = {i: int(hr.GetBinContent(i + 1)) for i in range(7)}
        cutflow_matrix_r_err[stage_name] = {i: float(hr.GetBinError(i + 1)) for i in range(7)}

    # Extract Stage Hits and Energy Histograms
    stage_sf_mc_tot = [ptr.GetValue() for ptr in stage_sf_mc_tot_ptrs]
    stage_sf_mc_sig = [ptr.GetValue() for ptr in stage_sf_mc_sig_ptrs]
    stage_sf_mc_em  = [ptr.GetValue() for ptr in stage_sf_mc_em_ptrs]
    stage_sf_mc_had = [ptr.GetValue() for ptr in stage_sf_mc_had_ptrs]
    stage_sf_mc_raw_sig = [ptr.GetValue() for ptr in stage_sf_mc_raw_sig_ptrs]

    stage_ds_mc_tot = [ptr.GetValue() for ptr in stage_ds_mc_tot_ptrs]
    stage_ds_mc_sig = [ptr.GetValue() for ptr in stage_ds_mc_sig_ptrs]
    stage_ds_mc_em  = [ptr.GetValue() for ptr in stage_ds_mc_em_ptrs]
    stage_ds_mc_had = [ptr.GetValue() for ptr in stage_ds_mc_had_ptrs]
    stage_ds_mc_raw_sig = [ptr.GetValue() for ptr in stage_ds_mc_raw_sig_ptrs]

    stage_veto_mc_tot = [ptr.GetValue() for ptr in stage_veto_mc_tot_ptrs]
    stage_veto_mc_sig = [ptr.GetValue() for ptr in stage_veto_mc_sig_ptrs]
    stage_veto_mc_em  = [ptr.GetValue() for ptr in stage_veto_mc_em_ptrs]
    stage_veto_mc_had = [ptr.GetValue() for ptr in stage_veto_mc_had_ptrs]
    stage_veto_mc_raw_sig = [ptr.GetValue() for ptr in stage_veto_mc_raw_sig_ptrs]

    stage_us_mc_tot = [ptr.GetValue() for ptr in stage_us_mc_tot_ptrs]
    stage_us_mc_sig = [ptr.GetValue() for ptr in stage_us_mc_sig_ptrs]
    stage_us_mc_em  = [ptr.GetValue() for ptr in stage_us_mc_em_ptrs]
    stage_us_mc_had = [ptr.GetValue() for ptr in stage_us_mc_had_ptrs]
    stage_us_mc_raw_sig = [ptr.GetValue() for ptr in stage_us_mc_raw_sig_ptrs]

    stage_energy_mc_sig = [ptr.GetValue() for ptr in stage_energy_mc_sig_ptrs]
    stage_energy_mc_all = [ptr.GetValue() for ptr in stage_energy_mc_all_ptrs]
    stage_energy_mc_raw_sig = [ptr.GetValue() for ptr in stage_energy_mc_raw_sig_ptrs]

    stage_sf_data   = [scale_data_th1(ptr.GetValue(), args.data_scaling) for ptr in stage_sf_data_ptrs]   if args.input_data else None
    stage_ds_data   = [scale_data_th1(ptr.GetValue(), args.data_scaling) for ptr in stage_ds_data_ptrs]   if args.input_data else None
    stage_veto_data = [scale_data_th1(ptr.GetValue(), args.data_scaling) for ptr in stage_veto_data_ptrs] if args.input_data else None
    stage_us_data   = [scale_data_th1(ptr.GetValue(), args.data_scaling) for ptr in stage_us_data_ptrs]   if args.input_data else None

    # Extract Data Results (if Data provided)
    raw_data_counts = None
    data_counts = None
    h_cutflow_data = None
    data_final_count = None
    if args.input_data:
        raw_data_counts = {s_name: int(ptr.GetValue()) for s_name, ptr in data_count_ptrs.items()}
        data_counts = {s_name: float(raw_data_counts[s_name]) * args.data_scaling for s_name in raw_data_counts}
        data_final_count = data_counts.get(cutflow_stages[-1][0], 0.0)

    # Extract Trimuon Results (if Trimuon MC provided)
    tri_counts_w = {}
    tri_counts_r = {}
    cat_tri_w_all = {}
    cat_tri_r_all = {}
    cat_tri_w_pass = {}
    cat_tri_r_pass = {}
    cutflow_matrix_tri_w = {}
    cutflow_matrix_tri_w_err = {}
    cutflow_matrix_tri_r = {}
    cutflow_matrix_tri_r_err = {}

    stage_sf_tri_tot = None
    stage_sf_tri_sig = None
    stage_sf_tri_multimu = None
    stage_sf_tri_raw = None

    stage_ds_tri_tot = None
    stage_ds_tri_sig = None
    stage_ds_tri_multimu = None
    stage_ds_tri_raw = None

    stage_veto_tri_tot = None
    stage_veto_tri_sig = None
    stage_veto_tri_multimu = None
    stage_veto_tri_raw = None

    stage_us_tri_tot = None
    stage_us_tri_sig = None
    stage_us_tri_multimu = None
    stage_us_tri_raw = None

    h_cutflow_tri_w = None
    h_cutflow_tri_r = None
    c_cutflow_tri_w = None
    c_cutflow_tri_r = None
    c_sf_stages_tri = None
    c_ds_stages_tri = None
    c_veto_stages_tri = None
    c_us_stages_tri = None

    if df_tri is not None:
        tri_counts_w = {s_name: float(ptr.GetValue()) for s_name, ptr in stage_tri_w_ptrs.items()}
        tri_counts_r = {s_name: int(ptr.GetValue()) for s_name, ptr in stage_tri_r_ptrs.items()}

        h_tri_all_w_val = h_tri_cat_all_w.GetValue()
        h_tri_all_r_val = h_tri_cat_all_r.GetValue()
        h_tri_pass_w_val = h_tri_cat_pass_w.GetValue()
        h_tri_pass_r_val = h_tri_cat_pass_r.GetValue()

        cat_tri_w_all = {i: float(h_tri_all_w_val.GetBinContent(i + 1)) * scale_tri for i in range(7)}
        cat_tri_r_all = {i: int(h_tri_all_r_val.GetBinContent(i + 1)) for i in range(7)}
        cat_tri_w_pass = {i: float(h_tri_pass_w_val.GetBinContent(i + 1)) * scale_tri for i in range(7)}
        cat_tri_r_pass = {i: int(h_tri_pass_r_val.GetBinContent(i + 1)) for i in range(7)}

        for stage_name in stage_tri_hist_ptrs_w.keys():
            hw = stage_tri_hist_ptrs_w[stage_name].GetValue()
            hr = stage_tri_hist_ptrs_r[stage_name].GetValue()
            cutflow_matrix_tri_w[stage_name] = {i: float(hw.GetBinContent(i + 1)) * scale_tri for i in range(7)}
            cutflow_matrix_tri_w_err[stage_name] = {i: float(hw.GetBinError(i + 1)) * scale_tri for i in range(7)}
            cutflow_matrix_tri_r[stage_name] = {i: int(hr.GetBinContent(i + 1)) for i in range(7)}
            cutflow_matrix_tri_r_err[stage_name] = {i: float(hr.GetBinError(i + 1)) for i in range(7)}

        stage_sf_tri_tot = [ptr.GetValue() for ptr in stage_sf_tri_tot_ptrs]
        stage_sf_tri_sig = [ptr.GetValue() for ptr in stage_sf_tri_sig_ptrs]
        stage_sf_tri_multimu = [ptr.GetValue() for ptr in stage_sf_tri_multimu_ptrs]
        stage_sf_tri_raw = [ptr.GetValue() for ptr in stage_sf_tri_raw_ptrs]

        stage_ds_tri_tot = [ptr.GetValue() for ptr in stage_ds_tri_tot_ptrs]
        stage_ds_tri_sig = [ptr.GetValue() for ptr in stage_ds_tri_sig_ptrs]
        stage_ds_tri_multimu = [ptr.GetValue() for ptr in stage_ds_tri_multimu_ptrs]
        stage_ds_tri_raw = [ptr.GetValue() for ptr in stage_ds_tri_raw_ptrs]

        stage_veto_tri_tot = [ptr.GetValue() for ptr in stage_veto_tri_tot_ptrs]
        stage_veto_tri_sig = [ptr.GetValue() for ptr in stage_veto_tri_sig_ptrs]
        stage_veto_tri_multimu = [ptr.GetValue() for ptr in stage_veto_tri_multimu_ptrs]
        stage_veto_tri_raw = [ptr.GetValue() for ptr in stage_veto_tri_raw_ptrs]

        stage_us_tri_tot = [ptr.GetValue() for ptr in stage_us_tri_tot_ptrs]
        stage_us_tri_sig = [ptr.GetValue() for ptr in stage_us_tri_sig_ptrs]
        stage_us_tri_multimu = [ptr.GetValue() for ptr in stage_us_tri_multimu_ptrs]
        stage_us_tri_raw = [ptr.GetValue() for ptr in stage_us_tri_raw_ptrs]

    # 10. Print Comprehensive Assessment Tables
    print_summary_tables(
        cat_w_all, cat_r_all, cat_w_pass, cat_r_pass, cutflow_matrix_w, cutflow_matrix_r,
        data_counts=data_counts, raw_data_counts=raw_data_counts,
        pmu_scale_factor=pmu_scale_factor, data_meta=data_meta,
        section_offset=0, sample_tag="PASSING MUON MC", print_exposure=True
    )

    if df_tri is not None:
        tri_meta = dict(data_meta) if data_meta else {}
        tri_meta["lumi_str"] = lumi_str if args.input_data else f"{lumi_mc_tri_eff:.4f} fb^-1"
        print_summary_tables(
            cat_tri_w_all, cat_tri_r_all, cat_tri_w_pass, cat_tri_r_pass,
            cutflow_matrix_tri_w, cutflow_matrix_tri_r,
            data_counts=data_counts, raw_data_counts=raw_data_counts,
            pmu_scale_factor=scale_tri, data_meta=tri_meta,
            section_offset=3, sample_tag="TRIMUON MC", print_exposure=False
        )

        print_trimuon_summary_table(
            cutflow_stages, tri_counts_w, tri_counts_r,
            data_counts=data_counts, raw_data_counts=raw_data_counts,
            scale_tri=scale_tri, lumi_str=lumi_str if args.input_data else f"{lumi_mc_tri_eff:.4f} fb^-1",
            table_number=7
        )


    last_stage_name = cutflow_stages[-1][0]
    y_title_w = "Events" if args.input_data else "FLUKA Weighted Yield / 10^{8} p-p"

    hists_cutflow_w = build_cutflow_histograms(
        cutflow_stages, stage_axis_labels, cutflow_matrix_w, cutflow_matrix_w_err,
        is_weighted=True, y_title=y_title_w
    )
    hists_cutflow_r = build_cutflow_histograms(
        cutflow_stages, stage_axis_labels, cutflow_matrix_r, cutflow_matrix_r_err,
        is_weighted=False
    )

    if args.input_data and data_counts is not None:
        h_cutflow_data = ROOT.TH1D("h_cutflow_data", "Collision Data Sequential Cutflow;;Events", len(cutflow_stages), 0.5, len(cutflow_stages) + 0.5)
        ROOT.SetOwnership(h_cutflow_data, False)
        h_cutflow_data.SetStats(0)
        h_cutflow_data.Sumw2()
        for s_idx, (stage_name, _) in enumerate(cutflow_stages, start=1):
            h_cutflow_data.GetXaxis().SetBinLabel(s_idx, stage_axis_labels[s_idx - 1])
            c_raw = float(raw_data_counts.get(stage_name, 0)) if raw_data_counts else float(data_counts.get(stage_name, 0))
            c_val = c_raw * args.data_scaling
            h_cutflow_data.SetBinContent(s_idx, c_val)
            h_cutflow_data.SetBinError(s_idx, math.sqrt(c_raw) * args.data_scaling if c_raw > 0 else 0.0)

        h_cutflow_data.SetLineColor(ROOT.kBlack)
        h_cutflow_data.SetLineWidth(2)
        h_cutflow_data.SetMarkerStyle(20)
        h_cutflow_data.SetMarkerSize(1.2)
        h_cutflow_data.SetMarkerColor(ROOT.kBlack)

    if args.input_data:
        scale_txt = f", Data #times{args.data_scaling:g}" if abs(args.data_scaling - 1.0) > 1e-9 else ""
        header_right_w = f"Data: Run {data_meta['run']} (T = {dur_str}, L_{{int}} = {lumi_str}){scale_txt}"
        leg_header = f"PMU MC Scaled to Lumi ({lumi_str})"
    else:
        header_right_w = "160 #murad, 100#times10^{6} p-p, FLUKA E_{cut} = 10 GeV"
        leg_header = None

    c_superimposed_w = build_superimposed_cutflow_canvas(
        hists_cutflow_w, last_stage_name, cutflow_matrix_w, is_weighted=True,
        canvas_name="c_cutflow_superimposed_weighted",
        h_data=h_cutflow_data, data_final_count=data_final_count, header_right=header_right_w,
        legend_header=leg_header
    )
    c_superimposed_r = build_superimposed_cutflow_canvas(
        hists_cutflow_r, last_stage_name, cutflow_matrix_r, is_weighted=False,
        canvas_name="c_cutflow_superimposed_raw",
        h_data=h_cutflow_data, data_final_count=data_final_count, header_right=header_right_w,
        legend_header=None
    )

    # 12. Build Multi-Stage Hit and Muon Energy Canvases
    c_sf_stages = build_multi_stage_canvas(
        "c_scifi_hits_stages", "SciFi Hit Multiplicity across Cut Stages",
        stage_short_titles, stage_sf_mc_tot, stage_sf_mc_sig, stage_sf_data,
        cut_lines=[float(args.scifi_min), float(args.scifi_max)], x_title="SciFi Hits", log_y=True,
        header_right=header_right_w
    )
    c_ds_stages = build_multi_stage_canvas(
        "c_ds_hits_stages", "Downstream MuFilter Hit Multiplicity across Cut Stages",
        stage_short_titles, stage_ds_mc_tot, stage_ds_mc_sig, stage_ds_data,
        cut_lines=None, x_title="DS System Hits", log_y=True,
        header_right=header_right_w
    )
    c_veto_stages = build_multi_stage_canvas(
        "c_veto_hits_stages", "Veto Hit Multiplicity across Cut Stages",
        stage_short_titles, stage_veto_mc_tot, stage_veto_mc_sig, stage_veto_data,
        cut_lines=None, x_title="Veto Hits", log_y=True,
        header_right=header_right_w
    )
    c_us_stages = build_multi_stage_canvas(
        "c_us_hits_stages", "Upstream MuFilter Hit Multiplicity across Cut Stages",
        stage_short_titles, stage_us_mc_tot, stage_us_mc_sig, stage_us_data,
        cut_lines=None, x_title="US System Hits", log_y=True,
        header_right=header_right_w
    )
    c_energy_stages = build_muon_energy_stages_canvas(
        "c_muon_energy_stages", "Passing Muon Energy Distribution across Cut Stages",
        stage_short_titles, stage_energy_mc_sig, stage_energy_mc_all,
        header_right=header_right_w
    )
    c_energy_overlay = build_energy_overlay_canvas(
        "c_muon_energy_overlay", stage_short_titles, stage_energy_mc_sig,
        header_right=header_right_w
    )

    # 12b. Build Trimuon MC Cutflow & Evolution Canvases (if Trimuon MC enabled & processed)
    if df_tri is not None:
        h_cutflow_tri_w = ROOT.TH1D("h_cutflow_trimuon_weighted", "Trimuon MC Sequential Cutflow (Weighted);;Events", len(cutflow_stages), 0.5, len(cutflow_stages) + 0.5)
        ROOT.SetOwnership(h_cutflow_tri_w, False)
        h_cutflow_tri_w.SetStats(0)
        for s_idx, (s_name, _) in enumerate(cutflow_stages, start=1):
            h_cutflow_tri_w.GetXaxis().SetBinLabel(s_idx, stage_axis_labels[s_idx - 1])
            h_cutflow_tri_w.SetBinContent(s_idx, tri_counts_w.get(s_name, 0.0))

        h_cutflow_tri_r = ROOT.TH1D("h_cutflow_trimuon_raw", "Trimuon MC Sequential Cutflow (Raw);;Raw Events", len(cutflow_stages), 0.5, len(cutflow_stages) + 0.5)
        ROOT.SetOwnership(h_cutflow_tri_r, False)
        h_cutflow_tri_r.SetStats(0)
        for s_idx, (s_name, _) in enumerate(cutflow_stages, start=1):
            h_cutflow_tri_r.GetXaxis().SetBinLabel(s_idx, stage_axis_labels[s_idx - 1])
            h_cutflow_tri_r.SetBinContent(s_idx, float(tri_counts_r.get(s_name, 0)))

        header_right_tri_w = f"Data vs Trimuon MC (L_{{int}} = {lumi_str})" if args.input_data else f"Trimuon MC (L_{{MC}} = {lumi_mc_tri_eff:.4f} fb^{{-1}})"
        leg_header_tri = f"Trimuon MC Scaled to Lumi ({lumi_str})" if args.input_data else None

        hists_cutflow_tri_w = build_cutflow_histograms(
            cutflow_stages, stage_axis_labels, cutflow_matrix_tri_w, cutflow_matrix_tri_w_err,
            is_weighted=True, y_title=y_title_w, name_prefix="trimuon_"
        )
        hists_cutflow_tri_r = build_cutflow_histograms(
            cutflow_stages, stage_axis_labels, cutflow_matrix_tri_r, cutflow_matrix_tri_r_err,
            is_weighted=False, name_prefix="trimuon_"
        )

        c_cutflow_tri_w = build_superimposed_cutflow_canvas(
            hists_cutflow_tri_w, last_stage_name, cutflow_matrix_tri_w,
            is_weighted=True, canvas_name="c_cutflow_trimuon_superimposed_weighted",
            h_data=h_cutflow_data, data_final_count=data_final_count,
            header_right=header_right_tri_w, legend_header=leg_header_tri,
            mc_total_label="Trimuon MC Total"
        )
        c_cutflow_tri_r = build_superimposed_cutflow_canvas(
            hists_cutflow_tri_r, last_stage_name, cutflow_matrix_tri_r,
            is_weighted=False, canvas_name="c_cutflow_trimuon_superimposed_raw",
            h_data=h_cutflow_data, data_final_count=data_final_count,
            header_right=header_right_tri_w, legend_header=None,
            mc_total_label="Trimuon MC Total"
        )

        c_sf_stages_tri = build_multi_stage_canvas(
            "c_scifi_hits_stages_trimuon", "SciFi Hit Multiplicity across Cut Stages (Trimuon MC vs Data)",
            stage_short_titles, stage_sf_tri_tot, stage_sf_tri_sig, stage_sf_data,
            cut_lines=[float(args.scifi_min), float(args.scifi_max)], x_title="SciFi Hits", log_y=True,
            header_right=header_right_tri_w, mc_legend_label="Trimuon MC Total",
            mc_color=ROOT.kMagenta + 2, mc_line_style=1,
            sig_legend_label="Clean #mu (Truth)",
            hists_mc_extra=stage_sf_tri_multimu,
            extra_legend_label="Multi-#mu (Truth)",
            extra_color=ROOT.kTeal + 2, extra_line_style=7
        )
        c_ds_stages_tri = build_multi_stage_canvas(
            "c_ds_hits_stages_trimuon", "Downstream MuFilter Hit Multiplicity across Cut Stages (Trimuon MC vs Data)",
            stage_short_titles, stage_ds_tri_tot, stage_ds_tri_sig, stage_ds_data,
            cut_lines=None, x_title="DS System Hits", log_y=True,
            header_right=header_right_tri_w, mc_legend_label="Trimuon MC Total",
            mc_color=ROOT.kMagenta + 2, mc_line_style=1,
            sig_legend_label="Clean #mu (Truth)",
            hists_mc_extra=stage_ds_tri_multimu,
            extra_legend_label="Multi-#mu (Truth)",
            extra_color=ROOT.kTeal + 2, extra_line_style=7
        )
        c_veto_stages_tri = build_multi_stage_canvas(
            "c_veto_hits_stages_trimuon", "Veto Hit Multiplicity across Cut Stages (Trimuon MC vs Data)",
            stage_short_titles, stage_veto_tri_tot, stage_veto_tri_sig, stage_veto_data,
            cut_lines=None, x_title="Veto Hits", log_y=True,
            header_right=header_right_tri_w, mc_legend_label="Trimuon MC Total",
            mc_color=ROOT.kMagenta + 2, mc_line_style=1,
            sig_legend_label="Clean #mu (Truth)",
            hists_mc_extra=stage_veto_tri_multimu,
            extra_legend_label="Multi-#mu (Truth)",
            extra_color=ROOT.kTeal + 2, extra_line_style=7
        )
        c_us_stages_tri = build_multi_stage_canvas(
            "c_us_hits_stages_trimuon", "Upstream MuFilter Hit Multiplicity across Cut Stages (Trimuon MC vs Data)",
            stage_short_titles, stage_us_tri_tot, stage_us_tri_sig, stage_us_data,
            cut_lines=None, x_title="US System Hits", log_y=True,
            header_right=header_right_tri_w, mc_legend_label="Trimuon MC Total",
            mc_color=ROOT.kMagenta + 2, mc_line_style=1,
            sig_legend_label="Clean #mu (Truth)",
            hists_mc_extra=stage_us_tri_multimu,
            extra_legend_label="Multi-#mu (Truth)",
            extra_color=ROOT.kTeal + 2, extra_line_style=7
        )

    # 13. Save Histograms and Canvases to Master ROOT File (Atomic write)
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
        tmp_path = tmp.name

    print(f"[*] Saving validation histograms, cutflow trees, and superimposed canvases to: {tmp_path}")
    tfile = ROOT.TFile(tmp_path, "RECREATE")

    # Summary directory
    d_summary = get_or_create_dir(tfile, "Summary")
    d_summary.cd()
    h_all_w_val.Write("h_categories_all_weighted")
    h_all_r_val.Write("h_categories_all_raw")
    h_pass_w_val.Write("h_categories_passed_weighted")
    h_pass_r_val.Write("h_categories_passed_raw")
    if df_tri is not None:
        h_tri_all_w_val.Write("h_tri_categories_all_weighted")
        h_tri_all_r_val.Write("h_tri_categories_all_raw")
        h_tri_pass_w_val.Write("h_tri_categories_passed_weighted")
        h_tri_pass_r_val.Write("h_tri_categories_passed_raw")

    # Cutflow directory
    d_cutflow = get_or_create_dir(tfile, "Cutflow")
    d_cutflow.cd()
    c_superimposed_w.Write()
    c_superimposed_r.Write()

    # 2D Cutflow Matrices
    h2_cutflow_w = ROOT.TH2D("h2_cutflow_matrix_weighted", "Sequential Cutflow Matrix (Weighted);Cut Stage;Category;FLUKA Yield",
                            len(cutflow_stages), 0.5, len(cutflow_stages) + 0.5, 6, 0.5, 6.5)
    h2_cutflow_r = ROOT.TH2D("h2_cutflow_matrix_raw", "Sequential Cutflow Matrix (Raw);Cut Stage;Category;Raw Events",
                            len(cutflow_stages), 0.5, len(cutflow_stages) + 0.5, 6, 0.5, 6.5)

    for s_idx, (stage_name, _) in enumerate(cutflow_stages, start=1):
        h2_cutflow_w.GetXaxis().SetBinLabel(s_idx, stage_name)
        h2_cutflow_r.GetXaxis().SetBinLabel(s_idx, stage_name)
        for cat_id in range(1, 7):
            if s_idx == 1:
                h2_cutflow_w.GetYaxis().SetBinLabel(cat_id, CATEGORY_NAMES[cat_id])
                h2_cutflow_r.GetYaxis().SetBinLabel(cat_id, CATEGORY_NAMES[cat_id])
            h2_cutflow_w.SetBinContent(s_idx, cat_id, cutflow_matrix_w[stage_name].get(cat_id, 0.0))
            h2_cutflow_r.SetBinContent(s_idx, cat_id, cutflow_matrix_r[stage_name].get(cat_id, 0))

    h2_cutflow_w.Write()
    h2_cutflow_r.Write()

    if df_tri is not None:
        h2_cutflow_tri_w = ROOT.TH2D("h2_cutflow_matrix_tri_weighted", "Trimuon Sequential Cutflow Matrix (Weighted);Cut Stage;Category;Scaled Yield",
                                     len(cutflow_stages), 0.5, len(cutflow_stages) + 0.5, 6, 0.5, 6.5)
        h2_cutflow_tri_r = ROOT.TH2D("h2_cutflow_matrix_tri_raw", "Trimuon Sequential Cutflow Matrix (Raw);Cut Stage;Category;Raw Events",
                                     len(cutflow_stages), 0.5, len(cutflow_stages) + 0.5, 6, 0.5, 6.5)
        for s_idx, (stage_name, _) in enumerate(cutflow_stages, start=1):
            h2_cutflow_tri_w.GetXaxis().SetBinLabel(s_idx, stage_name)
            h2_cutflow_tri_r.GetXaxis().SetBinLabel(s_idx, stage_name)
            for cat_id in range(1, 7):
                if s_idx == 1:
                    h2_cutflow_tri_w.GetYaxis().SetBinLabel(cat_id, CATEGORY_NAMES[cat_id])
                    h2_cutflow_tri_r.GetYaxis().SetBinLabel(cat_id, CATEGORY_NAMES[cat_id])
                h2_cutflow_tri_w.SetBinContent(s_idx, cat_id, cutflow_matrix_tri_w[stage_name].get(cat_id, 0.0))
                h2_cutflow_tri_r.SetBinContent(s_idx, cat_id, cutflow_matrix_tri_r[stage_name].get(cat_id, 0))
        h2_cutflow_tri_w.Write()
        h2_cutflow_tri_r.Write()

    # 1D Cutflows for each category
    d_cutflow_w = get_or_create_dir(tfile, "Cutflow/Histograms_Weighted")
    d_cutflow_w.cd()
    for h in hists_cutflow_w.values():
        h.Write()
    if h_cutflow_data is not None:
        h_cutflow_data.Write("h_cutflow_data")
    if df_tri is not None:
        for h in hists_cutflow_tri_w.values():
            h.Write()
    elif h_cutflow_tri_w is not None:
        h_cutflow_tri_w.Write("h_cutflow_trimuon_weighted")

    d_cutflow_r = get_or_create_dir(tfile, "Cutflow/Histograms_Raw")
    d_cutflow_r.cd()
    for h in hists_cutflow_r.values():
        h.Write()
    if h_cutflow_data is not None:
        h_cutflow_data.Write("h_cutflow_data")
    if df_tri is not None:
        for h in hists_cutflow_tri_r.values():
            h.Write()
    elif h_cutflow_tri_r is not None:
        h_cutflow_tri_r.Write("h_cutflow_trimuon_raw")

    # Hit Distributions directory
    d_sf = get_or_create_dir(tfile, "HitDistributions/SciFi_Hits")
    d_sf.cd()
    for s_idx in range(len(cutflow_stages)):
        stage_sf_mc_tot[s_idx].Write(f"h_sf_hits_mc_tot_stage{s_idx}")
        stage_sf_mc_sig[s_idx].Write(f"h_sf_hits_mc_sig_stage{s_idx}")
        stage_sf_mc_em[s_idx].Write(f"h_sf_hits_mc_em_stage{s_idx}")
        stage_sf_mc_had[s_idx].Write(f"h_sf_hits_mc_had_stage{s_idx}")
        stage_sf_mc_raw_sig[s_idx].Write(f"h_sf_hits_mc_raw_sig_stage{s_idx}")
        if stage_sf_data:
            stage_sf_data[s_idx].Write(f"h_sf_hits_data_stage{s_idx}")
        if stage_sf_tri_tot:
            stage_sf_tri_tot[s_idx].Write(f"h_sf_hits_tri_tot_stage{s_idx}")
            stage_sf_tri_sig[s_idx].Write(f"h_sf_hits_tri_sig_stage{s_idx}")
            stage_sf_tri_multimu[s_idx].Write(f"h_sf_hits_tri_multimu_stage{s_idx}")
            stage_sf_tri_raw[s_idx].Write(f"h_sf_hits_tri_raw_stage{s_idx}")

    d_ds = get_or_create_dir(tfile, "HitDistributions/DS_Hits")
    d_ds.cd()
    for s_idx in range(len(cutflow_stages)):
        stage_ds_mc_tot[s_idx].Write(f"h_ds_hits_mc_tot_stage{s_idx}")
        stage_ds_mc_sig[s_idx].Write(f"h_ds_hits_mc_sig_stage{s_idx}")
        stage_ds_mc_em[s_idx].Write(f"h_ds_hits_mc_em_stage{s_idx}")
        stage_ds_mc_had[s_idx].Write(f"h_ds_hits_mc_had_stage{s_idx}")
        stage_ds_mc_raw_sig[s_idx].Write(f"h_ds_hits_mc_raw_sig_stage{s_idx}")
        if stage_ds_data:
            stage_ds_data[s_idx].Write(f"h_ds_hits_data_stage{s_idx}")
        if stage_ds_tri_tot:
            stage_ds_tri_tot[s_idx].Write(f"h_ds_hits_tri_tot_stage{s_idx}")
            stage_ds_tri_sig[s_idx].Write(f"h_ds_hits_tri_sig_stage{s_idx}")
            stage_ds_tri_multimu[s_idx].Write(f"h_ds_hits_tri_multimu_stage{s_idx}")
            stage_ds_tri_raw[s_idx].Write(f"h_ds_hits_tri_raw_stage{s_idx}")

    d_veto = get_or_create_dir(tfile, "HitDistributions/Veto_Hits")
    d_veto.cd()
    for s_idx in range(len(cutflow_stages)):
        stage_veto_mc_tot[s_idx].Write(f"h_veto_hits_mc_tot_stage{s_idx}")
        stage_veto_mc_sig[s_idx].Write(f"h_veto_hits_mc_sig_stage{s_idx}")
        stage_veto_mc_em[s_idx].Write(f"h_veto_hits_mc_em_stage{s_idx}")
        stage_veto_mc_had[s_idx].Write(f"h_veto_hits_mc_had_stage{s_idx}")
        stage_veto_mc_raw_sig[s_idx].Write(f"h_veto_hits_mc_raw_sig_stage{s_idx}")
        if stage_veto_data:
            stage_veto_data[s_idx].Write(f"h_veto_hits_data_stage{s_idx}")
        if stage_veto_tri_tot:
            stage_veto_tri_tot[s_idx].Write(f"h_veto_hits_tri_tot_stage{s_idx}")
            stage_veto_tri_sig[s_idx].Write(f"h_veto_hits_tri_sig_stage{s_idx}")
            stage_veto_tri_multimu[s_idx].Write(f"h_veto_hits_tri_multimu_stage{s_idx}")
            stage_veto_tri_raw[s_idx].Write(f"h_veto_hits_tri_raw_stage{s_idx}")

    d_us = get_or_create_dir(tfile, "HitDistributions/US_Hits")
    d_us.cd()
    for s_idx in range(len(cutflow_stages)):
        stage_us_mc_tot[s_idx].Write(f"h_us_hits_mc_tot_stage{s_idx}")
        stage_us_mc_sig[s_idx].Write(f"h_us_hits_mc_sig_stage{s_idx}")
        stage_us_mc_em[s_idx].Write(f"h_us_hits_mc_em_stage{s_idx}")
        stage_us_mc_had[s_idx].Write(f"h_us_hits_mc_had_stage{s_idx}")
        stage_us_mc_raw_sig[s_idx].Write(f"h_us_hits_mc_raw_sig_stage{s_idx}")
        if stage_us_data:
            stage_us_data[s_idx].Write(f"h_us_hits_data_stage{s_idx}")
        if stage_us_tri_tot:
            stage_us_tri_tot[s_idx].Write(f"h_us_hits_tri_tot_stage{s_idx}")
            stage_us_tri_sig[s_idx].Write(f"h_us_hits_tri_sig_stage{s_idx}")
            stage_us_tri_multimu[s_idx].Write(f"h_us_hits_tri_multimu_stage{s_idx}")
            stage_us_tri_raw[s_idx].Write(f"h_us_hits_tri_raw_stage{s_idx}")

    # Muon Energy directory
    d_e = get_or_create_dir(tfile, "MuonEnergy")
    d_e.cd()
    for s_idx in range(len(cutflow_stages)):
        stage_energy_mc_sig[s_idx].Write(f"h_muon_energy_sig_stage{s_idx}")
        stage_energy_mc_all[s_idx].Write(f"h_muon_energy_all_stage{s_idx}")
        stage_energy_mc_raw_sig[s_idx].Write(f"h_muon_energy_raw_sig_stage{s_idx}")

    # Canvases directory
    d_canvases = get_or_create_dir(tfile, "Canvases")
    d_canvases.cd()
    c_superimposed_w.Write()
    c_superimposed_r.Write()
    c_sf_stages.Write()
    c_ds_stages.Write()
    c_veto_stages.Write()
    c_us_stages.Write()
    c_energy_stages.Write()
    c_energy_overlay.Write()
    if c_cutflow_tri_w is not None:
        c_cutflow_tri_w.Write()
        c_cutflow_tri_r.Write()
        c_sf_stages_tri.Write()
        c_ds_stages_tri.Write()
        c_veto_stages_tri.Write()
        c_us_stages_tri.Write()

    # Diagnostic distributions
    d_diag = get_or_create_dir(tfile, "Diagnostics")
    d_diag.cd()
    h_chi2_sig.GetValue().Write("h_chi2_signal")
    h_chi2_em.GetValue().Write("h_chi2_em_shower")
    h_chi2_had.GetValue().Write("h_chi2_hadronic")
    h_hits_sig.GetValue().Write("h_sf_hits_signal")
    h_hits_em.GetValue().Write("h_sf_hits_em_shower")
    h_hits_had.GetValue().Write("h_sf_hits_hadronic")
    h_veto_hits_sig.GetValue().Write("h_veto_hits_signal")
    h_us_hits_sig.GetValue().Write("h_us_hits_signal")
    h_p_sig_pass.GetValue().Write("h_p_signal_selected")

    if h_chi2_data_ptr is not None:
        scale_data_th1(h_chi2_data_ptr.GetValue(), args.data_scaling).Write("h_chi2_data")
    if h_sf_hits_data_ptr is not None:
        scale_data_th1(h_sf_hits_data_ptr.GetValue(), args.data_scaling).Write("h_sf_hits_data")
    if h_ds_hits_data_ptr is not None:
        scale_data_th1(h_ds_hits_data_ptr.GetValue(), args.data_scaling).Write("h_ds_hits_data")
    if h_veto_hits_data_ptr is not None:
        scale_data_th1(h_veto_hits_data_ptr.GetValue(), args.data_scaling).Write("h_veto_hits_data")
    if h_us_hits_data_ptr is not None:
        scale_data_th1(h_us_hits_data_ptr.GetValue(), args.data_scaling).Write("h_us_hits_data")

    if h_chi2_tri_ptr is not None:
        h_chi2_tri_ptr.GetValue().Write("h_chi2_trimuon")
    if h_sf_hits_tri_ptr is not None:
        h_sf_hits_tri_ptr.GetValue().Write("h_sf_hits_trimuon")
    if h_ds_hits_tri_ptr is not None:
        h_ds_hits_tri_ptr.GetValue().Write("h_ds_hits_trimuon")
    if h_veto_hits_tri_ptr is not None:
        h_veto_hits_tri_ptr.GetValue().Write("h_veto_hits_trimuon")
    if h_us_hits_tri_ptr is not None:
        h_us_hits_tri_ptr.GetValue().Write("h_us_hits_trimuon")

    tfile.Flush()
    tfile.Close()

    shutil.move(tmp_path, args.output_file)

    # Export standalone canvas graphics (PNG and PDF)
    out_base = os.path.splitext(args.output_file)[0]
    png_w = f"{out_base}_cutflow_superimposed.png"
    pdf_w = f"{out_base}_cutflow_superimposed.pdf"
    png_r = f"{out_base}_cutflow_superimposed_raw.png"
    pdf_r = f"{out_base}_cutflow_superimposed_raw.pdf"

    png_sf = f"{out_base}_scifi_hits_stages.png"
    pdf_sf = f"{out_base}_scifi_hits_stages.pdf"
    png_ds = f"{out_base}_ds_hits_stages.png"
    pdf_ds = f"{out_base}_ds_hits_stages.pdf"
    png_veto = f"{out_base}_veto_hits_stages.png"
    pdf_veto = f"{out_base}_veto_hits_stages.pdf"
    png_us = f"{out_base}_us_hits_stages.png"
    pdf_us = f"{out_base}_us_hits_stages.pdf"
    png_e_stages = f"{out_base}_muon_energy_stages.png"
    pdf_e_stages = f"{out_base}_muon_energy_stages.pdf"
    png_e_over = f"{out_base}_muon_energy_overlay.png"
    pdf_e_over = f"{out_base}_muon_energy_overlay.pdf"

    c_superimposed_w.SaveAs(png_w)
    c_superimposed_w.SaveAs(pdf_w)
    c_superimposed_r.SaveAs(png_r)
    c_superimposed_r.SaveAs(pdf_r)

    c_sf_stages.SaveAs(png_sf)
    c_sf_stages.SaveAs(pdf_sf)
    c_ds_stages.SaveAs(png_ds)
    c_ds_stages.SaveAs(pdf_ds)
    c_veto_stages.SaveAs(png_veto)
    c_veto_stages.SaveAs(pdf_veto)
    c_us_stages.SaveAs(png_us)
    c_us_stages.SaveAs(pdf_us)
    c_energy_stages.SaveAs(png_e_stages)
    c_energy_stages.SaveAs(pdf_e_stages)
    c_energy_overlay.SaveAs(png_e_over)
    c_energy_overlay.SaveAs(pdf_e_over)

    if df_tri is not None:
        png_tri_w = f"{out_base}_cutflow_trimuon_superimposed.png"
        pdf_tri_w = f"{out_base}_cutflow_trimuon_superimposed.pdf"
        png_tri_r = f"{out_base}_cutflow_trimuon_superimposed_raw.png"
        pdf_tri_r = f"{out_base}_cutflow_trimuon_superimposed_raw.pdf"

        png_sf_tri = f"{out_base}_scifi_hits_stages_trimuon.png"
        pdf_sf_tri = f"{out_base}_scifi_hits_stages_trimuon.pdf"
        png_ds_tri = f"{out_base}_ds_hits_stages_trimuon.png"
        pdf_ds_tri = f"{out_base}_ds_hits_stages_trimuon.pdf"
        png_veto_tri = f"{out_base}_veto_hits_stages_trimuon.png"
        pdf_veto_tri = f"{out_base}_veto_hits_stages_trimuon.pdf"
        png_us_tri = f"{out_base}_us_hits_stages_trimuon.png"
        pdf_us_tri = f"{out_base}_us_hits_stages_trimuon.pdf"

        c_cutflow_tri_w.SaveAs(png_tri_w)
        c_cutflow_tri_w.SaveAs(pdf_tri_w)
        c_cutflow_tri_r.SaveAs(png_tri_r)
        c_cutflow_tri_r.SaveAs(pdf_tri_r)

        c_sf_stages_tri.SaveAs(png_sf_tri)
        c_sf_stages_tri.SaveAs(pdf_sf_tri)
        c_ds_stages_tri.SaveAs(png_ds_tri)
        c_ds_stages_tri.SaveAs(pdf_ds_tri)
        c_veto_stages_tri.SaveAs(png_veto_tri)
        c_veto_stages_tri.SaveAs(pdf_veto_tri)
        c_us_stages_tri.SaveAs(png_us_tri)
        c_us_stages_tri.SaveAs(pdf_us_tri)

    print(f"[✓] Successfully exported publication graphics:")
    print(f"    --> {png_w}")
    print(f"    --> {pdf_w}")
    print(f"    --> {png_r}")
    print(f"    --> {pdf_r}")
    print(f"    --> {png_sf}")
    print(f"    --> {pdf_sf}")
    print(f"    --> {png_ds}")
    print(f"    --> {pdf_ds}")
    print(f"    --> {png_veto}")
    print(f"    --> {pdf_veto}")
    print(f"    --> {png_us}")
    print(f"    --> {pdf_us}")
    print(f"    --> {png_e_stages}")
    print(f"    --> {pdf_e_stages}")
    print(f"    --> {png_e_over}")
    print(f"    --> {pdf_e_over}")
    if df_tri is not None:
        print(f"    --> {png_tri_w}")
        print(f"    --> {pdf_tri_w}")
        print(f"    --> {png_tri_r}")
        print(f"    --> {pdf_tri_r}")
        print(f"    --> {png_sf_tri}")
        print(f"    --> {pdf_sf_tri}")
        print(f"    --> {png_ds_tri}")
        print(f"    --> {pdf_ds_tri}")
        print(f"    --> {png_veto_tri}")
        print(f"    --> {pdf_veto_tri}")
        print(f"    --> {png_us_tri}")
        print(f"    --> {pdf_us_tri}")

    fsize_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"[✓] Successfully committed output ROOT file ({fsize_mb:.2f} MB):")
    print(f"    --> {args.output_file}")
    print(f"[✓] Total elapsed execution time: {time.time() - t0:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
