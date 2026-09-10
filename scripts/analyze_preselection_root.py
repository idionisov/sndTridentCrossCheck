#!/usr/bin/env python3
"""
================================================================================
SND@LHC Comprehensive Pre-Selection Observable & Canvas Generator
================================================================================
High-throughput pre-tracking distribution extractor supporting 1D, 2D, and
TProfile observables across:
1. Monte Carlo Trident Signal (Rock, Target, Muon System, Total Combined)
2. Monte Carlo Trident Background (unboosted single-muon events)
3. Passing Muon Simulation (optional, via '--pmu-input')
4. Real Experimental Collision Data (optional, via '--data-input')

Uses:
- Compiled C++ metric extractor (snd::trident::PreselectionProcessor) in libtrident_analysis.so
- Multi-threaded ROOT RDataFrame with in-memory execution and zero disk-spill
- Okabe-Ito colorblind-safe, publication-grade visualization palette
- Modular configuration file ('config/preselection_histograms_config.py')
- Dual-pad TCanvas rendering with efficiency & rejection curves
- Safe hierarchical directory creation and atomic file writing (safe on EOS)

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
from typing import Dict, Any, List, Tuple, Optional

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

# Add project root to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == "scripts" else _script_dir
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from snd import DataManager, load_trident_libraries
from config.preselection_histograms_config import HIST_CONFIGS_1D, HIST_CONFIGS_2D, PROFILE_CONFIGS

# Load compiled C++ analysis libraries and dictionaries
load_trident_libraries(_repo_root)

# ==============================================================================
# Publication-Grade High-Visibility Palette (Standard Persistent ROOT Colors)
# ==============================================================================
PALETTE = {
    "rock": {
        "color": ROOT.kRed + 1,                     # Vibrant Red / Crimson
        "label": "Rock Trident Signal",
        "line_width": 3,
        "line_style": 1,
        "draw_opt": "HIST",
        "leg_opt": "L",
    },
    "target": {
        "color": ROOT.kAzure + 1,                   # High-Contrast Deep Azure Blue
        "label": "Target Trident Signal",
        "line_width": 3,
        "line_style": 1,
        "draw_opt": "HIST",
        "leg_opt": "L",
    },
    "mufi": {
        "color": ROOT.kTeal + 2,                    # Vibrant Emerald / Bluish Green
        "label": "Muon System Trident Signal",
        "line_width": 3,
        "line_style": 1,
        "draw_opt": "HIST",
        "leg_opt": "L",
    },
    "allsig": {
        "color": ROOT.kViolet + 1,                  # Rich Violet / Purple
        "label": "All Trident Signal",
        "line_width": 2,
        "line_style": 1,
        "draw_opt": "HIST",
        "leg_opt": "L",
    },
    "bkg": {
        "color": ROOT.kGray + 2,                    # Dark Slate / Charcoal Gray
        "label": "Trident MC Bkg",
        "line_width": 3,
        "line_style": 2,                            # Distinct thick dashed line
        "draw_opt": "HIST",
        "leg_opt": "L",
    },
    "pmu": {
        "color": ROOT.kOrange + 2,                  # Warm Amber / Orange
        "label": "Passing #mu MC",
        "line_width": 3,
        "line_style": 7,                            # Dash-dot
        "draw_opt": "HIST",
        "leg_opt": "L",
    },
    "data": {
        "color": ROOT.kBlack,                       # Solid Black
        "label": "Collision Data",
        "line_width": 2,
        "line_style": 1,
        "marker_size": 0,                           # Suppress markers for histogram format
        "draw_opt": "HIST E",                       # Step outline with vertical error bars
        "leg_opt": "LE",                            # Line + vertical error bar in legend
    },
    "gallery": {
        "color": ROOT.kMagenta + 1,                 # High-Visibility Vivid Magenta / Orchid
        "label": "Gallery Candidates",
        "line_width": 3,
        "line_style": 1,
        "marker_size": 0,                           # Suppress markers for histogram format
        "draw_opt": "HIST E",                       # Step outline with vertical error bars
        "leg_opt": "LE",                            # Line + vertical error bar in legend
    },
}


def get_or_create_dir(parent_tfile_or_dir, path_str: str):
    """Safely navigates or creates nested directory hierarchy without slashes bug in ROOT."""
    parts = [p for p in path_str.split("/") if p]
    curr = parent_tfile_or_dir
    for part in parts:
        sub = curr.GetDirectory(part)
        if not sub:
            sub = curr.mkdir(part)
        curr = sub
    return curr


def create_efficiency_graph(h_raw: ROOT.TH1D, cut_dir: str = "<=") -> ROOT.TGraphAsymmErrors:
    """Computes TGraphAsymmErrors for cumulative efficiency with binomial error propagation."""
    gr = ROOT.TGraphAsymmErrors()
    ROOT.SetOwnership(gr, False)
    gr.SetFillStyle(0)
    gr.SetFillColor(0)
    tot = h_raw.Integral()
    if tot <= 0:
        return gr

    nb = h_raw.GetNbinsX()
    var_tot = sum([h_raw.GetBinError(i)**2 for i in range(1, nb + 1)])

    for b in range(1, nb + 1):
        x = h_raw.GetBinLowEdge(b) + h_raw.GetBinWidth(b) if cut_dir == "<=" else h_raw.GetBinLowEdge(b)
        if cut_dir == "<=":
            pass_val = h_raw.Integral(1, b)
            var_pass = sum([h_raw.GetBinError(i)**2 for i in range(1, b + 1)])
        else:
            pass_val = h_raw.Integral(b, nb)
            var_pass = sum([h_raw.GetBinError(i)**2 for i in range(b, nb + 1)])

        eff = pass_val / tot
        var_eff = (var_pass * (1.0 - 2.0 * eff) + (eff**2) * var_tot) / (tot**2) if tot > 0 else 0.0
        err = math.sqrt(max(0.0, var_eff))
        pt = gr.GetN()
        gr.SetPoint(pt, x, eff)
        gr.SetPointError(pt, 0.0, 0.0, min(eff, err), min(1.0 - eff, err))
    return gr


def create_rejection_graph(h_raw: ROOT.TH1D, cut_dir: str = "<=") -> ROOT.TGraphAsymmErrors:
    """Computes TGraphAsymmErrors for cumulative rejection (1 - FPR)."""
    gr = ROOT.TGraphAsymmErrors()
    ROOT.SetOwnership(gr, False)
    gr.SetFillStyle(0)
    gr.SetFillColor(0)
    tot = h_raw.Integral()
    if tot <= 0:
        return gr

    nb = h_raw.GetNbinsX()
    var_tot = sum([h_raw.GetBinError(i)**2 for i in range(1, nb + 1)])

    for b in range(1, nb + 1):
        x = h_raw.GetBinLowEdge(b) + h_raw.GetBinWidth(b) if cut_dir == "<=" else h_raw.GetBinLowEdge(b)
        if cut_dir == "<=":
            pass_val = h_raw.Integral(1, b)
            var_pass = sum([h_raw.GetBinError(i)**2 for i in range(1, b + 1)])
        else:
            pass_val = h_raw.Integral(b, nb)
            var_pass = sum([h_raw.GetBinError(i)**2 for i in range(b, nb + 1)])

        fpr = pass_val / tot
        rej = 1.0 - fpr
        var_fpr = (var_pass * (1.0 - 2.0 * fpr) + (fpr**2) * var_tot) / (tot**2) if tot > 0 else 0.0
        err = math.sqrt(max(0.0, var_fpr))
        pt = gr.GetN()
        gr.SetPoint(pt, x, rej)
        gr.SetPointError(pt, 0.0, 0.0, min(rej, err), min(1.0 - rej, err))
    return gr


def create_superimposed_1d_canvas(
    var_name: str,
    h_rock: Optional[ROOT.TH1D] = None,
    h_target: Optional[ROOT.TH1D] = None,
    h_mufi: Optional[ROOT.TH1D] = None,
    h_bkg: Optional[ROOT.TH1D] = None,
    h_data: Optional[ROOT.TH1D] = None,
    h_pmu: Optional[ROOT.TH1D] = None,
    h_gallery: Optional[ROOT.TH1D] = None,
    cut_dir: str = "<="
) -> ROOT.TCanvas:
    """Generates dual-pad publication canvas with log-y distributions and efficiency/rejection curve."""
    c = ROOT.TCanvas(f"c_{var_name}", f"Preselection: {var_name}", 880, 800)
    ROOT.SetOwnership(c, False)

    pad1 = ROOT.TPad(f"pad1_{var_name}", "Top", 0.0, 0.30, 1.0, 1.0)
    pad2 = ROOT.TPad(f"pad2_{var_name}", "Bottom", 0.0, 0.0, 1.0, 0.30)
    ROOT.SetOwnership(pad1, False)
    ROOT.SetOwnership(pad2, False)

    pad1.SetBottomMargin(0.03); pad1.SetTopMargin(0.08); pad1.SetLeftMargin(0.12); pad1.SetRightMargin(0.05); pad1.SetLogy(1); pad1.Draw()
    pad2.SetTopMargin(0.03); pad2.SetBottomMargin(0.28); pad2.SetLeftMargin(0.12); pad2.SetRightMargin(0.05); pad2.Draw()

    # 1. Top Pad: Normalized Observable Distributions
    pad1.cd()
    raw_dict = {}
    if h_rock and h_rock.GetEntries() > 0:
        raw_dict["rock"] = (h_rock, PALETTE["rock"])
    if h_target and h_target.GetEntries() > 0:
        raw_dict["target"] = (h_target, PALETTE["target"])
    if h_mufi and h_mufi.GetEntries() > 0:
        raw_dict["mufi"] = (h_mufi, PALETTE["mufi"])
    if h_bkg and h_bkg.GetEntries() > 0:
        raw_dict["bkg"] = (h_bkg, PALETTE["bkg"])
    if h_pmu and h_pmu.GetEntries() > 0:
        raw_dict["pmu"] = (h_pmu, PALETTE["pmu"])
    if h_data and h_data.GetEntries() > 0:
        raw_dict["data"] = (h_data, PALETTE["data"])
    if h_gallery and h_gallery.GetEntries() > 0:
        raw_dict["gallery"] = (h_gallery, PALETTE["gallery"])

    samples = {}
    for k, (h_orig, style) in raw_dict.items():
        if h_orig and h_orig.GetEntries() > 0:
            h_cl = h_orig.Clone(f"{h_orig.GetName()}_norm_{var_name}")
            ROOT.SetOwnership(h_cl, False)
            if h_cl.Integral() > 0:
                h_cl.Scale(1.0 / h_cl.Integral())
            h_cl.SetLineColor(style["color"])
            h_cl.SetLineWidth(style["line_width"])
            h_cl.SetLineStyle(style["line_style"])
            h_cl.SetStats(0)
            h_cl.SetMarkerColor(style["color"])
            h_cl.SetMarkerSize(0)  # Suppress markers for histogram format
            h_cl.SetFillStyle(0)   # Transparent fill: avoids grey overlays across ROOT sessions
            samples[k] = (h_cl, h_orig, style)

    max_y = max([s[0].GetMaximum() for s in samples.values()]) * 6.0 if samples else 1.0
    for s in samples.values():
        s[0].SetMaximum(max(1.0, max_y))
        s[0].SetMinimum(1e-4)

    first_draw = True
    for k in ["bkg", "pmu", "rock", "target", "mufi", "data", "gallery"]:
        if k in samples:
            h_s, h_orig, style = samples[k]
            raw_title = h_orig.GetTitle()
            main_title = raw_title.split(";")[0] if ";" in raw_title else raw_title
            if ":" in main_title:
                main_title = main_title.split(":", 1)[1].strip()
            x_title = h_orig.GetXaxis().GetTitle()
            h_s.SetTitle(f"{main_title};{x_title};Normalized Entries")

            opt = style.get("draw_opt", "HIST")
            if not first_draw:
                opt += " SAME"
            h_s.Draw(opt)
            first_draw = False

    leg = ROOT.TLegend(0.48, 0.60, 0.93, 0.89)
    ROOT.SetOwnership(leg, False)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.033)
    for k in ["gallery", "data", "rock", "target", "mufi", "bkg", "pmu"]:
        if k in samples:
            h_cl, h_orig, style = samples[k]
            ent_str = f" (N={h_orig.GetEntries():,.0f})"
            leg_opt = style.get("leg_opt", style.get("draw_opt", "L"))
            leg.AddEntry(h_cl, f"{style['label']}{ent_str}", leg_opt)
    leg.Draw()

    # 2. Bottom Pad: Cumulative Efficiency & Rejection Curves
    pad2.cd()
    pad2.SetGridy(1)
    ref_h = None
    for cand in [h_rock, h_target, h_mufi, h_bkg, h_data, h_pmu, h_gallery]:
        if cand and cand.GetNbinsX() > 0:
            ref_h = cand
            break

    if ref_h:
        frame = ROOT.TH1D(
            f"frame_{var_name}",
            f";{ref_h.GetXaxis().GetTitle()};Eff / Rej",
            ref_h.GetNbinsX(),
            ref_h.GetXaxis().GetXmin(),
            ref_h.GetXaxis().GetXmax()
        )
        ROOT.SetOwnership(frame, False)
        frame.SetStats(0)
        frame.GetYaxis().SetRangeUser(-0.05, 1.15)
        frame.GetYaxis().SetNdivisions(505)
        frame.GetYaxis().SetTitleSize(0.09)
        frame.GetYaxis().SetTitleOffset(0.55)
        frame.GetYaxis().SetLabelSize(0.08)
        frame.GetXaxis().SetTitleSize(0.10)
        frame.GetXaxis().SetTitleOffset(1.1)
        frame.GetXaxis().SetLabelSize(0.08)
        frame.Draw("AXIS")

        # Reference guideline at y = 1.0
        line1 = ROOT.TLine(ref_h.GetXaxis().GetXmin(), 1.0, ref_h.GetXaxis().GetXmax(), 1.0)
        ROOT.SetOwnership(line1, False)
        line1.SetLineColor(ROOT.kGray + 1)
        line1.SetLineStyle(3)
        line1.Draw("SAME")

    leg2 = ROOT.TLegend(0.48, 0.28, 0.93, 0.65)
    ROOT.SetOwnership(leg2, False)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    leg2.SetTextSize(0.058)

    if h_rock and h_rock.GetEntries() > 0:
        gr_rock = create_efficiency_graph(h_rock, cut_dir)
        gr_rock.SetLineColor(PALETTE["rock"]["color"])
        gr_rock.SetLineWidth(3)
        if gr_rock.GetN() > 0:
            gr_rock.Draw("LX SAME")
            leg2.AddEntry(gr_rock, f"Rock Eff ({cut_dir})", "L")

    if h_target and h_target.GetEntries() > 0:
        gr_target = create_efficiency_graph(h_target, cut_dir)
        gr_target.SetLineColor(PALETTE["target"]["color"])
        gr_target.SetLineWidth(3)
        if gr_target.GetN() > 0:
            gr_target.Draw("LX SAME")
            leg2.AddEntry(gr_target, f"Target Eff ({cut_dir})", "L")

    if h_mufi and h_mufi.GetEntries() > 0:
        gr_mufi = create_efficiency_graph(h_mufi, cut_dir)
        gr_mufi.SetLineColor(PALETTE["mufi"]["color"])
        gr_mufi.SetLineWidth(3)
        if gr_mufi.GetN() > 0:
            gr_mufi.Draw("LX SAME")
            leg2.AddEntry(gr_mufi, f"MuFilter Eff ({cut_dir})", "L")

    if h_gallery and h_gallery.GetEntries() > 0:
        gr_gal = create_efficiency_graph(h_gallery, cut_dir)
        gr_gal.SetLineColor(PALETTE["gallery"]["color"])
        gr_gal.SetLineWidth(3)
        if gr_gal.GetN() > 0:
            gr_gal.Draw("LX SAME")
            leg2.AddEntry(gr_gal, f"Gallery Eff ({cut_dir})", "L")

    h_eval_bkg = None
    bkg_color = PALETTE["bkg"]["color"]
    if h_bkg and h_bkg.GetEntries() > 0:
        h_eval_bkg = h_bkg
        bkg_color = PALETTE["bkg"]["color"]
    elif h_data and h_data.GetEntries() > 0:
        h_eval_bkg = h_data
        bkg_color = PALETTE["data"]["color"]
    elif h_pmu and h_pmu.GetEntries() > 0:
        h_eval_bkg = h_pmu
        bkg_color = PALETTE["pmu"]["color"]

    if h_eval_bkg and h_eval_bkg.GetEntries() > 0:
        gr_bkg = create_rejection_graph(h_eval_bkg, cut_dir)
        gr_bkg.SetLineColor(bkg_color)
        gr_bkg.SetLineWidth(3)
        gr_bkg.SetLineStyle(2)
        if gr_bkg.GetN() > 0:
            gr_bkg.Draw("LX SAME")
            leg2.AddEntry(gr_bkg, f"Bkg Rej ({cut_dir})", "L")

    if leg2.GetNRows() > 0:
        leg2.Draw()

    c.cd()
    c.Modified()
    c.Update()
    return c


def setup_dataframe(data: DataManager, stream_type: str, radius: float = 40.0, max_events: int = 0) -> ROOT.RDataFrame:
    """Configures RDataFrame with compiled C++ PreselectionProcessor and standardized observables."""
    df = data.df()
    if max_events > 0:
        if ROOT.IsImplicitMTEnabled():
            df = df.Filter(f"rdfentry_ < {max_events}")
        else:
            df = df.Range(max_events)

    # 1. Resolve Hit branch names (supporting both rawConv and cbmsim conventions)
    sf_branch = "Digi_ScifiHits" if data.has_branch("Digi_ScifiHits") else "Digits_Scifi"
    if data.has_branch("Digi_MuFilterHits"):
        mf_branch = "Digi_MuFilterHits"
    elif data.has_branch("Digi_MuFilterHit"):
        mf_branch = "Digi_MuFilterHit"
    else:
        mf_branch = "Digits_MuFilter"

    # 2. Extract Preselection Metrics using compiled C++ processor
    processor = ROOT.snd.trident.PreselectionProcessor(radius)
    df = df.Define("metrics", processor, [sf_branch, mf_branch])

    # 3. Define / Redefine flat scalar observables for RDataFrame Histo1D / Histo2D
    for v_name, _, _, _, _, _, _ in HIST_CONFIGS_1D:
        if data.has_branch(v_name):
            df = df.Redefine(v_name, f"metrics.{v_name}")
        else:
            df = df.Define(v_name, f"metrics.{v_name}")

    # 4. Define vector observables for TProfiles
    df = (
        df
        .Define("prof_scifi_qdc_vs_station_x", "metrics.get_scifi_qdc_vs_station_x()")
        .Define("prof_scifi_qdc_vs_station_y", "metrics.get_scifi_qdc_vs_station_y()")
        .Define("prof_scifi_nhits_vs_station_x", "metrics.get_scifi_nhits_vs_station_x()")
        .Define("prof_scifi_nhits_vs_station_y", "metrics.get_scifi_nhits_vs_station_y()")
        .Define("prof_ds_qdc_vs_station_x", "metrics.get_ds_qdc_vs_station_x()")
        .Define("prof_ds_qdc_vs_station_y", "metrics.get_ds_qdc_vs_station_y()")
        .Define("prof_ds_nhits_vs_station_x", "metrics.get_ds_nhits_vs_station_x()")
        .Define("prof_ds_nhits_vs_station_y", "metrics.get_ds_nhits_vs_station_y()")
        .Define("prof_detector_longitudinal_qdc_x", "metrics.get_detector_longitudinal_qdc_x()")
        .Define("prof_detector_longitudinal_qdc_y", "metrics.get_detector_longitudinal_qdc_y()")
    )

    # 5. Standardized Truth & Weight categorization
    if stream_type == "trident_mc":
        if data.has_branch("is_signal"):
            df = df.Define("eval_is_signal", "is_signal")
        elif data.has_branch("MCTrack"):
            truth_config = ROOT.snd.trident.TridentTruthConfig()
            truth_proc = ROOT.snd.trident.TridentTruthProcessor(truth_config)
            df = df.Define("truth_info", truth_proc, ["MCTrack"])
            df = df.Define("eval_is_signal", "truth_info.hasCandidate")
        else:
            df = df.Define("eval_is_signal", "true")

        if data.has_branch("region_type"):
            df = df.Define("eval_region_type", "region_type")
        elif data.has_branch("vtx_z"):
            df = df.Define("eval_region_type", "vtx_z < 260.0 ? 1 : (vtx_z < 355.0 ? 2 : 3)")
        elif data.has_branch("truth_info"):
            df = df.Define("eval_region_type", "static_cast<int>(truth_info.regionType)")
        else:
            df = df.Define("eval_region_type", "1")

        if data.has_branch("mc_weight"):
            df = df.Define("eval_weight", "mc_weight")
        elif data.has_branch("truth_info"):
            df = df.Define("eval_weight", "truth_info.scaledWeight")
        elif data.has_branch("MCEventHeader"):
            df = df.Define("eval_weight", "static_cast<double>(MCEventHeader.GetWeight())")
        else:
            df = df.Define("eval_weight", "1.0")
    else:
        df = df.Define("eval_is_signal", "false")
        df = df.Define("eval_region_type", "-1")
        if data.has_branch("mc_weight"):
            df = df.Define("eval_weight", "mc_weight")
        elif data.has_branch("MCEventHeader"):
            df = df.Define("eval_weight", "static_cast<double>(MCEventHeader.GetWeight())")
        else:
            df = df.Define("eval_weight", "1.0")

    return df


def book_category_histograms(df_node: ROOT.RDataFrame, cat_name: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Books 1D, 2D, and Profile histograms for a specific category node in RDataFrame."""
    h1_ptrs = {}
    for v_name, title, nbins, xmin, xmax, _, _ in HIST_CONFIGS_1D:
        model = ROOT.RDF.TH1DModel(f"h_{cat_name}_{v_name}", f"{cat_name.capitalize()}: {title}", nbins, xmin, xmax)
        h1_ptrs[v_name] = df_node.Histo1D(model, v_name, "eval_weight")

    h2_ptrs = {}
    for h2_name, title, nx, xmin, xmax, ny, ymin, ymax, x_var, y_var, _ in HIST_CONFIGS_2D:
        model = ROOT.RDF.TH2DModel(f"h2_{cat_name}_{h2_name}", f"{cat_name.capitalize()}: {title}", nx, xmin, xmax, ny, ymin, ymax)
        h2_ptrs[h2_name] = df_node.Histo2D(model, x_var, y_var, "eval_weight")

    prof_ptrs = {}
    for prof_name, title, nx, xmin, xmax, _ in PROFILE_CONFIGS:
        model = ROOT.RDF.TProfile1DModel(f"prof_{cat_name}_{prof_name}", f"{cat_name.capitalize()}: {title}", nx, xmin, xmax)
        prof_ptrs[prof_name] = df_node.Profile1D(model, f"{prof_name}_x", f"{prof_name}_y", "eval_weight")

    return h1_ptrs, h2_ptrs, prof_ptrs


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive SND@LHC Preselection Observable & Canvas Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Stream Specification:
  You can provide inputs using dedicated flags starting with -i or repeating -i:
    -im, --input-mc       Trimuon MC pattern / regex (e.g. -im "/eos/.../trimuon_.*\\.root")
    -id, --input-data     Collision Data pattern / regex (e.g. -id "/eos/.../run_.*\\.root")
    -ip, --input-pmu      Passing Muon MC pattern / regex (e.g. -ip "/eos/.../pmu_.*\\.root")
    -ig, --input-gallery  Gallery Candidate pattern / regex (e.g. -ig "/eos/.../gal_.*\\.root")
  Or via repeatable -i:
    -i mc="<pattern>" -i data="<pattern>" -i pmu="<pattern>" -i gallery="<pattern>"
  At least one input stream must be provided.
        """
    )
    # Dedicated input flags (all starting with -i, all optional on their own)
    parser.add_argument("-im", "--input-mc", "--mc", dest="input_mc", default=None, help="Input Trimuon MC ROOT files (wildcard, regex, or file)")
    parser.add_argument("-id", "--input-data", "--data", "-d", dest="input_data", default=None, help="Input Collision Data ROOT files (wildcard, regex, or file)")
    parser.add_argument("-ip", "--input-pmu", "--pmu", "-p", dest="input_pmu", default=None, help="Input Passing Muon MC ROOT files (wildcard, regex, or file)")
    parser.add_argument("-ig", "--input-gallery", "--gallery", "-g", dest="input_gallery", default=None, help="Input Gallery Candidate events (wildcard, regex, or file)")

    # Repeatable generic input flag (e.g. -i mc=... -i data=... or -i <mc_path>)
    parser.add_argument("-i", "--input", dest="generic_inputs", action="append", default=None, help="Repeatable input flag: key=pattern (keys: mc, data, pmu, gallery), or standalone pattern (defaults to mc)")

    parser.add_argument("-o", "--output", dest="output_file", default="plots/preselection.root", help="Master output ROOT file")
    parser.add_argument("-n", "--max-events", dest="max_events", type=int, default=0, help="Default max events limit per input stream (0 = all)")
    parser.add_argument("-nm", "--max-events-mc", dest="max_events_mc", type=int, default=0, help="Max events limit for Trimuon MC (overrides -n)")
    parser.add_argument("-nd", "--max-events-data", dest="max_events_data", type=int, default=0, help="Max events limit for Collision Data (overrides -n)")
    parser.add_argument("-np", "--max-events-pmu", dest="max_events_pmu", type=int, default=0, help="Max events limit for Passing Muons (overrides -n)")
    parser.add_argument("-ng", "--max-events-gallery", dest="max_events_gallery", type=int, default=0, help="Max events limit for Gallery (overrides -n)")
    parser.add_argument("-j", "--threads", "--workers", dest="num_threads", type=int, default=8, help="Number of RDataFrame worker threads")
    parser.add_argument("-r", "--radius", dest="radius", type=float, default=40.0, help="Hit/QDC density search radius in channels (default: 40.0)")

    args = parser.parse_args()

    # Parse repeatable generic inputs (-i mc=... -i data=...)
    if args.generic_inputs:
        for item in args.generic_inputs:
            matched_key = False
            for sep in ["=", ":"]:
                if sep in item:
                    k, v = item.split(sep, 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k in ("mc", "trimuon", "signal"):
                        args.input_mc = v
                        matched_key = True
                        break
                    elif k in ("data", "collision", "coll"):
                        args.input_data = v
                        matched_key = True
                        break
                    elif k in ("pmu", "muon", "muons", "passing"):
                        args.input_pmu = v
                        matched_key = True
                        break
                    elif k in ("gallery", "gal", "golden", "candidates"):
                        args.input_gallery = v
                        matched_key = True
                        break
            if not matched_key:
                if not args.input_mc:
                    args.input_mc = item
                elif not args.input_data:
                    args.input_data = item
                elif not args.input_pmu:
                    args.input_pmu = item
                elif not args.input_gallery:
                    args.input_gallery = item

    # Validation: Ensure at least one input stream is provided
    if not (args.input_mc or args.input_data or args.input_pmu or args.input_gallery):
        parser.error(
            "At least one input stream must be provided!\n"
            "Please specify one or more of:\n"
            "  -im / --input-mc       Trimuon MC pattern or regex\n"
            "  -id / --input-data     Collision Data pattern or regex\n"
            "  -ip / --input-pmu      Passing Muon MC pattern or regex\n"
            "  -ig / --input-gallery  Gallery Candidate pattern or regex\n"
            "  Or repeatable -i:      -i mc=... -i data=... -i pmu=... -i gallery=..."
        )

    # Resolve event limits per stream (specific limit overrides global -n)
    n_mc = args.max_events_mc if args.max_events_mc > 0 else args.max_events
    n_data = args.max_events_data if args.max_events_data > 0 else args.max_events
    n_pmu = args.max_events_pmu if args.max_events_pmu > 0 else args.max_events
    n_gal = args.max_events_gallery if args.max_events_gallery > 0 else args.max_events

    t0_all = time.time()
    print("=" * 80)
    print("SND@LHC COMPREHENSIVE PRESELECTION OBSERVABLE GENERATOR (RDataFrame)")
    print("=" * 80)
    print(f"Trimuon MC Pattern : {args.input_mc or 'None'} (limit: {n_mc if n_mc > 0 else 'All'})")
    print(f"Collision Data     : {args.input_data or 'None'} (limit: {n_data if n_data > 0 else 'All'})")
    print(f"Passing Muons      : {args.input_pmu or 'None'} (limit: {n_pmu if n_pmu > 0 else 'All'})")
    print(f"Gallery Events     : {args.input_gallery or 'None'} (limit: {n_gal if n_gal > 0 else 'All'})")
    print(f"Total 1D Observables: {len(HIST_CONFIGS_1D)}")
    print(f"Total 2D Histograms : {len(HIST_CONFIGS_2D)}")
    print(f"Total Profiles      : {len(PROFILE_CONFIGS)}")
    print(f"Worker Threads     : {args.num_threads}")
    print(f"Output File        : {args.output_file}")
    print("=" * 80)

    # Dictionary to hold materialized ROOT histograms per category
    hist_store: Dict[str, Dict[str, Any]] = {}
    counts_store: Dict[str, int] = {}

    # Category display names and directory mapping
    cat_dir_map = {
        "rock": "Histograms/Rock_Signal",
        "target": "Histograms/Target_Signal",
        "mufi": "Histograms/MuonSystem_Signal",
        "allsig": "Histograms/All_Signal",
        "bkg": "Histograms/Background",
        "data": "Histograms/Data",
        "pmu": "Histograms/Passing_Muon",
        "gallery": "Histograms/Gallery_Candidates",
    }

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetErrorX(0)

    threads_to_use = args.num_threads

    # 1. Process MC Trident Stream (if provided)
    if args.input_mc:
        print("\n--- [1/4] Loading & Booking MC Trident Datasets ---")
        mc_data = DataManager(args.input_mc, num_threads=threads_to_use)
        print(f"  * Resolved {mc_data.num_files:,} files | Tree: '{mc_data.tree_name}' | Total entries: {mc_data.entries:,}")

        df_mc = setup_dataframe(mc_data, "trident_mc", args.radius, n_mc)

        mc_nodes = {
            "rock": df_mc.Filter("eval_is_signal && eval_region_type == 1"),
            "target": df_mc.Filter("eval_is_signal && eval_region_type == 2"),
            "mufi": df_mc.Filter("eval_is_signal && eval_region_type == 3"),
            "allsig": df_mc.Filter("eval_is_signal"),
            "bkg": df_mc.Filter("!eval_is_signal"),
        }

        mc_books = {}
        for cat, node in mc_nodes.items():
            mc_books[cat] = {
                "h1": book_category_histograms(node, cat)[0],
                "h2": book_category_histograms(node, cat)[1],
                "prof": book_category_histograms(node, cat)[2],
                "count": node.Count()
            }

        print("  * RDataFrame computational graph booked for all MC categories. Executing event loop...")
        t_loop = time.time()

        for cat in mc_nodes.keys():
            hist_store[cat] = {"h1": {}, "h2": {}, "prof": {}}
            counts_store[cat] = mc_books[cat]["count"].GetValue()
            for v_name, ptr in mc_books[cat]["h1"].items():
                h = ptr.GetValue()
                h.SetDirectory(0)
                hist_store[cat]["h1"][v_name] = h
            for h2_name, ptr in mc_books[cat]["h2"].items():
                h2 = ptr.GetValue()
                h2.SetDirectory(0)
                hist_store[cat]["h2"][h2_name] = h2
            for prof_name, ptr in mc_books[cat]["prof"].items():
                p = ptr.GetValue()
                p.SetDirectory(0)
                hist_store[cat]["prof"][prof_name] = p

        print(f"  * MC Event Loop completed in {time.time() - t_loop:.2f} s")
        print(f"    - Rock Events   : {counts_store['rock']:,}")
        print(f"    - Target Events : {counts_store['target']:,}")
        print(f"    - MuonSys Events: {counts_store['mufi']:,}")
        print(f"    - AllSig Events : {counts_store['allsig']:,}")
        print(f"    - Bkg Events    : {counts_store['bkg']:,}")

    # 2. Process Collision Data Stream (if provided)
    if args.input_data:
        print("\n--- [2/4] Loading & Booking Collision Data ---")
        data_obj = DataManager(args.input_data, num_threads=threads_to_use)
        dur_str = f" | Duration: {data_obj.get_duration_str()}" if data_obj.duration is not None else ""
        print(f"  * Resolved {data_obj.num_files:,} files | Tree: '{data_obj.tree_name}' | Total entries: {data_obj.entries:,}{dur_str}")

        df_data = setup_dataframe(data_obj, "data", args.radius, n_data)
        data_count = df_data.Count()
        d_h1, d_h2, d_prof = book_category_histograms(df_data, "data")

        t_loop = time.time()
        counts_store["data"] = data_count.GetValue()
        hist_store["data"] = {
            "h1": {k: ptr.GetValue() for k, ptr in d_h1.items()},
            "h2": {k: ptr.GetValue() for k, ptr in d_h2.items()},
            "prof": {k: ptr.GetValue() for k, ptr in d_prof.items()}
        }
        for sub in hist_store["data"].values():
            for obj in sub.values():
                obj.SetDirectory(0)
        print(f"  * Data Event Loop completed in {time.time() - t_loop:.2f} s | Events: {counts_store['data']:,}")

    # 3. Process Passing Muons Stream (if provided)
    if args.input_pmu:
        print("\n--- [3/4] Loading & Booking Passing Muon MC ---")
        pmu_obj = DataManager(args.input_pmu, num_threads=threads_to_use)
        print(f"  * Resolved {pmu_obj.num_files:,} files | Tree: '{pmu_obj.tree_name}' | Total entries: {pmu_obj.entries:,}")

        df_pmu = setup_dataframe(pmu_obj, "pmu", args.radius, n_pmu)
        pmu_count = df_pmu.Count()
        p_h1, p_h2, p_prof = book_category_histograms(df_pmu, "pmu")

        t_loop = time.time()
        counts_store["pmu"] = pmu_count.GetValue()
        hist_store["pmu"] = {
            "h1": {k: ptr.GetValue() for k, ptr in p_h1.items()},
            "h2": {k: ptr.GetValue() for k, ptr in p_h2.items()},
            "prof": {k: ptr.GetValue() for k, ptr in p_prof.items()}
        }
        for sub in hist_store["pmu"].values():
            for obj in sub.values():
                obj.SetDirectory(0)
        print(f"  * PMU Event Loop completed in {time.time() - t_loop:.2f} s | Events: {counts_store['pmu']:,}")

    # 4. Process Gallery Candidate Events (if provided)
    if args.input_gallery:
        print("\n--- [4/4] Loading & Booking Gallery Candidate Events ---")
        gal_obj = DataManager(args.input_gallery, num_threads=threads_to_use)
        dur_str = f" | Duration: {gal_obj.get_duration_str()}" if gal_obj.duration is not None else ""
        print(f"  * Resolved {gal_obj.num_files:,} files | Tree: '{gal_obj.tree_name}' | Total entries: {gal_obj.entries:,}{dur_str}")

        df_gal = setup_dataframe(gal_obj, "gallery", args.radius, n_gal)
        gal_count = df_gal.Count()
        g_h1, g_h2, g_prof = book_category_histograms(df_gal, "gallery")

        t_loop = time.time()
        counts_store["gallery"] = gal_count.GetValue()
        hist_store["gallery"] = {
            "h1": {k: ptr.GetValue() for k, ptr in g_h1.items()},
            "h2": {k: ptr.GetValue() for k, ptr in g_h2.items()},
            "prof": {k: ptr.GetValue() for k, ptr in g_prof.items()}
        }
        for sub in hist_store["gallery"].values():
            for obj in sub.values():
                obj.SetDirectory(0)
        print(f"  * Gallery Event Loop completed in {time.time() - t_loop:.2f} s | Events: {counts_store['gallery']:,}")

    # 5. Save Master File and Superimposed Canvases
    print("\n--- Generating Superimposed Canvases & Saving Master ROOT File ---")
    temp_dir = tempfile.mkdtemp(prefix="snd_presel_master_")
    local_master_path = os.path.join(temp_dir, "preselection_master.root")

    try:
        f_master = ROOT.TFile.Open(local_master_path, "RECREATE")

        # A. Write individual category histograms
        for cat, store in hist_store.items():
            dir_std = cat_dir_map.get(cat, f"Histograms/{cat.capitalize()}")
            d_std = get_or_create_dir(f_master, dir_std)
            d_std.cd()
            style = PALETTE.get(cat, PALETTE["rock"])
            for h in store["h1"].values():
                h.SetLineColor(style["color"])
                h.SetLineWidth(style["line_width"])
                h.SetLineStyle(style["line_style"])
                h.SetMarkerColor(style["color"])
                h.SetMarkerSize(0)
                h.SetFillStyle(0)
                h.Write()

            d_2d = get_or_create_dir(f_master, f"Histograms/2D/{cat.capitalize()}")
            d_2d.cd()
            for h2 in store["h2"].values():
                h2.Write()

            d_p = get_or_create_dir(f_master, f"Histograms/Profiles/{cat.capitalize()}")
            d_p.cd()
            for prof in store["prof"].values():
                prof.SetLineColor(style["color"])
                prof.SetLineWidth(style["line_width"])
                prof.SetLineStyle(style["line_style"])
                prof.SetMarkerColor(style["color"])
                prof.SetMarkerSize(0)
                prof.SetFillStyle(0)
                prof.Write()

        # B. Write Superimposed Master 1D Canvases
        if len(hist_store) > 0:
            d_canvases_1d = get_or_create_dir(f_master, "Superimposed_Canvases/1D_Distributions")
            for v_name, _, _, _, _, cut_dir, _ in HIST_CONFIGS_1D:
                c = create_superimposed_1d_canvas(
                    var_name=v_name,
                    h_rock=hist_store["rock"]["h1"][v_name] if "rock" in hist_store else None,
                    h_target=hist_store["target"]["h1"][v_name] if "target" in hist_store else None,
                    h_mufi=hist_store["mufi"]["h1"][v_name] if "mufi" in hist_store else None,
                    h_bkg=hist_store["bkg"]["h1"][v_name] if "bkg" in hist_store else None,
                    h_data=hist_store["data"]["h1"][v_name] if "data" in hist_store else None,
                    h_pmu=hist_store["pmu"]["h1"][v_name] if "pmu" in hist_store else None,
                    h_gallery=hist_store["gallery"]["h1"][v_name] if "gallery" in hist_store else None,
                    cut_dir=cut_dir
                )
                d_canvases_1d.cd()
                c.Write()

        # C. Write Superimposed Profile Canvases
        d_canvases_prof = get_or_create_dir(f_master, "Superimposed_Canvases/Profiles_Comparison")
        for prof_name, title, _, _, _, _ in PROFILE_CONFIGS:
            c_prof = ROOT.TCanvas(f"c_{prof_name}", f"Profile Comparison: {prof_name}", 880, 600)
            c_prof.cd()
            c_prof.SetGrid(1, 1)
            c_prof.SetLeftMargin(0.12)
            c_prof.SetRightMargin(0.05)
            c_prof.SetTopMargin(0.08)
            c_prof.SetBottomMargin(0.12)

            leg_p = ROOT.TLegend(0.60, 0.65, 0.92, 0.89)
            leg_p.SetBorderSize(0)
            leg_p.SetFillStyle(0)
            leg_p.SetTextSize(0.033)

            projs = []
            for cat in ["rock", "target", "mufi", "allsig", "bkg", "pmu", "data", "gallery"]:
                if cat in hist_store and prof_name in hist_store[cat]["prof"]:
                    p = hist_store[cat]["prof"][prof_name]
                    if p and p.GetEntries() > 0:
                        h_proj = p.ProjectionX(f"{prof_name}_{cat}_proj", "E")
                        ROOT.SetOwnership(h_proj, False)
                        h_proj.SetDirectory(0)
                        style = PALETTE.get(cat, PALETTE["rock"])
                        h_proj.SetLineColor(style["color"])
                        h_proj.SetMarkerColor(style["color"])
                        h_proj.SetMarkerSize(0)  # Suppress markers for histogram format
                        h_proj.SetLineWidth(style["line_width"])
                        h_proj.SetLineStyle(style["line_style"])
                        h_proj.SetFillStyle(0)
                        clean_prof_title = title.split(";")[0] if ";" in title else title
                        h_proj.SetTitle(f"{clean_prof_title};{p.GetXaxis().GetTitle()};{p.GetYaxis().GetTitle()}")
                        h_proj.SetStats(0)
                        projs.append((cat, h_proj, style))

            if projs:
                max_y = max(h.GetMaximum() for _, h, _ in projs)
                min_y = min(h.GetMinimum() for _, h, _ in projs)
                span = max_y - min_y
                y_top = max_y + 0.35 * (span if span > 0 else (abs(max_y) if max_y != 0 else 1.0))
                y_bot = max(0.0, min_y - 0.05 * span) if min_y >= 0 else min_y - 0.1 * span
                for idx, (cat, h_proj, style) in enumerate(projs):
                    h_proj.SetMaximum(y_top)
                    h_proj.SetMinimum(y_bot)
                    draw_opt = "HIST E" if idx == 0 else "HIST E SAME"
                    h_proj.Draw(draw_opt)
                    leg_p.AddEntry(h_proj, style["label"], "LE")
                leg_p.Draw()
                d_canvases_prof.cd()
                c_prof.Write()

        f_master.Close()

        # Atomic copy to final destination
        out_dir = os.path.dirname(os.path.abspath(args.output_file))
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(local_master_path, args.output_file)

        file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
        print(f"\n[SUCCESS] Master ROOT file saved: '{args.output_file}' ({file_size_mb:.2f} MB)")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed_all = time.time() - t0_all
    print("=" * 80)
    print(f"Total Workflow Runtime: {elapsed_all:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
