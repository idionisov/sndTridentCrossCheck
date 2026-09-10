#!/usr/bin/env python3
"""
================================================================================
SND@LHC Cutflow Analysis & Plotting Tool (Cutset 8)
================================================================================
Superimposes cutflow efficiencies and expected yields across:
  - Collision Data (Run 6640)
  - Passing Muon Background MC (PMU FLUKA+Geant4)
  - Trimuon Monte Carlo (All: signal + upstream rock background)
  - Trimuon Signal Monte Carlo
  - Gallery Candidate Events

Features:
  - Automatic multi-process histogram loading over EOS
  - JSON caching for instant (<0.1s) replotting
  - Exports all 5 individual cutflows as ROOT histograms in an output ROOT file
  - Luminosity scaling (Run 6640 lumi: 0.991586 fb^-1, MC lumi: 0.025 fb^-1)
  - Standard HEP stepped curves (--style step, default) eliminating narrow-bin clutter
  - Standard HEP header (SND@LHC on top-left, integrated luminosity on top-right)
  - Optional bar chart style (--style bar) and table-embedded style (--style table)
  - 16:9 side-by-side combined publication-quality figure
================================================================================
"""

import os
import sys
import glob
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Check ROOT availability
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False

# Default Cutset 8 Cut Labels (LaTeX formatted for matplotlib)
CUT_LABELS_CUTSET8 = [
    "1. Total Input",
    r"2. DS Hits $\geq$ 8",
    r"3. US Hits $\geq$ 5",
    r"4. SciFi Hits (9/3 or 3/9)",
    r"5. SciFi Max QDC $\geq$ 5",
    r"6. DS QDC $\geq$ 500",
    r"7. US QDC $\geq$ 40",
    r"8. US Max QDC $\geq$ 7.5",
    r"9. DS/SciFi QDC $\geq$ 0.05",
    r"10. Last DS Plane = 4",
    r"11. Event $\Delta$t > 100"
]

CUT_NAMES_PLAIN = [
    "Total Input",
    "DS Hits >= 8",
    "US Hits >= 5",
    "SciFi Hits (9/3 or 3/9)",
    "SciFi Max QDC >= 5",
    "DS QDC >= 500",
    "US QDC >= 40",
    "US Max QDC >= 7.5",
    "DS/SciFi QDC >= 0.05",
    "Last DS Plane = 4",
    "Event Delta_t > 100"
]

# Luminosity Constants
LUMI_RUN6640 = 0.9915864253712848  # fb^-1
LUMI_TRIMUON_MC = 0.025            # fb^-1
SCALE_TRIMUON_DEFAULT = LUMI_RUN6640 / LUMI_TRIMUON_MC  # ~39.663457
SCALE_PMU_DEFAULT = LUMI_RUN6640 * 100116.0             # ~99273.66


# ==============================================================================
# Single File Reader Worker (Pure ROOT C-speed)
# ==============================================================================

def _read_single_root_file(args: Tuple[str, int]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    fpath, max_bins = args
    if not os.path.exists(fpath):
        return None, None

    f = ROOT.TFile.Open(fpath, "READ")
    if not f or f.IsZombie():
        if f:
            f.Close()
        return None, None

    u_counts = None
    h = f.Get("cutFlow")
    if h:
        nb = min(h.GetNbinsX(), max_bins)
        u_counts = [float(h.GetBinContent(i)) for i in range(1, nb + 1)]

    w_counts = None
    hw = f.Get("cutFlow_weighted")
    if hw:
        nb = min(hw.GetNbinsX(), max_bins)
        w_counts = [float(hw.GetBinContent(i)) for i in range(1, nb + 1)]

    f.Close()
    return u_counts, w_counts


# ==============================================================================
# Parallel Cutflow Loader & Merging
# ==============================================================================

def load_and_merge_cutflow(
    file_pattern: str,
    dataset_name: str = "",
    max_bins: int = 11,
    n_workers: int = 16,
    is_mc: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Finds matching ROOT files and sums cutFlow and cutFlow_weighted histograms in parallel.
    """
    if not HAS_ROOT:
        raise RuntimeError("PyROOT module is required to read ROOT files.")

    files = sorted(glob.glob(file_pattern))
    if not files:
        if os.path.exists(file_pattern):
            files = [file_pattern]
        else:
            print(f"[{dataset_name}] Warning: No files matched pattern: '{file_pattern}'")
            return None, None, 0

    total_files = len(files)
    print(f"[{dataset_name}] Found {total_files:,} files. Loading histograms with {n_workers} workers...")

    tot_u = np.zeros(max_bins, dtype=float)
    tot_w = np.zeros(max_bins, dtype=float)
    has_u = False
    has_w = False
    processed_count = 0
    t0 = time.time()

    tasks = [(f, max_bins) for f in files]

    if total_files > 1 and n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_read_single_root_file, t) for t in tasks]
            for fut in as_completed(futures):
                u, w = fut.result()
                processed_count += 1
                if processed_count % 200 == 0 or processed_count == total_files:
                    elapsed = time.time() - t0
                    print(f"  -> [{dataset_name}] Loaded {processed_count:,} / {total_files:,} files ({processed_count/max(0.1, elapsed):.1f} files/s)...", flush=True)

                if u is not None:
                    has_u = True
                    for i, val in enumerate(u):
                        if i < max_bins:
                            tot_u[i] += val
                if w is not None:
                    has_w = True
                    for i, val in enumerate(w):
                        if i < max_bins:
                            tot_w[i] += val
    else:
        for t in tasks:
            u, w = _read_single_root_file(t)
            processed_count += 1
            if u is not None:
                has_u = True
                for i, val in enumerate(u):
                    if i < max_bins:
                        tot_u[i] += val
            if w is not None:
                has_w = True
                for i, val in enumerate(w):
                    if i < max_bins:
                        tot_w[i] += val

    # Forward fill 11th bin for MC datasets (event delta t cut not evaluated in simulation)
    def apply_mc_forward_fill(arr: np.ndarray) -> np.ndarray:
        if is_mc and len(arr) >= 11:
            if arr[10] == 0.0 and arr[9] > 0.0:
                arr[10] = arr[9]
        return arr

    res_u = apply_mc_forward_fill(tot_u) if has_u else None
    res_w = apply_mc_forward_fill(tot_w) if has_w else None

    print(f"[{dataset_name}] Completed {total_files:,} files in {time.time()-t0:.2f}s.")
    return res_u, res_w, total_files


# ==============================================================================
# ROOT File Storage for Cutflows
# ==============================================================================

def save_cutflows_to_root(
    cache_data: Dict[str, Any],
    datasets_dict: List[Dict[str, Any]],
    output_root_path: str = "cutflow_cutset8.root",
    cut_labels: List[str] = CUT_NAMES_PLAIN
):
    """
    Stores individual raw, weighted, scaled, and efficiency cutflows as ROOT histograms in an output .root file.
    """
    if not HAS_ROOT:
        print("PyROOT module is not available. Skipping ROOT output creation.")
        return

    out_file = ROOT.TFile.Open(output_root_path, "RECREATE")
    if not out_file or out_file.IsZombie():
        print(f"Error: Failed to create output ROOT file '{output_root_path}'")
        return

    n_bins = len(cut_labels)

    # 1. Write raw unweighted and generator weighted histograms
    for key, info in cache_data.items():
        u_arr = info.get("unweighted")
        w_arr = info.get("weighted")

        if u_arr:
            h_raw = ROOT.TH1D(f"cutFlow_raw_{key}", f"Raw Unweighted Cutflow ({key});Cut Step;Events", n_bins, 0.5, n_bins + 0.5)
            for i, val in enumerate(u_arr[:n_bins]):
                h_raw.SetBinContent(i + 1, val)
                h_raw.GetXaxis().SetBinLabel(i + 1, cut_labels[i])
            h_raw.Write()

        if w_arr:
            h_w = ROOT.TH1D(f"cutFlow_weighted_{key}", f"Generator Weighted Cutflow ({key});Cut Step;Events", n_bins, 0.5, n_bins + 0.5)
            for i, val in enumerate(w_arr[:n_bins]):
                h_w.SetBinContent(i + 1, val)
                h_w.GetXaxis().SetBinLabel(i + 1, cut_labels[i])
            h_w.Write()

    # 2. Write Lumi-Scaled Expected Yields and Efficiency Histograms
    for d in datasets_dict:
        name_key = d.get("key", d["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower())
        counts = d.get("counts")
        if counts is not None and len(counts) > 0:
            # Scaled Expected Yields
            h_scaled = ROOT.TH1D(f"cutFlow_scaled_{name_key}", f"Expected Yields Scaled to Lumi ({d['name']});Cut Step;Expected Events", n_bins, 0.5, n_bins + 0.5)
            for i, val in enumerate(counts[:n_bins]):
                h_scaled.SetBinContent(i + 1, val)
                h_scaled.GetXaxis().SetBinLabel(i + 1, cut_labels[i])
            h_scaled.Write()

            # Cumulative Efficiency (%)
            if counts[0] > 0:
                h_eff = ROOT.TH1D(f"cutFlow_efficiency_{name_key}", f"Cumulative Selection Efficiency ({d['name']});Cut Step;% Remaining", n_bins, 0.5, n_bins + 0.5)
                for i, val in enumerate(counts[:n_bins]):
                    h_eff.SetBinContent(i + 1, (val / counts[0]) * 100.0)
                    h_eff.GetXaxis().SetBinLabel(i + 1, cut_labels[i])
                h_eff.Write()

            # Step-by-Step Relative Efficiency (%)
            h_rel = ROOT.TH1D(f"cutFlow_relative_{name_key}", f"Step-by-Step Passing Fraction ({d['name']});Cut Step;% Relative Step Passing", n_bins, 0.5, n_bins + 0.5)
            for i, val in enumerate(counts[:n_bins]):
                if i == 0:
                    rel_val = 100.0
                else:
                    prev = counts[i - 1]
                    rel_val = (val / prev * 100.0) if prev > 0 else 0.0
                h_rel.SetBinContent(i + 1, rel_val)
                h_rel.GetXaxis().SetBinLabel(i + 1, cut_labels[i])
            h_rel.Write()

    out_file.Close()
    print(f"\n[+] Saved all cutflow ROOT histograms to: '{output_root_path}'")


# ==============================================================================
# Helper for Formatting Numbers
# ==============================================================================

def _format_yield_label(val: float) -> str:
    if np.isnan(val) or val <= 0:
        return ""
    if val >= 1e9:
        return f"{val/1e9:.2f}B"
    elif val >= 1e6:
        return f"{val/1e6:.1f}M"
    elif val >= 1e4:
        return f"{val/1e3:.0f}k"
    elif val >= 1e3:
        return f"{val/1e3:.1f}k"
    elif val >= 10:
        return f"{val:,.0f}"
    elif val >= 1:
        return f"{val:,.1f}"
    elif val >= 0.01:
        return f"{val:,.2f}"
    else:
        return f"{val:.1e}"

def _format_eff_label(h: float) -> str:
    if np.isnan(h) or h <= 0:
        return ""
    if h >= 99.95:
        return "100%"
    elif h >= 10.0:
        return f"{h:.1f}%"
    elif h >= 1.0:
        return f"{h:.1f}%"
    elif h >= 0.1:
        return f"{h:.2f}%"
    elif h >= 0.01:
        return f"{h:.3f}%"
    else:
        return f"{h:.1e}%"


# ==============================================================================
# Step / Line Plot Renderer (The Recommended Standard HEP Presentation)
# ==============================================================================

def render_step_yields_axis(
    ax,
    datasets_dict: List[Dict[str, Any]],
    cut_labels: List[str] = CUT_LABELS_CUTSET8,
    lumi: float = LUMI_RUN6640
):
    n_bins = len(cut_labels)
    x = np.arange(n_bins)

    for d in datasets_dict:
        c = np.maximum(d["counts"], 1e-4)
        ax.step(x, c, where="mid", label=d["name"], color=d["color"], ls=d.get("ls", "-"), lw=d.get("lw", 2.2), zorder=3)
        ax.plot(x, c, color=d["color"], marker=d.get("marker", "o"), markersize=6.5, ls="none", zorder=4)

    ax.set_yscale("log")
    ax.set_ylim(1e-1, 5e10)
    ax.set_xlim(-0.4, n_bins - 0.6)
    ax.set_ylabel(r"Expected Events", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cut_labels, rotation=30, ha="right", fontsize=10.5)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="#888888", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#bbbbbb", alpha=0.45)
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=11)
    ax.tick_params(which="major", length=6, width=1.0)
    ax.tick_params(which="minor", length=3.5, width=0.8)

    # Standard HEP Top Header: Left = Experiment, Right = Integrated Luminosity
    ax.set_title(r"$\mathbf{SND@LHC}$", loc="left", fontsize=14, pad=10)
    ax.text(1.0, 1.02, rf"$\mathcal{{L}} = {lumi:.3f}\ \mathrm{{fb}}^{{-1}}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=12.5)

    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=10.5, edgecolor="#cccccc")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")


def render_step_efficiency_axis(
    ax,
    datasets_dict: List[Dict[str, Any]],
    cut_labels: List[str] = CUT_LABELS_CUTSET8,
    lumi: float = LUMI_RUN6640
):
    n_bins = len(cut_labels)
    x = np.arange(n_bins)

    for d in datasets_dict:
        c = d["counts"]
        eff = (c / c[0]) * 100.0
        ax.step(x, eff, where="mid", label=d["name"], color=d["color"], ls=d.get("ls", "-"), lw=d.get("lw", 2.2), zorder=3)
        ax.plot(x, eff, color=d["color"], marker=d.get("marker", "o"), markersize=6.5, ls="none", zorder=4)

    ax.set_ylim(0, 108)
    ax.set_xlim(-0.4, n_bins - 0.6)
    ax.set_ylabel(r"% Remaining Events", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cut_labels, rotation=30, ha="right", fontsize=10.5)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="#888888", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#bbbbbb", alpha=0.45)
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=11)
    ax.tick_params(which="major", length=6, width=1.0)
    ax.tick_params(which="minor", length=3.5, width=0.8)

    # Standard HEP Top Header: Left = Experiment, Right = Integrated Luminosity
    ax.set_title(r"$\mathbf{SND@LHC}$", loc="left", fontsize=14, pad=10)
    ax.text(1.0, 1.02, rf"$\mathcal{{L}} = {lumi:.3f}\ \mathrm{{fb}}^{{-1}}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=12.5)

    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=10.5, edgecolor="#cccccc")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")


# ==============================================================================
# Bar Chart Renderer (With Cleaned Top Headers & Adaptive Text)
# ==============================================================================

def render_bar_yields_axis(
    ax,
    datasets_dict: List[Dict[str, Any]],
    cut_labels: List[str] = CUT_LABELS_CUTSET8,
    lumi: float = LUMI_RUN6640
):
    n_bins = len(cut_labels)
    plotted_entries = [d for d in datasets_dict if d["counts"] is not None and d["counts"][0] > 0]
    if not plotted_entries:
        return

    n_ds = len(plotted_entries)
    x = np.arange(n_bins)
    total_group_width = 0.78
    width = total_group_width / n_ds

    for idx, item in enumerate(plotted_entries):
        offset = (idx - (n_ds - 1) / 2.0) * width
        counts = np.maximum(item["counts"], 1e-4)
        rects = ax.bar(
            x + offset, counts, width,
            label=f"{item['name']}",
            facecolor=item.get("facecolor", item.get("color", "#008800")),
            edgecolor=item.get("edgecolor", "none"),
            hatch=item.get("hatch", None),
            linewidth=1.0 if item.get("hatch") is not None else 0.0,
            zorder=3
        )

        is_hatched = item.get("hatch") is not None
        for i_bar, r in enumerate(rects):
            val = item["counts"][i_bar]
            txt = _format_yield_label(val)
            h = r.get_height()
            if txt and h > 0:
                bbox_args = None
                if is_hatched:
                    bbox_args = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75)

                ax.annotate(
                    txt,
                    xy=(r.get_x() + r.get_width() / 2.0, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=6.2,
                    fontfamily="sans-serif",
                    fontweight="normal",
                    color="black",
                    bbox=bbox_args,
                    zorder=4
                )

    ax.set_yscale("log")
    ax.set_ylim(1e-1, 5e10)
    ax.set_xlim(-0.55, n_bins - 0.45)
    ax.set_ylabel(r"Expected Events", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cut_labels, rotation=30, ha="right", fontsize=9.5)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="#888888", alpha=0.7, zorder=1)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#bbbbbb", alpha=0.45, zorder=1)
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=10.5)

    ax.set_title(r"$\mathbf{SND@LHC}$", loc="left", fontsize=14, pad=10)
    ax.text(1.0, 1.02, rf"$\mathcal{{L}} = {lumi:.3f}\ \mathrm{{fb}}^{{-1}}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=12.5)

    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5, edgecolor="#cccccc")

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")


def render_bar_efficiency_axis(
    ax,
    datasets_dict: List[Dict[str, Any]],
    cut_labels: List[str] = CUT_LABELS_CUTSET8,
    lumi: float = LUMI_RUN6640
):
    n_bins = len(cut_labels)
    plotted_entries = []
    for d in datasets_dict:
        c = d["counts"]
        if c is not None and len(c) > 0 and not np.isnan(c[0]) and c[0] > 0:
            eff = (c / c[0]) * 100.0
            plotted_entries.append({
                "name": d["name"],
                "eff": eff,
                "counts": c,
                "color": d.get("color", "#008800"),
                "facecolor": d.get("facecolor", d.get("color", "#008800")),
                "edgecolor": d.get("edgecolor", "none"),
                "hatch": d.get("hatch", None),
                "is_hatched": d.get("hatch") is not None
            })

    if not plotted_entries:
        return

    n_ds = len(plotted_entries)
    x = np.arange(n_bins)
    total_group_width = 0.78
    width = total_group_width / n_ds

    for idx, item in enumerate(plotted_entries):
        offset = (idx - (n_ds - 1) / 2.0) * width
        rects = ax.bar(
            x + offset, item["eff"], width,
            label=item["name"],
            facecolor=item["facecolor"],
            edgecolor=item["edgecolor"],
            hatch=item["hatch"],
            linewidth=1.0 if item["is_hatched"] else 0.0,
            zorder=3
        )

        for r in rects:
            h = r.get_height()
            txt = _format_eff_label(h)
            if txt:
                bbox_args = None
                if item["is_hatched"]:
                    bbox_args = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75)

                ax.annotate(
                    txt,
                    xy=(r.get_x() + r.get_width() / 2.0, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=6.5,
                    fontfamily="sans-serif",
                    fontweight="normal",
                    color="black",
                    bbox=bbox_args,
                    zorder=4
                )

    ax.set_ylim(0, 115)
    ax.set_xlim(-0.55, n_bins - 0.45)
    ax.set_ylabel(r"% Remaining Events", fontsize=12, fontweight="bold", fontfamily="sans-serif")
    ax.set_xticks(x)
    ax.set_xticklabels(cut_labels, rotation=30, ha="right", fontsize=9.5, fontfamily="sans-serif")

    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=10.5)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="#888888", alpha=0.7, zorder=1)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#bbbbbb", alpha=0.45, zorder=1)

    ax.set_title(r"$\mathbf{SND@LHC}$", loc="left", fontsize=14, pad=10)
    ax.text(1.0, 1.02, rf"$\mathcal{{L}} = {lumi:.3f}\ \mathrm{{fb}}^{{-1}}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=12.5)

    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5, edgecolor="#cccccc")

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")


# ==============================================================================
# Plot Driver Functions (16:9 Aspect Ratio)
# ==============================================================================

def plot_combined_cutflows(
    datasets_dict: List[Dict[str, Any]],
    output_filename: str = "cutflow_cutset8_combined.png",
    cut_labels: List[str] = CUT_LABELS_CUTSET8,
    style: str = "step",
    lumi: float = LUMI_RUN6640
):
    """
    Produces a 16:9 side-by-side combined figure.
    """
    fig, (ax_yield, ax_eff) = plt.subplots(1, 2, figsize=(16, 9), dpi=300)

    if style == "step":
        render_step_yields_axis(ax_yield, datasets_dict, cut_labels=cut_labels, lumi=lumi)
        render_step_efficiency_axis(ax_eff, datasets_dict, cut_labels=cut_labels, lumi=lumi)
    else:
        render_bar_yields_axis(ax_yield, datasets_dict, cut_labels=cut_labels, lumi=lumi)
        render_bar_efficiency_axis(ax_eff, datasets_dict, cut_labels=cut_labels, lumi=lumi)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.18, wspace=0.18)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"[+] Generated Combined 16:9 Cutflow Figure ({style} style): {output_filename}")


def plot_cutflow_efficiency_standalone(
    datasets_dict: List[Dict[str, Any]],
    output_filename: str = "cutflow_cutset8_efficiency.png",
    cut_labels: List[str] = CUT_LABELS_CUTSET8,
    style: str = "step",
    lumi: float = LUMI_RUN6640
):
    fig, ax = plt.subplots(figsize=(12, 7.0), dpi=300)
    if style == "step":
        render_step_efficiency_axis(ax, datasets_dict, cut_labels=cut_labels, lumi=lumi)
    else:
        render_bar_efficiency_axis(ax, datasets_dict, cut_labels=cut_labels, lumi=lumi)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.18)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"[+] Generated Standalone Efficiency Plot ({style} style): {output_filename}")


def plot_cutflow_yields_standalone(
    datasets_dict: List[Dict[str, Any]],
    output_filename: str = "cutflow_cutset8_yields.png",
    cut_labels: List[str] = CUT_LABELS_CUTSET8,
    style: str = "step",
    lumi: float = LUMI_RUN6640
):
    fig, ax = plt.subplots(figsize=(12, 7.0), dpi=300)
    if style == "step":
        render_step_yields_axis(ax, datasets_dict, cut_labels=cut_labels, lumi=lumi)
    else:
        render_bar_yields_axis(ax, datasets_dict, cut_labels=cut_labels, lumi=lumi)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.18)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"[+] Generated Standalone Yields Plot ({style} style): {output_filename}")


# ==============================================================================
# Comprehensive Cutflow Table
# ==============================================================================

def print_cutflow_table(
    datasets_dict: List[Dict[str, Any]],
    cut_labels: List[str] = CUT_NAMES_PLAIN
):
    print("\n" + "=" * 140)
    print("                                            CUTFLOW SUMMARY TABLE (Cutset 8)")
    print("=" * 140)
    header = f"{'Cut Step':<26}"
    for d in datasets_dict:
        header += f" | {d['name']:<24}"
    print(header)
    print("-" * 140)

    for idx, label in enumerate(cut_labels):
        row = f"{label:<26}"
        for d in datasets_dict:
            c = d["counts"]
            if c is not None and len(c) > idx and not np.isnan(c[idx]):
                eff = (c[idx] / c[0]) * 100.0 if c[0] > 0 else 0.0
                val = c[idx]
                if val >= 10000:
                    val_str = f"{val:,.0f} ({eff:.1f}%)"
                elif val >= 10:
                    val_str = f"{val:,.1f} ({eff:.2f}%)"
                elif val >= 0.01:
                    val_str = f"{val:,.3f} ({eff:.3f}%)"
                else:
                    val_str = f"{val:.2e} ({eff:.3f}%)"
                row += f" | {val_str:<24}"
            else:
                row += f" | {'N/A':<24}"
        print(row)
    print("=" * 140)

    # Print Step-by-Step Relative Rejection
    print("\n" + "-" * 140)
    print("                                    STEP-BY-STEP RELATIVE PASSING FRACTIONS")
    print("-" * 140)
    header = f"{'Cut Step':<26}"
    for d in datasets_dict:
        header += f" | {d['name']:<24}"
    print(header)
    print("-" * 140)

    for idx, label in enumerate(cut_labels):
        row = f"{label:<26}"
        for d in datasets_dict:
            c = d["counts"]
            if c is not None and len(c) > idx and not np.isnan(c[idx]):
                if idx == 0:
                    rel_str = "100.00% (Init)"
                else:
                    prev = c[idx - 1]
                    rel = (c[idx] / prev * 100.0) if prev > 0 else 0.0
                    rel_str = f"{rel:.2f}%"
                row += f" | {rel_str:<24}"
            else:
                row += f" | {'N/A':<24}"
        print(row)
    print("=" * 140 + "\n")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cutflow comparison and reproduction script for Cutset 8 (Rock Trident Preselection)."
    )
    parser.add_argument(
        "--data",
        default="/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw-????_cutset8_hough_?.root",
        help="Collision Data files pattern (Run 6640)"
    )
    parser.add_argument(
        "--tri-all",
        default="/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP-???_hough_?_cutset8.root",
        help="Trimuon All (Signal + Rock Background) MC pattern"
    )
    parser.add_argument(
        "--tri-signal",
        default="/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100_signal/hough_trimuon_filtered_digCPP-???_cutset8.root",
        help="Trimuon Signal MC pattern"
    )
    parser.add_argument(
        "--pmu",
        default="/eos/user/i/idioniso/sndMuTri/data/cutset8/pmu/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_hough.root",
        help="Passing Muon Background MC (PMU) file pattern"
    )
    parser.add_argument(
        "--gallery",
        default="/eos/user/i/idioniso/sndMuTri/data/cutset8/gallery/mu3_search_run_??????_????_hough_cutset8.root",
        help="Gallery candidate files pattern"
    )
    parser.add_argument(
        "-o", "--output",
        default="cutflow_cutset8_combined.png",
        help="Output plot filename for combined 16:9 side-by-side figure (default: %(default)s)"
    )
    parser.add_argument(
        "--eff-output",
        default="cutflow_cutset8_efficiency.png",
        help="Output plot filename for standalone efficiency chart (default: %(default)s)"
    )
    parser.add_argument(
        "--yields-output",
        default="cutflow_cutset8_yields.png",
        help="Output plot filename for standalone yields chart (default: %(default)s)"
    )
    parser.add_argument(
        "--root-output",
        default="cutflow_cutset8.root",
        help="Output ROOT file to store all cutflow histograms (default: %(default)s)"
    )
    parser.add_argument(
        "--no-root-output",
        action="store_true",
        help="Disable saving ROOT histograms file"
    )
    parser.add_argument(
        "--style",
        choices=["step", "bar"],
        default="step",
        help="Plot visualization style: 'step' (recommended clean HEP curves) or 'bar' (grouped bars) (default: %(default)s)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "signal", "both"],
        default="both",
        help="Which trimuon MC cutflow to display (default: %(default)s)"
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=LUMI_RUN6640,
        help=f"Target Data Luminosity in fb^-1 (default: {LUMI_RUN6640:.6f})"
    )
    parser.add_argument(
        "--scale-tri",
        type=float,
        default=SCALE_TRIMUON_DEFAULT,
        help=f"Scaling factor for Trimuon MC (default: {SCALE_TRIMUON_DEFAULT:.4f} = lumi / 0.025 fb^-1)"
    )
    parser.add_argument(
        "--scale-pmu",
        type=float,
        default=SCALE_PMU_DEFAULT,
        help=f"Scaling factor for Passing Muon MC (default: {SCALE_PMU_DEFAULT:.2f} = lumi * 100116)"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=32,
        help="Number of parallel worker processes for scanning EOS (default: %(default)s)"
    )
    parser.add_argument(
        "--cache-file",
        default="cutflow_cache_cutset8.json",
        help="JSON file path for storing / loading merged cutflow arrays"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cutflows directly from JSON cache file if present"
    )
    parser.add_argument(
        "--unweighted-mc",
        action="store_true",
        help="Use raw unweighted MC counts instead of weighted histograms"
    )

    args = parser.parse_args()

    cache_data = {}
    if args.use_cache and os.path.exists(args.cache_file):
        print(f"Loading cutflow arrays from JSON cache: '{args.cache_file}'...")
        with open(args.cache_file, "r") as f_c:
            cache_data = json.load(f_c)

    def get_or_load(key: str, pattern: str, is_mc: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
        if key in cache_data:
            c_u = np.array(cache_data[key]["unweighted"], dtype=float) if cache_data[key]["unweighted"] is not None else None
            c_w = np.array(cache_data[key]["weighted"], dtype=float) if cache_data[key]["weighted"] is not None else None
            n_f = cache_data[key].get("n_files", 0)
            return c_u, c_w, n_f
        
        u, w, n = load_and_merge_cutflow(pattern, dataset_name=key, max_bins=11, n_workers=args.threads, is_mc=is_mc)
        cache_data[key] = {
            "unweighted": u.tolist() if u is not None else None,
            "weighted": w.tolist() if w is not None else None,
            "n_files": n
        }
        return u, w, n

    # 1. Load Data
    u_data, _, n_data = get_or_load("data", args.data, is_mc=False)
    data_counts = u_data

    # 2. Load Passing Muon MC (PMU)
    u_pmu, w_pmu, n_pmu = get_or_load("pmu", args.pmu, is_mc=True)
    pmu_base = u_pmu if (args.unweighted_mc or w_pmu is None) else w_pmu
    pmu_counts = (pmu_base * args.scale_pmu) if pmu_base is not None else None

    # 3. Load Trimuon MC (All)
    u_tri_all, w_tri_all, n_tri_all = get_or_load("trimuon_all", args.tri_all, is_mc=True)
    tri_all_base = u_tri_all if (args.unweighted_mc or w_tri_all is None) else w_tri_all
    tri_all_counts = (tri_all_base * args.scale_tri) if tri_all_base is not None else None

    # 4. Load Trimuon Signal MC
    u_tri_sig, w_tri_sig, n_tri_sig = get_or_load("trimuon_signal", args.tri_signal, is_mc=True)
    tri_sig_base = u_tri_sig if (args.unweighted_mc or w_tri_sig is None) else w_tri_sig
    tri_sig_counts = (tri_sig_base * args.scale_tri) if tri_sig_base is not None else None

    # 5. Load Gallery Data
    u_gal, _, n_gal = get_or_load("gallery", args.gallery, is_mc=False)
    gal_counts = u_gal

    # Save to JSON cache
    try:
        with open(args.cache_file, "w") as f_c:
            json.dump(cache_data, f_c, indent=2)
        print(f"Updated cache file: '{args.cache_file}'")
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")

    # Build plotting dictionary with customized styles
    datasets_to_plot = []
    if data_counts is not None:
        datasets_to_plot.append({
            "key": "data",
            "name": "Data (Run 6640)",
            "counts": data_counts,
            "color": "#0018a8",
            "facecolor": "#0018a8",
            "edgecolor": "none",
            "marker": "o",
            "ls": "-",
            "lw": 2.2,
            "hatch": None
        })
    if pmu_counts is not None:
        datasets_to_plot.append({
            "key": "pmu",
            "name": "PMU MC",
            "counts": pmu_counts,
            "color": "#c40000",
            "facecolor": "#c40000",
            "edgecolor": "none",
            "marker": "s",
            "ls": "-",
            "lw": 2.2,
            "hatch": None
        })

    if args.mode in ["all", "both"] and tri_all_counts is not None:
        datasets_to_plot.append({
            "key": "trimuon_all",
            "name": "Trimuon MC (All)",
            "counts": tri_all_counts,
            "color": "#2e7d32",
            "facecolor": "#d5edd5",
            "edgecolor": "#008800",
            "marker": "^",
            "ls": "--",
            "lw": 2.0,
            "hatch": "///"
        })

    if args.mode in ["signal", "both"] and tri_sig_counts is not None:
        datasets_to_plot.append({
            "key": "trimuon_signal",
            "name": "Trimuon MC (Signal)",
            "counts": tri_sig_counts,
            "color": "#00a000",
            "facecolor": "#00a000",
            "edgecolor": "none",
            "marker": "D",
            "ls": "-",
            "lw": 2.4,
            "hatch": None
        })

    if gal_counts is not None:
        datasets_to_plot.append({
            "key": "gallery",
            "name": "Gallery Data",
            "counts": gal_counts,
            "color": "#8a008a",
            "facecolor": "#8a008a",
            "edgecolor": "none",
            "marker": "*",
            "ls": "-.",
            "lw": 2.0,
            "hatch": None
        })

    # Print summary tables
    print_cutflow_table(datasets_to_plot)

    # 1. Generate Combined 16:9 Figure
    plot_combined_cutflows(datasets_to_plot, output_filename=args.output, style=args.style, lumi=args.lumi)

    # 2. Generate standalone individual figures
    plot_cutflow_efficiency_standalone(datasets_to_plot, output_filename=args.eff_output, style=args.style, lumi=args.lumi)
    plot_cutflow_yields_standalone(datasets_to_plot, output_filename=args.yields_output, style=args.style, lumi=args.lumi)

    # 3. Save all cutflows as ROOT histograms in an output ROOT file
    if not args.no_root_output and args.root_output:
        save_cutflows_to_root(cache_data, datasets_to_plot, output_root_path=args.root_output)


if __name__ == "__main__":
    main()
