#!/usr/bin/env python3
"""
================================================================================
Plot 2D Topological Criteria Scan (Cutset 8)
================================================================================
Reads 'topological_criteria_scan_cutset8.root' and creates the raw 2D Pareto scatter
plot of:
  - X-axis: Surviving collision data events (Run 6640, 0.992 fb^-1)
  - Y-axis: Rock trident selection efficiency (%)
  - Z-axis (Color): Surviving Gallery trident events (out of 744)
================================================================================
"""

import os
import json
import argparse
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize

DEFAULT_INPUT_ROOT = "topological_criteria_scan_cutset8.root"
DEFAULT_OUTPUT_IMG = "topological_criteria_scan_cutset8_raw.png"


def plot_topological_scan_raw(
    root_file: str = DEFAULT_INPUT_ROOT,
    output_png: str = DEFAULT_OUTPUT_IMG,
    max_data_events: float = 3000.0,
    cmap: str = "turbo"
):
    print(f"[*] Reading '{root_file}'...")
    with uproot.open(root_file) as f:
        df = f["topological_scan"].arrays(library="pd")
        meta = json.loads(str(f["metadata"])) if "metadata" in f else {}

    print(f"    Loaded {len(df):,} criteria configurations.")

    # Extract axes data
    # X: Surviving collision data events
    x = df["data_events"].to_numpy()
    
    # Y: Rock trident selection efficiency (%)
    if "eff_direct_rock" in df.columns:
        y = df["eff_direct_rock"].to_numpy()
    elif "rock_eff_pct" in df.columns:
        y = df["rock_eff_pct"].to_numpy()
    else:
        y = df["eff_total_signal"].to_numpy()

    # Z: Surviving gallery candidate events
    z = df["gallery_events"].to_numpy()
    tot_gal = meta.get("sample_statistics", {}).get("total_input_gallery_events", int(df["gallery_events"].max()))
    lumi_full = meta.get("normalization", {}).get("target_luminosity_fb", 0.991586)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

    # Filter/Zoom range for X if specified
    if max_data_events is not None and max_data_events > 0:
        mask = x <= max_data_events
    else:
        mask = np.ones(len(x), dtype=bool)

    x_plot = x[mask]
    y_plot = y[mask]
    z_plot = z[mask]

    # Scatter plot
    sc = ax.scatter(
        x_plot,
        y_plot,
        c=z_plot,
        cmap=cmap,
        s=45,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(f"Surviving Gallery Trident Events (out of {tot_gal})", fontsize=12, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=11)

    # Labels and Limits
    ax.set_xlabel(f"Surviving collision data events ($\mathcal{{L}} = {lumi_full:.3f}\\ \\mathrm{{fb}}^{{-1}}$)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Rock trident selection efficiency (%)", fontsize=13, fontweight="bold", labelpad=8)
    
    ax.set_ylim(0, 105)
    if max_data_events is not None and max_data_events > 0:
        ax.set_xlim(0, max_data_events)
    else:
        ax.set_xlim(0, max(x) * 1.05)

    # Grid & Ticks
    ax.grid(True, which="major", linestyle="--", alpha=0.5, color="gray", zorder=1)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.25, color="gray", zorder=1)
    ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)

    # HEP Top Headers
    ax.text(
        0.0, 1.02,
        r"$\mathbf{SND@LHC}$",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left"
    )
    ax.text(
        1.0, 1.02,
        f"$\mathcal{{L}} = {lumi_full:.3f}\\ \\mathrm{{fb}}^{{-1}}$ (Run 6640)",
        transform=ax.transAxes,
        fontsize=13,
        va="bottom",
        ha="right"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved plot to '{output_png}'.")


def main():
    parser = argparse.ArgumentParser(description="Plot raw 2D Topological Criteria Pareto Map.")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT_ROOT, help="Input ROOT file (default: %(default)s)")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_IMG, help="Output PNG file (default: %(default)s)")
    parser.add_argument("--max-data", type=float, default=3000.0, help="Max data events on X-axis (default: %(default)s, pass 0 for full range)")
    parser.add_argument("--cmap", default="turbo", help="Colormap for gallery retention (default: %(default)s)")

    args = parser.parse_args()
    max_d = args.max_data if args.max_data > 0 else None
    plot_topological_scan_raw(args.input, args.output, max_data_events=max_d, cmap=args.cmap)


if __name__ == "__main__":
    main()
