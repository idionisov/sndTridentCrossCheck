#!/usr/bin/env python3
"""
================================================================================
Comprehensive Topological Scan & Significance Findings Plot (Cutset 8)
================================================================================
Generates a 2-panel 16:9 side-by-side visualization:
  - Left Panel : Full Log-Scale Landscape: Rock Trident Efficiency vs Data Background,
                 color-coded by Asimov Discovery Significance Z_data.
  - Right Panel: Sweet-Spot Zoom (N_data <= 2500): Direct Rock Efficiency vs Data Background,
                 color-coded by Run 6640 Visual Gallery Survival (out of 66).
Highlighted Key Pareto Benchmarks with Non-Overlapping Callout Annotations.
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
from matplotlib.patches import Rectangle

DEFAULT_INPUT_ROOT = "topological_criteria_scan_cutset8.root"
DEFAULT_OUTPUT_IMG = "topological_scan_significance_findings.png"


def plot_significance_findings(
    root_file: str = DEFAULT_INPUT_ROOT,
    output_png: str = DEFAULT_OUTPUT_IMG
):
    print(f"[*] Loading '{root_file}'...")
    with uproot.open(root_file) as f:
        df = f["topological_scan"].arrays(library="pd")
        meta = json.loads(str(f["metadata"])) if "metadata" in f else {}

    print(f"    Loaded {len(df):,} criteria configurations.")

    # Key columns
    x_data = df["data_events"].to_numpy()
    y_rock_eff = df["eff_direct_rock"].to_numpy()
    z_sig_data = df["sig_data_direct_rock"].to_numpy()
    z_gal_6640 = df["gallery_6640_events"].to_numpy() if "gallery_6640_events" in df.columns else df["gallery_events"].to_numpy()

    # Benchmark configurations to highlight with clean offsets
    benchmarks = [
        {
            "label_pattern": "[SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)",
            "title": "Optimal Balanced Trade-Off",
            "xytext": (-150, 18),
            "color": "#E63946",
            "marker": "*",
            "size": 240
        },
        {
            "label_pattern": "(SF_3 | DS_3) & (SF:AND & DS:OR)",
            "title": "Max Z_data Optimum",
            "xytext": (25, 15),
            "color": "#2A9D8F",
            "marker": "D",
            "size": 130
        },
        {
            "label_pattern": "[SF(3,1) | DS(3,1)] & (SF:AND & DS:OR)",
            "title": "High Z_data Pure Mode",
            "xytext": (-200, -45),
            "color": "#E76F51",
            "marker": "^",
            "size": 130
        },
        {
            "label_pattern": "SF(3,3)_AND & DS(1,0)_AND",
            "title": "Ultra-High Purity (N <= 100)",
            "xytext": (25, 20),
            "color": "#7209B7",
            "marker": "s",
            "size": 120
        }
    ]

    # Setup 16:9 Dual Panel Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)

    # -------------------------------------------------------------------------
    # Left Panel: Full Log-Scale Landscape colored by Z_data
    # -------------------------------------------------------------------------
    sc1 = ax1.scatter(
        x_data,
        y_rock_eff,
        c=z_sig_data,
        cmap="viridis",
        s=38,
        alpha=0.80,
        edgecolors="black",
        linewidths=0.4,
        zorder=3
    )

    cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.02, fraction=0.046)
    cbar1.set_label(r"Asimov Discovery Significance $\mathcal{Z}_{\mathrm{data}}$ ($\sigma$)", fontsize=12, fontweight="bold", labelpad=10)
    cbar1.ax.tick_params(labelsize=11)

    ax1.set_xscale("log")
    ax1.set_xlim(10, 1.2e7)
    ax1.set_ylim(0, 105)
    ax1.set_xlabel(r"Surviving Collision Data Events ($B_{\mathrm{data}}$)", fontsize=13, fontweight="bold", labelpad=8)
    ax1.set_ylabel(r"Direct Rock Trident Selection Efficiency (%)", fontsize=13, fontweight="bold", labelpad=8)
    ax1.grid(True, which="major", linestyle="--", alpha=0.5, color="gray", zorder=1)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", linestyle=":", alpha=0.20, color="gray", zorder=1)
    ax1.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
    ax1.tick_params(axis="both", which="minor", length=3, width=0.8)

    # Top Headers
    ax1.text(0.0, 1.02, r"$\mathbf{SND@LHC}$", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax1.text(1.0, 1.02, r"$\int \mathcal{L}\,\mathrm{dt} = 0.991\ \mathrm{fb}^{-1},\ \sqrt{s} = 13.6\ \mathrm{TeV}$", transform=ax1.transAxes, fontsize=12, va="bottom", ha="right")

    # Title box
    ax1.text(
        0.04, 0.93,
        "Full Topological Search Space (3,366 Criteria)\nDiscovery Significance against Collision Data",
        transform=ax1.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.90),
        zorder=5
    )

    # -------------------------------------------------------------------------
    # Right Panel: Sweet-Spot Zoom (N_data <= 2,500) colored by Run 6640 Gallery
    # -------------------------------------------------------------------------
    zoom_mask = x_data <= 2500
    x_zoom = x_data[zoom_mask]
    y_zoom = y_rock_eff[zoom_mask]
    z_zoom = z_gal_6640[zoom_mask]

    sc2 = ax2.scatter(
        x_zoom,
        y_zoom,
        c=z_zoom,
        cmap="turbo",
        s=48,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )

    cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.02, fraction=0.046)
    cbar2.set_label("Run 6640 Visual Gallery Survival (out of 66)", fontsize=12, fontweight="bold", labelpad=10)
    cbar2.ax.tick_params(labelsize=11)

    ax2.set_xlim(0, 2500)
    ax2.set_ylim(0, 75)
    ax2.set_xlabel(r"Surviving Collision Data Events ($B_{\mathrm{data}}$ in Run 6640)", fontsize=13, fontweight="bold", labelpad=8)
    ax2.set_ylabel(r"Direct Rock Trident Selection Efficiency (%)", fontsize=13, fontweight="bold", labelpad=8)
    ax2.grid(True, which="major", linestyle="--", alpha=0.5, color="gray", zorder=1)
    ax2.minorticks_on()
    ax2.grid(True, which="minor", linestyle=":", alpha=0.25, color="gray", zorder=1)
    ax2.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
    ax2.tick_params(axis="both", which="minor", length=3, width=0.8)

    # Top Headers
    ax2.text(0.0, 1.02, r"$\mathbf{SND@LHC}$", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax2.text(1.0, 1.02, r"$\int \mathcal{L}\,\mathrm{dt} = 0.991\ \mathrm{fb}^{-1}$ (Run 6640)", transform=ax2.transAxes, fontsize=12, va="bottom", ha="right")

    # Shaded Sweet Spot Area
    rect = Rectangle((0, 44), 2000, 26, linewidth=1.2, edgecolor="#E63946", facecolor="#E63946", alpha=0.08, linestyle="--", zorder=2)
    ax2.add_patch(rect)
    ax2.text(80, 67.5, r"$\mathbf{Optimal\ Discovery\ Sweet\ Spot}$ ($N_{\mathrm{data}} \leq 2000,\ \epsilon_{\mathrm{rock}} \geq 45\%$)", fontsize=10, color="#B7094C", fontweight="bold", zorder=4)

    # Highlight Benchmarks on Right Panel
    for bm in benchmarks:
        match = df[df["clean_label"] == bm["label_pattern"]]
        if match.empty:
            continue
        row = match.iloc[0]
        bx, by = row["data_events"], row["eff_direct_rock"]
        bz_data, bz_mc = row["sig_data_direct_rock"], row["sig_mc_direct_rock"]
        bgal_6640 = int(row["gallery_6640_events"]) if "gallery_6640_events" in row else int(row["gallery_events"])

        # Plot prominent marker
        ax2.scatter(
            [bx], [by],
            marker=bm["marker"],
            s=bm["size"],
            color=bm["color"],
            edgecolors="black",
            linewidths=1.2,
            zorder=6
        )

        # Callout Annotation Box
        ann_text = (
            f"{bm['label_pattern']}\n"
            f"Data: {bx} | Sim Bkg: {row['bkg_mc_total']:.1f} | Rock Eff: {by:.1f}%\n"
            f"Gal 6640: {bgal_6640}/66 ({bgal_6640/66*100:.1f}%) | " + r"$\mathcal{Z}_{\mathrm{data}} = " + f"{bz_data:.1f}" + r"\sigma$ | $\mathcal{Z}_{\mathrm{MC}} = " + f"{bz_mc:.1f}" + r"\sigma$"
        )

        ax2.annotate(
            ann_text,
            xy=(bx, by),
            xytext=bm["xytext"],
            textcoords="offset points",
            fontsize=8.5,
            fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDF0", edgecolor=bm["color"], linewidth=1.2, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color=bm["color"], lw=1.2),
            zorder=7
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully generated findings visualization: '{output_png}'.")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive findings plot for topological scan.")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_IMG)
    args = parser.parse_args()

    plot_significance_findings(args.input, args.output)


if __name__ == "__main__":
    main()
