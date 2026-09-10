#!/usr/bin/env python3
"""
================================================================================
Plot Kolmogorov-Smirnov (KS) CDFs & Compatibility Summary (Cutset 8)
================================================================================
For the optimal criterion: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)

Generates:
  1. Multi-panel Empirical CDF comparison (Data vs. Weighted MC) highlighting D-stat.
  2. Summary bar chart of D-statistics and p-values across all kinematic variables.
================================================================================
"""

import os
import sys
import time
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Paths
CACHE_TRACKING_PATH = "/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/data_6640_tracking.parquet"
MC_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet"
PMU_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/pmu/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_hough.parquet"

OUTPUT_CDF_PNG = "ks_empirical_cdfs_optimal_criterion.png"
OUTPUT_SUMMARY_PNG = "ks_summary_barchart_optimal_criterion.png"

LUMI_FULL = 0.9915864253712848
LUMI_MC = 0.0250
SCALE_MC = LUMI_FULL / LUMI_MC
SCALE_PMU = LUMI_FULL * 100116.0

TRACKING_COLS = [
    "n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz",
    "xz_sf_m1", "xz_sf_c1", "yz_sf_m1", "yz_sf_c1",
    "xz_sf_m2", "xz_sf_c2", "yz_sf_m2", "yz_sf_c2",
    "xz_sf_m3", "xz_sf_c3", "yz_sf_m3", "yz_sf_c3",
    "xz_ds_m1", "xz_ds_c1", "yz_ds_m1", "yz_ds_c1",
    "xz_ds_m2", "xz_ds_c2", "yz_ds_m2", "yz_ds_c2",
    "xz_ds_m3", "xz_ds_c3", "yz_ds_m3", "yz_ds_c3"
]

KS_VARIABLES = [
    ("yz_sf_c1", "SciFi Track Intercept y(0)", r"SciFi $y(0)$ [cm]", (-45, 45)),
    ("xz_sf_c1", "SciFi Track Intercept x(0)", r"SciFi $x(0)$ [cm]", (-45, 45)),
    ("delta_slope_ds_xz_12", "DS Track Angular Separation (XZ)", r"DS $|\Delta\mathrm{slope}_{xz}|$ [rad]", (0.0, 0.10)),
    ("delta_slope_sf_xz_12", "SciFi Track Angular Separation (XZ)", r"SciFi $|\Delta\mathrm{slope}_{xz}|$ [rad]", (0.0, 0.10)),
    ("yz_ds_m1", "DS Track Slope dy/dz", r"DS Slope $dy/dz$", (-0.10, 0.10)),
    ("n_lines_ds_yz", "DS YZ Multiplicity", r"DS $N_{\mathrm{tracks}}(YZ)$", (0.5, 5.5)),
    ("yz_sf_m1", "SciFi Track Slope dy/dz", r"SciFi Slope $dy/dz$", (-0.10, 0.10)),
    ("xz_sf_m1", "SciFi Track Slope dx/dz", r"SciFi Slope $dx/dz$", (-0.10, 0.10)),
]


def weighted_ks_cdf(data_vals: np.ndarray, mc_vals: np.ndarray, mc_weights: np.ndarray):
    d_clean = data_vals[~np.isnan(data_vals)]
    mc_mask = ~np.isnan(mc_vals) & ~np.isnan(mc_weights) & (mc_weights > 0)
    m_clean = mc_vals[mc_mask]
    w_clean = mc_weights[mc_mask]

    n_d = len(d_clean)
    n_m = len(m_clean)

    if n_d == 0 or n_m == 0:
        return np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), 1.0, 0.0, 0.0

    sum_w = np.sum(w_clean)
    sum_w2 = np.sum(w_clean ** 2)
    n_eff_mc = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else n_m
    n_eff = (n_d * n_eff_mc) / (n_d + n_eff_mc)

    all_vals = np.concatenate([d_clean, m_clean])
    all_types = np.concatenate([np.ones(n_d), np.zeros(n_m)])
    all_weights = np.concatenate([np.ones(n_d), w_clean])

    order = np.argsort(all_vals)
    sorted_vals = all_vals[order]
    sorted_types = all_types[order]
    sorted_weights = all_weights[order]

    data_weights = np.where(sorted_types == 1, sorted_weights, 0.0)
    mc_weights_cum = np.where(sorted_types == 0, sorted_weights, 0.0)

    cdf_data = np.cumsum(data_weights) / n_d
    cdf_mc = np.cumsum(mc_weights_cum) / sum_w

    diff = np.abs(cdf_data - cdf_mc)
    max_idx = np.argmax(diff)
    d_stat = float(diff[max_idx])
    x_at_max_d = sorted_vals[max_idx]

    en = np.sqrt(n_eff)
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    p_val = float(stats.kstwobign.sf(lambda_val))
    
    return sorted_vals, cdf_data, cdf_mc, d_stat, p_val, x_at_max_d


def check_nc(df: pd.DataFrame, prefix: str, proj: str, z_min: float, z_max: float, strict: bool = True) -> np.ndarray:
    n = df[f"n_lines_{prefix}_{proj}"].to_numpy()
    m1 = np.nan_to_num(df[f"{proj}_{prefix}_m1"].to_numpy())
    c1 = np.nan_to_num(df[f"{proj}_{prefix}_c1"].to_numpy())
    m2 = np.nan_to_num(df[f"{proj}_{prefix}_m2"].to_numpy())
    c2 = np.nan_to_num(df[f"{proj}_{prefix}_c2"].to_numpy())
    m3 = np.nan_to_num(df[f"{proj}_{prefix}_m3"].to_numpy())
    c3 = np.nan_to_num(df[f"{proj}_{prefix}_c3"].to_numpy())

    x1_min, x1_max = m1 * z_min + c1, m1 * z_max + c1
    x2_min, x2_max = m2 * z_min + c2, m2 * z_max + c2
    x3_min, x3_max = m3 * z_min + c3, m3 * z_max + c3

    no_cross_12 = ((x1_min - x2_min) * (x1_max - x2_max)) > 0
    no_cross_13 = ((x1_min - x3_min) * (x1_max - x3_max)) > 0
    no_cross_23 = ((x2_min - x3_min) * (x2_max - x3_max)) > 0

    pass_3 = (n >= 3) & no_cross_12 & no_cross_13 & no_cross_23
    pass_2 = (n == 2) & no_cross_12

    if strict:
        return pass_3 | pass_2
    return pass_3 | pass_2 | (n <= 1)


def compute_derived_kinematics(df: pd.DataFrame):
    df["delta_slope_sf_xz_12"] = np.abs(df["xz_sf_m1"] - df["xz_sf_m2"])
    df["delta_slope_sf_yz_12"] = np.abs(df["yz_sf_m1"] - df["yz_sf_m2"])
    df["delta_slope_ds_xz_12"] = np.abs(df["xz_ds_m1"] - df["xz_ds_m2"])
    df["delta_slope_ds_yz_12"] = np.abs(df["yz_ds_m1"] - df["yz_ds_m2"])


def build_criterion_mask(df: pd.DataFrame) -> np.ndarray:
    nx = df["n_lines_sf_xz"].to_numpy()
    ny = df["n_lines_sf_yz"].to_numpy()
    nx_ds = df["n_lines_ds_xz"].to_numpy()
    ny_ds = df["n_lines_ds_yz"].to_numpy()

    sf_3_1 = ((nx >= 3) & (ny >= 1)) | ((ny >= 3) & (nx >= 1))
    ds_3_1 = ((nx_ds >= 3) & (ny_ds >= 1)) | ((ny_ds >= 3) & (nx_ds >= 1))

    nc_sf_xz_s = check_nc(df, "sf", "xz", 260.0, 355.0, strict=True)
    nc_sf_yz_s = check_nc(df, "sf", "yz", 260.0, 355.0, strict=True)
    nc_ds_xz_s = check_nc(df, "ds", "xz", 460.0, 590.0, strict=True)
    nc_ds_yz_s = check_nc(df, "ds", "yz", 460.0, 590.0, strict=True)

    sf_nc_or = nc_sf_xz_s | nc_sf_yz_s
    ds_nc_or = nc_ds_xz_s | nc_ds_yz_s

    return (sf_3_1 | ds_3_1) & (sf_nc_or & ds_nc_or)


def main():
    print("=" * 90)
    print("   PLOTTING KOLMOGOROV-SMIRNOV CDFS FOR: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)")
    print("=" * 90)

    # 1. Load PMU MC
    df_pmu = pq.read_table(PMU_PATH, columns=TRACKING_COLS + ["mc_weight"]).to_pandas()
    df_pmu["final_weight"] = df_pmu["mc_weight"] * SCALE_PMU
    compute_derived_kinematics(df_pmu)
    pmu_mask = build_criterion_mask(df_pmu)
    df_pmu_after = df_pmu[pmu_mask].copy()

    # 2. Load Trimuon MC
    df_mc = pq.read_table(MC_PATH, columns=TRACKING_COLS + ["is_signal", "mc_weight"]).to_pandas()
    df_mc["final_weight"] = df_mc["mc_weight"] * SCALE_MC
    compute_derived_kinematics(df_mc)
    mc_mask = build_criterion_mask(df_mc)
    df_mc_bkg_after = df_mc[(df_mc["is_signal"] == 0) & mc_mask].copy()
    df_mc_sig_after = df_mc[(df_mc["is_signal"] == 1) & mc_mask].copy()

    # 3. Load Collision Data Candidates
    tbl_n = pq.read_table(CACHE_TRACKING_PATH, columns=["n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz"])
    nx_dt = tbl_n["n_lines_sf_xz"].to_numpy()
    ny_dt = tbl_n["n_lines_sf_yz"].to_numpy()
    nx_ds_dt = tbl_n["n_lines_ds_xz"].to_numpy()
    ny_ds_dt = tbl_n["n_lines_ds_yz"].to_numpy()

    sf_3_1_dt = ((nx_dt >= 3) & (ny_dt >= 1)) | ((ny_dt >= 3) & (nx_dt >= 1))
    ds_3_1_dt = ((nx_ds_dt >= 3) & (ny_ds_dt >= 1)) | ((ny_ds_dt >= 3) & (nx_ds_dt >= 1))
    cand_indices = np.where(sf_3_1_dt | ds_3_1_dt)[0]

    dset_track = ds.dataset(CACHE_TRACKING_PATH, format="parquet")
    df_cand = dset_track.take(cand_indices, columns=TRACKING_COLS).to_pandas()
    compute_derived_kinematics(df_cand)

    opt_mask_cand = build_criterion_mask(df_cand)
    df_data_after = df_cand[opt_mask_cand].copy()

    print(f"Loaded: Data = {len(df_data_after):,} events | Sim Bkg = {df_mc_bkg_after['final_weight'].sum():.1f} | Signal = {df_mc_sig_after['final_weight'].sum():.1f}")

    # 4. Generate 8-Panel CDF Plot
    fig = plt.figure(figsize=(22, 11), dpi=300)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.25)

    summary_rows = []

    for i_var, (vname, vtitle, vlabel, vrange) in enumerate(KS_VARIABLES):
        ax = fig.add_subplot(gs[i_var // 4, i_var % 4])

        d_vals = df_data_after[vname].to_numpy()
        b_vals = df_mc_bkg_after[vname].to_numpy()
        b_w = df_mc_bkg_after["final_weight"].to_numpy()
        p_vals = df_pmu_after[vname].to_numpy()
        p_w = df_pmu_after["final_weight"].to_numpy()

        mc_vals = np.concatenate([b_vals, p_vals])
        mc_w = np.concatenate([b_w, p_w])

        vals, cdf_d, cdf_m, d_stat, p_val, x_max_d = weighted_ks_cdf(d_vals, mc_vals, mc_w)

        # Plot CDFs
        ax.plot(vals, cdf_d, color="#1D3557", linewidth=2.2, label=r"$\mathrm{Data\ CDF}\ F_{\mathrm{data}}(x)$")
        ax.plot(vals, cdf_m, color="#E63946", linewidth=2.2, linestyle="--", label=r"$\mathrm{Sim\ Bkg\ CDF}\ F_{\mathrm{MC}}(x)$")

        # Vertical line for max D-statistic
        idx_max = np.where(vals == x_max_d)[0][0]
        y_d = cdf_d[idx_max]
        y_m = cdf_m[idx_max]
        ax.vlines(x_max_d, min(y_d, y_m), max(y_d, y_m), color="#2A9D8F", linewidth=2.5, linestyle="-", label=f"$D = {d_stat:.3f}$")

        ax.set_xlabel(vlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Cumulative Probability $F(x)$", fontsize=10, fontweight="bold")
        ax.set_xlim(vrange)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)

        status = "EXCELLENT" if p_val >= 0.10 else ("GOOD" if p_val >= 0.01 else "DEVIATION")
        badge_color = "#2A9D8F" if p_val >= 0.10 else ("#F4A261" if p_val >= 0.01 else "#E76F51")

        ax.set_title(f"{vtitle}\n(D = {d_stat:.3f}, p = {p_val:.3g})", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.90)

        summary_rows.append({
            "variable": vtitle,
            "d_stat": d_stat,
            "p_val": p_val,
            "status": status,
            "badge_color": badge_color
        })

    fig.suptitle("SND@LHC Kolmogorov-Smirnov Empirical CDFs: Collision Data vs Simulated Background\nSelection: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) | Lumi = 0.991 fb^-1 (Run 6640)", fontsize=13, fontweight="bold", y=0.99)
    plt.savefig(OUTPUT_CDF_PNG, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved CDF figure to '{OUTPUT_CDF_PNG}'.")

    # 5. Generate Summary Bar Chart
    df_sum = pd.DataFrame(summary_rows).sort_values(by="d_stat", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(df_sum))
    bars = ax.barh(y_pos, df_sum["d_stat"], color=df_sum["badge_color"], edgecolor="black", height=0.6, alpha=0.85)

    for i, bar in enumerate(bars):
        p_v = df_sum["p_val"].iloc[i]
        d_v = df_sum["d_stat"].iloc[i]
        p_str = f"p = {p_v:.3g}" if p_v >= 1e-4 else f"p = {p_v:.1e}"
        ax.text(d_v + 0.015, bar.get_y() + bar.get_height() / 2, f"D = {d_v:.3f} ({p_str})", va="center", fontsize=9.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sum["variable"], fontsize=10.5, fontweight="bold")
    ax.set_xlabel(r"Kolmogorov-Smirnov Statistic $D = \sup |F_{\mathrm{data}}(x) - F_{\mathrm{MC}}(x)|$", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 0.85)
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2A9D8F", edgecolor="black", label=r"Excellent Compatibility ($p \geq 0.10$)"),
        Patch(facecolor="#F4A261", edgecolor="black", label=r"Good Compatibility ($0.01 \leq p < 0.10$)"),
        Patch(facecolor="#E76F51", edgecolor="black", label=r"Shape Deviation ($p < 0.01$)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9.5, framealpha=0.95)

    ax.set_title("SND@LHC Kinematic Shape Compatibility Summary (KS Test)\nCriterion: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_SUMMARY_PNG, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Summary Bar Chart to '{OUTPUT_SUMMARY_PNG}'.")


if __name__ == "__main__":
    main()
