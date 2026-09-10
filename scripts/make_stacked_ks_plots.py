#!/usr/bin/env python3
"""
================================================================================
Memory-Optimized Stacked Data vs. Total MC (Signal + Backgrounds) (Cutset 8)
================================================================================
Plots:
  - Collision Data (Run 6640, black markers)
  - Stacked MC Components:
      1. PMU MC Background (scaled by 100116 * Lumi)
      2. Trimuon Non-Signal MC Background (scaled by Lumi / 0.025)
      3. Direct Trident Signal MC (scaled by Lumi / 0.025)
  - Ratio Sub-Pad: Data / Total MC (Signal + All Backgrounds)
  - Legend: Displays exact yields and Weighted KS test against Total MC
================================================================================
"""

import os
import gc
import sys
import json
import time
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Paths
CACHE_TRACKING_PATH = "/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/data_6640_tracking.parquet"
MC_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet"
PMU_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/pmu/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_hough.parquet"

OUTPUT_ROOT = "kinematic_distributions_ks_cutset8.root"
OUTPUT_PNG = "kinematic_distributions_stacked_ks.png"

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

PLOT_VARIABLES = [
    {
        "name": "xz_sf_m1",
        "title": "SciFi Primary Track Slope dx/dz",
        "xlabel": r"SciFi Track Slope $dx/dz$",
        "bins": 40,
        "range": (-0.15, 0.15)
    },
    {
        "name": "yz_sf_m1",
        "title": "SciFi Primary Track Slope dy/dz",
        "xlabel": r"SciFi Track Slope $dy/dz$",
        "bins": 40,
        "range": (-0.15, 0.15)
    },
    {
        "name": "xz_sf_c1",
        "title": "SciFi Track Intercept x(0) [cm]",
        "xlabel": r"SciFi Intercept $x(0)$ [cm]",
        "bins": 50,
        "range": (-50.0, 50.0)
    },
    {
        "name": "yz_sf_c1",
        "title": "SciFi Track Intercept y(0) [cm]",
        "xlabel": r"SciFi Intercept $y(0)$ [cm]",
        "bins": 50,
        "range": (-50.0, 50.0)
    },
    {
        "name": "delta_slope_sf_xz_12",
        "title": "SciFi Track Angular Separation (XZ)",
        "xlabel": r"SciFi $|\Delta \mathrm{slope}_{xz}|$ [rad]",
        "bins": 30,
        "range": (0.0, 0.12)
    },
    {
        "name": "delta_slope_ds_xz_12",
        "title": "Downstream Track Angular Separation (XZ)",
        "xlabel": r"DS $|\Delta \mathrm{slope}_{xz}|$ [rad]",
        "bins": 30,
        "range": (0.0, 0.12)
    },
    {
        "name": "n_lines_sf_xz",
        "title": "SciFi XZ Track Multiplicity",
        "xlabel": r"SciFi $N_{\mathrm{tracks}}(XZ)$",
        "bins": 6,
        "range": (0.5, 6.5)
    },
    {
        "name": "n_lines_ds_yz",
        "title": "Downstream YZ Track Multiplicity",
        "xlabel": r"DS $N_{\mathrm{tracks}}(YZ)$",
        "bins": 6,
        "range": (0.5, 6.5)
    }
]


def weighted_ks_2sample(data_vals: np.ndarray, mc_vals: np.ndarray, mc_weights: np.ndarray) -> Tuple[float, float]:
    d_clean = data_vals[~np.isnan(data_vals)]
    mc_mask = ~np.isnan(mc_vals) & ~np.isnan(mc_weights) & (mc_weights > 0)
    m_clean = mc_vals[mc_mask]
    w_clean = mc_weights[mc_mask]

    n_d = len(d_clean)
    n_m = len(m_clean)

    if n_d == 0 or n_m == 0:
        return 1.0, 0.0

    sum_w = np.sum(w_clean)
    sum_w2 = np.sum(w_clean ** 2)
    n_eff_mc = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else n_m
    n_eff = (n_d * n_eff_mc) / (n_d + n_eff_mc)

    all_vals = np.concatenate([d_clean, m_clean])
    all_types = np.concatenate([np.ones(n_d), np.zeros(n_m)])
    all_weights = np.concatenate([np.ones(n_d), w_clean])

    order = np.argsort(all_vals)
    sorted_types = all_types[order]
    sorted_weights = all_weights[order]

    data_weights = np.where(sorted_types == 1, sorted_weights, 0.0)
    mc_weights_cum = np.where(sorted_types == 0, sorted_weights, 0.0)

    cdf_data = np.cumsum(data_weights) / n_d
    cdf_mc = np.cumsum(mc_weights_cum) / sum_w

    d_stat = float(np.max(np.abs(cdf_data - cdf_mc)))
    en = np.sqrt(n_eff)
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    p_val = float(stats.kstwobign.sf(lambda_val))
    return d_stat, p_val


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
    print("=" * 105)
    print("   GENERATING STACKED DATA vs TOTAL MC (SIGNAL + BACKGROUNDS) (ROOT & PNG)")
    print("=" * 105)

    # 1. Load PMU MC (Passing Muons)
    print("\n[1/4] Loading PMU MC...")
    df_pmu = pq.read_table(PMU_PATH, columns=TRACKING_COLS + ["mc_weight"]).to_pandas()
    df_pmu["final_weight"] = df_pmu["mc_weight"] * SCALE_PMU
    compute_derived_kinematics(df_pmu)
    uncut_pmu_yield = df_pmu["final_weight"].sum()
    print(f"  -> PMU uncut raw: {len(df_pmu):,}, expected yield: {uncut_pmu_yield:,.1f} events.")

    pmu_mask = build_criterion_mask(df_pmu)
    df_pmu_after = df_pmu[pmu_mask].copy()
    print(f"  -> PMU passing criterion: {len(df_pmu_after)} raw, {df_pmu_after['final_weight'].sum():.2f} weighted yield.")

    # 2. Load Trimuon MC
    print("\n[2/4] Loading Trimuon MC...")
    df_mc = pq.read_table(MC_PATH, columns=TRACKING_COLS + ["is_signal", "mc_weight"]).to_pandas()
    df_mc["final_weight"] = df_mc["mc_weight"] * SCALE_MC
    compute_derived_kinematics(df_mc)

    mc_mask = build_criterion_mask(df_mc)
    df_mc_bkg = df_mc[df_mc["is_signal"] == 0]
    df_mc_sig = df_mc[df_mc["is_signal"] == 1]

    df_mc_bkg_after = df_mc[(df_mc["is_signal"] == 0) & mc_mask].copy()
    df_mc_sig_after = df_mc[(df_mc["is_signal"] == 1) & mc_mask].copy()

    print(f"  -> Trimuon Bkg passing: {len(df_mc_bkg_after)} raw, {df_mc_bkg_after['final_weight'].sum():.1f} weighted yield.")
    print(f"  -> Trimuon Signal passing: {len(df_mc_sig_after)} raw, {df_mc_sig_after['final_weight'].sum():.1f} weighted yield.")

    # 3. Load Collision Data Candidates
    print("\n[3/4] Loading Collision Data Candidates from local cache...")
    tbl_n = pq.read_table(CACHE_TRACKING_PATH, columns=["n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz"])
    nx_dt = tbl_n["n_lines_sf_xz"].to_numpy()
    ny_dt = tbl_n["n_lines_sf_yz"].to_numpy()
    nx_ds_dt = tbl_n["n_lines_ds_xz"].to_numpy()
    ny_ds_dt = tbl_n["n_lines_ds_yz"].to_numpy()

    sf_3_1_dt = ((nx_dt >= 3) & (ny_dt >= 1)) | ((ny_dt >= 3) & (nx_dt >= 1))
    ds_3_1_dt = ((nx_ds_dt >= 3) & (ny_ds_dt >= 1)) | ((ny_ds_dt >= 3) & (nx_ds_dt >= 1))
    cand_indices = np.where(sf_3_1_dt | ds_3_1_dt)[0]
    del tbl_n, nx_dt, ny_dt, nx_ds_dt, ny_ds_dt
    gc.collect()

    dset_track = ds.dataset(CACHE_TRACKING_PATH, format="parquet")
    df_cand = dset_track.take(cand_indices, columns=TRACKING_COLS).to_pandas()
    compute_derived_kinematics(df_cand)

    opt_mask_cand = build_criterion_mask(df_cand)
    df_data_after = df_cand[opt_mask_cand].copy()
    print(f"  -> Surviving Data Events: {len(df_data_after):,}")
    print(f"  -> Total Surviving MC (Sig + Bkg): {df_pmu_after['final_weight'].sum() + df_mc_bkg_after['final_weight'].sum() + df_mc_sig_after['final_weight'].sum():.1f}")

    # Baseline before-cut data sample
    df_data_before = dset_track.take(np.linspace(0, 8873983, 30000, dtype=int), columns=TRACKING_COLS).to_pandas()
    compute_derived_kinematics(df_data_before)
    scale_before_data = 8873984.0 / len(df_data_before)

    # 4. Write ROOT File with Stacks (PMU + Trimuon Bkg + Signal)
    print(f"\n[4/4] Writing ROOT File '{OUTPUT_ROOT}' and PNG...")
    f_root = ROOT.TFile(OUTPUT_ROOT, "RECREATE")

    dir_before = f_root.mkdir("before_selection")
    dir_after = f_root.mkdir("after_selection_optimal")

    stages = [
        ("before", dir_before, df_data_before, df_pmu, df_mc_bkg, df_mc_sig, scale_before_data),
        ("after", dir_after, df_data_after, df_pmu_after, df_mc_bkg_after, df_mc_sig_after, 1.0)
    ]

    for stage_name, t_dir, d_df, p_df, b_df, s_df, d_scale in stages:
        t_dir.cd()
        for pvar in PLOT_VARIABLES:
            vname = pvar["name"]
            nbins = pvar["bins"]
            vmin, vmax = pvar["range"]

            # Combined Total MC (Signal + All Backgrounds) for KS test
            mc_tot_vals = np.concatenate([p_df[vname].to_numpy(), b_df[vname].to_numpy(), s_df[vname].to_numpy()])
            mc_tot_weights = np.concatenate([p_df["final_weight"].to_numpy(), b_df["final_weight"].to_numpy(), s_df["final_weight"].to_numpy()])
            data_vals = d_df[vname].to_numpy()

            d_stat, p_val = weighted_ks_2sample(data_vals, mc_tot_vals, mc_tot_weights)

            # Histograms
            h_data = ROOT.TH1D(f"h_data_{vname}_{stage_name}", f"Data {pvar['title']};{pvar['title']};Events", nbins, vmin, vmax)
            h_pmu = ROOT.TH1D(f"h_pmu_{vname}_{stage_name}", f"PMU MC;{pvar['title']};Events", nbins, vmin, vmax)
            h_trimu_bkg = ROOT.TH1D(f"h_trimu_bkg_{vname}_{stage_name}", f"Trimuon Bkg MC;{pvar['title']};Events", nbins, vmin, vmax)
            h_sig = ROOT.TH1D(f"h_signal_{vname}_{stage_name}", f"Trident Signal MC;{pvar['title']};Events", nbins, vmin, vmax)

            # Fill Data
            for val in data_vals:
                if not np.isnan(val):
                    h_data.Fill(val, d_scale)

            # Fill PMU
            for val, w in zip(p_df[vname].to_numpy(), p_df["final_weight"].to_numpy()):
                if not np.isnan(val):
                    h_pmu.Fill(val, w)

            # Fill Trimuon Bkg
            for val, w in zip(b_df[vname].to_numpy(), b_df["final_weight"].to_numpy()):
                if not np.isnan(val):
                    h_trimu_bkg.Fill(val, w)

            # Fill Signal
            for val, w in zip(s_df[vname].to_numpy(), s_df["final_weight"].to_numpy()):
                if not np.isnan(val):
                    h_sig.Fill(val, w)

            # Styling
            h_data.SetMarkerStyle(20)
            h_data.SetMarkerSize(0.9)
            h_data.SetMarkerColor(ROOT.kBlack)
            h_data.SetLineColor(ROOT.kBlack)

            h_pmu.SetFillColor(ROOT.TColor.GetColor("#F4A261"))
            h_pmu.SetLineColor(ROOT.kBlack)
            h_pmu.SetLineWidth(1)

            h_trimu_bkg.SetFillColor(ROOT.TColor.GetColor("#2A9D8F"))
            h_trimu_bkg.SetLineColor(ROOT.kBlack)
            h_trimu_bkg.SetLineWidth(1)

            h_sig.SetFillColor(ROOT.TColor.GetColor("#E76F51"))
            h_sig.SetLineColor(ROOT.kBlack)
            h_sig.SetLineWidth(1)

            # Stacked Histogram (PMU + Trimuon Bkg + Signal)
            stack = ROOT.THStack(f"hstack_{vname}_{stage_name}", f"{pvar['title']} Stack;{pvar['title']};Events")
            stack.Add(h_trimu_bkg)
            stack.Add(h_pmu)
            stack.Add(h_sig)

            # Canvas with Ratio Pad
            c = ROOT.TCanvas(f"c_{vname}_{stage_name}", f"{pvar['title']} ({stage_name})", 800, 800)
            pad1 = ROOT.TPad(f"pad1_{vname}_{stage_name}", "pad1", 0.0, 0.30, 1.0, 1.0)
            pad2 = ROOT.TPad(f"pad2_{vname}_{stage_name}", "pad2", 0.0, 0.0, 1.0, 0.30)
            pad1.SetBottomMargin(0.02)
            pad2.SetTopMargin(0.03)
            pad2.SetBottomMargin(0.30)
            pad1.Draw()
            pad2.Draw()

            pad1.cd()
            pad1.SetLogy(1 if stage_name == "before" else 0)

            max_y = max(h_data.GetMaximum(), stack.GetMaximum()) * (2.0 if stage_name == "before" else 1.35)
            h_data.SetMaximum(max_y)
            h_data.Draw("E")
            stack.Draw("HIST SAME")
            h_data.Draw("E SAME")

            # Legend with KS Test against Total MC
            leg = ROOT.TLegend(0.46, 0.60, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextFont(42)
            leg.SetTextSize(0.033)
            leg.AddEntry(h_data, f"Data Run 6640 ({h_data.Integral():.0f} ev)", "lep")
            leg.AddEntry(h_sig, f"Trident Signal ({h_sig.Integral():.1f} ev)", "f")
            leg.AddEntry(h_trimu_bkg, f"Trimuon Bkg MC ({h_trimu_bkg.Integral():.1f} ev)", "f")
            leg.AddEntry(h_pmu, f"PMU MC Bkg ({h_pmu.Integral():.1f} ev)", "f")
            leg.AddEntry("", f"#bf{{KS (Data vs Total MC):}} D = {d_stat:.3f}, p = {p_val:.3g}", "")
            leg.Draw()

            # Ratio Pad: Data / Total MC
            pad2.cd()
            pad2.SetGridy(1)
            h_ratio = h_data.Clone(f"hratio_{vname}_{stage_name}")
            h_mc_tot = h_pmu.Clone(f"hmctot_{vname}_{stage_name}")
            h_mc_tot.Add(h_trimu_bkg)
            h_mc_tot.Add(h_sig)
            h_ratio.Divide(h_mc_tot)
            h_ratio.SetTitle("")
            h_ratio.GetYaxis().SetTitle("Data / Total MC")
            h_ratio.GetYaxis().SetRangeUser(0.0, 2.5)
            h_ratio.GetYaxis().SetNdivisions(505)
            h_ratio.GetYaxis().SetTitleSize(0.10)
            h_ratio.GetYaxis().SetTitleOffset(0.45)
            h_ratio.GetYaxis().SetLabelSize(0.08)
            h_ratio.GetXaxis().SetTitleSize(0.11)
            h_ratio.GetXaxis().SetTitleOffset(1.0)
            h_ratio.GetXaxis().SetLabelSize(0.09)
            h_ratio.Draw("EP")

            # Write to ROOT dir
            h_data.Write()
            h_pmu.Write()
            h_trimu_bkg.Write()
            h_sig.Write()
            stack.Write()
            c.Write()

    f_root.Close()
    print(f"[+] Saved ROOT histograms & canvases to '{OUTPUT_ROOT}'.")

    # 5. Generate High-Resolution 8-Panel PNG Summary Figure with Stacked Signal
    fig = plt.figure(figsize=(22, 12), dpi=300)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.32, wspace=0.28)

    for i_var, pvar in enumerate(PLOT_VARIABLES):
        vname = pvar["name"]
        nbins = pvar["bins"]
        vmin, vmax = pvar["range"]
        bin_edges = np.linspace(vmin, vmax, nbins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        ax = fig.add_subplot(gs[i_var // 4, i_var % 4])

        # After selection data & MC
        d_vals = df_data_after[vname].to_numpy()
        p_vals = df_pmu_after[vname].to_numpy()
        p_w = df_pmu_after["final_weight"].to_numpy()
        b_vals = df_mc_bkg_after[vname].to_numpy()
        b_w = df_mc_bkg_after["final_weight"].to_numpy()
        s_vals = df_mc_sig_after[vname].to_numpy()
        s_w = df_mc_sig_after["final_weight"].to_numpy()

        # KS test against Total MC (Signal + Backgrounds)
        mc_tot_vals = np.concatenate([p_vals, b_vals, s_vals])
        mc_tot_weights = np.concatenate([p_w, b_w, s_w])
        d_stat, p_val = weighted_ks_2sample(d_vals, mc_tot_vals, mc_tot_weights)

        # Histograms
        h_d, _ = np.histogram(d_vals[~np.isnan(d_vals)], bins=bin_edges)
        h_p, _ = np.histogram(p_vals[~np.isnan(p_vals)], bins=bin_edges, weights=p_w[~np.isnan(p_vals)])
        h_b, _ = np.histogram(b_vals[~np.isnan(b_vals)], bins=bin_edges, weights=b_w[~np.isnan(b_vals)])
        h_s, _ = np.histogram(s_vals[~np.isnan(s_vals)], bins=bin_edges, weights=s_w[~np.isnan(s_vals)])

        # Stacked bar plot (Trimuon Bkg -> PMU Bkg -> Trident Signal)
        bottom_1 = h_b
        bottom_2 = h_b + h_p

        ax.bar(bin_centers, h_b, width=bin_width, color="#2A9D8F", label=f"Trimuon Bkg ({np.sum(h_b):.1f})", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(bin_centers, h_p, width=bin_width, bottom=bottom_1, color="#F4A261", label=f"PMU MC Bkg ({np.sum(h_p):.1f})", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(bin_centers, h_s, width=bin_width, bottom=bottom_2, color="#E76F51", label=f"Trident Signal ({np.sum(h_s):.1f})", alpha=0.85, edgecolor="black", linewidth=0.5)

        # Data points
        d_err = np.sqrt(h_d)
        ax.errorbar(bin_centers, h_d, yerr=d_err, fmt="o", color="black", markersize=4.5, capsize=2.0, label=f"Data ({np.sum(h_d):.0f})", zorder=5)

        ax.set_xlabel(pvar["xlabel"], fontsize=11, fontweight="bold")
        ax.set_ylabel("Events / Bin", fontsize=11, fontweight="bold")
        ax.set_xlim(vmin, vmax)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Legend with KS Test in header
        ks_compat_str = f"KS (Data vs Total MC):\nD = {d_stat:.3f}, p = {p_val:.3g}"
        ax.legend(title=ks_compat_str, title_fontsize=9.0, fontsize=8.2, loc="upper right", framealpha=0.90)

    fig.suptitle("SND@LHC Kinematic Distributions: Collision Data vs Stacked MC (Signal + Backgrounds)\nAfter [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) Selection | Lumi = 0.991 fb^-1 (Run 6640)", fontsize=13, fontweight="bold", y=0.99)

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PNG)), exist_ok=True)
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved PNG figure to '{OUTPUT_PNG}'.")


if __name__ == "__main__":
    main()
