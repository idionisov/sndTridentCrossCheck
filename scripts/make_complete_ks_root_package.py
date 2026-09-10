#!/usr/bin/env python3
"""
================================================================================
Complete ROOT Package & Canvas Visualizer with Error Bars & KS Tests (Cutset 8)
================================================================================
For: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)

Writes to 'kinematic_distributions_ks_cutset8.root':
  - Directories: 'before_selection/' and 'after_selection_optimal/'
  - Inside each directory:
      * h_data_<var>: Data TH1D with Poisson error bars (Sumw2)
      * h_pmu_<var>: PMU MC TH1D with Sumw2
      * h_trimu_bkg_<var>: Trimuon Bkg MC TH1D with Sumw2
      * h_signal_<var>: Trident Signal MC TH1D with Sumw2
      * h_mc_tot_<var>: Total MC TH1D with Sumw2 error bars
      * h_mc_err_<var>: Total MC uncertainty band (E2)
      * hstack_<var>: THStack of PMU + Trimuon Bkg + Signal
      * gr_data_cdf_<var>: TGraphErrors for Data Empirical CDF
      * gr_mc_cdf_<var>: TGraphErrors for MC Empirical CDF
      * hratio_<var>: Data / Total MC Ratio with full error propagation
      * c_<var>: Full TCanvas with main distribution pad, ratio pad, and KS legend
      * c_cdf_<var>: Full TCanvas showing empirical CDFs with error bars
================================================================================
"""

import os
import gc
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
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.SetDefaultSumw2(True)

# Paths
CACHE_TRACKING_PATH = "/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/data_6640_tracking.parquet"
MC_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet"
PMU_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/pmu/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_hough.parquet"

OUTPUT_ROOT = "kinematic_distributions_ks_cutset8.root"
OUTPUT_DIST_PNG = "kinematic_distributions_errorbars_optimal.png"
OUTPUT_CDF_PNG = "kinematic_cdfs_errorbars_optimal.png"

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
        "xlabel": "SciFi Track Slope dx/dz",
        "bins": 40,
        "range": (-0.15, 0.15)
    },
    {
        "name": "yz_sf_m1",
        "title": "SciFi Primary Track Slope dy/dz",
        "xlabel": "SciFi Track Slope dy/dz",
        "bins": 40,
        "range": (-0.15, 0.15)
    },
    {
        "name": "xz_sf_c1",
        "title": "SciFi Track Intercept x(0) [cm]",
        "xlabel": "SciFi Intercept x(0) [cm]",
        "bins": 50,
        "range": (-50.0, 50.0)
    },
    {
        "name": "yz_sf_c1",
        "title": "SciFi Track Intercept y(0) [cm]",
        "xlabel": "SciFi Intercept y(0) [cm]",
        "bins": 50,
        "range": (-50.0, 50.0)
    },
    {
        "name": "delta_slope_sf_xz_12",
        "title": "SciFi Track Angular Separation (XZ)",
        "xlabel": "SciFi |#Delta slope_{xz}| [rad]",
        "bins": 30,
        "range": (0.0, 0.12)
    },
    {
        "name": "delta_slope_ds_xz_12",
        "title": "Downstream Track Angular Separation (XZ)",
        "xlabel": "DS |#Delta slope_{xz}| [rad]",
        "bins": 30,
        "range": (0.0, 0.12)
    },
    {
        "name": "n_lines_sf_xz",
        "title": "SciFi XZ Track Multiplicity",
        "xlabel": "SciFi N_{tracks}(XZ)",
        "bins": 6,
        "range": (0.5, 6.5)
    },
    {
        "name": "n_lines_ds_yz",
        "title": "Downstream YZ Track Multiplicity",
        "xlabel": "DS N_{tracks}(YZ)",
        "bins": 6,
        "range": (0.5, 6.5)
    }
]


def weighted_ks_cdf(data_vals: np.ndarray, mc_vals: np.ndarray, mc_weights: np.ndarray):
    d_clean = data_vals[~np.isnan(data_vals)]
    mc_mask = ~np.isnan(mc_vals) & ~np.isnan(mc_weights) & (mc_weights > 0)
    m_clean = mc_vals[mc_mask]
    w_clean = mc_weights[mc_mask]

    n_d = len(d_clean)
    n_m = len(m_clean)

    if n_d == 0 or n_m == 0:
        return np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), 1.0, 0.0, 0.0, np.array([0, 0]), np.array([0, 0])

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

    # Standard errors on cumulative distributions
    err_data = np.sqrt(np.clip(cdf_data * (1.0 - cdf_data) / max(n_d, 1), 0.0, 1.0))
    err_mc = np.sqrt(np.clip(cdf_mc * (1.0 - cdf_mc) / max(n_eff_mc, 1), 0.0, 1.0))

    diff = np.abs(cdf_data - cdf_mc)
    max_idx = np.argmax(diff)
    d_stat = float(diff[max_idx])
    x_at_max_d = sorted_vals[max_idx]

    en = np.sqrt(n_eff)
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    p_val = float(stats.kstwobign.sf(lambda_val))

    return sorted_vals, cdf_data, cdf_mc, d_stat, p_val, x_at_max_d, err_data, err_mc


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
    print("=" * 110)
    print("   GENERATING COMPLETE ROOT PACKAGE WITH ERROR BARS & CANVASES")
    print("=" * 110)

    # 1. Load PMU MC
    print("\n[1/4] Loading PMU MC...")
    df_pmu = pq.read_table(PMU_PATH, columns=TRACKING_COLS + ["mc_weight"]).to_pandas()
    df_pmu["final_weight"] = df_pmu["mc_weight"] * SCALE_PMU
    compute_derived_kinematics(df_pmu)
    pmu_mask = build_criterion_mask(df_pmu)
    df_pmu_after = df_pmu[pmu_mask].copy()

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

    # 3. Load Collision Data
    print("\n[3/4] Loading Collision Data Candidates from local cache...")
    tbl_n = pq.read_table(CACHE_TRACKING_PATH, columns=["n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz"])
    nx_dt = tbl_n["n_lines_sf_xz"].to_numpy()
    ny_dt = tbl_n["n_lines_sf_yz"].to_numpy()
    nx_ds_dt = tbl_n["n_lines_ds_xz"].to_numpy()
    ny_ds_dt = tbl_n["n_lines_ds_yz"].to_numpy()

    cand_indices = np.where(((nx_dt >= 3) & (ny_dt >= 1)) | ((ny_dt >= 3) & (nx_dt >= 1)) | ((nx_ds_dt >= 3) & (ny_ds_dt >= 1)) | ((ny_ds_dt >= 3) & (nx_ds_dt >= 1)))[0]
    del tbl_n, nx_dt, ny_dt, nx_ds_dt, ny_ds_dt
    gc.collect()

    dset_track = ds.dataset(CACHE_TRACKING_PATH, format="parquet")
    df_cand = dset_track.take(cand_indices, columns=TRACKING_COLS).to_pandas()
    compute_derived_kinematics(df_cand)

    opt_mask_cand = build_criterion_mask(df_cand)
    df_data_after = df_cand[opt_mask_cand].copy()

    df_data_before = dset_track.take(np.linspace(0, 8873983, 30000, dtype=int), columns=TRACKING_COLS).to_pandas()
    compute_derived_kinematics(df_data_before)
    scale_before_data = 8873984.0 / len(df_data_before)

    print(f"  -> Surviving Data: {len(df_data_after):,} | Surviving PMU Bkg: {df_pmu_after['final_weight'].sum():.1f} | Surviving Trimu Bkg: {df_mc_bkg_after['final_weight'].sum():.1f} | Signal: {df_mc_sig_after['final_weight'].sum():.1f}")

    # 4. Write Complete ROOT File with Histograms, Graphs, Stacks, and Canvases
    print(f"\n[4/4] Writing ROOT File '{OUTPUT_ROOT}'...")
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

            # 1. Compute KS Test and CDFs with Error Bars
            d_vals = d_df[vname].to_numpy()
            p_vals = p_df[vname].to_numpy()
            p_w = p_df["final_weight"].to_numpy()
            b_vals = b_df[vname].to_numpy()
            b_w = b_df["final_weight"].to_numpy()
            s_vals = s_df[vname].to_numpy()
            s_w = s_df["final_weight"].to_numpy()

            mc_tot_vals = np.concatenate([p_vals, b_vals, s_vals])
            mc_tot_weights = np.concatenate([p_w, b_w, s_w])

            sorted_v, cdf_d, cdf_m, d_stat, p_val, x_max_d, err_d, err_m = weighted_ks_cdf(d_vals, mc_tot_vals, mc_tot_weights)

            # 2. Build ROOT Histograms with Error Bars
            h_data = ROOT.TH1D(f"h_data_{vname}_{stage_name}", f"Data {pvar['title']};{pvar['xlabel']};Events / Bin", nbins, vmin, vmax)
            h_pmu = ROOT.TH1D(f"h_pmu_{vname}_{stage_name}", f"PMU MC;{pvar['xlabel']};Events / Bin", nbins, vmin, vmax)
            h_trimu_bkg = ROOT.TH1D(f"h_trimu_bkg_{vname}_{stage_name}", f"Trimuon Bkg MC;{pvar['xlabel']};Events / Bin", nbins, vmin, vmax)
            h_sig = ROOT.TH1D(f"h_signal_{vname}_{stage_name}", f"Trident Signal MC;{pvar['xlabel']};Events / Bin", nbins, vmin, vmax)

            for val in d_vals:
                if not np.isnan(val):
                    h_data.Fill(val, d_scale)

            for val, w in zip(p_vals, p_w):
                if not np.isnan(val):
                    h_pmu.Fill(val, w)

            for val, w in zip(b_vals, b_w):
                if not np.isnan(val):
                    h_trimu_bkg.Fill(val, w)

            for val, w in zip(s_vals, s_w):
                if not np.isnan(val):
                    h_sig.Fill(val, w)

            # Total MC sum histogram with error bars
            h_mc_tot = ROOT.TH1D(f"h_mc_tot_{vname}_{stage_name}", f"Total MC;{pvar['xlabel']};Events / Bin", nbins, vmin, vmax)
            h_mc_tot.Add(h_trimu_bkg)
            h_mc_tot.Add(h_pmu)
            h_mc_tot.Add(h_sig)

            # Styling
            h_data.SetMarkerStyle(20)
            h_data.SetMarkerSize(0.95)
            h_data.SetMarkerColor(ROOT.kBlack)
            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetLineWidth(2)

            h_pmu.SetFillColor(ROOT.TColor.GetColor("#F4A261"))
            h_pmu.SetLineColor(ROOT.kBlack)
            h_pmu.SetLineWidth(1)

            h_trimu_bkg.SetFillColor(ROOT.TColor.GetColor("#2A9D8F"))
            h_trimu_bkg.SetLineColor(ROOT.kBlack)
            h_trimu_bkg.SetLineWidth(1)

            h_sig.SetFillColor(ROOT.TColor.GetColor("#E76F51"))
            h_sig.SetLineColor(ROOT.kBlack)
            h_sig.SetLineWidth(1)

            # MC Total Uncertainty Error Band (Hatched)
            h_mc_err = h_mc_tot.Clone(f"h_mc_err_{vname}_{stage_name}")
            h_mc_err.SetFillColor(ROOT.kGray + 2)
            h_mc_err.SetFillStyle(3354)
            h_mc_err.SetMarkerSize(0)

            # Stacked Histogram
            stack = ROOT.THStack(f"hstack_{vname}_{stage_name}", f"{pvar['title']} Stack;{pvar['xlabel']};Events / Bin")
            stack.Add(h_trimu_bkg)
            stack.Add(h_pmu)
            stack.Add(h_sig)

            # 3. Canvas 1: Differential Distribution Canvas with Ratio Pad
            c = ROOT.TCanvas(f"c_{vname}_{stage_name}", f"{pvar['title']} ({stage_name})", 850, 850)
            pad1 = ROOT.TPad(f"pad1_{vname}_{stage_name}", "pad1", 0.0, 0.30, 1.0, 1.0)
            pad2 = ROOT.TPad(f"pad2_{vname}_{stage_name}", "pad2", 0.0, 0.0, 1.0, 0.30)
            pad1.SetBottomMargin(0.03)
            pad1.SetTopMargin(0.08)
            pad2.SetTopMargin(0.03)
            pad2.SetBottomMargin(0.32)
            pad1.Draw()
            pad2.Draw()

            pad1.cd()
            pad1.SetLogy(1 if stage_name == "before" else 0)

            max_y = max(h_data.GetMaximum(), stack.GetMaximum()) * (2.2 if stage_name == "before" else 1.40)
            h_data.SetMaximum(max_y)
            h_data.Draw("E1")
            stack.Draw("HIST SAME")
            h_mc_err.Draw("E2 SAME")
            h_data.Draw("E1 SAME")

            # Legend with Error Bars and KS Test
            leg = ROOT.TLegend(0.44, 0.58, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextFont(42)
            leg.SetTextSize(0.033)
            leg.AddEntry(h_data, f"Data Run 6640 ({h_data.Integral():.0f} ev)", "lep")
            leg.AddEntry(h_sig, f"Trident Signal ({h_sig.Integral():.1f} ev)", "f")
            leg.AddEntry(h_trimu_bkg, f"Trimuon Bkg MC ({h_trimu_bkg.Integral():.1f} ev)", "f")
            leg.AddEntry(h_pmu, f"PMU MC Bkg ({h_pmu.Integral():.1f} ev)", "f")
            leg.AddEntry(h_mc_err, f"MC Stat. Uncert.", "f")
            leg.AddEntry("", f"#bf{{KS (Data vs Total MC):}} D = {d_stat:.3f}, p = {p_val:.3g}", "")
            leg.Draw()

            # Ratio Pad: Data / Total MC
            pad2.cd()
            pad2.SetGridy(1)
            h_ratio = h_data.Clone(f"hratio_{vname}_{stage_name}")
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

            # Ratio MC uncertainty band around 1.0
            h_ratio_err = h_mc_tot.Clone(f"hratio_err_{vname}_{stage_name}")
            h_ratio_err.Divide(h_mc_tot)
            h_ratio_err.SetFillColor(ROOT.kGray + 1)
            h_ratio_err.SetFillStyle(3354)
            h_ratio_err.SetMarkerSize(0)
            h_ratio_err.Draw("E2 SAME")
            h_ratio.Draw("EP SAME")

            # 4. Canvas 2: Empirical CDFs with Error Bars
            c_cdf = ROOT.TCanvas(f"c_cdf_{vname}_{stage_name}", f"CDF: {pvar['title']} ({stage_name})", 800, 700)
            c_cdf.SetGrid(1, 1)
            c_cdf.SetTopMargin(0.08)
            c_cdf.SetBottomMargin(0.12)
            c_cdf.SetLeftMargin(0.12)

            # Subsample points for clean graph drawing
            n_pts = min(len(sorted_v), 150)
            sub_idx = np.linspace(0, len(sorted_v) - 1, n_pts, dtype=int)

            x_pts = np.ascontiguousarray(sorted_v[sub_idx], dtype=np.float64)
            yd_pts = np.ascontiguousarray(cdf_d[sub_idx], dtype=np.float64)
            ym_pts = np.ascontiguousarray(cdf_m[sub_idx], dtype=np.float64)
            ex_pts = np.zeros(n_pts, dtype=np.float64)
            eyd_pts = np.ascontiguousarray(err_d[sub_idx], dtype=np.float64)
            eym_pts = np.ascontiguousarray(err_m[sub_idx], dtype=np.float64)

            gr_data_cdf = ROOT.TGraphErrors(n_pts, x_pts, yd_pts, ex_pts, eyd_pts)
            gr_data_cdf.SetName(f"gr_data_cdf_{vname}_{stage_name}")
            gr_data_cdf.SetTitle(f"Empirical CDF: {pvar['title']};{pvar['xlabel']};Cumulative Probability F(x)")
            gr_data_cdf.SetMarkerStyle(20)
            gr_data_cdf.SetMarkerSize(0.8)
            gr_data_cdf.SetMarkerColor(ROOT.kBlack)
            gr_data_cdf.SetLineColor(ROOT.kBlack)
            gr_data_cdf.SetLineWidth(2)
            gr_data_cdf.GetYaxis().SetRangeUser(0.0, 1.05)

            gr_mc_cdf = ROOT.TGraphErrors(n_pts, x_pts, ym_pts, ex_pts, eym_pts)
            gr_mc_cdf.SetName(f"gr_mc_cdf_{vname}_{stage_name}")
            gr_mc_cdf.SetMarkerStyle(24)
            gr_mc_cdf.SetMarkerSize(0.8)
            gr_mc_cdf.SetMarkerColor(ROOT.kRed + 1)
            gr_mc_cdf.SetLineColor(ROOT.kRed + 1)
            gr_mc_cdf.SetLineWidth(2)

            gr_data_cdf.Draw("APL")
            gr_mc_cdf.Draw("PL SAME")

            leg_cdf = ROOT.TLegend(0.48, 0.18, 0.88, 0.38)
            leg_cdf.SetBorderSize(0)
            leg_cdf.SetFillStyle(0)
            leg_cdf.SetTextFont(42)
            leg_cdf.SetTextSize(0.034)
            leg_cdf.AddEntry(gr_data_cdf, "Data CDF F_{data}(x) #pm #sigma", "lep")
            leg_cdf.AddEntry(gr_mc_cdf, "Total MC CDF F_{MC}(x) #pm #sigma", "lep")
            leg_cdf.AddEntry("", f"#bf{{Max Distance:}} D = {d_stat:.3f} (p = {p_val:.3g})", "")
            leg_cdf.Draw()

            # Write Objects to ROOT dir
            h_data.Write()
            h_pmu.Write()
            h_trimu_bkg.Write()
            h_sig.Write()
            h_mc_tot.Write()
            h_mc_err.Write()
            stack.Write()
            h_ratio.Write()
            gr_data_cdf.Write()
            gr_mc_cdf.Write()
            c.Write()
            c_cdf.Write()

    f_root.Close()
    print(f"[+] Saved ROOT file with all histograms, error bars, graphs, and canvases to '{OUTPUT_ROOT}'.")

    # 5. Generate Multi-Panel PNG Figures with Error Bars
    print("\n[5/5] Generating Summary PNG Figures...")
    
    # Figure A: Distributions with Error Bars
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

        d_vals = df_data_after[vname].to_numpy()
        p_vals = df_pmu_after[vname].to_numpy()
        p_w = df_pmu_after["final_weight"].to_numpy()
        b_vals = df_mc_bkg_after[vname].to_numpy()
        b_w = df_mc_bkg_after["final_weight"].to_numpy()
        s_vals = df_mc_sig_after[vname].to_numpy()
        s_w = df_mc_sig_after["final_weight"].to_numpy()

        mc_tot_vals = np.concatenate([p_vals, b_vals, s_vals])
        mc_tot_weights = np.concatenate([p_w, b_w, s_w])
        _, _, _, d_stat, p_val, _, _, _ = weighted_ks_cdf(d_vals, mc_tot_vals, mc_tot_weights)

        h_d, _ = np.histogram(d_vals[~np.isnan(d_vals)], bins=bin_edges)
        h_p, _ = np.histogram(p_vals[~np.isnan(p_vals)], bins=bin_edges, weights=p_w[~np.isnan(p_vals)])
        h_b, _ = np.histogram(b_vals[~np.isnan(b_vals)], bins=bin_edges, weights=b_w[~np.isnan(b_vals)])
        h_s, _ = np.histogram(s_vals[~np.isnan(s_vals)], bins=bin_edges, weights=s_w[~np.isnan(s_vals)])

        # MC stat error on each bin
        h_p_w2, _ = np.histogram(p_vals[~np.isnan(p_vals)], bins=bin_edges, weights=(p_w[~np.isnan(p_vals)])**2)
        h_b_w2, _ = np.histogram(b_vals[~np.isnan(b_vals)], bins=bin_edges, weights=(b_w[~np.isnan(b_vals)])**2)
        h_s_w2, _ = np.histogram(s_vals[~np.isnan(s_vals)], bins=bin_edges, weights=(s_w[~np.isnan(s_vals)])**2)
        mc_tot_err = np.sqrt(h_p_w2 + h_b_w2 + h_s_w2)
        mc_tot_yield = h_p + h_b + h_s

        # Stacked bars
        bottom_1 = h_b
        bottom_2 = h_b + h_p
        ax.bar(bin_centers, h_b, width=bin_width, color="#2A9D8F", label=f"Trimuon Bkg ({np.sum(h_b):.1f})", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(bin_centers, h_p, width=bin_width, bottom=bottom_1, color="#F4A261", label=f"PMU MC Bkg ({np.sum(h_p):.1f})", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(bin_centers, h_s, width=bin_width, bottom=bottom_2, color="#E76F51", label=f"Trident Signal ({np.sum(h_s):.1f})", alpha=0.85, edgecolor="black", linewidth=0.5)

        # MC Total Uncertainty Hatched Error Band
        ax.bar(bin_centers, 2.0 * mc_tot_err, width=bin_width, bottom=mc_tot_yield - mc_tot_err, color="none", edgecolor="black", hatch="////", linewidth=0.0, label="MC Stat. Uncert.", zorder=4)

        # Data points with Poisson error bars
        d_err = np.sqrt(h_d)
        ax.errorbar(bin_centers, h_d, yerr=d_err, fmt="o", color="black", markersize=4.5, capsize=2.0, elinewidth=1.2, label=f"Data ({np.sum(h_d):.0f})", zorder=5)

        ax.set_xlabel(pvar["xlabel"], fontsize=11, fontweight="bold")
        ax.set_ylabel("Events / Bin", fontsize=11, fontweight="bold")
        ax.set_xlim(vmin, vmax)
        ax.grid(True, linestyle="--", alpha=0.4)

        ks_compat_str = f"KS (Data vs Total MC):\nD = {d_stat:.3f}, p = {p_val:.3g}"
        ax.legend(title=ks_compat_str, title_fontsize=8.8, fontsize=8.0, loc="upper right", framealpha=0.90)

    fig.suptitle("SND@LHC Kinematic Distributions with Error Bars: Data vs Stacked MC\nAfter [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) Selection | Lumi = 0.991 fb^-1 (Run 6640)", fontsize=13, fontweight="bold", y=0.99)
    plt.savefig(OUTPUT_DIST_PNG, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Differential Distributions PNG to '{OUTPUT_DIST_PNG}'.")

    # Figure B: Empirical CDFs with Error Bars
    fig = plt.figure(figsize=(22, 11), dpi=300)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.25)

    for i_var, pvar in enumerate(PLOT_VARIABLES):
        vname = pvar["name"]
        vmin, vmax = pvar["range"]
        ax = fig.add_subplot(gs[i_var // 4, i_var % 4])

        d_vals = df_data_after[vname].to_numpy()
        p_vals = df_pmu_after[vname].to_numpy()
        p_w = df_pmu_after["final_weight"].to_numpy()
        b_vals = df_mc_bkg_after[vname].to_numpy()
        b_w = df_mc_bkg_after["final_weight"].to_numpy()
        s_vals = df_mc_sig_after[vname].to_numpy()
        s_w = df_mc_sig_after["final_weight"].to_numpy()

        mc_tot_vals = np.concatenate([p_vals, b_vals, s_vals])
        mc_tot_weights = np.concatenate([p_w, b_w, s_w])

        sorted_v, cdf_d, cdf_m, d_stat, p_val, x_max_d, err_d, err_m = weighted_ks_cdf(d_vals, mc_tot_vals, mc_tot_weights)

        # Plot CDFs with Error Bands / Error Bars
        ax.plot(sorted_v, cdf_d, color="#1D3557", linewidth=2.0, label=r"$\mathrm{Data\ CDF}\ F_{\mathrm{data}}(x)$")
        ax.fill_between(sorted_v, np.clip(cdf_d - err_d, 0, 1), np.clip(cdf_d + err_d, 0, 1), color="#1D3557", alpha=0.18, label=r"Data $\pm 1\sigma$")

        ax.plot(sorted_v, cdf_m, color="#E63946", linewidth=2.0, linestyle="--", label=r"$\mathrm{Total\ MC\ CDF}\ F_{\mathrm{MC}}(x)$")
        ax.fill_between(sorted_v, np.clip(cdf_m - err_m, 0, 1), np.clip(cdf_m + err_m, 0, 1), color="#E63946", alpha=0.18, label=r"MC $\pm 1\sigma$")

        ax.set_xlabel(pvar["xlabel"], fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Cumulative Probability $F(x)$", fontsize=10, fontweight="bold")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)

        ax.set_title(f"{pvar['title']}\n(D = {d_stat:.3f}, p = {p_val:.3g})", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.0, framealpha=0.90)

    fig.suptitle("SND@LHC Kolmogorov-Smirnov Empirical CDFs with Error Bands: Data vs Total MC\nAfter [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) Selection | Lumi = 0.991 fb^-1 (Run 6640)", fontsize=13, fontweight="bold", y=0.99)
    plt.savefig(OUTPUT_CDF_PNG, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved CDF Error Bands PNG to '{OUTPUT_CDF_PNG}'.")


if __name__ == "__main__":
    main()
