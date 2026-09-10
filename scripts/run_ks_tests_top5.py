#!/usr/bin/env python3
"""
================================================================================
Fast Weighted Kolmogorov-Smirnov (KS) Kinematic Shape Tests (Cutset 8)
================================================================================
Evaluates Data vs. Simulation Background Shape Agreement for the Top 5 Criteria
that maximize signal efficiency / significance relative to simulated background.

Simulation Background:
  - Trimuon MC Non-Signal Background (is_signal == 0)
  - Passing Muon MC (PMU FLUKA + Geant4 Background)
  - Scaled by mc_weight to full luminosity (0.992 fb^-1)

Kinematic Variables Tested:
  1. Track Slopes & Angles : xz_sf_m1, yz_sf_m1, xz_ds_m1, yz_ds_m1
  2. Track Intercepts & IP : xz_sf_c1, yz_sf_c1, xz_ds_c1, yz_ds_c1
  3. Angular Separations   : delta_slope_sf_xz_12, delta_slope_sf_yz_12, delta_slope_ds_xz_12, delta_slope_ds_yz_12
  4. Track Multiplicities  : n_lines_sf_xz, n_lines_sf_yz, n_lines_ds_xz, n_lines_ds_yz
================================================================================
"""

import os
import sys
import json
import time
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from scipy import stats

CACHE_TRACKING_PATH = "/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/data_6640_tracking.parquet"
MC_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet"
PMU_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/pmu/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_hough.parquet"

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
    ("xz_sf_m1", "SciFi Primary Track Slope dx/dz"),
    ("yz_sf_m1", "SciFi Primary Track Slope dy/dz"),
    ("xz_ds_m1", "Downstream Primary Track Slope dx/dz"),
    ("yz_ds_m1", "Downstream Primary Track Slope dy/dz"),
    
    ("xz_sf_c1", "SciFi Track Intercept x(0) [cm]"),
    ("yz_sf_c1", "SciFi Track Intercept y(0) [cm]"),
    ("xz_ds_c1", "Downstream Track Intercept x(0) [cm]"),
    ("yz_ds_c1", "Downstream Track Intercept y(0) [cm]"),
    
    ("delta_slope_sf_xz_12", "SciFi Track Angular Separation (XZ)"),
    ("delta_slope_sf_yz_12", "SciFi Track Angular Separation (YZ)"),
    ("delta_slope_ds_xz_12", "Downstream Track Angular Separation (XZ)"),
    ("delta_slope_ds_yz_12", "Downstream Track Angular Separation (YZ)"),
    
    ("n_lines_sf_xz", "SciFi XZ Multiplicity"),
    ("n_lines_sf_yz", "SciFi YZ Multiplicity"),
    ("n_lines_ds_xz", "Downstream XZ Multiplicity"),
    ("n_lines_ds_yz", "Downstream YZ Multiplicity")
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


def main():
    print("=" * 110)
    print("      WEIGHTED KOLMOGOROV-SMIRNOV KINEMATIC SHAPE TESTS: TOP 5 CRITERIA vs SIMULATED BACKGROUND")
    print("=" * 110)

    top5_criteria = [
        "[SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)",
        "(SF_3 | DS_3) & (SF_nc & DS_nc)",
        "[SF(3,2) | DS(3,2)] & (SF_nc & DS_nc)",
        "[SF(3,1) | DS(3,1)] & (SF_nc_both & DS_nc_phys)",
        "SF(3,2)_AND_phys & DS(1,0)_OR"
    ]

    print("\n[1/3] Loading simulation background datasets...")
    t0 = time.time()
    
    # 1. PMU MC
    df_pmu = pq.read_table(PMU_PATH, columns=TRACKING_COLS + ["mc_weight"]).to_pandas()
    df_pmu["final_weight"] = df_pmu["mc_weight"] * SCALE_PMU
    print(f"  -> Loaded {len(df_pmu):,} PMU background events.")

    # 2. Trimuon MC Non-Signal Background
    df_mc = pq.read_table(MC_PATH, columns=TRACKING_COLS + ["is_signal", "mc_weight"]).to_pandas()
    df_mc_bkg = df_mc[df_mc["is_signal"] == 0].copy()
    df_mc_bkg["final_weight"] = df_mc_bkg["mc_weight"] * SCALE_MC
    print(f"  -> Loaded {len(df_mc_bkg):,} Trimuon non-signal background events.")

    df_sim_bkg = pd.concat([df_mc_bkg, df_pmu], ignore_index=True)
    compute_derived_kinematics(df_sim_bkg)
    print(f"  -> Total Simulated Background Sample: {len(df_sim_bkg):,} events.")

    # Evaluate MC Masks
    nx_mc = df_sim_bkg["n_lines_sf_xz"].to_numpy()
    ny_mc = df_sim_bkg["n_lines_sf_yz"].to_numpy()
    nx_ds_mc = df_sim_bkg["n_lines_ds_xz"].to_numpy()
    ny_ds_mc = df_sim_bkg["n_lines_ds_yz"].to_numpy()

    sf_3_1_mc = ((nx_mc >= 3) & (ny_mc >= 1)) | ((ny_mc >= 3) & (nx_mc >= 1))
    ds_3_1_mc = ((nx_ds_mc >= 3) & (ny_ds_mc >= 1)) | ((ny_ds_mc >= 3) & (nx_ds_mc >= 1))
    sf_3_0_mc = (nx_mc >= 3) | (ny_mc >= 3)
    ds_3_0_mc = (nx_ds_mc >= 3) | (ny_ds_mc >= 3)
    sf_3_2_mc = ((nx_mc >= 3) & (ny_mc >= 2)) | ((ny_mc >= 3) & (nx_mc >= 2))
    ds_3_2_mc = ((nx_ds_mc >= 3) & (ny_ds_mc >= 2)) | ((ny_ds_mc >= 3) & (nx_ds_mc >= 2))

    nc_sf_xz_s_mc = check_nc(df_sim_bkg, "sf", "xz", 260.0, 355.0, strict=True)
    nc_sf_yz_s_mc = check_nc(df_sim_bkg, "sf", "yz", 260.0, 355.0, strict=True)
    nc_ds_xz_s_mc = check_nc(df_sim_bkg, "ds", "xz", 460.0, 590.0, strict=True)
    nc_ds_yz_s_mc = check_nc(df_sim_bkg, "ds", "yz", 460.0, 590.0, strict=True)

    nc_sf_xz_p_mc = check_nc(df_sim_bkg, "sf", "xz", 260.0, 355.0, strict=False)
    nc_sf_yz_p_mc = check_nc(df_sim_bkg, "sf", "yz", 260.0, 355.0, strict=False)
    nc_ds_xz_p_mc = check_nc(df_sim_bkg, "ds", "xz", 460.0, 590.0, strict=False)
    nc_ds_yz_p_mc = check_nc(df_sim_bkg, "ds", "yz", 460.0, 590.0, strict=False)

    sf_nc_or_mc = nc_sf_xz_s_mc | nc_sf_yz_s_mc
    ds_nc_or_mc = nc_ds_xz_s_mc | nc_ds_yz_s_mc
    sf_nc_and_mc = nc_sf_xz_s_mc & nc_sf_yz_s_mc
    ds_nc_phys_or_mc = nc_ds_xz_p_mc | nc_ds_yz_p_mc

    mc_masks = {
        "[SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)": (sf_3_1_mc | ds_3_1_mc) & (sf_nc_or_mc & ds_nc_or_mc),
        "(SF_3 | DS_3) & (SF_nc & DS_nc)": (sf_3_0_mc | ds_3_0_mc) & (sf_nc_or_mc & ds_nc_or_mc),
        "[SF(3,2) | DS(3,2)] & (SF_nc & DS_nc)": (sf_3_2_mc | ds_3_2_mc) & (sf_nc_or_mc & ds_nc_or_mc),
        "[SF(3,1) | DS(3,1)] & (SF_nc_both & DS_nc_phys)": (sf_3_1_mc | ds_3_1_mc) & (sf_nc_and_mc & ds_nc_phys_or_mc),
        "SF(3,2)_AND_phys & DS(1,0)_OR": (sf_3_2_mc & (nc_sf_xz_p_mc & nc_sf_yz_p_mc)) & (((nx_ds_mc >= 1) | (ny_ds_mc >= 1)) & ds_nc_or_mc)
    }

    # 2. Collision Data Candidate Filtering
    print("\n[2/3] Evaluating collision data candidates from local cache...")
    t1 = time.time()
    tbl_n = pq.read_table(CACHE_TRACKING_PATH, columns=["n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz"])
    nx_dt = tbl_n["n_lines_sf_xz"].to_numpy()
    ny_dt = tbl_n["n_lines_sf_yz"].to_numpy()
    nx_ds_dt = tbl_n["n_lines_ds_xz"].to_numpy()
    ny_ds_dt = tbl_n["n_lines_ds_yz"].to_numpy()

    sf_3_1_dt = ((nx_dt >= 3) & (ny_dt >= 1)) | ((ny_dt >= 3) & (nx_dt >= 1))
    ds_3_1_dt = ((nx_ds_dt >= 3) & (ny_ds_dt >= 1)) | ((ny_ds_dt >= 3) & (nx_ds_dt >= 1))
    sf_3_0_dt = (nx_dt >= 3) | (ny_dt >= 3)
    ds_3_0_dt = (nx_ds_dt >= 3) | (ny_ds_dt >= 3)
    sf_3_2_dt = ((nx_dt >= 3) & (ny_dt >= 2)) | ((ny_dt >= 3) & (nx_dt >= 2))
    ds_3_2_dt = ((nx_ds_dt >= 3) & (ny_ds_dt >= 2)) | ((ny_ds_dt >= 3) & (nx_ds_dt >= 2))

    cand_mask = sf_3_0_dt | ds_3_0_dt | sf_3_1_dt | ds_3_1_dt | sf_3_2_dt | ds_3_2_dt
    cand_indices = np.where(cand_mask)[0]
    print(f"  -> Found {len(cand_indices):,} candidate events out of {len(nx_dt):,} data events.")

    # Read candidate tracking rows from local cache in 0.5s
    dset_track = ds.dataset(CACHE_TRACKING_PATH, format="parquet")
    df_cand = dset_track.take(cand_indices, columns=TRACKING_COLS).to_pandas()
    compute_derived_kinematics(df_cand)

    nc_sf_xz_s_dt = check_nc(df_cand, "sf", "xz", 260.0, 355.0, strict=True)
    nc_sf_yz_s_dt = check_nc(df_cand, "sf", "yz", 260.0, 355.0, strict=True)
    nc_ds_xz_s_dt = check_nc(df_cand, "ds", "xz", 460.0, 590.0, strict=True)
    nc_ds_yz_s_dt = check_nc(df_cand, "ds", "yz", 460.0, 590.0, strict=True)

    nc_sf_xz_p_dt = check_nc(df_cand, "sf", "xz", 260.0, 355.0, strict=False)
    nc_sf_yz_p_dt = check_nc(df_cand, "sf", "yz", 260.0, 355.0, strict=False)
    nc_ds_xz_p_dt = check_nc(df_cand, "ds", "xz", 460.0, 590.0, strict=False)
    nc_ds_yz_p_dt = check_nc(df_cand, "ds", "yz", 460.0, 590.0, strict=False)

    sf_nc_or_dt = nc_sf_xz_s_dt | nc_sf_yz_s_dt
    ds_nc_or_dt = nc_ds_xz_s_dt | nc_ds_yz_s_dt
    sf_nc_and_dt = nc_sf_xz_s_dt & nc_sf_yz_s_dt
    ds_nc_phys_or_dt = nc_ds_xz_p_dt | nc_ds_yz_p_dt

    c_sf_3_1 = sf_3_1_dt[cand_indices]
    c_ds_3_1 = ds_3_1_dt[cand_indices]
    c_sf_3_0 = sf_3_0_dt[cand_indices]
    c_ds_3_0 = ds_3_0_dt[cand_indices]
    c_sf_3_2 = sf_3_2_dt[cand_indices]
    c_ds_3_2 = ds_3_2_dt[cand_indices]

    cand_masks_dict = {
        "[SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)": (c_sf_3_1 | c_ds_3_1) & (sf_nc_or_dt & ds_nc_or_dt),
        "(SF_3 | DS_3) & (SF_nc & DS_nc)": (c_sf_3_0 | c_ds_3_0) & (sf_nc_or_dt & ds_nc_or_dt),
        "[SF(3,2) | DS(3,2)] & (SF_nc & DS_nc)": (c_sf_3_2 | c_ds_3_2) & (sf_nc_or_dt & ds_nc_or_dt),
        "[SF(3,1) | DS(3,1)] & (SF_nc_both & DS_nc_phys)": (c_sf_3_1 | c_ds_3_1) & (sf_nc_and_dt & ds_nc_phys_or_dt),
        "SF(3,2)_AND_phys & DS(1,0)_OR": (c_sf_3_2 & (nc_sf_xz_p_dt & nc_sf_yz_p_dt)) & (((nx_ds_dt[cand_indices] >= 1) | (ny_ds_dt[cand_indices] >= 1)) & ds_nc_or_dt)
    }

    print(f"  -> Evaluated all candidate criteria masks in {time.time()-t1:.2f}s.")

    # 3. Compute Kolmogorov-Smirnov Tests
    print("\n[3/3] Computing Weighted Kolmogorov-Smirnov Tests across all kinematic variables...")
    results = []

    for i_crit, crit in enumerate(top5_criteria, 1):
        print(f"\n" + "=" * 110)
        print(f"  CRITERION {i_crit}/5: {crit}")
        print("=" * 110)

        sub_data = df_cand[cand_masks_dict[crit]]

        bkg_mask = mc_masks[crit]
        sub_bkg = df_sim_bkg[bkg_mask]

        n_dt_surv = len(sub_data)
        n_mc_surv_w = float(np.sum(sub_bkg["final_weight"]))
        n_mc_surv_raw = len(sub_bkg)

        print(f"    Surviving Collision Data Events : {n_dt_surv:,}")
        print(f"    Surviving Simulated Background : {n_mc_surv_w:.1f} expected ({n_mc_surv_raw} raw events)")

        print(f"\n    {'Variable':<26} | {'Description':<42} | {'D-Stat':<8} | {'p-value':<10} | {'Shape Compatibility'}")
        print("    " + "-" * 108)

        for var_name, var_desc in KS_VARIABLES:
            d_vals = sub_data[var_name].to_numpy()
            m_vals = sub_bkg[var_name].to_numpy()
            w_vals = sub_bkg["final_weight"].to_numpy()

            d_stat, p_val = weighted_ks_2sample(d_vals, m_vals, w_vals)

            compat = "EXCELLENT (p>=0.10)" if p_val >= 0.10 else ("GOOD (p>=0.01)" if p_val >= 0.01 else ("MODERATE (p>=1e-4)" if p_val >= 1e-4 else "DEVIATION"))

            print(f"    {var_name:<26} | {var_desc:<42} | {d_stat:.4f}   | {p_val:<10.4g} | {compat}")

            results.append({
                "criterion_rank": i_crit,
                "criterion": crit,
                "data_surviving": n_dt_surv,
                "mc_surviving_yield": n_mc_surv_w,
                "mc_surviving_raw": n_mc_surv_raw,
                "variable": var_name,
                "description": var_desc,
                "ks_d_stat": d_stat,
                "ks_p_val": p_val,
                "compatibility": compat
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv("ks_kinematic_shape_test_results.csv", index=False)
    print("\n" + "=" * 110)
    print("                         SUMMARY OF MEAN KS D-STATISTICS ACROSS ALL VARIABLES")
    print("=" * 110)
    summary = df_res.groupby("criterion").agg(
        mean_D=("ks_d_stat", "mean"),
        min_p=("ks_p_val", "min"),
        median_p=("ks_p_val", "median"),
        data_surv=("data_surviving", "first"),
        mc_surv=("mc_surviving_yield", "first")
    ).reset_index().sort_values(by="mean_D")

    print(summary.to_string(index=False))
    print("=" * 110)
    print("[+] Detailed test results exported to 'ks_kinematic_shape_test_results.csv'.")


if __name__ == "__main__":
    main()
