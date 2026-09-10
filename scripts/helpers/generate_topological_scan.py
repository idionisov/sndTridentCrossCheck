#!/usr/bin/env python3
"""
================================================================================
SND@LHC Modular Topological Criteria Scan Generator (Cutset 8)
================================================================================
Evaluates 3,366 topological track multiplicity and collinearity criteria across:
  - Collision Data (Run 6640)
  - Trimuon Monte Carlo:
      * 3 Processes : Direct Tridents, Gamma Conversion, e+e- Annihilation
      * 3 Regions   : Rock, Target, Muon Filter
      * Non-Signal Simulation Background (is_signal == 0)
  - Passing Muon MC (PMU FLUKA + Geant4 Background)
  - Gallery Candidate Events:
      * Run 6640 Candidates Specifically (66 events)
      * All Runs Candidates (744 events)

Computes:
  - Expected yields (S) and selection efficiencies (%) for each category
  - Rejection fractions on collision data and simulation backgrounds
  - Exact Asimov Poisson Profile Likelihood Discovery Significance (Cowan et al. 2011)
      (1) Data-driven background: B_data
      (2) Combined Simulation background: B_MC = B_trimuon_bkg + B_PMU

Outputs:
  - ROOT file with TTree ("topological_scan") and JSON metadata ("metadata")
  - Optional CSV table
  - Optional Parquet table
================================================================================
"""

import os
import sys
import ast
import json
import time
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot
import ROOT

try:
    import ddfUtils.root as ddf_root
    HAS_DDF_ROOT = True
except ImportError:
    HAS_DDF_ROOT = False

# Default File Paths for Cutset 8
DEFAULT_DATA_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw_cutset8_hough.parquet"
DEFAULT_MC_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet"
DEFAULT_PMU_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/pmu/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_hough.parquet"
DEFAULT_GALLERY_PATH = "/eos/user/i/idioniso/sndMuTri/data/cutset8/gallery/mu3_search_hough_cutset8.parquet"
CACHE_DATA_PATH = "/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/data_6640_tracking.parquet"

DEFAULT_OUTPUT_ROOT = "topological_criteria_scan_cutset8.root"
DEFAULT_OUTPUT_CSV = "topological_criteria_scan_cutset8.csv"
DEFAULT_OUTPUT_PARQUET = "topological_criteria_scan_cutset8.parquet"

# Luminosity Constants
LUMI_FULL_6640 = 0.9915864253712848  # fb^-1
LUMI_MC_DATASET = 0.0250            # fb^-1
SCALE_PMU_DEFAULT = LUMI_FULL_6640 * 100116.0  # ~99273.66

# Preselection Cuts Applied in Cutset 8
CUTSET8_PRIOR_CUTS = [
    "1. Total Input",
    "2. DS Hits >= 8",
    "3. US Hits >= 5",
    "4. SciFi Hits (9/3 or 3/9)",
    "5. SciFi Max QDC >= 5",
    "6. DS QDC >= 500",
    "7. US QDC >= 40",
    "8. US Max QDC >= 7.5",
    "9. DS/SciFi QDC >= 0.05",
    "10. Last DS Plane = 4",
    "11. Event Delta_t > 100"
]

# Tracking / Hough columns needed for evaluation
TRACKING_COLS = [
    "n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz",
    "xz_sf_m1", "xz_sf_c1", "yz_sf_m1", "yz_sf_c1",
    "xz_sf_m2", "xz_sf_c2", "yz_sf_m2", "yz_sf_c2",
    "xz_sf_m3", "xz_sf_c3", "yz_sf_m3", "yz_sf_c3",
    "xz_ds_m1", "xz_ds_c1", "yz_ds_m1", "yz_ds_c1",
    "xz_ds_m2", "xz_ds_c2", "yz_ds_m2", "yz_ds_c2",
    "xz_ds_m3", "xz_ds_c3", "yz_ds_m3", "yz_ds_c3"
]


# ==============================================================================
# 1. Exact Asimov Profile Likelihood Discovery Significance (Cowan et al. 2011)
# ==============================================================================

def calculate_asimov_significance(s: float, b: float) -> float:
    """
    Computes Exact Asimov Profile Likelihood Discovery Significance (Cowan et al. 2011, Eq. 20):
      Z = sqrt(2 * [ (s + b) * ln(1 + s/b) - s ])
    """
    if s <= 0.0:
        return 0.0
    
    b_val = max(b, 0.0)
    
    if b_val <= 1e-6:
        val = 2.0 * (s * np.log(1.0 + s))
        return float(np.sqrt(max(0.0, val)))
    
    val = 2.0 * ((s + b_val) * np.log(1.0 + s / b_val) - s)
    return float(np.sqrt(max(0.0, val)))


# ==============================================================================
# 2. Pairwise Non-Crossing & Multiplicity Pattern Evaluation
# ==============================================================================

def check_pairwise_non_crossing(
    df: pd.DataFrame,
    prefix: str,
    proj: str,
    z_min: float,
    z_max: float,
    strict: bool = True
) -> np.ndarray:
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


def extract_system_topologies(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    nx = df["n_lines_sf_xz"].to_numpy()
    ny = df["n_lines_sf_yz"].to_numpy()
    nx_ds = df["n_lines_ds_xz"].to_numpy()
    ny_ds = df["n_lines_ds_yz"].to_numpy()

    mult_sf = {
        "(3,3)": (nx >= 3) & (ny >= 3),
        "(3,2)": ((nx >= 3) & (ny >= 2)) | ((ny >= 3) & (nx >= 2)),
        "(3,1)": ((nx >= 3) & (ny >= 1)) | ((ny >= 3) & (nx >= 1)),
        "(3,0)": (nx >= 3) | (ny >= 3),
        "(2,2)": (nx >= 2) & (ny >= 2),
        "(2,1)": ((nx >= 2) & (ny >= 1)) | ((ny >= 2) & (nx >= 1)),
        "(2,0)": (nx >= 2) | (ny >= 2),
        "(1,1)": (nx >= 1) & (ny >= 1),
        "(1,0)": (nx >= 1) | (ny >= 1),
    }

    mult_ds = {
        "(3,3)": (nx_ds >= 3) & (ny_ds >= 3),
        "(3,2)": ((nx_ds >= 3) & (ny_ds >= 2)) | ((ny_ds >= 3) & (nx_ds >= 2)),
        "(3,1)": ((nx_ds >= 3) & (ny_ds >= 1)) | ((ny_ds >= 3) & (nx_ds >= 1)),
        "(3,0)": (nx_ds >= 3) | (ny_ds >= 3),
        "(2,2)": (nx_ds >= 2) & (ny_ds >= 2),
        "(2,1)": ((nx_ds >= 2) & (ny_ds >= 1)) | ((ny_ds >= 2) & (nx_ds >= 1)),
        "(2,0)": (nx_ds >= 2) | (ny_ds >= 2),
        "(1,1)": (nx_ds >= 1) & (ny_ds >= 1),
        "(1,0)": (nx_ds >= 1) | (ny_ds >= 1),
    }

    nc_sf_xz_s = check_pairwise_non_crossing(df, "sf", "xz", 260.0, 355.0, strict=True)
    nc_sf_yz_s = check_pairwise_non_crossing(df, "sf", "yz", 260.0, 355.0, strict=True)
    nc_ds_xz_s = check_pairwise_non_crossing(df, "ds", "xz", 460.0, 590.0, strict=True)
    nc_ds_yz_s = check_pairwise_non_crossing(df, "ds", "yz", 460.0, 590.0, strict=True)

    nc_sf_xz_p = check_pairwise_non_crossing(df, "sf", "xz", 260.0, 355.0, strict=False)
    nc_sf_yz_p = check_pairwise_non_crossing(df, "sf", "yz", 260.0, 355.0, strict=False)
    nc_ds_xz_p = check_pairwise_non_crossing(df, "ds", "xz", 460.0, 590.0, strict=False)
    nc_ds_yz_p = check_pairwise_non_crossing(df, "ds", "yz", 460.0, 590.0, strict=False)

    coll_sf = {
        "OFF": np.ones(len(df), dtype=bool),
        "OR": nc_sf_xz_s | nc_sf_yz_s,
        "AND": nc_sf_xz_s & nc_sf_yz_s,
        "OR_phys": nc_sf_xz_p | nc_sf_yz_p,
        "AND_phys": nc_sf_xz_p & nc_sf_yz_p,
    }
    coll_ds = {
        "OFF": np.ones(len(df), dtype=bool),
        "OR": nc_ds_xz_s | nc_ds_yz_s,
        "AND": nc_ds_xz_s & nc_ds_yz_s,
        "OR_phys": nc_ds_xz_p | nc_ds_yz_p,
        "AND_phys": nc_ds_xz_p & nc_ds_yz_p,
    }

    return mult_sf, mult_ds, coll_sf, coll_ds


def build_base_criteria(mult_sf, mult_ds, coll_sf, coll_ds) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    crit_sf = {f"{m}_{c}": mult_sf[m] & coll_sf[c] for m in mult_sf for c in coll_sf}
    crit_ds = {f"{m}_{c}": mult_ds[m] & coll_ds[c] for m in mult_ds for c in coll_ds}
    return crit_sf, crit_ds


def parse_k(k: str) -> list:
    parts = k.split("_")
    p = parts[0]
    c = "_".join(parts[1:])
    n1, n2 = [int(x) for x in p.strip("()").split(",")]
    return [n1, n2, c]


# ==============================================================================
# 3. Intuitive Label Formatter
# ==============================================================================

def format_intuitive_label(lbl: str) -> str:
    try:
        parts = ast.literal_eval(lbl)
    except Exception:
        return lbl

    mode = parts[0]
    if mode == "FACTORIZED_OR_AND":
        n1, n2 = parts[1]
        csf, cds = parts[2]
        m_str = f"[SF({n1},{n2}) | DS({n1},{n2})]" if (n1, n2) != (3, 0) else "(SF_3 | DS_3)"
        if csf == "OR" and cds == "OR":
            c_str = "(SF_nc & DS_nc)"
        elif csf == "AND" and cds == "OFF":
            c_str = "(SF_nc_both)"
        elif csf == "AND" and cds == "OR_phys":
            c_str = "(SF_nc_both & DS_nc_phys)"
        elif csf == "OR" and cds == "OFF":
            c_str = "(SF_nc)"
        else:
            c_str = f"(SF:{csf} & DS:{cds})"
        return f"{m_str} & {c_str}"

    if mode == "SF_ONLY":
        sf_m, sf_c = parts[1][:2], parts[1][2]
        return f"SF({sf_m[0]},{sf_m[1]})_{sf_c} ONLY"

    if mode == "DS_ONLY":
        ds_m, ds_c = parts[2][:2], parts[2][2]
        return f"DS({ds_m[0]},{ds_m[1]})_{ds_c} ONLY"

    if mode == "AND":
        sf_m, sf_c = parts[1][:2], parts[1][2]
        ds_m, ds_c = parts[2][:2], parts[2][2]
        return f"SF({sf_m[0]},{sf_m[1]})_{sf_c} & DS({ds_m[0]},{ds_m[1]})_{ds_c}"

    if mode == "OR":
        sf_m, sf_c = parts[1][:2], parts[1][2]
        ds_m, ds_c = parts[2][:2], parts[2][2]
        return f"SF({sf_m[0]},{sf_m[1]})_{sf_c} OR DS({ds_m[0]},{ds_m[1]})_{ds_c}"

    return lbl


# ==============================================================================
# 4. Data Loading & Extraction
# ==============================================================================

def load_parquet_datasets(
    data_path: str,
    mc_path: str,
    pmu_path: str,
    gallery_path: str
) -> Dict[str, pd.DataFrame]:
    t0 = time.time()
    print("[1/5] Loading Parquet tracking columns...")

    print("  -> Loading Gallery Data (partitioning Run 6640 and All Runs)...")
    gal_cols = TRACKING_COLS + (["run_id"] if "run_id" in pq.ParquetFile(gallery_path).schema.names else [])
    df_gal = pq.read_table(gallery_path, columns=gal_cols).to_pandas()
    print(f"     Loaded {len(df_gal):,} total gallery events across all runs.")

    print("  -> Loading Passing Muon MC (PMU)...")
    pmu_cols = TRACKING_COLS + ["mc_weight"]
    df_pmu = pq.read_table(pmu_path, columns=pmu_cols).to_pandas()
    print(f"     Loaded {len(df_pmu):,} PMU events.")

    print("  -> Loading Trimuon MC...")
    mc_cols = TRACKING_COLS + ["is_signal", "mc_weight", "region_type", "proc_type"]
    df_mc = pq.read_table(mc_path, columns=mc_cols).to_pandas()
    print(f"     Loaded {len(df_mc):,} Trimuon MC events.")

    print("  -> Loading Collision Data (Run 6640)...")
    if os.path.exists(CACHE_DATA_PATH):
        print(f"     [Using fast local cache: {CACHE_DATA_PATH}]")
        df_data = pq.read_table(CACHE_DATA_PATH).to_pandas()
    else:
        df_data = pq.read_table(data_path, columns=TRACKING_COLS).to_pandas()
    print(f"     Loaded {len(df_data):,} Data events in {time.time() - t0:.2f}s total.")

    return {
        "data": df_data,
        "mc": df_mc,
        "pmu": df_pmu,
        "gallery": df_gal
    }


def extract_sample_masks_and_weights(
    df_mc: pd.DataFrame,
    df_pmu: pd.DataFrame,
    df_gal: pd.DataFrame,
    scale_mc: float,
    scale_pmu: float
) -> Dict[str, Any]:
    is_sig = (df_mc["is_signal"] == 1).to_numpy()
    is_bkg = (df_mc["is_signal"] == 0).to_numpy()
    proc = df_mc["proc_type"].to_numpy()
    reg = df_mc["region_type"].to_numpy()
    w_mc = df_mc["mc_weight"].to_numpy()
    w_pmu = df_pmu["mc_weight"].to_numpy()

    # Gallery Run 6640 mask
    gal_6640_mask = (df_gal["run_id"] == 6640).to_numpy() if "run_id" in df_gal.columns else np.ones(len(df_gal), dtype=bool)

    masks = {
        "direct_rock": is_sig & (proc == 0) & (reg == 1),
        "direct_tgt": is_sig & (proc == 0) & (reg == 2),
        "direct_mu": is_sig & (proc == 0) & (reg == 3),
        "gamma_rock": is_sig & (proc == 1) & (reg == 1),
        "gamma_tgt": is_sig & (proc == 1) & (reg == 2),
        "gamma_mu": is_sig & (proc == 1) & (reg == 3),
        "annihil_rock": is_sig & (proc == 2) & (reg == 1),
        "annihil_tgt": is_sig & (proc == 2) & (reg == 2),
        "annihil_mu": is_sig & (proc == 2) & (reg == 3),
        
        "rock_all": is_sig & (reg == 1),
        "tgt_all": is_sig & (reg == 2),
        "mu_all": is_sig & (reg == 3),
        
        "direct_all": is_sig & (proc == 0),
        "gamma_all": is_sig & (proc == 1),
        "annihil_all": is_sig & (proc == 2),
        
        "total_signal": is_sig,
        "trimuon_bkg": is_bkg
    }

    initial_weights = {k: float(np.sum(w_mc[m])) for k, m in masks.items()}
    initial_yields = {k: w * scale_mc for k, w in initial_weights.items()}

    pmu_init_weight = float(np.sum(w_pmu))
    pmu_init_yield = pmu_init_weight * scale_pmu

    return {
        "masks": masks,
        "w_mc": w_mc,
        "w_pmu": w_pmu,
        "gal_6640_mask": gal_6640_mask,
        "initial_weights": initial_weights,
        "initial_yields": initial_yields,
        "pmu_init_weight": pmu_init_weight,
        "pmu_init_yield": pmu_init_yield
    }


# ==============================================================================
# 5. Core Variety Evaluation Function
# ==============================================================================

def evaluate_single_variety(
    regime: str,
    label: str,
    mask_mc: np.ndarray,
    mask_pmu: np.ndarray,
    mask_dt: np.ndarray,
    mask_gal: np.ndarray,
    sample_info: Dict[str, Any],
    tot_data_len: int,
    tot_gal_all: int,
    tot_gal_6640: int,
    scale_mc: float,
    scale_pmu: float
) -> Dict[str, Any]:
    w_mc = sample_info["w_mc"]
    w_pmu = sample_info["w_pmu"]
    masks = sample_info["masks"]
    gal_6640_mask = sample_info["gal_6640_mask"]
    init_w = sample_info["initial_weights"]

    n_dt = int(np.count_nonzero(mask_dt))
    data_rej = float((1.0 - (n_dt / tot_data_len)) * 100.0)

    # Gallery Survivals: Run 6640 and All Runs
    n_gal_6640 = int(np.count_nonzero(mask_gal & gal_6640_mask))
    gal_6640_surv = float((n_gal_6640 / tot_gal_6640) * 100.0 if tot_gal_6640 > 0 else 0.0)

    n_gal_all = int(np.count_nonzero(mask_gal))
    gal_all_surv = float((n_gal_all / tot_gal_all) * 100.0 if tot_gal_all > 0 else 0.0)

    # Simulation Backgrounds
    bkg_tri_w = float(np.sum(w_mc[masks["trimuon_bkg"] & mask_mc]))
    bkg_tri_yield = bkg_tri_w * scale_mc
    bkg_pmu_w = float(np.sum(w_pmu[mask_pmu]))
    bkg_pmu_yield = bkg_pmu_w * scale_pmu
    bkg_mc_total = bkg_tri_yield + bkg_pmu_yield

    b_data = float(n_dt)
    b_mc = float(bkg_mc_total)

    category_keys = [
        "direct_rock", "direct_tgt", "direct_mu",
        "gamma_rock", "gamma_tgt", "gamma_mu",
        "annihil_rock", "annihil_tgt", "annihil_mu",
        "rock_all", "tgt_all", "mu_all",
        "direct_all", "gamma_all", "annihil_all",
        "total_signal"
    ]

    row_data = {
        "regime": regime,
        "label": label,
        "clean_label": format_intuitive_label(label),
        "data_events": n_dt,
        "data_rej_pct": data_rej,
        "gallery_events": n_gal_6640,          # Primary: Run 6640 specifically
        "gallery_surv_pct": gal_6640_surv,
        "gallery_6640_events": n_gal_6640,
        "gallery_6640_surv_pct": gal_6640_surv,
        "gallery_all_events": n_gal_all,       # Secondary: Across all 22 runs
        "gallery_all_surv_pct": gal_all_surv,
        "bkg_mc_total": bkg_mc_total,
        "bkg_trimuon": bkg_tri_yield,
        "bkg_pmu": bkg_pmu_yield
    }

    for cat in category_keys:
        surv_w = float(np.sum(w_mc[masks[cat] & mask_mc]))
        s_yield = surv_w * scale_mc
        eff_pct = (surv_w / init_w[cat] * 100.0) if init_w[cat] > 0 else 0.0
        
        z_data = calculate_asimov_significance(s_yield, b_data)
        z_mc = calculate_asimov_significance(s_yield, b_mc)

        row_data[f"yield_{cat}"] = s_yield
        row_data[f"eff_{cat}"] = eff_pct
        row_data[f"sig_data_{cat}"] = z_data
        row_data[f"sig_mc_{cat}"] = z_mc

    return row_data


# ==============================================================================
# 6. Evaluation Orchestrator
# ==============================================================================

def run_scan_evaluation(
    datasets: Dict[str, pd.DataFrame],
    sample_info: Dict[str, Any],
    scale_mc: float,
    scale_pmu: float
) -> pd.DataFrame:
    df_data = datasets["data"]
    df_mc = datasets["mc"]
    df_pmu = datasets["pmu"]
    df_gal = datasets["gallery"]

    tot_data_len = len(df_data)
    tot_gal_all = len(df_gal)
    tot_gal_6640 = int(np.count_nonzero(sample_info["gal_6640_mask"]))

    t0 = time.time()
    print("\n[3/5] Extracting detector multiplicity & non-crossing components...")
    m_sf_mc, m_ds_mc, c_sf_mc, c_ds_mc = extract_system_topologies(df_mc)
    m_sf_pmu, m_ds_pmu, c_sf_pmu, c_ds_pmu = extract_system_topologies(df_pmu)
    m_sf_dt, m_ds_dt, c_sf_dt, c_ds_dt = extract_system_topologies(df_data)
    m_sf_gal, m_ds_gal, c_sf_gal, c_ds_gal = extract_system_topologies(df_gal)

    crit_sf_mc, crit_ds_mc = build_base_criteria(m_sf_mc, m_ds_mc, c_sf_mc, c_ds_mc)
    crit_sf_pmu, crit_ds_pmu = build_base_criteria(m_sf_pmu, m_ds_pmu, c_sf_pmu, c_ds_pmu)
    crit_sf_dt, crit_ds_dt = build_base_criteria(m_sf_dt, m_ds_dt, c_sf_dt, c_ds_dt)
    crit_sf_gal, crit_ds_gal = build_base_criteria(m_sf_gal, m_ds_gal, c_sf_gal, c_ds_gal)

    keys_45 = list(crit_sf_mc.keys())
    mult_keys = list(m_sf_mc.keys())
    coll_keys = ["OFF", "OR", "AND", "OR_phys", "AND_phys"]

    print(f"  Extracted base dictionaries in {time.time() - t0:.2f}s.")

    print("\n[4/5] Evaluating all 3,366 varieties on-the-fly...")
    t1 = time.time()
    records = []

    # A. SF_ONLY (45)
    for k in keys_45:
        records.append(evaluate_single_variety(
            "SF_ONLY",
            f"('SF_ONLY', {parse_k(k)}, None)",
            crit_sf_mc[k], crit_sf_pmu[k], crit_sf_dt[k], crit_sf_gal[k],
            sample_info, tot_data_len, tot_gal_all, tot_gal_6640, scale_mc, scale_pmu
        ))

    # B. DS_ONLY (45)
    for k in keys_45:
        records.append(evaluate_single_variety(
            "DS_ONLY",
            f"('DS_ONLY', None, {parse_k(k)})",
            crit_ds_mc[k], crit_ds_pmu[k], crit_ds_dt[k], crit_ds_gal[k],
            sample_info, tot_data_len, tot_gal_all, tot_gal_6640, scale_mc, scale_pmu
        ))

    # C. MODULAR_AND (2,025)
    for k_sf in keys_45:
        m_mc_sf = crit_sf_mc[k_sf]
        m_pmu_sf = crit_sf_pmu[k_sf]
        m_dt_sf = crit_sf_dt[k_sf]
        m_gal_sf = crit_sf_gal[k_sf]
        for k_ds in keys_45:
            records.append(evaluate_single_variety(
                "MODULAR_AND",
                f"('AND', {parse_k(k_sf)}, {parse_k(k_ds)})",
                m_mc_sf & crit_ds_mc[k_ds],
                m_pmu_sf & crit_ds_pmu[k_ds],
                m_dt_sf & crit_ds_dt[k_ds],
                m_gal_sf & crit_ds_gal[k_ds],
                sample_info, tot_data_len, tot_gal_all, tot_gal_6640, scale_mc, scale_pmu
            ))

    # D. MODULAR_OR (1,035)
    for i in range(len(keys_45)):
        k_sf = keys_45[i]
        m_mc_sf = crit_sf_mc[k_sf]
        m_pmu_sf = crit_sf_pmu[k_sf]
        m_dt_sf = crit_sf_dt[k_sf]
        m_gal_sf = crit_sf_gal[k_sf]
        for j in range(i, len(keys_45)):
            k_ds = keys_45[j]
            records.append(evaluate_single_variety(
                "MODULAR_OR",
                f"('OR', {parse_k(k_sf)}, {parse_k(k_ds)})",
                m_mc_sf | crit_ds_mc[k_ds],
                m_pmu_sf | crit_ds_pmu[k_ds],
                m_dt_sf | crit_ds_dt[k_ds],
                m_gal_sf | crit_ds_gal[k_ds],
                sample_info, tot_data_len, tot_gal_all, tot_gal_6640, scale_mc, scale_pmu
            ))

    # E. FACTORIZED_OR_AND (216)
    for m_name in mult_keys:
        n1, n2 = [int(x) for x in m_name.strip("()").split(",")]
        m_mult_mc = m_sf_mc[m_name] | m_ds_mc[m_name]
        m_mult_pmu = m_sf_pmu[m_name] | m_ds_pmu[m_name]
        m_mult_dt = m_sf_dt[m_name] | m_ds_dt[m_name]
        m_mult_gal = m_sf_gal[m_name] | m_ds_gal[m_name]

        for c_sf in coll_keys:
            for c_ds in coll_keys:
                if c_sf == "OFF" and c_ds == "OFF":
                    continue
                label = f"('FACTORIZED_OR_AND', [{n1}, {n2}], ['{c_sf}', '{c_ds}'] )"
                m_coll_mc = c_sf_mc[c_sf] & c_ds_mc[c_ds]
                m_coll_pmu = c_sf_pmu[c_sf] & c_ds_pmu[c_ds]
                m_coll_dt = c_sf_dt[c_sf] & c_ds_dt[c_ds]
                m_coll_gal = c_sf_gal[c_sf] & c_ds_gal[c_ds]

                records.append(evaluate_single_variety(
                    "FACTORIZED_OR_AND",
                    label,
                    m_mult_mc & m_coll_mc,
                    m_mult_pmu & m_coll_pmu,
                    m_mult_dt & m_coll_dt,
                    m_mult_gal & m_coll_gal,
                    sample_info, tot_data_len, tot_gal_all, tot_gal_6640, scale_mc, scale_pmu
                ))

    df_result = pd.DataFrame(records)
    print(f"  Completed all {len(df_result):,} evaluations in {time.time() - t1:.2f}s.")
    return df_result


# ==============================================================================
# 7. Metadata Builder & Exporter
# ==============================================================================

def generate_column_descriptions() -> Dict[str, Any]:
    return {
        "topological_scan": {
            "regime": "Combination category: SF_ONLY, DS_ONLY, MODULAR_AND, MODULAR_OR, FACTORIZED_OR_AND",
            "label": "Technical tuple representation of the criteria configuration",
            "clean_label": "Human-readable intuitive criteria formula (e.g. [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc))",
            "data_events": "Number of surviving collision data events in Run 6640 (Data Background B_data)",
            "data_rej_pct": "Background rejection efficiency on collision data: (1 - N_data / N_tot) * 100%",
            "gallery_events": "Number of surviving gallery candidate events in Run 6640 specifically (out of 66)",
            "gallery_surv_pct": "Percentage of Run 6640 gallery candidate events surviving the topological cut",
            "gallery_6640_events": "Number of surviving gallery candidate events in Run 6640 specifically (out of 66)",
            "gallery_6640_surv_pct": "Percentage of Run 6640 gallery candidate events surviving the topological cut",
            "gallery_all_events": "Number of surviving gallery candidate events across all 22 runs (out of 744)",
            "gallery_all_surv_pct": "Percentage of total gallery candidate events surviving the cut",
            "bkg_mc_total": "Total expected simulation background yield: B_MC = B_trimuon + B_PMU (0.992 fb^-1)",
            "bkg_trimuon": "Expected non-signal background yield in Trimuon MC sample (is_signal == 0)",
            "bkg_pmu": "Expected passing muon background yield from FLUKA+Geant4 PMU MC (scaled by lumi*100116)",
            "yield_<cat>": "Expected signal yield in Run 6640 (0.992 fb^-1) for category <cat>",
            "eff_<cat>": "Cumulative selection efficiency (%) for category <cat> relative to initial input",
            "sig_data_<cat>": "Exact Asimov Poisson Profile Likelihood significance Z using Collision Data background (B_data)",
            "sig_mc_<cat>": "Exact Asimov Poisson Profile Likelihood significance Z using Combined Simulation background (B_MC)"
        },
        "categories_glossary": {
            "direct_rock": "Direct / Genuine electromagnetic trident (mu -> mu mu+ mu-) in upstream rock (Z < 260 cm)",
            "direct_tgt": "Direct / Genuine trident in tungsten target & SciFi (260 <= Z <= 355 cm)",
            "direct_mu": "Direct / Genuine trident in upstream/downstream muon filter (Z > 355 cm)",
            "gamma_rock": "Bremsstrahlung photon conversion (gamma -> mu+ mu-) in upstream rock",
            "gamma_tgt": "Bremsstrahlung photon conversion (gamma -> mu+ mu-) in target & SciFi",
            "gamma_mu": "Bremsstrahlung photon conversion (gamma -> mu+ mu-) in muon filter",
            "annihil_rock": "Positron-electron annihilation (e+ e- -> mu+ mu-) in upstream rock",
            "annihil_tgt": "Positron-electron annihilation (e+ e- -> mu+ mu-) in target & SciFi",
            "annihil_mu": "Positron-electron annihilation (e+ e- -> mu+ mu-) in muon filter",
            "rock_all": "All signal tridents produced in the upstream cavern rock (region_type == 1)",
            "tgt_all": "All signal tridents produced in target & SciFi (region_type == 2)",
            "mu_all": "All signal tridents produced in muon filter (region_type == 3)",
            "direct_all": "All direct / genuine tridents across all fiducial regions (proc_type == 0)",
            "gamma_all": "All gamma conversion tridents across all fiducial regions (proc_type == 1)",
            "annihil_all": "All positron annihilation tridents across all fiducial regions (proc_type == 2)",
            "total_signal": "Total inclusive signal tridents across all processes and regions"
        }
    }


def build_metadata(
    data_path: str,
    mc_path: str,
    pmu_path: str,
    gallery_path: str,
    lumi_full: float,
    lumi_mc: float,
    scale_mc: float,
    scale_pmu: float,
    sample_info: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    return {
        "name": "metadata",
        "description": "SND@LHC 3,366 Topological Criteria Scan Results with Exact Asimov Poisson Significance",
        "generator_script": "scripts/helpers/generate_topological_scan.py",
        "created_at": datetime.datetime.now().isoformat(),
        "cutset_applied_prior": "cutset8",
        "prior_cutflow_steps": CUTSET8_PRIOR_CUTS,
        "input_files": {
            "collision_data": data_path,
            "trimuon_mc": mc_path,
            "pmu_mc": pmu_path,
            "gallery_data": gallery_path
        },
        "normalization": {
            "target_luminosity_fb": lumi_full,
            "mc_generated_luminosity_fb": lumi_mc,
            "scale_factor_mc_to_full": scale_mc,
            "scale_factor_pmu_to_full": scale_pmu,
            "initial_uncut_yields": sample_info["initial_yields"],
            "initial_pmu_expected_bkg": sample_info["pmu_init_yield"]
        },
        "sample_statistics": {
            "total_input_data_events": len(datasets["data"]),
            "total_input_mc_events": len(datasets["mc"]),
            "total_input_pmu_events": len(datasets["pmu"]),
            "total_input_gallery_all": len(datasets["gallery"]),
            "total_input_gallery_6640": int(np.count_nonzero(sample_info["gal_6640_mask"])),
            "total_topological_varieties": 3366
        },
        "significance_method": {
            "name": "Exact Asimov Profile Likelihood Discovery Significance (Cowan et al. 2011)",
            "formula": "Z = sqrt(2 * [(S+B)*ln(1 + S/B) - S])",
            "background_models": ["B_data = N_data_events", "B_MC = B_trimuon_bkg + B_PMU"]
        },
        "column_descriptions": generate_column_descriptions()
    }


def export_results(
    df_result: pd.DataFrame,
    metadata: Dict[str, Any],
    output_root: Optional[str] = DEFAULT_OUTPUT_ROOT,
    output_csv: Optional[str] = DEFAULT_OUTPUT_CSV,
    output_parquet: Optional[str] = DEFAULT_OUTPUT_PARQUET,
    tree_name: str = "topological_scan"
):
    print("\n[5/5] Exporting scan tables & ROOT files...")

    if output_root:
        os.makedirs(os.path.dirname(os.path.abspath(output_root)), exist_ok=True)
        with uproot.recreate(output_root) as f_root:
            f_root[tree_name] = df_result

        if HAS_DDF_ROOT:
            ddf_root.write_metadata(metadata, output_root, close_file=True)
        else:
            f_r = ROOT.TFile.Open(output_root, "UPDATE")
            r_str = ROOT.TObjString(json.dumps(metadata))
            r_str.Write("metadata")
            f_r.Close()

        root_size_kb = os.path.getsize(output_root) / 1024
        print(f"  [+] Saved ROOT file: '{output_root}' (TTree: '{tree_name}', Metadata: 'metadata', {root_size_kb:.1f} KB)")

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df_result.to_csv(output_csv, index=False)
        csv_size_kb = os.path.getsize(output_csv) / 1024
        print(f"  [+] Saved CSV table: '{output_csv}' ({len(df_result):,} rows, {csv_size_kb:.1f} KB)")

    if output_parquet:
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
        df_result.to_parquet(output_parquet, index=False)
        pq_size_kb = os.path.getsize(output_parquet) / 1024
        print(f"  [+] Saved Parquet table: '{output_parquet}' ({pq_size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Modular 3,366 Topological Criteria Scan Generator with Asimov Significance for Cutset 8."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--mc", default=DEFAULT_MC_PATH)
    parser.add_argument("--pmu", default=DEFAULT_PMU_PATH)
    parser.add_argument("--gallery", default=DEFAULT_GALLERY_PATH)
    parser.add_argument("-o", "--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tree-name", default="topological_scan")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-parquet", default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--lumi", type=float, default=LUMI_FULL_6640)
    parser.add_argument("--lumi-mc", type=float, default=LUMI_MC_DATASET)
    parser.add_argument("--scale-pmu", type=float, default=SCALE_PMU_DEFAULT)

    args = parser.parse_args()
    scale_mc = args.lumi / args.lumi_mc
    scale_pmu = args.scale_pmu

    datasets = load_parquet_datasets(args.data, args.mc, args.pmu, args.gallery)
    sample_info = extract_sample_masks_and_weights(datasets["mc"], datasets["pmu"], datasets["gallery"], scale_mc, scale_pmu)
    df_result = run_scan_evaluation(datasets, sample_info, scale_mc, scale_pmu)
    metadata = build_metadata(args.data, args.mc, args.pmu, args.gallery, args.lumi, args.lumi_mc, scale_mc, scale_pmu, sample_info, datasets)

    csv_out = args.output_csv if args.output_csv != "" else None
    pq_out = args.output_parquet if args.output_parquet != "" else None
    export_results(df_result, metadata, args.output_root, csv_out, pq_out, args.tree_name)


if __name__ == "__main__":
    main()
