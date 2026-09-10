#!/usr/bin/env python3
"""
Produce High-Purity Topological Distributions: [SF(3,3)_AND & DS(1,0)]
========================================================================
Applies the Ultra-High Purity selection:
  - SF(3,3)_AND: n_lines_sf_xz >= 3 & n_lines_sf_yz >= 3
  - DS(1,0):     n_lines_ds_xz >= 1
  - SciFi Non-Crossing in both XZ and YZ projections.

Outputs to /eos/user/i/idioniso/sndMuTri/out/high_purity/:
  1. Parquet Tables:
     - gallery_all_passed.parquet & gallery_all_failed.parquet
     - gallery_6640_passed.parquet & gallery_6640_failed.parquet
     - data_6640_passed.parquet
     - trimuon_mc_signal_passed.parquet
     - trimuon_mc_nonsignal_passed.parquet
     - trimuon_mc_all_passed.parquet
  2. ROOT File with TNtuples & TH1D Histograms:
     - topological_distributions_high_purity.root
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ROOT
import uproot
import ddfUtils.root as ddf_root

ROOT.gROOT.SetBatch(True)

# ==============================================================================
# 0. CONFIGURATION & FILE PATHS
# ==============================================================================
OUT_DIR = Path("/eos/user/i/idioniso/sndMuTri/out/high_purity")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT = OUT_DIR / "topological_distributions_high_purity.root"

PGUN_ROOT = Path("/eos/user/i/idioniso/sndMuTri/out/pGun_response.root")
MC_PATH = Path("/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet")
DATA_PATH = Path("/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw_cutset8_hough.parquet")
DATA_CACHE = Path("/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/data_6640_tracking.parquet")
GAL_PATH = Path("/eos/user/i/idioniso/sndMuTri/data/cutset8/gallery/mu3_search_hough_cutset8.parquet")

# Columns needed for tracking and detector variables
DET_COLS = ["scifi_nhits", "scifi_sum_qdc", "ds_nhits", "ds_sum_qdc"]
TRACK_COLS = [
    "n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz",
    "xz_sf_m1", "xz_sf_c1", "yz_sf_m1", "yz_sf_c1",
    "xz_sf_m2", "xz_sf_c2", "yz_sf_m2", "yz_sf_c2",
    "xz_sf_m3", "xz_sf_c3", "yz_sf_m3", "yz_sf_c3",
] + DET_COLS

# Load baselines from pGun metadata
with uproot.open(PGUN_ROOT) as f_meta:
    meta = json.loads(str(f_meta["metadata"])) if "metadata" in f_meta else {}
    w_bar_sf = meta.get("w_bar_SF", 2.183)
    qdc_bar_sf = meta.get("qdc_bar_SF", 2.576)
    w_bar_ds = meta.get("w_bar_DS", 2.120)
    qdc_bar_ds = meta.get("qdc_bar_DS", 102.340)

print(f"[*] Loaded MIP Baselines: SF (w={w_bar_sf:.3f}, QDC={qdc_bar_sf:.3f}) | DS (w={w_bar_ds:.3f}, QDC={qdc_bar_ds:.2f})")

# ==============================================================================
# 1. SELECTION & VARIABLE COMPUTATION FUNCTIONS
# ==============================================================================
def check_nc(df: pd.DataFrame, prefix: str, proj: str, z_min: float, z_max: float) -> np.ndarray:
    n = df[f"n_lines_{prefix}_{proj}"].to_numpy()
    m1, c1 = np.nan_to_num(df[f"{proj}_{prefix}_m1"].to_numpy()), np.nan_to_num(df[f"{proj}_{prefix}_c1"].to_numpy())
    m2, c2 = np.nan_to_num(df[f"{proj}_{prefix}_m2"].to_numpy()), np.nan_to_num(df[f"{proj}_{prefix}_c2"].to_numpy())
    m3, c3 = np.nan_to_num(df[f"{proj}_{prefix}_m3"].to_numpy()), np.nan_to_num(df[f"{proj}_{prefix}_c3"].to_numpy())
    
    x1_min, x1_max = m1 * z_min + c1, m1 * z_max + c1
    x2_min, x2_max = m2 * z_min + c2, m2 * z_max + c2
    x3_min, x3_max = m3 * z_min + c3, m3 * z_max + c3
    
    no_cross_12 = ((x1_min - x2_min) * (x1_max - x2_max)) > 0
    no_cross_13 = ((x1_min - x3_min) * (x1_max - x3_max)) > 0
    no_cross_23 = ((x2_min - x3_min) * (x2_max - x3_max)) > 0
    
    pass_3 = (n >= 3) & no_cross_12 & no_cross_13 & no_cross_23
    pass_2 = (n == 2) & no_cross_12
    return pass_3 | pass_2

def apply_high_purity_mask(df: pd.DataFrame) -> np.ndarray:
    """Evaluates [SF(3,3)_AND & DS(1,0)] condition with fast candidate pre-filtering."""
    nx, ny = df["n_lines_sf_xz"].to_numpy(), df["n_lines_sf_yz"].to_numpy()
    nx_ds = df["n_lines_ds_xz"].to_numpy()
    
    cand_mask = (nx >= 3) & (ny >= 3) & (nx_ds >= 1)
    final_mask = np.zeros(len(df), dtype=bool)
    
    if not np.any(cand_mask):
        return final_mask
        
    df_cand = df[cand_mask]
    coll_sf = check_nc(df_cand, "sf", "xz", 260.0, 355.0) & check_nc(df_cand, "sf", "yz", 260.0, 355.0)
    final_mask[cand_mask] = coll_sf
    return final_mask

def compute_event_variables(df: pd.DataFrame, is_mc: bool = False) -> pd.DataFrame:
    """Computes the 8 per-plane and MIP-normalized physics variables."""
    hits_sf = df["scifi_nhits"].to_numpy(dtype=np.float64) / 10.0
    qdc_sf = df["scifi_sum_qdc"].to_numpy(dtype=np.float64) / 10.0
    hits_ds = df["ds_nhits"].to_numpy(dtype=np.float64) / 4.0
    qdc_ds = df["ds_sum_qdc"].to_numpy(dtype=np.float64) / 4.0

    emip_sf = 0.5 * ((hits_sf / w_bar_sf) + (qdc_sf / qdc_bar_sf))
    valid_sf = (hits_sf > 0) & (qdc_sf >= 0)
    rhomip_sf = np.where(valid_sf, (qdc_sf / qdc_bar_sf) / (hits_sf / w_bar_sf), 0.0)

    emip_ds = 0.5 * ((hits_ds / w_bar_ds) + (qdc_ds / qdc_bar_ds))
    valid_ds = (hits_ds > 0) & (qdc_ds >= 0)
    rhomip_ds = np.where(valid_ds, (qdc_ds / qdc_bar_ds) / (hits_ds / w_bar_ds), 0.0)

    weights = df["mc_weight"].to_numpy(dtype=np.float64) if is_mc and "mc_weight" in df.columns else np.ones(len(df), dtype=np.float64)
    run_ids = df["run_id"].to_numpy(dtype=np.int64) if "run_id" in df.columns else np.full(len(df), 6640, dtype=np.int64)

    return pd.DataFrame({
        "hits_plane_sf": hits_sf,
        "qdc_plane_sf": qdc_sf,
        "emip_sf": emip_sf,
        "rhomip_sf": rhomip_sf,
        "hits_plane_ds": hits_ds,
        "qdc_plane_ds": qdc_ds,
        "emip_ds": emip_ds,
        "rhomip_ds": rhomip_ds,
        "weight": weights,
        "run_id": run_ids,
    })

# ==============================================================================
# 2. PROCESSING SAMPLES
# ==============================================================================
t_start = time.time()

print("\n[+] 1. Processing Gallery Samples...")
t0 = time.time()
gal_full = pq.read_table(GAL_PATH, columns=TRACK_COLS + ["run_id"]).to_pandas()
mask_gal_all = apply_high_purity_mask(gal_full)

df_gal_all_passed = gal_full[mask_gal_all].copy()
df_gal_all_failed = gal_full[~mask_gal_all].copy()

df_gal_6640_full = gal_full[gal_full["run_id"] == 6640].copy()
mask_gal_6640 = apply_high_purity_mask(df_gal_6640_full)
df_gal_6640_passed = df_gal_6640_full[mask_gal_6640].copy()
df_gal_6640_failed = df_gal_6640_full[~mask_gal_6640].copy()

print(f"    Gallery All ({len(gal_full)} total): Passed = {len(df_gal_all_passed)}, Failed = {len(df_gal_all_failed)} in {time.time()-t0:.2f}s")
print(f"    Gallery 6640 ({len(df_gal_6640_full)} total): Passed = {len(df_gal_6640_passed)}, Failed = {len(df_gal_6640_failed)}")

print("\n[+] 2. Processing Trimuon MC Samples...")
t0 = time.time()
mc_full = pq.read_table(MC_PATH, columns=TRACK_COLS + ["is_signal", "region_type", "mc_weight"]).to_pandas()
mask_mc = apply_high_purity_mask(mc_full)
df_mc_all_passed = mc_full[mask_mc].copy()

sig_mask = (mc_full["is_signal"] == 1) & (mc_full["region_type"] == 1)
nonsig_mask = (mc_full["is_signal"] == 0)

df_mc_sig_passed = mc_full[sig_mask & mask_mc].copy()
df_mc_nonsig_passed = mc_full[nonsig_mask & mask_mc].copy()

print(f"    Trimuon MC Signal ({np.sum(sig_mask):,} total): Passed = {len(df_mc_sig_passed):,}")
print(f"    Trimuon MC Non-Signal ({np.sum(nonsig_mask):,} total): Passed = {len(df_mc_nonsig_passed):,}")
print(f"    Trimuon MC All ({len(mc_full):,} total): Passed = {len(df_mc_all_passed):,} in {time.time()-t0:.2f}s")

print("\n[+] 3. Processing Collision Data (Run 6640)...")
t0 = time.time()
df_data_tr = pd.read_parquet(DATA_CACHE)
mask_data = apply_high_purity_mask(df_data_tr)
pass_indices = np.where(mask_data)[0]
pass_set = set(pass_indices)
print(f"    Data Run 6640 Tracking: {len(df_data_tr):,} events -> Passed = {len(pass_indices):,} in {time.time()-t0:.2f}s")

# Extract the 4 detector columns for the passing rows
pf_data = pq.ParquetFile(DATA_PATH)
row_offset = 0
collected_dfs = []

for rg_i in range(pf_data.num_row_groups):
    n_rows_rg = pf_data.metadata.row_group(rg_i).num_rows
    rg_matched_idx = [idx - row_offset for idx in pass_indices if row_offset <= idx < row_offset + n_rows_rg]
    if rg_matched_idx:
        t_rg = time.time()
        rg_tab = pf_data.read_row_group(rg_i, columns=DET_COLS)
        rg_df = rg_tab.to_pandas()
        matched = rg_df.iloc[rg_matched_idx].copy()
        matched.index = [i + row_offset for i in rg_matched_idx]
        collected_dfs.append(matched)
        print(f"    RG {rg_i}/{pf_data.num_row_groups}: Extracted {len(matched)} passing events in {time.time()-t_rg:.2f}s")
    row_offset += n_rows_rg

df_data_det = pd.concat(collected_dfs)
df_data_passed = df_data_tr.loc[df_data_det.index].copy()
for col in DET_COLS:
    df_data_passed[col] = df_data_det[col]

print(f"    [+] Assembled {len(df_data_passed)} Data events with complete tracking & detector columns.")

# ==============================================================================
# 3. SAVE PARQUET TABLES
# ==============================================================================
print("\n[+] 4. Saving Parquet Files in:", OUT_DIR)

df_gal_all_passed.to_parquet(OUT_DIR / "gallery_all_passed.parquet", index=False)
df_gal_all_failed.to_parquet(OUT_DIR / "gallery_all_failed.parquet", index=False)
df_gal_6640_passed.to_parquet(OUT_DIR / "gallery_6640_passed.parquet", index=False)
df_gal_6640_failed.to_parquet(OUT_DIR / "gallery_6640_failed.parquet", index=False)
df_data_passed.to_parquet(OUT_DIR / "data_6640_passed.parquet", index=False)
df_mc_sig_passed.to_parquet(OUT_DIR / "trimuon_mc_signal_passed.parquet", index=False)
df_mc_nonsig_passed.to_parquet(OUT_DIR / "trimuon_mc_nonsignal_passed.parquet", index=False)
df_mc_all_passed.to_parquet(OUT_DIR / "trimuon_mc_all_passed.parquet", index=False)

print("    [+] Successfully saved all 8 Parquet tables.")

# ==============================================================================
# 4. COMPUTE PHYSICS VARIABLES FOR HISTOGRAMS & ROOT TREES
# ==============================================================================
proc_data_passed = compute_event_variables(df_data_passed, is_mc=False)
proc_mc_sig = compute_event_variables(df_mc_sig_passed, is_mc=True)
proc_mc_nonsig = compute_event_variables(df_mc_nonsig_passed, is_mc=True)
proc_mc_all = compute_event_variables(df_mc_all_passed, is_mc=True)
proc_gal_6640_pass = compute_event_variables(df_gal_6640_passed, is_mc=False)
proc_gal_6640_fail = compute_event_variables(df_gal_6640_failed, is_mc=False)
proc_gal_all_pass = compute_event_variables(df_gal_all_passed, is_mc=False)
proc_gal_all_fail = compute_event_variables(df_gal_all_failed, is_mc=False)

# ==============================================================================
# 5. BUILD ROOT FILE (TNTUPLE + TH1D HISTOGRAMS)
# ==============================================================================
print("\n[+] 5. Writing ROOT File:", OUTPUT_ROOT)

BIN_CONFIGS = {
    "hits_plane_sf": ("h_hits_plane_sf", "SciFi Hits/Plane;N_{hits}^{SF}/10;Events", 60, 0.0, 15.0),
    "qdc_plane_sf":  ("h_qdc_plane_sf",  "SciFi QDC/Plane;QDC^{SF}/10;Events", 80, 0.0, 40.0),
    "emip_sf":       ("h_emip_sf",       "SciFi MIP-Equivalence;#varepsilon_{mip}^{SF};Events", 70, 0.0, 14.0),
    "rhomip_sf":     ("h_rhomip_sf",     "SciFi MIP Charge Density;#rho_{mip}^{SF};Events", 60, 0.0, 3.0),
    "hits_plane_ds": ("h_hits_plane_ds", "DS Hits/Plane;N_{hits}^{DS}/4;Events", 50, 0.0, 20.0),
    "qdc_plane_ds":  ("h_qdc_plane_ds",  "DS QDC/Plane;QDC^{DS}/4;Events", 80, 0.0, 500.0),
    "emip_ds":       ("h_emip_ds",       "DS MIP-Equivalence;#varepsilon_{mip}^{DS};Events", 70, 0.0, 14.0),
    "rhomip_ds":     ("h_rhomip_ds",     "DS MIP Charge Density;#rho_{mip}^{DS};Events", 60, 0.0, 3.0),
}

def build_sample_histograms(df_proc: pd.DataFrame, prefix_name: str):
    hists = []
    w = df_proc["weight"].to_numpy()
    for col, (h_name, h_title, n_bins, x_min, x_max) in BIN_CONFIGS.items():
        h = ROOT.TH1D(f"{h_name}", f"{prefix_name} {h_title}", n_bins, x_min, x_max)
        h.Sumw2(True)
        vals = df_proc[col].to_numpy()
        for v, wt in zip(vals, w):
            h.Fill(v, wt)
        hists.append(h)
    return hists

f_out = ROOT.TFile(str(OUTPUT_ROOT), "RECREATE")

metadata_dict = {
    "name": "metadata",
    "w_bar_SF": w_bar_sf,
    "qdc_bar_SF": qdc_bar_sf,
    "w_bar_DS": w_bar_ds,
    "qdc_bar_DS": qdc_bar_ds,
    "n_events_data_passed": len(proc_data_passed),
    "n_events_mc_sig_passed": len(proc_mc_sig),
    "n_events_mc_nonsig_passed": len(proc_mc_nonsig),
    "n_events_mc_all_passed": len(proc_mc_all),
    "n_events_gal_6640_passed": len(proc_gal_6640_pass),
    "n_events_gal_6640_failed": len(proc_gal_6640_fail),
    "n_events_gal_all_passed": len(proc_gal_all_pass),
    "n_events_gal_all_failed": len(proc_gal_all_fail),
}
ddf_root.write_metadata(metadata_dict, f_out, directory="", close_file=False)

SAMPLES = [
    ("data_passed", "Data Run 6640 Passed", proc_data_passed),
    ("trimuon_mc_signal", "Trimuon MC Signal Passed", proc_mc_sig),
    ("trimuon_mc_nonsignal", "Trimuon MC Non-Signal Passed", proc_mc_nonsig),
    ("trimuon_mc_all", "Trimuon MC All Passed", proc_mc_all),
    ("gallery_6640_passed", "Gallery Run 6640 Passed", proc_gal_6640_pass),
    ("gallery_6640_failed", "Gallery Run 6640 Failed", proc_gal_6640_fail),
    ("gallery_all_passed", "Gallery All Runs Passed", proc_gal_all_pass),
    ("gallery_all_failed", "Gallery All Runs Failed", proc_gal_all_fail),
]

for dir_name, label, df_p in SAMPLES:
    f_out.mkdir(dir_name)
    f_out.cd(dir_name)
    
    hists = build_sample_histograms(df_p, label)
    for h in hists:
        h.Write("", ROOT.TObject.kOverwrite)
        
    ntuple = ROOT.TNtuple(
        "tree_events",
        f"{label} Event Variables",
        "hits_plane_sf:qdc_plane_sf:emip_sf:rhomip_sf:hits_plane_ds:qdc_plane_ds:emip_ds:rhomip_ds:weight:run_id"
    )
    for r in df_p.itertuples(index=False):
        ntuple.Fill(
            r.hits_plane_sf,
            r.qdc_plane_sf,
            r.emip_sf,
            r.rhomip_sf,
            r.hits_plane_ds,
            r.qdc_plane_ds,
            r.emip_ds,
            r.rhomip_ds,
            r.weight,
            r.run_id
        )
    ntuple.Write("", ROOT.TObject.kOverwrite)
    print(f"  [+] Saved directory: {dir_name}/ ({len(hists)} histograms + tree_events NTuple)")

f_out.Close()
print(f"\n[+] Total Pipeline Runtime: {time.time()-t_start:.1f}s")
print(f"[+] All artifacts successfully produced in: '{OUT_DIR}'")
