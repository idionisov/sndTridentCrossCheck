#!/usr/bin/env python3
"""
================================================================================
SciFi Tracker Detector Response (Hits & QDC) & Post-Selection Trident Muon Spectra
================================================================================
Selection applied: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)
================================================================================
"""

from pathlib import Path
import os
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ROOT
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Paths
PGUN_DIR = Path("/eos/user/i/idioniso/1_Data/Monte_Carlo/pGun/muons")
CACHE_PGUN_SF = Path("/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/pgun_scifi_response.csv")
MC_PATH = Path("/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet")
OUTPUT_PNG = "scifi_response_and_trident_spectra_optimal.png"
OUTPUT_ROOT = "scifi_response_and_trident_spectra_optimal.root"

ENERGY_DIRS = [
    "10.GeV",
    "12.5.GeV",
    "15.GeV",
    "17.5.GeV",
    "20.GeV",
    "25.GeV",
    "30.GeV",
    "35.GeV",
    "55.GeV",
    "100.GeV",
    "200.GeV",
    "300.GeV",
    "450.GeV",
    "600.GeV",
    "800.GeV",
    "1010.GeV",
]

TRACKING_COLS = [
    "n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz",
    "xz_sf_m1", "xz_sf_c1", "yz_sf_m1", "yz_sf_c1",
    "xz_sf_m2", "xz_sf_c2", "yz_sf_m2", "yz_sf_c2",
    "xz_sf_m3", "xz_sf_c3", "yz_sf_m3", "yz_sf_c3",
    "xz_ds_m1", "xz_ds_c1", "yz_ds_m1", "yz_ds_c1",
    "xz_ds_m2", "xz_ds_c2", "yz_ds_m2", "yz_ds_c2",
    "xz_ds_m3", "xz_ds_c3", "yz_ds_m3", "yz_ds_c3"
]

KINEMATIC_COLS = ["is_signal", "region_type", "mc_weight", "p_mu_in", "p_mu_minus", "p_mu_plus"]


def extract_pgun_scifi_response():
    CACHE_PGUN_SF.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PGUN_SF.exists():
        print(f"[1/3] Loading Particle Gun SciFi response from cache: '{CACHE_PGUN_SF}'...")
        return pd.read_csv(CACHE_PGUN_SF).sort_values(by="energy_gev")

    print("[1/3] Scanning Particle Gun Energy Directories for SciFi Tracker (Hits & QDC)...")
    t0 = time.time()
    results_sf = []

    for edir in ENERGY_DIRS:
        dir_path = PGUN_DIR / edir
        digi_file = dir_path / "sndLHC.PG_13-TGeant4_digCPP.root"
        if not digi_file.exists():
            continue

        energy_val = float(edir.replace(".GeV", "").replace("GeV", ""))
        f = ROOT.TFile.Open(str(digi_file))
        if not f or f.IsZombie():
            continue

        tree = f.Get("cbmsim")
        if not tree:
            f.Close()
            continue

        n_entries = min(1000, tree.GetEntries())
        hit_counts_sf = []
        qdc_totals_sf = []
        qdc_per_hit_sf = []

        for i in range(n_entries):
            tree.GetEntry(i)
            valid_signals = [h.GetSignal() for h in tree.Digi_ScifiHits if h.isValid()]
            n_sf = len(valid_signals)
            if n_sf >= 5:  # Muon traversed SciFi tracker
                total_qdc = sum(valid_signals)
                hit_counts_sf.append(n_sf)
                qdc_totals_sf.append(total_qdc)
                qdc_per_hit_sf.append(total_qdc / n_sf)

        f.Close()

        if len(hit_counts_sf) > 0:
            arr_hits = np.array(hit_counts_sf)
            arr_qdc = np.array(qdc_totals_sf)
            arr_qph = np.array(qdc_per_hit_sf)

            # 10 SciFi planes (5 stations x 2 views XZ & YZ)
            results_sf.append({
                "energy_gev": energy_val,
                # Channels per plane
                "mean_hits_plane": float(np.mean(arr_hits)) / 10.0,
                "sem_hits_plane": float(np.std(arr_hits) / np.sqrt(len(arr_hits))) / 10.0,
                "median_hits_plane": float(np.median(arr_hits)) / 10.0,
                "q16_hits_plane": float(np.percentile(arr_hits, 16)) / 10.0,
                "q84_hits_plane": float(np.percentile(arr_hits, 84)) / 10.0,
                # Total QDC per plane
                "mean_qdc_plane": float(np.mean(arr_qdc)) / 10.0,
                "sem_qdc_plane": float(np.std(arr_qdc) / np.sqrt(len(arr_qdc))) / 10.0,
                "median_qdc_plane": float(np.median(arr_qdc)) / 10.0,
                "q16_qdc_plane": float(np.percentile(arr_qdc, 16)) / 10.0,
                "q84_qdc_plane": float(np.percentile(arr_qdc, 84)) / 10.0,
                # Specific QDC per hit
                "mean_qdc_per_hit": float(np.mean(arr_qph)),
                "sem_qdc_per_hit": float(np.std(arr_qph) / np.sqrt(len(arr_qph))),
                "n_events": len(arr_hits),
            })

    df_pgun_sf = pd.DataFrame(results_sf).sort_values(by="energy_gev")
    df_pgun_sf.to_csv(CACHE_PGUN_SF, index=False)
    print(f"  -> Extracted and cached {len(df_pgun_sf)} energy points in {time.time() - t0:.2f}s.")
    return df_pgun_sf


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


def apply_optimal_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Applies [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) selection."""
    nx = df["n_lines_sf_xz"].to_numpy()
    ny = df["n_lines_sf_yz"].to_numpy()
    nx_ds = df["n_lines_ds_xz"].to_numpy()
    ny_ds = df["n_lines_ds_yz"].to_numpy()

    sf_3_1 = ((nx >= 3) & (ny >= 1)) | ((ny >= 3) & (nx >= 1))
    ds_3_1 = ((nx_ds >= 3) & (ny_ds >= 1)) | ((ny_ds >= 3) & (nx_ds >= 1))
    mult_mask = sf_3_1 | ds_3_1

    coll_sf = check_nc(df, "sf", "xz", 260.0, 355.0, strict=True) | check_nc(df, "sf", "yz", 260.0, 355.0, strict=True)
    coll_ds = check_nc(df, "ds", "xz", 460.0, 590.0, strict=True) | check_nc(df, "ds", "yz", 460.0, 590.0, strict=True)

    return df[mult_mask & coll_sf & coll_ds].copy()


def extract_post_selection_energies():
    print("\n[2/3] Extracting Post-Selection Trident Muon Energies from MC...")
    t0 = time.time()
    mc_df = pq.read_table(MC_PATH, columns=TRACKING_COLS + KINEMATIC_COLS).to_pandas()
    sig_mc = mc_df[(mc_df["is_signal"] == 1) & (mc_df["region_type"] == 1)].copy()

    sel_mc = apply_optimal_selection(sig_mc)
    print(f"  -> Rock Trident Signal Surviving Selection: {len(sel_mc):,} / {len(sig_mc):,} ({100*len(sel_mc)/len(sig_mc):.2f}%) in {time.time() - t0:.2f}s.")

    m_mu = 0.10566
    e1 = np.sqrt(sel_mc["p_mu_in"] ** 2 + m_mu ** 2).to_numpy()
    e2 = np.sqrt(sel_mc["p_mu_minus"] ** 2 + m_mu ** 2).to_numpy()
    e3 = np.sqrt(sel_mc["p_mu_plus"] ** 2 + m_mu ** 2).to_numpy()
    e_mat = np.sort(np.vstack([e1, e2, e3]).T, axis=1)

    e_soft = e_mat[:, 0]
    e_sublead = e_mat[:, 1]
    e_lead = e_mat[:, 2]
    w_mc = sel_mc["mc_weight"].to_numpy()

    return e_soft, e_sublead, e_lead, w_mc


def plot_and_save_all(df_pgun_sf, e_soft, e_sublead, e_lead, w_mc):
    print("\n[3/3] Generating Publication-Quality SciFi Detector Response & Energy Spectra Plot...")

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.0,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.2), dpi=300)
    bins_energy = np.logspace(1.0, 3.01, 35)  # 10 GeV to ~1010 GeV

    # ==========================================================================
    # PANEL A: SciFi Activated Channels / Plane <N_hits^SF / 10>
    # ==========================================================================
    ax1_twin = ax1.twinx()

    # Trident Histograms (Right Axis)
    ax1_twin.hist(e_soft, bins=bins_energy, weights=w_mc, density=True, histtype="stepfilled", facecolor="#9E9E9E", alpha=0.25, edgecolor="#616161", linestyle=":", linewidth=1.4, label=r"Muon 1 $E_{\mathrm{soft}}$")
    ax1_twin.hist(e_sublead, bins=bins_energy, weights=w_mc, density=True, histtype="step", color="#424242", linestyle="--", linewidth=1.6, label=r"Muon 2 $E_{\mathrm{sublead}}$")
    ax1_twin.hist(e_lead, bins=bins_energy, weights=w_mc, density=True, histtype="step", color="black", linestyle="-", linewidth=1.8, label=r"Parent muon $E_{\mathrm{lead}}$")

    ax1_twin.set_ylabel(r"Normalized Events [ $\mathrm{GeV}^{-1}$ ]", fontweight="bold", color="#212121")
    ax1_twin.set_ylim(0, ax1_twin.get_ylim()[1] * 1.35)

    # SciFi Hits Response (Left Axis)
    ax1.fill_between(df_pgun_sf["energy_gev"], df_pgun_sf["q16_hits_plane"], df_pgun_sf["q84_hits_plane"], color="#1976D2", alpha=0.15, label=r"$\mu$ particle gun $68\%$ containment")
    ax1.plot(df_pgun_sf["energy_gev"], df_pgun_sf["median_hits_plane"], marker="s", markersize=3.5, markerfacecolor="none", markeredgecolor="#64B5F6", linestyle="--", color="#64B5F6", lw=1.2, label=r"Median channels / plane")
    ax1.errorbar(df_pgun_sf["energy_gev"], df_pgun_sf["mean_hits_plane"], yerr=df_pgun_sf["sem_hits_plane"], fmt="o-", color="#1565C0", ecolor="#1565C0", lw=2.2, markersize=5.0, capsize=2.5, label=r"Mean activated channels $\langle N_{\mathrm{hits}}^{\mathrm{SF}} / 10 \rangle$")
    ax1.axhline(2.55, color="#388E3C", linestyle="--", lw=1.6, label=r"SciFi MIP baseline ($\bar{w}_{\mathrm{SF}} = 2.55$)")

    ax1.set_xscale("log")
    ax1.set_xlim(9, 1050)
    ax1.set_ylim(1.0, 7.5)
    ax1.set_xlabel(r"Muon Energy $E_{\mu}$ [ GeV ]", fontweight="bold")
    ax1.set_ylabel(r"Activated SciFi Channels per Plane $\langle N_{\mathrm{hits}}^{\mathrm{SF}} / 10 \rangle$", fontweight="bold", color="#1565C0")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    ax1.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.92, title=r"SciFi Cluster Size ($\mu$ pGun)")
    ax1_twin.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, framealpha=0.92, title=r"Rock Trident $3\mu$ Post-Cut Spectrum")
    ax1.set_title(r"$\mathbf{(a)}$ SciFi Channel Multiplicity Response", fontsize=11.5, fontweight="bold")

    # ==========================================================================
    # PANEL B: SciFi Calibrated QDC Energy Deposit / Plane <QDC^SF / 10>
    # ==========================================================================
    ax2_twin = ax2.twinx()

    # Trident Histograms (Right Axis)
    ax2_twin.hist(e_soft, bins=bins_energy, weights=w_mc, density=True, histtype="stepfilled", facecolor="#9E9E9E", alpha=0.25, edgecolor="#616161", linestyle=":", linewidth=1.4, label=r"Muon 1 $E_{\mathrm{soft}}$")
    ax2_twin.hist(e_sublead, bins=bins_energy, weights=w_mc, density=True, histtype="step", color="#424242", linestyle="--", linewidth=1.6, label=r"Muon 2 $E_{\mathrm{sublead}}$")
    ax2_twin.hist(e_lead, bins=bins_energy, weights=w_mc, density=True, histtype="step", color="black", linestyle="-", linewidth=1.8, label=r"Parent muon $E_{\mathrm{lead}}$")

    ax2_twin.set_ylabel(r"Normalized Events [ $\mathrm{GeV}^{-1}$ ]", fontweight="bold", color="#212121")
    ax2_twin.set_ylim(0, ax2_twin.get_ylim()[1] * 1.35)

    # SciFi QDC Response (Left Axis)
    ax2.fill_between(df_pgun_sf["energy_gev"], df_pgun_sf["q16_qdc_plane"], df_pgun_sf["q84_qdc_plane"], color="#D32F2F", alpha=0.15, label=r"$\mu$ particle gun $68\%$ containment")
    ax2.plot(df_pgun_sf["energy_gev"], df_pgun_sf["median_qdc_plane"], marker="s", markersize=3.5, markerfacecolor="none", markeredgecolor="#E57373", linestyle="--", color="#E57373", lw=1.2, label=r"Median QDC / plane")
    ax2.errorbar(df_pgun_sf["energy_gev"], df_pgun_sf["mean_qdc_plane"], yerr=df_pgun_sf["sem_qdc_plane"], fmt="o-", color="#C62828", ecolor="#C62828", lw=2.2, markersize=5.0, capsize=2.5, label=r"Mean total QDC $\langle \mathrm{QDC}^{\mathrm{SF}} / 10 \rangle$")
    ax2.axhline(3.30, color="#7B1FA2", linestyle="--", lw=1.6, label=r"SciFi MIP QDC baseline ($\overline{\mathrm{QDC}}_{\mathrm{SF}} = 3.30$)")

    ax2.set_xscale("log")
    ax2.set_xlim(9, 1050)
    ax2.set_ylim(1.0, 12.0)
    ax2.set_xlabel(r"Muon Energy $E_{\mu}$ [ GeV ]", fontweight="bold")
    ax2.set_ylabel(r"Calibrated SciFi QDC per Plane $\langle \mathrm{QDC}^{\mathrm{SF}} / 10 \rangle$", fontweight="bold", color="#C62828")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    ax2.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.92, title=r"SciFi Energy Deposit ($\mu$ pGun)")
    ax2_twin.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, framealpha=0.92, title=r"Rock Trident $3\mu$ Post-Cut Spectrum")
    ax2.set_title(r"$\mathbf{(b)}$ SciFi Calibrated QDC Response", fontsize=11.5, fontweight="bold")

    fig.suptitle(
        "SND@LHC SciFi Tracker Response (Hits & QDC) & Post-Selection Trident Energies\nSelection: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) | Lumi = 0.991 fb^-1",
        fontsize=12.5,
        fontweight="bold",
        y=0.99
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved PNG plot to '{OUTPUT_PNG}'.")

    # --------------------------------------------------------------------------
    # ROOT File Output (TGraphErrors, TH1D, TCanvas)
    # --------------------------------------------------------------------------
    f_root = ROOT.TFile(OUTPUT_ROOT, "RECREATE")

    n_pts = len(df_pgun_sf)
    x_e = np.ascontiguousarray(df_pgun_sf["energy_gev"], dtype=np.float64)
    ex = np.zeros(n_pts, dtype=np.float64)

    # 1. TGraphErrors for Hits
    y_mean_hits = np.ascontiguousarray(df_pgun_sf["mean_hits_plane"], dtype=np.float64)
    ey_hits = np.ascontiguousarray(df_pgun_sf["sem_hits_plane"], dtype=np.float64)
    gr_hits = ROOT.TGraphErrors(n_pts, x_e, y_mean_hits, ex, ey_hits)
    gr_hits.SetName("gr_scifi_mean_hits_per_plane")
    gr_hits.SetTitle("SciFi Mean Activated Channels per Plane;Muon Energy [GeV];<N_{hits}^{SF} / 10>")
    gr_hits.SetMarkerStyle(20)
    gr_hits.SetMarkerColor(ROOT.kBlue + 1)
    gr_hits.SetLineColor(ROOT.kBlue + 1)
    gr_hits.SetLineWidth(2)
    gr_hits.Write()

    # 2. TGraphErrors for QDC
    y_mean_qdc = np.ascontiguousarray(df_pgun_sf["mean_qdc_plane"], dtype=np.float64)
    ey_qdc = np.ascontiguousarray(df_pgun_sf["sem_qdc_plane"], dtype=np.float64)
    gr_qdc = ROOT.TGraphErrors(n_pts, x_e, y_mean_qdc, ex, ey_qdc)
    gr_qdc.SetName("gr_scifi_mean_qdc_per_plane")
    gr_qdc.SetTitle("SciFi Mean Total QDC per Plane;Muon Energy [GeV];<QDC^{SF} / 10>")
    gr_qdc.SetMarkerStyle(21)
    gr_qdc.SetMarkerColor(ROOT.kRed + 1)
    gr_qdc.SetLineColor(ROOT.kRed + 1)
    gr_qdc.SetLineWidth(2)
    gr_qdc.Write()

    # 3. TH1D for Trident Muon Energy Spectra
    h_e_soft = ROOT.TH1D("h_e_soft", "Softest Trident Muon Energy E_{soft};Muon Energy [GeV];Events / Bin", 35, 10.0, 1010.0)
    h_e_sublead = ROOT.TH1D("h_e_sublead", "Subleading Trident Muon Energy E_{sublead};Muon Energy [GeV];Events / Bin", 35, 10.0, 1010.0)
    h_e_lead = ROOT.TH1D("h_e_lead", "Leading Trident Muon Energy E_{lead};Muon Energy [GeV];Events / Bin", 35, 10.0, 1010.0)

    for val, w in zip(e_soft, w_mc):
        h_e_soft.Fill(val, w)
    for val, w in zip(e_sublead, w_mc):
        h_e_sublead.Fill(val, w)
    for val, w in zip(e_lead, w_mc):
        h_e_lead.Fill(val, w)

    h_e_soft.Write()
    h_e_sublead.Write()
    h_e_lead.Write()

    # 4. Canvas
    c = ROOT.TCanvas("c_scifi_response_trident_energy", "SciFi Hits and QDC Response vs Trident Energy", 1000, 700)
    c.SetLogx(1)
    gr_hits.Draw("APL")
    gr_qdc.Draw("PL SAME")
    c.Write()

    f_root.Close()
    print(f"[+] Saved ROOT file to '{OUTPUT_ROOT}'.")


if __name__ == "__main__":
    df_pgun_sf = extract_pgun_scifi_response()
    e_soft, e_sublead, e_lead, w_mc = extract_post_selection_energies()
    plot_and_save_all(df_pgun_sf, e_soft, e_sublead, e_lead, w_mc)
