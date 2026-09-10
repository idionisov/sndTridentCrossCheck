#!/usr/bin/env python3
"""
================================================================================
SciFi Tracker Unified Response & Specific Ionization vs Trident Energy Spectra
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

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Paths
CACHE_PGUN_SF = Path("/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/pgun_scifi_response.csv")
MC_PATH = Path("/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet")
OUTPUT_PNG = "scifi_unified_response_and_trident_spectra.png"
OUTPUT_ROOT = "scifi_unified_response_and_trident_spectra.root"

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


def check_nc(df: pd.DataFrame, prefix: str, proj: str, z_min: float, z_max: float, strict: bool = True) -> np.ndarray:
    n = df[f"n_lines_{prefix}_{proj}"].to_numpy()
    m1 = np.nan_to_num(df[f"{proj}_{prefix}_m1"].to_numpy())
    c1 = np.nan_to_num(df[f"{proj}_{prefix}_c1"].to_numpy())
    m2 = np.nan_to_num(df[f"{proj}_{prefix}_m2"].to_numpy())
    c2 = np.nan_to_num(df[f"{proj}_{prefix}_c2"].to_numpy())
    m3 = np.nan_to_num(df[f"{proj}_{prefix}_m3"].to_numpy())
    c3 = np.nan_to_num(df[f"{proj}_{prefix}_c3"].to_numpy())

    no_cross_12 = ((m1 * z_min + c1 - (m2 * z_min + c2)) * (m1 * z_max + c1 - (m2 * z_max + c2))) > 0
    no_cross_13 = ((m1 * z_min + c1 - (m3 * z_min + c3)) * (m1 * z_max + c1 - (m3 * z_max + c3))) > 0
    no_cross_23 = ((m2 * z_min + c2 - (m3 * z_min + c3)) * (m2 * z_max + c2 - (m3 * z_max + c3))) > 0

    pass_3 = (n >= 3) & no_cross_12 & no_cross_13 & no_cross_23
    pass_2 = (n == 2) & no_cross_12

    if strict:
        return pass_3 | pass_2
    return pass_3 | pass_2 | (n <= 1)


def apply_optimal_selection(df: pd.DataFrame) -> pd.DataFrame:
    nx = df["n_lines_sf_xz"].to_numpy()
    ny = df["n_lines_sf_yz"].to_numpy()
    nx_ds = df["n_lines_ds_xz"].to_numpy()
    ny_ds = df["n_lines_ds_yz"].to_numpy()

    sf_3_1 = ((nx >= 3) & (ny >= 1)) | ((ny >= 3) & (nx >= 1))
    ds_3_1 = ((nx_ds >= 3) & (ny_ds >= 1)) | ((ny_ds >= 3) & (nx_ds >= 1))

    coll_sf = check_nc(df, "sf", "xz", 260.0, 355.0, strict=True) | check_nc(df, "sf", "yz", 260.0, 355.0, strict=True)
    coll_ds = check_nc(df, "ds", "xz", 460.0, 590.0, strict=True) | check_nc(df, "ds", "yz", 460.0, 590.0, strict=True)

    return df[(sf_3_1 | ds_3_1) & coll_sf & coll_ds].copy()


def main():
    # 1. Load SciFi PGUN response
    df_pgun_sf = pd.read_csv(CACHE_PGUN_SF).sort_values(by="energy_gev")

    # Define MIP baseline standards for normalization
    N_MIP_PLANE = 2.55
    QDC_MIP_PLANE = 3.30

    # Unified MIP Multiplicity: E_MIP = 0.5 * (N_hits / N_MIP + QDC / QDC_MIP)
    df_pgun_sf["mean_mip_eq"] = 0.5 * (df_pgun_sf["mean_hits_plane"] / N_MIP_PLANE + df_pgun_sf["mean_qdc_plane"] / QDC_MIP_PLANE)
    df_pgun_sf["sem_mip_eq"] = 0.5 * np.sqrt((df_pgun_sf["sem_hits_plane"] / N_MIP_PLANE)**2 + (df_pgun_sf["sem_qdc_plane"] / QDC_MIP_PLANE)**2)
    df_pgun_sf["median_mip_eq"] = 0.5 * (df_pgun_sf["median_hits_plane"] / N_MIP_PLANE + df_pgun_sf["median_qdc_plane"] / QDC_MIP_PLANE)
    df_pgun_sf["q16_mip_eq"] = 0.5 * (df_pgun_sf["q16_hits_plane"] / N_MIP_PLANE + df_pgun_sf["q16_qdc_plane"] / QDC_MIP_PLANE)
    df_pgun_sf["q84_mip_eq"] = 0.5 * (df_pgun_sf["q84_hits_plane"] / N_MIP_PLANE + df_pgun_sf["q84_qdc_plane"] / QDC_MIP_PLANE)

    # 2. Load MC Signal Post-Selection
    mc_df = pq.read_table(MC_PATH, columns=TRACKING_COLS + KINEMATIC_COLS).to_pandas()
    sig_mc = mc_df[(mc_df["is_signal"] == 1) & (mc_df["region_type"] == 1)].copy()
    sel_mc = apply_optimal_selection(sig_mc)

    m_mu = 0.10566
    e1 = np.sqrt(sel_mc["p_mu_in"] ** 2 + m_mu ** 2).to_numpy()
    e2 = np.sqrt(sel_mc["p_mu_minus"] ** 2 + m_mu ** 2).to_numpy()
    e3 = np.sqrt(sel_mc["p_mu_plus"] ** 2 + m_mu ** 2).to_numpy()
    e_mat = np.sort(np.vstack([e1, e2, e3]).T, axis=1)

    e_soft, e_sublead, e_lead = e_mat[:, 0], e_mat[:, 1], e_mat[:, 2]
    w_mc = sel_mc["mc_weight"].to_numpy()

    # 3. Plotting
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
    bins_energy = np.logspace(1.0, 3.01, 35)

    # --------------------------------------------------------------------------
    # Panel A: Unified MIP-Equivalent Energy Metric (Hits + QDC)
    # --------------------------------------------------------------------------
    ax1_twin = ax1.twinx()
    ax1_twin.hist(e_soft, bins=bins_energy, weights=w_mc, density=True, histtype="stepfilled", facecolor="#9E9E9E", alpha=0.25, edgecolor="#616161", linestyle=":", linewidth=1.4, label=r"Muon 1 $E_{\mathrm{soft}}$")
    ax1_twin.hist(e_sublead, bins=bins_energy, weights=w_mc, density=True, histtype="step", color="#424242", linestyle="--", linewidth=1.6, label=r"Muon 2 $E_{\mathrm{sublead}}$")
    ax1_twin.hist(e_lead, bins=bins_energy, weights=w_mc, density=True, histtype="step", color="black", linestyle="-", linewidth=1.8, label=r"Parent muon $E_{\mathrm{lead}}$")
    ax1_twin.set_ylabel(r"Normalized Events [ $\mathrm{GeV}^{-1}$ ]", fontweight="bold", color="#212121")
    ax1_twin.set_ylim(0, ax1_twin.get_ylim()[1] * 1.35)

    ax1.fill_between(df_pgun_sf["energy_gev"], df_pgun_sf["q16_mip_eq"], df_pgun_sf["q84_mip_eq"], color="#2A9D8F", alpha=0.18, label=r"$\mu$ particle gun $68\%$ containment")
    ax1.plot(df_pgun_sf["energy_gev"], df_pgun_sf["median_mip_eq"], marker="s", markersize=3.5, markerfacecolor="none", markeredgecolor="#52B788", linestyle="--", color="#52B788", lw=1.2, label=r"Median MIP equivalent")
    ax1.errorbar(df_pgun_sf["energy_gev"], df_pgun_sf["mean_mip_eq"], yerr=df_pgun_sf["sem_mip_eq"], fmt="o-", color="#1B4332", ecolor="#1B4332", lw=2.2, markersize=5.0, capsize=2.5, label=r"Mean MIP equivalent $\langle \mathcal{E}_{\mathrm{MIP}} \rangle$")
    ax1.axhline(1.00, color="#E76F51", linestyle="--", lw=1.8, label=r"Single-MIP baseline ($\mathcal{E}_{\mathrm{MIP}} = 1.00$)")
    ax1.axhline(3.00, color="#D90429", linestyle=":", lw=1.8, label=r"Ideal $3\mu$ Trident standard ($\mathcal{E}_{\mathrm{MIP}} = 3.00$)")

    ax1.set_xscale("log")
    ax1.set_xlim(9, 1050)
    ax1.set_ylim(0.4, 5.2)
    ax1.set_xlabel(r"Muon Energy $E_{\mu}$ [ GeV ]", fontweight="bold")
    ax1.set_ylabel(r"Unified SciFi MIP Equivalents $\langle \mathcal{E}_{\mathrm{MIP}} \rangle$", fontweight="bold", color="#1B4332")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.92, title=r"Unified Response ($\mu$ pGun)")
    ax1_twin.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, framealpha=0.92, title=r"Rock Trident $3\mu$ Post-Cut Spectrum")
    ax1.set_title(r"$\mathbf{(a)}$ Unified SciFi MIP-Equivalent Multiplicity $\mathcal{E}_{\mathrm{MIP}}$", fontsize=11.5, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel B: Specific Ionization / Charge per Hit (QDC / N_hit)
    # --------------------------------------------------------------------------
    ax2_twin = ax2.twinx()
    ax2_twin.hist(e_soft, bins=bins_energy, weights=w_mc, density=True, histtype="stepfilled", facecolor="#9E9E9E", alpha=0.25, edgecolor="#616161", linestyle=":", linewidth=1.4, label=r"Muon 1 $E_{\mathrm{soft}}$")
    ax2_twin.hist(e_sublead, bins=bins_energy, weights=w_mc, density=True, histtype="step", color="#424242", linestyle="--", linewidth=1.6, label=r"Muon 2 $E_{\mathrm{sublead}}$")
    ax2_twin.hist(e_lead, bins=bins_energy, weights=w_mc, density=True, histtype="step", color="black", linestyle="-", linewidth=1.8, label=r"Parent muon $E_{\mathrm{lead}}$")
    ax2_twin.set_ylabel(r"Normalized Events [ $\mathrm{GeV}^{-1}$ ]", fontweight="bold", color="#212121")
    ax2_twin.set_ylim(0, ax2_twin.get_ylim()[1] * 1.35)

    ax2.errorbar(df_pgun_sf["energy_gev"], df_pgun_sf["mean_qdc_per_hit"], yerr=df_pgun_sf["sem_qdc_per_hit"], fmt="o-", color="#6A0572", ecolor="#6A0572", lw=2.2, markersize=5.0, capsize=2.5, label=r"Specific charge density $\langle \mathrm{QDC} / N_{\mathrm{hits}} \rangle$")
    ax2.axhline(1.17, color="#AB83A1", linestyle="--", lw=1.6, label=r"MIP fiber ionization ($\rho_{\mathrm{MIP}} = 1.17\ \mathrm{QDC/fiber}$)")

    ax2.set_xscale("log")
    ax2.set_xlim(9, 1050)
    ax2.set_ylim(1.05, 1.48)
    ax2.set_xlabel(r"Muon Energy $E_{\mu}$ [ GeV ]", fontweight="bold")
    ax2.set_ylabel(r"Specific Charge per Active Fiber $\langle \mathrm{QDC} / N_{\mathrm{hits}} \rangle$ [ QDC/hit ]", fontweight="bold", color="#6A0572")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.92, title=r"SciFi Specific Ionization ($\mu$ pGun)")
    ax2_twin.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, framealpha=0.92, title=r"Rock Trident $3\mu$ Post-Cut Spectrum")
    ax2.set_title(r"$\mathbf{(b)}$ Specific Ionization Density $\langle \mathrm{QDC} / N_{\mathrm{hits}} \rangle$", fontsize=11.5, fontweight="bold")

    fig.suptitle(
        "SND@LHC SciFi Tracker Unified Calibration & Post-Selection Trident Energies\nSelection: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) | Lumi = 0.991 fb^-1",
        fontsize=12.5,
        fontweight="bold",
        y=0.99
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved PNG plot to '{OUTPUT_PNG}'.")


if __name__ == "__main__":
    main()
