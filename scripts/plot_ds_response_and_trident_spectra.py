#!/usr/bin/env python3
"""
================================================================================
Downstream (DS) Detector Response & Post-Selection Trident Muon Energy Spectra
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
PGUN_DIR = Path("/eos/user/i/idioniso/1_Data/Monte_Carlo/pGun/muons")
CACHE_PGUN = Path("/afs/cern.ch/work/i/idioniso/sndMuTri/.cache/pgun_ds_response.csv")
MC_PATH = Path("/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet")
OUTPUT_PNG = "ds_response_and_trident_spectra_optimal.png"
OUTPUT_ROOT = "ds_response_and_trident_spectra_optimal.root"

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


def extract_pgun_ds_response():
    CACHE_PGUN.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PGUN.exists():
        print(f"[1/3] Loading Particle Gun DS response from cache: '{CACHE_PGUN}'...")
        return pd.read_csv(CACHE_PGUN).sort_values(by="energy_gev")

    print("[1/3] Scanning Particle Gun Energy Directories for Downstream (DS, System 3)...")
    t0 = time.time()
    results_ds = []

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

        n_entries = min(1500, tree.GetEntries())
        hit_counts_ds = []

        for i in range(n_entries):
            tree.GetEntry(i)
            n_ds = sum(1 for h in tree.Digi_MuFilterHits if h.GetSystem() == 3)
            if n_ds >= 2:
                hit_counts_ds.append(n_ds)

        f.Close()

        if len(hit_counts_ds) > 0:
            arr = np.array(hit_counts_ds)
            mean_val = float(np.mean(arr))
            median_val = float(np.median(arr))
            sem_val = float(np.std(arr) / np.sqrt(len(arr)))
            q16 = float(np.percentile(arr, 16))
            q84 = float(np.percentile(arr, 84))

            results_ds.append({
                "energy_gev": energy_val,
                "mean_total_hits": mean_val,
                "median_total_hits": median_val,
                "sem_total_hits": sem_val,
                "q16_total_hits": q16,
                "q84_total_hits": q84,
                "mean_per_plane": mean_val / 4.0,
                "sem_per_plane": sem_val / 4.0,
                "median_per_plane": median_val / 4.0,
                "q16_per_plane": q16 / 4.0,
                "q84_per_plane": q84 / 4.0,
                "n_events": len(arr),
            })

    df_pgun_ds = pd.DataFrame(results_ds).sort_values(by="energy_gev")
    df_pgun_ds.to_csv(CACHE_PGUN, index=False)
    print(f"  -> Extracted and cached {len(df_pgun_ds)} energy points in {time.time() - t0:.2f}s.")
    return df_pgun_ds


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


def plot_and_save_all(df_pgun_ds, e_soft, e_sublead, e_lead, w_mc):
    print("\n[3/3] Generating Publication-Quality Plot & ROOT File...")
    
    # --------------------------------------------------------------------------
    # Matplotlib High-Resolution Dual-Axis Plot
    # --------------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9.5,
    })

    fig, ax1 = plt.subplots(figsize=(8.5, 6.0), dpi=300)
    bins_energy = np.logspace(1.0, 3.01, 35)  # 10 GeV to ~1010 GeV

    # 1. RIGHT AXIS (Background): Post-Selection Trident Muon Energy Hierarchy
    ax1_twin = ax1.twinx()

    # Softest Muon
    ax1_twin.hist(
        e_soft,
        bins=bins_energy,
        weights=w_mc,
        density=True,
        histtype="stepfilled",
        facecolor="#9E9E9E",
        alpha=0.25,
        edgecolor="#616161",
        linestyle=":",
        linewidth=1.4,
        zorder=1,
        label=r"Muon 1 $E_{\mathrm{soft}}$",
    )

    # Sub-Leading Muon
    ax1_twin.hist(
        e_sublead,
        bins=bins_energy,
        weights=w_mc,
        density=True,
        histtype="step",
        color="#424242",
        linestyle="--",
        linewidth=1.6,
        zorder=2,
        label=r"Muon 2 $E_{\mathrm{sublead}}$",
    )

    # Leading Muon
    ax1_twin.hist(
        e_lead,
        bins=bins_energy,
        weights=w_mc,
        density=True,
        histtype="step",
        color="black",
        linestyle="-",
        linewidth=1.8,
        zorder=3,
        label=r"Parent muon $E_{\mathrm{lead}}$",
    )

    ax1_twin.set_ylabel(
        r"Normalized Events [ $\mathrm{GeV}^{-1}$ ]",
        fontweight="bold",
        fontsize=11.5,
        color="#212121",
    )
    ax1_twin.tick_params(axis="y", labelcolor="#212121")
    ax1_twin.set_ylim(0, ax1_twin.get_ylim()[1] * 1.35)

    # 2. LEFT AXIS (Foreground): Particle Gun DS Detector Response
    # 68% containment interval band
    ax1.fill_between(
        df_pgun_ds["energy_gev"],
        df_pgun_ds["q16_per_plane"],
        df_pgun_ds["q84_per_plane"],
        color="#E53935",
        alpha=0.15,
        zorder=4,
        label=r"$\mu$ particle gun $68\%$ containment",
    )

    # Median response
    ax1.plot(
        df_pgun_ds["energy_gev"],
        df_pgun_ds["median_per_plane"],
        marker="s",
        markersize=3.8,
        markerfacecolor="none",
        markeredgecolor="#E57373",
        linestyle="--",
        color="#E57373",
        lw=1.3,
        alpha=0.85,
        zorder=5,
        label=r"Median channels / plane",
    )

    # Mean cluster size per plane
    ax1.errorbar(
        df_pgun_ds["energy_gev"],
        df_pgun_ds["mean_per_plane"],
        yerr=df_pgun_ds["sem_per_plane"],
        fmt="o-",
        color="#D32F2F",
        ecolor="#D32F2F",
        lw=2.2,
        markersize=5.5,
        capsize=2.5,
        zorder=7,
        label=r"Mean activated channels $\langle N_{\mathrm{hits}}^{\mathrm{DS}} / 4 \rangle$",
    )

    # DS MIP Plateau reference line (bar width ~ 1 cm -> w_DS ~ 2.10)
    ax1.axhline(
        2.10,
        color="#1976D2",
        linestyle="--",
        lw=1.6,
        zorder=6,
        label=r"DS MIP baseline ($\bar{w}_{\mathrm{DS}} = 2.10$)",
    )

    ax1.set_xscale("log")
    ax1.set_xlim(9, 1050)
    ax1.set_ylim(1.0, 6.5)

    ax1.set_xlabel(r"Muon Energy $E_{\mu}$ [ GeV ]", fontweight="bold")
    ax1.set_ylabel(
        r"Activated DS Channels per Plane $\langle N_{\mathrm{hits}}^{\mathrm{DS}} / 4 \rangle$",
        fontweight="bold",
        fontsize=11.5,
        color="#D32F2F",
    )
    ax1.tick_params(axis="y", labelcolor="#D32F2F")

    # Minor tick formatters
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto"))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1_twin.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax1.tick_params(which="both", direction="in", top=False, right=False, length=5, width=1.0)
    ax1_twin.tick_params(which="both", direction="in", top=False, right=True, length=5, width=1.0)
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Legends
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.92,
        edgecolor="lightgray",
        title=r"DS Detector Response ($\mu$ pGun)",
        title_fontsize=9.5,
    )

    ax1_twin.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.92,
        edgecolor="lightgray",
        title=r"Rock Trident $3\mu$ Post-Cut Spectrum",
        title_fontsize=9.5,
    )

    ax1.set_title(
        "SND@LHC Downstream Response & Post-Selection Trident Energies\nSelection: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) | Lumi = 0.991 fb^-1",
        fontsize=11.5,
        fontweight="bold",
        pad=12
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved PNG plot to '{OUTPUT_PNG}'.")

    # --------------------------------------------------------------------------
    # ROOT File Output (TGraphErrors, TH1D, TCanvas)
    # --------------------------------------------------------------------------
    f_root = ROOT.TFile(OUTPUT_ROOT, "RECREATE")

    # 1. TGraphErrors for Particle Gun Response
    n_pts = len(df_pgun_ds)
    x_e = np.ascontiguousarray(df_pgun_ds["energy_gev"], dtype=np.float64)
    y_mean = np.ascontiguousarray(df_pgun_ds["mean_per_plane"], dtype=np.float64)
    ex = np.zeros(n_pts, dtype=np.float64)
    ey = np.ascontiguousarray(df_pgun_ds["sem_per_plane"], dtype=np.float64)

    gr_mean = ROOT.TGraphErrors(n_pts, x_e, y_mean, ex, ey)
    gr_mean.SetName("gr_ds_mean_hits_per_plane")
    gr_mean.SetTitle("DS Mean Activated Channels per Plane;Muon Energy [GeV];<N_{hits}^{DS} / 4>")
    gr_mean.SetMarkerStyle(20)
    gr_mean.SetMarkerColor(ROOT.kRed + 1)
    gr_mean.SetLineColor(ROOT.kRed + 1)
    gr_mean.SetLineWidth(2)
    gr_mean.Write()

    y_med = np.ascontiguousarray(df_pgun_ds["median_per_plane"], dtype=np.float64)
    gr_med = ROOT.TGraphErrors(n_pts, x_e, y_med, ex, ex)
    gr_med.SetName("gr_ds_median_hits_per_plane")
    gr_med.SetTitle("DS Median Channels per Plane;Muon Energy [GeV];Median <N_{hits}^{DS} / 4>")
    gr_med.SetMarkerStyle(21)
    gr_med.SetMarkerColor(ROOT.kRed - 4)
    gr_med.SetLineColor(ROOT.kRed - 4)
    gr_med.SetLineStyle(2)
    gr_med.Write()

    # 2. TH1D for Trident Muon Energy Spectra
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

    # 3. Canvas
    c = ROOT.TCanvas("c_ds_response_trident_energy", "DS Response and Trident Muon Spectra", 900, 700)
    c.SetLogx(1)
    gr_mean.Draw("APL")
    gr_med.Draw("PL SAME")
    c.Write()

    f_root.Close()
    print(f"[+] Saved ROOT file to '{OUTPUT_ROOT}'.")


if __name__ == "__main__":
    df_pgun = extract_pgun_ds_response()
    e_soft, e_sublead, e_lead, w_mc = extract_post_selection_energies()
    plot_and_save_all(df_pgun, e_soft, e_sublead, e_lead, w_mc)
