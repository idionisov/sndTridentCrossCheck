import sys
import os
import warnings
from pathlib import Path
import ctypes

warnings.filterwarnings("ignore")

sys.path.insert(0, "/afs/cern.ch/user/i/idioniso")

import ROOT
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import mplhep

import ddfUtils.root as dr
from ddfUtils.mpl_styling import add_label

plt.style.use("root")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

tfile = ROOT.TFile.Open("plots/passing_muon_selection.root")

def compute_metrics(h_data, h_mc, name, x_min=None, x_max=None):
    df_d = dr.to_pandas(h_data)
    df_m = dr.to_pandas(h_mc)
    x = df_d["x"].to_numpy()
    yd = df_d["y"].to_numpy()
    eyd = df_d["ey"].to_numpy()
    ym = df_m["y"].to_numpy()
    eym = df_m["ey"].to_numpy()
    
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None: mask &= (x >= x_min)
    if x_max is not None: mask &= (x <= x_max)
    
    x = x[mask]; yd = yd[mask]; eyd = eyd[mask]; ym = ym[mask]; eym = eym[mask]
    sum_d = np.sum(yd); sum_m = np.sum(ym)
    scale = sum_d / sum_m if sum_m > 0 else 1.0
    ym_s = ym * scale; eym_s = eym * scale
    
    active = (yd > 0) | (ym > 0)
    var = np.where(eyd**2 + eym_s**2 > 0, eyd**2 + eym_s**2, 1.0)
    pulls = (yd[active] - ym_s[active]) / np.sqrt(var[active])
    chi2 = np.sum(pulls**2)
    ndf = max(1, np.sum(active) - 1)
    
    cdf_d = np.cumsum(yd) / sum_d if sum_d > 0 else np.zeros_like(yd)
    cdf_m = np.cumsum(ym) / sum_m if sum_m > 0 else np.zeros_like(ym)
    d_ks = float(np.max(np.abs(cdf_d - cdf_m)))
    p_ks = float(h_data.KolmogorovTest(h_mc, ""))
    
    chi2_c = ctypes.c_double(0.0)
    ndf_c = ctypes.c_int(0)
    igood_c = ctypes.c_int(0)
    p_chi2_root = float(h_data.Chi2TestX(h_mc, chi2_c, ndf_c, igood_c, "UW"))
    
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    w1 = float(np.sum(np.abs(cdf_d - cdf_m)) * dx)
    
    return {
        "name": name, "n_data": int(sum_d), "yield_mc": sum_m, "scale": scale,
        "chi2_ndf": chi2 / ndf, "p_chi2": float(stats.chi2.sf(chi2, ndf)),
        "root_chi2_ndf": chi2_c.value / max(1, ndf_c.value), "p_chi2_root": p_chi2_root,
        "d_ks": d_ks, "p_ks": p_ks, "w1": w1,
        "peak_d": x[np.argmax(yd)], "peak_m": x[np.argmax(ym)],
        "x": x, "yd": yd, "eyd": eyd, "ym_s": ym_s, "eym_s": eym_s
    }

configs = [
    ("SciFi Target Tracker", "HitDistributions/SciFi_Hits/h_sf_hits_data_stage7", "HitDistributions/SciFi_Hits/h_sf_hits_mc_tot_stage7", 9, 36, "SciFi Hits", (8.5, 36.5), (0.1, 4e4)),
    ("Veto Scintillator", "HitDistributions/Veto_Hits/h_veto_hits_data_stage7", "HitDistributions/Veto_Hits/h_veto_hits_mc_tot_stage7", None, 15, "Veto Hits", (-0.5, 14.5), (0.1, 4e4)),
    ("Upstream MuFilter (US)", "HitDistributions/US_Hits/h_us_hits_data_stage7", "HitDistributions/US_Hits/h_us_hits_mc_tot_stage7", None, 18, "US System Hits", (-0.5, 16.5), (0.1, 4e4)),
    ("Downstream MuFilter (DS)", "HitDistributions/DS_Hits/h_ds_hits_data_stage7", "HitDistributions/DS_Hits/h_ds_hits_mc_tot_stage7", None, 25, "DS System Hits", (-0.5, 25.5), (0.1, 4e4))
]

fig = plt.figure(figsize=(16, 13.5))
subfigs = fig.subfigures(2, 2, hspace=0.08, wspace=0.10)

panel_letters = ["(a)", "(b)", "(c)", "(d)"]

for idx, (sf, (title, d_path, m_path, xmin, xmax, xlabel, xlim, ylim)) in enumerate(zip(subfigs.flat, configs)):
    hd = tfile.Get(d_path)
    hm = tfile.Get(m_path)
    m = compute_metrics(hd, hm, title, xmin, xmax)
    
    ax_top, ax_bot = sf.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.07})
    
    x = m["x"]; yd = m["yd"]; eyd = m["eyd"]; ym = m["ym_s"]; eym = m["eym_s"]
    n_data_val = m["n_data"]
    
    # Upper plot
    ax_top.step(np.append(x - 0.5, x[-1] + 0.5), np.append(ym, ym[-1]), where="post", color="#1f77b4", lw=2, label="PMU MC (Shape Norm)")
    ax_top.fill_between(x, ym - eym, ym + eym, step="mid", color="#1f77b4", alpha=0.25)
    ax_top.errorbar(x, yd, yerr=eyd, fmt="o", color="black", markersize=5, capsize=2, lw=1.3, label=f"Data ({n_data_val:,})")
    
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Events / Bin", fontsize=12)
    ax_top.set_ylim(ylim)
    ax_top.set_xlim(xlim)
    ax_top.grid(True, which="both", ls=":", alpha=0.5)
    
    # Stats badge
    stats_text = (
        f"{panel_letters[idx]} {m['name']}\n"
        f"Peak: Data {m['peak_d']:.0f} vs MC {m['peak_m']:.0f} hits\n"
        f"Wasserstein $W_1$: {m['w1']:.3f} hits\n"
        f"KS $D_{{\\mathrm{{KS}}}}$: {m['d_ks']:.4f} ($p = {m['p_ks']:.3f}$)\n"
        f"$\\chi^2 / \\mathrm{{ndf}}$: {m['chi2_ndf']:.2f}\n"
        f"Data / MC Area: {m['scale']:.3f}"
    )
    ax_top.text(0.48, 0.88, stats_text, transform=ax_top.transAxes, fontsize=9.8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
    
    add_label(ax=ax_top, mainText="SND@LHC", extraText="Preliminary", suppText=r"$1.83\times 10^{-4}\ \mathrm{fb}^{-1}$", com=13.6, lumi=None, fontsize=12)
    ax_top.legend(loc="upper left", framealpha=0.9, fontsize=10.5)
    
    # Lower plot
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ym > 0, yd / ym, np.nan)
        rel_yd = np.where(yd > 0, eyd / yd, 0.0)
        rel_ym = np.where(ym > 0, eym / ym, 0.0)
        ratio_err = ratio * np.sqrt(rel_yd**2 + rel_ym**2)
        
    ax_bot.axhline(1.0, color="red", ls="--", lw=1.3)
    ax_bot.axhspan(0.9, 1.1, color="gray", alpha=0.20)
    ax_bot.axhspan(0.8, 1.2, color="gray", alpha=0.10)
    ax_bot.errorbar(x, ratio, yerr=ratio_err, fmt="o", color="black", markersize=4, capsize=2, lw=1.2)
    ax_bot.set_ylabel("Data / MC", fontsize=11)
    ax_bot.set_xlabel(xlabel, fontsize=12)
    ax_bot.set_ylim(0.4, 1.6)
    ax_bot.grid(True, which="both", ls=":", alpha=0.5)

out_png = "/tmp/all_subdetectors_comparison_2x2.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Successfully generated {out_png}")
