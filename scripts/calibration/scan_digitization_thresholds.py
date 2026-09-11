#!/usr/bin/env python3
"""
================================================================================
SND@LHC Digitization Threshold Grid Scanner & Optimization Engine
================================================================================
Orchestrates systematic digitization scans across a user-defined grid of
photo-electron thresholds on Monte Carlo simulations, executes track
reconstruction and calibration histogram extraction, and scores emulation
fidelity against real collision data using quantitative statistical metrics.

Metrics Computed:
1. Wasserstein Distance (W1 / Earth Mover's Distance in hits)
2. Peak / Mode Alignment (Delta Peak = |Peak_MC - Peak_Data|)
3. Mean Multiplicity Discrepancy (Delta mu = |mu_MC - mu_Data|)
4. Kolmogorov-Smirnov Distance (D_KS and p-value)
5. Normalized Shape Chi-square (chi2 / ndf)

Author: SND@LHC Collaboration
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add repository root to sys.path
_curr = os.path.abspath(os.path.dirname(__file__))
_repo_root = _curr
while _curr != "/" and _curr != os.path.dirname(_curr):
    if os.path.exists(os.path.join(_curr, "analysis")) and os.path.exists(os.path.join(_curr, "snd")):
        _repo_root = _curr
        break
    _curr = os.path.dirname(_curr)

if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Default environment configuration
DEFAULT_SND_HOME = os.environ.get("SNDSW_ROOT", "/home/idioniso/snd_master/sndsw")
if not os.path.exists(DEFAULT_SND_HOME):
    DEFAULT_SND_HOME = "/afs/cern.ch/user/i/idioniso/snd_master/sndsw"

DEFAULT_RAW_SIM = "/eos/experiment/sndlhc/MonteCarlo/from_sndprodmachine/run200/sndLHC.Ntuple-TGeant4_boost100.0.root"
DEFAULT_GEOFILE = "/eos/experiment/sndlhc/MonteCarlo/from_sndprodmachine/run200/geofile_full.Ntuple-TGeant4_boost100.0.root"
DEFAULT_DATA_REF = "plots/passing_muon_selection.root"
DEFAULT_DATA_HIST = "HitDistributions/SciFi_Hits/h_sf_hits_data_stage6"


def compute_hist_metrics(h_data: ROOT.TH1D, h_mc: ROOT.TH1D) -> Dict[str, float]:
    """
    Computes statistical similarity metrics between Data and MC TH1 histograms.
    """
    # 1. Peak / Mode
    peak_data = h_data.GetBinCenter(h_data.GetMaximumBin())
    peak_mc = h_mc.GetBinCenter(h_mc.GetMaximumBin())
    delta_peak = abs(peak_mc - peak_data)

    # 2. Mean and RMS
    mean_data = h_data.GetMean()
    mean_mc = h_mc.GetMean()
    delta_mean = abs(mean_mc - mean_data)

    rms_data = h_data.GetRMS()
    rms_mc = h_mc.GetRMS()

    # 3. Bin arrays & normalization
    nbins = h_data.GetNbinsX()
    x_edges = np.array([h_data.GetBinLowEdge(i) for i in range(1, nbins + 2)])
    dx = h_data.GetBinWidth(1)

    counts_d = np.array([h_data.GetBinContent(i) for i in range(1, nbins + 1)])
    counts_m = np.array([h_mc.GetBinContent(i) for i in range(1, nbins + 1)])

    sum_d = np.sum(counts_d)
    sum_m = np.sum(counts_m)

    if sum_d == 0 or sum_m == 0:
        return {
            "peak_data": peak_data, "peak_mc": peak_mc, "delta_peak": delta_peak,
            "mean_data": mean_data, "mean_mc": mean_mc, "delta_mean": delta_mean,
            "rms_data": rms_data, "rms_mc": rms_mc,
            "w1": float("inf"), "ks_stat": 1.0, "ks_pval": 0.0,
            "chi2": float("inf"), "ndf": 1, "chi2_ndf": float("inf")
        }

    pdf_d = counts_d / sum_d
    pdf_m = counts_m / sum_m

    # 4. Wasserstein Distance (W1 / EMD)
    cdf_d = np.cumsum(pdf_d)
    cdf_m = np.cumsum(pdf_m)
    w1 = float(np.sum(np.abs(cdf_d - cdf_m)) * dx)

    # 5. Kolmogorov-Smirnov Statistic
    ks_stat = float(np.max(np.abs(cdf_d - cdf_m)))
    n_eff = (sum_d * sum_m) / (sum_d + sum_m)
    ks_pval = float(stats.kstwo.sf(ks_stat, int(round(n_eff)))) if n_eff > 1 else 0.0

    # 6. Shape Chi-square
    alpha = sum_d / sum_m
    err_d = np.array([h_data.GetBinError(i) for i in range(1, nbins + 1)])
    err_m = np.array([h_mc.GetBinError(i) for i in range(1, nbins + 1)])

    chi2 = 0.0
    ndf = 0
    for i in range(nbins):
        if counts_d[i] > 0 or counts_m[i] > 0:
            var = err_d[i]**2 + (alpha * err_m[i])**2
            if var > 0:
                diff = counts_d[i] - alpha * counts_m[i]
                chi2 += (diff**2) / var
                ndf += 1

    chi2_ndf = (chi2 / (ndf - 1)) if ndf > 1 else chi2

    return {
        "peak_data": peak_data,
        "peak_mc": peak_mc,
        "delta_peak": delta_peak,
        "mean_data": mean_data,
        "mean_mc": mean_mc,
        "delta_mean": delta_mean,
        "rms_data": rms_data,
        "rms_mc": rms_mc,
        "w1": w1,
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "chi2": float(chi2),
        "ndf": max(1, ndf - 1),
        "chi2_ndf": float(chi2_ndf)
    }


def run_command(cmd: str, desc: str = "") -> bool:
    """Executes a bash shell command with real-time stdout streaming."""
    if desc:
        print(f"[*] {desc}")
    print(f"    --> {cmd}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[-] Command failed with exit code {res.returncode} ({dt:.1f} s)")
        return False
    print(f"[✓] Step completed successfully in {dt:.1f} s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="SND@LHC Digitization Threshold Grid Scanner & Optimization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input & Output Specifications
    parser.add_argument("-f", "--raw-sim", dest="raw_sim", default=DEFAULT_RAW_SIM,
                        help=f"Raw simulation Geant4 ROOT file (default: {DEFAULT_RAW_SIM})")
    parser.add_argument("-g", "--geofile", dest="geofile", default=DEFAULT_GEOFILE,
                        help=f"Geometry ROOT file (default: {DEFAULT_GEOFILE})")
    parser.add_argument("-d", "--data-ref", dest="data_ref", default=DEFAULT_DATA_REF,
                        help=f"Reference Collision Data ROOT file (default: {DEFAULT_DATA_REF})")
    parser.add_argument("--data-hist", dest="data_hist", default=DEFAULT_DATA_HIST,
                        help=f"Path to reference SciFi hits histogram in data ROOT file (default: {DEFAULT_DATA_HIST})")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default="plots/calibration_scan",
                        help="Output directory for scan results, plots, and reports (default: plots/calibration_scan)")
    parser.add_argument("-w", "--work-dir", dest="work_dir", default="",
                        help="Scratch working directory for intermediate ROOT files (default: <output_dir>/work)")

    # Grid definition
    parser.add_argument("-ts", "--thresholds", dest="thresholds", nargs="+", type=float,
                        default=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                        help="Grid of SciFi thresholds in p.e. (default: 3.0 3.5 4.0 4.5 5.0 5.5 6.0)")
    parser.add_argument("-ss", "--saturation", dest="saturation", type=float, default=104.0,
                        help="SciFi SiPM saturation in p.e. (default: 104.0)")
    parser.add_argument("-n", "--events", dest="num_events", type=int, default=2000,
                        help="Number of events to digitize/reconstruct per threshold (default: 2000, 0 = all)")
    parser.add_argument("-j", "--threads", dest="num_threads", type=int, default=8,
                        help="Number of threads for extraction (default: 8)")

    # Execution controls
    parser.add_argument("--mode", choices=["offline", "pipeline"], default="offline",
                        help="Scan mode: 'offline' (recommended) scans thresholds directly on already-digitized files in seconds; 'pipeline' runs full digitization and tracking from raw Geant4 (default: offline)")
    parser.add_argument("-i", "--input-file", "--input-reco", dest="input_file",
                        default="/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200_trks.root",
                        help="Input reconstructed file for offline mode (default: Trimuon MC digCPP-200_trks.root)")
    parser.add_argument("--snd-home", dest="snd_home", default=DEFAULT_SND_HOME,
                        help=f"Path to sndsw installation directory (default: {DEFAULT_SND_HOME})")
    parser.add_argument("--skip-digi-if-exists", dest="skip_digi_if_exists", action="store_true", default=False,
                        help="Skip digitization and reconstruction if output file for this threshold exists")
    parser.add_argument("--skip-tracking", dest="skip_tracking", action="store_true", default=False,
                        help="Skip track reconstruction if track files already exist")

    args = parser.parse_args()

    # Paths resolution
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).resolve() if args.work_dir else (out_dir / "work")
    work_dir.mkdir(parents=True, exist_ok=True)

    snd_home = Path(args.snd_home).resolve()
    digi_script = snd_home / "shipLHC" / "run_digiSND.py"
    reco_script = snd_home / "shipLHC" / "scripts" / "run_TrackSelections.py"
    extract_script = Path(_repo_root) / "scripts" / "calibration" / "extract_scifi_calibration_histograms.py"

    if args.mode == "pipeline":
        assert digi_script.exists(), f"run_digiSND.py not found at {digi_script}"
        assert reco_script.exists(), f"run_TrackSelections.py not found at {reco_script}"
    assert extract_script.exists(), f"extract script not found at {extract_script}"

    print("=" * 80)
    print("SND@LHC SCIFI DIGITIZATION THRESHOLD GRID SCANNER & CALIBRATION ENGINE")
    print("=" * 80)
    print(f"Raw Simulation     : {args.raw_sim}")
    print(f"Geofile            : {args.geofile}")
    print(f"Data Reference     : {args.data_ref} [{args.data_hist}]")
    print(f"Threshold Grid     : {args.thresholds} p.e.")
    print(f"SiPM Saturation    : {args.saturation} p.e.")
    print(f"Events per Point   : {args.num_events if args.num_events > 0 else 'All'}")
    print(f"Output Directory   : {out_dir}")
    print(f"Scratch Directory  : {work_dir}")
    print("=" * 80)

    # 1. Load Data Reference Histogram
    data_file = ROOT.TFile.Open(args.data_ref)
    assert data_file and not data_file.IsZombie(), f"Failed to open Data reference file: {args.data_ref}"
    h_data = data_file.Get(args.data_hist)
    assert h_data, f"Failed to find histogram '{args.data_hist}' in {args.data_ref}"
    h_data.SetDirectory(0)
    data_file.Close()

    data_peak = h_data.GetBinCenter(h_data.GetMaximumBin())
    data_mean = h_data.GetMean()
    data_rms = h_data.GetRMS()
    print(f"[✓] Loaded Data Reference: Peak = {data_peak:.1f} hits | Mean = {data_mean:.2f} | RMS = {data_rms:.2f}")
    print("-" * 80)

    # 2. Iterate over Threshold Grid
    results: List[Dict[str, Any]] = []

    for idx, ts in enumerate(args.thresholds, start=1):
        tag = f"ts_{ts:.2f}".replace(".", "p")
        print(f"\n[{idx}/{len(args.thresholds)}] === Processing Threshold: {ts:.2f} p.e. ({tag}) ===")

        # Target file paths
        calib_file = work_dir / f"calib_{tag}.root"

        if args.mode == "offline":
            reco_file = Path(args.input_file)
            assert reco_file.exists(), f"Input reconstructed file not found: {reco_file}"
        else:
            digi_file = work_dir / f"sim_{tag}_digCPP.root"
            reco_file = work_dir / f"sim_{tag}_trks.root"

            # Step A: Digitization
            if args.skip_digi_if_exists and digi_file.exists():
                print(f"[i] Reusing existing digitized file: {digi_file.name}")
            else:
                # run_digiSND creates outfile in current directory as <input_base>_digCPP.root
                # We run inside work_dir so the file is created there cleanly
                cmd_digi = (
                    f"cd {work_dir} && "
                    f"python {digi_script} "
                    f"-f {args.raw_sim} "
                    f"-g {args.geofile} "
                    f"-n {args.num_events} "
                    f"-ts {ts:.2f} "
                    f"-ss {args.saturation:.1f} "
                    f"-cpp"
                )
                ok = run_command(cmd_digi, desc=f"Running FairRoot C++ Digitization at ts = {ts:.2f} p.e.")
                if not ok:
                    print(f"[-] Skipping threshold {ts:.2f} due to digitization failure.")
                    continue

                # Identify created file and move to standard name
                # run_digiSND creates <raw_sim_basename>_digCPP.root
                raw_base = Path(args.raw_sim).name.replace(".root", "")
                created_digi = work_dir / f"{raw_base}_digCPP.root"
                if created_digi.exists() and created_digi != digi_file:
                    shutil.move(str(created_digi), str(digi_file))

            # Step B: Track Reconstruction
            if (args.skip_digi_if_exists or args.skip_tracking) and reco_file.exists():
                print(f"[i] Reusing existing reconstructed track file: {reco_file.name}")
            else:
                cmd_reco = (
                    f"python {reco_script} "
                    f"-f {digi_file} "
                    f"-g {args.geofile} "
                    f"-o {reco_file} "
                    f"-sc 1 -ht -st -n {args.num_events}"
                )
                ok = run_command(cmd_reco, desc=f"Running Hough + Simple Track Reconstruction for {tag}")
                if not ok:
                    print(f"[-] Skipping threshold {ts:.2f} due to tracking failure.")
                    continue

        # Step C: Single Muon Histogram Extraction
        thresh_arg = f"--scifi-threshold {ts:.2f}" if args.mode == "offline" else ""
        ev_arg = f"--max-events {args.num_events}" if args.num_events > 0 else ""
        cmd_extract = (
            f"python {extract_script} "
            f"-i {reco_file} "
            f"-o {calib_file} "
            f"--skip-scifi-hits-cut "
            f"--no-save-events "
            f"{thresh_arg} "
            f"{ev_arg} "
            f"-j {args.num_threads}"
        )
        ok = run_command(cmd_extract, desc=f"Extracting Single-Muon Calibration Histograms for {tag}")
        if not ok:
            print(f"[-] Skipping threshold {ts:.2f} due to histogram extraction failure.")
            continue

        # Step D: Load MC Histogram & Evaluate Statistical Metrics
        mc_file = ROOT.TFile.Open(str(calib_file))
        if not mc_file or mc_file.IsZombie():
            print(f"[-] Failed to open {calib_file}")
            continue

        h_mc = mc_file.Get("Histograms/1D/h1_scifi_nhits")
        if not h_mc:
            print(f"[-] Histogram Histograms/1D/h1_scifi_nhits not found in {calib_file}")
            mc_file.Close()
            continue

        h_mc.SetDirectory(0)
        mc_file.Close()

        metrics = compute_hist_metrics(h_data, h_mc)
        entry = {
            "threshold": ts,
            "tag": tag,
            "calib_file": str(calib_file),
            "h_mc": h_mc,
            **metrics
        }
        results.append(entry)

        print(f"    --> Results for ts = {ts:.2f} p.e.:")
        print(f"        Peak: MC = {metrics['peak_mc']:.1f} (Data = {metrics['peak_data']:.1f}, Δ = {metrics['delta_peak']:.1f})")
        print(f"        Mean: MC = {metrics['mean_mc']:.2f} (Data = {metrics['mean_data']:.2f}, Δ = {metrics['delta_mean']:.2f})")
        print(f"        Wasserstein W1 = {metrics['w1']:.3f} hits | KS Dist = {metrics['ks_stat']:.4f} | chi2/ndf = {metrics['chi2_ndf']:.2f}")

    if not results:
        print("[-] No thresholds evaluated successfully. Exiting.")
        sys.exit(1)

    # 3. Sort & Rank Results by Wasserstein Distance (W1) and Delta Peak
    results.sort(key=lambda r: (r["delta_peak"], r["w1"]))
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    best = results[0]

    # 4. Print Summary Ranking Table
    print("\n" + "=" * 95)
    print(f"{'Rank':<5} | {'Threshold':<11} | {'Peak (MC)':<10} | {'Mean (MC)':<10} | {'ΔPeak':<7} | {'ΔMean':<7} | {'W1 [hits]':<10} | {'KS Dist':<9} | {'χ²/ndf':<8}")
    print("-" * 95)
    for r in results:
        marker = " [BEST]" if r["rank"] == 1 else ""
        print(f"{r['rank']:<5} | {r['threshold']:<5.2f} p.e.  | {r['peak_mc']:<10.1f} | {r['mean_mc']:<10.2f} | {r['delta_peak']:<7.1f} | {r['delta_mean']:<7.2f} | {r['w1']:<10.3f} | {r['ks_stat']:<9.4f} | {r['chi2_ndf']:<8.2f}{marker}")
    print("=" * 95)
    print(f"\n[★] OPTIMAL DIGITIZATION THRESHOLD: {best['threshold']:.2f} p.e.")
    print(f"    Achieves minimal hit discrepancy: W1 = {best['w1']:.3f} hits (Peak = {best['peak_mc']:.1f} vs Data {data_peak:.1f})")

    # 5. Generate Publication Plots
    print("\n[*] Generating calibration scan diagnostic plots...")
    # Plot A: Overlaid Distributions
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Plot Data reference
    nbins = h_data.GetNbinsX()
    x_centers = np.array([h_data.GetBinCenter(i) for i in range(1, nbins + 1)])
    counts_d = np.array([h_data.GetBinContent(i) for i in range(1, nbins + 1)])
    norm_d = counts_d / np.sum(counts_d) if np.sum(counts_d) > 0 else counts_d

    ax_top.step(x_centers, norm_d, where='mid', label=f'Collision Data (Peak = {data_peak:.0f})', color='black', linewidth=2.5)

    # Plot MC at various thresholds (sorted by threshold)
    results_by_thresh = sorted(results, key=lambda r: r["threshold"])
    cmap = plt.get_cmap("viridis", len(results_by_thresh))

    for idx, r in enumerate(results_by_thresh):
        h = r["h_mc"]
        counts_m = np.array([h.GetBinContent(i) for i in range(1, nbins + 1)])
        norm_m = counts_m / np.sum(counts_m) if np.sum(counts_m) > 0 else counts_m
        is_best = (r["threshold"] == best["threshold"])
        lw = 2.5 if is_best else 1.2
        lbl = f"MC ts={r['threshold']:.2f} p.e. (W1={r['w1']:.2f})" + (" ★ BEST" if is_best else "")
        col = 'crimson' if is_best else cmap(idx)
        ax_top.step(x_centers, norm_m, where='mid', label=lbl, color=col, linewidth=lw, linestyle='-' if is_best else '--')

        # Ratio to Data in bottom panel
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(norm_d > 0, norm_m / norm_d, np.nan)
        ax_bot.step(x_centers, ratio, where='mid', color=col, linewidth=lw, linestyle='-' if is_best else '--')

    ax_top.set_ylabel("Normalized Density")
    ax_top.set_xlim(5, 35)
    ax_top.legend(frameon=True, fontsize=9, loc='upper right')
    ax_top.set_title("SND@LHC SciFi Hit Multiplicity vs Digitization Threshold", fontsize=12, fontweight='bold')
    ax_top.grid(True, linestyle=':', alpha=0.5)

    ax_bot.axhline(1.0, color='black', linestyle=':')
    ax_bot.set_ylabel("MC / Data")
    ax_bot.set_xlabel("SciFi Valid Hits Across 10 Planes")
    ax_bot.set_ylim(0.0, 2.0)
    ax_bot.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plot_dist_path = out_dir / "threshold_scan_distributions.png"
    fig.savefig(plot_dist_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved distribution comparison plot: {plot_dist_path}")

    # Plot B: Metrics vs Threshold Curve
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    thresh_vals = [r["threshold"] for r in results_by_thresh]
    w1_vals = [r["w1"] for r in results_by_thresh]
    peak_vals = [r["peak_mc"] for r in results_by_thresh]
    mean_vals = [r["mean_mc"] for r in results_by_thresh]
    ks_vals = [r["ks_stat"] for r in results_by_thresh]

    ax1.plot(thresh_vals, w1_vals, 'o-', color='navy', linewidth=2, markersize=7, label='Wasserstein Distance ($W_1$)')
    ax1.axvline(best["threshold"], color='crimson', linestyle='--', label=f'Optimal Threshold = {best["threshold"]:.2f} p.e.')
    ax1.set_xlabel("SciFi Threshold $t_s$ [p.e.]")
    ax1.set_ylabel("Wasserstein Distance $W_1$ [hits]")
    ax1.set_title("Emulation Discrepancy ($W_1$) vs Threshold")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(thresh_vals, peak_vals, 's-', color='darkgreen', linewidth=2, label='MC Peak Bin')
    ax2.plot(thresh_vals, mean_vals, '^-', color='teal', linewidth=2, label='MC Mean Hits')
    ax2.axhline(data_peak, color='black', linestyle=':', label=f'Data Peak ({data_peak:.1f})')
    ax2.axhline(data_mean, color='gray', linestyle=':', label=f'Data Mean ({data_mean:.2f})')
    ax2.axvline(best["threshold"], color='crimson', linestyle='--')
    ax2.set_xlabel("SciFi Threshold $t_s$ [p.e.]")
    ax2.set_ylabel("SciFi Hits")
    ax2.set_title("Peak and Mean Multiplicity vs Threshold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_metric_path = out_dir / "threshold_scan_metrics.png"
    fig2.savefig(plot_metric_path, dpi=300)
    plt.close(fig2)
    print(f"[✓] Saved metric scan curves plot: {plot_metric_path}")

    # 6. Save ROOT Summary File with TGraphs
    root_summary_path = out_dir / "threshold_scan_summary.root"
    tfile_out = ROOT.TFile(str(root_summary_path), "RECREATE")

    h_data.Write("h_data_reference")

    n_pts = len(thresh_vals)
    g_w1 = ROOT.TGraph(n_pts, np.array(thresh_vals, dtype=np.float64), np.array(w1_vals, dtype=np.float64))
    g_w1.SetName("g_w1_vs_threshold")
    g_w1.SetTitle("Wasserstein Distance W1 vs SciFi Threshold;Threshold [p.e.];W1 [hits]")
    g_w1.Write()

    g_peak = ROOT.TGraph(n_pts, np.array(thresh_vals, dtype=np.float64), np.array(peak_vals, dtype=np.float64))
    g_peak.SetName("g_peak_vs_threshold")
    g_peak.SetTitle("Peak Hit Bin vs SciFi Threshold;Threshold [p.e.];Peak Bin [hits]")
    g_peak.Write()

    g_mean = ROOT.TGraph(n_pts, np.array(thresh_vals, dtype=np.float64), np.array(mean_vals, dtype=np.float64))
    g_mean.SetName("g_mean_vs_threshold")
    g_mean.SetTitle("Mean Hits vs SciFi Threshold;Threshold [p.e.];Mean Hits")
    g_mean.Write()

    g_ks = ROOT.TGraph(n_pts, np.array(thresh_vals, dtype=np.float64), np.array(ks_vals, dtype=np.float64))
    g_ks.SetName("g_ks_vs_threshold")
    g_ks.SetTitle("Kolmogorov-Smirnov Distance vs SciFi Threshold;Threshold [p.e.];D_KS")
    g_ks.Write()

    for r in results:
        r["h_mc"].Write(f"h_mc_{r['tag']}")

    tfile_out.Close()
    print(f"[✓] Saved master ROOT summary file: {root_summary_path}")

    # 7. Write Markdown Summary Report
    md_path = out_dir / "threshold_scan_report.md"
    with open(md_path, "w") as f:
        f.write("# SND@LHC SciFi Digitization Threshold Calibration Report\n\n")
        f.write(f"- **Evaluated Simulation**: `{args.raw_sim}`\n")
        f.write(f"- **Data Reference**: `{args.data_ref}` (`{args.data_hist}`)\n")
        f.write(f"- **Optimal Threshold**: **`{best['threshold']:.2f} p.e.`**\n")
        f.write(f"- **Minimum Wasserstein Distance**: **`{best['w1']:.3f} hits`**\n")
        f.write(f"- **Peak Alignment**: MC = `{best['peak_mc']:.1f}` vs Data = `{data_peak:.1f}` (Delta = `{best['delta_peak']:.1f}`)\n")
        f.write(f"- **Mean Multiplicity**: MC = `{best['mean_mc']:.2f}` vs Data = `{data_mean:.2f}` (Delta = `{best['delta_mean']:.2f}`)\n\n")
        f.write("## Quantitative Ranking Table\n\n")
        f.write("| Rank | Threshold [p.e.] | Peak (MC) | Mean (MC) | ΔPeak | ΔMean | W1 [hits] | KS Distance | χ²/ndf | Status |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            status = "**BEST**" if r["rank"] == 1 else "Sub-optimal"
            f.write(f"| {r['rank']} | {r['threshold']:.2f} | {r['peak_mc']:.1f} | {r['mean_mc']:.2f} | {r['delta_peak']:.1f} | {r['delta_mean']:.2f} | {r['w1']:.3f} | {r['ks_stat']:.4f} | {r['chi2_ndf']:.2f} | {status} |\n")

    print(f"[✓] Saved Markdown report: {md_path}")
    print("\n" + "=" * 80)
    print("CALIBRATION GRID SCAN COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
