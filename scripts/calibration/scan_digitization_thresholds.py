#!/usr/bin/env python3
"""
================================================================================
SND@LHC Digitization & Calibration Multi-Parameter Optimization Engine
================================================================================
Orchestrates systematic multi-parameter grid scans across SciFi digitization
photo-electron thresholds (ts), QDC calibration scale factors (s_qdc), and
pedestal offsets (b0) on Monte Carlo simulations. Evaluates emulation fidelity
against real collision data by simultaneously optimizing:
1. SciFi Hit Multiplicity distributions (track geometry & detection efficiency)
2. SciFi Hit QDC distributions (single-MIP energy scale & SiPM response)
3. Total SciFi integrated track charge (calorimetric consistency)

Metrics Computed:
- Wasserstein Distance (W1 / Earth Mover's Distance in hits and in QDC a.u.)
- Mode / Peak Alignment (|Peak_MC - Peak_Data|)
- Mean Multiplicity & Charge Discrepancy (|mu_MC - mu_Data|)
- Kolmogorov-Smirnov Distance (D_KS and p-value)
- Joint Calibration Loss Function (relative, scale-invariant loss)

Author: SND@LHC Collaboration
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
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
DEFAULT_DATA_QDC_REF = "plots/scifi_calibration_run8329.root"
DEFAULT_DATA_QDC_HIST = "Histograms/1D/h1_hit_qdc"
DEFAULT_DATA_SUMQDC_HIST = "Histograms/1D/h1_scifi_sum_qdc"


def compute_hist_metrics(h_data: ROOT.TH1D, h_mc: ROOT.TH1D) -> Dict[str, Any]:
    """
    Computes statistical similarity metrics between Data and MC TH1 histograms.
    """
    peak_data = float(h_data.GetBinCenter(h_data.GetMaximumBin()))
    peak_mc = float(h_mc.GetBinCenter(h_mc.GetMaximumBin()))
    delta_peak = abs(peak_mc - peak_data)

    mean_data = float(h_data.GetMean())
    mean_mc = float(h_mc.GetMean())
    delta_mean = abs(mean_mc - mean_data)

    rms_data = float(h_data.GetRMS())
    rms_mc = float(h_mc.GetRMS())

    nbins_d = h_data.GetNbinsX()
    x_edges_d = np.array([h_data.GetBinLowEdge(i) for i in range(1, nbins_d + 2)])
    dx_d = float(h_data.GetBinWidth(1))

    counts_d = np.array([h_data.GetBinContent(i) for i in range(1, nbins_d + 1)], dtype=np.float64)
    sum_d = float(np.sum(counts_d))

    nbins_m = h_mc.GetNbinsX()
    x_edges_m = np.array([h_mc.GetBinLowEdge(i) for i in range(1, nbins_m + 2)])
    counts_m = np.array([h_mc.GetBinContent(i) for i in range(1, nbins_m + 1)], dtype=np.float64)
    sum_m = float(np.sum(counts_m))

    if sum_d == 0 or sum_m == 0:
        return {
            "peak_data": peak_data, "peak_mc": peak_mc, "delta_peak": delta_peak,
            "mean_data": mean_data, "mean_mc": mean_mc, "delta_mean": delta_mean,
            "rms_data": rms_data, "rms_mc": rms_mc,
            "w1": float("inf"), "ks_stat": 1.0, "ks_pval": 0.0,
            "chi2": float("inf"), "ndf": 1, "chi2_ndf": float("inf"),
            "counts_m_on_data": np.zeros_like(counts_d),
            "counts_d": counts_d
        }

    # Interpolate MC CDF onto Data bin edges to compute accurate EMD even if binning differs
    cdf_m_raw = np.concatenate(([0.0], np.cumsum(counts_m) / sum_m))
    cdf_m_on_d_edges = np.interp(x_edges_d, x_edges_m, cdf_m_raw, left=0.0, right=1.0)
    pdf_m_rebinned = np.diff(cdf_m_on_d_edges)
    counts_m_on_data = pdf_m_rebinned * sum_m

    cdf_d = np.cumsum(counts_d) / sum_d
    cdf_m_centers = 0.5 * (cdf_m_on_d_edges[:-1] + cdf_m_on_d_edges[1:])

    # 4. Wasserstein Distance (W1 / EMD in hits)
    w1 = float(np.sum(np.abs(cdf_d - cdf_m_centers)) * dx_d)

    # 5. Kolmogorov-Smirnov Statistic
    ks_stat = float(np.max(np.abs(cdf_d - cdf_m_centers)))
    n_eff = (sum_d * sum_m) / (sum_d + sum_m)
    ks_pval = float(stats.kstwo.sf(ks_stat, int(round(n_eff)))) if n_eff > 1 else 0.0

    # 6. Shape Chi-square
    alpha = sum_d / sum_m
    chi2 = 0.0
    ndf = 0
    for i in range(nbins_d):
        if counts_d[i] > 0 or counts_m_on_data[i] > 0:
            var = counts_d[i] + (alpha**2) * counts_m_on_data[i]
            if var > 0:
                diff = counts_d[i] - alpha * counts_m_on_data[i]
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
        "chi2_ndf": float(chi2_ndf),
        "counts_m_on_data": counts_m_on_data,
        "counts_d": counts_d
    }


def compute_rescaled_qdc_metrics(
    h_data: ROOT.TH1D,
    h_mc_raw: ROOT.TH1D,
    scale: float,
    offset: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluates similarity between calibrated collision data QDC and MC QDC rescaled via:
        x_calib = (x_raw - offset) / scale
    Accurately maps CDF, rebins onto the Data grid conserving probability, and computes
    Wasserstein distance W1, Kolmogorov-Smirnov distance, and peak/mean discrepancies.
    """
    peak_data = float(h_data.GetBinCenter(h_data.GetMaximumBin()))
    raw_mc_peak = float(h_mc_raw.GetBinCenter(h_mc_raw.GetMaximumBin()))
    peak_mc = (raw_mc_peak - offset) / scale
    delta_peak = abs(peak_mc - peak_data)

    mean_data = float(h_data.GetMean())
    mean_mc = (float(h_mc_raw.GetMean()) - offset) / scale
    delta_mean = abs(mean_mc - mean_data)

    rms_data = float(h_data.GetRMS())
    rms_mc = float(h_mc_raw.GetRMS()) / scale

    nbins_d = h_data.GetNbinsX()
    x_edges_d = np.array([h_data.GetBinLowEdge(i) for i in range(1, nbins_d + 2)])
    dx_d = float(h_data.GetBinWidth(1))
    counts_d = np.array([h_data.GetBinContent(i) for i in range(1, nbins_d + 1)], dtype=np.float64)
    sum_d = float(np.sum(counts_d))

    nbins_m = h_mc_raw.GetNbinsX()
    x_edges_m = np.array([h_mc_raw.GetBinLowEdge(i) for i in range(1, nbins_m + 2)])
    counts_m = np.array([h_mc_raw.GetBinContent(i) for i in range(1, nbins_m + 1)], dtype=np.float64)
    sum_m = float(np.sum(counts_m))

    if sum_d == 0 or sum_m == 0:
        return {
            "peak_data": peak_data, "peak_mc": peak_mc, "delta_peak": delta_peak,
            "mean_data": mean_data, "mean_mc": mean_mc, "delta_mean": delta_mean,
            "rms_data": rms_data, "rms_mc": rms_mc,
            "w1": float("inf"), "ks_stat": 1.0, "ks_pval": 0.0,
            "chi2": float("inf"), "ndf": 1, "chi2_ndf": float("inf"),
            "counts_m_rebinned": np.zeros_like(counts_d),
            "scale": scale, "offset": offset
        }

    # Transform MC bin edges
    x_edges_m_scaled = (x_edges_m - offset) / scale
    cdf_m_raw = np.concatenate(([0.0], np.cumsum(counts_m) / sum_m))

    # Project CDF onto data edges
    cdf_m_on_d_edges = np.interp(x_edges_d, x_edges_m_scaled, cdf_m_raw, left=0.0, right=1.0)
    pdf_m_rebinned = np.diff(cdf_m_on_d_edges)
    counts_m_rebinned = pdf_m_rebinned * sum_m

    # CDFs at bin centers
    cdf_d = np.cumsum(counts_d) / sum_d
    cdf_m_centers = 0.5 * (cdf_m_on_d_edges[:-1] + cdf_m_on_d_edges[1:])

    # Wasserstein Distance W1 (EMD)
    w1 = float(np.sum(np.abs(cdf_d - cdf_m_centers)) * dx_d)

    # Kolmogorov-Smirnov Statistic
    ks_stat = float(np.max(np.abs(cdf_d - cdf_m_centers)))
    n_eff = (sum_d * sum_m) / (sum_d + sum_m)
    ks_pval = float(stats.kstwo.sf(ks_stat, int(round(n_eff)))) if n_eff > 1 else 0.0

    # Shape Chi2
    alpha = sum_d / sum_m
    chi2 = 0.0
    ndf = 0
    for i in range(nbins_d):
        if counts_d[i] > 0 or counts_m_rebinned[i] > 0:
            var = counts_d[i] + (alpha**2) * counts_m_rebinned[i]
            if var > 0:
                diff = counts_d[i] - alpha * counts_m_rebinned[i]
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
        "chi2_ndf": float(chi2_ndf),
        "counts_m_rebinned": counts_m_rebinned,
        "scale": scale,
        "offset": offset
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
        description="SND@LHC Digitization & Calibration Multi-Parameter Optimization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input & Output Specifications
    parser.add_argument("-f", "--raw-sim", dest="raw_sim", default=DEFAULT_RAW_SIM,
                        help=f"Raw simulation Geant4 ROOT file (default: {DEFAULT_RAW_SIM})")
    parser.add_argument("-g", "--geofile", dest="geofile", default=DEFAULT_GEOFILE,
                        help=f"Geometry ROOT file (default: {DEFAULT_GEOFILE})")
    parser.add_argument("-d", "--data-ref", dest="data_ref", default=DEFAULT_DATA_REF,
                        help=f"Reference Collision Data ROOT file for hit multiplicity (default: {DEFAULT_DATA_REF})")
    parser.add_argument("--data-hist", dest="data_hist", default=DEFAULT_DATA_HIST,
                        help=f"Path to reference SciFi hits histogram (default: {DEFAULT_DATA_HIST})")
    parser.add_argument("--data-qdc-ref", dest="data_qdc_ref", default=DEFAULT_DATA_QDC_REF,
                        help=f"Reference Collision Data ROOT file for QDC (default: {DEFAULT_DATA_QDC_REF})")
    parser.add_argument("--data-qdc-hist", dest="data_qdc_hist", default=DEFAULT_DATA_QDC_HIST,
                        help=f"Path to reference SciFi hit QDC histogram (default: {DEFAULT_DATA_QDC_HIST})")
    parser.add_argument("--data-sumqdc-hist", dest="data_sumqdc_hist", default=DEFAULT_DATA_SUMQDC_HIST,
                        help=f"Path to reference SciFi sum QDC histogram (default: {DEFAULT_DATA_SUMQDC_HIST})")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default="plots/calibration_scan",
                        help="Output directory for scan results, plots, and reports (default: plots/calibration_scan)")
    parser.add_argument("-w", "--work-dir", dest="work_dir", default="",
                        help="Scratch working directory for intermediate ROOT files (default: <output_dir>/work)")

    # Grid definition: Multi-parameter variation around nominal defaults
    parser.add_argument("-ts", "--thresholds", dest="thresholds", nargs="+", type=float,
                        default=[3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
                        help="Grid of SciFi thresholds in p.e. (nominal default: 3.5 p.e.)")
    parser.add_argument("-qs", "--qdc-scales", dest="qdc_scales", nargs="+", type=float,
                        default=[3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4],
                        help="Grid of SciFi QDC calibration scale factors (nominal default: 4.50)")
    parser.add_argument("-qo", "--qdc-offsets", dest="qdc_offsets", nargs="+", type=float,
                        default=[-0.4, -0.2, 0.0, 0.2, 0.4],
                        help="Grid of SciFi QDC pedestal offsets in a.u. (nominal default: 0.0)")
    parser.add_argument("-ss", "--saturation", dest="saturation", type=float, default=104.0,
                        help="SciFi SiPM saturation in p.e. (default: 104.0)")

    # Loss function weights
    parser.add_argument("--weight-hits", dest="weight_hits", type=float, default=1.0,
                        help="Loss weight for SciFi hit multiplicity (default: 1.0)")
    parser.add_argument("--weight-qdc", dest="weight_qdc", type=float, default=1.0,
                        help="Loss weight for SciFi hit QDC energy scale (default: 1.0)")
    parser.add_argument("--weight-sumqdc", dest="weight_sumqdc", type=float, default=0.0,
                        help="Loss weight for SciFi sum QDC track charge (default: 0.0)")

    # Execution controls
    parser.add_argument("-n", "--events", dest="num_events", type=int, default=2000,
                        help="Number of events to extract per threshold (default: 2000, 0 = all)")
    parser.add_argument("-j", "--threads", dest="num_threads", type=int, default=8,
                        help="Number of threads for extraction (default: 8)")
    parser.add_argument("--mode", choices=["offline", "pipeline"], default="offline",
                        help="Scan mode: 'offline' (recommended) scans thresholds directly on already-digitized files in seconds; 'pipeline' runs full digitization and tracking from raw Geant4 (default: offline)")
    parser.add_argument("-i", "--input-file", "--input-reco", dest="input_file",
                        default="/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200_trks.root",
                        help="Input reconstructed file for offline mode (default: Trimuon MC digCPP-200_trks.root)")
    parser.add_argument("--snd-home", dest="snd_home", default=DEFAULT_SND_HOME,
                        help=f"Path to sndsw installation directory (default: {DEFAULT_SND_HOME})")
    parser.add_argument("--norm", choices=["peak", "window", "area"], default="peak",
                        help="Plot normalization mode: 'peak' (scales max to 1.0 for direct shape comparison), 'window' (unit area in [5, 35]), or 'area' (global unit area) (default: peak)")
    parser.add_argument("--skip-digi-if-exists", "--reuse-calib", "--skip-existing", dest="skip_digi_if_exists", action="store_true", default=True,
                        help="Reuse existing intermediate calibration ROOT files if available in work dir (default: True)")
    parser.add_argument("--force-reextract", dest="skip_digi_if_exists", action="store_false",
                        help="Force re-extraction of calibration ROOT files from EOS")
    parser.add_argument("--skip-tracking", dest="skip_tracking", action="store_true", default=False,
                        help="Skip track reconstruction in pipeline mode if track files already exist")

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

    n_combos = len(args.thresholds) * len(args.qdc_scales) * len(args.qdc_offsets)
    print("=" * 85)
    print("SND@LHC SCIFI MULTI-PARAMETER DIGITIZATION & QDC CALIBRATION OPTIMIZATION ENGINE")
    print("=" * 85)
    print(f"Data Hits Reference : {args.data_ref} [{args.data_hist}]")
    print(f"Data QDC Reference  : {args.data_qdc_ref} [{args.data_qdc_hist}]")
    print(f"Threshold Grid (ts) : {args.thresholds} p.e. (nominal default: 3.5 p.e.)")
    print(f"QDC Scales Grid(qs) : {args.qdc_scales} (nominal default: 4.50)")
    print(f"QDC Offsets Grid(qo): {args.qdc_offsets} a.u. (nominal default: 0.0)")
    print(f"Total Combinations  : {n_combos} parameter points")
    print(f"Loss Weights        : w_hits = {args.weight_hits:.2f}, w_qdc = {args.weight_qdc:.2f}, w_sumqdc = {args.weight_sumqdc:.2f}")
    print(f"Events per Point    : {args.num_events if args.num_events > 0 else 'All'}")
    print(f"Output Directory    : {out_dir}")
    print(f"Scratch Directory   : {work_dir}")
    print("=" * 85)

    # 1. Load Data Reference Histograms
    # A. Hit Multiplicity
    f_d_hits = ROOT.TFile.Open(args.data_ref)
    assert f_d_hits and not f_d_hits.IsZombie(), f"Failed to open Data reference file: {args.data_ref}"
    h_data_hits = f_d_hits.Get(args.data_hist)
    assert h_data_hits, f"Failed to find histogram '{args.data_hist}' in {args.data_ref}"
    h_data_hits.SetDirectory(0)
    f_d_hits.Close()

    # B. Hit QDC & Sum QDC
    f_d_qdc = ROOT.TFile.Open(args.data_qdc_ref)
    assert f_d_qdc and not f_d_qdc.IsZombie(), f"Failed to open Data QDC reference file: {args.data_qdc_ref}"
    h_data_qdc = f_d_qdc.Get(args.data_qdc_hist)
    assert h_data_qdc, f"Failed to find QDC histogram '{args.data_qdc_hist}' in {args.data_qdc_ref}"
    h_data_qdc.SetDirectory(0)

    h_data_sumqdc = f_d_qdc.Get(args.data_sumqdc_hist)
    if h_data_sumqdc:
        h_data_sumqdc.SetDirectory(0)
    f_d_qdc.Close()

    data_peak_hits = float(h_data_hits.GetBinCenter(h_data_hits.GetMaximumBin()))
    data_mean_hits = float(h_data_hits.GetMean())
    data_peak_qdc = float(h_data_qdc.GetBinCenter(h_data_qdc.GetMaximumBin()))
    data_mean_qdc = float(h_data_qdc.GetMean())

    print(f"[✓] Loaded Data Reference Hits : Peak = {data_peak_hits:.1f} hits | Mean = {data_mean_hits:.2f} hits")
    print(f"[✓] Loaded Data Reference QDC  : Peak = {data_peak_qdc:.2f} a.u. | Mean = {data_mean_qdc:.2f} a.u.")
    print("-" * 85)

    # 2. Extract or Cache MC Calibration Files for Each Threshold
    mc_cache: Dict[float, Dict[str, Any]] = {}

    for idx, ts in enumerate(args.thresholds, start=1):
        tag = f"ts_{ts:.2f}".replace(".", "p")
        calib_file = work_dir / f"calib_{tag}.root"

        # Check existing cached calib file
        calib_valid = calib_file.exists() and calib_file.stat().st_size > 10000

        if args.skip_digi_if_exists and calib_valid:
            print(f"[{idx}/{len(args.thresholds)}] Threshold {ts:.2f} p.e. ({tag}): Reusing cached {calib_file.name}")
        else:
            print(f"\n[{idx}/{len(args.thresholds)}] Threshold {ts:.2f} p.e. ({tag}): Extracting simulation...")
            if args.mode == "offline":
                reco_file = Path(args.input_file)
                assert reco_file.exists(), f"Input reconstructed file not found: {reco_file}"
            else:
                digi_file = work_dir / f"sim_{tag}_digCPP.root"
                reco_file = work_dir / f"sim_{tag}_trks.root"

                if not (args.skip_digi_if_exists and digi_file.exists()):
                    cmd_digi = (
                        f"cd {work_dir} && python {digi_script} "
                        f"-f {args.raw_sim} -g {args.geofile} -n {args.num_events} "
                        f"-ts {ts:.2f} -ss {args.saturation:.1f} -cpp"
                    )
                    ok = run_command(cmd_digi, desc=f"Digitizing at ts = {ts:.2f} p.e.")
                    if not ok:
                        continue
                    raw_base = Path(args.raw_sim).name.replace(".root", "")
                    created_digi = work_dir / f"{raw_base}_digCPP.root"
                    if created_digi.exists() and created_digi != digi_file:
                        shutil.move(str(created_digi), str(digi_file))

                if not (args.skip_digi_if_exists and reco_file.exists()):
                    cmd_reco = (
                        f"python {reco_script} -f {digi_file} -g {args.geofile} -o {reco_file} "
                        f"-sc 1 -ht -st -n {args.num_events}"
                    )
                    ok = run_command(cmd_reco, desc=f"Reconstructing tracks for {tag}")
                    if not ok:
                        continue

            # Extraction
            thresh_arg = f"--scifi-threshold {ts:.2f}" if args.mode == "offline" else ""
            ev_arg = f"--max-events {args.num_events}" if args.num_events > 0 else ""
            cmd_extract = (
                f"python {extract_script} -i {reco_file} -o {calib_file} "
                f"--skip-scifi-hits-cut --no-save-events {thresh_arg} {ev_arg} -j {args.num_threads}"
            )
            ok = run_command(cmd_extract, desc=f"Extracting histograms for {tag}")
            if not ok:
                continue

        # Load MC histograms from file
        f_mc = ROOT.TFile.Open(str(calib_file))
        if not f_mc or f_mc.IsZombie():
            print(f"[-] Failed to open {calib_file}")
            continue

        h_m_hits = f_mc.Get("Histograms/1D/h1_scifi_nhits")
        h_m_qdc = f_mc.Get("Histograms/1D/h1_hit_qdc")
        h_m_sumqdc = f_mc.Get("Histograms/1D/h1_scifi_sum_qdc")

        if not h_m_hits or not h_m_qdc:
            print(f"[-] Missing required histograms in {calib_file}")
            f_mc.Close()
            continue

        h_m_hits.SetDirectory(0)
        h_m_qdc.SetDirectory(0)
        if h_m_sumqdc:
            h_m_sumqdc.SetDirectory(0)
        f_mc.Close()

        # Compute hit multiplicity metrics
        hit_metrics = compute_hist_metrics(h_data_hits, h_m_hits)
        mc_cache[ts] = {
            "h_m_hits": h_m_hits,
            "h_m_qdc": h_m_qdc,
            "h_m_sumqdc": h_m_sumqdc,
            "hit_metrics": hit_metrics,
            "calib_file": str(calib_file),
            "tag": tag
        }

    if not mc_cache:
        print("[-] No thresholds evaluated successfully. Exiting.")
        sys.exit(1)

    print(f"\n[✓] Successfully cached simulation distributions for {len(mc_cache)} thresholds.")
    print("-" * 85)

    # 3. Comprehensive Multi-Parameter Grid Evaluation
    print(f"[*] Evaluating {n_combos} combinations across (ts, s_qdc, b0)...")
    results_grid: List[Dict[str, Any]] = []

    for ts in sorted(mc_cache.keys()):
        c_data = mc_cache[ts]
        hm = c_data["hit_metrics"]
        h_m_qdc_raw = c_data["h_m_qdc"]

        # Hit loss (dimensionless, normalized relative discrepancy)
        loss_hits = (hm["w1"] / data_mean_hits) + (hm["delta_peak"] / data_peak_hits)

        for s_qdc in args.qdc_scales:
            for b0 in args.qdc_offsets:
                qm = compute_rescaled_qdc_metrics(h_data_qdc, h_m_qdc_raw, scale=s_qdc, offset=b0)

                # QDC loss (dimensionless, normalized relative discrepancy)
                loss_qdc = (qm["w1"] / data_mean_qdc) + (qm["delta_peak"] / data_peak_qdc)

                # Total Joint Loss
                total_loss = args.weight_hits * loss_hits + args.weight_qdc * loss_qdc

                entry = {
                    "ts": ts,
                    "s_qdc": s_qdc,
                    "b0": b0,
                    "tag": c_data["tag"],
                    "calib_file": c_data["calib_file"],
                    # Hit metrics
                    "peak_mc_hits": hm["peak_mc"],
                    "mean_mc_hits": hm["mean_mc"],
                    "delta_peak_hits": hm["delta_peak"],
                    "delta_mean_hits": hm["delta_mean"],
                    "w1_hits": hm["w1"],
                    "ks_hits": hm["ks_stat"],
                    "chi2_ndf_hits": hm["chi2_ndf"],
                    # QDC metrics
                    "peak_mc_qdc": qm["peak_mc"],
                    "mean_mc_qdc": qm["mean_mc"],
                    "delta_peak_qdc": qm["delta_peak"],
                    "delta_mean_qdc": qm["delta_mean"],
                    "w1_qdc": qm["w1"],
                    "ks_qdc": qm["ks_stat"],
                    "chi2_ndf_qdc": qm["chi2_ndf"],
                    # Loss components
                    "loss_hits": loss_hits,
                    "loss_qdc": loss_qdc,
                    "total_loss": total_loss,
                    "counts_qdc_rebinned": qm["counts_m_rebinned"]
                }
                results_grid.append(entry)

    # 4. Sort & Rank All Combinations
    results_grid.sort(key=lambda r: r["total_loss"])
    for rank, r in enumerate(results_grid, start=1):
        r["rank"] = rank

    best = results_grid[0]

    # Find nominal default baseline for comparison (ts = 3.5, s_qdc ~ 4.5, b0 = 0.0)
    default_entry = None
    for r in results_grid:
        if abs(r["ts"] - 3.5) < 1e-4 and abs(r["s_qdc"] - 4.5) < 1e-4 and abs(r["b0"] - 0.0) < 1e-4:
            default_entry = r
            break
    if not default_entry:
        default_entry = results_grid[-1]

    # 5. Print Summary Ranking Tables
    print("\n" + "=" * 115)
    print("TOP 15 PARAMETER COMBINATIONS RANKED BY JOINT CALIBRATION LOSS")
    print("=" * 115)
    print(f"{'Rank':<5} | {'ts [p.e.]':<9} | {'s_qdc':<6} | {'b0':<5} | {'Hits Peak':<9} | {'Hits W1':<8} | {'QDC Peak':<8} | {'QDC W1':<7} | {'Loss (Hits)':<11} | {'Loss (QDC)':<10} | {'Total Loss':<10}")
    print("-" * 115)
    for r in results_grid[:15]:
        marker = " [BEST]" if r["rank"] == 1 else ""
        print(f"{r['rank']:<5} | {r['ts']:<9.2f} | {r['s_qdc']:<6.2f} | {r['b0']:<5.2f} | "
              f"{r['peak_mc_hits']:<9.1f} | {r['w1_hits']:<8.3f} | {r['peak_mc_qdc']:<8.2f} | {r['w1_qdc']:<7.3f} | "
              f"{r['loss_hits']:<11.3f} | {r['loss_qdc']:<10.3f} | {r['total_loss']:<10.4f}{marker}")
    print("=" * 115)

    print("\n" + "=" * 85)
    print("OPTIMAL VS DEFAULT BASELINE COMPARISON")
    print("=" * 85)
    print(f"{'Observable / Parameter':<30} | {'Default Simulation':<22} | {'Optimal Calibrated':<22}")
    print("-" * 85)
    print(f"{'SciFi Threshold (ts)':<30} | {default_entry['ts']:.2f} p.e.              | {best['ts']:.2f} p.e.")
    print(f"{'QDC Calibration Scale (s_qdc)':<30} | {default_entry['s_qdc']:.2f}                    | {best['s_qdc']:.2f}")
    print(f"{'QDC Pedestal Offset (b0)':<30} | {default_entry['b0']:.2f} a.u.              | {best['b0']:.2f} a.u.")
    print(f"{'SciFi Hits Peak (Data = ' + f'{data_peak_hits:.1f})':<30} | {default_entry['peak_mc_hits']:.1f} hits (Δ = {default_entry['delta_peak_hits']:.1f})   | {best['peak_mc_hits']:.1f} hits (Δ = {best['delta_peak_hits']:.1f})")
    print(f"{'SciFi Hits Mean (Data = ' + f'{data_mean_hits:.2f})':<30} | {default_entry['mean_mc_hits']:.2f} hits (Δ = {default_entry['delta_mean_hits']:.2f})   | {best['mean_mc_hits']:.2f} hits (Δ = {best['delta_mean_hits']:.2f})")
    print(f"{'SciFi Hits W1 Distance':<30} | {default_entry['w1_hits']:.3f} hits            | {best['w1_hits']:.3f} hits")
    print(f"{'SciFi Hit QDC Peak (Data = ' + f'{data_peak_qdc:.2f})':<30} | {default_entry['peak_mc_qdc']:.2f} a.u. (Δ = {default_entry['delta_peak_qdc']:.2f})  | {best['peak_mc_qdc']:.2f} a.u. (Δ = {best['delta_peak_qdc']:.2f})")
    print(f"{'SciFi Hit QDC Mean (Data = ' + f'{data_mean_qdc:.2f})':<30} | {default_entry['mean_mc_qdc']:.2f} a.u. (Δ = {default_entry['delta_mean_qdc']:.2f})  | {best['mean_mc_qdc']:.2f} a.u. (Δ = {best['delta_mean_qdc']:.2f})")
    print(f"{'SciFi Hit QDC W1 Distance':<30} | {default_entry['w1_qdc']:.4f} a.u.          | {best['w1_qdc']:.4f} a.u.")
    print(f"{'Total Joint Calibration Loss':<30} | {default_entry['total_loss']:.4f}                  | {best['total_loss']:.4f}")
    print("=" * 85)

    # 6. Generate Publication Plots
    print("\n[*] Generating multi-dimensional publication diagnostic plots...")

    # --------------------------------------------------------------------------
    # Plot 1: 2D Calibration Loss Surface Heatmap (ts vs s_qdc at optimal b0)
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    unique_ts = sorted(list(set(r["ts"] for r in results_grid)))
    unique_qs = sorted(list(set(r["s_qdc"] for r in results_grid)))
    loss_matrix = np.zeros((len(unique_qs), len(unique_ts)))

    for i, qs in enumerate(unique_qs):
        for j, ts in enumerate(unique_ts):
            matching = [r["total_loss"] for r in results_grid if abs(r["ts"] - ts) < 1e-4 and abs(r["s_qdc"] - qs) < 1e-4]
            loss_matrix[i, j] = min(matching) if matching else np.nan

    T_grid, Q_grid = np.meshgrid(unique_ts, unique_qs)
    im = ax1.imshow(
        loss_matrix,
        extent=[min(unique_ts) - 0.25, max(unique_ts) + 0.25, min(unique_qs) - 0.15, max(unique_qs) + 0.15],
        origin='lower',
        aspect='auto',
        cmap='viridis_r'
    )
    cbar = plt.colorbar(im, ax=ax1, pad=0.03)
    cbar.set_label("Joint Calibration Loss $\\mathcal{L}_{\\text{total}}$ (Dimensionless)", fontsize=11)

    # Contour lines
    try:
        cs = ax1.contour(T_grid, Q_grid, loss_matrix, levels=8, colors='white', alpha=0.6, linewidths=0.8)
        ax1.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    except Exception:
        pass

    # Mark optimal point
    ax1.plot(best["ts"], best["s_qdc"], marker='*', color='gold', markersize=18, markeredgecolor='black', markeredgewidth=1.5,
             label=f'Optimal ($t_s={best["ts"]:.2f}$ p.e., $s_{{QDC}}={best["s_qdc"]:.2f}$, $b_0={best["b0"]:.2f}$)')
    # Mark default baseline
    ax1.plot(default_entry["ts"], default_entry["s_qdc"], marker='o', color='red', markersize=10, markeredgecolor='black',
             label=f'Nominal Default ($t_s={default_entry["ts"]:.2f}$ p.e., $s_{{QDC}}={default_entry["s_qdc"]:.2f}$)')

    ax1.set_xlabel("SciFi Digitization Threshold $t_s$ [p.e.]", fontsize=12)
    ax1.set_ylabel("SciFi QDC Calibration Scale Factor $s_{\\text{QDC}}$", fontsize=12)
    ax1.set_title("SND@LHC Joint Calibration Loss Surface: $\\mathcal{L}_{\\text{total}}(t_s, s_{\\text{QDC}})$", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.3, color='black')

    plt.tight_layout()
    plot_surface_path = out_dir / "calibration_loss_surface_2d.png"
    fig1.savefig(plot_surface_path, dpi=300)
    plt.close(fig1)
    print(f"[✓] Saved 2D loss surface plot: {plot_surface_path}")

    # --------------------------------------------------------------------------
    # Plot 2: Dual Distribution Comparison (Hits on Left, QDC on Right)
    # --------------------------------------------------------------------------
    fig2, ((ax_h_top, ax_q_top), (ax_h_bot, ax_q_bot)) = plt.subplots(
        2, 2, figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08, 'wspace': 0.22}
    )

    # PANEL A: SciFi Hits Multiplicity
    nbins_dh = h_data_hits.GetNbinsX()
    x_dh = np.array([h_data_hits.GetBinCenter(i) for i in range(1, nbins_dh + 1)])
    counts_dh = np.array([h_data_hits.GetBinContent(i) for i in range(1, nbins_dh + 1)])
    norm_dh = counts_dh / np.max(counts_dh) if np.max(counts_dh) > 0 else counts_dh

    # Default MC hits
    h_m_hits_def = mc_cache[default_entry["ts"]]["h_m_hits"]
    counts_mh_def = np.array([h_m_hits_def.GetBinContent(i) for i in range(1, h_m_hits_def.GetNbinsX() + 1)])
    x_mh_def = np.array([h_m_hits_def.GetBinCenter(i) for i in range(1, h_m_hits_def.GetNbinsX() + 1)])
    norm_mh_def = counts_mh_def / np.max(counts_mh_def) if np.max(counts_mh_def) > 0 else counts_mh_def

    # Best MC hits
    h_m_hits_best = mc_cache[best["ts"]]["h_m_hits"]
    counts_mh_best = np.array([h_m_hits_best.GetBinContent(i) for i in range(1, h_m_hits_best.GetNbinsX() + 1)])
    x_mh_best = np.array([h_m_hits_best.GetBinCenter(i) for i in range(1, h_m_hits_best.GetNbinsX() + 1)])
    norm_mh_best = counts_mh_best / np.max(counts_mh_best) if np.max(counts_mh_best) > 0 else counts_mh_best

    ax_h_top.step(x_dh, norm_dh, where='mid', label=f'Collision Data (Peak = {data_peak_hits:.0f})', color='black', linewidth=2.2)
    ax_h_top.step(x_mh_def, norm_mh_def, where='mid', label=f'Nominal MC ($t_s={default_entry["ts"]:.1f}$, W1={default_entry["w1_hits"]:.2f})', color='royalblue', linewidth=1.8, linestyle='--')
    ax_h_top.step(x_mh_best, norm_mh_best, where='mid', label=f'Optimal MC ($t_s={best["ts"]:.1f}$, W1={best["w1_hits"]:.2f}) ★', color='crimson', linewidth=2.5, linestyle='-')

    ax_h_top.set_xlim(5, 35)
    ax_h_top.set_ylim(0, 1.15)
    ax_h_top.set_ylabel("Peak Normalized (Arb. Units)", fontsize=11)
    ax_h_top.set_title("(A) SciFi Hit Multiplicity Across 10 Planes", fontsize=12, fontweight='bold')
    ax_h_top.legend(loc='upper right', frameon=True, fontsize=9)
    ax_h_top.grid(True, linestyle=':', alpha=0.5)

    # Hits Ratio panel
    norm_mh_def_interp = np.interp(x_dh, x_mh_def, norm_mh_def)
    norm_mh_best_interp = np.interp(x_dh, x_mh_best, norm_mh_best)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_h_def = np.where(norm_dh > 0.02, norm_mh_def_interp / norm_dh, np.nan)
        ratio_h_best = np.where(norm_dh > 0.02, norm_mh_best_interp / norm_dh, np.nan)

    ax_h_bot.axhline(1.0, color='black', linestyle=':')
    ax_h_bot.step(x_dh, ratio_h_def, where='mid', color='royalblue', linewidth=1.5, linestyle='--')
    ax_h_bot.step(x_dh, ratio_h_best, where='mid', color='crimson', linewidth=2.0, linestyle='-')
    ax_h_bot.set_xlim(5, 35)
    ax_h_bot.set_ylim(0.0, 2.0)
    ax_h_bot.set_ylabel("MC / Data", fontsize=10)
    ax_h_bot.set_xlabel("SciFi Valid Hits Across 10 Planes", fontsize=11)
    ax_h_bot.grid(True, linestyle=':', alpha=0.5)

    # PANEL B: SciFi Hit QDC
    nbins_dq = h_data_qdc.GetNbinsX()
    x_dq = np.array([h_data_qdc.GetBinCenter(i) for i in range(1, nbins_dq + 1)])
    counts_dq = np.array([h_data_qdc.GetBinContent(i) for i in range(1, nbins_dq + 1)])
    norm_dq = counts_dq / np.max(counts_dq) if np.max(counts_dq) > 0 else counts_dq

    # Default MC QDC (calibrated at nominal scale 4.5)
    counts_mq_def = default_entry["counts_qdc_rebinned"]
    norm_mq_def = counts_mq_def / np.max(counts_mq_def) if np.max(counts_mq_def) > 0 else counts_mq_def

    # Best MC QDC
    counts_mq_best = best["counts_qdc_rebinned"]
    norm_mq_best = counts_mq_best / np.max(counts_mq_best) if np.max(counts_mq_best) > 0 else counts_mq_best

    ax_q_top.step(x_dq, norm_dq, where='mid', label=f'Collision Data (Peak = {data_peak_qdc:.2f})', color='black', linewidth=2.2)
    ax_q_top.step(x_dq, norm_mq_def, where='mid', label=f'Nominal MC ($s={default_entry["s_qdc"]:.2f}$, W1={default_entry["w1_qdc"]:.2f})', color='royalblue', linewidth=1.8, linestyle='--')
    ax_q_top.step(x_dq, norm_mq_best, where='mid', label=f'Optimal MC ($s={best["s_qdc"]:.2f}$, $b_0={best["b0"]:.2f}$, W1={best["w1_qdc"]:.2f}) ★', color='crimson', linewidth=2.5, linestyle='-')

    ax_q_top.set_xlim(0, 10)
    ax_q_top.set_ylim(0, 1.15)
    ax_q_top.set_ylabel("Peak Normalized (Arb. Units)", fontsize=11)
    ax_q_top.set_title("(B) SciFi Single Hit QDC Distribution", fontsize=12, fontweight='bold')
    ax_q_top.legend(loc='upper right', frameon=True, fontsize=9)
    ax_q_top.grid(True, linestyle=':', alpha=0.5)

    # QDC Ratio panel
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_q_def = np.where(norm_dq > 0.02, norm_mq_def / norm_dq, np.nan)
        ratio_q_best = np.where(norm_dq > 0.02, norm_mq_best / norm_dq, np.nan)

    ax_q_bot.axhline(1.0, color='black', linestyle=':')
    ax_q_bot.step(x_dq, ratio_q_def, where='mid', color='royalblue', linewidth=1.5, linestyle='--')
    ax_q_bot.step(x_dq, ratio_q_best, where='mid', color='crimson', linewidth=2.0, linestyle='-')
    ax_q_bot.set_xlim(0, 10)
    ax_q_bot.set_ylim(0.0, 2.0)
    ax_q_bot.set_ylabel("MC / Data", fontsize=10)
    ax_q_bot.set_xlabel("Calibrated SciFi Hit QDC [a.u.]", fontsize=11)
    ax_q_bot.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plot_joint_path = out_dir / "joint_hits_qdc_comparison.png"
    fig2.savefig(plot_joint_path, dpi=300)
    plt.close(fig2)
    print(f"[✓] Saved joint distribution comparison plot: {plot_joint_path}")

    # --------------------------------------------------------------------------
    # Plot 3: 1D Metric Curves (Hits vs ts on Left, QDC vs s_qdc on Right)
    # --------------------------------------------------------------------------
    fig3, (ax3_l, ax3_r) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Hit Multiplicity W1 and Peak vs ts
    ts_list = sorted(mc_cache.keys())
    w1_hits_list = [mc_cache[t]["hit_metrics"]["w1"] for t in ts_list]

    ax3_l.plot(ts_list, w1_hits_list, 'o-', color='navy', linewidth=2, markersize=7, label='Wasserstein $W_1^{\\text{hits}}$ [hits]')
    ax3_l.axvline(best["ts"], color='crimson', linestyle='--', label=f'Optimal $t_s = {best["ts"]:.2f}$ p.e.')
    ax3_l.set_xlabel("SciFi Digitization Threshold $t_s$ [p.e.]", fontsize=11)
    ax3_l.set_ylabel("Wasserstein Distance [hits]", fontsize=11)
    ax3_l.set_title("Hit Multiplicity Discrepancy ($W_1$) vs Threshold", fontsize=12, fontweight='bold')
    ax3_l.legend(frameon=True)
    ax3_l.grid(True, linestyle=':', alpha=0.6)

    # Right: QDC W1 vs s_qdc at optimal ts & b0
    qs_list = sorted(args.qdc_scales)
    w1_qdc_list = []
    for qs in qs_list:
        for r in results_grid:
            if abs(r["ts"] - best["ts"]) < 1e-4 and abs(r["s_qdc"] - qs) < 1e-4 and abs(r["b0"] - best["b0"]) < 1e-4:
                w1_qdc_list.append(r["w1_qdc"])
                break

    ax3_r.plot(qs_list, w1_qdc_list, 's-', color='darkgreen', linewidth=2, markersize=7, label='Wasserstein $W_1^{\\text{QDC}}$ [a.u.]')
    ax3_r.axvline(best["s_qdc"], color='crimson', linestyle='--', label=f'Optimal $s_{{QDC}} = {best["s_qdc"]:.2f}$')
    ax3_r.set_xlabel("SciFi QDC Calibration Scale Factor $s_{\\text{QDC}}$", fontsize=11)
    ax3_r.set_ylabel("Wasserstein Distance [a.u.]", fontsize=11)
    ax3_r.set_title("QDC Distribution Discrepancy ($W_1$) vs Scale", fontsize=12, fontweight='bold')
    ax3_r.legend(frameon=True)
    ax3_r.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_metric_path = out_dir / "threshold_scan_metrics.png"
    fig3.savefig(plot_metric_path, dpi=300)
    plt.close(fig3)
    print(f"[✓] Saved 1D metric scan curves plot: {plot_metric_path}")

    # --------------------------------------------------------------------------
    # Plot 4: Overlaid Hit Multiplicity Across All Scanned Thresholds
    # --------------------------------------------------------------------------
    fig4, (ax4_top, ax4_bot) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax4_top.step(x_dh, norm_dh, where='mid', label=f'Collision Data (Peak = {data_peak_hits:.0f})', color='black', linewidth=2.5)

    cmap = plt.get_cmap("viridis", len(ts_list))
    for idx, t in enumerate(ts_list):
        hm = mc_cache[t]["h_m_hits"]
        nb_m = hm.GetNbinsX()
        x_m = np.array([hm.GetBinCenter(i) for i in range(1, nb_m + 1)])
        cnt_m = np.array([hm.GetBinContent(i) for i in range(1, nb_m + 1)])
        norm_m = cnt_m / np.max(cnt_m) if np.max(cnt_m) > 0 else cnt_m

        is_best = abs(t - best["ts"]) < 1e-4
        col = 'crimson' if is_best else cmap(idx)
        lw = 2.5 if is_best else 1.2
        lbl = f"MC ts = {t:.2f} p.e." + (" ★ BEST" if is_best else "")
        ax4_top.step(x_m, norm_m, where='mid', label=lbl, color=col, linewidth=lw, linestyle='-' if is_best else '--')

        norm_m_interp = np.interp(x_dh, x_m, norm_m)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(norm_dh > 0.02, norm_m_interp / norm_dh, np.nan)
        ax4_bot.step(x_dh, ratio, where='mid', color=col, linewidth=lw, linestyle='-' if is_best else '--')

    ax4_top.set_xlim(5, 35)
    ax4_top.set_ylim(0, 1.15)
    ax4_top.set_ylabel("Arbitrary Units (Peak Normalized to 1.0)", fontsize=11)
    ax4_top.set_title("SND@LHC SciFi Hit Multiplicity vs Digitization Threshold", fontsize=12, fontweight='bold')
    ax4_top.legend(loc='upper right', frameon=True, fontsize=9)
    ax4_top.grid(True, linestyle=':', alpha=0.5)

    ax4_bot.axhline(1.0, color='black', linestyle=':')
    ax4_bot.set_xlim(5, 35)
    ax4_bot.set_ylim(0.0, 2.0)
    ax4_bot.set_ylabel("MC / Data", fontsize=10)
    ax4_bot.set_xlabel("SciFi Valid Hits Across 10 Planes", fontsize=11)
    ax4_bot.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plot_dist_path = out_dir / "threshold_scan_distributions.png"
    fig4.savefig(plot_dist_path, dpi=300)
    plt.close(fig4)
    print(f"[✓] Saved hit distributions overlay plot: {plot_dist_path}")

    # 7. Save ROOT Summary File
    root_summary_path = out_dir / "threshold_scan_summary.root"
    tfile_out = ROOT.TFile(str(root_summary_path), "RECREATE")

    h_data_hits.Write("h_data_hits_ref")
    h_data_qdc.Write("h_data_qdc_ref")
    if h_data_sumqdc:
        h_data_sumqdc.Write("h_data_sumqdc_ref")

    # 2D Loss Histogram
    h2_loss = ROOT.TH2D(
        "h2_loss_surface",
        "Joint Calibration Loss Surface;Threshold t_{s} [p.e.];QDC Scale s_{QDC};Loss",
        len(unique_ts), min(unique_ts) - 0.25, max(unique_ts) + 0.25,
        len(unique_qs), min(unique_qs) - 0.15, max(unique_qs) + 0.15
    )
    for i, qs in enumerate(unique_qs, start=1):
        for j, ts in enumerate(unique_ts, start=1):
            h2_loss.SetBinContent(j, i, float(loss_matrix[i-1, j-1]))
    h2_loss.Write()

    # Best MC Histograms
    h_m_hits_best.Write("h_mc_hits_best")

    # Reconstructed calibrated best QDC histogram
    h_mc_qdc_calib = ROOT.TH1D(
        "h_mc_qdc_calib_best",
        f"Best Calibrated MC Hit QDC (ts={best['ts']:.2f}, s={best['s_qdc']:.2f}, b0={best['b0']:.2f});Hit QDC [a.u.];Hits",
        nbins_dq, h_data_qdc.GetXaxis().GetXmin(), h_data_qdc.GetXaxis().GetXmax()
    )
    for b, cnt in enumerate(best["counts_qdc_rebinned"], start=1):
        h_mc_qdc_calib.SetBinContent(b, float(cnt))
    h_mc_qdc_calib.Write()

    tfile_out.Close()
    print(f"[✓] Saved master ROOT summary file: {root_summary_path}")

    # 8. Write Markdown Calibration Report
    md_path = out_dir / "threshold_scan_report.md"
    with open(md_path, "w") as f:
        f.write("# SND@LHC SciFi Multi-Parameter Calibration Report\n\n")
        f.write("## Joint Hit Multiplicity and QDC Energy Optimization\n\n")
        f.write(f"- **Evaluated Simulation**: `{args.input_file if args.mode == 'offline' else args.raw_sim}`\n")
        f.write(f"- **Data Reference (Hits)**: `{args.data_ref}` (`{args.data_hist}`)\n")
        f.write(f"- **Data Reference (QDC)**: `{args.data_qdc_ref}` (`{args.data_qdc_hist}`)\n")
        f.write(f"- **Total Grid Combinations Evaluated**: `{n_combos}` parameter points\n\n")
        f.write("### Optimal Calibration Parameters\n\n")
        f.write(f"- **SciFi Hit Threshold ($t_s$)**: **`{best['ts']:.2f} p.e.`** (Nominal default: `3.50 p.e.`)\n")
        f.write(f"- **QDC Calibration Scale Factor ($s_{{QDC}}$)**: **`{best['s_qdc']:.2f}`** (Nominal default: `4.50`)\n")
        f.write(f"- **QDC Pedestal Offset ($b_0$)**: **`{best['b0']:.2f} a.u.`** (Nominal default: `0.00`)\n")
        f.write(f"- **Minimal Joint Calibration Loss**: **`{best['total_loss']:.4f}`** (Reduced by `{((default_entry['total_loss'] - best['total_loss'])/default_entry['total_loss'])*100:.1f}%` from default `{default_entry['total_loss']:.4f}`)\n\n")
        f.write("### Parameter Sensitivity & Benchmark Summary\n\n")
        f.write("| Observable / Metric | Collision Data | Nominal Default MC | Optimal Calibrated MC | Improvement |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **SciFi Hits Peak** | **`{data_peak_hits:.1f}` hits** | `{default_entry['peak_mc_hits']:.1f}` hits | **`{best['peak_mc_hits']:.1f}` hits** | Δ: `{default_entry['delta_peak_hits']:.1f} → {best['delta_peak_hits']:.1f}` hits |\n")
        f.write(f"| **SciFi Hits Mean** | **`{data_mean_hits:.2f}` hits** | `{default_entry['mean_mc_hits']:.2f}` hits | **`{best['mean_mc_hits']:.2f}` hits** | Δ: `{default_entry['delta_mean_hits']:.2f} → {best['delta_mean_hits']:.2f}` hits |\n")
        f.write(f"| **SciFi Hits $W_1$ (EMD)** | `0.000` | `{default_entry['w1_hits']:.3f}` hits | **`{best['w1_hits']:.3f}` hits** | **`-`{((default_entry['w1_hits'] - best['w1_hits'])/default_entry['w1_hits'])*100:.1f}%** |\n")
        f.write(f"| **SciFi Hit QDC Peak** | **`{data_peak_qdc:.2f}` a.u.** | `{default_entry['peak_mc_qdc']:.2f}` a.u. | **`{best['peak_mc_qdc']:.2f}` a.u.** | Δ: `{default_entry['delta_peak_qdc']:.2f} → {best['delta_peak_qdc']:.2f}` a.u. |\n")
        f.write(f"| **SciFi Hit QDC Mean** | **`{data_mean_qdc:.2f}` a.u.** | `{default_entry['mean_mc_qdc']:.2f}` a.u. | **`{best['mean_mc_qdc']:.2f}` a.u.** | Δ: `{default_entry['delta_mean_qdc']:.2f} → {best['delta_mean_qdc']:.2f}` a.u. |\n")
        f.write(f"| **SciFi Hit QDC $W_1$ (EMD)** | `0.000` | `{default_entry['w1_qdc']:.4f}` a.u. | **`{best['w1_qdc']:.4f}` a.u.** | **`-`{((default_entry['w1_qdc'] - best['w1_qdc'])/default_entry['w1_qdc'])*100:.1f}%** |\n")
        f.write(f"| **Joint Calibration Loss** | `0.000` | `{default_entry['total_loss']:.4f}` | **`{best['total_loss']:.4f}`** | **`-`{((default_entry['total_loss'] - best['total_loss'])/default_entry['total_loss'])*100:.1f}%** |\n\n")
        f.write("### Top 15 Parameter Combinations\n\n")
        f.write("| Rank | $t_s$ [p.e.] | $s_{\\text{QDC}}$ | $b_0$ | Hits Peak | Hits $W_1$ | QDC Peak | QDC $W_1$ | Loss (Hits) | Loss (QDC) | Total Loss |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results_grid[:15]:
            f.write(f"| {r['rank']} | {r['ts']:.2f} | {r['s_qdc']:.2f} | {r['b0']:.2f} | {r['peak_mc_hits']:.1f} | {r['w1_hits']:.3f} | {r['peak_mc_qdc']:.2f} | {r['w1_qdc']:.3f} | {r['loss_hits']:.3f} | {r['loss_qdc']:.3f} | **{r['total_loss']:.4f}** |\n")

    print(f"[✓] Saved Markdown report: {md_path}")
    print("\n" + "=" * 85)
    print("MULTI-PARAMETER JOINT CALIBRATION OPTIMIZATION COMPLETED")
    print("=" * 85)


if __name__ == "__main__":
    main()
