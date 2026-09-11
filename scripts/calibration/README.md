# SND@LHC SciFi Calibration & Threshold Optimization Tools

This directory contains automated calibration and simulation tuning utilities for the **SND@LHC** experiment.

---

## Tools in This Directory

### 1. `scan_digitization_thresholds.py`
**Automated Multi-Threshold Grid Scanner & Optimizer**
- Systematically re-digitizes Monte Carlo simulations across a configurable grid of SciFi photo-electron thresholds ($t_s$).
- Automatically runs:
  1. `run_digiSND.py -cpp -ts <threshold>`
  2. `run_TrackSelections.py -sc 1 -ht -st`
  3. `extract_scifi_calibration_histograms.py --skip-scifi-hits-cut`
- Evaluates statistical similarity metrics against Collision Data:
  - **Wasserstein Distance ($W_1$)**: Physical discrepancy in number of hits.
  - **Peak Alignment ($\Delta \text{Peak}$)**: Modal hit multiplicity match.
  - **Mean Multiplicity Difference ($\Delta \mu$)**.
  - **Kolmogorov-Smirnov Statistic ($D_{\text{KS}}$ & $p$-value)**.
  - **Normalized Shape $\chi^2/\text{ndf}$**.
- Generates:
  - Ranked summary table identifying the optimal threshold $t_s^*$.
  - Comparison plot: Data vs MC overlay at all thresholds (`threshold_scan_distributions.png`).
  - Metric response curves: $W_1$ and Peak vs $t_s$ (`threshold_scan_metrics.png`).
  - ROOT summary file with `TGraph` curves (`threshold_scan_summary.root`).
  - Markdown report (`threshold_scan_report.md`).

#### Usage Example:
```bash
python scripts/calibration/scan_digitization_thresholds.py \
    -f /eos/experiment/sndlhc/MonteCarlo/from_sndprodmachine/run200/sndLHC.Ntuple-TGeant4_boost100.0.root \
    -g /eos/experiment/sndlhc/MonteCarlo/from_sndprodmachine/run200/geofile_full.Ntuple-TGeant4_boost100.0.root \
    -ts 3.0 3.5 4.0 4.5 5.0 5.5 6.0 \
    -n 2000 \
    -j 8 \
    -o plots/calibration_scan
```

---

### 2. `extract_scifi_calibration_histograms.py`
**Single-Muon Calibration Histogram & Event Skimmer**
- Multi-threaded RDataFrame analysis engine using compiled C++ [`MuonCalibrationProcessor`](file:///afs/cern.ch/work/i/idioniso/sndMuTri/analysis/MuonCalibrationProcessor.cxx).
- Filters clean passing single muons through sequential tracking and DS matching cuts (Stages 1 through 5).
- By default skips the final $[10, 35]$ SciFi hits cut (`--skip-scifi-hits-cut`) to preserve full hit spectra for calibration and threshold studies.
- Preserves full event trees and branches in the output ROOT file (`--save-events`).

#### Usage Example:
```bash
# Run on Collision Data:
python scripts/calibration/extract_scifi_calibration_histograms.py \
    -i "/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-0_100_8329_muonReco.root" \
    -o "plots/calibration/run_8329_clean_muons.root" \
    -j 8

# Run on Monte Carlo:
python scripts/calibration/extract_scifi_calibration_histograms.py \
    -i "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200_trks.root" \
    -o "plots/calibration/trimuon_clean_muons.root" \
    -j 8
```

### Running Threshold Scans on Already-Digitized Files (`--mode offline`)

Because raw Geant4 files are typically cleaned up, and already-digitized files (`..._trks.root`) contain all SciFi hits with their QDC/signals, you can run `scan_digitization_thresholds.py` directly on already-digitized files without re-running Geant4, digitization, or tracking:

```bash
python scripts/calibration/scan_digitization_thresholds.py \
    --mode offline \
    -i "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200_trks.root" \
    -ts 3.5 3.8 4.0 4.2 4.5 4.8 5.0 5.5 \
    -o "plots/calibration_scan" \
    -j 8
```
This evaluates the entire threshold grid in under 2 minutes, generates quantitative ranking metrics ($W_1$, $\Delta\text{Peak}$, $\Delta\mu$, KS distance, $\chi^2/\text{ndf}$), and saves diagnostic plots (`threshold_scan_distributions.png` and `threshold_scan_metrics.png`).

