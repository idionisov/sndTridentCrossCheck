# SND@LHC Trident Analysis & SciFi Calibration Framework (`sndMuTri`)

A high-performance C++ and Python framework for **SND@LHC** data analysis, focusing on:
1. **Clean Single Passing Muon Selection & SciFi Detector Calibration**: Kinematic and geometric track-based selection, unbiased hit charge (QDC), and fiber attenuation length measurements.
2. **Simulation Tuning & Optimization**: Photo-electron threshold scanning and metric-based calibration against LHC collision data.
3. **Trident Cross-Check & BDT Feature Extraction**: Multi-track reconstruction (Hough transform) and multivariate event selection.

---

## Table of Contents
- [1. Environment & Prerequisites](#1-environment--prerequisites)
- [2. Compilation & Build](#2-compilation--build)
- [3. Key Scripts & How to Run](#3-key-scripts--how-to-run)
  - [A. SciFi Calibration Extraction Engine](#a-scifi-calibration-extraction-engine)
  - [B. Passing Muon Truth & Purity Assessment](#b-passing-muon-truth--purity-assessment)
  - [C. Digitization Threshold Scanner & Optimizer](#c-digitization-threshold-scanner--optimizer)
  - [D. End-to-End Trident Reconstruction Pipeline](#d-end-to-end-trident-reconstruction-pipeline)
- [4. Repository Architecture](#4-repository-architecture)
- [5. Troubleshooting & FAQ](#5-troubleshooting--faq)
- [6. Notes: Quick Data Exploration & Analysis Guide](notes/data_manager_exploration_and_analysis_guide.md)

---

## 1. Environment & Prerequisites

The framework requires the standard CERN `sndsw` environment with **ROOT 6 (C++17, RDataFrame)** and Python 3.9+.

On CERN AlmaLinux 9 / `lxplus`:

```bash
source /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh
alienv enter sndsw/latest
```

Verify that the required environment variables are exported:
```bash
echo $SNDSW_ROOT
echo $ROOTSYS
```

---

## 2. Compilation & Build

The core analysis algorithms and event processors are implemented in C++ and compiled into shared libraries located in `build/lib/`:
- `libtrident_analysis.so`: Contains high-throughput RDataFrame processors (`MuonCalibrationProcessor`, `IP1Filter`, `DataManager`, `TridentTruthProcessor`, `PreselectionProcessor`).
- `libtrident_cuts.so`: C++ cuts framework (`tridentSelection`, `bdtPreselection`).
- `tridentSample`: Standalone C++ binary for rapid event skimming.

### Build Steps

From the repository root (`/afs/cern.ch/work/i/idioniso/sndMuTri`):

```bash
cmake -B build -S . && cmake --build build -j$(nproc)
```

> [!TIP]
> For incremental builds after modifying C++ headers or source files under `analysis/` or `cuts/`, you only need to run:
> ```bash
> cmake --build build -j$(nproc)
> ```

---

## 3. Key Scripts & How to Run

### A. SciFi Calibration Extraction Engine

**Script**: [`scripts/extract_scifi_calibration_histograms.py`](file:///afs/cern.ch/work/i/idioniso/sndMuTri/scripts/extract_scifi_calibration_histograms.py)  
**Configuration**: [`config/scifi_calibration_config.yaml`](file:///afs/cern.ch/work/i/idioniso/sndMuTri/config/scifi_calibration_config.yaml)

A multi-threaded ROOT RDataFrame engine that selects clean through-going single muons and extracts 1D, 2D, and Profile histograms (hit QDC, cluster charges, track kinematics, fiber attenuation vs distance).

#### Configuration-Driven Execution
The script is **100% configuration-driven**. All input patterns, thread counts, cut criteria, and histogram binning are defined in the YAML file.

Run with default configuration (`config/scifi_calibration_config.yaml`):
```bash
python3 scripts/extract_scifi_calibration_histograms.py
```

Run with custom configuration:
```bash
python3 scripts/extract_scifi_calibration_histograms.py -c path/to/custom_config.yaml
```

#### Selection Logic Configured in YAML
- **Stage 0 (`ip1_bunch`)**: Restricts real collision data to colliding bunch crossings (`EventHeader.isIP1()`). Automatically skipped on MC where `EventHeader` is absent.
- **Stage 1 (`single_scifi_track`)**: Exactly 1 reconstructed SciFi Hough track (Type 11).
- **Stage 2 (`ds_track`)**: Exactly 1 reconstructed Downstream MuFilter Hough track (Type 13).
- **Stage 3 (`chi2_quality`)**: Track fit quality $\chi^2/\text{ndf} \le 10$ for both tracks.
- **Stage 4 (`angular_slope`)**: Near-normal incidence ($|\text{slope}| < 0.05\text{ rad}$).
- **Stage 5 (`fiducial_430`)**: Tracks intersect reference plane $z = 430\text{ cm}$ within fiducial boundaries ($\ge 1.5\text{ cm}$ margin).
- **Stage 6 (`ds_match`)**: Spatial matching ($\Delta R \le 3.0\text{ cm}$) and angular collinearity ($\Delta\theta \le 0.015\text{ rad}$) at $z = 430\text{ cm}$.
- **Hit Multiplicity Cuts**: Deliberately bypassed to prevent sculpting/biasing the SciFi response distributions during detector calibration.

---

### B. Passing Muon Truth & Purity Assessment

**Script**: [`scripts/passing_muon_selection.py`](file:///afs/cern.ch/work/i/idioniso/sndMuTri/scripts/passing_muon_selection.py)

Classifies Monte Carlo events against truth information using [`PassingMuonTruthProcessor`](file:///afs/cern.ch/work/i/idioniso/sndMuTri/analysis/PassingMuonTruthProcessor.h) to assess the efficiency and purity of single-muon selection cuts:
- **Truth Categories**: Clean Passing Muon, Stopping Muon, Halo Muon, Catastrophic EM Shower, Hadronic Shower, and Trimuon Signal.
- Outputs detailed yield tables, rejection factors, and purity assessments.

```bash
python3 scripts/passing_muon_selection.py
```

---

### C. Digitization Threshold Scanner & Optimizer

**Script**: [`scripts/calibration/scan_digitization_thresholds.py`](file:///afs/cern.ch/work/i/idioniso/sndMuTri/scripts/calibration/scan_digitization_thresholds.py)

Optimizes the SciFi light yield and photo-electron threshold ($t_s$) in Monte Carlo to accurately reproduce the hit multiplicity and QDC spectra of collision data.

#### Offline Mode (on already reconstructed files, recommended):
```bash
python3 scripts/calibration/scan_digitization_thresholds.py \
    --mode offline \
    -i "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200_trks.root" \
    -ts 3.0 3.5 3.8 4.0 4.2 4.5 5.0 \
    -j 8 \
    -o plots/calibration_scan
```
Generates Wasserstein distance ($W_1$), peak alignment ($\Delta\text{Peak}$), and shape $\chi^2/\text{ndf}$ curves.

---

### D. End-to-End Trident Reconstruction Pipeline

**Driver Script**: [`script.sh`](file:///afs/cern.ch/work/i/idioniso/sndMuTri/script.sh)

Coordinates the multi-stage trident cross-check analysis across large batches:
1. **Stage 1**: Event preselection using compiled `build/bin/tridentSample` (cut set 6 or 7).
2. **Stage 2**: Multi-track Hough reconstruction via `hough_line_reco.py`.
3. **Stage 3**: BDT feature extraction into a unified parquet dataset via `extract_trident_bdt_features.py`.

```bash
chmod +x script.sh
./script.sh
```

---

## 4. Repository Architecture

```text
sndMuTri/
├── CMakeLists.txt
├── README.md
├── script.sh
│
├── analysis/
│   ├── MuonCalibrationProcessor.h / .cxx
│   ├── DataManager.h / .cxx
│   ├── TridentTruthProcessor.h / .cxx
│   ├── PassingMuonTruthProcessor.h / .cxx
│   └── ProgressPrinter.h / .cxx
│
├── cuts/
│   ├── tridentSelection.cxx
│   └── bdtPreselection.cxx
│
├── config/
│   ├── scifi_calibration_config.yaml
│   └── calibration_histograms_config.py
│
├── notes/
│   └── data_manager_exploration_and_analysis_guide.md
│
├── scripts/
│   ├── extract_scifi_calibration_histograms.py
│   ├── passing_muon_selection.py
│   ├── hough_line_reco.py
│   └── calibration/
│       ├── scan_digitization_thresholds.py
│       └── extract_scifi_calibration_histograms.py
│
├── snd/
│   ├── data_manager.py
│   ├── progress.py
│   ├── truth_branches.py
│   └── yield_summary.py
│
└── build/
    ├── bin/
    └── lib/
```

---

## 5. Troubleshooting & FAQ

### `libtrident_analysis.so not found`
Ensure you have built the C++ targets:
```bash
cmake -B build -S . && cmake --build build -j$(nproc)
```
The Python framework automatically searches `build/lib/` for the shared libraries.

### `SNDSW_ROOT environment variable is not set`
Remember to source the SND@LHC environment before building or running:
```bash
source /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh
alienv enter sndsw/latest
```

### `Could not open file /eos/...: Permission denied`
Ensure your CERN Kerberos ticket and AFS/EOS tokens are active:
```bash
kinit <your_cern_username>@CERN.CH
aklog
```
