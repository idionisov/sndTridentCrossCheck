# Data Exploration, Plotting, and Analysis Guide

A complete technical manual and reference guide for the analysis classes introduced in `sndMuTri`:
- **`DataManager`**: Multi-file discovery, tree resolving, geometry access, ROOT `RDataFrame` (`data.rdf()`), and flat Python pandas DataFrame (`data.df()`).
- **`ProgressPrinter`**: Thread-safe periodic status reporting (every 30 seconds) with zero event-loop overhead.
- **Histogram Configuration Wrapper**: Centralized YAML-driven observable booking (`CALIB_CONFIGS_1D`, `2D`, `Profiles`).
- **`MuonCalibrationProcessor` & `IP1Filter`**: Track-based clean single muon selection and hit-level unrolling.
- **`PassingMuonTruthProcessor`**: Monte Carlo truth genealogy and passing muon classification.

---

## Table of Contents
- [1. DataManager Reference](#1-datamanager-reference)
  - [A. Constructor Signatures](#a-constructor-signatures)
  - [B. Complete Method and Property Reference](#b-complete-method-and-property-reference)
  - [C. `rdf()` vs `df()`: RDataFrame vs Pandas DataFrame](#c-rdf-vs-df-rdataframe-vs-pandas-dataframe)
  - [D. Usage Modes (C++, Direct PyROOT, Python Wrapper)](#d-usage-modes)
- [2. ProgressPrinter Reference](#2-progressprinter-reference)
  - [A. Constructor Signatures & Parameters](#a-constructor-signatures--parameters)
  - [B. Method & Accessor Reference](#b-method--accessor-reference)
  - [C. Usage Modes (C++, Direct PyROOT, Python Wrapper)](#c-usage-modes-1)
- [3. Histogram Configuration & Booking Wrapper](#3-histogram-configuration--booking-wrapper)
  - [A. Tuple Schemas](#a-tuple-schemas)
  - [B. Available Pre-Configured Sets](#b-available-pre-configured-sets)
  - [C. Usage with and without Wrapper](#c-usage-with-and-without-wrapper)
- [4. MuonCalibrationProcessor & IP1Filter](#4-muoncalibrationprocessor--ip1filter)
  - [A. MuonCalibrationConfig Fields](#a-muoncalibrationconfig-fields)
  - [B. MuonCalibrationMetrics Fields & Accessors](#b-muoncalibrationmetrics-fields--accessors)
  - [C. IP1Filter for Collision Bunch Crossings](#c-ip1filter-for-collision-bunch-crossings)
  - [D. Usage Modes](#d-usage-modes)
- [5. PassingMuonTruthProcessor (Monte Carlo)](#5-passingmuontruthprocessor-monte-carlo)
  - [A. Truth Categories](#a-truth-categories)
  - [B. Config & Info Structures](#b-config--info-structures)
  - [C. Usage in RDataFrame](#c-usage-in-rdataframe)
- [6. Complete Practical Workflows](#6-complete-practical-workflows)
  - [Workflow 1: Quick Interactive Dataset Exploration & Metadata Query](#workflow-1-quick-interactive-dataset-exploration--metadata-query)
  - [Workflow 2: Run-Based Analysis with Automatic Geometry Initialization](#workflow-2-run-based-analysis-with-automatic-geometry-initialization)
  - [Workflow 3: Multi-Threaded Calibration with 30s Status Updates (`data.rdf()`)](#workflow-3-multi-threaded-calibration-with-30s-status-updates-datardf)
  - [Workflow 4: Extracting Flattened Python DataFrames (`data.df()`) for Pandas & Seaborn](#workflow-4-extracting-flattened-python-dataframes-datadf-for-pandas--seaborn)
  - [Workflow 5: Direct PyROOT Execution Without Any Python Wrapper](#workflow-5-direct-pyroot-execution-without-any-python-wrapper)
  - [Workflow 6: Pure C++ Application](#workflow-6-pure-c-application)
  - [Workflow 7: Monte Carlo Truth vs Reconstruction Efficiency & Purity](#workflow-7-monte-carlo-truth-vs-reconstruction-efficiency--purity)

---

## 1. DataManager Reference

`DataManager` encapsulates file discovery, tree auto-detection (`rawConv` vs `cbmsim`), geometry integration, high-performance ROOT `RDataFrame` generation (`data.rdf()`), and flat Python pandas DataFrame generation (`data.df()`).

### A. Constructor Signatures

#### C++ (`snd::DataManager`)
```cpp
// 1. Default constructor
DataManager();

// 2. From file path or glob pattern
explicit DataManager(const std::string& source,
                     const std::string& treeName = "",
                     int numThreads = 0);

// 3. From a vector of file paths
explicit DataManager(const std::vector<std::string>& sources,
                     const std::string& treeName = "",
                     int numThreads = 0);

// 4. From an LHC run number
explicit DataManager(int runNumber,
                     int nFiles = -1,
                     const std::string& csvDataPath = "",
                     const std::string& treeName = "",
                     int numThreads = 0);

// 5. Adopt existing TChain (with optional ownership transfer)
explicit DataManager(TChain* chain, bool ownChain = true, int numThreads = 0);
explicit DataManager(std::unique_ptr<TChain> chain, int numThreads = 0);
```

#### Python Wrapper (`snd.DataManager`)
```python
DataManager(
    source: Optional[Union[str, Sequence[str], int, ROOT.TChain]] = None,
    tree_name: Optional[str] = None,
    num_threads: Optional[int] = None,
    run: Optional[int] = None,
    n_files: int = -1,
    csv_data_path: Optional[str] = None,
    geo_path: Optional[str] = None,
    csv_geo_path: Optional[str] = None,
    init_geo: bool = False,
    load_libraries: bool = True,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `str`, `List[str]`, `int`, `TChain` | `None` | Path pattern, file list, run number, or active `TChain` |
| `tree_name` | `str` | `""` | Tree name. If empty, auto-detects `rawConv` or `cbmsim` |
| `num_threads` | `int` | `0` | Threads for implicit multi-threading (`0` = keep current setting) |
| `run` | `int` | `None` | Explicit LHC run number (e.g. `8329`) |
| `n_files` | `int` | `-1` | Maximum files to read when using run-based construction (`-1` = all) |
| `csv_data_path` | `str` | `None` | Custom path to SND@LHC data CSV catalog |
| `geo_path` | `str` | `None` | Custom path to detector geometry ROOT file |
| `csv_geo_path` | `str` | `None` | Custom path to geometry catalog CSV |
| `init_geo` | `bool` | `False` | Automatically load SND@LHC geometry upon initialization |
| `load_libraries` | `bool` | `True` | Automatically load `libtrident_analysis.so` and dependencies |

---

### B. Complete Method and Property Reference

#### File & Tree Inspection
- **`tree_name`** (`str`) / **`GetTreeName()`** (`std::string`): The active TTree name.
- **`files`** (`List[str]`) / **`GetFiles()`** (`const std::vector<std::string>&`): List of all resolved ROOT file paths.
- **`num_files`** (`int`) / **`GetNumFiles()`** (`size_t`): Number of resolved files.
- **`entries`** (`int`) / **`GetEntries()`** (`Long64_t`): Total entries in dataset.
- **`__len__()`** (`int`): Returns total entries (`len(data)`).
- **`branch_names`** (`Set[str]`) / **`GetBranchNamesList()`** (`std::vector<std::string>`): Names of all branches in the active tree.
- **`has_branch(name: str)`** (`bool`) / **`HasBranch(name)`** (`bool`): Checks if a branch exists.
- **`has_event_header()`** (`bool`) / **`HasEventHeader()`** (`bool`): Checks if `EventHeader` or `EventHeader.` exists.
- **`is_mc`** (`bool`) / **`IsMC()`** (`bool`): Checks if `MCTrack` is present.

#### Run, Fill & Luminosity
- **`run_number`** (`int`) / **`GetRunNumber()`** (`int`): The active LHC run number.
- **`fill_number`** (`int`) / **`GetFillNumber()`** (`int`): The active LHC fill number.
- **`lumi_path`** (`str`) / **`GetLumiPath(lumiDir)`** (`std::string`): Path to ATLAS luminosity file.
- **`data_base_path`** (`str`) / **`GetDataBasePath(csvPath)`** (`std::string`): Path to run storage folder on EOS.
- **`GetLumi(start_ts, end_ts, lumiDir, cached)`** (`double`): Integrated luminosity in $\text{fb}^{-1}$.
- **`GetLumiFast()`** / **`GetLumiExact()`** (`double`): Quick boundary or exact traversed luminosity.

#### Geometry Access
- **`geo_path`** (`str`) / **`GetGeoPath()`** (`std::string`): Path to active geofile.
- **`has_geometry`** (`bool`) / **`HasGeometry()`** (`bool`): Returns `true` if geometry is loaded into memory.
- **`init_geometry(geo_path, csv_file_path)`** / **`InitGeometry(...)`**: Instantiates `Scifi` and `MuFilter` detector objects.
- **`scifi`** / **`GetScifi()`**: Pointer to `Scifi` detector class.
- **`mufilter`** / **`GetMuFilter()`**: Pointer to `MuFilter` detector class.
- **`geometry`** / **`GetGeometry()`**: Returns `(scifi, mufilter)`.
- **`configuration`** / **`GetConfiguration()`**: Cached `snd::Configuration` pointer.

#### ROOT RDataFrame Access (`rdf`)
- **`rdf(range_limit=None, progress=False, every_seconds=30.0, every_events=0, progress_label=None)`** (`ROOT.RDataFrame`):
  Constructs and returns the multi-threaded ROOT `RDataFrame`.
  - In C++: `data.rdf()` or `data.GetRDF()`.
- **`get_rdataframe(range_limit=None)`**: Alias for `rdf(range_limit)`.
- **`rdataframe`**: Property alias for `data.rdf()`.
- **`get_chain()`** (`ROOT.TChain*`): Raw underlying `TChain` pointer.

#### Flattened Python DataFrame Access (`df`)
- **`df(columns=None, range_limit=None, query=None, rdf=None, progress=False, every_seconds=30.0, progress_label=None)`** (`pandas.DataFrame`):
  Converts dataset branches and defined columns into a flattened Python `pandas.DataFrame`.
  - `columns`: Optional list of branch or column names to extract (e.g. `["reco_chi2", "n_hits"]`).
  - `range_limit`: Maximum number of entries to extract.
  - `query`: Optional string selection filter (e.g. `"reco_chi2 < 10 && n_hits > 15"`).
  - `rdf`: Optional pre-configured `RDataFrame` node (with custom `.Define()` columns) to extract from.
  - `progress`: If `True`, prints 30-second progress status while extracting.

---

### C. `rdf()` vs `df()`: RDataFrame vs Pandas DataFrame

| Feature | `data.rdf(...)` | `data.df(...)` |
| :--- | :--- | :--- |
| **Return Type** | `ROOT.RDataFrame` | `pandas.DataFrame` |
| **Execution** | Lazy computational graph | Eager materialized table |
| **Target Use Case** | Large-scale event processing, C++ filters, ROOT histograms | Tabular analysis, Python ML (Scikit-Learn/XGBoost), Seaborn plotting |
| **Memory** | Minimal (streamed chunk-by-chunk) | In-memory RAM table |
| **Branch Support** | Full (raw `TClonesArray`, objects, scalar branches) | Flattened scalar & 1D array columns |

---

### D. Usage Modes

#### 1. Python with High-Level Wrapper
```python
from snd import DataManager

data = DataManager("/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-*.root", num_threads=8)

# A. Get ROOT RDataFrame for fast C++ event processing
rdf = data.rdf(progress=True, every_seconds=30.0, progress_label="Run8329")
h_hits = rdf.Define("nhits", "Digi_ScifiHits.GetEntries()").Histo1D("nhits")

# B. Get flattened pandas DataFrame for tabular analysis
rdf_with_cols = rdf.Define("nhits", "Digi_ScifiHits.GetEntries()").Define("is_clean", "nhits > 10")
pdf = data.df(columns=["nhits", "is_clean"], rdf=rdf_with_cols, range_limit=10000)
print(pdf.describe())
```

#### 2. Python Direct PyROOT (Without Python Wrapper)
```python
import ROOT
from snd import load_trident_libraries

load_trident_libraries()

# Call C++ DataManager directly
data = ROOT.snd.DataManager("/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-*.root", "", 8)
print("Entries:", data.GetEntries())
print("Active Tree:", data.GetTreeName())

# Call rdf() in C++
rdf = data.rdf()
```

#### 3. Pure C++
```cpp
#include "DataManager.h"
#include <iostream>

int main() {
    snd::DataManager data("/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-*.root", "", 8);
    std::cout << "Files: " << data.GetNumFiles() << " | Entries: " << data.GetEntries() << std::endl;

    ROOT::RDataFrame rdf = data.rdf();
    auto count = rdf.Count();
    std::cout << "Count: " << count.GetValue() << std::endl;
    return 0;
}
```

---

## 2. ProgressPrinter Reference

`ProgressPrinter` monitors `RDataFrame` event loops in multi-threaded workflows, calculating processing speed, ETA, and elapsed time with lock-free atomic counters.

### A. Constructor Signatures & Parameters

#### C++ (`snd::trident::ProgressPrinter`)
```cpp
ProgressPrinter(uint64_t every_n = 0,
                uint64_t total_events = 0,
                double every_sec = 30.0,
                const std::string& label = "Progress",
                bool log_mode = true);
```

#### Python Wrapper (`snd.ProgressPrinter`)
```python
ProgressPrinter(
    total_events: int = 0,
    every_seconds: float = 30.0,
    every_events: int = 0,
    label: str = "Progress",
    log_mode: bool = True,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `total_events` | `int` | `0` | Total target events. If $>0$, computes percentage and ETA. |
| `every_seconds`| `float` | `30.0` | Minimum elapsed seconds between status updates. |
| `every_events` | `int` | `0` | Event-count trigger (e.g. `50000`). `0` = purely timer-based. |
| `label` | `str` | `"Progress"` | Prefix label displayed in status line. |
| `log_mode` | `bool` | `True` | Outputs distinct newline log entries suitable for batch jobs and terminal. |

---

### B. Method & Accessor Reference

- **`operator()(ULong64_t entry = 0)`** (`bool`): RDataFrame functor call operator. Returns `true` for all entries.
- **`GetProcessed()`** (`uint64_t`) / **`processed`** (`int`): Current total events processed.
- **`GetTotal()`** (`uint64_t`) / **`total`** (`int`): Target total events.
- **`SetTotal(total: int)`**: Updates target total events.
- **`GetElapsedSeconds()`** (`double`) / **`elapsed_seconds`** (`float`): Elapsed wall time in seconds.
- **`GetRate()`** (`double`) / **`rate`** (`float`): Throughput rate in events per second.
- **`PrintSummary()`** / **`print_summary()`**: Prints the final completion summary line.
- **`Reset()`** / **`reset()`**: Resets event counters, timer, and completion state.

---

### C. Usage Modes

#### 1. Python with Wrapper
```python
from snd import add_progress_printer

# Returns updated DataFrame and printer object
rdf, printer = add_progress_printer(rdf, total_events=data.entries, every_seconds=30.0, label="SciFi")

res = rdf.Count().GetValue()
printer.print_summary()
```

#### 2. Python Direct PyROOT (Without Python Wrapper)
```python
import ROOT
from snd import load_trident_libraries

load_trident_libraries()

# Instantiate C++ class directly
printer = ROOT.snd.trident.ProgressPrinter(0, 500000, 30.0, "MyAnalysis", True)
rdf = rdf.Filter(printer, ["rdfentry_"])

count = rdf.Count()
print("Total:", count.GetValue())
printer.PrintSummary()
```

#### 3. Pure C++
```cpp
#include "ProgressPrinter.h"

snd::trident::ProgressPrinter printer(0, total_events, 30.0, "SciFi Calib", true);
auto rdf_monitored = rdf.Filter(printer, {"rdfentry_"});

auto count = rdf_monitored.Count();
std::cout << "Done: " << count.GetValue() << std::endl;
printer.PrintSummary();
```

---

## 3. Histogram Configuration & Booking Wrapper

The histogram system couples the master YAML file (`config/scifi_calibration_config.yaml`) with `config.calibration_histograms_config` to eliminate hardcoded histogram definitions.

### A. Tuple Schemas

- **1D Histograms**: `(name, title_and_axes, nbins, xmin, xmax, column_name)`
- **2D Histograms**: `(name, title_and_axes, xbins, xmin, xmax, ybins, ymin, ymax, x_col, y_col)`
- **Profile Histograms**: `(name, title_and_axes, xbins, xmin, xmax, ymin, ymax, x_col, y_col)`

### B. Available Pre-Configured Sets

Imported from `config.calibration_histograms_config`:
- **`CALIB_CONFIGS_1D`**: 27 standard distributions including hit QDC, plane sums, track $\chi^2/\text{ndf}$, angles, and matching residuals.
- **`CALIB_CONFIGS_2D`**: 5 standard 2D correlations (e.g. QDC vs fiber distance, track matching $\Delta x$ vs $\Delta y$).
- **`CALIB_PROFILE_CONFIGS`**: 3 TProfile graphs for fiber attenuation ($QDC(d)$).

### C. Usage with and without Wrapper

#### Python (With Configuration Wrapper)
```python
from config.calibration_histograms_config import CALIB_CONFIGS_1D

h1_books = {}
for name, title, nbins, xmin, xmax, col in CALIB_CONFIGS_1D:
    if rdf.HasColumn(col):
        h1_books[name] = rdf.Histo1D((name, title, nbins, xmin, xmax), col)
```

#### Python Direct PyROOT (Without Wrapper)
```python
import ROOT

model = ROOT.RDF.TH1DModel("h_qdc", "Hit QDC;QDC;Hits", 100, 0, 100)
h_qdc = rdf.Histo1D(model, "hit_qdc")
```

---

## 4. MuonCalibrationProcessor & IP1Filter

### A. `MuonCalibrationConfig` Fields

| Field Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `scifi_track_type` | `int` | `11` | SciFi Hough transform track type |
| `ds_track_type` | `int` | `13` | Downstream MuFilter Hough track type |
| `chi2_max_scifi` | `double` | `10.0` | Maximum allowed SciFi track fit $\chi^2/\text{ndf}$ |
| `chi2_max_ds` | `double` | `10.0` | Maximum allowed DS track fit $\chi^2/\text{ndf}$ |
| `max_slope` | `double` | `0.05` | Maximum angular slope $|\text{slope}| < 0.05\text{ rad}$ |
| `fiducial_margin` | `double` | `1.5` | Safety border margin from detector envelope (cm) |
| `z_match` | `double` | `430.0` | Reference plane $z$ for track extrapolation (cm) |
| `pos_match_max` | `double` | `3.0` | Maximum transverse residual $\Delta R \le 3.0\text{ cm}$ |
| `angle_match_max` | `double` | `0.015` | Maximum collinearity angle $\Delta\theta \le 0.015\text{ rad}$ |
| `ds_match_slope_max` | `double` | `0.04` | Max slope difference between SF and DS |
| `scifi_hits_min` | `int` | `10` | Minimum SciFi hit multiplicity (MIP band) |
| `scifi_hits_max` | `int` | `35` | Maximum SciFi hit multiplicity |
| `ds_hits_min` | `int` | `2` | Minimum Downstream hits (penetration) |
| `scifi_qdc_min` | `double` | `0.0` | Offline SciFi threshold in p.e./QDC |

---

### B. `MuonCalibrationMetrics` Fields & Accessors

When processed via `rdf.Define("calib", processor, [...])`, the returned `calib` struct contains:

#### Cut Decision Flags (`bool`)
- `is_clean`: Master flag (true if event passes all active baseline cuts).
- `pass_cut_single_scifi_track`: Exactly 1 SciFi Hough track (Type 11).
- `pass_cut_ds_track`: Exactly 1 DS Hough track (Type 13).
- `pass_cut_chi2_scifi`, `pass_cut_chi2_ds`: Track $\chi^2/\text{ndf}$ quality.
- `pass_cut_slope_scifi`, `pass_cut_slope_ds`: Angular slope criteria.
- `pass_cut_fiducial_430`: Fiducial border check at $z = 430\text{ cm}$.
- `pass_cut_pos_match_430`, `pass_cut_angle_match`, `pass_cut_ds_match`: Spatial and angular matching.

#### Kinematic Scalars
- `reco_chi2_scifi`, `reco_chi2_ds` (`double`)
- `reco_slope_x`, `reco_slope_y` (`double`)
- `match_dr_430`, `match_dtheta` (`double`)

#### Vector Observables for Hit-Level Analysis
- `calib.get_hit_qdc()` $\rightarrow$ `std::vector<double>` (charge for all hits on track)
- `calib.get_hit_distance()` $\rightarrow$ `std::vector<double>` (reconstructed distance to SiPM [cm])
- `calib.get_hit_station()` $\rightarrow$ `std::vector<double>` (station index 1 to 5)
- `calib.get_hit_orientation()` $\rightarrow$ `std::vector<double>` (0 = Horizontal, 1 = Vertical)
- `calib.get_plane_qdc()` $\rightarrow$ `std::vector<double>` (summed QDC per plane)
- `calib.get_plane_nhits()` $\rightarrow$ `std::vector<double>` (hit multiplicity per plane)

---

### C. `IP1Filter` for Collision Bunch Crossings

In raw data (`rawConv`), collision bunch crossings must be checked on the event header branch `'EventHeader.'`:

```python
# IP1Filter takes SNDLHCEventHeader& and returns header.isIP1()
rdf = rdf.Define("is_ip1", ROOT.snd.trident.IP1Filter(), ["EventHeader."])
rdf = rdf.Filter("is_ip1", "IP1 Bunch Crossing")
```

---

### D. Usage Modes

#### Python (PyROOT)
```python
import ROOT
from snd import load_trident_libraries

load_trident_libraries()

cfg = ROOT.snd.trident.MuonCalibrationConfig()
proc = ROOT.snd.trident.MuonCalibrationProcessor(cfg)

rdf = rdf.Define("calib", proc, ["Reco_MuonTracks", "Digi_ScifiHits", "Digi_MuFilterHits"])
rdf_clean = rdf.Filter("calib.is_clean")
rdf_clean = rdf_clean.Define("hit_qdc", "calib.get_hit_qdc()")
```

#### Pure C++
```cpp
#include "MuonCalibrationProcessor.h"

snd::trident::MuonCalibrationConfig cfg;
snd::trident::MuonCalibrationProcessor proc(cfg);

auto rdf_calib = rdf.Define("calib", proc, {"Reco_MuonTracks", "Digi_ScifiHits", "Digi_MuFilterHits"})
                    .Filter("calib.is_clean", "Clean Single Muon")
                    .Define("hit_qdc", "calib.get_hit_qdc()");
```

---

## 5. PassingMuonTruthProcessor (Monte Carlo)

### A. Truth Categories

| Category ID | Enum Name | Description |
| :---: | :--- | :--- |
| `0` | `kNoDetectorHit` | No activity in SciFi or MuFilter |
| `1` | `kCleanPassingMuon` | Single clean through-going muon (Signal) |
| `2` | `kCatastrophicEMShower` | Muon initiates hard radiative cascade ($E_{sec} > 2.0\text{ GeV}$) |
| `3` | `kHadronicShower` | Muon initiates nuclear / deep-inelastic shower |
| `4` | `kMultiMuon` | Multiple muons traversing detector simultaneously |
| `5` | `kHaloShowerNoMuon` | Secondary shower enters detector without primary muon |
| `6` | `kStoppingScatteredMuon`| Muon enters SciFi but stops, decays, or scatters out before DS |

### B. Config & Info Structures

`PassingMuonTruthConfig`:
- `shower_energy_threshold`: Energy threshold for hard cascades (default: `2.0` GeV).
- `max_scifi_points_clean`: Upper limit of Geant4 SciFi points for clean MIP (default: `150`).
- `fiducial_margin`: Boundary margin in cm (default: `1.5`).

`PassingMuonTruthInfo`:
- `category_id`: Integer ID (0 to 6).
- `category_name`: String description.
- `is_signal`: Boolean (`category == kCleanPassingMuon`).
- `primary_muon_p`, `primary_muon_pt`, `primary_muon_eta`: Primary kinematics.
- `n_scifi_points`, `n_mufi_points`: Geant4 point multiplicities.

### C. Usage in RDataFrame

```python
truth_cfg = ROOT.snd.trident.PassingMuonTruthConfig()
truth_proc = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)

rdf = rdf.Define("truth", truth_proc, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
rdf_signal = rdf.Filter("truth.is_signal", "True Single Passing Muon")
```

---

## 6. Complete Practical Workflows

### Workflow 1: Quick Interactive Dataset Exploration & Metadata Query

```python
from snd import DataManager

# 1. Inspect run 8329
data = DataManager(run=8329, n_files=5, num_threads=4)

print("Files:", data.num_files)
print("Entries:", data.entries)
print("Active Tree:", data.tree_name)
print("Run Number:", data.run_number)
print("Fill Number:", data.fill_number)
print("Has Muon Tracks:", data.has_branch("Reco_MuonTracks"))
print("Has Digi SciFi:", data.has_branch("Digi_ScifiHits"))

# 2. Get quick 5,000-event ROOT RDataFrame sample
rdf = data.rdf(range_limit=5000)
count = rdf.Count().GetValue()
print(f"Sample count: {count}")
```

---

### Workflow 2: Run-Based Analysis with Automatic Geometry Initialization

```python
import ROOT
from snd import DataManager

# Initialize run 8329 and load SND@LHC detector geometry
data = DataManager(run=8329, n_files=10, init_geo=True, num_threads=4)

# Access geometry classes directly
scifi_det = data.scifi
mufilter_det = data.mufilter
print(f"Geometry initialized: {data.has_geometry}")
print(f"SciFi Stations: {scifi_det.GetConf().size()}")

rdf = data.rdf(progress=True, every_seconds=30.0, progress_label="GeoAnalysis")
```

---

### Workflow 3: Multi-Threaded Calibration with 30s Status Updates (`data.rdf()`)

```python
import ROOT
from snd import DataManager, add_progress_printer
from config.calibration_histograms_config import CALIB_CONFIGS_1D

ROOT.EnableImplicitMT(8)

data = DataManager("/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-*.root")
rdf = data.rdf()

# 1. Attach progress tracking (every 30 seconds)
rdf, printer = add_progress_printer(rdf, total_events=data.entries, every_seconds=30.0, label="SciFi Calib")

# 2. Setup Calibration Processor
calib_cfg = ROOT.snd.trident.MuonCalibrationConfig()
processor = ROOT.snd.trident.MuonCalibrationProcessor(calib_cfg)

rdf = rdf.Define("calib", processor, ["Reco_MuonTracks", "Digi_ScifiHits", "Digi_MuFilterHits"])
rdf_clean = rdf.Filter("calib.is_clean", "Clean Passing Muon")

# 3. Unroll vector columns
rdf_clean = (
    rdf_clean
    .Define("hit_qdc", "calib.get_hit_qdc()")
    .Define("hit_distance", "calib.get_hit_distance()")
    .Define("hit_station", "calib.get_hit_station()")
)

# 4. Book all 1D histograms from YAML
books = {}
for name, title, nbins, xmin, xmax, col in CALIB_CONFIGS_1D:
    if rdf_clean.HasColumn(col):
        books[name] = rdf_clean.Histo1D((name, title, nbins, xmin, xmax), col)

count_book = rdf_clean.Count()

# 5. Execute event loop
selected = count_book.GetValue()
printer.PrintSummary()

print(f"Selected clean muons: {selected:,}")
print(f"Hit QDC entries: {books['h1_hit_qdc'].GetValue().GetEntries():,}")
```

---

### Workflow 4: Extracting Flattened Python DataFrames (`data.df()`) for Pandas & Seaborn

```python
import ROOT
from snd import DataManager

data = DataManager("/eos/user/i/idioniso/1_Data/Tracks/run_008329/sndsw_raw-0_0_8329_muonReco.root")

# 1. Prepare RDataFrame with custom defined analysis quantities
rdf = (
    data.rdf()
    .Define("n_scifi_hits", "Digi_ScifiHits.GetEntries()")
    .Define("n_ds_hits", "Digi_MuFilterHits.GetEntries()")
    .Define("n_tracks", "Reco_MuonTracks.GetEntries()")
)

# 2. Extract into a clean, flat pandas.DataFrame with 30s progress
pdf = data.df(
    columns=["n_scifi_hits", "n_ds_hits", "n_tracks"],
    rdf=rdf,
    query="n_tracks > 0",
    range_limit=50000,
    progress=True,
    every_seconds=30.0
)

# 3. Immediate tabular manipulation & correlation plotting
print(pdf.head())
print(pdf.corr())

# Plot with seaborn or matplotlib
# import seaborn as sns
# sns.scatterplot(data=pdf, x="n_scifi_hits", y="n_ds_hits")
```

---

### Workflow 5: Direct PyROOT Execution Without Any Python Wrapper

```python
import ROOT

# 1. Load C++ shared libraries manually
ROOT.gSystem.Load("build/lib/libtrident_analysis.so")

# 2. Use C++ DataManager directly
data = ROOT.snd.DataManager("/eos/.../run_008329/sndsw_raw-0_0_8329_muonReco.root", "", 4)
total_entries = data.GetEntries()

# 3. Use C++ ProgressPrinter directly
printer = ROOT.snd.trident.ProgressPrinter(0, total_entries, 30.0, "DirectPyROOT", True)

rdf = data.rdf()
rdf = rdf.Filter(printer, ["rdfentry_"])

# 4. Use C++ MuonCalibrationProcessor directly
cfg = ROOT.snd.trident.MuonCalibrationConfig()
proc = ROOT.snd.trident.MuonCalibrationProcessor(cfg)

rdf = rdf.Define("calib", proc, ["Reco_MuonTracks", "Digi_ScifiHits", "Digi_MuFilterHits"])
rdf = rdf.Filter("calib.is_clean")

# 5. Trigger
count = rdf.Count()
print("Clean muons:", count.GetValue())
printer.PrintSummary()
```

---

### Workflow 6: Pure C++ Application

```cpp
#include <iostream>
#include <ROOT/RDataFrame.hxx>
#include "DataManager.h"
#include "ProgressPrinter.h"
#include "MuonCalibrationProcessor.h"

int main() {
    ROOT::EnableImplicitMT(8);

    // 1. DataManager
    snd::DataManager data("/eos/.../run_008329/sndsw_raw-*.root", "", 8);
    auto total_events = data.GetEntries();

    // 2. ProgressPrinter
    snd::trident::ProgressPrinter printer(0, total_events, 30.0, "CppWorkflow", true);

    ROOT::RDataFrame rdf = data.rdf();
    auto rdf_monitored = rdf.Filter(printer, {"rdfentry_"});

    // 3. Calibration Processor
    snd::trident::MuonCalibrationConfig cfg;
    snd::trident::MuonCalibrationProcessor proc(cfg);

    auto rdf_clean = rdf_monitored.Define("calib", proc, {"Reco_MuonTracks", "Digi_ScifiHits", "Digi_MuFilterHits"})
                                  .Filter("calib.is_clean", "Clean Single Muon")
                                  .Define("hit_qdc", "calib.get_hit_qdc()");

    auto h_qdc = rdf_clean.Histo1D({"h_qdc", "Hit QDC;QDC;Hits", 100, 0, 100}, "hit_qdc");
    auto count = rdf_clean.Count();

    // 4. Execution
    std::cout << "Selected: " << count.GetValue() << " events." << std::endl;
    std::cout << "Hits: " << h_qdc->GetEntries() << std::endl;
    printer.PrintSummary();

    return 0;
}
```

---

### Workflow 7: Monte Carlo Truth vs Reconstruction Efficiency & Purity

```python
import ROOT
from snd import DataManager, load_trident_libraries

load_trident_libraries()
ROOT.EnableImplicitMT(8)

mc_path = "/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/..._trks.root"
data = DataManager(mc_path)

# Attach progress printer with 30s reporting
rdf = data.rdf(progress=True, every_seconds=30.0, progress_label="MCPurity")

# Truth classification
truth_proc = ROOT.snd.trident.PassingMuonTruthProcessor()
rdf = rdf.Define("truth", truth_proc, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
rdf = rdf.Define("is_true_signal", "truth.category_id == 1")

# Reconstruction selection
calib_proc = ROOT.snd.trident.MuonCalibrationProcessor()
rdf = rdf.Define("calib", calib_proc, ["fittedTracks", "Digi_ScifiHits", "Digi_MuFilterHits"])
rdf = rdf.Define("is_reco_clean", "calib.is_clean")

# Book counters
n_gen_signal = rdf.Filter("is_true_signal").Count()
n_reco_pass = rdf.Filter("is_reco_clean").Count()
n_reco_pass_true_signal = rdf.Filter("is_reco_clean && is_true_signal").Count()

# Evaluate
n_gen = n_gen_signal.GetValue()
n_rec = n_reco_pass.GetValue()
n_pass_sig = n_reco_pass_true_signal.GetValue()

print(f"Generated True Signal Muons : {n_gen:,}")
print(f"Reconstructed Clean Muons   : {n_rec:,}")
print(f"True Signal Passing Cuts    : {n_pass_sig:,}")
print(f"Selection Efficiency        : {100.0 * n_pass_sig / n_gen:.2f} %")
print(f"Selection Purity            : {100.0 * n_pass_sig / n_rec:.2f} %")
```
