# sndTridentCrossCheck

SND@LHC Muonic trident cross check with event cuts and Hough transform selection.

## Project Structure

### (1) Cut Flow Implementation (C++)

* **`tridentSample.cxx`**: The main executable with the cut flows.
* **`cuts/`**: All implemented event cuts
  * SciFi and DS fiducial volume cuts
  * Veto and DS activity cuts
  * Event time delta cuts
  * Dense tracking / trident hits selection cuts
* **`CMakeLists.txt`**

### (2) Hough Selection Implementation (Python)

Located in [hough.py](file:///afs/cern.ch/work/i/idioniso/sndTridentCrossCheck/pythonHelpers/hough.py), this module provides Python helpers to reconstruct tracks using the Hough transform with vertex constraints:

* **[run_hough_transform](file:///afs/cern.ch/work/i/idioniso/sndTridentCrossCheck/pythonHelpers/hough.py#L5)**:
  * Extracts positions, channel dimensions, and orientations for active SciFi hits (`event.Digi_ScifiHits`).
  * Separates horizontal and vertical projections (`XZ` and `YZ`) and applies randomized Hough track fits (`fit_randomize`) from `SndlhcMuonReco`.
  * **Vertex Constraint Filtering**: Reconstructed tracks are cross-checked for intersections (vertices) against other existing tracks. If the reconstructed intersection `z_vertex` falls outside a user-defined interval `[z_vtx_min, z_vtx_max]`, the track is filtered out as a conflict.
  * Masks hits associated with validated tracks to avoid double-counting in subsequent Hough search iterations.
* **[get_line_params](file:///afs/cern.ch/work/i/idioniso/sndTridentCrossCheck/pythonHelpers/hough.py#L116)**:
  * Safely returns slope and intercept parameters for reconstructed lines in each projection.

---

## Prerequisites

- **SNDSW**: The SND@LHC software envronment.

Required environment variables:

- `SNDSW_ROOT`
- `ROOTSYS`

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/idionisov/sndTridentCrossCheck.git
    cd sndTridentCrossCheck
    ```

2. **Environment Setup**:
   `alienv enter sndsw/latest`

3. **Build**:

    ```bash
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    cd ..
    ```

---

## Running the Trident Filter

Standard event cut executable:

```bash
./build/bin/tridentSample <input_file_name_or_regexp> <output_file_name> <cut_set>
```

The new **Cut Set Option (`<cut_set> = 6`)** is for Trident Selection. It demands veto hits, minimum SciFi planes, trident hits (9/3 or 3/9 distribution), total hits limits, and track/signal density cuts.
