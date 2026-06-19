# sndTridentCrossCheck

SND@LHC Muonic trident cross check with event cuts and Hough transform selection.

## Project Structure

* **`tridentSample.cxx`**: The main executable with the cut flows.
* **`cuts/`**: All implemented event cuts
  * SciFi and DS fiducial volume cuts
  * Veto and DS activity cuts
  * Event time delta cuts
  * Dense tracking / trident hits selection cuts
* **`CMakeLists.txt`**

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
