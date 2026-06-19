# sndTridentCrossCheck

A selection and filtering framework for Trident events and neutrino candidates in the SND@LHC experiment.

## Project Structure

* **`tridentSample.cxx`**: The main executable source code that sets up the cut flows, processes input files, runs the event loop, and fills/saves selection histrograms.
* **`cuts/`**: A library of modular selections (`sndBaseCut` implementations) including:
  * SciFi and DS fiducial volume cuts
  * Veto and DS activity cuts
  * Event time delta cuts
  * Dense tracking / trident hits selection cuts
* **`CMakeLists.txt`**: Standard CMake build configuration that generates the dictionary and outputs PCM, rootmap, and binary targets under `build/`.

---

## Build Instructions

To compile the codebase, ensure that you have your `SNDSW` environment set up (`SNDSW_ROOT` environment variable is required) along with standard ROOT libraries.

1. Create a build directory:
   ```bash
   mkdir -p build && cd build
   ```
2. Configure with CMake:
   ```bash
   cmake ..
   ```
3. Build the targets:
   ```bash
   make
   ```

This will produce:
* `build/lib/libtrident_cuts.so` (and associated ROOT `.pcm` / `.rootmap` files)
* `build/bin/tridentSample` (main executable)

---

## Running the Trident Filter

Run the compiled executable with three arguments:
```bash
./build/bin/tridentSample <input_file_name_or_regexp> <output_file_name> <cut_set>
```

### Cut Set Options (`<cut_set>`)
Choose from the following integer modes:
* **`0` (Stage 1 Selection)**: Average SciFi and DS fiducial, veto cut, no hits in 1st & 2nd SciFi planes, vertex not in 5th wall, consecutive SciFi planes, and DS activity constraint.
* **`1` (No Veto / SciFi 1st Layer)**: Similar to stage 1 but without veto constraints.
* **`2` (FV Sideband)**: Fiducial volume sideband selections.
* **`3` (Include Walls 2 and 5)**: Custom wall selection allowing SciFi station hits.
* **`4` (Nue Filter)**: Electron neutrino candidate filter.
* **`6` (Trident Selection)**: Demands veto hits, minimum SciFi planes, trident hits (9/3 or 3/9 distribution), total hits limits, and track/signal density cuts.
