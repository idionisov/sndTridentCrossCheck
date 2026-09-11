# SND@LHC SciFi Digitization Threshold Calibration Report

- **Evaluated Simulation**: `/eos/experiment/sndlhc/MonteCarlo/from_sndprodmachine/run200/sndLHC.Ntuple-TGeant4_boost100.0.root`
- **Data Reference**: `plots/passing_muon_selection.root` (`HitDistributions/SciFi_Hits/h_sf_hits_data_stage6`)
- **Optimal Threshold**: **`4.50 p.e.`**
- **Minimum Wasserstein Distance**: **`33.095 hits`**
- **Peak Alignment**: MC = `14.5` vs Data = `12.0` (Delta = `2.5`)
- **Mean Multiplicity**: MC = `20.51` vs Data = `15.29` (Delta = `5.23`)

## Quantitative Ranking Table

| Rank | Threshold [p.e.] | Peak (MC) | Mean (MC) | ΔPeak | ΔMean | W1 [hits] | KS Distance | χ²/ndf | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 4.50 | 14.5 | 20.51 | 2.5 | 5.23 | 33.095 | 0.6367 | 94.34 | **BEST** |
| 2 | 4.00 | 14.5 | 21.39 | 2.5 | 6.10 | 34.323 | 0.6585 | 141.33 | Sub-optimal |
| 3 | 3.50 | 16.5 | 22.36 | 4.5 | 7.07 | 35.471 | 0.6793 | 176.26 | Sub-optimal |
