# SND@LHC SciFi Multi-Parameter Calibration Report

## Joint Hit Multiplicity and QDC Energy Optimization

- **Evaluated Simulation**: `/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200_trks.root`
- **Data Reference (Hits)**: `plots/passing_muon_selection.root` (`HitDistributions/SciFi_Hits/h_sf_hits_data_stage6`)
- **Data Reference (QDC)**: `plots/scifi_calibration_run8329.root` (`Histograms/1D/h1_hit_qdc`)
- **Total Grid Combinations Evaluated**: `140` parameter points

### Optimal Calibration Parameters

- **SciFi Hit Threshold ($t_s$)**: **`5.00 p.e.`** (Nominal default: `3.50 p.e.`)
- **QDC Calibration Scale Factor ($s_{QDC}$)**: **`4.20`** (Nominal default: `4.50`)
- **QDC Pedestal Offset ($b_0$)**: **`0.40 a.u.`** (Nominal default: `0.00`)
- **Minimal Joint Calibration Loss**: **`0.7956`** (Reduced by `40.8%` from default `1.3444`)

### Parameter Sensitivity & Benchmark Summary

| Observable / Metric | Collision Data | Nominal Default MC | Optimal Calibrated MC | Improvement |
|:---|:---:|:---:|:---:|:---:|
| **SciFi Hits Peak** | **`12.0` hits** | `16.5` hits | **`13.5` hits** | Δ: `4.5 → 1.5` hits |
| **SciFi Hits Mean** | **`15.29` hits** | `22.36` hits | **`21.00` hits** | Δ: `7.07 → 5.71` hits |
| **SciFi Hits $W_1$ (EMD)** | `0.000` | `8.124` hits | **`6.785` hits** | **`-`16.5%** |
| **SciFi Hit QDC Peak** | **`1.10` a.u.** | `0.82` a.u. | **`1.12` a.u.** | Δ: `0.28 → 0.02` a.u. |
| **SciFi Hit QDC Mean** | **`2.18` a.u.** | `2.08` a.u. | **`2.41` a.u.** | Δ: `0.10 → 0.23` a.u. |
| **SciFi Hit QDC $W_1$ (EMD)** | `0.000` | `0.4034` a.u. | **`0.4556` a.u.** | **`-`-12.9%** |
| **Joint Calibration Loss** | `0.000` | `1.3444` | **`0.7956`** | **`-`40.8%** |

### Top 15 Parameter Combinations

| Rank | $t_s$ [p.e.] | $s_{\text{QDC}}$ | $b_0$ | Hits Peak | Hits $W_1$ | QDC Peak | QDC $W_1$ | Loss (Hits) | Loss (QDC) | Total Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 5.00 | 4.20 | 0.40 | 13.5 | 6.785 | 1.12 | 0.456 | 0.569 | 0.227 | **0.7956** |
| 2 | 5.00 | 4.50 | 0.20 | 13.5 | 6.785 | 1.09 | 0.480 | 0.569 | 0.231 | **0.7994** |
| 3 | 5.00 | 4.80 | -0.20 | 13.5 | 6.785 | 1.10 | 0.520 | 0.569 | 0.243 | **0.8115** |
| 4 | 5.00 | 4.50 | 0.00 | 13.5 | 6.785 | 1.13 | 0.492 | 0.569 | 0.256 | **0.8251** |
| 5 | 4.50 | 4.20 | 0.20 | 14.5 | 6.284 | 1.12 | 0.424 | 0.619 | 0.212 | **0.8315** |
| 6 | 4.50 | 4.20 | 0.40 | 14.5 | 6.284 | 1.07 | 0.407 | 0.619 | 0.213 | **0.8327** |
| 7 | 4.50 | 4.50 | 0.00 | 14.5 | 6.284 | 1.09 | 0.449 | 0.619 | 0.217 | **0.8360** |
| 8 | 5.00 | 4.50 | 0.40 | 13.5 | 6.785 | 1.04 | 0.473 | 0.569 | 0.268 | **0.8366** |
| 9 | 5.00 | 4.80 | 0.00 | 13.5 | 6.785 | 1.06 | 0.513 | 0.569 | 0.270 | **0.8389** |
| 10 | 5.00 | 5.10 | -0.40 | 13.5 | 6.785 | 1.08 | 0.553 | 0.569 | 0.274 | **0.8428** |
| 11 | 5.00 | 4.20 | 0.20 | 13.5 | 6.785 | 1.17 | 0.475 | 0.569 | 0.279 | **0.8479** |
| 12 | 4.50 | 4.80 | -0.40 | 14.5 | 6.284 | 1.10 | 0.494 | 0.619 | 0.231 | **0.8501** |
| 13 | 5.00 | 4.80 | -0.40 | 13.5 | 6.785 | 1.15 | 0.528 | 0.569 | 0.284 | **0.8533** |
| 14 | 4.50 | 4.50 | -0.20 | 14.5 | 6.284 | 1.13 | 0.461 | 0.619 | 0.242 | **0.8615** |
| 15 | 4.00 | 4.20 | 0.20 | 14.5 | 7.157 | 1.12 | 0.380 | 0.677 | 0.192 | **0.8684** |
