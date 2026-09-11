# SND@LHC SciFi Multi-Parameter Calibration Report

## Joint Hit Multiplicity and QDC Energy Optimization

- **Evaluated Simulation**: `/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200_trks.root`
- **Data Reference (Hits)**: `plots/passing_muon_selection.root` (`HitDistributions/SciFi_Hits/h_sf_hits_data_stage6`)
- **Data Reference (QDC)**: `plots/scifi_calibration_run8329.root` (`Histograms/1D/h1_hit_qdc`)
- **Total Grid Combinations Evaluated**: `280` parameter points

### Optimal Calibration Parameters

- **SciFi Hit Threshold ($t_s$)**: **`6.50 p.e.`** (Nominal default: `3.50 p.e.`)
- **QDC Calibration Scale Factor ($s_{QDC}$)**: **`5.40`** (Nominal default: `4.50`)
- **QDC Pedestal Offset ($b_0$)**: **`0.40 a.u.`** (Nominal default: `0.00`)
- **Minimal Joint Calibration Loss**: **`0.6384`** (Reduced by `52.5%` from default `1.3444`)

### Parameter Sensitivity & Benchmark Summary

| Observable / Metric | Collision Data | Nominal Default MC | Optimal Calibrated MC | Improvement |
|:---|:---:|:---:|:---:|:---:|
| **SciFi Hits Peak** | **`12.0` hits** | `16.5` hits | **`11.5` hits** | Δ: `4.5 → 0.5` hits |
| **SciFi Hits Mean** | **`15.29` hits** | `22.36` hits | **`18.19` hits** | Δ: `7.07 → 2.91` hits |
| **SciFi Hits $W_1$ (EMD)** | `0.000` | `8.124` hits | **`3.985` hits** | **`-`50.9%** |
| **SciFi Hit QDC Peak** | **`1.10` a.u.** | `0.82` a.u. | **`1.17` a.u.** | Δ: `0.28 → 0.07` a.u. |
| **SciFi Hit QDC Mean** | **`2.18` a.u.** | `2.08` a.u. | **`2.12` a.u.** | Δ: `0.10 → 0.05` a.u. |
| **SciFi Hit QDC $W_1$ (EMD)** | `0.000` | `0.4034` a.u. | **`0.5993` a.u.** | **`-`-48.6%** |
| **Joint Calibration Loss** | `0.000` | `1.3444` | **`0.6384`** | **`-`52.5%** |

### Top 15 Parameter Combinations

| Rank | $t_s$ [p.e.] | $s_{\text{QDC}}$ | $b_0$ | Hits Peak | Hits $W_1$ | QDC Peak | QDC $W_1$ | Loss (Hits) | Loss (QDC) | Total Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 6.50 | 5.40 | 0.40 | 11.5 | 3.985 | 1.17 | 0.599 | 0.302 | 0.336 | **0.6384** |
| 2 | 7.00 | 5.40 | 0.40 | 11.5 | 3.235 | 1.24 | 0.613 | 0.253 | 0.410 | **0.6628** |
| 3 | 6.50 | 5.40 | 0.20 | 11.5 | 3.985 | 1.20 | 0.607 | 0.302 | 0.373 | **0.6758** |
| 4 | 6.50 | 5.10 | 0.40 | 11.5 | 3.985 | 1.24 | 0.577 | 0.302 | 0.388 | **0.6906** |
| 5 | 7.00 | 5.40 | 0.20 | 11.5 | 3.235 | 1.28 | 0.615 | 0.253 | 0.444 | **0.6975** |
| 6 | 6.00 | 5.10 | 0.40 | 13.5 | 4.578 | 1.12 | 0.563 | 0.425 | 0.275 | **0.6993** |
| 7 | 6.00 | 5.40 | 0.20 | 13.5 | 4.578 | 1.09 | 0.597 | 0.425 | 0.281 | **0.7055** |
| 8 | 6.50 | 5.40 | 0.00 | 11.5 | 3.985 | 1.24 | 0.612 | 0.302 | 0.409 | **0.7114** |
| 9 | 5.50 | 4.80 | 0.40 | 13.5 | 5.338 | 1.10 | 0.517 | 0.474 | 0.242 | **0.7158** |
| 10 | 7.00 | 5.10 | 0.40 | 11.5 | 3.235 | 1.31 | 0.592 | 0.253 | 0.466 | **0.7198** |
| 11 | 6.00 | 5.40 | 0.00 | 13.5 | 4.578 | 1.13 | 0.597 | 0.425 | 0.301 | **0.7258** |
| 12 | 6.50 | 5.10 | 0.20 | 11.5 | 3.985 | 1.27 | 0.585 | 0.302 | 0.427 | **0.7297** |
| 13 | 7.00 | 5.40 | 0.00 | 11.5 | 3.235 | 1.31 | 0.621 | 0.253 | 0.481 | **0.7341** |
| 14 | 6.00 | 5.10 | 0.20 | 13.5 | 4.578 | 1.16 | 0.567 | 0.425 | 0.312 | **0.7370** |
| 15 | 6.00 | 5.40 | 0.40 | 13.5 | 4.578 | 1.06 | 0.597 | 0.425 | 0.315 | **0.7392** |
