# SND@LHC SciFi Multi-Parameter Calibration Report

## Joint Hit Multiplicity and QDC Energy Optimization

- **Evaluated Simulation**: `/eos/user/i/idioniso/1_Data/Monte_Carlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200_trks.root`
- **Data Reference (Hits)**: `plots/passing_muon_selection.root` (`HitDistributions/SciFi_Hits/h_sf_hits_data_stage6`)
- **Data Reference (QDC)**: `plots/scifi_calibration_run8329.root` (`Histograms/1D/h1_hit_qdc`)
- **Total Grid Combinations Evaluated**: `2805` parameter points

### Optimal Calibration Parameters

- **SciFi Hit Threshold ($t_s$)**: **`6.00 p.e.`** (Nominal default: `3.50 p.e.`)
- **QDC Calibration Scale Factor ($s_{QDC}$)**: **`4.05`** (Nominal default: `4.50`)
- **QDC Pedestal Offset ($b_0$)**: **`0.40 a.u.`** (Nominal default: `0.00`)
- **Minimal Joint Calibration Loss**: **`0.9122`** (Reduced by `11.5%` from default `1.0311`)

### Parameter Sensitivity & Benchmark Summary

| Observable / Metric | Collision Data | Nominal Default MC | Optimal Calibrated MC | Improvement |
|:---|:---:|:---:|:---:|:---:|
| **SciFi Hits Peak** | **`12.0` hits** | `15.5` hits | **`14.5` hits** | Δ: `3.5 → 2.5` hits |
| **SciFi Hits Mean** | **`15.29` hits** | `22.42` hits | **`22.44` hits** | Δ: `7.13 → 7.16` hits |
| **SciFi Hits $W_1$ (EMD)** | `0.000` | `8.190` hits | **`8.219` hits** | **`-`-0.4%** |
| **SciFi Hit QDC Peak** | **`1.10` a.u.** | `1.09` a.u. | **`1.11` a.u.** | Δ: `0.01 → 0.01` a.u. |
| **SciFi Hit QDC Mean** | **`2.18` a.u.** | `2.08` a.u. | **`2.22` a.u.** | Δ: `0.09 → 0.04` a.u. |
| **SciFi Hit QDC $W_1$ (EMD)** | `0.000` | `0.4211` a.u. | **`0.3395` a.u.** | **`-`19.4%** |
| **Joint Calibration Loss** | `0.000` | `1.0311` | **`0.9122`** | **`-`11.5%** |

### Top 15 Parameter Combinations

| Rank | $t_s$ [p.e.] | $s_{\text{QDC}}$ | $b_0$ | Hits Peak | Hits $W_1$ | QDC Peak | QDC $W_1$ | Loss (Hits) | Loss (QDC) | Total Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 6.00 | 4.05 | 0.40 | 14.5 | 8.219 | 1.11 | 0.340 | 0.746 | 0.166 | **0.9122** |
| 2 | 6.00 | 4.20 | 0.30 | 14.5 | 8.219 | 1.10 | 0.366 | 0.746 | 0.172 | **0.9185** |
| 3 | 6.00 | 4.05 | 0.35 | 14.5 | 8.219 | 1.12 | 0.342 | 0.746 | 0.179 | **0.9246** |
| 4 | 6.00 | 4.20 | 0.35 | 14.5 | 8.219 | 1.08 | 0.364 | 0.746 | 0.183 | **0.9286** |
| 5 | 6.00 | 4.35 | 0.10 | 14.5 | 8.219 | 1.10 | 0.394 | 0.746 | 0.184 | **0.9303** |
| 6 | 6.00 | 4.20 | 0.20 | 14.5 | 8.219 | 1.12 | 0.368 | 0.746 | 0.187 | **0.9327** |
| 7 | 6.00 | 4.35 | 0.15 | 14.5 | 8.219 | 1.09 | 0.393 | 0.746 | 0.188 | **0.9338** |
| 8 | 6.00 | 4.05 | 0.30 | 14.5 | 8.219 | 1.14 | 0.344 | 0.746 | 0.191 | **0.9369** |
| 9 | 6.00 | 4.20 | 0.40 | 14.5 | 8.219 | 1.07 | 0.363 | 0.746 | 0.193 | **0.9387** |
| 10 | 6.00 | 4.50 | -0.05 | 14.5 | 8.219 | 1.10 | 0.420 | 0.746 | 0.193 | **0.9390** |
| 11 | 3.00 | 4.05 | -0.35 | 14.5 | 8.375 | 1.10 | 0.400 | 0.756 | 0.185 | **0.9411** |
| 12 | 5.50 | 4.05 | -0.35 | 14.5 | 8.375 | 1.10 | 0.400 | 0.756 | 0.185 | **0.9411** |
| 13 | 7.00 | 4.05 | -0.35 | 14.5 | 8.375 | 1.10 | 0.400 | 0.756 | 0.185 | **0.9411** |
| 14 | 6.00 | 4.35 | 0.05 | 14.5 | 8.219 | 1.11 | 0.395 | 0.746 | 0.195 | **0.9414** |
| 15 | 6.00 | 4.35 | 0.20 | 14.5 | 8.219 | 1.08 | 0.391 | 0.746 | 0.198 | **0.9437** |
