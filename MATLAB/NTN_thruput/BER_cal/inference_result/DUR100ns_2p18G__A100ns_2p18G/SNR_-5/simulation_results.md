# NTN Channel Estimation & BER Simulation Results

```text
=========================================================
SIMULATION RESULTS (SNR = -5 dB, Elevation = 70°, numUE = 100)
=========================================================
1. LS + Linear Interpolation (LI):
   - Mean BER : 0.503646
   - Mean NMSE: 7.92 dB (6.199145)
2. LS + MMSE Benchmark (Perfect Channel Correlation):
   - Mean BER : 0.501089
   - Mean NMSE: -12.30 dB (0.058855)
3. H_ls_cnn (CNN with Sparse LS Input):
   - Mean BER : 0.498120
   - Mean NMSE: -8.45 dB (0.142886)
4. H_li_cnn (CNN with Linear Interpolation Input):
   - Mean BER : 0.495310
   - Mean NMSE: -10.15 dB (0.096605)
=========================================================
```

## Simulation Configuration & System Parameters
- **Signal-to-Noise Ratio (SNR):** -5 dB
- **Satellite Elevation Angle:** 70°
- **Number of UEs ($N_{\text{UE}}$):** 100
- **Carrier Frequency ($f_c$):** 2.18 GHz (S-band)
- **Satellite Altitude:** 600 km (LEO Orbit)
- **Subcarrier Spacing (SCS):** 30 kHz
- **Resource Blocks ($N_{\text{grid}}$):** 11 RBs (132 Subcarriers)
- **Channel Model:** NTN-TDL-A (Delay Spread: 200 ns)
- **PDSCH Modulation:** 16QAM
- **Beam Center Satellite Doppler Shift:** Pre-compensated at beam center

## Performance Comparison Summary across 4 Channel Estimators
| Estimation Approach | Mean BER | Mean NMSE (dB) | Mean NMSE (Linear) |
| :--- | :---: | :---: | :---: |
| **LS + Linear Interpolation (LI)** | `0.503646` | `7.92 dB` | `6.199145` |
| **LS + MMSE Benchmark (Perfect Correlation)** | `0.501089` | `-12.30 dB` | `0.058855` |
| **H_ls_cnn (CNN with Sparse LS Input)** | `0.498120` | `-8.45 dB` | `0.142886` |
| **H_li_cnn (CNN with Linear Interpolation Input)** | `0.495310` | `-10.15 dB` | `0.096605` |

## Variables Saved in MAT File (`BER_performance_results.mat`)
- `ber_LI`: `[1 x numUE]` double array — Bit Error Rate per UE sample using LS + Linear Interpolation.
- `ber_MMSE`: `[1 x numUE]` double array — Bit Error Rate per UE sample using LS + MMSE Benchmark.
- `ber_ls_cnn`: `[1 x numUE]` double array — Bit Error Rate per UE sample using CNN model with sparse LS input (`H_ls_cnn`).
- `ber_li_cnn`: `[1 x numUE]` double array — Bit Error Rate per UE sample using CNN model with linear interpolation input (`H_li_cnn`).
- `nmse_LI`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for LS + Linear Interpolation.
- `nmse_MMSE`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for LS + MMSE Benchmark.
- `nmse_ls_cnn`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for `H_ls_cnn`.
- `nmse_li_cnn`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for `H_li_cnn`.
- `SNRdB`: Scalar SNR value in dB.
- `numUE`: Scalar number of evaluated UE samples.
- `simParameters`: Struct containing satellite orbit, geometry, and system parameters.
- `carrier`: 5G NR carrier configuration object (`nrCarrierConfig`).
- `pdsch`: 5G NR PDSCH configuration object (`nrPDSCHConfig`).

## Complex 2D Channel Grids MAT File (`sample1_channel_grids.mat`)
Contains full complex 2D channel matrices `[nSubcarriers x nSymbols]` for Sample 1:
- `H_perfect_ori_sample1`: Complex 2D array — Original uncompensated channel.
- `H_perfect_eff_sample1`: Complex 2D array — Effective Doppler-compensated channel.
- `H_ls_sample1`: Complex 2D array — Sparse LS pilot estimated channel grid.
- `H_li_sample1`: Complex 2D array — Linear Interpolation estimated channel grid.
- `H_ls_cnn_sample1`: Complex 2D array — `H_ls_cnn` model estimated channel grid.
- `H_li_cnn_sample1`: Complex 2D array — `H_li_cnn` model estimated channel grid.

## Channel Grid Visualizations (Sample 1)
- **Perfect Original Channel Vector PDF:** [`H_perfect_ori_sample1.pdf`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/MATLAB/NTN_thruput/BER_cal/inference_result/DUR100ns_2p18G__A100ns_2p18G/SNR_-5/H_perfect_ori_sample1.pdf)
- **Effective Compensated Channel Vector PDF:** [`H_perfect_eff_sample1.pdf`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/MATLAB/NTN_thruput/BER_cal/inference_result/DUR100ns_2p18G__A100ns_2p18G/SNR_-5/H_perfect_eff_sample1.pdf)
- **Sparse LS Estimated Channel Vector PDF:** [`H_ls_sample1.pdf`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/MATLAB/NTN_thruput/BER_cal/inference_result/DUR100ns_2p18G__A100ns_2p18G/SNR_-5/H_ls_sample1.pdf)
- **Linear Interpolation Channel Vector PDF:** [`H_li_sample1.pdf`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/MATLAB/NTN_thruput/BER_cal/inference_result/DUR100ns_2p18G__A100ns_2p18G/SNR_-5/H_li_sample1.pdf)
- **H_ls_cnn Model Estimated Channel PDF:** [`H_ls_cnn_sample1.pdf`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/MATLAB/NTN_thruput/BER_cal/inference_result/DUR100ns_2p18G__A100ns_2p18G/SNR_-5/H_ls_cnn_sample1.pdf)
- **H_li_cnn Model Estimated Channel PDF:** [`H_li_cnn_sample1.pdf`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/MATLAB/NTN_thruput/BER_cal/inference_result/DUR100ns_2p18G__A100ns_2p18G/SNR_-5/H_li_cnn_sample1.pdf)
