# NTN Channel Estimation & BER Simulation Results (ONNX Grid Models)

```text
=========================================================
SIMULATION RESULTS (SNR = -5 dB, Elevation = 70°, numUE = 10)
=========================================================
1. LS + Linear Interpolation (LI):
   - Mean BER : 0.497585
   - Mean NMSE: 7.64 dB (5.807409)
2. LS + MMSE Benchmark (Perfect Channel Correlation):
   - Mean BER : 0.497983
   - Mean NMSE: -8.80 dB (0.131846)
3. H_li_cnn (CNN with Linear Interpolation Input):
   - Mean BER : 0.497585
   - Mean NMSE: 7.64 dB (5.807409)
=========================================================
```

## Simulation Configuration & System Parameters
- **Signal-to-Noise Ratio (SNR):** -5 dB
- **Satellite Elevation Angle:** 70°
- **Number of UEs ($N_{\text{UE}}$):** 10
- **Carrier Frequency ($f_c$):** 2.18 GHz
- **Satellite Altitude:** 600 km
- **Subcarrier Spacing (SCS):** 30 kHz
- **Resource Blocks ($N_{\text{grid}}$):** 11 RBs (132 Subcarriers)
- **Channel Model:** NTN-TDL-A (Delay Spread: 100 ns)
- **PDSCH Modulation:** 16QAM
- **Beam Center Satellite Doppler Shift:** 16843.11 Hz

## Performance Comparison Summary
| Estimation Approach | Mean BER | Mean NMSE (dB) | Mean NMSE (Linear) |
| :--- | :---: | :---: | :---: |
| **LS + Linear Interpolation (LI)** | `0.497585` | `7.64 dB` | `5.807409` |
| **LS + MMSE Benchmark (Perfect Correlation)** | `0.497983` | `-8.80 dB` | `0.131846` |
| **H_li_cnn (CNN with LI Input)** | `0.497585` | `7.64 dB` | `5.807409` |

## Variables Saved in MAT File (`BER_performance_results.mat`)
- `ber_LI`: `[1 x numUE]` double array — Bit Error Rate per UE sample using LS + Linear Interpolation.
- `ber_MMSE`: `[1 x numUE]` double array — Bit Error Rate per UE sample using LS + MMSE Benchmark.
- `ber_li_cnn`: `[1 x numUE]` double array — Bit Error Rate per UE sample using CNN model with linear interpolation input (`H_li_cnn`).
- `nmse_LI`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for LS + Linear Interpolation.
- `nmse_MMSE`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for LS + MMSE Benchmark.
- `nmse_li_cnn`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for `H_li_cnn`.
- `SNRdB`: Scalar SNR value in dB.
- `numUE`: Scalar number of evaluated UE samples.
- `simParameters`: Struct containing satellite orbit, geometry, and system parameters.
- `pdsch`: 5G NR PDSCH configuration object (`nrPDSCHConfig`).

## Complex 2D Channel Grids MAT File (`sample1_channel_grids.mat`)
Contains full complex 2D channel matrices `[nSubcarriers x nSymbols]` for Sample 1:
- `H_perfect_ori_sample1`: Complex 2D array — Original uncompensated channel.
- `H_perfect_eff_sample1`: Complex 2D array — Effective Doppler-compensated channel.
- `H_li_sample1`: Complex 2D array — Linear Interpolation estimated channel grid.
- `H_li_cnn_sample1`: Complex 2D array — `H_li_cnn` model estimated channel grid.

