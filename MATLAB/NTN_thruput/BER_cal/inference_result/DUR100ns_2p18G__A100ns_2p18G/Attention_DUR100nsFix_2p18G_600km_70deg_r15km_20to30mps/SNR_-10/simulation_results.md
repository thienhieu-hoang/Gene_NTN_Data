# NTN Channel Estimation & BER Simulation Results (ONNX Sequence Attention Model)

```text
=========================================================
SIMULATION RESULTS (SNR = -10 dB, Elevation = 70°, numUE = 512)
=========================================================
1. LS + Linear Interpolation (LI Benchmark):
   - Mean BER : 0.499749
   - Mean NMSE: 13.14 dB (20.588071)
2. LS + MMSE Benchmark (Perfect Channel Correlation):
   - Mean BER : 0.500079
   - Mean NMSE: -4.78 dB (0.332621)
3. H_ls_attention (ONNX Attention Model with LS Sequence Input):
   - Mean BER : 0.500005
   - Mean NMSE: -1.93 dB (0.640689)
=========================================================
```

## Simulation Configuration & System Parameters
- **Signal-to-Noise Ratio (SNR):** -10 dB
- **Satellite Elevation Angle:** 70°
- **Number of UEs ($N_{\text{UE}}$):** 512
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
| **LS + Linear Interpolation (LI Benchmark)** | `0.499749` | `13.14 dB` | `20.588071` |
| **LS + MMSE Benchmark (Perfect Correlation)** | `0.500079` | `-4.78 dB` | `0.332621` |
| **H_ls_attention (ONNX Attention Model with LS Sequence Input)** | `0.500005` | `-1.93 dB` | `0.640689` |

## Variables Saved in MAT File (`BER_performance_results.mat`)
- `ber_LI`: `[1 x numUE]` double array — Bit Error Rate per UE sample using LS + Linear Interpolation.
- `ber_MMSE`: `[1 x numUE]` double array — Bit Error Rate per UE sample using LS + MMSE Benchmark.
- `ber_ls_atten`: `[1 x numUE]` double array — Bit Error Rate per UE sample using ONNX Attention model with LS sequence input (`H_ls_attention`).
- `nmse_LI`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for LS + Linear Interpolation.
- `nmse_MMSE`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for LS + MMSE Benchmark.
- `nmse_ls_atten`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for `H_ls_attention`.
- `SNRdB`: Scalar SNR value in dB.
- `numUE`: Scalar number of evaluated UE samples.
- `simParameters`: Struct containing satellite orbit, geometry, and system parameters.
- `pdsch`: 5G NR PDSCH configuration object (`nrPDSCHConfig`).

## Complex 2D Channel Grids MAT File (`sample1_channel_grids.mat`)
Contains full complex 2D channel matrices `[nSubcarriers x nSymbols]` for Sample 1:
- `H_perfect_ori_sample1`: Complex 2D array — Original uncompensated channel.
- `H_perfect_eff_sample1`: Complex 2D array — Effective Doppler-compensated channel.
- `H_ls_sample1`: Complex 2D array — Sparse LS pilot estimated channel grid.
- `H_li_sample1`: Complex 2D array — Linear Interpolation estimated channel grid.
- `H_ls_atten_sample1`: Complex 2D array — `H_ls_attention` model estimated channel grid.

