# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention
- **Target Dataset Folder**: DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 1.801704e-19 | 0.702709 | -1.53 dB | 0.926911 |
| -5 | 1.236851e-19 | 0.385133 | -4.14 dB | 0.947630 |
| +0 | 1.784733e-19 | 0.254563 | -5.94 dB | 0.960536 |
| +5 | 6.962652e-20 | 0.195139 | -7.10 dB | 0.968573 |
| +10 | 6.691735e-20 | 0.179235 | -7.47 dB | 0.973145 |
| +15 | 5.793766e-20 | 0.160093 | -7.96 dB | 0.974110 |

## Inferred MAT File Field Reference
All variables are saved combined in **`inferredChannel.mat`** inside each target SNR subfolder.

### Belong to Inference Results
- `H_LS_infer`: The complex estimated/inferred channel matrix (shape: `(N, 132, 14)`).
- `mmse`: Average Mean Squared Error compared to perfect label (scalar).
- `nmse`: Average Normalized Mean Squared Error (scalar).
- `nmse_db`: Average NMSE in dB (scalar).
- `ssim`: Average Structural Similarity Index (scalar).

### Belong to Original Dataset
- `H_li`: Original linear-interpolated input channel (shape: `(N, 132, 14)`).
- `H_ls_pilots`: Original sparse pilot values (shape: `(N, 88)`).
- `H_prac`: Original practical estimated channel (shape: `(N, 132, 14)`).
- `H_perfect` / `H_perfect_ori`: True channel labels (shape: `(N, 132, 14)`).
- `pilot_rows` / `pilot_cols` / `pilot_indices`: Grid positions of the pilot symbols.
- Sim geometry & propagation vectors: `r_ue_ECEF_all`, `ut_loc_ENU_all`, `slant_ranges`, `doppler_shifts_all`, `pl_dB_all`, `elevation_angles`, etc.
- Constant system variables: `bs_loc_ENU`, `r_sat_ECEF`, `v_sat_ECEF`, `v_sat_ENU`, `satelliteDopplerShift_bc`, etc.

## Inference Details
- **ONNX Model File**: auto
- **Number of Samples**: All
- **Extrapolation Clipping**: False
- **Standardization**: False
- **MATLAB Variable Key**: H_LS_infer
