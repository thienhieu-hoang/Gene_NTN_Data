# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps
- **Target Dataset Folder**: MATLAB\sampleWiseDoppler_wGeometry_A100_2p18e9_600km_70deg_30kHz

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 5.263333e-17 | 0.250277 | -6.02 dB | 0.449359 |
| -5 | 4.160407e-17 | 0.197643 | -7.04 dB | 0.492860 |
| +0 | 2.291581e-17 | 0.122597 | -9.12 dB | 0.598419 |
| +5 | 1.943645e-17 | 0.096124 | -10.17 dB | 0.629471 |
| +10 | 1.377124e-17 | 0.068545 | -11.64 dB | 0.667592 |
| +15 | 8.858805e-18 | 0.051925 | -12.85 dB | 0.720416 |

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
- **ONNX Model File**: best_model.onnx
- **Number of Samples**: 512
- **Extrapolation Clipping**: False
- **MATLAB Variable Key**: H_LS_infer
