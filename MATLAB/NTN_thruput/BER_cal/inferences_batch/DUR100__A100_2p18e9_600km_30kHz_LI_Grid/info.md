# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\Clipped_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps
- **Target Dataset Folder**: MATLAB\sampleWiseDoppler_wGeometry_A100_2p18e9_600km_70deg_30kHz

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 5.918119e-17 | 0.266585 | -5.74 dB | 0.173876 |
| -5 | 4.028424e-17 | 0.176399 | -7.54 dB | 0.286693 |
| +0 | 1.052094e-17 | 0.049527 | -13.05 dB | 0.470113 |
| +5 | 5.046970e-18 | 0.022423 | -16.49 dB | 0.642179 |
| +10 | 2.914804e-18 | 0.014542 | -18.37 dB | 0.766068 |
| +15 | 1.505972e-18 | 0.008413 | -20.75 dB | 0.857237 |

## Inferred MAT File Field Reference
All variables are saved combined in **`inferredChannel.mat`** inside each target SNR subfolder.

### Belong to Inference Results
- `H_LI_infer`: The complex estimated/inferred channel matrix (shape: `(N, 132, 14)`).
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
- **ONNX Model File**: best_net.onnx
- **Number of Samples**: 512
- **Extrapolation Clipping**: True
- **MATLAB Variable Key**: H_LI_infer
