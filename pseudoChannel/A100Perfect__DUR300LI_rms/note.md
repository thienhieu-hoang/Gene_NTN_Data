# Pseudo Dataset Overview: FDA Domain Adaptation

## General Information

- **Method:** Delay-Doppler Fourier Domain Adaptation (FDA) + 5G NR DM-RS Pilot & Noise Realization
- **Fourier Transfer (FDA):** Translation of **Perfect (source) -> LI (target)** (`src_H_perf` -> `tgt_H_li`)
- **Source Dataset Path:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\generatedChannel_Results\A100_2p18e9_600km_70deg_30kHz`
- **Target Dataset Path:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\Sionna\OpenNTN\channel_wGeometry\results\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps`
- **Result Folder:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\A100Perfect__DUR300LI_rms`
- **Generation Date:** 2026-09-18 15:25:09

> **Note on Fourier Transfer & Sampling Strategy:**
> - Low-frequency Delay-Doppler components from target domain linear interpolated channel (`tgt_H_li`, pre-processed with extrapolation clipping) are transferred onto the clean source channel (`src_H_perf`) to generate pseudo ground truth (`H_perfect`).
> - When dataset sizes differ or when target pseudo sample count exceeds dataset sizes:
>   * Samples `1` to `N_min`: Strict 1-to-1 sequential mapping (Source `i` -> Target `i`).
>   * Samples `N_min+1` to `N_max`: Sequential mapping for the larger dataset, uniform random sampling from the smaller dataset.
>   * Samples `> N_max`: Uniform random sampling from both Source and Target datasets.
> - Corresponding index mappings are preserved in variables `idx_map_source` and `idx_map_target`.

---

## Configuration Summary

| Parameter | Value |
| :--- | :--- |
| **Fourier Transfer (FDA)** | **Perfect (source)** $\rightarrow$ **LI (target)** (`src_H_perf` $\rightarrow$ `tgt_H_li`) |
| **Source Domain** | NTN TDL-A (NLOS, 70° elevation, 100 ns delay spread, 30 kHz SCS) |
| **Target Domain** | OpenNTN DUR NLOS (30° elevation, 300 ns delay spread, 30 kHz SCS) |
| **FDA Preprocessing Scaling** | `rms` (Source & Target normalized to unit RMS, descaled by Target RMS) |
| **FDA Window ($13 \times w_w$)** | [3 5] |
| **SNR Range** | [-10 -5 0 5 10 15] dB |
| **Requested Pseudo Samples** | [] |
| **Grid Dimensions** | 132 Subcarriers × 14 OFDM Symbols (11 RBs) |
| **Pilot Configuration** | DM-RS Type 2 Port 1 (88 pilots per slot: symbols 3, 12; subcarriers 1-128) |
| **Pre-processing** | Target LI extrapolated elements clipped to inner pilot region min/max prior to FDA |
| **Saved Variables** | `H_perfect`, `H_li`, `H_ls_pilots`, `pilot_rows`, `pilot_cols`, `pilot_indices`, `idx_map_source`, `idx_map_target`, `nmse_li`, `nmse_ls_pilot`, `ssim_li`, `ssim_li_pilot`, `ssim_ls` |

## Linked Reference Notes

- [Source Dataset Note](note_source.md)
- [Target Dataset Note](note_target.md)
