# Pseudo Dataset Overview: FDA Domain Adaptation (LS_Attention)

## General Information

- **Method:** Delay-Doppler Fourier Domain Adaptation (FDA) + 5G NR DM-RS Pilot & Noise Realization
- **Fourier Transfer (FDA):** Translation of **Perfect (source) -> LS_infer (target)** (`src_H_perf` -> `tgt_H_infer`)
- **Source Dataset Path:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\generatedChannel_Results\A100_2p18e9_600km_70deg_30kHz`
- **Target Dataset Path:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\inferred_dataset\A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz\LS_Attention_standardize`
- **Result Folder:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\A100Perfect__DUR300LSAttention_rms`
- **Generation Date:** 2026-09-18 15:14:33

> **Note on Fourier Transfer & Sampling Strategy:**
> - FDA transfers low-frequency Delay-Doppler components from target domain `tgt_H_infer` onto clean source channel `src_H_perf`.
> - When dataset sizes differ or when target pseudo sample count exceeds dataset sizes:
>   * Samples `1` to `N_min`: Strict 1-to-1 sequential mapping (Source `i` -> Target `i`).
>   * Samples `N_min+1` to `N_max`: Sequential mapping for the larger dataset, uniform random sampling from the smaller dataset.
>   * Samples `> N_max`: Uniform random sampling from both Source and Target datasets.
> - Corresponding index mappings are preserved in variables `idx_map_source` and `idx_map_target`.

---

## Configuration Summary

| Parameter | Value |
| :--- | :--- |
| **Fourier Transfer (FDA)** | **Perfect (source)** $\rightarrow$ **LS_infer (target)** (`src_H_perf` $\rightarrow$ `tgt_H_infer`) |
| **Source Domain** | NTN TDL-A (NLOS, 70° elevation, 100 ns delay spread, 30 kHz SCS) |
| **Target Domain** | OpenNTN DUR NLOS (30° elevation, 300 ns delay spread, 30 kHz SCS) - LS_Attention Inferred |
| **FDA Preprocessing Scaling** | `rms` (Source & Target normalized to unit RMS, descaled by Target RMS) |
| **FDA Window ($13 \times w_w$)** | [3 5] |
| **SNR Range** | [-10 -5 0 5 10 15] dB |
| **Requested Pseudo Samples** | [] |
| **Grid Dimensions** | 132 Subcarriers × 14 OFDM Symbols (11 RBs) |
| **Pilot Configuration** | DM-RS Type 2 Port 1 (88 pilots per slot: symbols 3, 12; subcarriers 1-128) |
| **Saved Variables** | `H_perfect`, `H_li`, `H_ls_pilots`, `pilot_rows`, `pilot_cols`, `pilot_indices`, `idx_map_source`, `idx_map_target`, `nmse_li`, `nmse_ls_pilot`, `ssim_li`, `ssim_li_pilot`, `ssim_ls` |

## Linked Reference Notes

- [Source Dataset Note](note_source.md)
- [Target Dataset Note](note_target.md)
