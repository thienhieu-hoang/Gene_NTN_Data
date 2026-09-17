# Pseudo Dataset Overview: FDA Domain Adaptation

## General Information

- **Method:** Delay-Doppler Fourier Domain Adaptation (FDA) + 5G NR DM-RS Pilot & Noise Realization
- **Source Dataset Path:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\generatedChannel_Results\A100_2p18e9_600km_70deg_30kHz`
- **Target Dataset Path:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\Sionna\OpenNTN\channel_wGeometry\results\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps`
- **Result Folder:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\A100__DUR300`
- **Generation Date:** 2026-09-17 17:05:01

---

## Configuration Summary

| Parameter | Value |
| :--- | :--- |
| **Source Domain** | NTN TDL-A (NLOS, 70° elevation, 100 ns delay spread, 30 kHz SCS) |
| **Target Domain** | OpenNTN DUR NLOS (30° elevation, 300 ns delay spread, 30 kHz SCS) |
| **FDA Window ($13 \times w_w$)** | [3 5] |
| **SNR Range** | [-10 -5 0 5 10 15] dB |
| **Grid Dimensions** | 132 Subcarriers × 14 OFDM Symbols (11 RBs) |
| **Pilot Configuration** | DM-RS Type 2 Port 1 (88 pilots per slot: symbols 3, 12; subcarriers 1-128) |
| **Saved Variables** | `H_perfect`, `H_li`, `H_ls_pilots`, `pilot_rows`, `pilot_cols`, `pilot_indices`, `nmse_li`, `nmse_ls_pilot`, `ssim_li`, `ssim_li_pilot`, `ssim_ls` |

## Linked Reference Notes

- [Source Dataset Note](note_source.md)
- [Target Dataset Note](note_target.md)
