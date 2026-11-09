# Channel Dataset Physical Characterization Report

- **Target Folder:** `DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps/15dB`
- **Absolute Path:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps\15dB`
- **Total Realizations:** `2048`
- **Grid Size:** `132 Subcarriers × 14 Symbols` (SCS = `30.0 kHz`)
- **Delay-Doppler Resolution:** `Δτ = 252.53 ns/bin`, `Δν = 2000.0 Hz/bin`

## 1. Physical Propagation & Coherence Metrics

| Physical Parameter | Value | Standard Interpretation |
| :--- | :---: | :--- |
| **RMS Delay Spread ($\tau_{\text{rms}}$)** | `310.50 ns` | Multipath dispersion along subcarriers |
| **Mean Excess Delay ($\bar{\tau}$)** | `217.60 ns` | Center of multipath power distribution |
| **Coherence Bandwidth ($B_{c, 0.5}$)** | `991.5 kHz` (`33.05` sc) | Frequency lag where correlation drops to 0.5 |
| **Mean Doppler Shift ($\bar{\nu}$)** | `-1.48 Hz` | Bulk orbital/velocity Doppler frequency |
| **RMS Doppler Spread ($\nu_{\text{rms}}$)** | `490.65 Hz` | Temporal fading velocity across OFDM symbols |
| **Coherence Time ($T_{c, 0.5}$)** | `0.46 ms` (`13.00` sym) | Time lag where correlation drops to 0.5 |
| **Channel Power (RMS)** | `6.7916e-10` | Average energy magnitude |
| **Dynamic Range** | `146.36 dB` | Peak-to-minimum power variation |

## 2. Baseline Pilot Estimation Performance

| Baseline Estimator | NMSE (dB) | SSIM |
| :--- | :---: | :---: |
| **2D Linear Interpolation ($H_{\text{li}}$)** | `-14.39 dB` | `1.0000` |

## 3. Visual Characterization Plots

### 2D Delay-Doppler Energy Distribution (Ensemble Average over all N = 2048 samples)
![2D Delay-Doppler](./fig_delay_doppler_2d.png)

### 1D Physical Curves (Ensemble Average over all N = 2048 samples)
![1D Physics](./fig_channel_physics_1d.png)

### Single Realization Snapshot (Sample #0 in Time-Frequency Domain)
![TF Heatmaps](./fig_samples_tf_heatmaps.png)
