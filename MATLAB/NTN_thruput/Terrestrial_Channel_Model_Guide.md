# Terrestrial 3GPP TR 38.901 TDL Channel Modeling Guide

This document specifies the design rationale, physical parameters, and execution instructions for the terrestrial channel generator script:
[`Gen_Chan_Diverse_Terrestrial_Dataset.m`](./Gen_Chan_Diverse_Terrestrial_Dataset.m).

---

## 1. Domain Gap: Why the Terrestrial Channel is Harsher than NTN

When evaluating domain adaptation (UDA, CORAL, FDA) from an NTN-pretrained channel estimation network to a terrestrial system, the channel experiences several fundamental structural shifts:

| Physical Metric | NTN Channel (3GPP TR 38.811) | Terrestrial Channel (3GPP TR 38.901 UMa) | Domain Gap & Impact on Estimator |
| :--- | :--- | :--- | :--- |
| **Carrier / Bulk Doppler** | Massive raw shift ($\pm 48\text{ kHz}$ at LEO), but **pre-compensated** for beam-center. | **Zero** satellite offset; Base Station is stationary on the ground. | NTN networks learn residual phase rotations around 0 Hz. In terrestrial, no bulk phase ramp exists. |
| **Multipath Doppler Spectrum** | Highly concentrated/directional around the LOS satellite vector. | **Isotropic / U-shaped (Jakes / Clarke spectrum)** from 360° scattering. | Every multipath tap experiences independent diffuse Doppler spread ($f_D \in [-f_{D,\max}, +f_{D,\max}]$), accelerating time-decorrelation across OFDM symbols. |
| **RMS Delay Spread ($\tau_{\text{rms}}$)** | **Small**: $30\text{ ns} - 150\text{ ns}$ (high elevation, limited clutter around satellite). | **Large**: $150\text{ ns} - 800\text{ ns}$ (dense urban macro clutter, multi-bounce reflections). | **2× to 5× wider delay spread** drastically reduces coherence bandwidth, causing deep frequency-selective notches across the 132 subcarriers. |
| **Fading Character** | Predominantly strong LOS ($K = 4\text{ dB} - 18\text{ dB}$ for TDL-C/D at $\ge 50^\circ$ elevation). | Predominantly NLOS (~75% NLOS) Rayleigh fading ($K = -\infty$). | Deep fading nulls test the neural network's interpolation and denoising robustness much more severely. |

---

## 2. Parameter Specifications & Physical Coupling (3GPP TR 38.901 UMa)

### A. Antenna Heights & Cell Geometry
* **Base Station Height ($h_{\text{BS}}$)**: $25.0\text{ m}$ (standard Urban Macrocell rooftop / dedicated tower height, elevated above average 15–20 m urban buildings).
* **UE Terminal Height ($h_{\text{UE}}$)**: $1.5\text{ m}$ (standard handheld user or in-car mobile terminal).
* **Cell Radius ($R_{\text{cell}}$)**: $500.0\text{ m}$ (dense urban macrocell coverage).
* **Minimum Distance ($r_{\min}$)**: $20.0\text{ m}$ (avoids the immediate near-field tower exclusion zone).
* **Spatial Density**: Uniform area sampling:
  $$r_{\text{UE}} = \sqrt{r_{\min}^2 + (R_{\text{cell}}^2 - r_{\min}^2) \cdot u}, \quad u \sim U(0, 1), \quad \theta \sim U(0, 2\pi)$$
* **3D Euclidean Distance**:
  $$d_{3D} = \sqrt{r_{\text{UE}}^2 + (h_{\text{BS}} - h_{\text{UE}})^2}$$

### B. Distance-Dependent LOS Probability
Following 3GPP TR 38.901 UMa formula for $h_{\text{UE}} = 1.5\text{ m}$:
$$P_{\text{LOS}}(d_{2D}) = \begin{cases} 1.0, & d_{2D} \le 18\text{ m} \\ \frac{18}{d_{2D}} + \exp\left(-\frac{d_{2D}}{63}\right)\left(1 - \frac{18}{d_{2D}}\right), & d_{2D} > 18\text{ m} \end{cases}$$
Across a 500 m macrocell, this produces realistic urban distribution (~25% LOS, ~75% NLOS).

### C. Channel Profile Selection & Delay Spread Coupling
* **When NLOS is sampled (~75%)**:
  * Profile distribution: 75% **`TDL-A`** (23 taps, rich urban scattering) and 25% **`TDL-B`** (23 taps, steeper delay decay).
  * Fading type: Pure Rayleigh ($K = -\infty$).
  * Delay spread: **$150\text{ ns}$ to $800\text{ ns}$** (mean ~350 ns), scaling with cell distance:
    $$\tau_{\text{rms, base}} = 200\text{ ns} + \left(\frac{d_{2D}}{R_{\text{cell}}}\right) \cdot 350\text{ ns} \pm \text{jitter}$$
* **When LOS is sampled (~25%)**:
  * Profile distribution: 75% **`TDL-C`** (24 taps, Rician fading) and 25% **`TDL-D`** (14 taps, Rician fading).
  * Fading type: Rician with distance-dependent $K$-factor ($K \in [0, 9]\text{ dB}$):
    $$K_{\text{base}} = 9.0 - \left(\frac{d_{2D}}{R_{\text{cell}}}\right) \cdot 7.0 \pm \text{jitter}$$
  * Delay spread: **$30\text{ ns}$ to $250\text{ ns}$** (mean ~100 ns).

### D. UE Mobility & Doppler Shift
* **UE Velocity ($v_{\text{UE}}$)**: Uniformly sampled in $[5.0, 35.0]\text{ m/s}$ ($18\text{ km/h}$ to $126\text{ km/h}$, covering urban city traffic to highway speeds).
* **Max Doppler Shift ($f_{D,\max}$)**:
  $$f_{D,\max} = \frac{v_{\text{UE}} \cdot f_c}{c}$$
  At $f_c = 2.18\text{ GHz}$, this corresponds to $36.3\text{ Hz} \le f_{D,\max} \le 254.3\text{ Hz}$.
* **No Satellite Pre-compensation**: In terrestrial cellular, the transmitted waveform is directly transmitted over the fading channel without phase modulation pre-compensation.

### E. Large-Scale Path Loss (3GPP TR 38.901 UMa)
* **LOS**:
  $$PL_{\text{LOS}} = 28.0 + 22.0 \log_{10}(d_{3D}) + 20.0 \log_{10}(f_c[\text{GHz}]) + \chi_{\sigma_{\text{LOS}}}, \quad \sigma_{\text{LOS}} = 4.0\text{ dB}$$
* **NLOS**:
  $$PL_{\text{NLOS}} = 13.54 + 39.03 \log_{10}(d_{3D}) + 20.0 \log_{10}(f_c[\text{GHz}]) - 0.6(h_{\text{UE}} - 1.5) + \chi_{\sigma_{\text{NLOS}}}, \quad \sigma_{\text{NLOS}} = 6.0\text{ dB}$$

---

## 3. Output Dataset Format & Numerology

To guarantee 100% plug-and-play compatibility with existing NTN channel estimation models (`LS+Attention`, `LI+DnCNN`) and evaluation pipelines:

- **Carrier Frequency**: $2.18\text{ GHz}$ (S-band, matched to NTN)
- **Subcarrier Spacing (SCS)**: $30\text{ kHz}$ ($\mu = 1$)
- **Grid Size**: $N_{\text{RB}} = 11$ (132 subcarriers)
- **Time Slot**: 14 OFDM symbols
- **Output Tensor Shape**: `(N_samples, 132, 14)` with `complex64` (`<c8`)
- **Output Files**:
  - `H_train.npy` (80% of samples)
  - `H_val.npy` (10% of samples)
  - `H_test.npy` (10% of samples)
  - `sample_metadata.mat` (contains exact ground-truth `meta_d2d_m`, `meta_d3d_m`, `meta_v_ue`, `meta_ds_ns`, `meta_profile`, `meta_is_los`, `meta_k_factor`, `meta_doppler`, `meta_pathloss`)
  - `note.md` (automatic execution log and parameter distributions summary)

---

## 4. Execution Commands

### Quick Sanity Run (200 samples)
```powershell
.\run_generate_terrestrial_dataset.ps1 -NumUE 200 -FolderName "Test_Terrestrial_200"
```

### Full Diverse Dataset (20,000 samples)
```powershell
.\run_generate_terrestrial_dataset.ps1 -NumUE 20000 -FolderName "Diverse_Terrestrial_2p18e9_UMa_30kHz"
```

Or invoke directly from MATLAB:
```matlab
Gen_Chan_Diverse_Terrestrial_Dataset('NumUE', 20000, 'FolderName', 'Diverse_Terrestrial_2p18e9_UMa_30kHz');
```
