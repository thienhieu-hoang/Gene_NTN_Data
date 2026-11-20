# Dataset Overview: Diverse NTN Channel (2.18 GHz, 600 km, 30°–80°, 30 kHz)

## General Information

- **Dataset Type:** Physically Grounded Diverse NTN Channel Dataset (3GPP TR 38.811)
- **Data Generator:** MATLAB 5G Toolbox (`Gen_Chan_Diverse_Dataset.m`)
- **Total Samples:** 20000 drops (Train: 16000, Val: 2000, Test: 2000)
- **Available Files:** `H_train.npy` (16000), `H_val.npy` (2000), `H_test.npy` (2000)
- **Data Type:** `complex64`, shape `(N, 132, 14)` (132 subcarriers × 14 OFDM symbols)
- **Power Normalization:** Raw unnormalized physical channel scale preserved

---

## Parameter Distributions & Diversity Summary

| Parameter | Tested Range | Distribution / Modeling |
| :--- | :--- | :--- |
| **Elevation Angle ($\theta$)** | [30.0°, 80.0°] (Mean: 55.0°) | Uniform random across pass range |
| **UE Velocity ($v_{\text{UE}}$)** | [10.0, 30.0] m/s (36–108 km/h) | Uniform vehicular mobility range |
| **Channel Profiles** | NTN-TDL-A, B, C, D | LOS: 57.5% (`NTN-TDL-C/D`), NLOS: 42.5% (`NTN-TDL-A/B`) |
| **Delay Spread ($\tau_{\text{rms}}$)** | [30.0, 486.7] ns (Mean: 170.0 ns) | **Coupled to Elevation & LOS:**<br>• LOS: 30–150 ns (mean ~60 ns)<br>• NLOS: 80–500 ns (mean ~260 ns) |
| **Rician $K$-factor (LOS)** | [4.0, 17.0] dB (Mean: 10.8 dB) | Scales monotonically with elevation angle (3GPP TR 38.811) |
| **UE Spatial Distribution** | Uniform disk inside 15 km beam | $r \le 14.5$ km beam footprint offset |
| **Carrier Frequency ($f_c$)** | 2.18 GHz | S-band NTN carrier |
| **Subcarrier Spacing (SCS)** | 30 kHz | 5G NR Numerology $\mu=1$ |
| **Grid Bandwidth** | 11 RBs (132 subcarriers) | 5G NR slot with 14 symbols |
| **Antenna Configuration** | 1 × 1 (SISO) | Single-input single-output |

---

## Physical Coupling Justification (3GPP TR 38.811)

1. **Elevation-Dependent LOS Probability**: At higher elevation angles ($\\theta \\ge 60^\\circ$), obstacles and terrain clutter are avoided, yielding up to ~85% LOS. At lower elevation ($\\theta \\le 35^\\circ$), building blockage increases NLOS up to ~70%.
2. **Reasonable Delay Spread Scaling**: In NTN, scattering occurs solely near the UE. When LOS is present, the direct path dominates, keeping $\tau_{\text{rms}} \le 150$ ns. In NLOS, rich urban multipath stretches $\tau_{\text{rms}}$ up to 500 ns.
3. **Rician $K$-factor Scaling**: Direct-to-multipath ratio increases as the satellite approaches zenith ($K$ rises from ~5 dB at 30° to ~16 dB at 80°).
