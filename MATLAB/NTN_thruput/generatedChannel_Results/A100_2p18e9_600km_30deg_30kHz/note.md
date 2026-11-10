# Dataset Overview: NTN-TDL-A (2.18 GHz, 600 km, 30°, 30 kHz)

## General Information

- **Channel Model:** NTN-TDL-A (NLOS (Rayleigh))
- **Data Generator:** MATLAB 5G Toolbox
- **Benchmark Role:** Elevation-only domain shift (70° -> 30°)
- **Total Samples:** 20,000 drops (Train: 16,000, Val: 2,000, Test: 2,000)
- **Available Files:** `H_train.npy` (16,000), `H_val.npy` (2,000), `H_test.npy` (2,000)
- **Data Type:** `complex64`, shape `(N, 132, 14)`
- **Power Normalization:** Sample-wise RMS unit power ($\mathbb{E}[|H|^2] = 1.0$)

---

## Configuration Summary

| Parameter | Value |
| :--- | :--- |
| **Environment / Scenario** | Urban Expressway |
| **Carrier Frequency ($f_c$)** | 2.18 GHz |
| **Subcarrier Spacing (SCS)** | 30 kHz |
| **OFDM Grid Size** | 11 RBs (132 Subcarriers × 14 Symbols) |
| **UE Velocity** | [20, 30] m/s |
| **Satellite Altitude** | 600 km |
| **Nominal Elevation Angle** | 30° |
| **Delay Spread** | 100 ns |
| **Antenna Configuration** | 1 × 1 (SISO) |
| **Pilot Configuration (DM-RS)** | **5G NR DM-RS Type 2, Port 1:**<br>• **Frequency Allocation:** Subcarriers port 1 (`[0, 1, 6, 7]`) per RB (44 pilots/symbol across 11 RBs)<br>• **Time Allocation:** Mapping Type A, Pos 2 (OFDM symbols 2 and 11, 0-indexed)<br>• **Total Pilot REs:** 88 REs per slot |
