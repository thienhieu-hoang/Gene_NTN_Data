# Dataset Overview: OpenNTN DUR LoS (2.18 GHz, 600 km, 70°, 30 kHz)

## General Information

- **Channel Model:** Dense Urban (DUR) (with setting LoS)
- **Data Generator:** OpenNTN

---

## Configuration Summary

| Parameter | Value |
| :--- | :--- |
| **Environment / Scenario** | Urban Expressway (Dense Urban, LoS) |
| **Carrier Frequency ($f_c$)** | 2.18 GHz |
| **Subcarrier Spacing (SCS)** | 30 kHz |
| **OFDM Grid Size** | 11 RBs (132 Subcarriers × 14 Symbols) |
| **UE Velocity** | [20, 30] m/s |
| **Satellite Altitude** | 600 km |
| **Nominal Elevation Angle** | 70° |
| **Delay Spread** | 100 ns |
| **Antenna Configuration** | 1 × 1 (SISO) |
| **Pilot Configuration (DM-RS)** | **5G NR DM-RS Type 2, Port 2:**<br>• **Frequency Allocation:** Subcarriers port 2 (`[2, 3, 8, 9]`) per RB (44 pilots/symbol across 11 RBs)<br>• **Time Allocation:** Mapping Type A, Pos 2 (OFDM symbols 2 and 11, 0-indexed)<br>• **Total Pilot REs:** 88 REs per slot |
