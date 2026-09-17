# Dataset Overview: OpenNTN DUR NLoS (2.18 GHz, 600 km, 30°, 30 kHz)

## General Information

- **Channel Model:** Dense Urban (DUR) (with setting NLoS)
- **Data Generator:** OpenNTN

---

## Configuration Summary

| Parameter | Value |
| :--- | :--- |
| **Environment / Scenario** | Urban Expressway (Dense Urban, NLoS) |
| **Carrier Frequency ($f_c$)** | 2.18 GHz |
| **Subcarrier Spacing (SCS)** | 30 kHz |
| **OFDM Grid Size** | 11 RBs (132 Subcarriers × 14 Symbols) |
| **UE Velocity** | [20, 30] m/s |
| **Satellite Altitude** | 600 km |
| **Nominal Elevation Angle** | 30° |
| **Delay Spread** | 300 ns |
| **Antenna Configuration** | 1 × 1 (SISO) |
| **Pilot Configuration (DM-RS)** | **5G NR DM-RS Type 2, Port 1:**<br>• **Frequency Allocation:** Subcarriers port 1 (`[0, 1, 6, 7]`) per RB (44 pilots/symbol across 11 RBs)<br>• **Time Allocation:** Mapping Type A, Pos 2 (OFDM symbols 2 and 11, 0-indexed)<br>• **Total Pilot REs:** 88 REs per slot |
