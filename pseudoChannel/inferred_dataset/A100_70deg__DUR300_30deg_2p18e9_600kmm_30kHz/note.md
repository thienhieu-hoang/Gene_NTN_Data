# Experiment Note: A100 (70°) → DUR300 (30°)

## Model & Dataset Paths

- **Source Trained Model Folder:** `single_dataset\A100_2p18e9_600km_70deg_30kHz\<model>`
- **Source Dataset**: `generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz`
- **Target Dataset Folder:** `generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps`

---

## Dataset Configurations

| Parameter | Source Dataset | Target Dataset |
| :--- | :--- | :--- |
| **Channel Model** | NTN TDL-A (NLOS) | DUR (Dense Urban, NLoS) |
| **Data Generator** | MATLAB | OpenNTN |
| **Carrier Frequency ($f_c$)** | 2.18 GHz | 2.18 GHz |
| **Subcarrier Spacing (SCS)** | 30 kHz | 30 kHz |
| **UE Velocity** | [20, 30] m/s | [20, 30] m/s |
| **Satellite Altitude** | 600 km | 600 km |
| **Elevation Angle** | 70° | 30° |
| **Delay Spread** | 100 ns | 300 ns |
| **Pilot Pattern (DM-RS)** | **5G NR DM-RS Type 2, Port 1:**<br>• Subcarriers port 1 (`[0, 1, 6, 7]`) per RB<br>• Mapping Type A, Pos 2 (Symbols 2 & 11, 0-indexed)<br>• 88 REs per slot | *(Identical to Source)*<br>• Subcarriers port 1 (`[0, 1, 6, 7]`) per RB<br>• Mapping Type A, Pos 2 (Symbols 2 & 11, 0-indexed)<br>• 88 REs per slot |

> [!NOTE]
> **Cross-Geometry & Delay-Spread Shift Experiment:**
> - **Elevation Angle:** Shifts from high-elevation $70^\circ$ (steep incidence) down to low-elevation $30^\circ$ (longer slant path, severe urban multipath/blockage).
> - **Delay Spread:** Increases from $100\text{ ns}$ to $300\text{ ns}$ (tripled delay spread, creating much stronger frequency-selective fading).
> - **Pilot Layout:** Both source and target maintain the **exact same pilot positions** (Type 2, Port 1, Pos 2), isolating domain shift to geometry and channel propagation characteristics.

