# Dataset Overview: openNTN_KaBand_120kHz_20GHz_Diverse_50kmBeam

## 1. General Information
- **Channel Generator:** OpenNTN 3GPP TR 38.811 GSCM via Sionna
- **Carrier Frequency:** 20.00 GHz (3GPP Rel-17/18 Ka-band NTN)
- **Subcarrier Spacing (SCS):** 120 kHz (Numerology $\mu = 3$)
- **Active Bandwidth:** 15.84 MHz (132 subcarriers = 11 Resource Blocks)
- **Total Drops:** 20,000 (Train: 16,000, Val: 2,000, Test: 2,000)
  - **Train (`H_train.npy`):** First 16,000 samples (`SampleIdx` 1 – 16,000, 0-based indices `[0:16000]`)
  - **Val (`H_val.npy`):** Next 2,000 samples (`SampleIdx` 16,001 – 18,000, 0-based indices `[16000:18000]`)
  - **Test (`H_test.npy`):** Next 2,000 samples (`SampleIdx` 18,001 – 20,000, 0-based indices `[18000:20000]`)
- **Grid Size:** 132 Subcarriers × 14 OFDM Symbols (Fixed for model compatibility)
- **Data Type:** `complex64`, raw unnormalized channel matrices `(N, 132, 14)`

## 2. Multi-Environment & Multi-Ripple Fading Parameters
- **Scenarios:** Dense Urban (`dur`: 6624), Urban (`urb`: 7488), Suburban (`sur`: 5888)
- **Elevation Angle:** Continuous mixture from 20.0° to 80.0° (Mean: 50.1°)
- **LoS Ratio:** 31.7% (Physically coupled per 3GPP TR 38.811 Table 6.6.1-1)
- **Delay Spread ($\tau_{\text{rms}}$):** [20.0 ns, 463.4 ns] (Mean: 162.0 ns)
- **Multi-Ripple Fading Span:** Across the 15.84 MHz bandwidth, a 162 ns delay spread produces $\Delta\Phi \approx 2.6$ full fading cycles across the 132 subcarriers (prominent horizontal stripes).
- **Beam Footprint:** 50.0 km radius (100 km diameter wide beam)
- **UE Ground Velocity:** 10.0 m/s to 35.0 m/s (36 km/h to 126 km/h)
- **Doppler Model:** Beam-center precompensation with residual differential Doppler (Max residual: $\pm$38457.7 Hz)

## 3. Metadata Files & Split Order Mapping
- `sample_metadata.csv`: Ground-truth parameters per sample for pandas / Python analysis. Contains all 20,000 drops in sequential order:
  | Split | `.npy` File | `SampleIdx` (1-based CSV) | Python Slice `[start:stop]` | Count |
  | :--- | :--- | :--- | :--- | :--- |
  | **Train** | `H_train.npy` | `1` to `16000` | `[0 : 16000]` | 16,000 |
  | **Val** | `H_val.npy` | `16001` to `18000` | `[16000 : 18000]` | 2,000 |
  | **Test** | `H_test.npy` | `18001` to `20000` | `[18000 : 20000]` | 2,000 |
- `H_train.npy`, `H_val.npy`, `H_test.npy`: Pure channel tensor splits `(N, 132, 14)` matching the exact sequential rows above.
