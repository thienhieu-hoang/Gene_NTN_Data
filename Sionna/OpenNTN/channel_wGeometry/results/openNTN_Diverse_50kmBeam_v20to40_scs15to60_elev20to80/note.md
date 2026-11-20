# Dataset Overview: openNTN_Diverse_50kmBeam_v20to40_scs15to60_elev20to80

## 1. General Information
- **Channel Generator:** OpenNTN 3GPP TR 38.811 GSCM via Sionna
- **Carrier Frequency:** 2.18 GHz (S-band)
- **Total Drops:** 20,000 (Train: 16,000, Val: 2,000, Test: 2,000)
- **Grid Size:** 132 Subcarriers × 14 OFDM Symbols (Fixed for model compatibility)
- **Data Type:** `complex64`, raw unnormalized channel matrices `(N, 132, 14)`

## 2. Diverse Multi-Environment Parameters
- **Scenarios:** Dense Urban (`dur`: 6336), Urban (`urb`: 7104), Suburban (`sur`: 6560)
- **Elevation Angle:** Continuous mixture from 20° to 80° (Mean: 49.7°)
- **LoS Ratio:** 27.7% (Physically coupled to elevation per 3GPP TR 38.811 Table 6.6.1-1)
- **Delay Spread ($	au_{\text{rms}}$):** [20.0 ns, 539.7 ns] (Mean: 173.0 ns)
- **Subcarrier Spacings:** Diverse mixture of 15 kHz, 30 kHz, and 60 kHz
- **Beam Footprint:** 50.0 km radius (100 km diameter wide beam)
- **UE Ground Velocity:** 20 m/s to 40 m/s (72 km/h to 144 km/h)
- **Doppler Model:** Beam-center precompensation with residual differential Doppler (Max residual: $\pm$4052.9 Hz)

## 3. Metadata Files
- `sample_metadata.mat`: MATLAB-compatible structure containing per-sample physics (`meta_scenario`, `meta_elev_deg`, `meta_is_los`, `meta_ds_ns`, `meta_scs_khz`, `meta_v_ue`, `meta_dist_center_m`, `meta_residual_doppler_hz`).
- `metadata_summary.json`: High-level statistical summary.
