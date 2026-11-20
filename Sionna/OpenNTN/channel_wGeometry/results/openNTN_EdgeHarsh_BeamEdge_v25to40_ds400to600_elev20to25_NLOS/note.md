# Dataset Overview: openNTN_EdgeHarsh_BeamEdge_v25to40_ds400to600_elev20to25_NLOS

## 1. General Information
- **Channel Generator:** OpenNTN 3GPP TR 38.811 GSCM via Sionna
- **Carrier Frequency:** 2.18 GHz (S-band)
- **Total Drops:** 20,000 (Train: 16,000, Val: 2,000, Test: 2,000)
- **Grid Size:** 132 Subcarriers × 14 OFDM Symbols (Fixed for model compatibility)
- **Data Type:** `complex64`, raw unnormalized channel matrices `(N, 132, 14)`

## 2. Multi-Environment Parameters
- **Scenarios:** Dense Urban (`dur`: 5952), Urban (`urb`: 6784), Suburban (`sur`: 7264)
- **Elevation Angle:** Continuous mixture from 20.0° to 25.0° (Mean: 22.5°)
- **LoS Ratio:** 0.0% (Condition: nlos)
- **Delay Spread ($	au_{\text{rms}}$):** [401.8 ns, 597.2 ns] (Mean: 503.8 ns)
- **Subcarrier Spacings:** Diverse mixture of ['15 kHz', '30 kHz', '60 kHz']
- **Beam Footprint:** 50.0 km radius (Outer ring ratio: 0.70)
- **UE Ground Velocity:** 25.0 m/s to 40.0 m/s (90 km/h to 144 km/h)
- **Doppler Model:** Beam-center precompensation with residual differential Doppler (Max residual: $\pm$782.8 Hz)

## 3. Metadata Files
- `sample_metadata.mat`: MATLAB-compatible structure containing per-sample physics (`meta_scenario`, `meta_elev_deg`, `meta_is_los`, `meta_ds_ns`, `meta_scs_khz`, `meta_v_ue`, `meta_dist_center_m`, `meta_residual_doppler_hz`).
- `metadata_summary.json`: High-level statistical summary.
