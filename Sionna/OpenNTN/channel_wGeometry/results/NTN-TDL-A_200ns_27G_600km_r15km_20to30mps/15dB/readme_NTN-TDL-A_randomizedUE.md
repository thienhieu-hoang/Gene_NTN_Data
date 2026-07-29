# Channel & Geometry Generation Settings - NTN-TDL-A (Randomized UE)

- **TDL Model Profile**: NTN-TDL-A (3GPP TR 38.811 NTN Profile)
- **Carrier Frequency**: 27.00 GHz
- **Link Direction**: downlink
- **Satellite (LEO) Height**: 600 km
- **Satellite Nominal Elevation Angle**: 87.86 degrees
- **Subcarrier Spacing (SCS)**: 60 kHz
- **FFT Size**: 256
- **Active Subcarriers**: 132 (out of 256)
- **SNR (for LS estimation)**: 15 dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 7, 11]
- **Total Samples Generated**: 16
- **Target Delay Spread Configuration**: 200.0 ns (Custom Overridden)
- **Average RMS Delay Spread (Realized)**: 181.83 ns (Range: [143.45, 206.31] ns)

## Satellite (LEO) Settings (Fixed Snapshot)
- **Temporal State**: Single snapshot at orbital closest approach ($t = 0$ seconds)
- **Satellite Position (ENU)**: Fixed at [0.00, 20699.48, 607915.78] meters
- **Satellite Velocity Vector (ENU)**: Fixed at [5082.21, 5200.24, -0.00] m/s (Speed: 7271.27 m/s)

## Beam Boresight & Footprint Settings
- **Beam Center (ECEF)**: [-2706217.22, -4261126.21, 3885786.75] meters
- **Beam Center (ENU)**: [0.00, 0.00, 0.00] meters (Origin of local tangent plane)
- **Beam Footprint Radius**: 15.0 km

## UE Randomization Settings
- **Generation Method**: Randomized UE Positions and Velocities (GPU Mini-Batched)
- **Position Area (ENU)**: 
  * Shape: Uniformly distributed inside a circle of radius 14.50 km around the beam center
  * Height (Z): 1.5 meters above ground
- **Velocity (ENU)**:
  * Speed Range: [20.0, 30.0] m/s
  * Heading (Direction): Randomized uniformly over [0, 360] degrees (full direction randomization across all generated samples)
