# Channel & Geometry Generation Settings - SUR (Randomized UE)

- **Scenario Type**: SUR (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: 20.00 GHz
- **Link Direction**: downlink
- **Satellite (LEO) Height**: 1000 km
- **Configured Target Elevation Angle**: 70.0°
- **Nominal Beam-Center Elevation Angle**: 70.01° (Snapshot time t = 49.1 s)
- **Subcarrier Spacing (SCS)**: 120 kHz
- **FFT Size**: 256
- **Active Subcarriers**: 132 (out of 256)
- **SNR (for LS estimation)**: -5 dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 7, 11]
- **Total Samples Generated**: 16
- **Target Delay Spread Configuration**: 100.0 ns (Custom Overridden)
- **Average RMS Delay Spread (Realized)**: 67.20 ns (Range: [4.16, 199.05] ns)

## Satellite Orbital Pass & Elevation Angle Timeline
- **Pass Start (t_start = -379.2 s)**: Elevation = 11.63° (Horizon Rise)
- **Peak Zenith (t_peak = 0.0 s)**: Elevation = 88.64° (Overhead Peak)
- **Snapshot Point (t_snap = 49.1 s)**: Elevation = 70.01° (Single Position Generated)
- **Pass End (t_end = 379.2 s)**: Elevation = 11.26° (Horizon Set)

## Spatial Elevation Variation Across 15km Beam Footprint (16 UEs)
- **UE Farthest from Satellite (Min Elevation)**: 69.32°
- **UE Closest to Satellite (Max Elevation)**: 70.68°
- **Average Across All UEs (Mean Elevation)**: 70.07°

## Satellite (LEO) Settings (Fixed Snapshot)
- **Temporal State**: Single snapshot at orbital time $t = 49.1$ seconds
- **Satellite Position (ENU)**: Fixed at [241473.41, 268369.58, 999802.91] meters
- **Satellite Velocity Vector (ENU)**: Fixed at [4925.63, 5028.78, -330.38] m/s (Speed: 7046.95 m/s)

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
