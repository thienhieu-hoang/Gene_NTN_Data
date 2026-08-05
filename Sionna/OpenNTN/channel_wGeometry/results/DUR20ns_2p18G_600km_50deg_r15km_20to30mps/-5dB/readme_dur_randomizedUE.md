# Channel & Geometry Generation Settings - DUR (Randomized UE)

- **Scenario Type**: DUR (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: 2.18 GHz
- **Link Direction**: downlink
- **Satellite (LEO) Height**: 600 km
- **Configured Target Elevation Angle**: 50.0°
- **Nominal Beam-Center Elevation Angle**: 50.51° (Snapshot time t = 64.8 s)
- **Subcarrier Spacing (SCS)**: 30 kHz
- **FFT Size**: 256
- **Active Subcarriers**: 132 (out of 256)
- **SNR (for LS estimation)**: -5 dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 7, 11]
- **Total Samples Generated**: 16
- **Target Delay Spread Configuration**: 20.0 ns (Custom Overridden)
- **Average RMS Delay Spread (Realized)**: 68.44 ns (Range: [5.10, 301.18] ns)

## Satellite Orbital Pass & Elevation Angle Timeline
- **Pass Start (t_start = -255.0 s)**: Elevation = 11.42° (Horizon Rise)
- **Peak Zenith (t_peak = 0.0 s)**: Elevation = 87.86° (Overhead Peak)
- **Snapshot Point (t_snap = 64.8 s)**: Elevation = 50.51° (Single Position Generated)
- **Pass End (t_end = 255.0 s)**: Elevation = 10.99° (Horizon Set)

## Spatial Elevation Variation Across 15km Beam Footprint (16 UEs)
- **UE Farthest from Satellite (Min Elevation)**: 49.75°
- **UE Closest to Satellite (Max Elevation)**: 51.27°
- **Average Across All UEs (Mean Elevation)**: 50.58°

## Satellite (LEO) Settings (Fixed Snapshot)
- **Temporal State**: Single snapshot at orbital time $t = 64.8$ seconds
- **Satellite Position (ENU)**: Fixed at [329876.79, 356211.05, 592034.83] meters
- **Satellite Velocity Vector (ENU)**: Fixed at [5101.55, 5156.86, -490.22] m/s (Speed: 7270.44 m/s)

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
