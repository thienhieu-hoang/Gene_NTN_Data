# Channel & Geometry Generation Settings - DUR (Randomized UE)

- **Scenario Type**: DUR (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: 2.18 GHz
- **Link Direction**: downlink
- **Satellite (LEO) Height**: 600 km
- **Configured Target Elevation Angle**: 70.0°
- **Nominal Beam-Center Elevation Angle**: 69.52° (Snapshot time t = 28.7 s)
- **Subcarrier Spacing (SCS)**: 30 kHz
- **FFT Size**: 256
- **Active Subcarriers**: 132 (out of 256)
- **SNR (for LS estimation)**: -5 dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 11]
- **Total Samples Generated**: 1024
- **Target Delay Spread Configuration**: 100.0 ns (Custom Overridden, Fixed)
- **Average RMS Delay Spread (Realized)**: 81.04 ns (Range: [7.38, 325.68] ns)

## Satellite Orbital Pass & Elevation Angle Timeline
- **Pass Start (t_start = -255.0 s)**: Elevation = 11.42° (Horizon Rise)
- **Peak Zenith (t_peak = 0.0 s)**: Elevation = 87.86° (Overhead Peak)
- **Snapshot Point (t_snap = 28.7 s)**: Elevation = 69.52° (Single Position Generated)
- **Pass End (t_end = 255.0 s)**: Elevation = 10.99° (Horizon Set)

## Spatial Elevation Variation Across 15km Beam Footprint (1024 UEs)
- **UE Farthest from Satellite (Min Elevation)**: 68.21°
- **UE Closest to Satellite (Max Elevation)**: 70.87°
- **Average Across All UEs (Mean Elevation)**: 69.53°

## Satellite (LEO) Settings (Fixed Snapshot)
- **Temporal State**: Single snapshot at orbital time $t = 28.7$ seconds
- **Satellite Position (ENU)**: Fixed at [146216.87, 169917.38, 604787.78] meters
- **Satellite Velocity Vector (ENU)**: Fixed at [5093.45, 5184.15, -217.68] m/s (Speed: 7270.90 m/s)

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
