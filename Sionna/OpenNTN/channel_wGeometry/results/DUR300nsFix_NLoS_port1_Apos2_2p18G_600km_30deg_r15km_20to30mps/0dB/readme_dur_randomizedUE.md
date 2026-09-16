# Channel & Geometry Generation Settings - DUR (Randomized UE)

- **Scenario Type**: DUR (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Propagation Condition**: Only NLoS (Non-Line-of-Sight, Rayleigh Fading)
- **Carrier Frequency**: 2.18 GHz
- **Link Direction**: downlink
- **Satellite (LEO) Height**: 600 km
- **Configured Target Elevation Angle**: 30.0°
- **Nominal Beam-Center Elevation Angle**: 31.00° (Snapshot time t = 123.6 s)
- **Subcarrier Spacing (SCS)**: 30 kHz
- **FFT Size**: 256
- **Active Subcarriers**: 132 (out of 256)
- **SNR (for LS estimation)**: 0 dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 11] (typeAposition=2)
- **Pilot Density**: Sparse (88 pilots, port 1: subcarrier mod 6 = 0 or 1 on symbols 2 and 11)
- **Total Samples Generated**: 2048
- **Target Delay Spread Configuration**: 300.0 ns (Custom Overridden, Fixed)
- **Average RMS Delay Spread (Realized)**: 230.45 ns (Range: [4.94, 792.78] ns)

## Satellite Orbital Pass & Elevation Angle Timeline
- **Pass Start (t_start = -255.0 s)**: Elevation = 11.42° (Horizon Rise)
- **Peak Zenith (t_peak = 0.0 s)**: Elevation = 87.86° (Overhead Peak)
- **Snapshot Point (t_snap = 123.6 s)**: Elevation = 31.00° (Single Position Generated)
- **Pass End (t_end = 255.0 s)**: Elevation = 10.99° (Horizon Set)

## Spatial Elevation Variation Across 15km Beam Footprint (2048 UEs)
- **UE Farthest from Satellite (Min Elevation)**: 30.48°
- **UE Closest to Satellite (Max Elevation)**: 31.54°
- **Average Across All UEs (Mean Elevation)**: 31.00°

## Satellite (LEO) Settings (Fixed Snapshot)
- **Temporal State**: Single snapshot at orbital time $t = 123.6$ seconds
- **Satellite Position (ENU)**: Fixed at [629944.09, 657775.75, 550161.73] meters
- **Satellite Velocity Vector (ENU)**: Fixed at [5100.36, 5095.49, -933.35] m/s (Speed: 7269.72 m/s)

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
