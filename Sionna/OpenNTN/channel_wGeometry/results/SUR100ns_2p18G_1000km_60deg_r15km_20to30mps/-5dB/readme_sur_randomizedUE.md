# Channel & Geometry Generation Settings - SUR (Randomized UE)

- **Scenario Type**: SUR (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: 2.18 GHz
- **Link Direction**: downlink
- **Satellite (LEO) Height**: 1000 km
- **Configured Target Elevation Angle**: 60.0°
- **Nominal Beam-Center Elevation Angle**: 60.47° (Snapshot time t = 76.9 s)
- **Subcarrier Spacing (SCS)**: 30 kHz
- **FFT Size**: 256
- **Active Subcarriers**: 132 (out of 256)
- **SNR (for LS estimation)**: -5 dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 7, 11]
- **Total Samples Generated**: 16
- **Target Delay Spread Configuration**: 100.0 ns (Custom Overridden)
- **Average RMS Delay Spread (Realized)**: 492.34 ns (Range: [0.57, 3277.48] ns)

## Satellite Orbital Pass & Elevation Angle Timeline
- **Pass Start (t_start = -379.2 s)**: Elevation = 11.63° (Horizon Rise)
- **Peak Zenith (t_peak = 0.0 s)**: Elevation = 88.64° (Overhead Peak)
- **Snapshot Point (t_snap = 76.9 s)**: Elevation = 60.47° (Single Position Generated)
- **Pass End (t_end = 379.2 s)**: Elevation = 11.26° (Horizon Set)

## Spatial Elevation Variation Across 15km Beam Footprint (16 UEs)
- **UE Farthest from Satellite (Min Elevation)**: 59.86°
- **UE Closest to Satellite (Max Elevation)**: 61.07°
- **Average Across All UEs (Mean Elevation)**: 60.53°

## Satellite (LEO) Settings (Fixed Snapshot)
- **Temporal State**: Single snapshot at orbital time $t = 76.9$ seconds
- **Satellite Position (ENU)**: Fixed at [378598.55, 407992.61, 988010.41] meters
- **Satellite Velocity Vector (ENU)**: Fixed at [4930.95, 5007.25, -517.26] m/s (Speed: 7046.58 m/s)

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
