# Channel & Geometry Generation Settings - DUR (Randomized UE)

- **Scenario Type**: DUR (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: 27.00 GHz
- **Link Direction**: downlink
- **Satellite (LEO) Height**: 600 km
- **Satellite Nominal Elevation Angle**: 87.86 degrees
- **Subcarrier Spacing (SCS)**: 60 kHz
- **FFT Size**: 256
- **Active Subcarriers**: 132 (out of 256)
- **SNR (for LS estimation)**: 20.0 dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 7, 11]
- **Total Samples Generated**: 16
- **Average RMS Delay Spread**: 3.49 ns (Range: [1.39, 7.34] ns)

## UE Randomization Settings
- **Generation Method**: Randomized UE Positions and Velocities (GPU Mini-Batched)
- **Position Area (ENU)**: 
  * East (X): [-1000.0, 1000.0] meters (Span: 2x2 km)
  * North (Y): [-1000.0, 1000.0] meters
  * Height (Z): 1.5 meters above ground
- **Velocity (ENU)**:
  * Speed Range: [10.0, 30.0] m/s
  * Heading (Direction): Randomized uniformly over [0, 360] degrees (full direction randomization)
