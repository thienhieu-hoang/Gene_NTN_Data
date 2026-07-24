# Quick Start & Configuration Guide for NTN Channel Simulation

This guide explains how to configure and run the modified OpenNTN channel simulation scripts (`gen_channel_v2_wGeometry_straightDoppler.py` and `gen_channel_v3_wGeometry_straightDoppler_Compensate.py`).

---

## 1. Import Modified Scenario & Channel Classes

To use the 3D ENU vector Doppler features (per-cluster departure angles and real orbital mechanics), import the `_modify` classes from `OpenNTN`:

```python
from OpenNTN import DenseUrban_modify, Urban_modify, SubUrban_modify

# Map scenario strings to modified channel classes
scenario_classes = {
    "dur": DenseUrban_modify,
    "sur": SubUrban_modify,
    "urb": Urban_modify
}

channel_class = scenario_classes[scenario]
```

---

## 2. Configure Channel Model Parameters & Doppler Modes

Instantiate `channel_class` with `doppler_enabled=True` and select the desired `doppler_mode`:

```python
channel_model = channel_class(
    carrier_frequency = carrier_frequency,
    ut_array          = ut_array,
    bs_array          = bs_array,
    direction         = direction,
    elevation_angle   = elevation_angle,
    doppler_enabled   = True,                 # Enable Doppler calculation
    doppler_mode      = 'precompensated'      # Choose: 'full' | 'precompensated' | 'ue_only'
)
```

### Doppler Mode Descriptions:

| `doppler_mode` | Channel Type | Physical Behavior |
| :--- | :--- | :--- |
| **`'full'`** | **Straight / Uncompensated Channel** | Simulates the full, uncompensated 3D satellite Doppler ($\sim 200 - 400\text{ kHz}$) plus the UE mobility Doppler. Use this to model the raw physical link before satellite precompensation. |
| **`'precompensated'`** | **Effective Channel (Precompensated)** | Precompensates the bulk Doppler along the Satellite-to-Beam-Center link. The resulting channel contains the UE mobility Doppler plus the per-cluster satellite angular-spread residual ($\sim 10 - 50\text{ Hz}$). Use this to model the post-precompensation channel. |
| **`'ue_only'`** | **UE Mobility Only** | Ignores the satellite motion entirely, modeling only the UE ground movement Doppler shift. |

---

## 3. Set Up Topology with 3D Velocities

Pass both the UE 3D velocity (`v_UE_ENU`) and the Satellite 3D velocity (`v_sat_ENU`) into `set_topology`:

```python
# Convert ENU position and velocity arrays to TensorFlow constants
ut_loc_tensor        = tf.constant([[ut_loc_ENU]], dtype=tf.float32)         # [1, 1, 3]
bs_loc_tensor        = tf.constant([[bs_loc_ENU]], dtype=tf.float32)         # [1, 1, 3]
ut_orientations      = tf.zeros([1, 1, 3])                                    # [1, 1, 3]
bs_orientations      = tf.zeros([1, 1, 3])                                    # [1, 1, 3]
ut_velocities_tensor = tf.constant([[v_UE_ENU]], dtype=tf.float32)           # [1, 1, 3] UE Velocity
bs_velocities_tensor = tf.constant([[v_sat_ENU]], dtype=tf.float32)          # [1, 1, 3] SAT Velocity (ENU)
in_state             = tf.constant([[False]], dtype=tf.bool)                 # [1, 1]

# Pack topology tuple
topology_data = (
    ut_loc_tensor,
    bs_loc_tensor,
    ut_orientations,
    bs_orientations,
    ut_velocities_tensor,
    bs_velocities_tensor,    # Pass 3D satellite velocity vector
    in_state
)

# Apply custom ENU geometry to the topology
channel_model.set_topology(*topology_data, los=True)
```

---

## 4. Customizing Beam Center for Off-Center Precompensation (Optional)

By default, when `doppler_mode='precompensated'`, the beam center is assumed to track the UE directly ($\text{BC} = \text{UE}$). 

If you want to simulate a UE located off-center (e.g. at the beam edge), set a custom beam center location in ENU coordinates before generating channel coefficients:

```python
# Define beam center offset relative to UE (e.g. 100 meters East)
beam_center_offset = np.array([100.0, 0.0, 0.0])
beam_center_ENU    = ut_loc_ENU + beam_center_offset

# Set custom beam center on channel model
channel_model.set_beam_center(beam_center_ENU)
```

---

## 5. Generate Channel Coefficients

Generate path coefficients and channel impulse responses:

```python
# Generate channel path coefficients and delays
num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)

# Generate OFDM Channel Matrix
ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
h_freq = ofdm_channel()         # Shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym, num_subcarriers]
```