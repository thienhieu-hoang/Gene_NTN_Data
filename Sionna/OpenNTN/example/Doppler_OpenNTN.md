*(similar to MATLAB - separate **SAT-BC** (Beam Center), **BC-UE**)* 


Based on the source code of Sionna/OpenNTN, here is how the Doppler shift is modeled, how `doppler_enabled` operates, and how it maps to the MATLAB pre-compensation model.

---

## Separate Dopplers
### 1. Are the satellite and UE Dopplers added separately?
**Yes.** 
In [channel_coefficients.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py#L604-L632), the Doppler phase exponent is computed as:
```python
# 1. Compute phase shift due to UE mobility (Local UE Doppler)
exponent = 2*PI/lambda_0*tf.reduce_sum(r_hat_ut*v_uts_bar, -2)*t

# 2. Compute phase shift due to LEO satellite movement (if enabled)
if tf.logical_and(tf.less_equal(600000.0, topology.bs_height), topology.doppler_enabled):
    max_sat_speed_for_elevation_angle = topology.sat_speed
    max_rotation_per_time = (2.0*PI/lambda_0)*max_sat_speed_for_elevation_angle
    rotation_for_time = tf.matmul(tf.expand_dims(max_rotation_per_time,axis=1), tf.expand_dims(t,axis=0))
    ...
    # Add the satellite Doppler exponent to the local UE mobility exponent
    exponent = tf.math.add(exponent, rotation_for_time)
```
*   **UE Doppler**: Modeled as a time-varying phase rotation for each ray individually (`reduce_sum(r_hat_ut * v_uts_bar, -2) * t`), where $r\_hat\_ut$ is the ray arrival angle vector and $v\_uts\_bar$ is the UE velocity vector. This represents local multipath scattering.
*   **Satellite Doppler**: Modeled as a common, flat phase offset `rotation_for_time` ($2\pi f_{d,\text{sat}} t$) added uniformly to the exponent of all rays.
*   Because the two exponents are summed together ($e^{j(\phi_{\text{UE}}(t) + \phi_{\text{sat}}(t))}$), they are mathematically multiplied in the complex domain (cascaded).

---

### 2. Does the code not consider the Doppler between UE and satellite straightly?
**Correct, it does not.**

Just like the MATLAB toolbox, OpenNTN splits the Doppler into two decoupled components:
1.  **UE-to-Beam-Center (Local UE mobility)**: Evaluated using the UE's local velocity vector and local angle-of-arrival rays.
2.  **Satellite-to-Beam-Center (Orbit Doppler)**: Evaluated as a single, static frequency translation based on a flat projection of the satellite's orbital speed at the configured elevation angle:
    ```python
    max_sat_speed_for_elevation_angle = tf.math.cos(elevation_angle * (PI/180.0)) * sat_speed
    self.sat_speed = tf.math.cos(random_direction_per_batch) * max_sat_speed_for_elevation_angle
    ```
There is no sample-wise or time-varying coordinate/geometry tracking to compute a unified relative velocity vector between the satellite and the UE.

---

### 3. What does `doppler_enabled=False` mean?
When you set `doppler_enabled=False` in `DenseUrban(...)`:
*   The check `if tf.logical_and(..., topology.doppler_enabled)` evaluates to `False`.
*   As a result, **the satellite-induced common Doppler phase shift (`rotation_for_time`) is bypassed and set to 0**.
*   However, the local UE mobility Doppler (`exponent = 2*PI/lambda_0*tf.reduce_sum(...)`) **remains active** as long as the UE has a non-zero velocity.

---

### 4. Does setting `doppler_enabled=False` equal the effective channel of the Doppler pre-compensation in the MATLAB code?
**Yes, they are mathematically identical.**

*   **In MATLAB**: The transmitter pre-compensates by multiplying the waveform by $e^{-j 2\pi f_{d,\text{sat}} t}$. The channel adds $+f_{d,\text{sat}}$ Doppler, and the two cancel out ($e^{-j \theta} \cdot e^{+j \theta} = 1$). The effective channel seen at the receiver has **zero LEO satellite Doppler** (only local UE fading remains).
*   **In Sionna/OpenNTN (`doppler_enabled=False`)**: The channel model simply doesn't add the $+f_{d,\text{sat}}$ term in the first place ($f_{d,\text{sat}} = 0$). 
*   In both cases, the LEO satellite's relative frequency offset is completely removed from the simulation, and the receiver experiences a channel where the only dynamic temporal variations are caused by the local UE ground mobility.

---

# Realistic Doppler 

Doppler straightly between SAT and UE 

Yes, **it is entirely possible** to modify the library to model the Doppler shift directly (straightly) between the UE and the satellite. This change would make the simulation physically consistent and spatially accurate (especially for multi-user scenarios).

Here is a breakdown of how it works and which files and lines of code would need to change:

---

### 1. Why this is more correct/practical
In a realistic system:
*   The satellite moves at velocity $\mathbf{v}_{\text{sat}}(t)$ and the UE moves at velocity $\mathbf{v}_{\text{UE}}(t)$.
*   For any ray $k$ propagating between them, the true Doppler shift is determined by projecting the relative velocity onto the unit direction vector of the ray:
    $$f_{d,k}(t) = -\frac{\left(\mathbf{v}_{\text{sat}}(t) - \mathbf{v}_{\text{UE}}(t)\right) \cdot \mathbf{\hat{r}}_{k}(t)}{\lambda}$$
*   By modeling this directly, when you perform pre-compensation at the transmitter based on the **beam center** Doppler, the remaining (residual) Doppler seen at the receiver will naturally and correctly include:
    1.  The local UE mobility Doppler spread.
    2.  The **residual satellite Doppler** caused by the geographic offset between the UE's position and the beam center.

---

### 2. Which lines of code need to be changed?

All modifications are located inside [channel_coefficients.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py).

#### Change 1: In the `Topology` class initialization (Lines 98–103)
*   **Current code**: The code simplifies and randomizes the satellite speed projection using a random angle per batch:
    ```python
    sat_speed = compute_satellite_speed(bs_height)
    max_sat_speed_for_elevation_angle = tf.math.cos(elevation_angle * (PI/180.0)) * sat_speed
    batch_size = tf.shape(los_aoa)[0]
    random_direction_per_batch = tf.random.uniform(minval=0, maxval=2*PI, shape = [batch_size])
    self.sat_speed = tf.math.cos(random_direction_per_batch) * max_sat_speed_for_elevation_angle
    ```
*   **Modification**: Replace this block. Instead of using a randomized scalar, pass the actual **3D velocity vector** of the satellite ($\mathbf{v}_{\text{sat}}$) and the UE ($\mathbf{v}_{\text{UE}}$) in a shared coordinate system (like ECEF).

---

#### Change 2: Inside `_step_11_doppler_matrix` (Lines 609–629)
*   **Current code**: The code calculates the satellite phase rotation as a flat, common term broadcasted uniformly to all rays:
    ```python
    max_sat_speed_for_elevation_angle = topology.sat_speed
    max_rotation_per_time = (2.0*PI/lambda_0)*max_sat_speed_for_elevation_angle
    rotation_for_time = tf.matmul(tf.expand_dims(max_rotation_per_time,axis=1), tf.expand_dims(t,axis=0))
    ...
    exponent = tf.math.add(exponent, rotation_for_time)
    ```
*   **Modification**: 
    1.  Remove the separate `if tf.logical_and(..., topology.doppler_enabled)` block that adds `rotation_for_time` as a separate flat term.
    2.  Modify the main exponent calculation (around line 606) to use the **relative velocity vector** ($\mathbf{v}_{\text{rel}} = \mathbf{v}_{\text{sat}} - \mathbf{v}_{\text{UE}}$) projected onto the individual unit sphere ray vectors (`r_hat_ut`):
        ```python
        # Project the combined relative velocity vector directly onto each ray
        exponent = 2*PI/lambda_0 * tf.reduce_sum(r_hat_ut * v_relative, -2) * t
        ```
    3.  This applies the Doppler shift **directly to each ray of each path of the channel**, naturally incorporating both the satellite and UE velocities.