To set the simulation run with straight Doppler between sat and UE: 

```scenario_classes = {
    "dur": DenseUrban_modify,   <----
    "sur": SubUrban_modify,     <----
    "urb": Urban_modify         <----
}
channel_class = scenario_classes[scenario]
channel_model = channel_class(carrier_frequency=carrier_frequency,
                              ut_array=ut_array,
                              bs_array=bs_array,
                              direction=direction,
                              elevation_angle=elevation_angle,
                              doppler_enabled=True,      <----
                              doppler_mode='full')       <----
```


Here is the step-by-step chronological execution flow that occurs when you run [gen_channel_v2_wGeometry_straightDoppler.py]:

---

### Step 1: Instantiation
* **Where**: [gen_channel_v2_wGeometry_straightDoppler.py](Gene_NTN_Data/Sionna/OpenNTN/channel_wGeometry/gen_channel_v2_wGeometry_straightDoppler.py#L208-L214)
* **Action**:
  You instantiate the modified scenario and channel classes:
  `channel_model = DenseUrban_modify(..., doppler_enabled=True, doppler_mode='full')`
* **Under the Hood**:
  1. `DenseUrban_modify` (defined in [dense_urban.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/dense_urban.py)) instantiates `DenseUrbanScenario_modify` (defined in [dense_urban_scenario.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/dense_urban_scenario.py)).
  2. The scenario constructor calls `SystemLevelScenario_modify.__init__()` (in [system_level_scenario.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py)), storing the properties:
     `self._doppler_mode = 'full'`
     `self._bs_velocities = None`

---

### Step 2: Topology Setup
* **Where**: [gen_channel_v2_wGeometry_straightDoppler.py](Gene_NTN_Data/Sionna/OpenNTN/channel_wGeometry/gen_channel_v2_wGeometry_straightDoppler.py#L230)
* **Action**:
  You call:
  `channel_model.set_topology(*topology_data, los=True)`
* **Under the Hood**:
  1. This enters `SystemLevelChannel_modify.set_topology()` (in [system_level_channel.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py)).
  2. It intercepts and calls the scenario's modified topology method:
     `self._scenario.set_topology_modify(...)`
  3. `SystemLevelScenario_modify.set_topology_modify()` (in [system_level_scenario.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py)) stores the 3D satellite velocity:
     `self._bs_velocities = bs_velocities` (which is your `v_sat_ENU`)
     It then calls the original `set_topology()` method to set the locations.

---

### Step 3: Simulation Call
* **Where**: [gen_channel_v2_wGeometry_straightDoppler.py](Gene_NTN_Data/Sionna/OpenNTN/channel_wGeometry/gen_channel_v2_wGeometry_straightDoppler.py#L234)
* **Action**:
  You execute the channel coefficients generator:
  `path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)`
* **Under the Hood**:
  1. This calls `SystemLevelChannel_modify.__call__()` (in [system_level_channel.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py)).
  2. Rather than using the original `Topology` class, it constructs a **`Topology_modify`** object:
     ```python
     topology = Topology_modify(
         velocities=self._scenario.ut_velocities,
         bs_velocities=self._scenario.bs_velocities,  # v_sat_ENU passed here!
         doppler_mode=self._scenario.doppler_mode,    # 'full' passed here!
         ...
     )
     ```
  3. `Topology_modify.__init__()` (in [channel_coefficients.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py)) stores `self.bs_velocities = bs_velocities`. The randomized orbit direction lines of the original class are completely bypassed.

---

### Step 4: Step 11 Doppler Matrix Generation
* **Where**: [system_level_channel.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py#L554)
* **Action**:
  Inside `SystemLevelChannel_modify.__call__()`, the simulator runs the channel impulse response generator:
  `self._cir_sampler(..., topology, ...)`
  where `self._cir_sampler` is **`ChannelCoefficientsGenerator_modify`**.
* **Under the Hood**:
  1. `ChannelCoefficientsGenerator_modify` runs its overridden **`_step_11_doppler_matrix()`** method (in [channel_coefficients.py](Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py)).
  2. The code checks:
     `if is_ntn and doppler_mode in ('full', 'precompensated') and bs_velocities is not None:`
     (This evaluates to `True` because altitude $\ge 600\text{ km}$, mode is `'full'`, and `bs_velocities` is present).
  3. Instead of adding a uniform randomized scalar to all paths, it computes the **true projection** of the satellite's velocity vector onto the departure angle vector of each cluster $l$:
     `exp_sat_cl = 2*PI/lambda_0 * tf.reduce_sum(r_hat_sat_cl * v_sat_bar, -2) * t`
  4. Because `doppler_mode == 'full'`, it directly adds this uncompensated satellite Doppler term to the UE's local Doppler term:
     `exponent = exponent + exp_sat_cl`
  5. The resulting array is returned as `h_doppler` and applied to the fading paths.


---

# Summary Doppler different

Both the old and new code, the satellite Doppler shift is ultimately applied to **all rays**. However, the **mathematical representation and the angles used** are completely different.

Here is the side-by-side comparison of the old vs. the new logic:

---

### 1. The Old Version (Uniform Scalar, No Real Angles)
In the original OpenNTN library, the satellite Doppler is a **single, flat number** for the entire simulation drop.

* **How it was calculated**:
  The code took the satellite's scalar speed ($v_{\text{sat}} \approx 7600\text{ m/s}$), projected it onto the elevation angle, and multiplied it by a **random angle** to represent a random orbit direction.
  $$f_{D,\text{sat}} = \frac{v_{\text{sat}}}{\lambda_0} \cos(\theta_{\text{elevation}}) \cos(\theta_{\text{random\_direction}})$$
  This produces a single scalar value (e.g., $+250\text{ kHz}$).
* **How it was applied**:
  The library broadcasted (copied) this **exact same number** to every single cluster and every single ray:
  ```python
  rotation_for_time = tf.broadcast_to(rotation_for_time, tf.shape(exponent))
  ```
* **The Problem**: 
  **No physical departure angles were used.** Whether a path departed the satellite at $+5^\circ$ or $-5^\circ$ relative to the direct LoS path, it received the exact same $+250\text{ kHz}$ shift.

---

### 2. The New Version (3D Vector Projection, Per-Cluster Angles)
In our modified code, the satellite Doppler is calculated using the **satellite's actual 3D velocity vector** and the **specific 3D angle** of each path.

* **How it is calculated**:
  For each cluster $l$, the code look up its specific departure angles: Azimuth of Departure (AoD) and Zenith of Departure (ZoD). It creates a 3D unit vector pointing from the satellite to that scatter cluster:
  $$\hat{r}_{\text{dep},l} = \text{unit\_sphere\_vector}(\text{zod}_l, \text{aod}_l)$$
  Then, it computes the Doppler shift by taking the **3D dot product** between the satellite's real 3D velocity vector ($\vec{v}_{\text{sat}}$) and that path's departure vector:
  $$\text{Doppler}_{\text{sat}, l} = \frac{1}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep},l} \right)$$
* **How it is applied**:
  * **Different clusters get different satellite Doppler shifts** because their departure vectors ($\hat{r}_{\text{dep}, l}$) are different.
  * Within a single cluster $l$, the 20 sub-rays share that cluster's center departure direction, meaning the 20 sub-rays of Cluster 1 share one satellite Doppler shift, and the 20 sub-rays of Cluster 2 share another.

---

### Comparison Summary

| Feature | Old Version | New Version (Modified) |
| :--- | :--- | :--- |
| **Satellite Velocity** | Randomized scalar approximation | Real 3D velocity vector ($\vec{v}_{\text{sat}}$) from orbit dynamics |
| **Path Angles Used** | **None** (same scalar broadcasted to all) | **Specific departure angles (AoD, ZoD)** per cluster |
| **Physical Effect** | All multipaths shift by the exact same frequency | Different multipath clusters experience slightly different Doppler shifts |