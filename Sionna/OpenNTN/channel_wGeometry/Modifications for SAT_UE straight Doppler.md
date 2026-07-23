# A. Straight Doppler SAT-UE
To set the simulation run with straight Doppler between sat and UE: 

```
scenario_classes = {
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

# B. Summary Doppler different

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

| Feature                | Old Version                                      | New Version (Modified)                                                    |
| :--------------------- | :----------------------------------------------- | :------------------------------------------------------------------------ |
| **Satellite Velocity** | Randomized scalar approximation                  | Real 3D velocity vector ($\vec{v}_{\text{sat}}$) from orbit dynamics      |
| **Path Angles Used**   | **None** (same scalar broadcasted to all)        | **Specific departure angles (AoD, ZoD)** per cluster                      |
| **Physical Effect**    | All multipaths shift by the exact same frequency | Different multipath clusters experience slightly different Doppler shifts |

# C. SAT-BC Doppler Compensation to the Straight SAT-UE Doppler

When `doppler_mode='precompensated'` is set in the simulation, the gateway/satellite precompensates for the bulk Doppler along the direct Line-of-Sight (LoS) path. The simulator then computes the residual Doppler from the satellite angular spread and adds it to the UE's own motion Doppler.

### How it works under the hood (The Physics):

When `doppler_mode='precompensated'`, the modified coefficients generator runs the following logic inside its `_step_11_doppler_matrix()` function:

1. **Calculate Full Satellite Doppler (Per Cluster)**: It calculates the raw satellite Doppler phase component $\Phi_{\text{sat}, l}(t)$ along each cluster path's departure vector $\hat{r}_{\text{dep},l}$:
   $$\Phi_{\text{sat}, l}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep},l} \right) t$$
2. **Calculate Beam-Center/LoS precompensation**: It calculates the reference Doppler phase shift $\Phi_{\text{precomp}}(t)$ along the direct Line-of-Sight vector $\hat{r}_{\text{LoS}}$ pointing directly to the UE (which is at the center of the beam):
   $$\Phi_{\text{precomp}}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{LoS}} \right) t$$
3. **Subtract the Compensation (The Residual)**: It subtracts the LoS precompensation from the per-cluster Doppler, and adds the UE motion Doppler phase component $\Phi_{\text{UE}, l, m}(t)$ to compute the composite ray phase exponent $\theta_{l,m}(t)$:
   $$\theta_{l,m}(t) = \Phi_{\text{UE}, l, m}(t) + \Big( \Phi_{\text{sat}, l}(t) - \Phi_{\text{precomp}}(t) \Big)$$
   where:
   $$\Phi_{\text{UE}, l, m}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr},l,m} \right) t$$

#### 4. Defining the "Beam Center"
In this simulation setup, **the beam center is defined as the exact position of the UE**. Because the simulation focuses on a single link, the satellite spot beam is assumed to track and center directly on the UE. 
* **The Direction Vector**: The code uses the direct Line-of-Sight (LoS) path angles (`los_zod` and `los_aod` from the `topology` structure) as the direction vector pointing to the beam center ($\hat{r}_{\text{LoS}}$).
* **The Physics of the Residual**: Because scattering clusters (e.g. reflections off nearby buildings) depart the satellite at slightly different angles ($\hat{r}_{\text{dep},l}$) than the direct path ($\hat{r}_{\text{LoS}}$), they experience a slightly different satellite Doppler shift. Subtracting the beam-center precompensation leaves a small per-cluster **residual Doppler** ($\Phi_{\text{sat}, l} - \Phi_{\text{precomp}}$).

#### 5. Customizing the Beam Center (Off-Center Precompensation)
If you want to simulate a scenario where the satellite's spot beam is pointed at a reference point offset from the UE (e.g., to simulate a UE at the edge of the beam), you can set a custom beam center in the code:

* **What to add in the Python script**:
  Before running the channel model, calculate your custom beam center location in ENU coordinates and pass it to the channel model using `set_beam_center()`:
  ```python
  # Define the beam center offset relative to the UE (e.g., 100 meters East)
  beam_center_offset = np.array([100.0, 0.0, 0.0])
  beam_center_ENU = ut_loc_ENU + beam_center_offset

  # Set the custom beam center
  channel_model.set_beam_center(beam_center_ENU)
  ```
* **How it works under the hood**:
  - The channel coefficients and physical path delays/powers are still calculated strictly between the **actual UE and Satellite locations**.
  - However, the precompensation phase $\Phi_{\text{precomp}}(t)$ is now calculated along the unit vector pointing from the satellite to this custom beam center coordinate ($\hat{r}_{\text{sat}\to\text{BC}}$):
    $$\Phi_{\text{precomp}}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{sat}\to\text{BC}} \right) t$$
  - As a result, because $\hat{r}_{\text{LoS}} \neq \hat{r}_{\text{sat}\to\text{BC}}$, the direct Line-of-Sight path between the satellite and the UE will **no longer cancel to $0\text{ Hz}$**; it will experience a realistic residual Doppler shift based on its distance from the beam center.

---
**Explain in Details**
## 1. Where in the Code this is Implemented
This logic is implemented in the overridden `_step_11_doppler_matrix()` function of the **`ChannelCoefficientsGenerator_modify`** class:
* **File**: [OpenNTN/channel_coefficients.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py)
* **Lines**: **Lines 1177–1182**

### The exact code block:
```python
elif doppler_mode == 'precompensated':
    los_zod_exp = tf.expand_dims(tf.expand_dims(topology.los_zod, -1), -1)
    los_aod_exp = tf.expand_dims(tf.expand_dims(topology.los_aod, -1), -1)
    r_hat_los = self._unit_sphere_vector(los_zod_exp, los_aod_exp)
    exp_sat_los = 2*PI/lambda_0 * tf.reduce_sum(r_hat_los * v_sat_bar, -2) * t
    exponent = exponent + (exp_sat_cl - exp_sat_los)
```

---

## 2. Detailed Step-by-Step Code Explanation

### Step A: Expanding LoS Angles for Dimensions
```python
los_zod_exp = tf.expand_dims(tf.expand_dims(topology.los_zod, -1), -1)
los_aod_exp = tf.expand_dims(tf.expand_dims(topology.los_aod, -1), -1)
```
The direct Line-of-Sight zenith-of-departure (`los_zod`) and azimuth-of-departure (`los_aod`) have shapes of `[batch_size, num_rx, num_tx]`. 
To perform computations alongside per-ray arrays, these are expanded twice along the trailing dimensions using `tf.expand_dims` to match the target shape: `[batch_size, num_rx, num_tx, num_clusters, num_rays]`.

### Step B: Building the 3D LoS Unit Vector
```python
r_hat_los = self._unit_sphere_vector(los_zod_exp, los_aod_exp)
```
This converts the spherical angles (elevation/azimuth of the direct link) into a 3D Cartesian unit vector $\hat{r}_{\text{LoS}}$ pointing from the satellite straight to the UE:
$$\hat{r}_{\text{LoS}} = \begin{bmatrix} \cos(\theta_{\text{aod,LoS}}) \sin(\theta_{\text{zod,LoS}}) \\ \sin(\theta_{\text{aod,LoS}}) \sin(\theta_{\text{zod,LoS}}) \\ \cos(\theta_{\text{zod,LoS}}) \end{bmatrix}$$

### Step C: Computing the Direct LoS Doppler Projection
```python
exp_sat_los = 2*PI/lambda_0 * tf.reduce_sum(r_hat_los * v_sat_bar, -2) * t
```
This performs the element-wise multiplication of the 3D unit vector $\hat{r}_{\text{LoS}}$ and the satellite's 3D velocity vector `v_sat_bar` ($\vec{v}_{\text{sat}}$), and sums over the spatial dimension index (`-2`) to compute the 3D dot product:
$$\Phi_{\text{precomp}}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{LoS}} \right) t$$
This is the bulk Doppler shift that the transmitter precompensates.

### Step D: Subtracting the Compensation (Residual Calculation)
```python
exponent = exponent + (exp_sat_cl - exp_sat_los)
```
The phase contribution of the satellite's motion on cluster path $l$ is:
$$\Phi_{\text{sat}, l}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep},l} \right) t$$
The code subtracts the bulk precompensation $\Phi_{\text{precomp}}(t)$ from the per-cluster phase $\Phi_{\text{sat}, l}(t)$ to calculate the satellite residual Doppler phase. It then adds this to the UE's own Doppler phase exponent:
$$\theta_{l,m}(t) = \Phi_{\text{UE}, l, m}(t) + \Big( \Phi_{\text{sat}, l}(t) - \Phi_{\text{precomp}}(t) \Big)$$
where $\Phi_{\text{UE}, l, m}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr},l,m} \right) t$.

The resulting composite exponent is converted to a complex phase rotation matrix in Line 1196:
$$h_{\text{doppler}} = \exp\left( j \cdot \theta_{l,m}(t) \right)$$
This phase matrix is applied to the final time-domain channel impulse response coefficients.