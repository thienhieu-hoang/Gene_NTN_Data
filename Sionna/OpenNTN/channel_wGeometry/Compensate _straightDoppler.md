Viewed gen_channel_v3_wGeometry_straightDoppler_Compensate.py:1-26
Listed directory channel_wGeometry
Viewed Doppler_Limitations_and_Modifications.md:1-401
Viewed gen_channel_v3_wGeometry_straightDoppler_Compensate.py:200-220
Viewed gen_channel_v3_wGeometry_straightDoppler_Compensate.py:25-35
Viewed gen_channel_v3_wGeometry_straightDoppler_Compensate.py:220-235
Edited gen_channel_v3_wGeometry_straightDoppler_Compensate.py

You **do not need to write any new `_modify` functions or classes** for this! 

The modified classes we implemented in the library (`Topology_modify` and `ChannelCoefficientsGenerator_modify`) were proactively designed to support three distinct Doppler modes:
1. `full`: Straight/uncompensated sat-to-UE Doppler.
2. `precompensated`: Satellite-to-beam-center (LoS) Doppler precompensated, leaving only the residual satellite Doppler and the UE's Doppler.
3. `ue_only`: Bypasses the satellite Doppler entirely.

---

### How to use it in your new script:
In [gen_channel_v3_wGeometry_straightDoppler_Compensate.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/channel_wGeometry/gen_channel_v3_wGeometry_straightDoppler_Compensate.py), I have already changed the configuration to:
```python
channel_model = channel_class(carrier_frequency=carrier_frequency,
                              ut_array=ut_array,
                              bs_array=bs_array,
                              direction=direction,
                              elevation_angle=elevation_angle,
                              doppler_enabled=True,
                              doppler_mode='precompensated') # <--- Set to precompensated!
```

---

### How it works under the hood (The Physics):

When `doppler_mode='precompensated'`, the modified coefficients generator runs the following logic inside its `_step_11_doppler_matrix()` function:

1. **Calculate Full Satellite Doppler (Per Cluster)**:
   It calculates the raw satellite Doppler projection onto each cluster path's departure vector $\hat{r}_{\text{dep},l}$:
   $$\text{exp\_sat\_cl}_l = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep},l} \right) \cdot t$$
2. **Calculate Beam-Center/LoS precompensation**:
   It calculates the reference Doppler shift along the direct Line-of-Sight vector $\hat{r}_{\text{LoS}}$ pointing directly to the UE (which is at the center of the beam):
   $$\text{exp\_sat\_los} = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{LoS}} \right) \cdot t$$
3. **Subtract the Compensation (The Residual)**:
   It subtracts the LoS precompensation from the per-cluster Doppler, and adds the UE Doppler:
   $$\text{Total Phase}(t) = \text{Phase}_{\text{UE}}(t) + \Big( \text{exp\_sat\_cl}_l - \text{exp\_sat\_los} \Big)$$

This correctly simulates the residual Doppler shift (due to the angular spread of the clusters leaving the satellite) combined with the UE's movement Doppler shift.