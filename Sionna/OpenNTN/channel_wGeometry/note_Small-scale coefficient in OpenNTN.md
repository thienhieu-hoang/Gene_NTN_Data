
The channel models in OpenNTN (`Urban`, `DenseUrban`, and `SubUrban`) use a **Geometry-Based Stochastic Channel Model (GSCM)** to generate small-scale fading. 


### How the Scenarios Differ (Urban vs. DenseUrban vs. SubUrban)

The underlying mathematical steps to generate the channel are **identical** for all three classes. They differ entirely in the **statistical configuration tables (JSON files)** they load, which define:

* **Number of Clusters (Paths):**
  * `SubUrban`: Fewer paths (**3 for LoS, 4 for NLoS**).
  * `Urban` & `DenseUrban`: More paths due to denser clutter (up to **12+ paths**).
* **RMS Delay Spread (DS):**
  * `SubUrban` has a short delay spread (smaller delay differences between paths).
  * `DenseUrban` has a larger delay spread (reflections travel further, arriving with longer delays).
* **Angular Spreads (ASA, ASD, ZSA, ZSD):**
  * Urban environments have a wider angular spread of arrivals/departures because waves scatter off tall buildings in all directions.
  * Suburban environments have narrow angular spreads because the houses are shorter and sparser.

---

## Small-Scale

**Yes, there is small-scale randomness in the channel, and it is indeed modeled as $h = \sqrt{\beta} \cdot g$.** 

Even if you keep the positions, distances, and elevation angles completely fixed, the channel coefficients will still vary randomly between different runs. 

Here is the exact verification from the OpenNTN/Sionna source code:

---

### 1. Where the Randomness Comes From (`phi` in Step 10)

Inside [channel_coefficients.py:L491–L509](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py#L491-L509), the channel generator runs **Step 10** of the 3GPP standard:

```python
def _step_10(self, shape):
    # Generate random and uniformly distributed phases for all rays
    phi = tf.random.uniform(tf.concat([shape, [4]], axis=0), minval=-PI,
        maxval=PI, dtype=self.rdtype)
    return phi
```

* **Uniform Random Phase:** `phi` ($\phi$) represents the microscopic phase offset of each sub-ray. Because the scatterers in the real world have random microscopic surfaces, their initial phases are modeled as independent uniform random variables:
  $$\phi \sim U(-\pi, \pi)$$

---

### 2. How the Fading Coefficient $g$ is Built

Each of the paths (clusters) is composed of 20 scattered sub-rays. Inside the code, these sub-rays are summed up together:
$$g_l(t) = \sum_{m=1}^{20} e^{j \phi_m} \cdot \text{array\_offset}_m \cdot e^{j 2\pi f_{d, m} t}$$

* By the **Central Limit Theorem**, summing these 20 random-phase sub-rays creates a complex Gaussian random variable:
  $$g_l \sim \mathcal{CN}(0, \sigma^2)$$
* The magnitude of this fading coefficient $|g_l|$ follows a **Rayleigh distribution** (or a **Rician distribution** if it is a LoS path).

---

### 3. Step 12: Scaling by Path Loss and Shadow Fading ($\sqrt{\beta}$)

After the small-scale fading coefficients are generated, the base class applies Step 12 ([channel_coefficients.py:L403–L430](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py#L403-L430)):

```python
gain = tf.math.pow(10.0, -pl_db / 20.0) * tf.sqrt(sf)
h *= tf.complex(gain, 0.0)
```

Here:
* `pl_db` is the path loss (deterministic distance-based).
* `sf` is the shadow fading (log-normal random variable).
* The combined term `gain` corresponds to the large-scale channel gain $\sqrt{\beta}$.

---

### Summary of What is Deterministic vs. Random

| Channel Component | Type | Source in Code |
| :--- | :--- | :--- |
| **Path Loss ($\beta_{\text{PL}}$)** | **Deterministic** | Calculated from 3D distance and elevation angle. |
| **Shadow Fading ($\beta_{\text{SF}}$)** | **Random** (Slow) | Log-Normal random variable generated per drop. |
| **Small-Scale Phase ($\phi$)** | **Random** (Fast) | Uniformly generated in `_step_10()` for each ray. |
| **Path Coefficient ($h$)** | **Random** | Combined product: $h = \sqrt{\beta_{\text{PL}} \cdot \beta_{\text{SF}}} \cdot g$ |

---

## Rays and Paths (taps)
### 1. Do the rays within a path only differ in phase? Are the large-scale elements the same?

**Almost, but they also have slight angular offsets.**

* **What they share (Same):** All 20 sub-rays of a path (cluster) share the **same large-scale parameters** (average delay, path loss, and shadow fading).
* **What they differ in (Different):**
  1. **Random initial phases ($\phi_m$):** Each sub-ray has a different random phase.
  2. **Microscopic angles:** Each sub-ray has a tiny angular offset relative to the path's central direction (determined by standard 3GPP tables, e.g., $\pm 0.04^\circ$ up to $\pm 2.15^\circ$). 
* **Rician vs. Rayleigh Fading:**
  * **NLoS Taps (and delayed LoS taps):** Since there is no dominant ray, summing these 20 random-phase, equal-power sub-rays yields a complex Gaussian variable. This results in **Rayleigh fading**.
  * **The First Tap (Tap 0) in LoS:** This tap contains the deterministic direct ray (constant amplitude and phase) plus the 20 random scattered rays. This results in **Rician fading**.

---

### 2. Is limiting the scenario to 4 paths equivalent to limiting it to 4 taps?

**Yes, exactly.**

Limiting the scenario to 4 paths means the discrete channel impulse response $h_t[n, l]$ will have exactly **4 delay taps**. Each of these 4 delay taps is generated by summing up its respective 20 sub-rays in the spatial domain before discretization.

---

### 3. Is the number of rays per path fixed or varied?

**It is fixed at 20 rays per path.**

* **No variation between taps:** Tap 1, Tap 2, Tap 3, and Tap 4 all sum up **exactly 20 sub-rays** each.
* **No variation between scenarios:** The number of rays per path is fixed at 20 for **DenseUrban, Urban, and SubUrban** alike (this is a standard constant defined by 3GPP TR 38.901).
* **What actually changes between the scenarios:**
  1. **The number of paths (taps):** (e.g., 3 for SubUrban LoS, but 12 for Urban/DenseUrban).
  2. **The statistical profiles:** The average delays, power decay profile, and the spread of departure/arrival angles of the paths.