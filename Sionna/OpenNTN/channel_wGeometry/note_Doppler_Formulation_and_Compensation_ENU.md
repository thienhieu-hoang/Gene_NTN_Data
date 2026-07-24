# Mathematical Formulation of NTN Doppler Shift and Precompensation in ENU Coordinates

> **Document Summary**: This document provides a mathematical formulation of 3D Doppler shifts for Non-Terrestrial Networks (NTN) in the local **East-North-Up (ENU)** coordinate frame. It covers:
> 1. Coordinate initialization in ECEF and transformation into ENU coordinates.
> 2. Straight Satellite-to-UE Doppler formulation and per-ray phase rotation.
> 3. Comparison with legacy models (old OpenNTN & MATLAB 5G Toolbox 2-segment split) and ray-level application.
> 4. Doppler precompensation using a Beam Center (BC) link ($\text{BC} \neq \text{UE}$).
> 5. Special case when the Beam Center is located exactly at the UE ($\text{BC} = \text{UE}$).
> 6. Step-by-step per-ray application in time-domain channel impulse responses.

---

## 1. Local Coordinate System (ECEF to ENU Frame Transformation)

In satellite communication systems, orbital dynamics and global physical trajectories are natively modeled in the **Earth-Centered Earth-Fixed (ECEF)** Cartesian coordinate frame $(X, Y, Z)$ based on the WGS-84 ellipsoid. 

To model 3D wireless propagation (angles of arrival, departure, and local mobility), all physical 3D positions and velocity vectors are transformed into the local tangent plane **East-North-Up (ENU)** Cartesian coordinate system $(x, y, z)$, centered at the initial UE ground position $(\phi_{\text{UE}}, \lambda_{\text{UE}}, h_{\text{UE}})$:

* **ECEF-to-ENU Transformation**: Using the rotation matrix $\mathbf{R}_{\text{ECEF}\to\text{ENU}}$ defined in [Trajectory_Explanation.md](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/channel_wGeometry/Trajectory_Explanation.md#3-global-ecef-to-local-tangent-frame-enu-transformation):
  $$\vec{p}_{\text{ENU}} = \mathbf{R}_{\text{ECEF}\to\text{ENU}} \cdot \left( \vec{r}_{\text{ECEF}} - \vec{r}_{\text{UE, ECEF}}(0) \right), \quad \vec{v}_{\text{ENU}} = \mathbf{R}_{\text{ECEF}\to\text{ENU}} \cdot \vec{v}_{\text{ECEF}}$$

In the local ENU frame $(x, y, z)$:
* **Satellite Position**: $\vec{p}_{\text{sat}} = [x_{\text{sat}}, y_{\text{sat}}, z_{\text{sat}}]^T \in \mathbb{R}^3$ (altitude $h_{\text{sat}} = z_{\text{sat}} \approx 600\text{ km}$ for LEO).
* **Satellite Velocity Vector**: $\vec{v}_{\text{sat}} = [v_{\text{sat}, E}, v_{\text{sat}, N}, v_{\text{sat}, U}]^T \in \mathbb{R}^3$ (speed $\|\vec{v}_{\text{sat}}\| \approx 7.56\text{ km/s}$).
* **User Equipment (UE) Position**: $\vec{p}_{\text{UE}} = [x_{\text{UE}}, y_{\text{UE}}, z_{\text{UE}}]^T \in \mathbb{R}^3$ (typically $z_{\text{UE}} \approx 1.5\text{ m}$).
* **UE Velocity Vector**: $\vec{v}_{\text{UE}} = [v_{\text{UE}, E}, v_{\text{UE}, N}, v_{\text{UE}, U}]^T \in \mathbb{R}^3$ (speed $\|\vec{v}_{\text{UE}}\| \approx 0 - 50\text{ m/s}$).
* **Carrier Frequency & Wavelength**: $f_c$ [Hz] and $\lambda_0 = \frac{c}{f_c}$ [m], where $c \approx 2.99792458 \times 10^8\text{ m/s}$.

---

## 2. Satellite-to-UE Doppler Formulation

### 2.1 Spherical Angle Representation of Rays

Consider a channel drop consisting of $L$ multipath clusters, where each cluster $l \in \{1, \dots, L\}$ contains $M$ sub-rays $m \in \{1, \dots, M\}$.

#### A. Departure Unit Vector at Satellite (TX)
Each cluster $l$ departs the satellite at a specific **Zenith of Departure** $\theta_{\text{zod}, l}$ and **Azimuth of Departure** $\phi_{\text{aod}, l}$. The 3D unit vector pointing along the departure path of cluster $l$ is:

$$\hat{r}_{\text{dep}, l} = \begin{bmatrix} \cos(\phi_{\text{aod}, l}) \sin(\theta_{\text{zod}, l}) \\ \sin(\phi_{\text{aod}, l}) \sin(\theta_{\text{zod}, l}) \\ \cos(\theta_{\text{zod}, l}) \end{bmatrix} \in \mathbb{R}^3, \quad \|\hat{r}_{\text{dep}, l}\| = 1$$

#### B. Arrival Unit Vector at UE (RX)
Each sub-ray $m$ within cluster $l$ arrives at the UE with a **Zenith of Arrival** $\theta_{\text{zoa}, l, m}$ and **Azimuth of Arrival** $\phi_{\text{aoa}, l, m}$. The 3D unit vector pointing along the arrival direction of sub-ray $(l,m)$ is:

$$\hat{r}_{\text{arr}, l, m} = \begin{bmatrix} \cos(\phi_{\text{aoa}, l, m}) \sin(\theta_{\text{zoa}, l, m}) \\ \sin(\phi_{\text{aoa}, l, m}) \sin(\theta_{\text{zoa}, l, m}) \\ \cos(\theta_{\text{zoa}, l, m}) \end{bmatrix} \in \mathbb{R}^3, \quad \|\hat{r}_{\text{arr}, l, m}\| = 1$$

---

### 2.2 Mathematical Expression for Straight SAT-UE Doppler

The total Doppler frequency shift $f_{D, l, m}^{\text{straight}}$ experienced by sub-ray $(l, m)$ on the direct, uncompensated Satellite-to-UE link is the sum of two projections:

1. **Satellite Motion Projection**: Satellite 3D velocity vector $\vec{v}_{\text{sat}}$ projected onto the cluster departure direction $\hat{r}_{\text{dep}, l}$.
2. **UE Motion Projection**: UE 3D velocity vector $\vec{v}_{\text{UE}}$ projected onto the sub-ray arrival direction $\hat{r}_{\text{arr}, l, m}$.

$$f_{D, l, m}^{\text{straight}} = \underbrace{\frac{1}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l} \right)}_{\text{Satellite Doppler Component } f_{D, \text{sat}, l}} + \underbrace{\frac{1}{\lambda_0} \left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m} \right)}_{\text{UE Doppler Component } f_{D, \text{UE}, l, m}}$$

### 2.3 Per-Ray Phase Rotation over Time

Over a time duration $t$, the accumulated Doppler phase shift $\Phi_{l, m}^{\text{straight}}(t)$ for sub-ray $(l, m)$ is:

$$\Phi_{l, m}^{\text{straight}}(t) = 2\pi f_{D, l, m}^{\text{straight}} t = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l} + \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m} \right) t$$

To avoid confusion with the channel impulse response / transfer function $h_{l,m}(t)$, we denote the complex **Doppler phase rotation operator** for sub-ray $(l,m)$ as $d_{\text{phase}, l, m}(t)$:

$$d_{\text{phase}, l, m}(t) = \exp\left( j \cdot \Phi_{l, m}^{\text{straight}}(t) \right) = \exp\left( j \frac{2\pi}{\lambda_0} \left[ (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l}) + (\vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m}) \right] t \right)$$

---

## 3. Comparison with Original (Previous Code Versions) Doppler Models

### 3.1 Original OpenNTN Code (Scalar & Randomized Approximation)
In the original OpenNTN implementation, the satellite Doppler shift was **not** calculated using 3D departure vectors $\hat{r}_{\text{dep}, l}$ or the satellite's true 3D velocity vector $\vec{v}_{\text{sat}}$. Instead:

1. **Scalar Orbital Speed**: It computed a scalar orbital speed $v_{\text{sat}} = \sqrt{\frac{GM}{R_E + h_{\text{sat}}}}$.
2. **LoS Projection with Random Angle**:
   $$f_{D, \text{sat}}^{\text{old}} = \frac{v_{\text{sat}}}{\lambda_0} \cos(\theta_{\text{elevation}}) \cos(\theta_{\text{random}})$$
   where $\theta_{\text{random}} \sim \mathcal{U}(0, 2\pi)$ was a random orbit direction angle drawn per batch.
3. **Flat Scalar Broadcast**: This single scalar frequency $f_{D, \text{sat}}^{\text{old}}$ (e.g., $+250\text{ kHz}$) was broadcasted identically to **all** clusters and **all** sub-rays:
   $$\text{Phase}_{\text{old}}(t) = \frac{2\pi}{\lambda_0} (\vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m}) t + 2\pi f_{D, \text{sat}}^{\text{old}} t$$

* **Limitation**: Multipath departure angle diversity was ignored. Rays departing at $+5^\circ$ or $-5^\circ$ relative to the Line-of-Sight experienced the exact same satellite Doppler shift.

---

### 3.2 MATLAB 5G Toolbox / 3GPP TR 38.811 Model (2-Segment Split)

The MATLAB 5G Toolbox (and standard 3GPP TR 38.811 formulations) often simplifies the NTN channel into two decoupled segments:

1. **Segment 1 (Sat-to-Beam-Center Link $\text{Sat}\to\text{BC}$)**:
   The satellite Doppler is evaluated along a single reference vector $\hat{r}_{\text{Sat}\to\text{BC}}$ pointing to the center of the beam:
   $$f_{D, \text{Sat-BC}} = \frac{1}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{Sat}\to\text{BC}} \right)$$
2. **Segment 2 (Beam-Center-to-UE Link $\text{BC}\to\text{UE}$)**:
   The local ground scattering and UE mobility produce local Doppler shifts relative to local arrival vectors $\hat{r}_{\text{arr}, l, m}$:
   $$f_{D, \text{BC-UE}, l, m} = \frac{1}{\lambda_0} \left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m} \right)$$
3. **Total segment sum**:
   $$f_{D, \text{total}}^{\text{2-segment}} = f_{D, \text{Sat-BC}} + f_{D, \text{BC-UE}, l, m}$$

* **Comparison**: The 2-segment model assumes the satellite Doppler is uniform across all multipaths within a beam. Our **modified 3D ENU model** goes beyond this by taking the dot product $\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l}$ per cluster $l$, capturing the true physical angular spread from the satellite.

---

### 3.3 How Doppler is Applied to Each Ray (Ray-Level Mapping)

> **Relationship to Sections 3.1 & 3.2**: 
> Sections 3.1 and 3.2 present high-level models for calculating satellite Doppler shifts. Section 3.3 explains **how any Doppler model maps onto individual sub-rays** inside the 3GPP TR 38.901 stochastic channel matrix generator.

For a cluster $l$ containing $M$ sub-rays ($m = 1, \dots, M$):

* **Cluster-Level Departure**: All $M$ sub-rays in cluster $l$ share the cluster-center departure direction $\hat{r}_{\text{dep}, l}$. Therefore, all sub-rays in cluster $l$ share the **same satellite Doppler component**:
  $$f_{D, \text{sat}, l} = \frac{1}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l} \right)$$
* **Sub-Ray-Level Arrival**: Each sub-ray $m$ within cluster $l$ has an individual arrival direction vector $\hat{r}_{\text{arr}, l, m}$. Therefore, each sub-ray $m$ has a **unique UE Doppler component**:
  $$f_{D, \text{UE}, l, m} = \frac{1}{\lambda_0} \left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m} \right)$$

The time-domain channel impulse response coefficient $\gamma_{l, m}(t)$ for sub-ray $(l,m)$ is constructed as:

$$\gamma_{l, m}(t) = a_{l, m} \cdot \exp\left(j \psi_{l, m}\right) \cdot d_{\text{phase}, l, m}(t)$$

$$\gamma_{l, m}(t) = a_{l, m} \cdot \exp\left(j \psi_{l, m}\right) \cdot \exp\left(j \frac{2\pi}{\lambda_0} \left[ (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l}) + (\vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m}) \right] t \right)$$

where $a_{l, m}$ is the sub-ray amplitude and $\psi_{l, m}$ is the random initial phase.

---

## 4. Doppler Compensation (Precompensation with Sat-BC Link)

> 📌 **Summary Note on Precompensation & Residual Doppler**:
> In practical NTN systems, the satellite bulk Doppler shift (up to $\pm 400\text{ kHz}$ at Ka-band) exceeds the subcarrier spacing. The transmitter/gateway precompensates by shifting the carrier frequency by $-\Phi_{\text{precomp}}(t) = -\frac{2\pi}{\lambda_0}(\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{sat}\to\text{BC}})t$ along the reference link to the **Beam Center (BC)**. 
> Because multipath scatter rays depart at angles $\hat{r}_{\text{dep}, l} \neq \hat{r}_{\text{sat}\to\text{BC}}$, subtracting this single bulk shift leaves a **residual satellite Doppler** on each cluster alongside the UE's own ground mobility Doppler.

```
       [ LEO Satellite ] (v_sat)
             /     \
            /       \  r_dep,l (Cluster path l)
           /         \
          v           v
  (Beam Center BC)   [ UE ] (v_UE)
  (p_BC != p_UE)
```

---

### 4.1 General Case: Beam Center Offset from UE ($\text{BC} \neq \text{UE}$)

Let $\vec{p}_{\text{BC}} = [x_{\text{BC}}, y_{\text{BC}}, z_{\text{BC}}]^T$ be the ENU coordinates of the designated **Beam Center**.

#### A. Satellite-to-Beam-Center Unit Vector
The 3D unit vector pointing from the satellite to the Beam Center is:

$$\hat{r}_{\text{sat}\to\text{BC}} = \frac{\vec{p}_{\text{BC}} - \vec{p}_{\text{sat}}}{\|\vec{p}_{\text{BC}} - \vec{p}_{\text{sat}}\|} \in \mathbb{R}^3$$

#### B. Precompensation Frequency & Phase
The satellite precompensates the Doppler shift evaluated along $\hat{r}_{\text{sat}\to\text{BC}}$:

$$f_{D, \text{precomp}} = \frac{1}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{sat}\to\text{BC}} \right)$$

$$\Phi_{\text{precomp}}(t) = 2\pi f_{D, \text{precomp}} t = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{sat}\to\text{BC}} \right) t$$

#### C. Net Effective (Compensated) Doppler Phase
The net Doppler phase $\Phi_{l, m}^{\text{compensated}}(t)$ remaining on sub-ray $(l, m)$ after precompensation is:

$$\Phi_{l, m}^{\text{compensated}}(t) = \Phi_{l, m}^{\text{straight}}(t) - \Phi_{\text{precomp}}(t)$$

Substituting the expressions for $\Phi_{l, m}^{\text{straight}}(t)$ and $\Phi_{\text{precomp}}(t)$:

$$\Phi_{l, m}^{\text{compensated}}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m} \right) t + \frac{2\pi}{\lambda_0} \left[ \vec{v}_{\text{sat}} \cdot \left( \hat{r}_{\text{dep}, l} - \hat{r}_{\text{sat}\to\text{BC}} \right) \right] t$$

#### D. Physical Analysis of the Net Residual Doppler

The total Doppler remaining on sub-ray $(l, m)$ after precompensation consists of **two distinct physical terms**:

$$\text{Total Post-Precompensation Residual}_{l, m} = \underbrace{\frac{1}{\lambda_0} \left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m} \right)}_{\text{Term 1: UE Motion Doppler}} + \underbrace{\frac{1}{\lambda_0} \left[ \vec{v}_{\text{sat}} \cdot \left( \hat{r}_{\text{dep}, l} - \hat{r}_{\text{sat}\to\text{BC}} \right) \right]}_{\text{Term 2: Satellite Residual Doppler}}$$

1. **Term 1 (UE Motion Doppler)**: Driven by the UE ground velocity $\vec{v}_{\text{UE}}$. Precompensation at the satellite/gateway only counteracts the satellite's movement ($\vec{v}_{\text{sat}}$) and does **not** compensate for $\vec{v}_{\text{UE}}$. Therefore, the full UE motion Doppler $\frac{1}{\lambda_0}(\vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m})$ remains in the channel.
2. **Term 2 (Satellite Residual Doppler)**: Driven by two spatial/angular offsets:
   - **Location Offset ($\vec{p}_{\text{UE}} \neq \vec{p}_{\text{BC}}$)**: When the UE is offset from the Beam Center (e.g., positioned at the edge of the spot beam), the direct Line-of-Sight vector $\hat{r}_{\text{LoS}}$ differs from $\hat{r}_{\text{sat}\to\text{BC}}$. This causes a satellite Doppler mismatch even on the direct path ($\hat{r}_{\text{LoS}} \neq \hat{r}_{\text{sat}\to\text{BC}}$).
   - **Cluster Departure Angular Spread ($\hat{r}_{\text{dep}, l} \neq \hat{r}_{\text{sat}\to\text{BC}}$)**: Multipath scatter paths depart the satellite at angles offset from $\hat{r}_{\text{sat}\to\text{BC}}$, generating per-cluster satellite Doppler variations ($\approx 10 - 50\text{ Hz}$).

---

### 4.2 Special Case: Beam Center at UE Position ($\text{BC} = \text{UE}$)

When the satellite spot beam is steered to track the UE directly ($\vec{p}_{\text{BC}} = \vec{p}_{\text{UE}}$):

$$\hat{r}_{\text{sat}\to\text{BC}} = \hat{r}_{\text{LoS}} = \begin{bmatrix} \cos(\phi_{\text{aod, LoS}}) \sin(\theta_{\text{zod, LoS}}) \\ \sin(\phi_{\text{aod, LoS}}) \sin(\theta_{\text{zod, LoS}}) \\ \cos(\theta_{\text{zod, LoS}}) \end{bmatrix}$$

The precompensation phase simplifies to:

$$\Phi_{\text{precomp}}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{sat}} \cdot \hat{r}_{\text{LoS}} \right) t$$

#### A. Direct Line-of-Sight Path ($l = 1, \text{LoS}$)
For the direct LoS path, $\hat{r}_{\text{dep}, 1} = \hat{r}_{\text{LoS}}$. The residual satellite Doppler cancels to **zero**:

$$\Delta f_{D, \text{sat}, \text{LoS}}^{\text{residual}} = \frac{1}{\lambda_0} \vec{v}_{\text{sat}} \cdot \left( \hat{r}_{\text{LoS}} - \hat{r}_{\text{LoS}} \right) = 0\text{ Hz}$$

The net Doppler phase on the direct LoS path is purely due to UE motion:

$$\Phi_{\text{LoS}}^{\text{compensated}}(t) = \frac{2\pi}{\lambda_0} \left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, \text{LoS}} \right) t$$

#### B. Non-Line-of-Sight Scatter Clusters ($l > 1$)
For scattered multipath clusters ($l > 1$), $\hat{r}_{\text{dep}, l} \neq \hat{r}_{\text{LoS}}$ due to angular spread. A small **cluster angular-spread residual** remains:

$$\Delta f_{D, \text{sat}, l}^{\text{residual}} = \frac{1}{\lambda_0} \vec{v}_{\text{sat}} \cdot \left( \hat{r}_{\text{dep}, l} - \hat{r}_{\text{LoS}} \right) \approx 10 - 50\text{ Hz}$$

---

### 4.3 Per-Ray Application of Compensated Doppler

The composite complex fading channel matrix $H_{u, s}(f, t)$ over time $t$ and subcarrier frequency $f$ for receiver antenna $u$ and transmitter antenna $s$ is obtained by summing over all clusters $l$ and sub-rays $m$:

$$H_{u, s}(f, t) = \sum_{l=1}^{L} \sum_{m=1}^{M} F_{u, l, m} \cdot F_{s, l, m} \cdot \exp\left( -j 2\pi f \tau_{l, m} \right) \cdot \exp\left( j \Phi_{l, m}^{\text{compensated}}(t) \right)$$

where:
* $F_{u, l, m}$ and $F_{s, l, m}$ are receive and transmit antenna field patterns.
* $\tau_{l, m}$ is the propagation delay of sub-ray $(l,m)$.
* $\Phi_{l, m}^{\text{compensated}}(t)$ is the per-ray Doppler phase exponent:

$$\Phi_{l, m}^{\text{compensated}}(t) = \frac{2\pi}{\lambda_0} \left[ \underbrace{\left( \vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m} \right)}_{\text{UE Motion Phase}} + \underbrace{\left( \vec{v}_{\text{sat}} \cdot (\hat{r}_{\text{dep}, l} - \hat{r}_{\text{sat}\to\text{BC}}) \right)}_{\text{Satellite Residual Phase}} \right] t$$

---

## 5. Summary Comparison Matrix of Doppler Modes

| Feature / Mode | Straight (Full) Doppler (`full`) | Precompensated ($\text{BC} \neq \text{UE}$) | Precompensated ($\text{BC} = \text{UE}$) (`precompensated`) | Legacy OpenNTN (`ue_only`) |
| :--- | :--- | :--- | :--- | :--- |
| **Satellite Velocity Used** | Real 3D vector $\vec{v}_{\text{sat}}$ | Real 3D vector $\vec{v}_{\text{sat}}$ | Real 3D vector $\vec{v}_{\text{sat}}$ | None (or randomized scalar) |
| **Sat Doppler per Cluster** | $\frac{1}{\lambda_0} (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l})$ | $\frac{1}{\lambda_0} (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l})$ | $\frac{1}{\lambda_0} (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l})$ | Flat scalar $f_{D, \text{sat}}^{\text{old}}$ (or $0$) |
| **Precompensation Frequency** | $0\text{ Hz}$ | $\frac{1}{\lambda_0} (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{sat}\to\text{BC}})$ | $\frac{1}{\lambda_0} (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{LoS}})$ | N/A |
| **LoS Path Satellite Residual** | $\frac{1}{\lambda_0} (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{LoS}}) \approx 200-400\text{ kHz}$ | $\frac{1}{\lambda_0} \vec{v}_{\text{sat}} \cdot (\hat{r}_{\text{LoS}} - \hat{r}_{\text{sat}\to\text{BC}})$ | **$0\text{ Hz}$** (Fully cancelled) | $0\text{ Hz}$ |
| **NLoS Cluster Satellite Residual**| $\frac{1}{\lambda_0} (\vec{v}_{\text{sat}} \cdot \hat{r}_{\text{dep}, l})$ | $\frac{1}{\lambda_0} \vec{v}_{\text{sat}} \cdot (\hat{r}_{\text{dep}, l} - \hat{r}_{\text{sat}\to\text{BC}})$ | $\frac{1}{\lambda_0} \vec{v}_{\text{sat}} \cdot (\hat{r}_{\text{dep}, l} - \hat{r}_{\text{LoS}}) \approx 10-50\text{ Hz}$ | $0\text{ Hz}$ |
| **UE Motion Doppler per Ray** | $\frac{1}{\lambda_0} (\vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m})$ | $\frac{1}{\lambda_0} (\vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m})$ | $\frac{1}{\lambda_0} (\vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m})$ | $\frac{1}{\lambda_0} (\vec{v}_{\text{UE}} \cdot \hat{r}_{\text{arr}, l, m})$ |

---

*Document updated based on inline review comments — Sionna/OpenNTN integration.*
