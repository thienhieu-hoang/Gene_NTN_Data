# Geometry and Channel Modeling in Satellite Non-Terrestrial Networks (NTN)

This document provides a detailed mathematical and physical reference for the geometric relationships, coordinate systems, trajectories, Doppler dynamics, and channel fading models used in Non-Terrestrial Networks (NTN). 

---

## 1. Coordinate Systems and Transformations

To model the propagation channel between a Low Earth Orbit (LEO) satellite and a ground User Equipment (UE), we represent their positions in two primary 3D Cartesian coordinates: the Earth-Centered Inertial (ECI) frame and the Earth-Centered, Earth-Fixed (ECEF) frame.

### 1.1 Earth-Centered Inertial (ECI) Frame
The ECI frame is a non-rotating coordinate system where Kepler's laws of planetary motion hold. The origin is at the Earth's center of mass.
- **Position Vector**: $\mathbf{r}_{sat, ECI}(t) = \left[ x_{sat}(t), y_{sat}(t), z_{sat}(t) \right]^T$
- **Velocity Vector**: $\mathbf{v}_{sat, ECI}(t) = \left[ v_{x, sat}(t), v_{y, sat}(t), v_{z, sat}(t) \right]^T$

### 1.2 Earth-Centered, Earth-Fixed (ECEF) Frame
The ECEF frame rotates with the Earth at angular speed $\omega_E \approx 7.292115 \times 10^{-5} \text{ rad/s}$. The origin is at the Earth's center of mass.
- **Position Vector**: $\mathbf{r}_{UE, ECEF}(t) = \left[ x_{UE}(t), y_{UE}(t), z_{UE}(t) \right]^T$
- **Velocity Vector**: $\mathbf{v}_{UE, ECEF}(t) = \left[ v_{x, UE}(t), v_{y, UE}(t), v_{z, UE}(t) \right]^T$

### 1.3 UE Position Model (Geodetic to ECEF)
For a UE located at latitude $\phi$, longitude $\lambda$, and ellipsoidal height $h$, its 3D ECEF position vector is:
$$\mathbf{r}_{UE, ECEF} = \begin{bmatrix} (N(\phi) + h) \cos\phi \cos\lambda \\ (N(\phi) + h) \cos\phi \sin\lambda \\ \left(N(\phi)(1 - e^2) + h\right) \sin\phi \end{bmatrix}$$
Where:
- $N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$ is the prime vertical radius of curvature.
- $a = 6378137.0 \text{ m}$ (semi-major axis) and $e^2 = 6.69437999 \times 10^{-3}$ (first eccentricity squared) are defined by the WGS-84 ellipsoid.

### 1.4 ECI and ECEF Frame Rotations
The rotation from ECI to ECEF is a rotation about the Z-axis by the Greenwich Sidereal Time (GST) angle $\theta_G(t) = \omega_E t + \theta_G(0)$:
$$\mathbf{r}_{ECEF}(t) = \mathbf{R}_z(\theta_G(t)) \mathbf{r}_{ECI}(t)$$
$$\mathbf{r}_{ECI}(t) = \mathbf{R}_z(-\theta_G(t)) \mathbf{r}_{ECEF}(t)$$
Where the Z-axis rotation matrix $\mathbf{R}_z(\theta)$ is defined as:
$$\mathbf{R}_z(\theta) = \begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

For velocity conversion, the rotation must account for the Coriolis acceleration term:
$$\mathbf{v}_{ECI}(t) = \mathbf{R}_z(-\theta_G(t)) \mathbf{v}_{ECEF}(t) + \boldsymbol{\omega}_E \times \mathbf{r}_{ECI}(t)$$
$$\mathbf{v}_{ECEF}(t) = \mathbf{R}_z(\theta_G(t)) \left( \mathbf{v}_{ECI}(t) - \boldsymbol{\omega}_E \times \mathbf{r}_{ECI}(t) \right)$$
where $\boldsymbol{\omega}_E = [0, 0, \omega_E]^T$ is the Earth's rotation vector.

---

## 2. Satellite & UE Trajectory and Velocity Models

### 2.1 Satellite 3D Trajectory and Velocity (ECI Frame)
For a satellite in a circular orbit with altitude $h_s$ (orbit radius $r = R_E + h_s$), inclination $i$, Right Ascension of the Ascending Node (RAAN) $\Omega$, and argument of latitude $u(t) = \omega_s t + u_0$:

#### Position Vector in ECI
$$\mathbf{r}_{sat, ECI}(t) = \begin{bmatrix} x_{sat}(t) \\ y_{sat}(t) \\ z_{sat}(t) \end{bmatrix} = \begin{bmatrix} r \left( \cos u(t) \cos \Omega - \sin u(t) \sin \Omega \cos i \right) \\ r \left( \cos u(t) \sin \Omega + \sin u(t) \cos \Omega \cos i \right) \\ r \sin u(t) \sin i \end{bmatrix}$$

#### Velocity Vector in ECI
$$\mathbf{v}_{sat, ECI}(t) = \begin{bmatrix} v_{x, sat}(t) \\ v_{y, sat}(t) \\ v_{z, sat}(t) \end{bmatrix} = \begin{bmatrix} v_{sat} \left( -\sin u(t) \cos \Omega - \cos u(t) \sin \Omega \cos i \right) \\ v_{sat} \left( -\sin u(t) \sin \Omega + \cos u(t) \cos \Omega \cos i \right) \\ v_{sat} \cos u(t) \sin i \end{bmatrix}$$
Where:
- $v_{sat} = \sqrt{\frac{\mu}{R_E + h_s}}$ is the constant orbital speed.
- $\omega_s = \frac{v_{sat}}{r} = \sqrt{\frac{\mu}{(R_E + h_s)^3}}$ is the angular orbital frequency.
- $\mu \approx 3.986004 \times 10^{14} \text{ m}^3/\text{s}^2$ is Earth's gravitational parameter.

### 2.2 UE 3D Trajectory and Velocity (ECEF Frame)
For a UE moving on the ground at speed $v_{UE, ground}$ with a heading azimuth $\alpha$ (relative to North) at geodetic coordinate $(\phi, \lambda, h)$:

#### Velocity in Local East-North-Up (ENU) Frame
$$\mathbf{v}_{UE, ENU} = \begin{bmatrix} v_E \\ v_N \\ v_U \end{bmatrix} = \begin{bmatrix} v_{UE, ground} \sin\alpha \\ v_{UE, ground} \cos\alpha \\ 0 \end{bmatrix}$$

#### Velocity in ECEF Frame
The velocity vector is transformed from ENU to ECEF using the local transformation matrix:
$$\mathbf{v}_{UE, ECEF} = \mathbf{R}_{ENU\to ECEF} \mathbf{v}_{UE, ENU}$$
Where:
$$\mathbf{R}_{ENU\to ECEF} = \begin{bmatrix} -\sin\lambda & -\sin\phi \cos\lambda & \cos\phi \cos\lambda \\ \cos\lambda & -\sin\phi \sin\lambda & \cos\phi \sin\lambda \\ 0 & \cos\phi & \sin\phi \end{bmatrix}$$
Expanding the components gives the 3D ECEF velocity:
$$v_{x, UE} = -v_E \sin\lambda - v_N \sin\phi \cos\lambda$$
$$v_{y, UE} = v_E \cos\lambda - v_N \sin\phi \sin\lambda$$
$$v_{z, UE} = v_N \cos\phi$$

#### Position over time in ECEF
For short time intervals, the position is modeled as:
$$\mathbf{r}_{UE, ECEF}(t) = \mathbf{r}_{UE, ECEF}(0) + \mathbf{v}_{UE, ECEF} t$$

---

## 3. Time-Varying NTN Channel Formulation

The propagation delay between a LEO satellite and a ground UE is dynamic due to high-speed satellite motion.

### 3.1 Multipath Channel Impulse Response
The time-varying channel impulse response $h(t, \tau)$ is represented as:
$$h(t, \tau) = \sum_{l=0}^{L-1} a_l(t) e^{-j 2\pi f_c \tau_l(t)} \delta(\tau - \tau_l(t))$$
Where:
- $L$ is the number of resolvable propagation paths.
- $a_l(t)$ is the time-varying complex path gain.
- $f_c$ is the carrier frequency.
- $\tau_l(t)$ is the time-varying delay of the $l$-th path:
  $$\tau_l(t) = \frac{d_l(t)}{c}$$
  with $d_l(t)$ being the time-varying distance of the $l$-th path and $c$ being the speed of light.

### 3.2 Time-Varying Slant Range
For the direct line-of-sight (LoS) path, the distance $d_0(t)$ (known as the slant range $S_U(t)$) between the satellite and UE is calculated using their 3D ECI coordinates:
$$S_U(t) = \|\mathbf{r}_{sat, ECI}(t) - \mathbf{r}_{UE, ECI}(t)\| = \sqrt{(x_{sat}(t) - x_{UE, ECI}(t))^2 + (y_{sat}(t) - y_{UE, ECI}(t))^2 + (z_{sat}(t) - z_{UE, ECI}(t))^2}$$
Alternatively, in terms of Earth-centered geometry, it can be computed using the Law of Cosines:
$$S_U(t) = \sqrt{R_{UE}^2 + r_{sat}^2 - 2 R_{UE} r_{sat} \cos(\theta(t))}$$
Where:
- $R_{UE} = R_E + h_g$ is the distance of the UE from the Earth's center.
- $\theta(t)$ is the geocentric separation angle between the satellite and the UE.

---

## 4. Channel Temporal Autocorrelation

NTN propagation environments are often modeled under the **Wide-Sense Stationary Uncorrelated Scattering (WSSUS)** assumption over short observation intervals.

### 4.1 Diffuse Scattering (Jakes' Model)
For a purely scattered (Non-Line-of-Sight, NLoS) channel component, isotropic scattering leads to the classic Jakes' correlation model:
$$\rho(\Delta t) = J_0(2\pi f_{d,UE} \Delta t)$$
Where:
- $J_0(\cdot)$ is the zeroth-order Bessel function of the first kind.
- $f_{d,UE} = \frac{v_{UE}}{\lambda}$ is the maximum Doppler shift caused by the local motion of the UE.

### 4.2 Rician Fading Temporal Autocorrelation
For NTN channels, a strong dominant Line-of-Sight path is present alongside multipath components. The overall channel coefficient for a tap can be written as:
$$h(t) = \sqrt{\frac{K_R}{K_R + 1}} h_{LoS}(t) + \sqrt{\frac{1}{K_R + 1}} h_{NLoS}(t)$$
Where:
- $K_R$ is the Rician K-factor.
- $h_{LoS}(t) = e^{j 2\pi f_{d,sat} t}$ is the coherent component experiencing satellite Doppler shift.
- $h_{NLoS}(t)$ is the diffuse Rayleigh component.

The normalized autocorrelation function $R_h(\Delta t) = \mathbb{E}[h(t) h^*(t - \Delta t)]$ is:
$$R_h(\Delta t) = \frac{K_R}{K_R + 1} e^{j 2\pi f_{d,sat} \Delta t} + \frac{1}{K_R + 1} J_0(2\pi f_{d,UE} \Delta t)$$

---

## 5. Doppler Shift Mechanics

The high relative velocity between a LEO satellite and the ground results in severe Doppler shifts.

### 5.1 Vector Calculation of Doppler Shift
The instantaneous Doppler shift $f_{d,sat}(t)$ is given by:
$$f_{d,sat}(t) = -\frac{1}{\lambda} \frac{\mathrm{d}}{\mathrm{d}t} S_U(t) = \frac{\mathbf{v}_{rel}(t) \cdot \mathbf{\hat{r}}_{LOS}(t)}{c} f_c$$
Where:
- $\mathbf{v}_{rel}(t) = \mathbf{v}_{sat, ECI}(t) - \mathbf{v}_{UE, ECI}(t)$ is the relative velocity vector in ECI.
- $\mathbf{\hat{r}}_{LOS}(t) = \frac{\mathbf{r}_{sat, ECI}(t) - \mathbf{r}_{UE, ECI}(t)}{S_U(t)}$ is the unit line-of-sight vector pointing from the UE to the satellite.

```
       Satellite [r_sat, v_sat]
          *
         / \
        /   \
       /     \  LOS Vector (r_sat - r_UE)
      /       \
     /         \
    /           \
   *-------------*
 Earth Center   Ground UE [r_UE, v_UE]
```

### 5.2 Beam-Level vs. UE-Specific Doppler Shift
LEO satellites illuminate designated ground footprint zones via steerable beams.
1. **Beam-Level (Common) Doppler ($f_{d,beam}$)**: Calculated from the satellite to the geocentric center of the beam spot.
2. **UE-Specific Doppler ($f_{d,UE}$)**: Calculated from the satellite to the specific UE coordinates.

```
        LEO Satellite
             *
           / | \
          /  |  \
         /   |   \
        /    |    \
  Edge /     |     \ Edge
      v      v      v
   [ UE1 ] [Beam] [ UE2 ]
           [Ctr ]
```

#### Doppler Pre-compensation
To minimize the receiver frequency tracking requirements at the UE, the satellite transmitter (or gateway) pre-compensates the downlink frequency by shifting the carrier frequency based on the beam center:
$$f_{tx}(t) = f_c - f_{d,beam}(t)$$
The residual Doppler shift $\Delta f_{d,residual}(t)$ observed by a UE in the beam is:
$$\Delta f_{d,residual}(t) = f_{d,UE}(t) - f_{d,beam}(t)$$
This reduces the maximum frequency tracking offset from tens of kilohertz (full LEO Doppler) to a few kilohertz or hundreds of hertz, which can be compensated by fine tracking algorithms at the UE.

---

## 6. Doppler Dynamics Over Time (S-Curve)

As a LEO satellite passes overhead, the Doppler shift changes from a positive frequency shift (approaching) to a negative frequency shift (receding), tracing an S-shaped curve over time.

### 6.1 Range-Rate Derivation
Let the satellite move in a circular orbit at angular frequency $\omega_s$. The distance is:
$$d(t) = \sqrt{R_{UE}^2 + r_{sat}^2 - 2 R_{UE} r_{sat} \cos(\omega_s t)}$$
Taking the derivative with respect to time yields the range-rate $\dot{d}(t)$:
$$\dot{d}(t) = \frac{R_{UE} r_{sat} \omega_s \sin(\omega_s t)}{\sqrt{R_{UE}^2 + r_{sat}^2 - 2 R_{UE} r_{sat} \cos(\omega_s t)}}$$
The Doppler shift curve is then:
$$f_d(t) = -\frac{f_c}{c} \dot{d}(t) = -\frac{f_c}{c} \frac{R_{UE} r_{sat} \omega_s \sin(\omega_s t)}{d(t)}$$

```
 Doppler (Hz)
   ^
   |      +Max Doppler (Approaching)
   |       .---.
   |      /     \
   |     /       \
 0 +----+---------+----> Time (t)
   |   /  Zenith / (Closest approach: Doppler = 0)
   |  /         /
   | /     .---'
   v      -Max Doppler (Receding)
```

At the point of closest approach ($t = 0$), $\dot{d}(t) = 0$, resulting in $f_d(t) = 0$. The slope of the S-curve at this point is at its maximum absolute value:
$$\left. \frac{\mathrm{d}f_d(t)}{\mathrm{d}t} \right|_{t=0} = -\frac{f_c}{c} \frac{R_{UE} r_{sat} \omega_s^2}{r_{sat} - R_{UE}}$$
For higher carrier frequencies (e.g., Ka-band, 28 GHz) and lower orbit altitudes, this maximum rate of change is extremely high, requiring robust, continuous phase tracking loops.

---

## 7. Small-Scale Fading Models in NTN

NTN propagation channels differ significantly from terrestrial channels due to the presence of a strong direct path and a relative lack of dense local scatterers around the terminal under high elevation angles.

### 7.1 Rician Fading Model (Line-of-Sight - LoS)
Rician fading is used to model channels where a dominant direct Line-of-Sight (LoS) path is present alongside multiple scattered, random paths. The probability density function (PDF) of the Rician fading envelope $r$ is:
$$p(r) = \frac{r}{\sigma^2} \exp\left(-\frac{r^2 + s^2}{2\sigma^2}\right) I_0\left(\frac{r s}{\sigma^2}\right), \quad r \ge 0$$
Where:
- $s^2$ is the power of the dominant LoS component.
- $2\sigma^2$ is the average power of the scattered multipaths.
- $I_0(\cdot)$ is the zero-order modified Bessel function of the first kind.

The **Rician K-factor** is defined as the ratio of the dominant path power to the scattered path power:
$$K_R = \frac{s^2}{2\sigma^2}$$
In NTN, $K_R$ is highly dependent on the elevation angle $\epsilon$. As the satellite moves closer to the zenith (higher $\epsilon$), shadowing decreases, and the direct path dominates, causing $K_{dB} = 10\log_{10}(K_R)$ to increase. This is modeled empirically as:
$$K_{dB} = a \epsilon + b$$
where $a$ and $b$ are environment-specific parameters.

### 7.2 Rayleigh Fading Model (Non-Line-of-Sight - NLoS)
When the direct path is completely blocked (Non-Line-of-Sight or NLoS conditions), there is no dominant component ($s = 0$). Mathematically, this reduces the Rician K-factor to zero ($K_R = 0$, or $K_{dB} = -\infty$).
Since $I_0(0) = 1$, the Rician PDF simplifies directly to the **Rayleigh distribution**:
$$p(r) = \frac{r}{\sigma^2} \exp\left(-\frac{r^2}{2\sigma^2}\right), \quad r \ge 0$$
Thus, **NLoS channels are modeled as Rayleigh fading channels**, which represents a special case of the Rician model where the dominant component is absent.

### 7.3 Loo's Model (Shadowed Rician)
For land mobile satellite (LMS) links where the direct LoS path is shadowed by foliage or buildings, the amplitude of the dominant component $s$ is modeled as a log-normal random variable instead of a constant. The total envelope PDF is:
$$p(r) = \frac{r}{\sigma^2 \sqrt{2\pi d_0}} \int_0^\infty \frac{1}{z} \exp\left( - \frac{(\ln z - \mu_0)^2}{2 d_0} - \frac{r^2 + z^2}{2\sigma^2} \right) I_0\left( \frac{r z}{\sigma^2} \right) \mathrm{d}z$$
Where $\mu_0$ and $d_0$ are the mean and variance of the log-normal shadowing process.

### 7.4 3GPP TR 38.901 NTN Profiles
Standardized link-level simulations utilize Tapped Delay Line (TDL) profiles adapted for satellite channels:

*   **NLoS Profiles (Rayleigh Fading):**
    *   **NTN-TDL-A** and **NTN-TDL-B**: Model Non-Line-of-Sight environments (e.g., urban canyons, heavy forestry) where all taps undergo independent Rayleigh fading ($K_R = 0$).
*   **LoS Profiles (Rician Fading):**
    *   **NTN-TDL-C** and **NTN-TDL-D**: Model Line-of-Sight environments (e.g., open rural or suburban terrains). In these profiles, the first (shortest delay) tap represents the direct path and is modeled with **Rician fading** (incorporating a configurable or elevation-dependent K-factor), while the remaining delayed taps are modeled with Rayleigh fading.
    
> [!NOTE]
> Terrestrial 3GPP TR 38.901 profiles classify TDL-A, B, and C as NLoS, and TDL-D and E as LoS. However, in the satellite NTN standard (3GPP TR 38.811), **NTN-TDL-C is classified as a Line-of-Sight (LoS) profile** along with NTN-TDL-D.