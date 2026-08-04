# Technical Note on Doppler Precompensation and Effective Channel Estimation in NTN

This document provides a concise reference for **Doppler Precompensation**, **Effective Channel Formulation**, **Pilot-based Channel Estimation**, and **Reconstruction Evaluation** in Non-Terrestrial Network (NTN) communication systems.

---

## 1. System Pipeline & Signal Flow

### 1.1 Uncompensated Baseline
In high-velocity LEO (Low Earth Orbit) satellite communications, large relative velocities cause significant Doppler shifts ($f_d$). The **perfect original channel** is defined including the physical Doppler effect:

$$h_{\text{original}}(t) = h(t) \cdot e^{j 2\pi f_d t}$$

Without Tx precompensation, the received waveform experiences severe carrier frequency offset (CFO) and inter-carrier interference (ICI):

$$\text{txWaveform } x(t) \stackrel{h_{\text{original}}}{\longrightarrow} \text{rxWaveform } y(t) = (x * h_{\text{original}})(t) + n(t) = (x * h)(t) \cdot e^{j 2\pi f_d t} + n(t)$$

where:
- $h(t, \tau)$ is the base multipath channel impulse response.
- $f_d(t)$ is the dynamic physical Doppler shift: $f_d(t) = \frac{v(t)}{\lambda} \cos\theta(t)$.
- $n(t)$ is additive white Gaussian noise (AWGN).

---

### 1.2 Doppler Precompensation & Effective Channel ($H_{\text{effective}}$)

To mitigate large Doppler frequency offsets before the signal reaches the receiver:

1. **Tx Doppler Precompensation**: The transmitter pre-rotates the transmit waveform using estimated/predicted ephemeris Doppler frequency $f_{\text{pre}}(t)$:
   $$x_{\text{pre}}(t) = x(t) \cdot e^{-j 2\pi f_{\text{pre}} t}$$

   **Physical Sequential Transmission**:
   $$y_{\text{comp}}(t) = (x_{\text{pre}} * h_{\text{original}})(t) + n(t)$$
   $$\Rightarrow y_{\text{comp}}(t) = \left( (x(t) \cdot e^{-j 2\pi f_{\text{pre}} t}) * h_{\text{original}} \right)(t) + n(t)$$

   **Effective Channel Formulation**:
   $$\Rightarrow y_{\text{comp}}(t) = \left( x(t) * (h_{\text{original}}(t) \cdot e^{-j 2\pi f_{\text{pre}} t}) \right) + n(t)$$
   $$\Rightarrow y_{\text{comp}}(t) = (x(t) * h_{\text{effective}}(t)) + n(t)$$

   where the **Effective Channel** is defined as:
   $$h_{\text{effective}}(t) = h_{\text{original}}(t) \cdot e^{-j 2\pi f_{\text{pre}} t} = h(t) \cdot e^{j 2\pi (f_d - f_{\text{pre}}) t}$$

2. **Frequency-Domain Grid Representation**:
   - Physical Sequential grid:
     $$Y_{\text{comp}} = X_{\text{comp}} \odot H_{\text{original}} + N$$
   - Effective Channel grid:
     $$Y_{\text{comp}} = X \odot H_{\text{effective}} + N$$

---

## 2. Matrix Signal Model & LS Channel Estimation

### 2.1 Discrete OFDM Signal Model
For an OFDM system with subcarrier index $k \in \{1, \dots, N_f\}$ and symbol index $l \in \{1, \dots, N_t\}$, the received grid $Y$ is expressed as:

$$Y[k, l] = X[k, l] \cdot H_{\text{effective}}[k, l] + N[k, l]$$

or element-wise:
$$Y = X \odot H_{\text{effective}} + N$$

where:
- $X \in \mathbb{C}^{N_f \times N_t}$ is the transmitted symbol matrix (data + pilots).
- $Y \in \mathbb{C}^{N_f \times N_t}$ is the received subcarrier grid.
- $H_{\text{effective}} \in \mathbb{C}^{N_f \times N_t}$ is the effective complex channel matrix.
- $N \sim \mathcal{CN}(0, \sigma_n^2)$ is complex AWGN matrix.

---

### 2.2 Least-Squares (LS) Pilot Estimation
At pilot symbol locations $(k_p, l_p) \in \mathcal{P}$ (e.g., DM-RS or CSI-RS subcarriers):

$$\hat{H}_{\text{LS}}[k_p, l_p] = \frac{Y[k_p, l_p]}{X[k_p, l_p]}$$

Substituting the received signal model:
$$\hat{H}_{\text{LS}}[k_p, l_p] = H_{\text{effective}}[k_p, l_p] + \frac{N[k_p, l_p]}{X[k_p, l_p]}$$

**Key Characteristics**:
- Unbiased channel estimate: $\mathbb{E}[\hat{H}_{\text{LS}}] = H_{\text{effective}}$.
- Noise Variance: $\text{Var}\left(\hat{H}_{\text{LS}}\right) = \frac{\sigma_n^2}{|X[k_p, l_p]|^2}$.

---

## 3. Channel Interpolation & Reconstruction

To recover the complete complex channel matrix $\hat{H} \in \mathbb{C}^{N_f \times N_t}$ across all subcarriers and symbols from sparse pilot estimates $\hat{H}_{\text{LS}}$:

### 3.1 Conventional Interpolation Methods
1. **2D Linear / Spline Interpolation**:
   Uses scattered pilot points to interpolate over the subcarrier ($k$) and symbol ($l$) grid:
   $$\hat{H}_{\text{linear}}[k, l] = \text{interp2D}\left(\{(k_p, l_p), \hat{H}_{\text{LS}}[k_p, l_p]\}\right)$$
2. **LMMSE (Linear Minimum Mean Square Error)**:
   Leverages channel autocorrelation matrices in time $R_t$ and frequency $R_f$ to suppress noise:
   $$\hat{H}_{\text{LMMSE}} = R_{H H_p} \left(R_{H_p H_p} + \sigma_n^2 (X X^H)^{-1}\right)^{-1} \hat{H}_{\text{LS}}$$

### 3.2 Deep Learning / Attention-based Estimation
Modern NTN estimation networks (e.g., CNN, U-Net, Attention, UDA) process noisy $\hat{H}_{\text{LS}}$ or linearly interpolated grids $\hat{H}_{\text{linear}}$ to output high-fidelity channel predictions $\hat{H}$.

---

## 4. Evaluation Metrics

To quantify how closely the estimated/predicted channel matrix $\hat{H}$ matches the true effective channel $H$ (or $H_{\text{original}}$):

### 4.1 Structural Similarity Index Measure (SSIM)
While traditional metrics (e.g., MSE) measure pixel-wise error, **SSIM** evaluates structural patterns (fade duration, Doppler trajectory, time-frequency correlation) of the channel magnitude response $|H| \in \mathbb{R}^{N_f \times N_t}$:

$$\text{SSIM}(H, \hat{H}) = \frac{(2\mu_{H} \mu_{\hat{H}} + C_1)(2\sigma_{H\hat{H}} + C_2)}{(\mu_{H}^2 + \mu_{\hat{H}}^2 + C_1)(\sigma_{H}^2 + \sigma_{\hat{H}}^2 + C_2)}$$

where:
- $\mu_H, \mu_{\hat{H}}$: Mean channel response magnitudes of $H$ and $\hat{H}$.
- $\sigma_H^2, \sigma_{\hat{H}}^2$: Variances of $H$ and $\hat{H}$.
- $\sigma_{H\hat{H}}$: Covariance between $H$ and $\hat{H}$.
- $C_1 = (K_1 L)^2, C_2 = (K_2 L)^2$: Stability constants to prevent division by zero ($L$ is the dynamic range of magnitude values).

> **Interpretation**: $\text{SSIM} \in [-1, 1]$. An SSIM close to **1.0** indicates near-perfect spatial and structural preservation of channel fading features across time and frequency.

---

### 4.2 Complementary Metrics
1. **Normalized Mean Squared Error (NMSE)**:
   $$\text{NMSE (dB)} = 10 \log_{10} \left( \frac{\|H - \hat{H}\|_F^2}{\|H\|_F^2} \right)$$
2. **Peak Signal-to-Noise Ratio (PSNR)**:
   $$\text{PSNR (dB)} = 10 \log_{10} \left( \frac{\max(|H|)^2}{\text{MSE}(H, \hat{H})} \right)$$
