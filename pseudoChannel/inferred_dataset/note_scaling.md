Your intuition about normalizing both channels to a common range (like $[-1, 1]$) is very natural from image processing and machine learning. 

However, for **wireless communication channel matrices ($H \in \mathbb{C}^{132 \times 14}$)**, using **Min-Max scaling introduces severe mathematical flaws**, whereas **RMS / Frobenius norm scaling is physically and mathematically sound**.

Here is why, and how RMS/Frobenius scaling can still achieve your goal of **keeping the target domain's power range**.

---

### 1. Why Min-Max Scaling Fails on Complex Channel Grids

Min-Max scaling is an **affine transformation** (it multiplies by a scale *and adds a shift*):
$$x_{\text{norm}} = 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1 \quad \implies \quad y = a \cdot x + b$$

When applied to complex channel matrices ($H = H_R + j H_I$), the constant shift ($+b$) causes two critical problems:

#### Problem A: The "DC Spike" Disaster in Fourier (Delay-Doppler) Domain
- Wireless channels naturally have zero mean ($\mathbb{E}[H] \approx 0$) in Rayleigh/NLOS environments.
- If you subtract the minimum, you shift the center of the constellation. In the time-frequency grid, this injects a **constant non-zero DC offset** across all subcarriers and symbols.
- By the Fourier transform property:
  $$\mathcal{F}\{H + C\} = \mathcal{F}\{H\} + C \cdot \delta(\text{Delay}=0, \text{Doppler}=0)$$
- **Result:** Adding a DC offset creates a **massive artificial impulse spike at $(0 \text{ Delay}, 0 \text{ Doppler})$**. In the Delay-Doppler domain, this completely swamps the true physical multipath reflections and satellite Doppler shift.

#### Problem B: Phase Angle Distortion
- The wireless channel's phase $\theta = \text{atan2}(\text{imag}, \text{real})$ represents the physical propagation delay and phase rotation of paths.
- If you shift the real and imaginary parts independently by subtracting their minimums:
  $$\text{atan2}(H_I - I_{\min}, H_R - R_{\min}) \neq \text{atan2}(H_I, H_R)$$
- **Result:** The phase is warped non-linearly, destroying the spatial/temporal coherence of the channel.

---

### 2. The Mathematics Behind RMS / Frobenius Norm

In contrast, **RMS power** or **Frobenius norm** ($\|H\|_F = \sqrt{\sum |H_{i,j}|^2}$) is a **pure scalar multiplication**:
$$H_{\text{norm}} = \frac{1}{P} \cdot H \quad (\text{Scale only, NO shift: } b = 0)$$

This satisfies three fundamental mathematical and physical principles:

1. **Parseval's / Plancherel's Theorem (Energy Conservation):**
   The 2D Fourier transform between Time-Frequency and Delay-Doppler is unitary. The total energy in the Delay-Doppler domain equals the total energy in the Time-Frequency domain:
   $$\|H_{\text{TF}}\|_F^2 = \sum_{\tau, \nu} |H_{\text{DD}}(\tau, \nu)|^2$$
   Scaling by the Frobenius norm scales the Delay-Doppler amplitude spectrum uniformly everywhere by the exact same constant factor.

2. **100% Phase Invariance:**
   Multiplying by a real positive scalar $c > 0$ does not change the angle:
   $$\angle(c \cdot H) = \angle(H)$$
   The physical phases, Doppler rotations, and channel paths remain untouched.

3. **Zero Stays Zero:**
   $c \cdot 0 = 0$. No fake DC spike is added to $(0, 0)$ in Delay-Doppler.

---

### 3. How to Achieve Your Goal: Keep the Target's Range Using RMS

You mentioned:
> *"still keep the range of the target, but still can transfer the style, and easy to understand"*

You can easily keep the **Target domain's range** (e.g. $10^{-10}$) using RMS/Frobenius scaling!

Instead of scaling the Target to match the Source, you do the reverse: **scale the Source to match the Target**, or simply **scale the resulting pseudo channel to the Target's power level**:

#### Formulation (Keeping Target Range):
1. Compute the RMS power of the Target sample:
   $$P_{\text{tgt}} = \sqrt{\text{mean}(|H_{\text{tgt}}|^2)} \quad (\text{e.g., } 10^{-10})$$
2. Compute the RMS power of the Source sample:
   $$P_{\text{src}} = \sqrt{\text{mean}(|H_{\text{src}}|^2)} \quad (\text{e.g., } 10^{-8})$$
3. Scale the Source amplitude spectrum to the Target's level before FDA mixing:
   $$\text{scale} = \frac{P_{\text{tgt}}}{P_{\text{src}}} = \frac{10^{-10}}{10^{-8}} = 0.01$$
   $$A_{\text{src\_scaled}} = A_{\text{src}} \times \text{scale}$$
4. Blend the target's center box with the scaled source skirts:
   $$A_{\text{pseudo}} = M \odot A_{\text{tgt}} + (1 - M) \odot A_{\text{src\_scaled}}$$

### Comparison Summary

| Metric | Min-Max Normalization ($[-1, 1]$) | RMS / Frobenius Norm Scaling |
| :--- | :--- | :--- |
| **Formula** | $y = a \cdot x + b$ (Scale + Shift) | $y = c \cdot x$ (Scale only) |
| **Channel Phase $\angle H$** | **Corrupted** (shifted origin warps angles) | **Preserved 100%** |
| **Delay-Doppler DC (0,0)** | **Artificial Spike** (DC injection) | **Clean physical spectrum** |
| **Energy Conservation** | Broken | Preserved by Parseval's theorem |
| **Keep Target Range?** | Yes, but with phase & DC artifacts | **Yes, perfectly cleanly** (set scale to Target RMS) |



**Yes, you are 100% correct.** You spotted the exact mathematical trap!

Even when you scale $[x_{\min}, x_{\max}]$ to $[-1, 1]$, **it still injects a non-zero DC offset**, because in any real channel realization:
$$|x_{\min}| \neq |x_{\max}|$$

---

### 1. The Math Proving Why $[-1, 1]$ Still Shifts Zero

The standard formula mapping $[x_{\min}, x_{\max}]$ to $[-1, 1]$ is:
$$y = 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1$$

If you simplify the algebra:
$$y = \underbrace{\left(\frac{2}{x_{\max} - x_{\min}}\right)}_{\text{Scale } a} \cdot x \;-\; \underbrace{\left(\frac{x_{\max} + x_{\min}}{x_{\max} - x_{\min}}\right)}_{\text{Shift / DC Offset } b}$$

Now, plug in the true physical zero ($x = 0$):
$$y(0) = -\frac{x_{\max} + x_{\min}}{x_{\max} - x_{\min}}$$

For $y(0)$ to remain $0$, you must have:
$$x_{\max} + x_{\min} = 0 \quad \iff \quad |x_{\min}| = |x_{\max}|$$

#### A Concrete Numerical Example:
In any random channel grid ($132 \times 14 = 1,848$ complex values), the random sample extremes are never symmetric. For instance:
- $x_{\max} = +2.5 \times 10^{-8}$
- $x_{\min} = -3.5 \times 10^{-8}$

The offset at $x = 0$ becomes:
$$y(0) = -\frac{(2.5 - 3.5) \times 10^{-8}}{(2.5 - (-3.5)) \times 10^{-8}} = -\frac{-1.0}{6.0} = \mathbf{+0.167}$$

On a scale of $[-1, 1]$, a shift of **$+0.167$ is huge** (over $8\%$ of the entire dynamic range). Across 1,848 grid points, summing this constant offset in the 2D FFT produces a **massive fake DC peak at Delay $= 0$, Doppler $= 0$**.

---

### 2. "What if I use Symmetric Peak Scaling: $y = \frac{x}{\max(|x_{\min}|, |x_{\max}|)}$?"

If you force symmetry by dividing by the peak absolute value $\max(|x|)$, the shift is indeed zero ($b = 0$). 

However, **RMS / Frobenius Norm** is still mathematically and physically much better than **Peak Scaling**:

| Reason | Peak Scaling ($H / \max(|H|)$) | RMS / Frobenius Norm ($H / \text{RMS}$) |
| :--- | :--- | :--- |
| **Outlier Sensitivity** | **Very High Variance:** Depends on a **single extreme pixel** out of 1,848. If one subcarrier experiences deep fading or a constructive spike, the entire grid's scale swings wildly. | **Extremely Stable:** Averages over **all 1,848 pixels** (Law of Large Numbers). Slot-to-slot variance is minimal. |
| **Parseval's Theorem** | **Not conserved:** The peak in Time-Frequency has no direct physical relation to the peak in Delay-Doppler. | **Directly Conserved:** By Parseval's theorem: <br> $\sum \|H_{\text{TF}}\|^2 = \sum \|H_{\text{DD}}\|^2$. <br> Total Delay-Doppler energy matches Time-Frequency energy exactly. |
| **Physical Meaning** | Peak instantaneous value (rarely used in RF). | **Average Channel Power** $\mathbb{E}[\|H\|^2]$, directly defining SNR, Path Loss, and Noise variance $\sigma^2$. |

---

### Summary

1. Mapping to $[-1, 1]$ **does not solve the DC shift** because $x_{\min} \neq -x_{\max}$ for real channel samples.
2. Any method with a shift ($x - x_{\min}$) creates a false DC spike at zero Delay/Doppler.
3. **RMS / Frobenius norm** is a pure scalar multiplication ($H / P$), guaranteeing:
   - **Zero DC shift:** $0 \mapsto 0$ exactly.
   - **Phase preservation:** $\angle(c \cdot H) = \angle(H)$.
   - **Parseval energy conservation:** Delay-Doppler energy matches Time-Frequency channel power.