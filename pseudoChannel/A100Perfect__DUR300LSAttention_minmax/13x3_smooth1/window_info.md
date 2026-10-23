# FDA Smooth Window Information: `13x3_smooth1`

- **Window Variant:** `smooth1` (Tukey Outward (Compact Roll-Off))
- **Base Core Window Size:** $13 \times 13$ ($13$ Delay bins $\times 3$ Doppler bins)
- **Window Type:** `tukey_outward`
- **Delay Transition ($taper\_h$):** `2` rows
- **Doppler Transition ($taper\_w$):** `1` columns
- **Description:** Keeps core 13xWw flat at 1.0 (100% Target). Adds 1-element Doppler taper [0.50] and 2-element Delay taper [0.75, 0.25].
- **Generated On:** 2026-10-23 11:17:00

---

## Rationale: Eliminating the Boundary "Outside Ring"

In standard FDA with a hard rectangular boxcar window ($13 \times 3$), an abrupt jump discontinuity occurs between the Target and Source amplitude spectra at the window boundaries.
When transforming back to the Time-Frequency domain via 2D inverse Fourier transform (`F_inverse_DD`), this frequency step discontinuity convolves with sinc sidelobes, causing **Gibbs ringing** and a visible rectangular boundary "ring".

The `smooth1` window maintains the central core strictly at **$1.0$** (preserving 100% of the target domain dominant delay-Doppler energy) while softly tapering outward to **$0.0$** over smooth cosine transitions.

---

## Exact Profile Weights

### 1. Doppler Profile $w_x$ (OFDM Symbols, dimension 14, center column 8)

| Column Index | Distance from DC | Weight $w_x$ | Region |
| :---: | :---: | :---: | :--- |
| **Col 5** | $\vert x - 8 \vert = 3$ | `0.0000` | Outer (100% Source) |
| **Col 6** | $\vert x - 8 \vert = 2$ | `0.5000` | *Smooth Transition* |
| **Col 7** | $\vert x - 8 \vert = 1$ | `1.0000` | **Core (100% Target)** |
| **Col 8** | $\vert x - 8 \vert = 0$ | `1.0000` | **Core (100% Target)** |
| **Col 9** | $\vert x - 8 \vert = 1$ | `1.0000` | **Core (100% Target)** |
| **Col 10** | $\vert x - 8 \vert = 2$ | `0.5000` | *Smooth Transition* |
| **Col 11** | $\vert x - 8 \vert = 3$ | `0.0000` | Outer (100% Source) |

### 2. Delay Profile $w_y$ (Subcarriers, dimension 132, center row 67)

| Row Index | Distance from DC | Weight $w_y$ | Region |
| :---: | :---: | :---: | :--- |
| **Row 58** | $\vert y - 67 \vert = 9$ | `0.0000` | Outer (100% Source) |
| **Row 59** | $\vert y - 67 \vert = 8$ | `0.2500` | *Smooth Transition* |
| **Row 60** | $\vert y - 67 \vert = 7$ | `0.7500` | *Smooth Transition* |
| **Row 61** | $\vert y - 67 \vert = 6$ | `1.0000` | **Core (100% Target)** |
| **Row 62** | $\vert y - 67 \vert = 5$ | `1.0000` | **Core (100% Target)** |
| **Row 63** | $\vert y - 67 \vert = 4$ | `1.0000` | **Core (100% Target)** |
| **Row 64** | $\vert y - 67 \vert = 3$ | `1.0000` | **Core (100% Target)** |
| **Row 65** | $\vert y - 67 \vert = 2$ | `1.0000` | **Core (100% Target)** |
| **Row 66** | $\vert y - 67 \vert = 1$ | `1.0000` | **Core (100% Target)** |
| **Row 67** | $\vert y - 67 \vert = 0$ | `1.0000` | **Core (100% Target)** |
| **Row 68** | $\vert y - 67 \vert = 1$ | `1.0000` | **Core (100% Target)** |
| **Row 69** | $\vert y - 67 \vert = 2$ | `1.0000` | **Core (100% Target)** |
| **Row 70** | $\vert y - 67 \vert = 3$ | `1.0000` | **Core (100% Target)** |
| **Row 71** | $\vert y - 67 \vert = 4$ | `1.0000` | **Core (100% Target)** |
| **Row 72** | $\vert y - 67 \vert = 5$ | `1.0000` | **Core (100% Target)** |
| **Row 73** | $\vert y - 67 \vert = 6$ | `1.0000` | **Core (100% Target)** |
| **Row 74** | $\vert y - 67 \vert = 7$ | `0.7500` | *Smooth Transition* |
| **Row 75** | $\vert y - 67 \vert = 8$ | `0.2500` | *Smooth Transition* |
| **Row 76** | $\vert y - 67 \vert = 9$ | `0.0000` | Outer (100% Source) |

### 3. Central 2D Mask Slice ($M(\tau, \nu)$ around DC)

Values represent the fraction of Target amplitude spectrum ($1.0$ = 100% Target, $0.0$ = 100% Source):

| Row \\ Col | **C5** | **C6** | **C7** | **C8** | **C9** | **C10** | **C11** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R58** | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` |
| **R59** | `0.00` | *0.13* | *0.25* | *0.25* | *0.25* | *0.13* | `0.00` |
| **R60** | `0.00` | *0.38* | *0.75* | *0.75* | *0.75* | *0.38* | `0.00` |
| **R61** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R62** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R63** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R64** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R65** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R66** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R67** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R68** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R69** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R70** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R71** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R72** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R73** | `0.00` | *0.50* | **1.00** | **1.00** | **1.00** | *0.50* | `0.00` |
| **R74** | `0.00` | *0.38* | *0.75* | *0.75* | *0.75* | *0.38* | `0.00` |
| **R75** | `0.00` | *0.13* | *0.25* | *0.25* | *0.25* | *0.13* | `0.00` |
| **R76** | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` |

---

## Mathematical Formula

$$\text{For outward step } k = d - r_{\text{core}} \in \{1, \dots, \text{taper}\}:\quad w(k) = \frac{1}{2} \left(1 + \cos\left(\frac{k \pi}{\text{taper} + 1}\right)\right)$$

- Along Doppler ($taper\_w = 1$): $k=1 \implies \frac{1}{2}(1 + \cos(\pi/2)) = \mathbf{0.50}$.
- Along Delay ($taper\_h = 2$): $k=1 \implies \frac{1}{2}(1 + \cos(\pi/3)) = \mathbf{0.75}$, $k=2 \implies \frac{1}{2}(1 + \cos(2\pi/3)) = \mathbf{0.25}$.
