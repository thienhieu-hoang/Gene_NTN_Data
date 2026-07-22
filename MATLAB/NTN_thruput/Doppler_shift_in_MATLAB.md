Based on the source code in `nrTDLChannel.m`, here is the detailed breakdown of how the Doppler shifts are modeled, separated, and applied to the waveform:

---

### 1. How is the Doppler shift added? Are both added sample-wise?
**Yes, both are added/applied sample-wise, but they are applied in completely different ways:**

#### A. Satellite Doppler Shift (`SatelliteDopplerShift`)
The satellite Doppler shift is a **flat, constant frequency offset** ($f_{d,\text{sat}}$) applied sample-by-sample to the channel's path gains. 
In `nrTDLChannel.m`, this is implemented as follows:
```matlab
% Apply Doppler shift due to satellite for all the paths
if nrTDLChannel.isNTN(obj)
    % Get the phase shift to be applied at each sample
    dopplerPhase = obj.theSatelliteDopplerPhase + ...
        2*pi*(1/obj.theChannel.SampleRate)*obj.SatelliteDopplerShift * (0:obj.theChannel.NumSamples)';
    obj.theSatelliteDopplerPhase = dopplerPhase(end,1);
    % Apply the phase shifts to the path gains
    pathgains = pathgains .* exp(1j*dopplerPhase(1:(end-1),1));
end
```
*   **Sample-wise application**: A phase vector (`dopplerPhase`) is computed. It accumulates linearly with every sample: $\Delta \theta = 2\pi \cdot f_{d,\text{sat}} \cdot T_s$ (where $T_s = 1/F_s$).
*   The generated path gains are then multiplied sample-wise by `exp(1j * dopplerPhase)`. Because the input waveform is filtered by these path gains, this translates to shifting the entire output signal in frequency by $+f_{d,\text{sat}}$ Hz.

#### B. UE Doppler Shift (`MaximumDopplerShift`)
The Doppler shift due to the UE's velocity is **not** a simple frequency translation. Instead, it is modeled as **time-varying statistical multipath fading** (Doppler spread).
In `nrTDLChannel.m`
```matlab
c = comm.MIMOChannel;
...
c.MaximumDopplerShift = obj.MaximumDopplerShift;
c.DopplerSpectrum = doppler('Jakes');
```
*   **Sample-wise variation**: The underlying `comm.MIMOChannel` uses a sum-of-sinusoids method to generate Rayleigh or Rician fading coefficients that vary sample-by-sample. 
*   The rate of change (fading speed) of these coefficients is determined by the Jakes' Doppler spectrum, governed by `MaximumDopplerShift` ($f_{d,\text{UE}} = \frac{v_{\text{UE}} \cdot f_c}{c}$). 
*   For Rician fading (LOS), it also adds a direct path Doppler shift component set to `0.7 * MaximumDopplerShift`.

---

### 2. Are the Doppler shifts separated?
**Yes, they are completely decoupled and treated as two separate physical phenomena:**
1.  **Orbit Doppler** is modeled as a **pure frequency offset** (`SatelliteDopplerShift`). It is common and uniform across all paths (co-phased translation).
2.  **UE Doppler** is modeled as **Doppler spread / fading** (`MaximumDopplerShift`). It represents diffuse multipath scattering around the moving terminal.

---

### 3. Does the code not simulate the Doppler between Sat and UE straightly?
**Correct, the code does not simulate the physical Doppler between the satellite and the UE directly.**

A high-fidelity physical simulation would:
1. Propagate the satellite orbit coordinates $\mathbf{r}_{\text{sat}}(t)$ and velocity vectors $\mathbf{v}_{\text{sat}}(t)$ sample-by-sample.
2. Track the moving UE coordinates $\mathbf{r}_{\text{UE}}(t)$ and velocity vectors $\mathbf{v}_{\text{UE}}(t)$ sample-by-sample.
3. Calculate the time-varying Line-of-Sight vector $\mathbf{\hat{r}}_{\text{LOS}}(t)$ and the relative velocity vector $\mathbf{v}_{\text{rel}}(t) = \mathbf{v}_{\text{sat}}(t) - \mathbf{v}_{\text{UE}}(t)$.
4. Apply a **time-varying** Doppler shift:
   $$f_d(t) = -\frac{\mathbf{v}_{\text{rel}}(t) \cdot \mathbf{\hat{r}}_{\text{LOS}}(t)}{\lambda}$$

**Instead, this toolbox model simplifies the math:**
*   The caller must supply a **static constant** `SatelliteDopplerShift` (usually computed at the starting position/beam-center).
*   The channel model shifts the entire signal frequency by this constant value.
*   It assumes that the orbital geometry remains static over the duration of the processed slot/frame, and does not model the changing velocity vector projections or S-curve dynamics during the signal transmission itself.


---

Here is the explanation for both of your questions, detailing the signal processing mechanics and why the end-to-end channel appears so simple in your plots.

---

### 1. Does pre-compensation completely cancel out the satellite Doppler?
**Yes, it cancels it out perfectly.**

In the simulation:
1.  **At the Transmitter**: You pre-compensate the waveform by shifting it in frequency by $-f_{d,\text{sat}}$:
    $$x_{\text{tx}}(t) = x(t) \cdot e^{-j 2\pi f_{d,\text{sat}} t}$$
2.  **In the Channel**: The channel filters the signal and shifts its frequency by $+f_{d,\text{sat}}$:
    $$y_{\text{rx}}(t) = \text{Channel}\big\{x_{\text{tx}}(t)\big\} \cdot e^{+j 2\pi f_{d,\text{sat}} t}$$
3.  Assuming a flat channel with gain $h(t)$ for simplicity, the received signal is:
    $$y_{\text{rx}}(t) = \left( x(t) \cdot e^{-j 2\pi f_{d,\text{sat}} t} \cdot h(t) \right) \cdot e^{+j 2\pi f_{d,\text{sat}} t} = x(t) \cdot h(t)$$
    The frequency shifts $e^{-j 2\pi f_{d,\text{sat}} t}$ and $e^{+j 2\pi f_{d,\text{sat}} t}$ **multiply to $1$**, completely neutralizing the satellite Doppler.

#### Why the plotted channel looks like a normal, simple TDL channel:
When you plot the **effective channel** seen after demodulation, the satellite Doppler shift has been completely removed by this cancelation. The only remaining impairment is:
*   The multipath delay profile (the static taps: TDL-A, TDL-B, etc.).
*   The slow time-variation (fading) of those taps due to the local UE movement (`MaximumDopplerShift`), which typically ranges from a few Hertz to a few hundred Hertz. 
This is why the channel plots look identical to a standard, terrestrial 3GPP TDL channel.

---

### 2. How does multiplying the channel's path gains by `exp(1j*dopplerPhase)` add Doppler to the waveform?
It seems counter-intuitive at first: **why are we applying the frequency shift to the channel coefficients (`pathgains`) rather than the waveform itself?**

Mathematically, they are equivalent. Here is the proof:

#### A. The Multipath Filtering Equation
When a signal $x(t)$ passes through a multipath fading channel, the output waveform $y(t)$ is computed by convolving the input with the time-varying channel impulse response:
$$y(t) = \sum_{l=0}^{L-1} g_l(t) \cdot x(t - \tau_l)$$
Where:
*   $g_l(t)$ is the complex path gain of the $l$-th path at sample time $t$.
*   $\tau_l$ is the delay of the $l$-th path.

#### B. Applying Phase to the Path Gains
In the code block you highlighted, the channel multiplies every path gain $g_l(t)$ by the satellite Doppler phase factor $e^{j \theta_{\text{sat}}(t)}$ at each sample time $t$:
$$\tilde{g}_l(t) = g_l(t) \cdot e^{j 2 \pi f_{d,\text{sat}} t}$$
When the input waveform $x(t)$ is filtered through this modified channel, the output waveform $\tilde{y}(t)$ becomes:
$$\tilde{y}(t) = \sum_{l=0}^{L-1} \tilde{g}_l(t) \cdot x(t - \tau_l) = \sum_{l=0}^{L-1} \left( g_l(t) \cdot e^{j 2 \pi f_{d,\text{sat}} t} \right) \cdot x(t - \tau_l)$$

#### C. Factoring out the Doppler term
Because the satellite Doppler shift is a **common frequency offset** that is identical for all paths, the term $e^{j 2 \pi f_{d,\text{sat}} t}$ does not depend on the path index $l$. 

We can pull it out of the summation:
$$\tilde{y}(t) = e^{j 2 \pi f_{d,\text{sat}} t} \cdot \sum_{l=0}^{L-1} g_l(t) \cdot x(t - \tau_l)$$
$$\tilde{y}(t) = e^{j 2 \pi f_{d,\text{sat}} t} \cdot y(t)$$

### Summary
Multiplying the channel path gains sample-wise by $e^{j 2 \pi f_{d,\text{sat}} t}$ is **mathematically identical to multiplying the output waveform of the channel by $e^{j 2 \pi f_{d,\text{sat}} t}$**. 

By applying the phase rotation directly to the `pathgains`, the toolbox:
1. Implements the physical effect of a Doppler shift (carrier frequency translation) during propagation.
2. Keeps the channel representation mathematically unified, allowing the `pathgains` output to reflect the true, combined physical channel state (fading + Doppler offset).