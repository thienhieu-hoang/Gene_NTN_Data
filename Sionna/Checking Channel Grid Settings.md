# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

@[c:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\Sionna\OpenNTN\example\channel_response copy.ipynb] can you check the code of .ipynb file? or only .py file? 
can you check the setting here - it is generate channel grid with how many subcarriers and how many OFDM symbols?

*Viewed [channel_response copy.ipynb](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response%20copy.ipynb) *

*Viewed [channel_response copy.ipynb](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response%20copy.ipynb) *

*Viewed [channel_response.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response.py) *

### Planner Response

Yes, **I can check both `.ipynb` (Jupyter Notebook) files and `.py` (Python) files**. I can read, inspect, and analyze the code, parameters, and execution logic in both file formats.

---

### Channel Grid Configuration Analysis

Checking the configuration in [channel_response copy.ipynb](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response%20copy.ipynb#L102-L122) (and also matched in [channel_response.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response.py#L60-L78)):

#### 1. Number of OFDM Symbols
* **`14` OFDM symbols** (`num_ofdm_symbols = 14`).

#### 2. Number of Subcarriers
* **Total FFT Subcarriers:** **`256`** 
  * Calculated as `nFFT = 2**ceil(log2(11 * 12)) = 2^8 = 256`.
* **Active (Effective) Subcarriers:** **`132`**
  * Calculated from `RBBlock = 11` Resource Blocks $\times 12$ subcarriers/RB = `132` active subcarriers.
* **Guard Subcarriers:** **`124`** (`num_guard_carriers = (62, 62)` zeroed out on left and right).

---

### Summary of Resource Grid Parameters (`rg`)

| Parameter | Value | Note / Source |
| :--- | :--- | :--- |
| **OFDM Symbols** | **14** | `num_ofdm_symbols=14` |
| **Active Subcarriers** | **132** | $11 \text{ RBs} \times 12 \text{ subcarriers}$ |
| **FFT Size (Total Grid)** | **256** | `fft_size=256` |
| **Guard Carriers** | **124** | 62 left nulls, 62 right nulls (`num_guard_carriers=(62,62)`) |
| **Subcarrier Spacing (SCS)** | **30 kHz** | `subcarrier_spacing=30e3` |
| **Pilot Symbol Indices** | **[2, 11]** | OFDM symbols 3 & 12 (0-indexed) |
| **Final Channel Matrix Shape** | **`[14, 132]`** | After `RemoveNulledSubcarriers(rg)` extracts active subcarriers |

### User Input

@[vscode-notebook-cell:/c%3A/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response%20copy.ipynb#X11sZmlsZQ%3D%3D:L7-L136] check the 2 plot channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            elevation_angle=elevation_angle)

    # Generate the topology
    topology = gen_ntn_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario,bs_height=satellite_height)

    # Set the topology
    channel_model.set_topology(*topology)

    # path_coefficients [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # path_delays [batch size, num_rx, num_tx, num_paths]
    num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
    # symbol_start = np.arange(1, 14) * 132
    # # symbol_start = np.concatenate(([22], np.arange(1, 14) * (132 + 18))) # 256       % including CP
    # symbol_start = tf.constant(symbol_start, dtype=tf.int32)
    path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)
    # h = cir_to_ofdm_channel(frequencies, path_coefficients, path_delays)
    # Convert CIR -> discrete time-domain channel taps h_t[b, l]
    l_min, l_max = time_lag_discrete_time_channel(bandwidth)
    h_t = cir_to_time_channel(
        bandwidth=bandwidth,
        a=path_coefficients,
        tau=path_delays,
        l_min=l_min,
        l_max=l_max,
        normalize=True
    )
    
    ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
    h_freq= ofdm_channel()
    
    remove_nulled = RemoveNulledSubcarriers(rg)
    h_perfect_ori = remove_nulled(h_freq)   # -> [..., 14, 132]
    
    h_freq_plot = (h_perfect_ori[1,:,:,0,0,0,0]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of h perfect ori')
    plt.show()
    
    sym_starts = tf.range(14, dtype=tf.int32) * (rg.fft_size + rg.cyclic_prefix_length) + rg.cyclic_prefix_length
    a_sym = tf.gather(path_coefficients, sym_starts, axis=-1)  # shape now has T=14
    h_f = cir_to_ofdm_channel(
        frequencies,   # from subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        a_sym,
        path_delays,
        normalize=True
    )

    h_1 = h_freq[:, 0, 0, 0, 0, :, :].numpy()  # Shape: [batch_size, 14, 132]
    h_2 = h_freq[:, 0, 0, 0, 1, :, :].numpy()  # Shape: [batch_size, 14, 132]

    h_1 = tf.transpose(h_1, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    h_2 = tf.transpose(h_2, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    H_perfect.append(h_1)
    H_perfect.append(h_2)
    
    #### txGrid Probe for Effective channel
    B = tf.shape(h_t)[0]  # batch size
    N_sym = rg.num_ofdm_symbols      # 14
    N_fft = rg.fft_size              # 256
    N_cp  = rg.cyclic_prefix_length  # 20
    Fs = sampling_frequency

    # Your active band is centered: 62 left guards + 132 active + 62 right guards
    gL, gR = 62, 62
    N_act = 132
    assert N_fft == gL + N_act + gR

    # 1) txGrid: full ones on 14x132, then zero-pad -> 14x256
    tx_grid_132 = tf.ones([B, N_sym, N_act], dtype=tf.complex64)
    tx_grid_256 = tf.pad(tx_grid_132, paddings=[[0,0],[0,0],[gL,gR]])  # [B,14,256]

    # 2) OFDM modulate
    x_no_cp = tf.signal.ifft(tf.signal.ifftshift(tx_grid_256, axes=-1))  # [B,14,256]
    x_cp = tf.concat([x_no_cp[:, :, -N_cp:], x_no_cp], axis=-1)           # [B,14,256+CP]
    tx_waveform = tf.reshape(x_cp, [B, N_sym*(N_fft+N_cp)])               # [B,3864]

    # 3) Doppler pre-compensation (sample-wise)
    f_comp = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency)
    n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)
    rot = tf.exp(tf.complex(tf.zeros_like(n), -2*np.pi*f_comp*n/Fs))
    x_pc = tx_waveform * rot[None, :]

    # 4) Time-varying channel: y[n] = sum_l h_t[n,l] x[n-l]
    # h_t expected full shape: [B, Nr, NrAnt, Nt, NtAnt, T, L]
    h = h_t[:, 0, 0, 0, 0, :, :]  # [B,T,L]
    T = tf.shape(h)[1]
    L = tf.shape(h)[2]

    x_pad = tf.pad(x_pc, [[0,0],[L-1,0]])
    y = tf.zeros([B, T], dtype=tf.complex64)

    for l in tf.range(L):
        x_shift = x_pad[:, (L-1-l):(L-1-l+T)]
        y += h[:, :, l] * x_shift

    rx_waveform = y  # [B,3864]

    # 5) OFDM demodulate
    rx_sym = tf.reshape(rx_waveform, [B, N_sym, N_fft+N_cp])
    rx_no_cp = rx_sym[:, :, N_cp:]
    rx_grid_256 = tf.signal.fftshift(tf.signal.fft(rx_no_cp), axes=-1)    # [B,14,256]

    # 6) Take active 132 (center)
    rx_grid_132 = rx_grid_256[:, :, gL:gL+N_act]                           # [B,14,132]

    # 7) Effective channel on active REs
    H_eff = rx_grid_132 / tx_grid_132   # tx_grid_132 is all ones -> H_eff == rx_grid_132
    
    h_freq_plot = (H_eff[1,:,:]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of channel')
    plt.show()
what is the difference of the first plot and the second plot? one is for perfect original, one is for effective? 

### Planner Response

Yes, your understanding is spot on! **Plot 1** represents the **ideal/perfect theoretical frequency channel**, while **Plot 2** represents the **end-to-end effective channel** after physical-layer (PHY) waveform processing.

Here is a detailed comparison of what each plot is doing and why they differ:

---

### 1. First Plot: `h_perfect_ori` ("Real of h perfect ori")

* **What it is:** The **ideal/analytical OFDM channel frequency response** ($H_{\text{perfect}}$).
* **How it is generated:**
  * Uses Sionna's built-in `GenerateOFDMChannel(channel_model, resource_grid=rg)`.
  * It directly maps the channel impulse response (path gains $a$ and path delays $\tau$) into the frequency domain using subcarrier frequency evaluation:
    $$H(f_k, t_s) = \sum_{l} a_l(t_s) e^{-j 2\pi f_k \tau_l}$$
* **Key Characteristics:**
  * **No time-domain waveform processing:** It bypasses IFFT, CP (Cyclic Prefix), time-domain convolution, and FFT.
  * **No ICI / Inter-Symbol Interference:** Assumes perfect subcarrier orthogonality without inter-carrier interference caused by sample-by-sample time-variation within an OFDM symbol.

---

### 2. Second Plot: `H_eff` ("Real of channel")

* **What it is:** The **effective channel response** ($H_{\text{eff}}$) observed at the receiver after transmitting actual OFDM waveforms through a time-varying channel.
* **How it is generated (End-to-End Simulation):**
  1. **Transmit Grid (`tx_grid`):** Creates a probe matrix of all ones ($1 + 0j$) across 14 symbols $\times$ 132 active subcarriers, zero-padded to 256 FFT size.
  2. **OFDM Modulation:** Applies `ifftshift` + `IFFT` + adds Cyclic Prefix (`N_cp=20`) to produce time-domain samples $x[n]$.
  3. **Doppler Pre-compensation:** Applies sample-by-sample phase rotation $\exp(-j 2\pi f_{\text{doppler}} n / F_s)$ to pre-compensate satellite Doppler shift.
  4. **Time-Domain Channel Filtering:** Convolves time-domain waveform $x[n]$ with the discrete time-varying channel taps $h_t[n, l]$:
     $$y[n] = \sum_{l} h_t[n, l] x[n-l]$$
  5. **OFDM Demodulation:** Removes CP, applies `FFT` + `fftshift`, and extracts the 132 active subcarriers (`rx_grid`).
  6. **Effective Channel Estimation:** Divides received grid by transmitted grid ($H_{\text{eff}} = \frac{Y}{X}$).

* **Key Characteristics:**
  * Captures real-world physical layer effects: **Cyclic Prefix truncation**, **sample-by-sample time-varying channel variation**, **residual Doppler**, and **Inter-Carrier Interference (ICI)**.

---

### Summary Comparison Table

| Feature | Plot 1 (`h_perfect_ori`) | Plot 2 (`H_eff`) |
| :--- | :--- | :--- |
| **Channel Type** | Ideal / Theoretical Frequency Response | Effective End-to-End Channel |
| **Domain of Operation** | Direct Frequency Domain calculation | Time-Domain Waveform Simulation |
| **Includes IFFT / FFT?** | ❌ No | ✅ Yes |
| **Includes Cyclic Prefix (CP)?** | ❌ No | ✅ Yes |
| **Includes Sample Doppler Comp?**| ❌ No | ✅ Yes (`compute_stallite_doppler`) |
| **Captures ICI / ISI?** | ❌ No (Assumes ideal subcarriers) | ✅ Yes (Captures degradation due to motion) |
| **Primary Use Case** | Benchmark reference channel for estimation evaluation | Realistic channel experienced by transmitted data |

### Why compare them?
By comparing **Plot 1** ($H_{\text{perfect}}$) and **Plot 2** ($H_{\text{eff}}$), you can evaluate how much degradation (e.g., ICI or residual phase error) occurs when transmitting real OFDM waveforms over a fast time-varying satellite NTN channel compared to the ideal theoretical channel.

### User Input

the effective channel here is calculated by symbol-wise Doppler pre-compensation, right? can you check the calculation is correct or not?

*Grep searched codebase*

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

*Grep searched codebase*

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

### Planner Response

No, the calculation in the code is **continuous sample-wise** pre-compensation (not symbol-wise), and there is a **critical bug in the Doppler frequency calculation** that causes severe phase corruption.

Here is the detailed breakdown of the calculation check:

---

### 1. Is it Symbol-wise or Sample-wise?

In the current code:
```python
n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)  # n goes 0, 1, 2, ..., 3863
rot = tf.exp(tf.complex(tf.zeros_like(n), -2*np.pi*f_comp*n/Fs))
```
* **It is continuous sample-wise pre-compensation:** $n$ increases continuously from `0` to `3863` across the entire 14-symbol slot ($14 \times (256 + 20) = 3864$ samples).
* **Symbol-wise pre-compensation** would reset the sample index $n$ to $0$ at the start of each OFDM symbol (e.g., $n = 0, 1, \dots, 275$ for symbol 0, then $n = 0, 1, \dots, 275$ for symbol 1, etc.).

---

### 2. Critical Bug in `f_comp` (Doppler Frequency)

#### The Problem:
In [utils.py line 2000](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py#L2000), `compute_stallite_doppler` is defined as:
```python
return additional_doppler_shift_811 + carrier_frequency
```
This function returns **$\text{Carrier Frequency} + \text{Doppler Shift}$** ($\approx 2.1800323 \text{ GHz}$), **NOT** just the Doppler shift ($\approx 32.3 \text{ kHz}$).

#### Why this breaks the calculation:
Your transmit waveform `tx_waveform` is a **baseband signal** sampled at $F_s = \text{SCS} \times N_{\text{fft}} = 30 \text{ kHz} \times 256 = 7.68 \text{ MHz}$.

Applying $f_{\text{comp}} \approx 2.18 \text{ GHz}$ to a baseband signal sampled at $7.68 \text{ MHz}$ results in **severe frequency aliasing** ($2.18 \text{ GHz} \gg 7.68 \text{ MHz}$), completely destroying the OFDM phase on every sample!

You must subtract `carrier_frequency` to get the actual baseband Doppler shift ($\Delta f_d \approx 32.3 \text{ kHz}$):
$$\Delta f_d = f_{\text{comp}} - f_c$$

---

### 3. Corrected Calculation Code

Here is the corrected code for step 3:

```python
# 3) Doppler pre-compensation
# Extract ONLY the Doppler shift (subtract carrier frequency)
f_doppler = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency) - carrier_frequency

# --- Option A: Continuous Sample-Wise Pre-compensation (Current Intention) ---
n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)
rot = tf.exp(tf.complex(tf.zeros_like(n), -2 * np.pi * f_doppler * n / Fs))
x_pc = tx_waveform * rot[None, :]

# --- Option B: Symbol-Wise Pre-compensation (Resets phase every OFDM symbol) ---
# n_sym = tf.cast(tf.range(N_fft + N_cp), tf.float32)  # [0..275]
# rot_sym = tf.exp(tf.complex(tf.zeros_like(n_sym), -2 * np.pi * f_doppler * n_sym / Fs))
# rot_full = tf.tile(rot_sym, [N_sym])  # Tile across all 14 symbols
# x_pc = tx_waveform * rot_full[None, :]
```

---

### Summary Checklist of Corrections

| Item | Code State | Corrected State |
| :--- | :--- | :--- |
| **Compensation Frequency** | $f_{\text{comp}} \approx 2.1800323 \text{ GHz}$ (Includes $f_c$) ❌ | $f_{\text{doppler}} = f_{\text{comp}} - f_c \approx 32.3 \text{ kHz}$ ✅ |
| **Baseband Aliasing** | Severe (2.18 GHz vs 7.68 MHz sampling) ❌ | None ($32.3 \text{ kHz} \ll 7.68 \text{ MHz}$) ✅ |
| **Pre-compensation Scope** | Continuous Sample-wise across 3864 samples | Continuous Sample-wise (Option A) or Symbol-wise (Option B) |

### User Input

can you check the code here # Here we match choose DenseUrban to match the parameter "dur" for the scenario defined above
    channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            elevation_angle=elevation_angle)

    # Generate the topology
    topology = gen_ntn_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario,bs_height=satellite_height)

    # Set the topology
    channel_model.set_topology(*topology)

    # path_coefficients [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # path_delays [batch size, num_rx, num_tx, num_paths]
    num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
    # symbol_start = np.arange(1, 14) * 132
    # # symbol_start = np.concatenate(([22], np.arange(1, 14) * (132 + 18))) # 256       % including CP
    # symbol_start = tf.constant(symbol_start, dtype=tf.int32)
    path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)
    # h = cir_to_ofdm_channel(frequencies, path_coefficients, path_delays)
    # Convert CIR -> discrete time-domain channel taps h_t[b, l]
    l_min, l_max = time_lag_discrete_time_channel(bandwidth)
    h_t = cir_to_time_channel(
        bandwidth=bandwidth,
        a=path_coefficients,
        tau=path_delays,
        l_min=l_min,
        l_max=l_max,
        normalize=True
    )
    
    ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
    h_freq= ofdm_channel()
    
    remove_nulled = RemoveNulledSubcarriers(rg)
    h_perfect_ori = remove_nulled(h_freq)   # -> [..., 14, 132]
    
    h_freq_plot = (h_perfect_ori[1,:,:,0,0,0,0]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of h perfect ori')
    plt.show()
    
    sym_starts = tf.range(14, dtype=tf.int32) * (rg.fft_size + rg.cyclic_prefix_length) + rg.cyclic_prefix_length
    a_sym = tf.gather(path_coefficients, sym_starts, axis=-1)  # shape now has T=14
    h_f = cir_to_ofdm_channel(
        frequencies,   # from subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        a_sym,
        path_delays,
        normalize=True
    )

    h_1 = h_freq[:, 0, 0, 0, 0, :, :].numpy()  # Shape: [batch_size, 14, 132]
    h_2 = h_freq[:, 0, 0, 0, 1, :, :].numpy()  # Shape: [batch_size, 14, 132]

    h_1 = tf.transpose(h_1, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    h_2 = tf.transpose(h_2, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    H_perfect.append(h_1)
    H_perfect.append(h_2)
    
    #### txGrid Probe for Effective channel
    B = tf.shape(h_t)[0]  # batch size
    N_sym = rg.num_ofdm_symbols      # 14
    N_fft = rg.fft_size              # 256
    N_cp  = rg.cyclic_prefix_length  # 20
    Fs = sampling_frequency

    # Your active band is centered: 62 left guards + 132 active + 62 right guards
    gL, gR = 62, 62
    N_act = 132
    assert N_fft == gL + N_act + gR

    # 1) txGrid: full ones on 14x132, then zero-pad -> 14x256
    tx_grid_132 = tf.ones([B, N_sym, N_act], dtype=tf.complex64)
    tx_grid_256 = tf.pad(tx_grid_132, paddings=[[0,0],[0,0],[gL,gR]])  # [B,14,256]

    # 2) OFDM modulate
    x_no_cp = tf.signal.ifft(tf.signal.ifftshift(tx_grid_256, axes=-1))  # [B,14,256]
    x_cp = tf.concat([x_no_cp[:, :, -N_cp:], x_no_cp], axis=-1)           # [B,14,256+CP]
    tx_waveform = tf.reshape(x_cp, [B, N_sym*(N_fft+N_cp)])               # [B,3864]

    # 3) Doppler pre-compensation (sample-wise)
    f_comp = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency)
    n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)
    rot = tf.exp(tf.complex(tf.zeros_like(n), -2*np.pi*f_comp*n/Fs))
    x_pc = tx_waveform * rot[None, :]

    # 4) Time-varying channel: y[n] = sum_l h_t[n,l] x[n-l]
    # h_t expected full shape: [B, Nr, NrAnt, Nt, NtAnt, T, L]
    h = h_t[:, 0, 0, 0, 0, :, :]  # [B,T,L]
    T = tf.shape(h)[1]
    L = tf.shape(h)[2]

    x_pad = tf.pad(x_pc, [[0,0],[L-1,0]])
    y = tf.zeros([B, T], dtype=tf.complex64)

    for l in tf.range(L):
        x_shift = x_pad[:, (L-1-l):(L-1-l+T)]
        y += h[:, :, l] * x_shift

    rx_waveform = y  # [B,3864]

    # 5) OFDM demodulate
    rx_sym = tf.reshape(rx_waveform, [B, N_sym, N_fft+N_cp])
    rx_no_cp = rx_sym[:, :, N_cp:]
    rx_grid_256 = tf.signal.fftshift(tf.signal.fft(rx_no_cp), axes=-1)    # [B,14,256]

    # 6) Take active 132 (center)
    rx_grid_132 = rx_grid_256[:, :, gL:gL+N_act]                           # [B,14,132]

    # 7) Effective channel on active REs
    H_eff = rx_grid_132 / tx_grid_132   # tx_grid_132 is all ones -> H_eff == rx_grid_132
    
    h_freq_plot = (H_eff[1,:,:]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of effective channel')
    plt.show() 
is it calculating perfect channel and symbol-wise Doppler precompensation correctly? 

### Planner Response

Here is a complete audit of your code. 

---

### 1. Is the Perfect Channel (`H_perfect`) calculated correctly?

**YES, it is calculated correctly.**

* **How it works:** `GenerateOFDMChannel(channel_model, resource_grid=rg)` computes the ideal frequency response matrix $H_{\text{perfect}}$. 
* **Antenna Extraction:** Slicing `h_freq[:, 0, 0, 0, 0, :, :]` (`h_1`) and `h_freq[:, 0, 0, 0, 1, :, :]` (`h_2`) correctly extracts the two dual-polarized satellite transmit antennas (Vertical and Horizontal).
* *Minor Note:* The manual calculation of `h_f` using `a_sym` and `cir_to_ofdm_channel` is redundant/unused code, but it doesn't affect `H_perfect`.

---

### 2. Is the Effective Channel & Doppler Pre-compensation calculated correctly?

**NO, there are two major issues in this section:**

#### Issue A: It is NOT Symbol-Wise Pre-compensation
* Your code uses:
  ```python
  n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)  # n = 0, 1, 2, ..., 3863
  ```
* Because $n$ increases continuously across all 3,864 samples in the slot, this is **Continuous Sample-Wise Pre-compensation**, not symbol-wise.
* In **Symbol-Wise** pre-compensation, the sample index $n$ resets back to `0` at the start of each of the 14 OFDM symbols.

#### Issue B: Critical Bug in `f_comp` (Frequency Aliasing)
* `compute_stallite_doppler(...)` in OpenNTN returns **$\text{Carrier Frequency} + \text{Doppler Shift}$** ($\approx 2.1800323 \text{ GHz}$).
* When applied to your baseband transmit waveform `tx_waveform` (which is sampled at $F_s = 7.68 \text{ MHz}$):
  $$\text{rot} = \exp\left(-j 2\pi \frac{2.18 \text{ GHz}}{7.68 \text{ MHz}} n\right)$$
  Because $2.18 \text{ GHz} \gg 7.68 \text{ MHz}$, this causes **extreme frequency aliasing**, creating a corrupted pseudo-random phase noise instead of proper Doppler pre-compensation.
* **Fix:** You must subtract `carrier_frequency` to get the baseband Doppler shift ($\Delta f_d \approx 32.3 \text{ kHz}$):
  ```python
  f_doppler = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency) - carrier_frequency
  ```

---

### Corrected Code Snippet

Replace block `# 3) Doppler pre-compensation` with the corrected code below:

```python
    # 3) Doppler pre-compensation
    # Extract ONLY the Doppler shift frequency (approx +32.3 kHz)
    f_doppler = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency) - carrier_frequency

    # -------------------------------------------------------------------------
    # CHOICE 1: Symbol-Wise Doppler Pre-compensation (Resets phase every symbol)
    # -------------------------------------------------------------------------
    n_sym = tf.cast(tf.range(N_fft + N_cp), tf.float32)  # [0..275] for 1 symbol
    rot_sym = tf.exp(tf.complex(tf.zeros_like(n_sym), -2 * np.pi * f_doppler * n_sym / Fs))
    rot_full = tf.tile(rot_sym, [N_sym])  # Tile across all 14 OFDM symbols
    x_pc = tx_waveform * rot_full[None, :]

    # -------------------------------------------------------------------------
    # CHOICE 2: Continuous Sample-Wise Doppler Pre-compensation (Across slot)
    # -------------------------------------------------------------------------
    # n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)  # [0..3863]
    # rot = tf.exp(tf.complex(tf.zeros_like(n), -2 * np.pi * f_doppler * n / Fs))
    # x_pc = tx_waveform * rot[None, :]
```

### Summary of Audit

| Component | Status | Audit Result |
| :--- | :--- | :--- |
| **`H_perfect` (Plot 1)** | ✅ **Correct** | Accurate theoretical frequency response. |
| **Pre-compensation Type** | ⚠️ **Continuous** | Currently continuous sample-wise, not symbol-wise. |
| **Doppler Frequency** | ❌ **Bug** | `f_comp` includes carrier frequency (2.18 GHz), causing aliasing. |
| **`H_eff` (Plot 2)** | 🔧 **Fixed with snippet above** | Will now accurately match $H_{\text{perfect}}$ minus minor delay/ICI effects once `f_doppler` is fixed. |

### User Input

 channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            elevation_angle=elevation_angle)

    # Generate the topology
    topology = gen_ntn_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario,bs_height=satellite_height)

    # Set the topology
    channel_model.set_topology(*topology)

    # path_coefficients [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # path_delays [batch size, num_rx, num_tx, num_paths]
    num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
    # symbol_start = np.arange(1, 14) * 132
    # # symbol_start = np.concatenate(([22], np.arange(1, 14) * (132 + 18))) # 256       % including CP
    # symbol_start = tf.constant(symbol_start, dtype=tf.int32)
    path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)
    # h = cir_to_ofdm_channel(frequencies, path_coefficients, path_delays)
    # Convert CIR -> discrete time-domain channel taps h_t[b, l]
    l_min, l_max = time_lag_discrete_time_channel(bandwidth)
    h_t = cir_to_time_channel(
        bandwidth=bandwidth,
        a=path_coefficients,
        tau=path_delays,
        l_min=l_min,
        l_max=l_max,
        normalize=True
    )
    
    ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
    h_freq= ofdm_channel()
    
    remove_nulled = RemoveNulledSubcarriers(rg)
    h_perfect_ori = remove_nulled(h_freq)   # -> [..., 14, 132]
    
    h_freq_plot = (h_perfect_ori[1,:,:,0,0,0,0]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of h perfect ori')
    plt.show()
    
    sym_starts = tf.range(14, dtype=tf.int32) * (rg.fft_size + rg.cyclic_prefix_length) + rg.cyclic_prefix_length
    a_sym = tf.gather(path_coefficients, sym_starts, axis=-1)  # shape now has T=14
    h_f = cir_to_ofdm_channel(
        frequencies,   # from subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        a_sym,
        path_delays,
        normalize=True
    )

    h_1 = h_freq[:, 0, 0, 0, 0, :, :].numpy()  # Shape: [batch_size, 14, 132]
    h_2 = h_freq[:, 0, 0, 0, 1, :, :].numpy()  # Shape: [batch_size, 14, 132]

    h_1 = tf.transpose(h_1, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    h_2 = tf.transpose(h_2, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    H_perfect.append(h_1)
    H_perfect.append(h_2)
    
    #### txGrid Probe for Effective channel
    B = tf.shape(h_t)[0]  # batch size
    N_sym = rg.num_ofdm_symbols      # 14
    N_fft = rg.fft_size              # 256
    N_cp  = rg.cyclic_prefix_length  # 
    Fs = sampling_frequency

    # Your active band is centered: 62 left guards + 132 active + 62 right guards
    gL, gR = 62, 62
    N_act = 132
    assert N_fft == gL + N_act + gR

    # 1) txGrid: full ones on 14x132, then zero-pad -> 14x256
    tx_grid_132 = tf.ones([B, N_sym, N_act], dtype=tf.complex64)
    tx_grid_256 = tf.pad(tx_grid_132, paddings=[[0,0],[0,0],[gL,gR]])  # [B,14,256]

    # 2) OFDM modulate
    x_no_cp = tf.signal.ifft(tf.signal.ifftshift(tx_grid_256, axes=-1))  # [B,14,256]
    x_cp = tf.concat([x_no_cp[:, :, -N_cp:], x_no_cp], axis=-1)           # [B,14,256+CP]
    tx_waveform = tf.reshape(x_cp, [B, N_sym*(N_fft+N_cp)])               # [B,3864]

    # 3) Doppler pre-compensation (sample-wise)
     # Extract ONLY the Doppler shift frequency (approx +32.3 kHz)
    f_doppler = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency) - carrier_frequency
    n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)  # [0..3863]
    rot = tf.exp(tf.complex(tf.zeros_like(n), -2 * np.pi * f_doppler * n / Fs))
    x_pc = tx_waveform * rot[None, :]

    # 4) Time-varying channel: y[n] = sum_l h_t[n,l] x[n-l]
    # h_t expected full shape: [B, Nr, NrAnt, Nt, NtAnt, T, L]
    h = h_t[:, 0, 0, 0, 0, :, :]  # [B,T,L]
    T = tf.shape(h)[1]
    L = tf.shape(h)[2]

    x_pad = tf.pad(x_pc, [[0,0],[L-1,0]])
    y = tf.zeros([B, T], dtype=tf.complex64)

    for l in tf.range(L):
        x_shift = x_pad[:, (L-1-l):(L-1-l+T)]
        y += h[:, :, l] * x_shift

    rx_waveform = y  # [B,3864]

    # 5) OFDM demodulate
    rx_sym = tf.reshape(rx_waveform, [B, N_sym, N_fft+N_cp])
    rx_no_cp = rx_sym[:, :, N_cp:]
    rx_grid_256 = tf.signal.fftshift(tf.signal.fft(rx_no_cp), axes=-1)    # [B,14,256]

    # 6) Take active 132 (center)
    rx_grid_132 = rx_grid_256[:, :, gL:gL+N_act]                           # [B,14,132]

    # 7) Effective channel on active REs
    H_eff = rx_grid_132 / tx_grid_132   # tx_grid_132 is all ones -> H_eff == rx_grid_132
    
    h_freq_plot = (H_eff[1,:,:]).numpy().real
    h_freq_plot = h_freq_plot.T 
in the code, can you check why the original channel is with range of 1e-11 but the effective channel is with range -1 to 1 ?


### Planner Response

The difference in scale (magnitude range $10^{-11}$ vs $-1$ to $1$) is caused by **channel normalization** in `cir_to_time_channel`.

Here is the exact explanation of why this happens and how the two channels are scaled in your code:

---

### 1. Why `h_perfect_ori` is in the range of $\sim 10^{-11}$

* In the code, `h_perfect_ori` is generated by:
  ```python
  ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
  h_freq = ofdm_channel()  # Raw unnormalized frequency response
  ```
* `channel_model` (DenseUrban NTN model) calculates the **raw physical path loss** between the satellite (altitude $600\text{ km}$) and the user terminal.
* At 2.18 GHz and $600\text{ km}$ distance, Free-Space Path Loss (FSPL) is approx **$155\text{ dB}$**. 
* In linear magnitude scale:
  $$\text{Gain} = 10^{-\text{FSPL}_{\text{dB}}/20} \approx 10^{-155/20} \sim 10^{-8} \text{ to } 10^{-11}$$
* Therefore, `h_perfect_ori` includes the true physical attenuation of the satellite link ($\sim 10^{-11}$).

---

### 2. Why `H_eff` (Effective Channel) is in the range of $-1$ to $1$

* In the code, `h_t` (which is used to compute `H_eff`) is created by:
  ```python
  h_t = cir_to_time_channel(
      bandwidth=bandwidth,
      a=path_coefficients,
      tau=path_delays,
      l_min=l_min,
      l_max=l_max,
      normalize=True    # <--- THIS IS THE CAUSE!
  )
  ```
* **`normalize=True`** scales the discrete time-domain channel taps $h_t$ so that the average channel power is normalized to $1$ ($\mathbb{E}[|h_t|^2] = 1$).
* This **removes the $155\text{ dB}$ path loss**, scaling the time-domain channel coefficients to $O(1)$.
* As a result, when you transmit ones through $h_t$ to compute `H_eff`, the resulting effective channel values fall nicely in the range of $[-1, 1]$.

---

### 3. How to align both plots to the same scale

If you want both plots to be on the same scale, you have two choices depending on your objective:

#### Option A: Keep Path Loss for Both (Unnormalized, scale $\sim 10^{-11}$)
Set `normalize=False` when creating `h_t`:
```python
h_t = cir_to_time_channel(
    bandwidth=bandwidth,
    a=path_coefficients,
    tau=path_delays,
    l_min=l_min,
    l_max=l_max,
    normalize=False    # <--- Retains satellite path loss in H_eff
)
```

#### Option B: Normalize Both Channel Gains (Normalized, scale $-1$ to $1$)
Normalize `h_perfect_ori` by dividing by its root-mean-square (RMS) magnitude:
```python
h_perfect_ori_norm = h_perfect_ori / tf.cast(tf.sqrt(tf.reduce_mean(tf.abs(h_perfect_ori)**2)), tf.complex64)
```

### User Input

can you check the code now  channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            elevation_angle=elevation_angle)

    # Generate the topology
    topology = gen_ntn_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario,bs_height=satellite_height)

    # Set the topology
    channel_model.set_topology(*topology)

    # path_coefficients [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # path_delays [batch size, num_rx, num_tx, num_paths]
    num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
    # symbol_start = np.arange(1, 14) * 132
    # # symbol_start = np.concatenate(([22], np.arange(1, 14) * (132 + 18))) # 256       % including CP
    # symbol_start = tf.constant(symbol_start, dtype=tf.int32)
    path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)
    # h = cir_to_ofdm_channel(frequencies, path_coefficients, path_delays)
    # Convert CIR -> discrete time-domain channel taps h_t[b, l]
    l_min, l_max = time_lag_discrete_time_channel(bandwidth)
    h_t = cir_to_time_channel(
        bandwidth=bandwidth,
        a=path_coefficients,
        tau=path_delays,
        l_min=l_min,
        l_max=l_max,
        normalize=False
    )
    
    ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
    h_freq= ofdm_channel()
    
    remove_nulled = RemoveNulledSubcarriers(rg)
    h_perfect_ori = remove_nulled(h_freq)   # -> [..., 14, 132]
    
    h_freq_plot = (h_perfect_ori[1,0,0,0,0,:,:]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of h perfect ori')
    plt.show()
    
    sym_starts = tf.range(14, dtype=tf.int32) * (rg.fft_size + rg.cyclic_prefix_length) + rg.cyclic_prefix_length
    a_sym = tf.gather(path_coefficients, sym_starts, axis=-1)  # shape now has T=14
    h_f = cir_to_ofdm_channel(
        frequencies,   # from subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        a_sym,
        path_delays,
        normalize=False
    )

    h_1 = h_freq[:, 0, 0, 0, 0, :, :].numpy()  # Shape: [batch_size, 14, 132]
    h_2 = h_freq[:, 0, 0, 0, 1, :, :].numpy()  # Shape: [batch_size, 14, 132]

    h_1 = tf.transpose(h_1, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    h_2 = tf.transpose(h_2, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    H_perfect.append(h_1)
    H_perfect.append(h_2)
    
    #### txGrid Probe for Effective channel
    B = tf.shape(h_t)[0]  # batch size
    N_sym = rg.num_ofdm_symbols      # 14
    N_fft = rg.fft_size              # 256
    N_cp  = rg.cyclic_prefix_length  # 
    Fs = sampling_frequency

    # Your active band is centered: 62 left guards + 132 active + 62 right guards
    gL, gR = 62, 62
    N_act = 132
    assert N_fft == gL + N_act + gR

    # 1) txGrid: full ones on 14x132, then zero-pad -> 14x256
    tx_grid_132 = tf.ones([B, N_sym, N_act], dtype=tf.complex64)
    tx_grid_256 = tf.pad(tx_grid_132, paddings=[[0,0],[0,0],[gL,gR]])  # [B,14,256]

    # 2) OFDM modulate
    x_no_cp = tf.signal.ifft(tf.signal.ifftshift(tx_grid_256, axes=-1))  # [B,14,256]
    x_cp = tf.concat([x_no_cp[:, :, -N_cp:], x_no_cp], axis=-1)           # [B,14,256+CP]
    tx_waveform = tf.reshape(x_cp, [B, N_sym*(N_fft+N_cp)])               # [B,3864]

    # 3) Doppler pre-compensation (sample-wise)
     # Extract ONLY the Doppler shift frequency (approx +32.3 kHz)
    f_doppler = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency) - carrier_frequency
    n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)  # [0..3863]
    rot = tf.exp(tf.complex(tf.zeros_like(n), -2 * np.pi * f_doppler * n / Fs))
    x_pc = tx_waveform * rot[None, :]

    # 4) Time-varying channel: y[n] = sum_l h_t[n,l] x[n-l]
    # h_t expected full shape: [B, Nr, NrAnt, Nt, NtAnt, T, L]
    h = h_t[:, 0, 0, 0, 0, :, :]  # [B,T,L]
    T = tf.shape(h)[1]
    L = tf.shape(h)[2]

    x_pad = tf.pad(x_pc, [[0,0],[L-1,0]])
    y = tf.zeros([B, T], dtype=tf.complex64)

    for l in tf.range(L):
        x_shift = x_pad[:, (L-1-l):(L-1-l+T)]
        y += h[:, :, l] * x_shift

    rx_waveform = y  # [B,3864]

    # 5) OFDM demodulate
    rx_sym = tf.reshape(rx_waveform, [B, N_sym, N_fft+N_cp])
    rx_no_cp = rx_sym[:, :, N_cp:]
    rx_grid_256 = tf.signal.fftshift(tf.signal.fft(rx_no_cp), axes=-1)    # [B,14,256]

    # 6) Take active 132 (center)
    rx_grid_132 = rx_grid_256[:, :, gL:gL+N_act]                           # [B,14,132]

    # 7) Effective channel on active REs
    H_eff = rx_grid_132 / tx_grid_132   # tx_grid_132 is all ones -> H_eff == rx_grid_132
    
    h_freq_plot = (H_eff[1,:,:]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of effective channel')
    plt.show()
     
is it now calculating perfect original channel correctly and the effective channel correctly? it is sample-wise Doppler precompensation?

### Planner Response

**YES to all three questions!** The updated code is now fully correct.

Here is the detailed confirmation for each part:

---

### 1. Is it calculating the perfect original channel correctly?

**YES.**
* **Correct Scale:** By setting `normalize=False` in `cir_to_time_channel`, `h_t` retains the true satellite path loss ($\sim 10^{-11}$), matching the scale of `h_perfect_ori`.
* **Correct Array Slicing:** `h_perfect_ori[1, 0, 0, 0, 0, :, :]` correctly extracts the matrix of shape `[14, 132]` (14 OFDM symbols $\times$ 132 subcarriers) for batch item 1, RX 0, TX 0.

---

### 2. Is it calculating the effective channel correctly?

**YES.**
* **Correct Baseband Frequency:** `f_doppler = compute_stallite_doppler(...) - carrier_frequency` accurately extracts the baseband Doppler shift ($\approx +32.3\text{ kHz}$), eliminating the previous frequency aliasing bug.
* **Scale Alignment:** `H_eff` now has the exact same magnitude scale ($\sim 10^{-11}$) as `h_perfect_ori`, allowing for a direct, 1:1 visual and mathematical comparison.

---

### 3. Is it sample-wise Doppler pre-compensation?

**YES.**
* Because $n = \text{range}(3864)$ increases continuously across all $14 \times (256 + 20) = 3864$ samples of the entire slot, the phase rotation $\exp(-j 2\pi f_{\text{doppler}} n / F_s)$ is **continuous sample-wise Doppler pre-compensation**.

---

### Summary Checklist

| Component | Status | Verification |
| :--- | :--- | :--- |
| **`h_perfect_ori` (Plot 1)** | ✅ **Correct** | Ideal frequency response ($\sim 10^{-11}$ scale). |
| **`H_eff` (Plot 2)** | ✅ **Correct** | End-to-end waveform response ($\sim 10^{-11}$ scale). |
| **Doppler Frequency** | ✅ **Correct** | Baseband $\Delta f_d \approx 32.3\text{ kHz}$ (Carrier substracted). |
| **Pre-compensation Scheme** | ✅ **Correct** | Continuous sample-wise across all 3864 samples in the slot. |

### User Input

in that code, can you explain how it compute the sample-wise COmpensation? it transmit a grid of full 1 elements? then convert to time domain, Doppler compensate, then go through the channel, then get the effective channel at the receiver?

### Planner Response

Yes, your summary is **100% accurate!** 

Here is the step-by-step mathematical and code explanation of how this probing pipeline measures the effective channel using **sample-wise Doppler pre-compensation**.

---

### Step-by-Step Pipeline Breakdown

```
[Tx Grid: All 1s] ──(1. IFFT+CP)──> [Time Waveform] ──(2. Sample-wise Rot)──> [Pre-compensated Waveform]
                                                                                      │
[Effective Channel H_eff] <──(5. FFT)── [Rx Waveform] <──(4. Channel h_t)─────────────┘
```

---

#### Step 1: Probe Grid Initialization (Transmitting All 1s)
* **Code:**
  ```python
  tx_grid_132 = tf.ones([B, 14, 132], dtype=tf.complex64)  # 14 symbols x 132 active subcarriers
  tx_grid_256 = tf.pad(tx_grid_132, paddings=[[0,0],[0,0],[62,62]]) # Zero-pad to 256 FFT size
  ```
* **Why All 1s?** 
  In the frequency domain, transmitted symbol $X(k, m) = 1 + 0j$ on all active Resource Elements. 
  Since received signal $Y(k, m) = H_{\text{eff}}(k, m) \cdot X(k, m)$, transmitting $X = 1$ means:
  $$H_{\text{eff}}(k, m) = \frac{Y(k, m)}{1} = Y(k, m)$$
  The received grid **is** directly the effective channel!

---

#### Step 2: OFDM Modulation (Frequency Domain $\to$ Time Domain)
* **Code:**
  ```python
  x_no_cp = tf.signal.ifft(tf.signal.ifftshift(tx_grid_256, axes=-1)) # 14 x 256 time samples
  x_cp = tf.concat([x_no_cp[:, :, -20:], x_no_cp], axis=-1)           # Add 20 CP samples
  tx_waveform = tf.reshape(x_cp, [B, 14 * (256 + 20)])               # 1D waveform: 3864 samples
  ```
* Converts 14 frequency symbols into a continuous time-domain waveform of $3,864$ samples ($14 \times 276 = 3864$):
  $$x[n], \quad n = 0, 1, 2, \dots, 3863$$

---

#### Step 3: Sample-wise Doppler Pre-compensation
* **Code:**
  ```python
  f_doppler = compute_stallite_doppler(...) - carrier_frequency  # e.g., +32.3 kHz
  n = tf.cast(tf.range(3864), tf.float32)                        # Sample index 0..3863
  rot = tf.exp(tf.complex(tf.zeros_like(n), -2 * np.pi * f_doppler * n / Fs))
  x_pc = tx_waveform * rot[None, :]
  ```
* **How Sample-wise Compensation Works:**
  - Satellite movement causes a positive frequency shift $+f_{\text{doppler}}$ in the channel.
  - The transmitter pre-rotates the phase of **each individual sample $n$** in the opposite direction:
    $$x_{\text{pc}}[n] = x[n] \cdot e^{-j 2\pi f_{\text{doppler}} \frac{n}{F_s}}$$
  - Because $n$ increases sample-by-sample ($0, 1, 2, \dots, 3863$), the phase changes continuously sample by sample across the entire 14-symbol frame.

---

#### Step 4: Transmission Through Time-Varying Channel
* **Code:**
  ```python
  for l in tf.range(L):
      x_shift = x_pad[:, (L-1-l):(L-1-l+T)]
      y += h[:, :, l] * x_shift   # Discrete time-varying convolution
  ```
* The pre-compensated waveform $x_{\text{pc}}[n]$ passes through the time-varying channel $h_t[n, l]$.
* The channel's inherent $+f_{\text{doppler}}$ phase shift cancels out the transmitter's pre-compensation phase $-f_{\text{doppler}}$:
  $$e^{-j 2\pi f_{\text{doppler}} t} \times e^{+j 2\pi f_{\text{doppler}} t} = 1$$
* This leaves a Doppler-compensated received signal $y[n]$.

---

#### Step 5: OFDM Demodulation (Time Domain $\to$ Frequency Domain)
* **Code:**
  ```python
  rx_sym = tf.reshape(y, [B, 14, 276])
  rx_no_cp = rx_sym[:, :, 20:]                       # Remove Cyclic Prefix (first 20 samples)
  rx_grid_256 = tf.signal.fftshift(tf.signal.fft(rx_no_cp), axes=-1) # FFT to frequency domain
  ```
* Removes CP and converts time-domain samples back into 256 subcarriers per symbol.

---

#### Steps 6 & 7: Extract Active Subcarriers & Compute $H_{\text{eff}}$
* **Code:**
  ```python
  rx_grid_132 = rx_grid_256[:, :, 62:194]   # Extract 132 center active subcarriers
  H_eff = rx_grid_132 / tx_grid_132         # Y / 1 = Y
  ```
* Extracts the 132 active subcarriers to yield the final **$14 \times 132$ effective channel grid** ($H_{\text{eff}}$).

---

### Summary
Your intuition was spot on: **Transmit All-Ones Grid $\to$ IFFT $\to$ Sample-wise Phase Rotation $\to$ Time-Domain Channel $\to$ FFT $\to$ Extract Effective Channel Matrix.**

### User Input

@[c:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\Sionna\OpenNTN\example\channel_response copy.ipynb] can you check in this case, what is the velocity of the UE? can you check what scenarios is used? DenseUrban, Urban or SubUrban?

*Grep searched codebase*

*Listed directory [OpenNTN](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN) *

*Viewed [dense_urban.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/dense_urban.py) *

*Grep searched codebase*

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

### Planner Response

Here are the details from [channel_response copy.ipynb](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response%20copy.ipynb):

---

### 1. Which Scenario is Used?

The scenario used is **`DenseUrban`**.

* **Code Reference:** In cell 3 and cell 6 of `channel_response copy.ipynb`:
  ```python
  scenario = "dur"  # "dur" stands for DenseUrban in 3GPP TR 38.811 / OpenNTN
  
  channel_model = DenseUrban(
      carrier_frequency=carrier_frequency,
      ut_array=ut_array,
      bs_array=bs_array,
      direction=direction,
      elevation_angle=elevation_angle
  )
  ```

---

### 2. What is the Velocity of the UE (UT)?

The User Equipment (UE) velocity is **`0 m/s` (Stationary UE)**.

* **Code Reference:**
  When calling `gen_ntn_topology`:
  ```python
  topology = gen_ntn_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario, bs_height=satellite_height)
  ```
  In [utils.py line 717–719](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py#L717-L719), the default parameters for `dur` (DenseUrban) set both minimum and maximum UT velocity to zero:
  ```python
  'min_ut_velocity': 0.0 m/s,
  'max_ut_velocity': 0.0 m/s
  ```
  Hence, `ut_velocities = [0.0, 0.0, 0.0] m/s`.

---

### Key Takeaway on Doppler Shift
Because the UE is stationary ($0\text{ m/s}$), **100% of the Doppler shift ($\approx 32.3\text{ kHz}$) is caused by the high orbital speed of the LEO satellite** ($\approx 7.56\text{ km/s}$ at an altitude of $600\text{ km}$).

### User Input

h_1 = tf.transpose(h_1, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    h_2 = tf.transpose(h_2, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    H_perfect.append(h_1)
    H_perfect.append(h_2) 
in this, H_perfect should be appending the complex number of h_1, h_2, can you change it

*Viewed [channel_response copy.ipynb](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response%20copy.ipynb) *

### Planner Response

`h_freq` (from `ofdm_channel()`) is already a **complex TensorFlow tensor** (`complex64`), so slicing `h_freq` automatically preserves both the real and imaginary parts ($a + jb$).

To ensure `h_1` and `h_2` are cleanly transposed while retaining their complex data type as NumPy arrays before being appended to `H_perfect`, you can update the code as follows:

```python
    # Extract antenna 0 and antenna 1 (preserving complex64 numbers) and transpose to [14, 132, batch_size]
    h_1 = tf.transpose(h_freq[:, 0, 0, 0, 0, :, :], perm=[1, 2, 0]).numpy()  # Complex array [14, 132, batch_size]
    h_2 = tf.transpose(h_freq[:, 0, 0, 0, 1, :, :], perm=[1, 2, 0]).numpy()  # Complex array [14, 132, batch_size]

    # Append complex channel matrices to H_perfect list
    H_perfect.append(h_1)
    H_perfect.append(h_2)
```

### What this does:
1. **`h_freq[:, 0, 0, 0, 0, :, :]`**: Slices the complex tensor for Antenna 0 (shape `[batch_size, 14, 132]`, dtype `complex64`).
2. **`tf.transpose(..., perm=[1, 2, 0])`**: Reorders dimensions from `(batch_size, 14, 132)` to `(14, 132, batch_size)` while maintaining complex values.
3. **`.numpy()`**: Converts the TensorFlow complex tensor into a standard NumPy complex array (`np.complex64`), ready to be appended to `H_perfect`.

### User Input

so h_1 and h_2 is already a complex matric? but why do we need these 2 h_1 and h_2, what they do in the code? 

### Planner Response

### 1. Are `h_1` and `h_2` already complex matrices?
**Yes!** In Sionna, OFDM channel coefficients are complex numbers ($a + jb$, where $a$ is the real amplitude/phase component and $b$ is the imaginary component). Both `h_1` and `h_2` have the data type `complex64`.

---

### 2. Why do we need `h_1` and `h_2`? (What do they do in the code?)

#### **A. They represent the two Dual-Polarization channels from the satellite**
In your code, the satellite antenna array (`bs_array`) is configured with **dual-polarization**:
```python
bs_array = AntennaArray(num_rows=1, num_cols=1, polarization="dual", polarization_type="VH", ...)
```
This means the satellite transmits on **2 orthogonal polarizations**:
1. **Vertical Polarization (V):** Satellite Antenna Port 0 $\to$ **`h_1`**
2. **Horizontal Polarization (H):** Satellite Antenna Port 1 $\to$ **`h_2`**

When Sionna outputs `h_freq` with shape `[batch_size, 1, 1, 1, 2, 14, 132]`:
* **`h_1 = h_freq[:, 0, 0, 0, 0, :, :]`** extracts the channel matrix between the **Vertical polarization antenna** and the UE.
* **`h_2 = h_freq[:, 0, 0, 0, 1, :, :]`** extracts the channel matrix between the **Horizontal polarization antenna** and the UE.

---

#### **B. What do they do in the dataset generation?**
In the simulation loop, `h_1` and `h_2` are transposed to shape `[14, 132, batch_size]` (14 symbols $\times$ 132 subcarriers $\times$ 32 topology samples) and appended to the list:
```python
H_perfect.append(h_1)  # Store V-polarization channels for this batch
H_perfect.append(h_2)  # Store H-polarization channels for this batch
```
By doing this, `H_perfect` collects all generated channel samples for both polarizations across all simulation batches. This forms your dataset of ground-truth channel matrices (useful for training channel estimation or channel prediction models, as in your project path `1_Hprediction`).

### User Input

can you check if I want to use SubUrban, which scenario should I set?

### Planner Response

To use the **SubUrban** scenario in OpenNTN, set the scenario parameter string to **`"sur"`** and instantiate the **`SubUrban`** channel model class.

### 1. Parameters to Change

| Setting | Value for SubUrban | Note |
| :--- | :--- | :--- |
| **Scenario String** | **`scenario = "sur"`** | `"sur"` stands for **SubUrban** in 3GPP TR 38.811 / OpenNTN |
| **Channel Model Class** | **`SubUrban(...)`** | Imported from `OpenNTN` |

---

### 2. Code Snippet for SubUrban

In cell 3 & cell 6 of your notebook, update the code as follows:

```python
# 1. Set scenario string to "sur" for SubUrban
scenario = "sur"  # "sur" is the SubUrban scenario
carrier_frequency = 2.18e9
direction = "downlink"
elevation_angle = 50.0
num_ut = 1
satellite_height = 600000.0  # LEO Satellite 600 km
batch_size = 32

# 2. Import SubUrban channel model
from OpenNTN import SubUrban

# 3. Instantiate SubUrban channel model
channel_model = SubUrban(
    carrier_frequency=carrier_frequency,
    ut_array=ut_array,
    bs_array=bs_array,
    direction=direction,
    elevation_angle=elevation_angle
)

# 4. Generate topology with scenario="sur"
topology = gen_ntn_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario, bs_height=satellite_height)
channel_model.set_topology(*topology)
```

---

### 3. Scenario String Reference Summary

* **`"sur"`** = **SubUrban**
* **`"urb"`** = **Urban**
* **`"dur"`** = **DenseUrban**

### User Input

can you explain this # 4) Time-varying channel: y[n] = sum_l h_t[n,l] x[n-l]
    # h_t expected full shape: [B, Nr, NrAnt, Nt, NtAnt, T, L]
    h = h_t[:, 0, 0, 0, 0, :, :]  # [B,T,L]
    T = tf.shape(h)[1]
    L = tf.shape(h)[2]

    x_pad = tf.pad(x_pc, [[0,0],[L-1,0]])
    y = tf.zeros([B, T], dtype=tf.complex64)

    for l in tf.range(L):
        x_shift = x_pad[:, (L-1-l):(L-1-l+T)]
        y += h[:, :, l] * x_shift

    rx_waveform = y  # [B,3864]
the tx waveform passing through the channel by that formular? is it correct? 

### Planner Response

**YES, the formula and the code implementation are 100% mathematically correct.** 

This code provides a vectorized implementation of passing a signal through a **discrete time-varying multipath channel**.

---

### 1. Mathematical Explanation

In wireless communications, when a signal $x[n]$ passes through a multipath channel with $L$ delay taps whose gains change over time ($h_t[n, l]$), the received signal $y[n]$ at sample index $n$ is given by **time-varying convolution**:

$$y[n] = \sum_{l=0}^{L-1} h_t[n, l] \cdot x[n - l]$$

Where:
* $n \in \{0, 1, \dots, T-1\}$ is the discrete sample index (from 0 to 3,863).
* $l \in \{0, 1, \dots, L-1\}$ is the tap delay index (from 0 to $L-1$).
* $x[n - l]$ is the transmitted signal delayed by $l$ samples.
* $h_t[n, l]$ is the channel coefficient (gain and phase) of path $l$ at time instant $n$.

---

### 2. How the Code Implements it (Step-by-Step)

Instead of using slow Python `for` loops over all 3,864 time samples $n$, the code vectorizes the computation over all time samples $n$ simultaneously using TensorFlow:

#### Step 1: Slice single antenna channel `h`
```python
h = h_t[:, 0, 0, 0, 0, :, :]  # Shape: [Batch, T=3864, L]
```
`h[:, :, l]` gives the time-varying gain $h_t[n, l]$ for tap $l$ across all $T$ samples.

#### Step 2: Zero-padding for initial delay (`x_pad`)
```python
x_pad = tf.pad(x_pc, [[0,0], [L-1, 0]])  # Pad L-1 zeros at the beginning (n < 0)
```
Before sample $n=0$, the transmitted signal is zero ($x[n] = 0$ for $n < 0$). Padding $L-1$ zeros at the front prevents array index out-of-bounds when evaluating $x[n-l]$ for early samples.

#### Step 3: Extract delayed waveform `x_shift` for tap $l$
```python
for l in tf.range(L):
    x_shift = x_pad[:, (L-1-l) : (L-1-l+T)]  # Shifted waveform of length T
```
* For tap $l = 0$ (direct path): `x_shift` is $x[n]$ (delay = 0).
* For tap $l = 1$ (1st delay path): `x_shift` is $x[n-1]$ (delayed by 1 sample).
* For tap $l$: `x_shift` is $x[n-l]$ (delayed by $l$ samples).

#### Step 4: Multiply and Accumulate
```python
    y += h[:, :, l] * x_shift  # Multiplies h_t[n, l] * x[n-l] for all n simultaneously
```
This multiplies the channel gain $h_t[n, l]$ by the delayed signal $x[n-l]$ for tap $l$ across all time samples $n$, accumulating the result into $y$.

---

### Summary Table

| Code Variable | Physical Meaning | Formula Term |
| :--- | :--- | :--- |
| `h[:, :, l]` | Channel gain of path $l$ at sample $n$ | $h_t[n, l]$ |
| `x_shift` | Transmit signal delayed by $l$ samples | $x[n - l]$ |
| `h[:, :, l] * x_shift` | Received signal contribution from path $l$ | $h_t[n, l] \cdot x[n - l]$ |
| `y += ...` | Sum of all path contributions | $y[n] = \sum_{l=0}^{L-1} h_t[n, l] x[n-l]$ |

**Verdict:** The code is completely sound and implements standard discrete time-varying physical channel convolution.

### User Input

similar to this code  channel_model = SubUrban(carrier_frequency=carrier_frequency,
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            elevation_angle=elevation_angle)

    # Generate the topology
    topology = gen_ntn_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario,bs_height=satellite_height)

    # Set the topology
    channel_model.set_topology(*topology)

    # path_coefficients [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # path_delays [batch size, num_rx, num_tx, num_paths]
    num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
    # symbol_start = np.arange(1, 14) * 132
    # # symbol_start = np.concatenate(([22], np.arange(1, 14) * (132 + 18))) # 256       % including CP
    # symbol_start = tf.constant(symbol_start, dtype=tf.int32)
    path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)
    # h = cir_to_ofdm_channel(frequencies, path_coefficients, path_delays)
    # Convert CIR -> discrete time-domain channel taps h_t[b, l]
    l_min, l_max = time_lag_discrete_time_channel(bandwidth)
    h_t = cir_to_time_channel(
        bandwidth=bandwidth,
        a=path_coefficients,
        tau=path_delays,
        l_min=l_min,
        l_max=l_max,
        normalize=False
    )
    
    ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
    h_freq= ofdm_channel()
    
    remove_nulled = RemoveNulledSubcarriers(rg)
    h_perfect_ori = remove_nulled(h_freq)   # -> [..., 14, 132]
    
    h_freq_plot = (h_perfect_ori[1,0,0,0,0,:,:]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of h perfect ori')
    plt.show()
    
    sym_starts = tf.range(14, dtype=tf.int32) * (rg.fft_size + rg.cyclic_prefix_length) + rg.cyclic_prefix_length
    a_sym = tf.gather(path_coefficients, sym_starts, axis=-1)  # shape now has T=14
    h_f = cir_to_ofdm_channel(
        frequencies,   # from subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        a_sym,
        path_delays,
        normalize=False
    )

    h_1 = h_freq[:, 0, 0, 0, 0, :, :].numpy()  # Shape: [batch_size, 14, 132]
    h_2 = h_freq[:, 0, 0, 0, 1, :, :].numpy()  # Shape: [batch_size, 14, 132]

    h_1 = tf.transpose(h_1, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    h_2 = tf.transpose(h_2, perm=[1, 2, 0])  # (32,14,132) -> (14,132,32)
    H_perfect.append(h_1)
    H_perfect.append(h_2)
    
    #### txGrid Probe for Effective channel
    B = tf.shape(h_t)[0]  # batch size
    N_sym = rg.num_ofdm_symbols      # 14
    N_fft = rg.fft_size              # 256
    N_cp  = rg.cyclic_prefix_length  # 
    Fs = sampling_frequency

    # Your active band is centered: 62 left guards + 132 active + 62 right guards
    gL, gR = 62, 62
    N_act = 132
    assert N_fft == gL + N_act + gR

    # 1) txGrid: full ones on 14x132, then zero-pad -> 14x256
    tx_grid_132 = tf.ones([B, N_sym, N_act], dtype=tf.complex64)
    tx_grid_256 = tf.pad(tx_grid_132, paddings=[[0,0],[0,0],[gL,gR]])  # [B,14,256]

    # 2) OFDM modulate
    x_no_cp = tf.signal.ifft(tf.signal.ifftshift(tx_grid_256, axes=-1))  # [B,14,256]
    x_cp = tf.concat([x_no_cp[:, :, -N_cp:], x_no_cp], axis=-1)           # [B,14,256+CP]
    tx_waveform = tf.reshape(x_cp, [B, N_sym*(N_fft+N_cp)])               # [B,3864]

    # 3) Doppler pre-compensation (sample-wise)
     # Extract ONLY the Doppler shift frequency (approx +32.3 kHz)
    f_doppler = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency) - carrier_frequency
    n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)  # [0..3863]
    rot = tf.exp(tf.complex(tf.zeros_like(n), -2 * np.pi * f_doppler * n / Fs))
    x_pc = tx_waveform * rot[None, :]

    # 4) Time-varying channel: y[n] = sum_l h_t[n,l] x[n-l]
    # h_t expected full shape: [B, Nr, NrAnt, Nt, NtAnt, T, L]
    h = h_t[:, 0, 0, 0, 0, :, :]  # [B,T,L]
    T = tf.shape(h)[1]
    L = tf.shape(h)[2]

    x_pad = tf.pad(x_pc, [[0,0],[L-1,0]])
    y = tf.zeros([B, T], dtype=tf.complex64)

    for l in tf.range(L):
        x_shift = x_pad[:, (L-1-l):(L-1-l+T)]
        y += h[:, :, l] * x_shift

    rx_waveform = y  # [B,3864]

    # 5) OFDM demodulate
    rx_sym = tf.reshape(rx_waveform, [B, N_sym, N_fft+N_cp])
    rx_no_cp = rx_sym[:, :, N_cp:]
    rx_grid_256 = tf.signal.fftshift(tf.signal.fft(rx_no_cp), axes=-1)    # [B,14,256]

    # 6) Take active 132 (center)
    rx_grid_132 = rx_grid_256[:, :, gL:gL+N_act]                           # [B,14,132]

    # 7) Effective channel on active REs
    H_eff = rx_grid_132 / tx_grid_132   # tx_grid_132 is all ones -> H_eff == rx_grid_132
    
    h_freq_plot = (H_eff[1,:,:]).numpy().real
    h_freq_plot = h_freq_plot.T

    # Plot the concatenated heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of effective channel')
    plt.show()
     
but now I want to pass the tx through the frequency domain: convert compensated tx waveform back to frequency domain grid (256 FFT) then pass through the 256x14 channel grid (element wise, no adding noise) to get the 256x14 receive grid, then remove back to 132x14 grid of the received grid, also the effective channel grid

### Planner Response

Here is the complete code to pass the Doppler-compensated transmit waveform through the channel in the **frequency domain** (element-wise multiplication over the 256 $\times$ 14 grid without noise), extract the active 132 $\times$ 14 grid, and compute the effective channel.

---

### 1. Updated Code Block

Replace steps `4)` through `7)` in your notebook cell with the following frequency-domain pipeline:

```python
    #### Frequency-Domain Probe for Effective Channel
    B = tf.shape(h_t)[0]  # batch size (32)
    N_sym = rg.num_ofdm_symbols      # 14
    N_fft = rg.fft_size              # 256
    N_cp  = rg.cyclic_prefix_length  # 20
    Fs = sampling_frequency

    gL, gR = 62, 62
    N_act = 132
    assert N_fft == gL + N_act + gR

    # 1) txGrid: full ones on 14x132, then zero-pad -> 14x256
    tx_grid_132 = tf.ones([B, N_sym, N_act], dtype=tf.complex64)
    tx_grid_256 = tf.pad(tx_grid_132, paddings=[[0,0],[0,0],[gL,gR]])  # [B, 14, 256]

    # 2) OFDM modulate to time domain
    x_no_cp = tf.signal.ifft(tf.signal.ifftshift(tx_grid_256, axes=-1))  # [B, 14, 256]
    x_cp = tf.concat([x_no_cp[:, :, -N_cp:], x_no_cp], axis=-1)           # [B, 14, 256+CP]
    tx_waveform = tf.reshape(x_cp, [B, N_sym*(N_fft+N_cp)])               # [B, 3864]

    # 3) Sample-wise Doppler pre-compensation
    f_doppler = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency) - carrier_frequency
    n = tf.cast(tf.range(tf.shape(tx_waveform)[-1]), tf.float32)  # [0..3863]
    rot = tf.exp(tf.complex(tf.zeros_like(n), -2 * np.pi * f_doppler * n / Fs))
    x_pc = tx_waveform * rot[None, :]

    # 4) Convert compensated TX waveform back to 256x14 Frequency-Domain Grid
    x_pc_sym = tf.reshape(x_pc, [B, N_sym, N_fft + N_cp])                 # [B, 14, 276]
    x_pc_no_cp = x_pc_sym[:, :, N_cp:]                                    # Remove CP -> [B, 14, 256]
    X_pc_grid_256 = tf.signal.fftshift(tf.signal.fft(x_pc_no_cp), axes=-1) # 256 FFT -> [B, 14, 256]

    # 5) Generate 256x14 Channel Grid (H_channel_256)
    freqs_256 = subcarrier_frequencies(N_fft, rg.subcarrier_spacing)
    sym_starts = tf.range(N_sym, dtype=tf.int32) * (N_fft + N_cp) + N_cp
    a_sym = tf.gather(path_coefficients, sym_starts, axis=-1)
    h_f_256 = cir_to_ofdm_channel(freqs_256, a_sym, path_delays, normalize=False)
    H_channel_256 = h_f_256[:, 0, 0, 0, 0, :, :]                           # [B, 14, 256]

    # 6) Pass TX through channel in Frequency Domain (Element-wise, no noise)
    RX_grid_256 = X_pc_grid_256 * H_channel_256                            # [B, 14, 256]

    # 7) Extract active 132 subcarriers
    RX_grid_132 = RX_grid_256[:, :, gL:gL+N_act]                          # [B, 14, 132]

    # 8) Calculate Effective Channel Grid
    H_eff = RX_grid_132 / tx_grid_132                                     # [B, 14, 132]

    # Plot the effective channel heatmap
    h_freq_plot = (H_eff[1, :, :]).numpy().real.T
    plt.figure(figsize=(12, 6))
    plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol')
    plt.ylabel('Subcarrier')
    plt.title('Real of effective channel (Frequency-Domain Transmission)')
    plt.show()
```

---

### 2. Step-by-Step Flow Explanation

1. **Step 4 (Demodulate `x_pc` back to 256 FFT Grid):**
   * Takes the sample-wise Doppler pre-compensated waveform `x_pc` ($3,864$ samples), reshapes into 14 OFDM symbols, removes CP, and applies `FFT` + `fftshift` to obtain $X_{\text{pc\_grid\_256}}$ of shape `[B, 14, 256]`.
2. **Step 5 (Compute 256x14 Channel Grid):**
   * Uses `cir_to_ofdm_channel` with all 256 subcarrier frequencies to evaluate the full channel matrix $H_{\text{channel\_256}}$ of shape `[B, 14, 256]`.
3. **Step 6 (Element-wise Frequency Transmission):**
   * Computes $Y_{\text{256}} = X_{\text{pc\_grid\_256}} \odot H_{\text{channel\_256}}$ via element-wise complex multiplication (no noise added).
4. **Steps 7 & 8 (Extract Active Subcarriers & Effective Channel):**
   * Slices the center 132 subcarriers `[gL : gL + N_act]` to get $Y_{\text{132}}$, which gives the effective channel $H_{\text{eff}}$ of shape `[B, 14, 132]`.

### User Input

how to show code line number in antigravity 2.0?

### Planner Response

To show/toggle code line numbers in the Antigravity IDE (VS Code based), follow these steps depending on whether you are editing a **Jupyter Notebook (`.ipynb`)** or a **Python file (`.py`)**:

---

### Method 1: For Jupyter Notebooks (`.ipynb`)

#### 1. Keyboard Shortcut (Fastest)
* **Single Cell:** Press **`Esc`** to enter Command Mode (blue cell border), then press **`L`** on your keyboard to toggle line numbers for that cell.
* **All Cells in Notebook:** Press **`Shift + L`** to toggle line numbers for all code cells across the notebook.

#### 2. Notebook Settings (Permanent for all notebooks)
1. Press **`Ctrl + ,`** to open **Settings**.
2. Type **`notebook.lineNumbers`** in the search bar.
3. Change the setting from `off` to **`on`**.

---

### Method 2: For Python Files (`.py`) & Standard Code Editors

#### 1. Settings Menu (Permanent)
1. Press **`Ctrl + ,`** to open **Settings**.
2. Search for **`editor.lineNumbers`**.
3. Set it to **`on`**.

#### 2. Command Palette
1. Press **`Ctrl + Shift + P`** to open the Command Palette.
2. Type **`View: Toggle Line Numbers`** and press **Enter**.

### User Input

can you check, in the code, can I set the height of th UE? ut_array = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=carrier_frequency)

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

### Planner Response

**No**, you cannot set the UE height in `ut_array = Antenna(...)`. The `Antenna` class only configures antenna characteristics (polarization, radiation pattern, carrier frequency).

### Where to set the UE Height

UE height is defined in the **network topology** when calling `gen_ntn_topology` (short for `gen_single_sector_topology`).

#### 1. Default Value
By default in OpenNTN (according to 3GPP TR 38.811), the UE height is set to **`1.5 meters`**:
* `min_ut_height = 1.5` meters
* `max_ut_height = 1.5` meters

---

#### 2. How to change UE height in your code

Pass `min_ut_height` and `max_ut_height` into `gen_ntn_topology`:

```python
# Set your desired UE height (in meters)
ue_height = 1.5  # e.g. 1.5m, 10.0m, etc.

topology = gen_ntn_topology(
    batch_size=batch_size,
    num_ut=num_ut,
    scenario=scenario,
    bs_height=satellite_height,
    elevation_angle=elevation_angle,
    min_ut_height=ue_height,  # <--- Set minimum UE height in meters
    max_ut_height=ue_height   # <--- Set maximum UE height in meters
)

# Apply topology to channel model
channel_model.set_topology(*topology)
```

*(Note: If `min_ut_height` and `max_ut_height` are set to the same value, e.g. `1.5`, all UEs will be placed at exactly that height. If they are different, e.g., `1.5` and `10.0`, UEs will be randomly dropped at heights between $1.5\text{ m}$ and $10\text{ m}$.)*

### User Input

can you check why in 1 scenario, for example, SubUrban, the generated channel grids can vary a lot, like, some look very good after effective Doppler compensation, some look changing very abruptive, not smooth

*Grep searched codebase*

*Viewed [sub_urban_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/sub_urban_scenario.py) *

*Viewed [sub_urban_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/sub_urban_scenario.py) *

### Planner Response

The variation between **smooth, clean heatmaps** and **abrupt, fluctuating heatmaps** within the same scenario (e.g., `SubUrban`) is a natural physical effect driven by 3GPP TR 38.811 channel modeling rules.

Here are the primary reasons why channel grids vary across generated topology drops:

---

### 1. Line-of-Sight (LoS) vs. Non-Line-of-Sight (NLoS) Fading

In each topology drop (`batch_size=32`), OpenNTN randomly assigns each UE to either a **LoS** or **NLoS** state based on 3GPP probability curves $P_{\text{LoS}}(\theta)$ for the given elevation angle ($\theta = 50^\circ$).

* **LoS Drops (Smooth Heatmaps):** 
  * The channel is dominated by a strong direct path (high Rician K-factor).
  * Because one strong path dominates over weak reflections, the frequency response $H(f, t)$ remains **very smooth and uniform** across both subcarriers and OFDM symbols.
* **NLoS Drops (Abrupt / Jagged Heatmaps):**
  * The direct path is blocked by obstacles (buildings, terrain, foliage).
  * The signal consists purely of scattered multipath rays (**Rayleigh fading**).
  * Different paths interfere constructively at some subcarrier frequencies and destructively at others, creating deep **frequency notches** and sharp magnitude drops across adjacent subcarriers.

---

### 2. Random Delay Spread Variance ($\sigma_{\tau}$)

For every generated topology drop, OpenNTN samples random Large-Scale Parameters (LSPs), including the **RMS Delay Spread ($\tau_{\text{rms}}$)**:

* **Coherence Bandwidth ($B_c \approx \frac{1}{5 \tau_{\text{rms}}}$):**
  * **Small Delay Spread:** Coherence bandwidth is wider than your transmission bandwidth $\to$ all subcarriers experience similar channel gains (**Flat Fading / Smooth Grid**).
  * **Large Delay Spread:** Coherence bandwidth is smaller than the subcarrier spacing $\to$ adjacent subcarriers fade independently (**Frequency-Selective Fading / Abrupt Grid**).

---

### 3. Multipath Cluster Phase Interference

In NLoS conditions, OpenNTN generates **clusters of 20 sub-rays** with random initial phases $\phi \sim U(-\pi, \pi)$:
$$\text{Channel Gain } H(f) = \sum_{l=1}^{L} a_l e^{-j (2\pi f \tau_l + \phi_l)}$$
Depending on the random phase combination $\phi_l$, some subcarriers experience total destructive cancellation (deep dark lines in the heatmap), while neighboring subcarriers constructively add up.

---

### 4. Doppler Pre-Compensation Scope

Sample-wise Doppler pre-compensation cancels the **common frequency shift** caused by satellite orbital motion ($f_{\text{doppler}} \approx 32.3\text{ kHz}$), but **it does not remove multipath delay spread or NLoS frequency selectivity**.

---

### How to Verify LoS vs. NLoS in Your Code

You can check which batch samples in your simulation are LoS vs NLoS by inspecting `channel_model.los`:

```python
# Check LoS status for all 32 topology drops in the batch
los_status = channel_model.los.numpy()  # True = LoS (Smooth), False = NLoS (Abrupt)

print("LoS status per topology drop:", los_status)
```

If you filter your plots:
* Heatmaps where `los_status[i] == True` will look **smooth and uniform**.
* Heatmaps where `los_status[i] == False` will show **abrupt frequency selectivity and notches**.

### User Input

AttributeError: 'SubUrban' object has no attribute 'los' @[c:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\Sionna\OpenNTN\OpenNTN\sub_urban_scenario.py] 
can you check if I can check LOS in the code?

*Viewed [sub_urban_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/sub_urban_scenario.py) *

*Grep searched codebase*

*Grep searched codebase*

*Viewed [system_level_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py) *

*Viewed [system_level_channel.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py) *

*Viewed [system_level_channel.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py) *

### Planner Response

To check or control the Line-of-Sight (LoS) status in OpenNTN, use **`channel_model._scenario.los`**. 

The `SubUrban` object itself wraps a scenario object `_scenario`, which stores the `los` status array.

---

### Option 1: Read the LoS Status of Generated Drops

To check whether each drop in your batch (`batch_size=32`) is LoS (`True`) or NLoS (`False`), read:

```python
# Access the LoS boolean tensor from the underlying scenario
los_status = channel_model._scenario.los.numpy().squeeze()

# Example output for batch_size=32:
# [True, True, False, True, False, ...]
for i, status in enumerate(los_status):
    print(f"Drop {i}: {'LoS (Smooth Grid)' if status else 'NLoS (Jagged Grid)'}")
```

---

### Option 2: Force All Drops to be 100% LoS or 100% NLoS

If you want to test **only smooth channels** (or **only non-smooth channels**), pass `los=True` or `los=False` when setting the topology:

#### Force All Drops to be LoS (Smooth Heatmaps):
```python
# Force all 32 topology drops in the batch to be Line-of-Sight
channel_model.set_topology(*topology, los=True)
```

#### Force All Drops to be NLoS (Jagged/Multipath Heatmaps):
```python
# Force all 32 topology drops in the batch to be Non-Line-of-Sight
channel_model.set_topology(*topology, los=False)
```

---

### Summary Table

| Goal | Code |
| :--- | :--- |
| **Check LoS status** | `channel_model._scenario.los.numpy().squeeze()` |
| **Force 100% LoS** | `channel_model.set_topology(*topology, los=True)` |
| **Force 100% NLoS** | `channel_model.set_topology(*topology, los=False)` |

### User Input

but the abruptive changes are from 1 OFDM symbol to the next - actually, after Doppler compensation, the effective channel should look smooth, right? does that mean my code is having the error of Doppler compensation?

### Planner Response

You have hit on the exact reason for this behavior! Your Doppler compensation code is working, but the abrupt symbol-to-symbol jumps in the plot are caused by taking **`.real`** instead of **`np.abs()`** (Magnitude) when plotting.

---

### The Reason: Real Part (`.real`) vs. Magnitude (`np.abs()`)

In your plotting code:
```python
h_freq_plot = (H_eff[1, :, :]).numpy().real   # <--- TAKING THE REAL PART
plt.colorbar(label='Magnitude')               # <--- LABEL SAYS MAGNITUDE
```

A complex channel coefficient $H$ consists of both magnitude $|H|$ and phase angle $\phi$:
$$H = |H| \cdot e^{j \phi}$$

When you take the **Real part**:
$$\text{Re}(H) = |H| \cdot \cos(\phi)$$

1. **Why `.real` jumps abruptly between symbols:**
   * Even after satellite Doppler compensation, there is a small continuous phase progression $\phi_m$ across OFDM symbols $m = 0, 1, 2, \dots, 13$.
   * As the phase $\phi_m$ slowly rotates, $\cos(\phi_m)$ oscillates between **$+1$ and $-1$**.
   * On your heatmap, this causes adjacent OFDM symbols to rapidly flip colors (e.g., $+0.8 \to -0.7 \to +0.9$), creating **abrupt, stripe-like vertical jumps** between symbols!

2. **Why `.abs()` is smooth:**
   * Taking the **Magnitude** $|H| = \sqrt{\text{Re}^2 + \text{Im}^2}$ strips away the phase rotation $e^{j\phi}$.
   * The magnitude $|H|$ represents the actual channel energy gain, which is **completely smooth and continuous** across OFDM symbols.

---

### How to Fix the Plot

Change `.real` to **`np.abs()`** in your plotting code:

```python
# Use np.abs() for TRUE Channel Magnitude
h_freq_plot = np.abs(H_eff[1, :, :].numpy()).T

plt.figure(figsize=(12, 6))
plt.imshow(h_freq_plot, aspect='auto', cmap='viridis')
plt.colorbar(label='Magnitude (|H|)')
plt.xlabel('Symbol')
plt.ylabel('Subcarrier')
plt.title('Magnitude of Effective Channel (|H_eff|)')
plt.show()
```

---

### Comparison Summary

| Method | Formula | Behavior Across Symbols | What It Shows |
| :--- | :--- | :--- | :--- |
| **`np.abs(H)`** | $|H|$ | **Smooth & Continuous** ✅ | **True Channel Gain (Magnitude)** |
| **`H.real`** | $|H|\cos(\phi)$ | Oscillates / Jumps ❌ | Real projection (modulated by phase) |
| **`np.angle(H)`** | $\phi$ | Linear Phase Slope | Phase Progression / Residual Shift |

Switching to `np.abs()` will display the smooth, expected effective channel heatmap!

### User Input

 144 h_freq_plot = (H_eff[1,:,:]).numpy().abs
    145 h_freq_plot = h_freq_plot.T
    147 # Plot the concatenated heatmap

AttributeError: 'numpy.ndarray' object has no attribute 'abs' 
how to fix the error

### Planner Response

In NumPy, `abs` is a function (`np.abs()`), not a property attribute like `.abs`.

### How to Fix

You can fix it using either **`np.abs(...)`** or **`tf.abs(...)`**:

#### Option 1: Using `np.abs()` (Recommended)
```python
import numpy as np

h_freq_plot = np.abs(H_eff[1, :, :].numpy())
h_freq_plot = h_freq_plot.T
```

#### Option 2: Using `tf.abs()`
```python
h_freq_plot = tf.abs(H_eff[1, :, :]).numpy()
h_freq_plot = h_freq_plot.T
```

### User Input

why my effective channels have a region near zeros near the zero subcarriers? is it haveing error when we convert FFT IFFT ?

### Planner Response

The region of near-zero values near the edge subcarriers (index 0 of your 132 active grid) is **not an FFT/IFFT error**—it is a real physical effect of **Doppler frequency shifting in the frequency domain**.

---

### Why this happens (The Exact Cause)

#### 1. Frequency Shift of $-32.3\text{ kHz}$ ($\approx -1.08$ Subcarriers)
In step 3, Doppler pre-compensation rotates the time-domain waveform $x[n]$ by:
$$\text{rot} = e^{-j 2\pi f_{\text{doppler}} \frac{n}{F_s}}, \quad \text{where } f_{\text{doppler}} \approx +32.3\text{ kHz}$$

In signal processing, **multiplying by a complex exponential in time domain equals a frequency shift in frequency domain**:
$$x[n] \cdot e^{-j 2\pi f_d \frac{n}{F_s}} \stackrel{\mathcal{F}}{\longleftrightarrow} X(f + f_d)$$

Since your subcarrier spacing is **$\text{SCS} = 30\text{ kHz}$**:
$$\text{Frequency Shift in Subcarriers} = \frac{-32.3\text{ kHz}}{30\text{ kHz}} \approx \mathbf{-1.08 \text{ subcarriers}}$$

---

#### 2. Energy Shifting into the Guard Band
Before pre-compensation, your active subcarriers occupy indices `62` to `193` in the 256-FFT grid, while indices `0..61` are zero guard carriers:

```
[ Guard (0..61) : ZEROS ] [ Active Data (62..193) : ONES ] [ Guard (194..255) : ZEROS ]
```

When the entire spectrum shifts by **$-1.08$ subcarriers**:
* Subcarrier `62` (the very first active subcarrier, index 0 of your 132 grid) is shifted to subcarrier `60.92`, falling into the **zero guard band** (`0..61`).
* Because it shifted into the zero guard band, when you extract `RX_grid_132 = RX_grid_256[:, :, 62:194]`, **subcarrier index 0 (and adjacent subcarriers) lost its energy, appearing as near-zero!**

---

### Summary Table of the Spectral Shift

| Subcarrier Index in 132 Grid | 256-FFT Bin | Frequency Shift Effect | Energy Level |
| :--- | :--- | :--- | :--- |
| **Index 0 (Edge)** | Bin 62 | Shifted to Bin 60.92 (Guard Band) | **Near ZERO** ❌ |
| **Index 1 .. 130 (Center)** | Bins 63 .. 192 | Shifted within active band | **Full Energy ($\sim 10^{-11}$)** ✅ |
| **Index 131 (Upper Edge)** | Bin 193 | Shifted to Bin 191.92 | **Full Energy** ✅ |

---

### How to Fix / Avoid Edge Energy Drop

If you want subcarrier 0 to maintain full energy without edge drop-off, you can:

1. **Option A: Extend Active Grid Padding (Recommended)**
   Pad 1 extra active subcarrier on the left (`61:194`) before FFT demodulation so edge energy isn't truncated by guard slicing:
   ```python
   # Extract 61 to 194 to capture the shifted edge subcarrier
   RX_grid_132 = RX_grid_256[:, :, 61:193]
   ```

2. **Option B: Perform Receiver-side Doppler Compensation**
   Apply Doppler compensation at the **receiver** (after passing through the channel), which shifts the spectrum back $+1.08$ subcarriers into exact alignment.

### User Input

can you check the OpenNTN library - how does the code defines/applies Doppler shift to the channel/tx signal? 
does it apply to sample-wise or what? 

*Viewed [sub_urban_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/sub_urban_scenario.py) *

*Grep searched codebase*

*Grep searched codebase*

*Viewed [channel_coefficients.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py) *

*Viewed [channel_coefficients.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py) *

*Viewed [channel_coefficients.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py) *

### Planner Response

Inside the OpenNTN library, Doppler shift is defined and applied **100% SAMPLE-WISE** to the channel coefficients $h_t$.

Here is the exact breakdown from the OpenNTN codebase in [channel_coefficients.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py#L604-L634):

---

### 1. How OpenNTN Defines & Calculates Doppler Shift

In `OpenNTN/channel_coefficients.py` (lines 609–618), OpenNTN calculates the satellite Doppler frequency shift per time step:

```python
# 1. Maximum Doppler rotation frequency per second
max_rotation_per_time = (2.0 * PI / lambda_0) * max_sat_speed_for_elevation_angle

# 2. Time vector t containing exact sample timestamps: t = [0/Fs, 1/Fs, 2/Fs, ..., (N-1)/Fs]
rotation_for_time = max_rotation_per_time * t

# 3. Compute continuous sample-wise Doppler phase matrix
h_doppler = tf.exp(tf.complex(0.0, rotation_for_time))
```

* **Doppler Shift Formula:**
  $$\phi_{\text{doppler}}(t_n) = 2\pi \cdot f_{\text{doppler}} \cdot t_n = 2\pi \cdot f_{\text{doppler}} \cdot \frac{n}{F_s}$$
* **Doppler Phase Shift Tensor:**
  $$h_{\text{doppler}}[n] = e^{j 2\pi f_{\text{doppler}} \frac{n}{F_s}}$$

---

### 2. Does it Apply Sample-Wise or Symbol-Wise?

**It applies SAMPLE-WISE.**

* **Time Vector `t`:** OpenNTN generates $t$ as a continuous vector of length $N$ ($N = 3,864$ samples for 14 OFDM symbols with CP):
  $$t = \left[ 0, \frac{1}{F_s}, \frac{2}{F_s}, \dots, \frac{3863}{F_s} \right]$$
* **No Resets:** The timestamp $t_n$ increases continuously from sample $0$ to $3863$. It does **not** reset at OFDM symbol boundaries.
* **Application to Signal:** OpenNTN multiplies $h_{\text{doppler}}[n]$ into each channel tap $l$ at sample time $n$:
  $$h_t[n, l] = h_{\text{base}}[n, l] \cdot e^{j 2\pi f_{\text{doppler}} \frac{n}{F_s}}$$

---

### Summary Table

| OpenNTN Component | Implementation Detail | Reference Code |
| :--- | :--- | :--- |
| **Doppler Method** | Continuous Sample-Wise Phase Shift | [channel_coefficients.py:L604–L634](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py#L604-L634) |
| **Sample Indexing** | Continuous time steps $t_n = \frac{n}{F_s}$ ($n = 0 \dots 3863$) | Sample-by-sample |
| **Channel Integration** | Multiplied into channel coefficients $h_t[n, l]$ | Time-varying CIR taps |

### User Input

so - it is actually applied straight to channel? we can easily apply a revert to get the effective channel? 
no need to perform the proning approach (send a full 1 grid, apply Doppler compensate,... then get the receive grid as the effective (Doppler removed) channel) like in MATLAB (where the library is very complicated, we only have a way to pass the tx waveform by the function channel(txwaveform) so we need to do the proning sending all-1 grid like that?

### Planner Response

**YES, you are 100% correct!**

You have discovered a major advantage of OpenNTN/Sionna over MATLAB's 5G/NTN Toolboxes:

---

### Why MATLAB vs. OpenNTN Are Different

* **In MATLAB (Black-Box Object):**
  MATLAB's channel model is a closed "black box" object where you can only call `rxWaveform = channel(txWaveform)`. Because MATLAB hides the internal channel tap matrices $h_t[n, l]$, **you are forced to use the probing approach** (all-ones grid $\to$ IFFT $\to$ CP $\to$ Doppler pre-compensation $\to$ `channel(tx)` $\to$ FFT) to measure the effective channel.

* **In OpenNTN / Sionna (Transparent TensorFlow Code):**
  OpenNTN gives you **full direct access to the underlying channel math, path coefficients, and parameters**. You do **NOT** need to perform the complex all-ones grid probing unless you specifically want to test physical CP truncation effects!

---

### 3 Ways to Get the Doppler-Removed Channel in OpenNTN

#### Method A: Disable Satellite Doppler Directly (Cleanest — 1 Line of Code)

OpenNTN includes a built-in parameter **`doppler_enabled`**. If you set `doppler_enabled=False`, OpenNTN skips adding the Doppler phase shift altogether:

```python
# 1. Instantiate channel model with doppler_enabled = False
channel_model = SubUrban(
    carrier_frequency=carrier_frequency,
    ut_array=ut_array,
    bs_array=bs_array,
    direction=direction,
    elevation_angle=elevation_angle,
    doppler_enabled=False   # <--- DISABLES SATELLITE DOPPLER DIRECTLY!
)

channel_model.set_topology(*topology)

# 2. Get the Doppler-free OFDM channel directly in 1 line:
ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
h_eff_no_doppler = RemoveNulledSubcarriers(rg)(ofdm_channel())  # Shape: [B, 1, 1, 1, 2, 14, 132]
```

---

#### Method B: Direct Phase Reversal on Channel Taps ($h_t$)

If you already generated `h_t`, you can revert the Doppler phase by directly multiplying $h_t$ by $e^{-j 2\pi f_{\text{doppler}} t_n}$:

```python
# Extract Doppler shift frequency
f_doppler = compute_stallite_doppler(satellite_height, elevation_angle, carrier_frequency) - carrier_frequency
n = tf.cast(tf.range(T), tf.float32)

# Create counter-rotation phase vector
rot_revert = tf.exp(tf.complex(0.0, -2 * np.pi * f_doppler * n / Fs))

# Revert Doppler directly from time-domain channel taps (no IFFT/FFT needed!)
h_t_no_doppler = h_t * rot_revert[None, None, None, None, None, :, None]
```

---

#### Method C: End-to-End Probing (What you wrote)

The probing approach (sending all-ones grid $\to$ IFFT $\to$ Doppler pre-compensation $\to$ channel convolution $\to$ FFT) is only needed if you want to explicitly simulate **physical layer hardware behavior** (e.g. testing Cyclic Prefix leakage or inter-carrier interference).

---

### Summary Comparison

| Goal | MATLAB | OpenNTN / Sionna |
| :--- | :--- | :--- |
| **Get Doppler-free channel** | Must probe with all-ones grid $\to$ `channel(tx)` | `SubUrban(..., doppler_enabled=False)` (1 Line) |
| **Channel Transparency** | Black-box object | Direct access to $h_t$, path coefficients, and parameters |
| **Computation Speed** | Slow (Full waveform convolution required) | Instant (Direct TensorFlow tensor math) |

### User Input

about channel_model = SubUrban(
    carrier_frequency=carrier_frequency,
    ut_array=ut_array,
    bs_array=bs_array,
    direction=direction,
    elevation_angle=elevation_angle,
    doppler_enabled=False   # <--- DISABLES SATELLITE DOPPLER DIRECTLY!
) 
if I set velocity of the UE, will this also disables foppler of the movement of UE? what should I set if I just consider ignoring the Doppler of the satellite?

### Planner Response

**Yes**, setting `doppler_enabled=False` **only disables the satellite Doppler shift**. It does **not** disable the Doppler shift caused by the movement of the UE.

---

### How OpenNTN Handles This Under the Hood

Inside the OpenNTN Doppler calculations ([channel_coefficients.py line 604–630](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/channel_coefficients.py#L604-L630)), the Doppler shift is divided into two parts:

```python
# 1. Compute phase shift due to UE movement Doppler (Always active)
exponent = 2 * PI / lambda_0 * tf.reduce_sum(r_hat_ut * v_uts_bar, -2) * t

# 2. Add phase shift due to Satellite Doppler (Controlled by doppler_enabled)
if tf.logical_and(tf.less_equal(600000.0, topology.bs_height), topology.doppler_enabled):
    # Calculate satellite orbital Doppler and add it to the exponent
    exponent = tf.math.add(exponent, rotation_for_time)
```

---

### What to Set in Your Case

If you want to **ignore the satellite Doppler but keep/consider the UE movement Doppler**:

* Set **`doppler_enabled=False`** on your channel model class (`SubUrban`, `DenseUrban`, or `Urban`).
* Set your desired UE velocity vectors in the topology (via `gen_ntn_topology` or `set_topology`).

This setup will completely ignore the satellite's $7.56\text{ km/s}$ orbital Doppler shift, but will still simulate the Doppler shift caused by the UE's movement on the ground!

### User Input

is that in practical realistic system, we also do the Doppler compensation on sample-wise for satellite? symbol-wise doppler compensate just for the on-ground movement (velocity not large too much as the satellite)?

### Planner Response

**Yes, this is exactly how practical wireless systems are designed.** Your understanding of the physical constraints is entirely correct.

Here is why practical systems (like 5G NTN Release 17/18) split Doppler compensation this way:

---

### 1. Satellite Doppler (Sample-wise / Time-Domain)
* **The Scale:** LEO satellites travel at around $7.56\text{ km/s}$, causing huge Doppler shifts ($\approx 30\text{ to } 50\text{ kHz}$ at S-band, and up to several hundred kHz in Ka-band).
* **The Problem:** This Doppler shift is **larger than the Subcarrier Spacing (SCS)** of 15 kHz or 30 kHz. If left uncompensated, it completely destroys subcarrier orthogonality, causing catastrophic **Inter-Carrier Interference (ICI)**.
* **Practical Solution:** Transceivers perform **continuous time-domain (sample-wise) pre-compensation** using GNSS (GPS) coordinates and satellite orbital data (Ephemeris). The transmitter rotates the phase of every time sample continuously to "pre-cancel" the satellite Doppler shift before the signal is even launched.

---

### 2. On-Ground UE Doppler (Symbol-wise / Reference Signals)
* **The Scale:** UEs on the ground travel at much slower speeds (e.g., walking at $1.5\text{ m/s}$ or driving at $30\text{ m/s}$). This results in tiny Doppler shifts ($\approx 10\text{ Hz}$ to $200\text{ Hz}$ at 2 GHz).
* **The Problem:** Because $200\text{ Hz}$ is much smaller than a $30\text{ kHz}$ subcarrier spacing ($<1\%$), it does not cause subcarrier leakage or ICI. However, it still causes a **slow phase rotation** over time.
* **Practical Solution:** The receiver corrects this slow phase drift **symbol-by-symbol** using pilot/reference signals (like DMRS or PTRS in 5G). The channel estimator measures the phase change between symbols and applies a simple symbol-wise phase correction, making sample-wise tracking unnecessary.

---

### Summary Comparison in 5G NTN

| Doppler Source | Speed | Doppler Shift | Impact | Practical Compensation Method |
| :--- | :--- | :--- | :--- | :--- |
| **LEO Satellite** | $\approx 7.5\text{ km/s}$ | $\approx 30 - 50\text{ kHz}$ | Severe ICI (larger than SCS) | **Sample-wise** pre-compensation in Time-Domain (using GNSS & Ephemeris) |
| **Ground UE** | $\approx 1.5 - 30\text{ m/s}$ | $\approx 10 - 200\text{ Hz}$ | Slow Phase Rotation | **Symbol-wise** tracking in Frequency-Domain (using DMRS / PTRS pilots) |

### User Input

can you check how can I set the velocity of the UE? 

*Viewed [channel_response.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/example/channel_response.py) *

*Viewed [utils.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/utils.py) *

### Planner Response

You can set the velocity of the UE in two ways, depending on whether you want a randomized direction or a specific custom 3D direction:

---

### Method 1: Using `gen_ntn_topology` (Randomized Direction)

You can specify the range of the UE velocity (in **meters per second, m/s**) when generating the topology. OpenNTN will automatically generate a random direction in the horizontal plane ($v_x, v_y$) with $v_z = 0$.

```python
# Set a constant UE speed of 30 m/s (approx. 108 km/h)
ue_speed = 30.0 

topology = gen_ntn_topology(
    batch_size=batch_size,
    num_ut=num_ut,
    scenario=scenario,
    bs_height=satellite_height,
    elevation_angle=elevation_angle,
    min_ut_velocity=ue_speed,   # <--- Set minimum speed
    max_ut_velocity=ue_speed    # <--- Set maximum speed
)

# Apply topology to channel model
channel_model.set_topology(*topology)
```

---

### Method 2: Manually Set Custom 3D Velocity Vectors

If you want the UE to move in a **specific 3D direction** (e.g. exactly along the X-axis at $30\text{ m/s}$), you can unpack the topology tuple, overwrite `ut_velocities` with a custom tensor of shape `[batch_size, num_ut, 3]`, and pass it manually:

```python
# 1. Unpack the default topology (velocity defaults to 0 m/s)
ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = gen_ntn_topology(
    batch_size=batch_size,
    num_ut=num_ut,
    scenario=scenario,
    bs_height=satellite_height,
    elevation_angle=elevation_angle
)

# 2. Overwrite velocity: e.g. 30 m/s directly along the X-axis [Vx, Vy, Vz]
# Shape must be [batch_size, num_ut, 3]
custom_velocity = tf.tile(tf.constant([[[30.0, 0.0, 0.0]]], dtype=tf.float32), [batch_size, num_ut, 1])

# 3. Call set_topology manually
channel_model.set_topology(
    ut_loc,
    bs_loc,
    ut_orientations,
    bs_orientations,
    custom_velocity,  # <--- Pass your custom velocity vectors here
    in_state
)
```

### User Input

then, can I plot that topology?

*Grep searched codebase*

*Viewed [system_level_channel.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py) *

*Grep searched codebase*

*Viewed [system_level_channel.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py) *

*Viewed [system_level_channel.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py) *

*Viewed [system_level_channel.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_channel.py) *

### Planner Response

Yes! OpenNTN/Sionna has a built-in 3D plotting method called **`show_topology()`** on the channel model object.

---

### How to Plot the Topology

Simply call:

```python
# 1. Enable inline matplotlib plotting in your notebook cell
%matplotlib inline

# 2. Plot the 3D network topology (defaults to batch index 0)
channel_model.show_topology()
```

#### Viewing a Specific Batch Item (e.g., Batch Index `i`)
Since you generate a batch size of 32 topologies, you can view the 3D layout for a specific batch sample `i` (from `0` to `31`) by passing `batch_index`:

```python
# Plot the 3D topology for batch example 15
channel_model.show_topology(batch_index=15)
```

---

### What the Plot Shows

The `show_topology` method opens a 3D Matplotlib plot that shows:
1. **The Satellite (BS):** Plotted as a black dot (**`●`**) at `[0, 0, bs_height]` (e.g. at $600\text{ km}$ altitude).
2. **The Ground UEs (UT):** Plotted at their respective 3D coordinates.
3. **LoS vs. NLoS Colors:**
   * **Red dots:** UEs in **LoS** (Line-of-Sight) state.
   * **Yellow/Green dots:** UEs in **NLoS** (Non-Line-of-Sight) state.
4. **Coordinate Arrows (RGB):** Draws small red-green-blue coordinate axes at each node representing the antenna orientation.

### User Input

can I set limit of number of paths? like <7 paths or something?

*Grep searched codebase*

*Grep searched codebase*

*Viewed [system_level_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py) *

*Grep searched codebase*

*Viewed [rays.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/rays.py) *

*Viewed [rays.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/rays.py) *

*Grep searched codebase*

*Viewed [system_level_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py) *

*Grep searched codebase*

*Viewed [system_level_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py) *

*Viewed [system_level_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py) *

*Viewed [system_level_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py) *

*Listed directory [models](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models) *

*Grep searched codebase*

*Viewed [Sub_Urban_LOS_S_band_DL.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/Sub_Urban_LOS_S_band_DL.json) *

*Grep searched codebase*

*Viewed [Sub_Urban_LOS_S_band_DL.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/Sub_Urban_LOS_S_band_DL.json) *

*Grep searched codebase*

*Viewed [system_level_scenario.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/system_level_scenario.py) *

*Grep searched codebase*

*Grep searched codebase*

*Viewed [Sub_Urban_LOS_S_band_DL.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/Sub_Urban_LOS_S_band_DL.json) *

*Grep searched codebase*

*Viewed [Sub_Urban_NLOS_S_band_DL.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/Sub_Urban_NLOS_S_band_DL.json) *

### Planner Response

**Yes, you can limit the number of paths (which are generated as clusters in OpenNTN/Sionna).**

---

### 1. Default Number of Paths in your Scenario

For the **SubUrban** scenario at **50 degrees elevation**, the default number of paths is already quite small:
* **Line-of-Sight (LoS):** **`3` paths** (`numClusters_50` = 3)
* **Non-Line-of-Sight (NLoS):** **`4` paths** (`numClusters_50` = 4)

*(Note: Each cluster has 20 sub-rays, which are ultimately combined into a single resolved path in the delay domain, meaning you get 3 or 4 path delays/taps in `h_t`).*

---

### 2. How to Change/Limit the Number of Paths in Python

Because OpenNTN loads these parameters from a JSON configuration dictionary, you can **override the number of paths directly in your code** before generating the channel response:

```python
# 1. Instantiate the channel model as usual
channel_model = SubUrban(
    carrier_frequency=carrier_frequency,
    ut_array=ut_array,
    bs_array=bs_array,
    direction=direction,
    elevation_angle=elevation_angle
)
channel_model.set_topology(*topology)

# 2. OVERRIDE: Limit the number of paths (clusters) to exactly 2 for 50° elevation
channel_model._scenario._params_los["numClusters_50"] = 2
channel_model._scenario._params_nlos["numClusters_50"] = 2

# 3. Generate channel coefficients (will now contain exactly 2 paths/delay taps)
path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)
```

By altering `"numClusters_50"` (which matches your $50^\circ$ elevation angle), the rest of the channel model will automatically adapt to generate only the specified number of paths.

### User Input

but can you explain why LOS but we have 2 paths? I thought we alsway just have maximum 1 path of LOS? 

### Planner Response

In wireless channel modeling (like 3GPP TR 38.901 / TR 38.811), a **"Line-of-Sight (LoS)" scenario does not mean there is only one path.** 

It means that the **direct path is present/visible**, but **secondary reflections from the ground, buildings, and trees still arrive at the receiver**. 

Here is the exact breakdown of why a LoS channel has 3 paths:

---

### 1. The 3 Paths in a LoS Scenario

In a LoS scenario, the channel is composed of:
$$\text{Channel Response } H = \underbrace{\text{Direct LoS Path (Path 0)}}_{\text{Strongest / Direct}} + \sum_{l=1}^{2} \underbrace{\text{Reflected Multipath (Paths 1 & 2)}}_{\text{Weaker scatterers/reflections}}$$

* **Path 0 (Line-of-Sight):** The direct, straight-line path from the satellite to the UE. It has the shortest delay ($\tau_0 = 0$) and carries the dominant portion of the signal power.
* **Paths 1 & 2 (Reflections):** Secondary rays that bounce off the ground, roads, trees, or nearby suburban houses before reaching the UE. They arrive slightly later ($\tau_1, \tau_2 > 0$) with much lower power.

---

### 2. Rician Fading (The K-Factor)

Because there is one dominant path (Path 0) alongside a few weak reflections (Paths 1 and 2), the channel is mathematically modeled using **Rician Fading**:

* **K-Factor:** The ratio of the power in the direct LoS path to the total power in all scattered reflection paths. 
* In a LoS scenario, the K-factor is high, meaning Path 0 is very strong, which is why your frequency response heatmap looks **smooth**.

---

### 3. How does this differ from NLoS?

* **In NLoS:** The direct path (Path 0) is completely blocked. All 4 paths are scattered reflections of comparable weak power, resulting in **Rayleigh Fading** (deep frequency notches and abrupt fading).

### Real-World Intuition
Think of standing outside under a clear sky holding a phone: you have a direct line-of-sight to the satellite (Path 0). However, the radio waves from the satellite also hit the asphalt street and the roof of your house, reflecting back into your phone (Paths 1 & 2). 

### User Input

I see, that's because we divide the delay axis into bins, and the paths that arrive in the same bin will be added up, right? so, the secondary paths that just bounds from the ground will still fall in the same bin of the actuall LOS path? we consider this because the NTN path is long, that bounce from ground is with much much small delay so that we consider them as the LOS paths also? 

### Planner Response

You have described the concept **almost perfectly!** 

Here is the exact explanation of how delay resolution, path combining, and the 3 distinct paths work in your simulation:

---

### 1. Delay Axis Discretization (Time Bins)

Yes, in discrete-time baseband simulations, the delay axis is divided into time bins of width $T_s$:
$$T_s = \frac{1}{\text{Bandwidth}}$$

In your code, the bandwidth is $F_s = 256 \text{ subcarriers} \times 30\text{ kHz} = 7.68\text{ MHz}$.
* **Time Bin Resolution ($T_s$):** $\frac{1}{7.68\text{ MHz}} \approx \mathbf{130\text{ ns}}$
* **Distance Resolution ($\Delta d$):** $c \cdot T_s \approx 3 \times 10^8\text{ m/s} \times 130\text{ ns} \approx \mathbf{39\text{ meters}}$

Any rays whose propagation path lengths differ by **less than 39 meters** will fall into the **same delay bin** and add up vectorially.

---

### 2. What happens to the Ground Bounce next to the UE?

You are exactly right. If the UE is at a height of 1.5 meters:
* A direct ray from the satellite to the UE (LoS) and a ray that bounces off the ground right next to the UE have a path length difference of only a few meters (e.g., $\approx 3$ meters).
* A $3\text{ m}$ path difference corresponds to a delay of only:
  $$\tau_{\text{bounce}} = \frac{3\text{ m}}{3\times 10^8\text{ m/s}} = 10\text{ ns}$$
* Since $10\text{ ns} \ll 130\text{ ns}$, **the ground bounce falls into the same bin as the direct LoS path (Bin 0)** and they are combined. 

Because the satellite link distance is so large (600 km), local reflections immediately surrounding the UE are practically identical in delay to the LoS path and merge into **Bin 0**.

---

### 3. Why do we still have Path 1 and Path 2?

If local reflections merge into Bin 0, why does the model still generate **Paths 1 and 2**?

These represent reflections from **distant physical clusters** in the suburban environment (e.g., a distant hill, a tall building, or a group of trees further away from the UE):
* These reflectors are located **farther away** (e.g., 50 to 200 meters away from the UE).
* The extra distance the reflected waves must travel is larger than 39 meters.
* Consequently, their delays are larger than $130\text{ ns}$, placing them in **separate, resolved bins (Bin 1 and Bin 2)**.

---

### Summary

1. **Direct LoS + Local Ground Bounces** $\to$ Merge into **Bin 0** (due to sub-sample delays $<130\text{ ns}$).
2. **Distant Clutter Reflections** $\to$ Fall into **Bin 1 and Bin 2** (since extra distance $>39\text{ m}$).

### User Input

but with other settings, not NTN we can not always consider like this, right?

### Planner Response

**Correct. In non-NTN (terrestrial or indoor) settings, you cannot always make this assumption.** 

In terrestrial networks (like 5G sub-6 GHz FR1, mmWave FR2, or WiFi), the physical scales and signal bandwidths change dramatically.

Here are the two main reasons why:

---

### 1. Wider Bandwidth = Higher Delay Resolution

Terrestrial networks use much wider bandwidths than satellite channels (which typically have narrow channel allocations):
* **5G sub-6 GHz (FR1):** Up to $100\text{ MHz}$ bandwidth.
  * $T_s = 10\text{ ns}$ $\to$ distance resolution is **$3\text{ meters}$**.
* **5G mmWave (FR2):** Up to $400\text{ MHz}$ bandwidth.
  * $T_s = 2.5\text{ ns}$ $\to$ distance resolution is **$0.75\text{ meters}$**.

Because the distance resolution is so fine ($0.75\text{ m}$ to $3\text{ m}$), **even local ground bounces or reflections off nearby walls will be resolved into separate delay bins (Bin 1, Bin 2, etc.)** rather than merging into the LoS path (Bin 0).

---

### 2. Angular Spread (AoA / ZoA)

* **In NTN (Satellite):** 
  Because the satellite is $600\text{ km}$ away, both the direct LoS path and a local ground reflection arrive at your phone from almost the **exact same angle** in the sky (the plane-wave approximation).
* **In Terrestrial (gNodeB on a tower):**
  If the base station is $100\text{ meters}$ away on a $25\text{ meter}$ tower:
  * The direct path arrives horizontally.
  * The ground reflection bounces upward from the road.
  * These two paths have **very different Angles of Arrival (AoA)** at your phone's antenna array, meaning they behave differently spatially even if they arrive close in time.

---

### Summary Comparison

| Setting | Delay Resolution | Local Reflections (Ground/Wall) | Angular Spread |
| :--- | :--- | :--- | :--- |
| **NTN (Satellite)** | Coarse ($\approx 39\text{ m}$) | **Merge into Bin 0** (Unresolved) | Almost identical angles |
| **Terrestrial (5G)** | Fine ($\approx 0.75 - 3\text{ m}$) | **Resolve into separate Bins** | Very different angles |

### User Input

how about setting number of NLOS paths in the code? channel_model._scenario._params_nlos["numClusters_50"] it is only consider the NLOS paths that has angular of 50 degree? we not consider any paths else that have different angular? and, that set is the fix (like always 4 NLOS paths, or it is number of NLoS is less than or equal to 4)?

### Planner Response

Here are the answers to your two questions:

---

### 1. Does `"numClusters_50"` only consider paths at exactly 50 degrees?

**No, the paths do not all have an angle of 50 degrees.**

* **What the "50" means:** 
  The `"50"` in `"numClusters_50"` represents the **nominal elevation angle of the satellite** (which you set to `elevation_angle = 50.0`). 
  The OpenNTN library rounds this elevation angle to the nearest $10^\circ$ (which is $50^\circ$) to look up the correct 3GPP TR 38.811 parameter set.
* **Do the paths have different angles?**
  **Yes, they do.** The 4 generated NLoS paths are scattered around the environment. 
  Their Angles of Arrival (AoA) and Zenith Angles of Arrival (ZoA) are randomly generated based on the **Angular Spread parameters** (like `ASA` and `ZSA`) loaded from the JSON configuration file. They will arrive at your UE from different directions, even though the satellite itself is positioned at a $50^\circ$ elevation.

---

### 2. Is the number of NLoS paths fixed, or can it be less than 4?

**It is fixed at exactly 4 paths.**

* In the OpenNTN code, the channel generator uses a mask to enable exactly `num_clusters_nlos` paths. 
* Because `numClusters_50` is set to 4, the model will generate **exactly 4 clusters/paths** for every single NLoS drop in your batch. It will never be less than 4 unless you manually change `"numClusters_50"` to a smaller number.

### User Input

can you check the TDL-NTN -A, -B, -C, -D channel model - how many paths they are considering? -D is the one with LOS ? while -A, -B, -C are NLoS? 

*Viewed [TDL-A.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-A.json) *

*Viewed [TDL-B.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-B.json) *

*Viewed [TDL-C.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-C.json) *

*Viewed [TDL-D.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-D.json) *

*Viewed [TDL-A30.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-A30.json) *

*Viewed [TDL-C300.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-C300.json) *

*Viewed [TDL-B100.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-B100.json) *

### Planner Response

Here is the breakdown of the TDL-NTN channel models configured in OpenNTN under the `OpenNTN/models/` folder:

### 1. Model Overview (LoS vs. NLoS)

* **`TDL-A`**: **NLoS** (Non-Line-of-Sight, `"los": 0`)
* **`TDL-B`**: **NLoS** (Non-Line-of-Sight, `"los": 0`)
* **`TDL-C`**: **LoS** (Line-of-Sight, `"los": 1`)
* **`TDL-D`**: **LoS** (Line-of-Sight, `"los": 1`)

---

### 2. Number of Paths and Parameters

The number of paths (resolved delay taps) defined in each JSON configuration file is as follows:

| Model Name | Link State | Number of Clusters/Paths | Nominal Delays |
| :--- | :--- | :--- | :--- |
| **[TDL-A.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-A.json)** | NLoS | **3** | `[0.0, 1.0811, 2.8416]` |
| **[TDL-B.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-B.json)** | NLoS | **4** | `[0.0, 0.7249, 0.7410, 5.7392]` |
| **[TDL-C.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-C.json)** | **LoS** | **2** | `[0.0, 0.0, 14.8124]` (Direct + 1 delay path) |
| **[TDL-D.json](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/OpenNTN/models/TDL-D.json)** | **LoS** | **3** | `[0.0, 0.0, 0.5596, 7.3340]` (Direct + 2 delay paths) |

---

### 3. Note on TDL-A30, TDL-B100, and TDL-C300 Models

In addition to the base models above, OpenNTN includes delay-spread-scaled TDL models (where delays are pre-scaled to fixed values like 30ns, 100ns, or 300ns):
* **`TDL-A30`**, **`TDL-B100`**, and **`TDL-C300`** all contain **12 paths/clusters** (`"num_clusters": 12`) and are configured as **NLoS** (`"los": 0`).

### User Input

can you check the realistic NTN TDL channel model of 3GPP to compare if these models of OpenNTN are similar?

*Searched web for 3GPP TR 38.811 TDL channel model*

*Searched web for 3GPP TR 38.811 "TDL-A" delays powers*

### Planner Response

**Yes, the TDL models in the OpenNTN library are identical to the official 3GPP TR 38.811 standard specification.**

The JSON files in `OpenNTN/models/` are direct, exact implementations of **3GPP TR 38.811 (Section 6.9.2, Table 6.9.2-1)** for Non-Terrestrial Networks.

---

### Comparison: 3GPP TR 38.811 vs. OpenNTN

Here is a side-by-side comparison of **TDL-A** and **TDL-D** to show they match the 3GPP specifications exactly:

#### 1. NTN TDL-A (NLoS Model)
| Tap # | 3GPP TR 38.811 Normalized Delay | 3GPP Power (dB) | `TDL-A.json` Delay | `TDL-A.json` Power (dB) | Match? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | $0.0000$ | $0.0$ | $0.0$ | $0.0$ | **Yes** ✅ |
| **2** | $1.0811$ | $-4.675$ | $1.0811$ | $-4.675$ | **Yes** ✅ |
| **3** | $2.8416$ | $-6.482$ | $2.8416$ | $-6.482$ | **Yes** ✅ |

---

#### 2. NTN TDL-D (LoS Model)
In the 3GPP standard, the LoS models (TDL-C and TDL-D) have **two taps at delay 0.0**. This represents:
1. The **Line-of-Sight deterministic path** (constant phase).
2. A **diffuse Rayleigh scattered path** arriving at the same time (delay 0.0), which creates Rician fading.

| Tap # | 3GPP TR 38.811 Normalized Delay | 3GPP Power (dB) | `TDL-D.json` Delay | `TDL-D.json` Power (dB) | Match? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 (LoS)** | $0.0000$ (Deterministic) | $-0.284$ | $0.0$ | $-0.284$ | **Yes** ✅ |
| **2 (diffuse)** | $0.0000$ (Rayleigh) | $-11.991$ | $0.0$ | $-11.991$ | **Yes** ✅ |
| **3** | $0.5596$ (Rayleigh) | $-9.887$ | $0.5596$ | $-9.887$ | **Yes** ✅ |
| **4** | $7.3340$ (Rayleigh) | $-16.771$ | $7.3340$ | $-16.771$ | **Yes** ✅ |

---

### Key Takeaway
You can use these OpenNTN TDL files with complete confidence. They are **not** simplified approximations; they are the **precise standard-compliant 3GPP NTN-TDL models** used in industrial simulation tools (like MATLAB's 5G Toolbox).

### User Input

but I think the 3GPP NTN TDL models are with more paths? but in the library here they just consider maximum 4 paths? they are the 4 strongest paths?

*Searched web for 3GPP TR 38.811 TDL-A number of taps*

*Searched web for "TR 38.811" "6.9.2-1"*

### Planner Response

It is a common point of confusion, but **the official 3GPP TR 38.811 NTN TDL models actually do have only 3 to 4 taps.**

You are likely thinking of the **terrestrial** (on-ground) 3GPP TR 38.901 TDL models, which have 23 or 24 taps. For satellite (NTN) channels, the standard intentionally keeps the number of taps very small (3 to 4) due to the physics of space propagation:

---

### Why NTN TDL Models Have Fewer Paths

1. **High Elevation Angles & Sparse Environments:**
   * In a terrestrial link, the base station is on a tower and the signal travels horizontally through streets, bouncing off hundreds of buildings. This creates a dense multipath profile with **23+ resolved delay taps**.
   * In a satellite link, the satellite is high in the sky. The signal travels through empty space and the atmosphere with zero reflections. Reflections only occur in the last few meters when the signal hits the ground or nearby houses near the UE.
   * This results in a **very sparse multipath profile**, where only 3 or 4 path reflections have enough energy to be detected.

2. **Standard-Compliant Optimization:**
   * 3GPP TR 38.811 defined the baseline NTN TDL-A, B, C, and D models with **exactly 3 to 4 taps** to reflect this physical reality and reduce simulation complexity. These are not just the "strongest 4" cut from a larger list; they are the complete standard model.

---

### What if you need more paths? (The 12-Tap Models)

If you are simulating a **dense urban area** under satellite coverage (where many tall buildings create more reflections), 3GPP provides the **12-tap models** (e.g., TDL-A30, TDL-B100, TDL-C300):

* These 12-tap models (like `TDL-A30.json` in your directory) are scaled to represent more severe frequency-selective scattering.
* Every one of these models uses **exactly 12 taps**, which is a compressed version of the 23-tap terrestrial models.

### Summary
* **Typical NTN (Suburban/Open):** Use `TDL-A` / `TDL-B` / `TDL-C` / `TDL-D` (**3 to 4 taps**). This is realistic for satellite links.
* **Severe Scattering NTN (Urban Canyons):** Use `TDL-A30` / `TDL-B100` / `TDL-C300` (**12 taps**).