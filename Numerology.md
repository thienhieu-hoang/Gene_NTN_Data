The technical details regarding the frequency bands and OFDM numerologies used for Low Earth Orbit (LEO) satellites, as standardized under **5G NR Non-Terrestrial Networks (NTN)** (3GPP Release 17 and beyond), are detailed below.

### 1. Which Carrier Frequencies can a LEO Satellite Use?

3GPP categorizes the frequency spectrum for 5G satellite communications into two primary ranges based on the terminal type and regulatory authorization:

#### A. Frequency Range 1 (FR1) – Below 6 GHz (Direct-to-Device / Handset)

These frequencies feature lower path attenuation and are optimized for direct-to-device scenarios (smartphones, IoT sensors, and handheld terminals):

- **S-band (Band n255):** Nominal carrier frequency around **2 GHz** (Uplink: 1980–2010 MHz, Downlink: 2170–2200 MHz).
    
- **L-band (Band n256):** Nominal carrier frequency around **1.5 GHz** (typically used for narrow-band or mass IoT connectivity).
    
- _Note:_ These bands operate primarily using **FDD (Frequency Division Duplexing)** to avoid timing overlap over the long round-trip propagation delays. Channel bandwidths are typically smaller (e.g., **5 MHz to 20 MHz**).
    

#### B. Frequency Range 2 (FR2-0 / FR2-1 / FR2) – Above 10 GHz (VSAT / Fixed Terminals)

For high-throughput broadband (fixed terminals, enterprise VSAT, moving platforms like planes/ships), higher frequency bands with large chunks of spectrum are assigned:

- **Ka-band (Bands n510, n511, n512 / FR2-0):** The highest priority band for broadband backhaul and VSAT.
    
    - **Downlink:** ~17.7 GHz to 20.2 GHz (centered around **20 GHz**).
        
    - **Uplink:** ~27.5 GHz to 30.0 GHz (centered around **30 GHz**).
        
- **Ku-band / Other FR2 bands:** Frequencies stretching from 26 GHz up to 48 GHz are also designated for high-capacity satellite links. Channel bandwidths here can scale much higher (e.g., **100 MHz to 400 MHz**).
    

### 2. OFDM Numerology for LEO Satellite Frequencies

Unlike stable terrestrial cell towers, LEO satellites move at high orbital velocities (~7.5 km/s), inducing a massive **Doppler shift** (e.g., up to $\pm 50\text{ kHz}$ at 2 GHz and much worse at 20/30 GHz). To withstand this severe frequency impairment, the Subcarrier Spacing (SCS) must be chosen carefully to minimize inter-subcarrier interference (ICI).

3GPP defines the multi-numerology structure where $\Delta f = 15 \times 2^\mu \text{ kHz}$. Depending on the chosen carrier frequency, the following configurations are utilized:

#### A. Numerology Configuration for FR1 (S/L-band @ ~1.5 GHz – 2 GHz)

For lower frequencies, 3GPP supports standard terrestrial configurations but shifts toward wider spacing for higher Doppler tolerance:

- **Subcarrier Spacing (SCS):** * **15 kHz** ($\mu = 0$) or **30 kHz** ($\mu = 1$).
    
    - **60 kHz** ($\mu = 2$) can be deployed for highly dynamic channels.
        
- **Number of Subcarriers:** * In 5G NR, resources are allocated in Resource Blocks (RBs), where **1 RB = 12 subcarriers**.
    
    - For a typical **5 MHz** profile at 15 kHz SCS: **25 RBs** are utilized $\rightarrow 25 \times 12 = \mathbf{300 \text{ subcarriers}}$.
        
    - For a maximum **20 MHz** profile at 30 kHz SCS: **51 RBs** are utilized $\rightarrow 51 \times 12 = \mathbf{612 \text{ subcarriers}}$.
        
- **Cyclic Prefix (CP):** Normal CP. (Doppler pre-compensation is performed dynamically by the User Equipment using GNSS/Ephemeris coordinates to map out the shift before it breaks the SCS spacing).
    

#### B. Numerology Configuration for Ka-band / FR2-0 (Broadband @ 20 GHz – 30 GHz)

At Ka-band frequencies, a narrow 15 kHz spacing would be entirely crushed by the Doppler variance. Therefore, wider spacing is mandated:

- **Subcarrier Spacing (SCS):** * **60 kHz** ($\mu = 2$) or **120 kHz** ($\mu = 3$).
    
- **Number of Subcarriers:**
    
    - With vast channel chunks up to **400 MHz bandwidth**, a 60 kHz SCS uses large resource grids.
        
    - For a **100 MHz** profile at 60 kHz SCS: **132 RBs** $\rightarrow 132 \times 12 = \mathbf{1,584 \text{ subcarriers}}$.
        
    - For a **400 MHz** profile at 120 kHz SCS: **275 RBs** $\rightarrow 275 \times 12 = \mathbf{3,300 \text{ subcarriers}}$.
        
- **Cyclic Prefix (CP):** * **Extended Cyclic Prefix (ECP)** is strictly supported for the **60 kHz SCS** scenario. ECP provides a significantly longer temporal guard interval per OFDM symbol, which absorbs the massive differential propagation path delays encountered across large satellite beam footprints on Earth.
    

### Summary Mapping Table

|**Parameter**|**FR1 (Direct Handset/IoT)**|**Ka-Band (VSAT/Broadhaul)**|
|---|---|---|
|**Typical Carrier Freq.**|~1.5 GHz (L-band) / ~2.1 GHz (S-band)|~20 GHz (DL) / ~30 GHz (UL)|
|**Duplex Mode**|FDD|FDD (or high-frequency TDD variants)|
|**Channel Bandwidth**|5 MHz – 20 MHz|100 MHz – 400 MHz|
|**Supported SCS**|**15 kHz, 30 kHz**, 60 kHz|**60 kHz, 120 kHz**|
|**Cyclic Prefix (CP)**|Normal CP|Extended CP (relevant for 60 kHz)|
|**Active Subcarriers**|~300 to 1,068 subcarriers|~1,500 to 3,300+ subcarriers|