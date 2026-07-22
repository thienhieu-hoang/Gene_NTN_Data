import os
from sionna.phy.channel import GenerateOFDMChannel
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.interpolate import interp1d

from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# These functions also exist in sionna.channel.tr38901 but are not compatable with 3GPP TR38.811
from sionna.phy.channel.tr38901 import Antenna, AntennaArray


# Import the NTN channel models from the local OpenNTN package
import sys
# Resolve the project root directory relative to this script's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OpenNTN import DenseUrban, Urban, SubUrban
from OpenNTN.utils import gen_single_sector_topology as gen_ntn_topology

scenario = "dur" # dur, sur, urb
carrier_frequency = 27e9 # DL 
direction = "downlink"
elevation_angle = 50.0
num_ut = 1
satellite_height = 600000.0 # Height in meters, this is a satellite in the Low Earth Orbit (LEO)
batch_size = 32 # Number of topologies we will generate later  

ut_array = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=carrier_frequency)

# The satellite is the basestation, so we name it bs. 
bs_array = AntennaArray(num_rows=1,
                        num_cols=1,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
N_samples = 128

num_streams_per_tx = 1 
num_time_steps = 14
RBBlock = 11
nFFT = int(2** (np.ceil(np.log2(RBBlock * 12)))) 
SCS = 60e3
sampling_frequency = SCS * nFFT # set equal to waveformInfo.SampleRate in MATLAB
        # maybe different because in MATLAB, number of subcariers is different to the Nfft

rg = ResourceGrid(num_ofdm_symbols=14,
                    fft_size=nFFT, # 
                    subcarrier_spacing=SCS,
                    num_tx=num_ut,
                    num_streams_per_tx=num_streams_per_tx,
                    cyclic_prefix_length=26,
                    num_guard_carriers=(62, 62),     # 256 - 132 = 124 zeroed carriers
                    dc_null=False, 
                    pilot_pattern="kronecker",
                    pilot_ofdm_symbol_indices=[2,7,11])

# Function that calculates the subcarrier frequencies of the OFDM frame
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)

from OpenNTN.utils import compute_stallite_doppler as compute_stallite_doppler 

from OpenNTN.utils import cir_to_time_channel, time_lag_discrete_time_channel
from sionna.phy.ofdm import RemoveNulledSubcarriers
bandwidth = rg.fft_size * rg.subcarrier_spacing

# Set a constant UE speed of 30 m/s (approx. 108 km/h)
ue_speed = 20.0 

def interpolate_channel(rx_grid_b, tx_grid_b, pilot_mask):
    # rx_grid_b: [14, 132]
    # tx_grid_b: [14, 132]
    # pilot_mask: [14, 132] bool
    
    # LS estimation at pilot positions
    h_est = np.zeros_like(rx_grid_b)
    h_est[pilot_mask] = rx_grid_b[pilot_mask] / tx_grid_b[pilot_mask]
    
    pilot_coords = np.where(pilot_mask)
    pilot_symbols = np.unique(pilot_coords[0]) # e.g. [2, 7, 11]
    
    # 1. Frequency interpolation for each pilot symbol
    h_freq_interp = np.zeros_like(rx_grid_b)
    for m in pilot_symbols:
        pilots_in_sym = np.where(pilot_mask[m])[0]
        if len(pilots_in_sym) > 1:
            f_interp = interp1d(pilots_in_sym, h_est[m, pilots_in_sym], kind='linear', fill_value='extrapolate')
            h_freq_interp[m, :] = f_interp(np.arange(132))
        elif len(pilots_in_sym) == 1:
            h_freq_interp[m, :] = h_est[m, pilots_in_sym[0]]
            
    # 2. Time interpolation for all subcarriers
    h_interp = np.zeros_like(rx_grid_b)
    for k in range(132):
        t_interp = interp1d(pilot_symbols, h_freq_interp[pilot_symbols, k], kind='linear', fill_value='extrapolate')
        h_interp[:, k] = t_interp(np.arange(14))
        
    return h_interp

H_eff_all = []
H_LS_est_all = []
H_interp_all = []
delay_spreads_all = []

SNR_dB = 20.0
SNR_linear = 10.0 ** (SNR_dB / 10.0)

# We want QPSK symbols as the data and pilot symbols
qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / np.sqrt(2)

pilot_mask = tf.squeeze(rg.pilot_pattern.mask).numpy() # shape [14, 132]

for i in range(N_samples // batch_size):
    # Map the scenario string to the corresponding class
    scenario_classes = {
        "dur": DenseUrban,
        "sur": SubUrban,
        "urb": Urban
    }
    channel_class = scenario_classes[scenario]
    
    channel_model = channel_class(carrier_frequency=carrier_frequency,
                                  ut_array=ut_array,
                                  bs_array=bs_array,
                                  direction=direction,
                                  elevation_angle=elevation_angle,
                                  doppler_enabled=False)
    # NOTE: Satellite-to-beam-center Doppler (common frequency shift) and UE-to-beam-center Doppler (local scattering/fading) are decoupled in the channel coefficients generator.
    # Setting doppler_enabled=False disables the satellite orbit Doppler component, making it equivalent to the effective channel experienced after perfect transmitter Doppler pre-compensation.

    channel_model._scenario._params_nlos["numClusters_50"] = 3

    # Generate the topology
    topology = gen_ntn_topology(batch_size=batch_size,
                            num_ut=num_ut, scenario=scenario,
                            bs_height=satellite_height,
                            min_ut_velocity=ue_speed,
                            max_ut_velocity=ue_speed)

    # Set the topology
    channel_model.set_topology(*topology)

    # path_coefficients [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # path_delays [batch size, num_rx, num_tx, num_paths]
    num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
    path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)

    # Calculate RMS delay spread for each topology in the batch
    coefs_np = np.abs(path_coefficients.numpy())**2
    delays_np = path_delays.numpy()
    for b in range(batch_size):
        # Average power of each path over all antennas and time steps
        p_b = np.mean(coefs_np[b], axis=(0, 1, 2, 3, 5)) # shape: [num_paths]
        tau_b = delays_np[b, 0, 0, :] # shape: [num_paths]
        
        sum_p = np.sum(p_b)
        if sum_p > 0:
            mean_tau = np.sum(p_b * tau_b) / sum_p
            rms_ds = np.sqrt(np.sum(p_b * (tau_b - mean_tau)**2) / sum_p)
            delay_spreads_all.append(rms_ds)
    
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
    h_freq = ofdm_channel() # shape [batch_size, 1, 1, 1, 2, 14, 256]
    
    remove_nulled = RemoveNulledSubcarriers(rg)
    h_eff = remove_nulled(h_freq)   # shape [batch_size, 1, 1, 1, 2, 14, 132]
    
    # Extract SISO channel for the first BS antenna to UT antenna
    h_eff_siso = h_eff[:, 0, 0, 0, 0, :, :].numpy() # shape [batch_size, 14, 132]
    H_eff_all.append(h_eff_siso)

    # Generate tx_grid with QPSK symbols
    tx_grid = np.random.choice(qpsk_symbols, size=[batch_size, 14, 132])

    # Pass through the channel and add complex white Gaussian noise
    rx_signal_clean = tx_grid * h_eff_siso
    sig_power = np.mean(np.abs(rx_signal_clean) ** 2)
    noise_var = sig_power / SNR_linear
    noise = (np.random.randn(*h_eff_siso.shape) + 1j * np.random.randn(*h_eff_siso.shape)) * np.sqrt(noise_var / 2.0)
    rx_grid = rx_signal_clean + noise

    # Least-Squares estimation
    h_LS = rx_grid / tx_grid # shape [batch_size, 14, 132]
    
    # Extract only the LS estimates at the pilot positions
    pilot_symbols, pilot_subcarriers = np.where(pilot_mask)
    h_LS_pilots = h_LS[:, pilot_symbols, pilot_subcarriers] # shape [batch_size, num_pilots]
    H_LS_est_all.append(h_LS_pilots)

    # Linear interpolation
    h_interp_batch = []
    for b in range(batch_size):
        h_interp_b = interpolate_channel(rx_grid[b], tx_grid[b], pilot_mask)
        h_interp_batch.append(h_interp_b)
    h_interp_batch = np.stack(h_interp_batch, axis=0) # [batch_size, 14, 132]
    H_interp_all.append(h_interp_batch)

# Concatenate all batches
H_eff_total = np.concatenate(H_eff_all, axis=0)      # [N_samples, 14, 132]
H_LS_total = np.concatenate(H_LS_est_all, axis=0)    # [N_samples, num_pilots]
H_interp_total = np.concatenate(H_interp_all, axis=0) # [N_samples, 14, 132]

pilot_symbols, pilot_subcarriers = np.where(pilot_mask)

# Save to dynamically named scenario directory inside generatedChannel
script_dir = os.path.dirname(os.path.abspath(__file__))
base_output_dir = os.path.join(script_dir, "generatedChannel")
dir_name = f"{scenario.upper()}_{int(carrier_frequency/1e9)}G_{int(satellite_height/1000)}km_{int(ue_speed)}ms_{int(SNR_dB)}dB"
output_dir = os.path.join(base_output_dir, dir_name)
os.makedirs(output_dir, exist_ok=True)

mat_filename = os.path.join(output_dir, f"channel_{scenario}.mat")
mat_data = {
    "H_eff": H_eff_total,
    "H_LS_values": H_LS_total,
    "pilot_symbols": pilot_symbols + 1,       # Convert to 1-indexed for MATLAB compatibility
    "pilot_subcarriers": pilot_subcarriers + 1, # Convert to 1-indexed for MATLAB compatibility
    "H_interp": H_interp_total
}
savemat(mat_filename, mat_data)
print(f"Saved channel simulation data to {mat_filename}")

# Calculate average RMS delay spread
avg_delay_spread_ns = np.mean(delay_spreads_all) * 1e9 if len(delay_spreads_all) > 0 else 0.0

# Save the scenario configurations and delay spread to a markdown note file
md_filename = os.path.join(output_dir, f"readme_{scenario}.md")
md_content = f"""# Channel Generation Settings - {scenario.upper()}

- **Scenario Type**: {scenario.upper()} (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: {carrier_frequency / 1e9:.2f} GHz
- **Link Direction**: {direction}
- **Satellite Elevation Angle**: {elevation_angle} degrees
- **Satellite (LEO) Height**: {satellite_height / 1000:.0f} km
- **UE Ground Speed**: {ue_speed} m/s
- **Subcarrier Spacing (SCS)**: {SCS / 1e3:.0f} kHz
- **FFT Size**: {nFFT}
- **Active Subcarriers**: 132 (out of {nFFT})
- **SNR (for LS estimation)**: {SNR_dB} dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 7, 11]
- **Total Samples Generated**: {N_samples}
- **Average RMS Delay Spread**: {avg_delay_spread_ns:.2f} ns
"""

with open(md_filename, "w") as f:
    f.write(md_content)
print(f"Saved scenario parameters documentation to {md_filename}")

    
