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

from sionna.phy.ofdm import RemoveNulledSubcarriers, ResourceGrid
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# These functions also exist in sionna.channel.tr38901 but are not compatable with 3GPP TR38.811
from sionna.phy.channel.tr38901 import Antenna, AntennaArray


# Import the NTN channel models from the local OpenNTN package
import sys
sys.path.append('../')  # Add parent directory to path to access OpenNTN
from OpenNTN import DenseUrban, Urban, SubUrban
from OpenNTN.utils import cir_to_time_channel, gen_single_sector_topology as gen_ntn_topology, time_lag_discrete_time_channel

scenario = "dur" # dur is the DenseUrban scenario
carrier_frequency = 2.18e9 # DL S-Band
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
N_samples = 2048
H_perfect = [] # to be in size of [14,132,N]

num_streams_per_tx = 1 
num_time_steps = 14
RBBlock = 11
nFFT = 2** (np.ceil(np.log2(RBBlock * 12))) 
SCS = 30e3
sampling_frequency = SCS * nFFT # set equal to waveformInfo.SampleRate in MATLAB
        # maybe different because in MATLAB, number of subcariers is different to the Nfft

rg = ResourceGrid(num_ofdm_symbols=14,
                    fft_size=nFFT, # 
                    subcarrier_spacing=SCS,
                    num_tx=num_ut,
                    num_streams_per_tx=num_streams_per_tx,
                    cyclic_prefix_length=40,
                    num_guard_carriers=(62, 62),     # 256 - 132 = 124 zeroed carriers
                    dc_null=False, 
                    pilot_pattern="kronecker",
                    pilot_ofdm_symbol_indices=[2,11])

# Function that calculates the subcarrier frequencies of the OFDM frame
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)

from OpenNTN.utils import compute_stallite_doppler as compute_stallite_doppler 
bandwidth = rg.fft_size * rg.subcarrier_spacing

for i in range(N_samples//(batch_size *2)):
    # Here we match choose DenseUrban to match the parameter "dur" for the scenario defined above
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
    