import os
import sys
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Configure the system to use only a single GPU and allocate memory dynamically
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import subcarrier_frequencies
from sionna.phy.channel.tr38901 import Antenna, AntennaArray

# Resolve the project root directory relative to this script's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OpenNTN import DenseUrban_modify, Urban_modify, SubUrban_modify
from helper import ecef_to_enu, get_satellite_state_ecef, get_local_earth_patch, save_simulation_readme
from OpenNTN.utils import compute_satellite_speed
from OpenNTN.utils import cir_to_time_channel, time_lag_discrete_time_channel
from OpenNTN.utils import compute_stallite_doppler as compute_stallite_doppler
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.ofdm import RemoveNulledSubcarriers

SPEED_OF_LIGHT = 299792458.0

# =========================================================================
# 1. GEOMETRY CONFIGURATION PARAMETERS (from visualize_geometry.py)
# =========================================================================
phi_UE_deg = 37.7749         # UE Initial Latitude (degrees)
lambda_UE_deg = -122.4194    # UE Initial Longitude (degrees)
h_UE = 100.0                 # UE Altitude (m)
satellite_height = 600000.0  # LEO Orbit altitude (m) (600 km)
inclination_deg = 55.0       # Orbit inclination (degrees)
ue_speed = 50.0              # UE speed along the ground (m/s)
heading_deg = 45.0           # UE heading azimuth (degrees, 0=North, 90=East)

# Physical constants
omega_E = 7.292115e-5        # Earth's rotation rate (rad/s)
mu = 3.986004418e14          # Earth's gravitational parameter (m^3/s^2)
a_wgs84 = 6378137.0          # WGS-84 semi-major axis (m)
e2 = 6.69437999e-3           # First eccentricity squared

# Convert to radians
inclination = np.deg2rad(inclination_deg)
phi_UE = np.deg2rad(phi_UE_deg)
lambda_UE = np.deg2rad(lambda_UE_deg)
heading = np.deg2rad(heading_deg)

# Orbit radius & speed
r_orbit = a_wgs84 + satellite_height
omega_s = np.sqrt(mu / r_orbit**3)
v_sat_orbit = np.sqrt(mu / r_orbit)

# UE Initial ECEF position
N_phi_0 = a_wgs84 / np.sqrt(1.0 - e2 * np.sin(phi_UE)**2)
r_ue_ECEF_0 = np.array([
    (N_phi_0 + h_UE) * np.cos(phi_UE) * np.cos(lambda_UE),
    (N_phi_0 + h_UE) * np.cos(phi_UE) * np.sin(lambda_UE),
    (N_phi_0 * (1.0 - e2) + h_UE) * np.sin(phi_UE)
])

# UE local velocity in ENU
v_UE_ENU = np.array([ue_speed * np.sin(heading), ue_speed * np.cos(heading), 0.0])

# Rotation matrix from local ENU to ECEF at UE's initial location
R_ENU2ECEF = np.array([
    [-np.sin(lambda_UE), -np.sin(phi_UE) * np.cos(lambda_UE), np.cos(phi_UE) * np.cos(lambda_UE)],
    [ np.cos(lambda_UE), -np.sin(phi_UE) * np.sin(lambda_UE), np.cos(phi_UE) * np.sin(lambda_UE)],
    [ 0.0,                np.cos(phi_UE),                 np.sin(phi_UE)]
])
R_ECEF2ENU = R_ENU2ECEF.T

# UE velocity in ECEF
ue_vel_ECEF = R_ENU2ECEF @ v_UE_ENU

# Orbit alignment (closest approach at t=0)
if inclination >= abs(phi_UE):
    u_mid = np.arcsin(np.sin(phi_UE) / np.sin(inclination))
else:
    u_mid = np.sign(phi_UE) * np.pi / 2.0
Omega_RAAN = lambda_UE - np.arctan2(np.sin(u_mid) * np.cos(inclination), np.cos(u_mid))

# Calculate initial state at snapshot t = 0 (nominal reference)
t0 = 0.0
r_sat_ECEF_0, v_sat_ECEF_0 = get_satellite_state_ecef(
    t0, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E
)
v_los_0 = r_sat_ECEF_0 - r_ue_ECEF_0
slant_range_0 = np.linalg.norm(v_los_0)
u_normal_0 = r_ue_ECEF_0 / np.linalg.norm(r_ue_ECEF_0)
u_los_0 = v_los_0 / slant_range_0
elev_rad_0 = np.arcsin(np.dot(u_normal_0, u_los_0))
elevation_angle_nom = float(np.degrees(elev_rad_0))

# =========================================================================
# 2. SYSTEM SIMULATION CONFIGURATION & DATASETS CONFIGURATION
# =========================================================================
scenario = "dur"             # dur (Dense Urban), sur (SubUrban), urb (Urban)
carrier_frequency = 27e9     # DL carrier frequency (Hz)
direction = "downlink"
num_ut = 1
SNR_dB = 20.0
SNR_linear = 10.0 ** (SNR_dB / 10.0)
r_beam = 15000.0                 # 15 km beam footprint radius

# Total sequential slots (N_samples) to generate
N_samples = 16
# Mini-batch size for GPU execution efficiency and memory safety
batch_size = 8

# ----------------- Resource Grid Setup -----------------
ut_array = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=carrier_frequency)

bs_array = AntennaArray(num_rows=1,
                        num_cols=1,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

num_streams_per_tx = 1 
RBBlock = 11
nFFT = int(2** (np.ceil(np.log2(RBBlock * 12)))) 
SCS = 60e3
sampling_frequency = SCS * nFFT
bandwidth = nFFT * SCS

rg = ResourceGrid(num_ofdm_symbols=14,
                    fft_size=nFFT,
                    subcarrier_spacing=SCS,
                    num_tx=num_ut,
                    num_streams_per_tx=num_streams_per_tx,
                    cyclic_prefix_length=26,
                    num_guard_carriers=(62, 62),
                    dc_null=False, 
                    pilot_pattern="kronecker",
                    pilot_ofdm_symbol_indices=[2, 7, 11])

frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
pilot_mask = tf.squeeze(rg.pilot_pattern.mask).numpy()
pilot_symbols, pilot_subcarriers = np.where(pilot_mask)
qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / np.sqrt(2)

# Calculate slot duration
num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
slot_duration = num_time_steps / sampling_frequency
print(f"Sampling Frequency: {sampling_frequency/1e6:.3f} MHz, Slot duration: {slot_duration*1e3:.6f} ms")

# ----------------- Precompute Sequential Geometry States -----------------
t_all = np.arange(N_samples) * slot_duration

# Arrays for accumulating coordinates and velocities in ENU
ut_loc_ENU_all = np.zeros((N_samples, 3))
bs_loc_ENU_all = np.zeros((N_samples, 3))
v_sat_ENU_all = np.zeros((N_samples, 3))
v_ue_ENU_all = np.tile(v_UE_ENU[np.newaxis, :], (N_samples, 1))

slant_ranges_all = np.zeros(N_samples)
elevation_angles_all = np.zeros(N_samples)
sat_speeds_all = np.zeros(N_samples)

r_sat_ECEF_all = np.zeros((N_samples, 3))
r_ue_ECEF_all = np.zeros((N_samples, 3))
v_sat_ECEF_all = np.zeros((N_samples, 3))

z_val = 1.5

for i in range(N_samples):
    t_n = t_all[i]
    # LEO state in ECEF
    r_sat, v_sat = get_satellite_state_ecef(
        t_n, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E
    )
    r_sat_ECEF_all[i] = r_sat
    v_sat_ECEF_all[i] = v_sat
    sat_speeds_all[i] = np.linalg.norm(v_sat)
    
    # UE state in ECEF
    r_ue = r_ue_ECEF_0 + ue_vel_ECEF * t_n
    r_ue = r_ue * (np.linalg.norm(r_ue_ECEF_0) / np.linalg.norm(r_ue))
    r_ue_ECEF_all[i] = r_ue
    
    # Slant range and elevation angle
    v_los = r_sat - r_ue
    slant_ranges_all[i] = np.linalg.norm(v_los)
    u_normal = r_ue / np.linalg.norm(r_ue)
    u_los = v_los / slant_ranges_all[i]
    elev_rad = np.arcsin(np.dot(u_normal, u_los))
    elevation_angles_all[i] = np.degrees(elev_rad)
    
    # ENU transformation (relative to reference origin r_ue_ECEF_0)
    ut_loc_ENU_all[i] = ecef_to_enu(r_ue, r_ue_ECEF_0, lambda_UE, phi_UE)
    ut_loc_ENU_all[i, 2] = z_val  # force vertical antenna height to standard 1.5m
    bs_loc_ENU_all[i] = ecef_to_enu(r_sat, r_ue_ECEF_0, lambda_UE, phi_UE)
    v_sat_ENU_all[i] = ecef_to_enu(v_sat, np.zeros(3), lambda_UE, phi_UE)

# =========================================================================
# 3. HELPER INTERPOLATOR
# =========================================================================
def interpolate_channel(rx_grid_b, tx_grid_b, pilot_mask):
    h_est = np.zeros_like(rx_grid_b)
    h_est[pilot_mask] = rx_grid_b[pilot_mask] / tx_grid_b[pilot_mask]
    
    pilot_coords = np.where(pilot_mask)
    pilot_symbols = np.unique(pilot_coords[0])
    
    h_freq_interp = np.zeros_like(rx_grid_b)
    for m in pilot_symbols:
        pilots_in_sym = np.where(pilot_mask[m])[0]
        if len(pilots_in_sym) > 1:
            f_interp = interp1d(pilots_in_sym, h_est[m, pilots_in_sym], kind='linear', fill_value='extrapolate')
            h_freq_interp[m, :] = f_interp(np.arange(132))
        elif len(pilots_in_sym) == 1:
            h_freq_interp[m, :] = h_est[m, pilots_in_sym[0]]
            
    h_interp = np.zeros_like(rx_grid_b)
    for k in range(132):
        t_interp = interp1d(pilot_symbols, h_freq_interp[pilot_symbols, k], kind='linear', fill_value='extrapolate')
        h_interp[:, k] = t_interp(np.arange(14))
        
    return h_interp

def compute_complex_ssim(h_true, h_est):
    def ssim_real_2d(x, y):
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        cov_xy = np.mean((x - mu_x) * (y - mu_y))
        
        val_max = max(np.max(x), np.max(y))
        val_min = min(np.min(x), np.min(y))
        L = val_max - val_min if val_max != val_min else 1.0
        
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        
        num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
        return num / den

    ssim_r = ssim_real_2d(np.real(h_true), np.real(h_est))
    ssim_i = ssim_real_2d(np.imag(h_true), np.imag(h_est))
    return (ssim_r + ssim_i) / 2.0

# Initialize openNTN Channel Model
scenario_classes = {
    "dur": DenseUrban_modify,
    "sur": SubUrban_modify,
    "urb": Urban_modify
}
channel_class = scenario_classes[scenario]
channel_model = channel_class(carrier_frequency=carrier_frequency,
                              ut_array=ut_array,
                              bs_array=bs_array,
                              direction=direction,
                              elevation_angle=elevation_angle_nom, # nominal for parameter lookup
                              doppler_enabled=True,
                              doppler_mode='full')

# Override clusters based on elevation angle (rounded to nearest 10 degrees)
rounded_elev = int(round(elevation_angle_nom / 10.0) * 10)
rounded_elev = max(10, min(90, rounded_elev))
channel_model._scenario._params_nlos[f"numClusters_{rounded_elev}"] = 3

ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
remove_nulled = RemoveNulledSubcarriers(rg)
channel_seed = 42

# Setup outputs directory
script_dir = os.path.dirname(os.path.abspath(__file__))
setting_dir = f"{scenario.upper()}_{int(carrier_frequency/1e9)}G_{int(satellite_height/1000)}km_{int(ue_speed)}mps_sequential"
output_dir = os.path.join(script_dir, "results", setting_dir, f"{int(SNR_dB)}dB")
os.makedirs(output_dir, exist_ok=True)

# =========================================================================
# 4. MINI-BATCH CHANNEL GENERATION & ESTIMATION
# =========================================================================
H_eff_full_list = []
H_eff_comp_list = []
H_LS_full_list = []
H_LS_comp_list = []
H_interp_full_list = []
H_interp_comp_list = []
delay_spreads_all = []
nmse_ls_list = []
nmse_ls_pilot_list = []
nmse_prac_list = []
nmse_li_list = []
ssim_ls_list = []
ssim_prac_list = []
ssim_li_list = []

num_batches = int(np.ceil(N_samples / batch_size))
print(f"Starting batched sequential generation of {N_samples} slots (Batch Size: {batch_size}, Total Batches: {num_batches})...")

for b in range(num_batches):
    start_idx = b * batch_size
    end_idx = min(start_idx + batch_size, N_samples)
    current_batch_size = end_idx - start_idx
    
    # 1. Slice and build tensors for current batch
    ut_loc_batch = ut_loc_ENU_all[start_idx:end_idx]
    bs_loc_batch = bs_loc_ENU_all[start_idx:end_idx]
    ut_vel_batch = v_ue_ENU_all[start_idx:end_idx]
    bs_vel_batch = v_sat_ENU_all[start_idx:end_idx]
    
    ut_loc_tensor = tf.constant(ut_loc_batch[:, np.newaxis, :], dtype=tf.float32)
    bs_loc_tensor = tf.constant(bs_loc_batch[:, np.newaxis, :], dtype=tf.float32)
    
    ut_orientations = tf.zeros([current_batch_size, 1, 3])
    bs_orientations = tf.zeros([current_batch_size, 1, 3])
    
    ut_velocities_tensor = tf.constant(ut_vel_batch[:, np.newaxis, :], dtype=tf.float32)
    bs_velocities_tensor = tf.constant(bs_vel_batch[:, np.newaxis, :], dtype=tf.float32)
    
    in_state = tf.constant(np.zeros((current_batch_size, 1), dtype=bool), dtype=tf.bool)
    
    # 2. Update Channel Model Topology
    channel_model.set_topology(ut_loc_tensor, bs_loc_tensor, ut_orientations, bs_orientations,
                               ut_velocities_tensor, bs_velocities_tensor, in_state, los=True)
    
    # Set the beam center to the local ENU origin (shape [current_batch_size, 3])
    channel_model._scenario.set_beam_center(tf.zeros([current_batch_size, 3], dtype=tf.float32))
    
    # Unique iteration seed for this batch, ensuring alignment of full and comp
    iteration_seed = channel_seed + b
    
    # ------------------ Part 1: Full/Uncompensated Channel ------------------
    channel_model._scenario._doppler_mode = 'full'
    tf.random.set_seed(iteration_seed)
    path_coefficients_full, path_delays_full = channel_model(num_time_steps, sampling_frequency)
    
    # Record RMS delay spreads for this batch
    coefs_np = np.abs(path_coefficients_full.numpy())**2
    delays_np = path_delays_full.numpy()
    for batch_i in range(current_batch_size):
        p_b = np.mean(coefs_np[batch_i], axis=(0, 1, 2, 3, 5))
        tau_b = delays_np[batch_i, 0, 0, :]
        sum_p = np.sum(p_b)
        if sum_p > 0:
            mean_tau = np.sum(p_b * tau_b) / sum_p
            rms_ds = np.sqrt(np.sum(p_b * (tau_b - mean_tau)**2) / sum_p)
            delay_spreads_all.append(rms_ds)
            
    tf.random.set_seed(iteration_seed)
    h_freq_full = ofdm_channel()
    h_eff_full = remove_nulled(h_freq_full)
    h_eff_siso_full = h_eff_full[:, 0, 0, 0, 0, :, :].numpy()  # shape [current_batch_size, 14, 132]
    
    # ------------------ Part 2: Precompensated Channel ------------------
    channel_model._scenario._doppler_mode = 'precompensated'
    tf.random.set_seed(iteration_seed)
    path_coefficients_comp, path_delays_comp = channel_model(num_time_steps, sampling_frequency)
    
    tf.random.set_seed(iteration_seed)
    h_freq_comp = ofdm_channel()
    h_eff_comp = remove_nulled(h_freq_comp)
    h_eff_siso_comp = h_eff_comp[:, 0, 0, 0, 0, :, :].numpy()  # shape [current_batch_size, 14, 132]
    
    # ------------------ Part 3: Channel Estimation & Interpolation ------------------
    np.random.seed(iteration_seed)
    tx_grid = np.random.choice(qpsk_symbols, size=[current_batch_size, 14, 132])
    
    np.random.seed(iteration_seed + 10)
    noise_pattern = (np.random.randn(*h_eff_siso_full.shape) + 1j * np.random.randn(*h_eff_siso_full.shape))
    
    # Full Doppler LS estimation
    rx_signal_clean_full = tx_grid * h_eff_siso_full
    sig_power_full = np.mean(np.abs(rx_signal_clean_full) ** 2, axis=(1, 2), keepdims=True)
    noise_var_full = sig_power_full / SNR_linear
    noise_realized_full = noise_pattern * np.sqrt(noise_var_full / 2.0)
    rx_grid_full = rx_signal_clean_full + noise_realized_full
    
    h_LS_full = rx_grid_full / tx_grid
    h_LS_pilots_full = h_LS_full[:, pilot_symbols, pilot_subcarriers]
    
    # Precompensated LS estimation
    rx_signal_clean_comp = tx_grid * h_eff_siso_comp
    sig_power_comp = np.mean(np.abs(rx_signal_clean_comp) ** 2, axis=(1, 2), keepdims=True)
    noise_var_comp = sig_power_comp / SNR_linear
    noise_realized_comp = noise_pattern * np.sqrt(noise_var_comp / 2.0)
    rx_grid_comp = rx_signal_clean_comp + noise_realized_comp
    
    h_LS_comp = rx_grid_comp / tx_grid
    h_LS_pilots_comp = h_LS_comp[:, pilot_symbols, pilot_subcarriers]
    
    # Interpolation
    h_interp_full_batch = []
    h_interp_comp_batch = []
    for idx_in_batch in range(current_batch_size):
        h_interp_f = interpolate_channel(rx_grid_full[idx_in_batch], tx_grid[idx_in_batch], pilot_mask)
        h_interp_c = interpolate_channel(rx_grid_comp[idx_in_batch], tx_grid[idx_in_batch], pilot_mask)
        h_interp_full_batch.append(h_interp_f)
        h_interp_comp_batch.append(h_interp_c)
        
    h_interp_full_batch = np.stack(h_interp_full_batch, axis=0)
    h_interp_comp_batch = np.stack(h_interp_comp_batch, axis=0)
    
    # Calculate batch NMSE and SSIM values
    nmse_ls_batch = []
    nmse_ls_pilot_batch = []
    nmse_prac_batch = []
    nmse_li_batch = []
    ssim_ls_batch = []
    ssim_prac_batch = []
    ssim_li_batch = []
    for idx_in_batch in range(current_batch_size):
        h_eff_c = h_eff_siso_comp[idx_in_batch]
        h_eff_f = h_eff_siso_full[idx_in_batch]
        
        # LS estimated full grid (un-interpolated)
        h_ls_c = h_LS_comp[idx_in_batch]
        
        # NMSE LS full
        n_ls = np.sum(np.abs(h_eff_c - h_ls_c)**2) / np.sum(np.abs(h_eff_c)**2)
        nmse_ls_batch.append(n_ls)
        
        # NMSE LS pilots (only at pilot locations)
        h_eff_pilots = h_eff_c[pilot_symbols, pilot_subcarriers]
        h_ls_pilots = h_LS_pilots_comp[idx_in_batch]
        n_ls_p = np.sum(np.abs(h_eff_pilots - h_ls_pilots)**2) / np.sum(np.abs(h_eff_pilots)**2)
        nmse_ls_pilot_batch.append(n_ls_p)
        
        # NMSE prac (interpolated precompensated)
        h_interp_c = h_interp_comp_batch[idx_in_batch]
        n_prac = np.sum(np.abs(h_eff_c - h_interp_c)**2) / np.sum(np.abs(h_eff_c)**2)
        nmse_prac_batch.append(n_prac)
        
        # NMSE li (interpolated uncompensated)
        h_interp_f = h_interp_full_batch[idx_in_batch]
        n_li = np.sum(np.abs(h_eff_f - h_interp_f)**2) / np.sum(np.abs(h_eff_f)**2)
        nmse_li_batch.append(n_li)
        
        # Calculate batch SSIM values
        s_ls = compute_complex_ssim(h_eff_c, h_ls_c)
        s_prac = compute_complex_ssim(h_eff_c, h_interp_c)
        s_li = compute_complex_ssim(h_eff_c, h_interp_f)
        
        ssim_ls_batch.append(s_ls)
        ssim_prac_batch.append(s_prac)
        ssim_li_batch.append(s_li)
        
    nmse_ls_list.extend(nmse_ls_batch)
    nmse_ls_pilot_list.extend(nmse_ls_pilot_batch)
    nmse_prac_list.extend(nmse_prac_batch)
    nmse_li_list.extend(nmse_li_batch)
    ssim_ls_list.extend(ssim_ls_batch)
    ssim_prac_list.extend(ssim_prac_batch)
    ssim_li_list.extend(ssim_li_batch)
    
    # Accumulate batch results
    H_eff_full_list.append(h_eff_siso_full)
    H_eff_comp_list.append(h_eff_siso_comp)
    H_LS_full_list.append(h_LS_pilots_full)
    H_LS_comp_list.append(h_LS_pilots_comp)
    H_interp_full_list.append(h_interp_full_batch)
    H_interp_comp_list.append(h_interp_comp_batch)
    
    # 4. Plots ONLY for the very first batch, first sample (Slot 0)
    if b == 0:
        print("Plotting geometry and channel visualizations once for Slot 0...")
        
        # Plot ECEF Geometry
        try:
            t_path = np.linspace(-150, 150, 100)
            sat_path_ECEF = []
            ue_path_ECEF = []
            for tp in t_path:
                r_sat_ECEF_p, _ = get_satellite_state_ecef(
                    tp, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E
                )
                sat_path_ECEF.append(r_sat_ECEF_p)
                r_ue_ECEF_p = r_ue_ECEF_all[0] + ue_vel_ECEF * tp
                r_ue_ECEF_p = r_ue_ECEF_p * (np.linalg.norm(r_ue_ECEF_all[0]) / np.linalg.norm(r_ue_ECEF_p))
                ue_path_ECEF.append(r_ue_ECEF_p)
                
            sat_path_ECEF = np.array(sat_path_ECEF).T
            ue_path_ECEF = np.array(ue_path_ECEF).T
            
            # Generate Beam Footprint circle (15 km radius) in ENU and rotate to ECEF
            theta_circle = np.linspace(0, 2 * np.pi, 100)
            circle_ENU = np.zeros((3, 100))
            circle_ENU[0] = r_beam * np.cos(theta_circle)
            circle_ENU[1] = r_beam * np.sin(theta_circle)
            circle_ENU[2] = 0.0  # ground plane
            
            # Convert to ECEF relative to r_ue_ECEF_0
            circle_ECEF = r_ue_ECEF_0[:, np.newaxis] + R_ENU2ECEF @ circle_ENU
            # Project to ellipsoid surface
            norm_r_ue_0 = np.linalg.norm(r_ue_ECEF_0)
            for i in range(100):
                circle_ECEF[:, i] = circle_ECEF[:, i] * (norm_r_ue_0 / np.linalg.norm(circle_ECEF[:, i]))
            
            fig_ecef = plt.figure(figsize=(10, 8), facecolor='white')
            ax_ecef = fig_ecef.add_subplot(111, projection='3d')
            ax_ecef.set_facecolor('white')
            XS, YS, ZS = get_local_earth_patch(lambda_UE, phi_UE, a_wgs84, delta_deg=12.0, num_pts=30)
            ax_ecef.plot_surface(XS, YS, ZS, color=(0.85, 0.90, 0.95), edgecolor=(0.70, 0.75, 0.80), alpha=0.5, linewidth=0.25)
            ax_ecef.plot(sat_path_ECEF[0], sat_path_ECEF[1], sat_path_ECEF[2], '--', color=(0.85, 0.60, 0.05), linewidth=1.5, label='LEO Orbit (ECEF)')
            ax_ecef.plot(ue_path_ECEF[0], ue_path_ECEF[1], ue_path_ECEF[2], '-', color=(0.85, 0.20, 0.20), linewidth=2, label='UE Path')
            
            # Draw Beam Footprint and Beam Center
            ax_ecef.plot(circle_ECEF[0], circle_ECEF[1], circle_ECEF[2], color='purple', linewidth=2, label='Beam Footprint (15 km radius)')
            ax_ecef.scatter(r_ue_ECEF_0[0], r_ue_ECEF_0[1], r_ue_ECEF_0[2], color='purple', marker='X', s=120, edgecolor='w', label='Beam Center')
            
            ax_ecef.scatter(r_sat_ECEF_all[0, 0], r_sat_ECEF_all[0, 1], r_sat_ECEF_all[0, 2], color=(0.85, 0.60, 0.05), s=150, edgecolor='w', label='LEO Satellite (Slot 0)')
            ax_ecef.scatter(r_ue_ECEF_all[0, 0], r_ue_ECEF_all[0, 1], r_ue_ECEF_all[0, 2], color=(0.85, 0.20, 0.20), s=100, edgecolor='w', label='UE (Slot 0)')
            ax_ecef.plot([r_ue_ECEF_all[0, 0], r_sat_ECEF_all[0, 0]], 
                         [r_ue_ECEF_all[0, 1], r_sat_ECEF_all[0, 1]], 
                         [r_ue_ECEF_all[0, 2], r_sat_ECEF_all[0, 2]], 
                         '--', color=(0.6, 0.6, 0.6), linewidth=1.2, label='Line of Sight (Slot 0)')
            
            # Draw Velocity Vectors (scaled for WGS-84 scale)
            vel_scale_sat = 1.0e5 / 7500 * 5
            vel_scale_ue = 1.0e5 / 50 * 3.5
            v_sat_scaled = v_sat_ECEF_all[0] * vel_scale_sat
            v_ue_scaled = ue_vel_ECEF * vel_scale_ue

            ax_ecef.quiver(r_sat_ECEF_all[0, 0], r_sat_ECEF_all[0, 1], r_sat_ECEF_all[0, 2],
                           v_sat_scaled[0], v_sat_scaled[1], v_sat_scaled[2],
                           color=(0.10, 0.65, 0.35), linewidth=1.5, arrow_length_ratio=0.25, label='Satellite Velocity Vector (Scaled)')
            ax_ecef.quiver(r_ue_ECEF_all[0, 0], r_ue_ECEF_all[0, 1], r_ue_ECEF_all[0, 2],
                           v_ue_scaled[0], v_ue_scaled[1], v_ue_scaled[2],
                           color=(0.85, 0.40, 0.05), linewidth=2.0, arrow_length_ratio=0.3, label='UE Velocity Vector (Scaled)')

            zoom_width = 1.2e6
            ax_ecef.set_xlim(r_ue_ECEF_0[0] - zoom_width, r_ue_ECEF_0[0] + zoom_width)
            ax_ecef.set_ylim(r_ue_ECEF_0[1] - zoom_width, r_ue_ECEF_0[1] + zoom_width)
            ax_ecef.set_zlim(r_ue_ECEF_0[2] - zoom_width, r_ue_ECEF_0[2] + zoom_width)
            ax_ecef.set_xlabel('ECEF X (meters)')
            ax_ecef.set_ylabel('ECEF Y (meters)')
            ax_ecef.set_zlabel('ECEF Z (meters)')
            ax_ecef.set_title(f"3D Geometry Snapshot (ECEF Coordinates)\nSlot 0 Elevation: {elevation_angles_all[0]:.2f}°")
            ax_ecef.legend(facecolor='white')
            plt.tight_layout()
            ecef_plot_filename = os.path.join(output_dir, "geometry_ecef.pdf")
            plt.savefig(ecef_plot_filename, format='pdf', bbox_inches='tight', pad_inches=0.02)
            plt.close('all')
        except Exception as ex:
            print(f"Warning: Could not save ECEF plot. Error: {ex}")
            
        # Plot Local Tangent Plane ENU Topology for Slot 0
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(ut_loc_batch[0, 0], ut_loc_batch[0, 1], ut_loc_batch[0, 2], color='red', s=100, label='User Equipment (Slot 0)')
            
            # Draw Beam Footprint circle and Beam Center
            ax.plot(circle_ENU[0], circle_ENU[1], np.zeros_like(circle_ENU[0]), color='purple', linewidth=2, linestyle='-', label='Beam Footprint (15 km)')
            ax.scatter(0, 0, 0, color='purple', marker='X', s=120, edgecolor='w', label='Beam Center')
            
            ax.scatter(bs_loc_batch[0, 0], bs_loc_batch[0, 1], bs_loc_batch[0, 2], color='blue', s=200, label='LEO Satellite (Slot 0)')
            ax.plot([ut_loc_batch[0, 0], bs_loc_batch[0, 0]], 
                    [ut_loc_batch[0, 1], bs_loc_batch[0, 1]], 
                    [ut_loc_batch[0, 2], bs_loc_batch[0, 2]], 
                    color='gray', linestyle='--', alpha=0.7, label='Line of Sight (Slot 0)')
            
            grid_range = 25000.0
            x_grid, y_grid = np.meshgrid(np.linspace(-grid_range, grid_range, 10), np.linspace(-grid_range, grid_range, 10))
            z_grid = np.zeros_like(x_grid)
            ax.plot_wireframe(x_grid, y_grid, z_grid, color='green', alpha=0.15, rstride=1, cstride=1)
            
            scale_ue_vel = 100.0  
            ax.quiver(ut_loc_batch[0, 0], ut_loc_batch[0, 1], ut_loc_batch[0, 2],
                      ut_vel_batch[0, 0]*scale_ue_vel, ut_vel_batch[0, 1]*scale_ue_vel, ut_vel_batch[0, 2]*scale_ue_vel,
                      color='orange', linewidth=2, label='UE Velocity (Scaled)')
            
            scale_sat_vel = 5.0
            ax.quiver(bs_loc_batch[0, 0], bs_loc_batch[0, 1], bs_loc_batch[0, 2],
                      bs_vel_batch[0, 0]*scale_sat_vel, bs_vel_batch[0, 1]*scale_sat_vel, bs_vel_batch[0, 2]*scale_sat_vel,
                      color='cyan', linewidth=2, label='Satellite Velocity (Scaled)')
            
            ax.set_xlabel('East (meters)')
            ax.set_ylabel('North (meters)')
            ax.set_zlabel('Up (meters)')
            ax.set_title(f"3D Geometry (Local Tangent Plane ENU)\nSlot 0 Elevation: {elevation_angles_all[0]:.2f}°")
            ax.legend()
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, "geometry_plot.pdf")
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight', pad_inches=0.02)
            plt.close('all')
        except Exception as ex:
            print(f"Warning: Could not save ENU plot. Error: {ex}")
        # Plot Channels (Transposed, viridis colormap, cropped)
        try:
            real_full_t = np.real(h_eff_siso_full[0]).T
            real_comp_t = np.real(h_eff_siso_comp[0]).T
            
            fig0, ax0 = plt.subplots(figsize=(8, 5), facecolor='white')
            im0 = ax0.imshow(real_full_t, aspect='auto', cmap='viridis', origin='lower')
            ax0.set_title(f"Full Doppler: Real Part of Slot 0 Channel ({scenario.upper()})\nElevation: {elevation_angles_all[0]:.2f}°")
            ax0.set_xlabel("OFDM Symbol Index")
            ax0.set_ylabel("Subcarrier Index")
            fig0.colorbar(im0, ax=ax0)
            plt.tight_layout()
            channel_real_full_filename = os.path.join(output_dir, f"channel_real_full_{scenario}.pdf")
            plt.savefig(channel_real_full_filename, format='pdf', bbox_inches='tight', pad_inches=0.01)
            plt.close(fig0)
            
            fig1, ax1 = plt.subplots(figsize=(8, 5), facecolor='white')
            im1 = ax1.imshow(real_comp_t, aspect='auto', cmap='viridis', origin='lower')
            ax1.set_title(f"Precompensated Doppler: Real Part of Slot 0 Channel ({scenario.upper()})\nElevation: {elevation_angles_all[0]:.2f}°")
            ax1.set_xlabel("OFDM Symbol Index")
            ax1.set_ylabel("Subcarrier Index")
            fig1.colorbar(im1, ax=ax1)
            plt.tight_layout()
            channel_real_comp_filename = os.path.join(output_dir, f"channel_real_comp_{scenario}.pdf")
            plt.savefig(channel_real_comp_filename, format='pdf', bbox_inches='tight', pad_inches=0.01)
            plt.close(fig1)
        except Exception as ex:
            print(f"Warning: Could not save channel plots. Error: {ex}")

# Concatenate all batched results
H_eff_full = np.concatenate(H_eff_full_list, axis=0)
H_eff_comp = np.concatenate(H_eff_comp_list, axis=0)
H_LS_full = np.concatenate(H_LS_full_list, axis=0)
H_LS_comp = np.concatenate(H_LS_comp_list, axis=0)
H_interp_full = np.concatenate(H_interp_full_list, axis=0)
H_interp_comp = np.concatenate(H_interp_comp_list, axis=0)

# Convert channel grids to MATLAB shape [14, 132, N_samples]
H_perfect = np.transpose(H_eff_comp, (1, 2, 0))      # [14, 132, N_samples]
H_perfect_ori = np.transpose(H_eff_full, (1, 2, 0))  # [14, 132, N_samples]
H_prac = np.transpose(H_interp_comp, (1, 2, 0))      # [14, 132, N_samples]
H_li = np.transpose(H_interp_full, (1, 2, 0))        # [14, 132, N_samples]
H_ls_pilots = np.transpose(H_LS_comp, (1, 0))        # [numPilots, N_samples]

# Pilot positions and indices (MATLAB style: 1-indexed, column-major)
pilot_rows = pilot_subcarriers + 1
pilot_cols = pilot_symbols + 1
pilot_indices = (pilot_cols - 1) * 132 + pilot_rows

# Convert NMSE lists to arrays, and average SSIM lists to scalars
nmse_prac = np.array(nmse_prac_list)
nmse_ls = np.array(nmse_ls_list)
nmse_ls_pilot = np.array(nmse_ls_pilot_list)
nmse_li = np.array(nmse_li_list)
ssim_prac = float(np.mean(ssim_prac_list)) if len(ssim_prac_list) > 0 else 0.0
ssim_ls = float(np.mean(ssim_ls_list)) if len(ssim_ls_list) > 0 else 0.0
ssim_li = float(np.mean(ssim_li_list)) if len(ssim_li_list) > 0 else 0.0

# Calculate common Doppler shifts at the beam center (ENU [0,0,0]) over time
lambda_0 = SPEED_OF_LIGHT / carrier_frequency
doppler_sat_bc_all = []
for i in range(N_samples):
    v_los_bc = -bs_loc_ENU_all[i]
    u_los_bc = v_los_bc / np.linalg.norm(v_los_bc)
    d_sat_bc = np.dot(v_sat_ENU_all[i], u_los_bc) / lambda_0
    doppler_sat_bc_all.append(d_sat_bc)
doppler_sat_bc_all = np.array(doppler_sat_bc_all)

# Save .mat file
mat_filename = os.path.join(output_dir, f"channel_{scenario}_sequentialTime.mat")
mat_data = {
    # Main requested MATLAB matrices
    "H_perfect": H_perfect,
    "H_perfect_ori": H_perfect_ori,
    "H_prac": H_prac,
    "H_li": H_li,
    "H_ls_pilots": H_ls_pilots,
    "pilot_indices": pilot_indices,
    "pilot_rows": pilot_rows,
    "pilot_cols": pilot_cols,
    "nmse_prac": nmse_prac,
    "nmse_ls": nmse_ls,
    "nmse_ls_pilot": nmse_ls_pilot,
    "nmse_li": nmse_li,
    "ssim_prac": ssim_prac,
    "ssim_ls": ssim_ls,
    "ssim_li": ssim_li,
    "doppler_sat_bc": doppler_sat_bc_all,
    
    # Metadata and other parameters
    "pilot_symbols": pilot_symbols + 1,       
    "pilot_subcarriers": pilot_subcarriers + 1, 
    "ut_loc_ENU": ut_loc_ENU_all,       
    "bs_loc_ENU": bs_loc_ENU_all,       
    "ut_velocity_ENU": v_ue_ENU_all,    
    "bs_velocity_ENU": v_sat_ENU_all,   
    "sat_speeds_all": sat_speeds_all,   
    "elevation_angles_all": elevation_angles_all, 
    "slant_ranges_all": slant_ranges_all, 
    "time_stamps": t_all,               
    "slot_duration": slot_duration      
}
savemat(mat_filename, mat_data)
print(f"Successfully saved batched sequential channel data of {N_samples} slots to {mat_filename}")

# Save scenario parameters documentation
delay_spreads_ns = np.array(delay_spreads_all) * 1e9
avg_delay_spread_ns = np.mean(delay_spreads_ns) if len(delay_spreads_ns) > 0 else 0.0
min_delay_spread_ns = np.min(delay_spreads_ns) if len(delay_spreads_ns) > 0 else 0.0
max_delay_spread_ns = np.max(delay_spreads_ns) if len(delay_spreads_ns) > 0 else 0.0

md_filename = os.path.join(output_dir, f"readme_{scenario}_sequentialTime.md")

md_content = f"""# Channel & Geometry Generation Settings - {scenario.upper()} (Sequential Time)

- **Scenario Type**: {scenario.upper()} (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: {carrier_frequency / 1e9:.2f} GHz
- **Link Direction**: {direction}
- **Satellite (LEO) Height**: {satellite_height / 1000:.0f} km
- **Satellite Nominal Elevation Angle**: {elevation_angle_nom:.2f} degrees
- **Subcarrier Spacing (SCS)**: {SCS / 1e3:.0f} kHz
- **FFT Size**: {nFFT}
- **Active Subcarriers**: 132 (out of {nFFT})
- **SNR (for LS estimation)**: {SNR_dB} dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 7, 11]
- **Total Samples Generated**: {N_samples} (Slots)
- **Average RMS Delay Spread**: {avg_delay_spread_ns:.2f} ns (Range: [{min_delay_spread_ns:.2f}, {max_delay_spread_ns:.2f}] ns)

## Beam Boresight & Footprint Settings
- **Beam Center (ECEF)**: [{r_ue_ECEF_0[0]:.2f}, {r_ue_ECEF_0[1]:.2f}, {r_ue_ECEF_0[2]:.2f}] meters
- **Beam Center (ENU)**: [0.00, 0.00, 0.00] meters (Origin of local tangent plane)
- **Beam Footprint Radius**: {r_beam / 1000:.1f} km

## Time-Series Sequential Settings
- **Generation Method**: Sequential Time-Series Generation (GPU Mini-Batched)
- **Number of Slots**: {N_samples} slots
- **Slot Duration**: {slot_duration * 1e3:.6f} ms (Total duration: {t_all[-1] * 1e3:.3f} ms)
- **UE Ground Speed**: {ue_speed} m/s (Constant heading: {heading_deg} degrees)
- **Satellite Speed (Nominal)**: {np.mean(sat_speeds_all):.2f} m/s
- **Elevation Angle Range**: [{np.min(elevation_angles_all):.2f}, {np.max(elevation_angles_all):.2f}] degrees
- **Slant Range Range**: [{np.min(slant_ranges_all)/1000:.2f}, {np.max(slant_ranges_all)/1000:.2f}] km
"""

with open(md_filename, "w") as f:
    f.write(md_content)
print(f"Saved scenario parameters documentation to {md_filename}")

