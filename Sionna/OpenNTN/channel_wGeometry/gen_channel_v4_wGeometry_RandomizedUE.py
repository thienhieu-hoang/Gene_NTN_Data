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


# ========================== Simulation Variables ==========================

satellite_height = 600000.0  # LEO Orbit altitude (m) (600 km)
scenario = "dur"             # dur (Dense Urban), sur (SubUrban), urb (Urban)
carrier_frequency = 2.18e9     # DL carrier frequency (Hz)
SCS = 30e3
delay_spread_ns_custom = 100 #None    # Custom delay spread in ns (e.g. 100.0) or None for standard 3GPP defaults
fix_delay_spread = True               # True to fix the delay spread value exactly; False to sample with standard variance
dense_pilot = False                   # False (default) to use sparse pilots (88 pilots); True for dense pilots (264 pilots)

# satellite_height = 600000.0  # LEO Orbit altitude (m) (600 km)
# scenario = "dur"             # dur (Dense Urban), sur (SubUrban), urb (Urban)
# carrier_frequency = 20e9     # DL carrier frequency (Hz)
# SCS = 120e3
# delay_spread_ns_custom = 100 #None    # Custom delay spread in ns (e.g. 100.0) or None for standard 3GPP defaults

v_min, v_max = 20.0, 30.0        # UE ground speed in m/s

# Total samples to generate (N_samples)
N_samples = 1024
batch_size = 32 #32
# Target Elevation Angle Configuration (e.g. 20, 30, 40, 50, 60, 70, 80, 90 deg, or None for peak 90 deg)
target_elevation_angle = 70.0   # Desired nominal elevation angle in degrees (e.g. 50.0)

SNR_dB = -5

# =========================================================================

# =========================================================================
# 1. GEOMETRY CONFIGURATION PARAMETERS (from visualize_geometry.py)
# =========================================================================
phi_UE_deg = 37.7749         # UE Reference Latitude (degrees)
lambda_UE_deg = -122.4194    # UE Reference Longitude (degrees)
h_UE = 100.0                 # UE Reference Altitude (m)
inclination_deg = 55.0       # Orbit inclination (degrees)

# Physical constants
omega_E = 7.292115e-5        # Earth's rotation rate (rad/s)
mu = 3.986004418e14          # Earth's gravitational parameter (m^3/s^2)
a_wgs84 = 6378137.0          # WGS-84 semi-major axis (m)
e2 = 6.69437999e-3           # First eccentricity squared

# Convert to radians
inclination = np.deg2rad(inclination_deg)
phi_UE = np.deg2rad(phi_UE_deg)
lambda_UE = np.deg2rad(lambda_UE_deg)

# Orbit radius & speed
r_orbit = a_wgs84 + satellite_height
omega_s = np.sqrt(mu / r_orbit**3)
v_sat_orbit = np.sqrt(mu / r_orbit)

# UE Reference ECEF position
N_phi_0 = a_wgs84 / np.sqrt(1.0 - e2 * np.sin(phi_UE)**2)
r_ue_ECEF_0 = np.array([
    (N_phi_0 + h_UE) * np.cos(phi_UE) * np.cos(lambda_UE),
    (N_phi_0 + h_UE) * np.cos(phi_UE) * np.sin(lambda_UE),
    (N_phi_0 * (1.0 - e2) + h_UE) * np.sin(phi_UE)
])

# Rotation matrix from local ENU to ECEF at UE's reference location
R_ENU2ECEF = np.array([
    [-np.sin(lambda_UE), -np.sin(phi_UE) * np.cos(lambda_UE), np.cos(phi_UE) * np.cos(lambda_UE)],
    [ np.cos(lambda_UE), -np.sin(phi_UE) * np.sin(lambda_UE), np.cos(phi_UE) * np.sin(lambda_UE)],
    [ 0.0,                np.cos(phi_UE),                 np.sin(phi_UE)]
])
R_ECEF2ENU = R_ENU2ECEF.T

# Orbit alignment (closest approach at t=0)
if inclination >= abs(phi_UE):
    u_mid = np.arcsin(np.sin(phi_UE) / np.sin(inclination))
else:
    u_mid = np.sign(phi_UE) * np.pi / 2.0
Omega_RAAN = lambda_UE - np.arctan2(np.sin(u_mid) * np.cos(inclination), np.cos(u_mid))

# Calculate Satellite state at snapshot time t (aligned to target_elevation_angle)
if target_elevation_angle is not None and target_elevation_angle < 89.9:
    theta_target_rad = np.deg2rad(target_elevation_angle)
    # Solve orbital central angle gamma for target elevation
    gamma_central = np.pi / 2.0 - theta_target_rad - np.arcsin((a_wgs84 / r_orbit) * np.cos(theta_target_rad))
    t_snapshot = float(gamma_central / omega_s)
else:
    t_snapshot = 0.0

t = t_snapshot
r_sat_ECEF, v_sat_ECEF = get_satellite_state_ecef(
    t, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E
)

# Reference geometry
v_los = r_sat_ECEF - r_ue_ECEF_0
slant_range_nom = np.linalg.norm(v_los)
u_normal = r_ue_ECEF_0 / np.linalg.norm(r_ue_ECEF_0)
u_los = v_los / slant_range_nom
elev_rad_nom = np.arcsin(np.dot(u_normal, u_los))
elevation_angle_nom = float(np.degrees(elev_rad_nom))

# Convert reference coordinates to local ENU
bs_loc_ENU = ecef_to_enu(r_sat_ECEF, r_ue_ECEF_0, lambda_UE, phi_UE)
v_sat_ENU = ecef_to_enu(v_sat_ECEF, np.zeros(3), lambda_UE, phi_UE)
sat_speed = np.linalg.norm(v_sat_ECEF)

# =========================================================================
# 2. SYSTEM SIMULATION CONFIGURATION & DATASETS CONFIGURATION
# =========================================================================

direction = "downlink"
num_ut = 1
SNR_linear = 10.0 ** (SNR_dB / 10.0)



# Randomized UE range in ENU (East-North-Up)
z_val = 1.5                      # Standard mobile antenna height (m)
r_beam = 15000.0                 # 15 km beam footprint radius
r_ue_max = 14500.0               # Maximum initial radius for UE generation (14.5 km)

# ----------------- Generate Randomized UE Positions & Velocities -----------------
np.random.seed(42)
ut_loc_ENU_all = np.zeros((N_samples, 3))

# Generate positions uniformly inside a circle of radius r_ue_max (polar coordinates)
theta_rand = np.random.uniform(0.0, 2 * np.pi, N_samples)
r_rand = r_ue_max * np.sqrt(np.random.uniform(0.0, 1.0, N_samples))
ut_loc_ENU_all[:, 0] = r_rand * np.cos(theta_rand)
ut_loc_ENU_all[:, 1] = r_rand * np.sin(theta_rand)
ut_loc_ENU_all[:, 2] = z_val

# Ensure the first data sample has an offset distance of at least 5000m (5km) compared to the beam center
dist_first = np.sqrt(ut_loc_ENU_all[0, 0]**2 + ut_loc_ENU_all[0, 1]**2)
while dist_first < 5000.0 or dist_first > r_ue_max:
    theta_f = np.random.uniform(0.0, 2 * np.pi)
    r_f = r_ue_max * np.sqrt(np.random.uniform(0.0, 1.0))
    ut_loc_ENU_all[0, 0] = r_f * np.cos(theta_f)
    ut_loc_ENU_all[0, 1] = r_f * np.sin(theta_f)
    dist_first = np.sqrt(ut_loc_ENU_all[0, 0]**2 + ut_loc_ENU_all[0, 1]**2)

ut_speed_all = np.random.uniform(v_min, v_max, N_samples)
ut_heading_all = np.random.uniform(0.0, 2 * np.pi, N_samples)
ut_velocity_ENU_all = np.zeros((N_samples, 3))
ut_velocity_ENU_all[:, 0] = ut_speed_all * np.sin(ut_heading_all)
ut_velocity_ENU_all[:, 1] = ut_speed_all * np.cos(ut_heading_all)
ut_velocity_ENU_all[:, 2] = 0.0

# Convert positions & velocities to ECEF for physical calculations
r_ue_ECEF_all = r_ue_ECEF_0[np.newaxis, :] + (R_ENU2ECEF @ ut_loc_ENU_all.T).T
ue_vel_ECEF_all = (R_ENU2ECEF @ ut_velocity_ENU_all.T).T

# Calculate slant ranges and elevation angles per randomized sample
v_los_all = r_sat_ECEF[np.newaxis, :] - r_ue_ECEF_all
slant_ranges_all = np.linalg.norm(v_los_all, axis=1)
u_normal_all = r_ue_ECEF_all / np.linalg.norm(r_ue_ECEF_all, axis=1, keepdims=True)
u_los_all = v_los_all / slant_ranges_all[:, np.newaxis]
elev_rad_all = np.arcsin(np.sum(u_normal_all * u_los_all, axis=1))
elevation_angles_all = np.degrees(elev_rad_all)

# Calculate orbital pass timeline (from 10 deg horizon rise to 10 deg horizon set)
theta_horizon_rad = np.deg2rad(10.0) # 3GPP minimum operational elevation angle
gamma_horizon = np.pi / 2.0 - theta_horizon_rad - np.arcsin((a_wgs84 / r_orbit) * np.cos(theta_horizon_rad))
t_pass_half = float(gamma_horizon / omega_s)

t_min = -t_pass_half
t_max = +t_pass_half

# Satellite position at t_min, t=0, and t_max
r_sat_min, _ = get_satellite_state_ecef(t_min, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E)
r_sat_peak, _ = get_satellite_state_ecef(0.0, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E)
r_sat_max, _ = get_satellite_state_ecef(t_max, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E)

def calc_elev_at_t(r_sat_pos):
    v_l = r_sat_pos - r_ue_ECEF_0
    u_l = v_l / np.linalg.norm(v_l)
    return float(np.degrees(np.arcsin(np.dot(u_normal, u_l))))

elev_at_tmin = calc_elev_at_t(r_sat_min)
elev_at_tpeak = calc_elev_at_t(r_sat_peak)
elev_at_tmax = calc_elev_at_t(r_sat_max)

# Print elevation angle timeline details
print("=" * 70)
print("SATELLITE ORBITAL PASS & ELEVATION ANGLE TIMELINE")
print(f"1. Satellite Pass Start (t_start = {t_min:.1f} s) : Elevation = {elev_at_tmin:.2f}° (Horizon Rise)")
print(f"2. Peak Zenith Approach  (t_peak  = 0.0 s)     : Elevation = {elev_at_tpeak:.2f}° (Overhead Peak)")
print(f"3. Dataset Snapshot Point(t_snap  = {t_snapshot:.1f} s) : Elevation = {elevation_angle_nom:.2f}° (Single Position Generated)")
print(f"4. Satellite Pass End   (t_end    = {t_max:.1f} s) : Elevation = {elev_at_tmax:.2f}° (Horizon Set)")
print(f"\nSpatial Elevation Angle Variation Across 15km Beam Footprint ({N_samples} Randomized UEs):")
print(f"   - UE Farthest from Satellite (Min Elevation) : {np.min(elevation_angles_all):.2f}°")
print(f"   - UE Closest to Satellite  (Max Elevation) : {np.max(elevation_angles_all):.2f}°")
print(f"   - Average Across All UEs   (Mean Elevation): {np.mean(elevation_angles_all):.2f}°")
print("=" * 70 + "\n")

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
                    pilot_ofdm_symbol_indices=[2, 11])

frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
pilot_mask = tf.squeeze(rg.pilot_pattern.mask).numpy().astype(bool)

if not dense_pilot:
    # Extract only elements where subcarrier index mod 6 == 0 or mod 6 == 1, others are set to False
    subcarrier_indices = np.arange(132)
    sparse_subc_mask = (subcarrier_indices % 6 == 0) | (subcarrier_indices % 6 == 1)
    for sym_idx in [2, 11]:
        pilot_mask[sym_idx] = pilot_mask[sym_idx] & sparse_subc_mask

pilot_symbols, pilot_subcarriers = np.where(pilot_mask)
qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / np.sqrt(2)

# =========================================================================
# 3. HELPER INTERPOLATOR
# =========================================================================
def interpolate_channel(rx_grid_b, tx_grid_b, pilot_mask):
    pilot_mask = pilot_mask.astype(bool)
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
                              elevation_angle=elevation_angle_nom, # nominal for cluster generation
                              doppler_enabled=True,
                              doppler_mode='full')

# Override clusters based on elevation angle (rounded to nearest 10 degrees)
rounded_elev = int(round(elevation_angle_nom / 10.0) * 10)
rounded_elev = max(10, min(90, rounded_elev))
channel_model._scenario._params_nlos[f"numClusters_{rounded_elev}"] = 3

# Apply custom delay spread if configured
if delay_spread_ns_custom is not None:
    target_log_mean_ds = np.log10(delay_spread_ns_custom * 1e-9)
    channel_model._scenario._params_los[f"muDS_{rounded_elev}"] = tf.constant(target_log_mean_ds, dtype=tf.float32)
    channel_model._scenario._params_nlos[f"muDS_{rounded_elev}"] = tf.constant(target_log_mean_ds, dtype=tf.float32)
    if fix_delay_spread:
        channel_model._scenario._params_los[f"sigmaDS_{rounded_elev}"] = tf.constant(0.0, dtype=tf.float32)
        channel_model._scenario._params_nlos[f"sigmaDS_{rounded_elev}"] = tf.constant(0.0, dtype=tf.float32)

ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
remove_nulled = RemoveNulledSubcarriers(rg)
num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
channel_seed = 42

# Setup outputs directory
script_dir = os.path.dirname(os.path.abspath(__file__))
elev_tag = f"_{int(round(elevation_angle_nom / 10.0) * 10)}deg"

fc_ghz = carrier_frequency / 1e9
if fc_ghz == int(fc_ghz):
    fc_str = f"{int(fc_ghz)}G"
else:
    fc_str = f"{fc_ghz:.2f}".rstrip('0').rstrip('.').replace('.', 'p') + "G"

pilot_suffix = "_dense" if dense_pilot else ""
if delay_spread_ns_custom is not None:
    ds_suffix = f"{int(delay_spread_ns_custom)}nsFix" if fix_delay_spread else f"{int(delay_spread_ns_custom)}ns"
    setting_dir = f"{scenario.upper()}{ds_suffix}{pilot_suffix}_{fc_str}_{int(satellite_height/1000)}km{elev_tag}_r{int(r_beam/1000)}km_{int(v_min)}to{int(v_max)}mps"
else:
    setting_dir = f"{scenario.upper()}{pilot_suffix}_{fc_str}_{int(satellite_height/1000)}km{elev_tag}_r{int(r_beam/1000)}km_{int(v_min)}to{int(v_max)}mps"
output_dir = os.path.join(script_dir, "results", setting_dir, f"{int(SNR_dB)}dB")
os.makedirs(output_dir, exist_ok=True)

# =========================================================================
# 4. MINI-BATCH CHANNEL GENERATION & ESTIMATION
# =========================================================================
H_extracted_ori_list = []
H_extracted_comp_list = []
H_LS_ori_list = []
H_LS_comp_list = []
H_interp_ori_list = []
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
print(f"Starting batched generation of {N_samples} channels (Batch Size: {batch_size}, Total Batches: {num_batches})...")

for b in range(num_batches):
    start_idx = b * batch_size
    end_idx = min(start_idx + batch_size, N_samples)
    current_batch_size = end_idx - start_idx
    
    # 1. Slice and build tensors for current batch
    ut_loc_batch = ut_loc_ENU_all[start_idx:end_idx]
    ut_vel_batch = ut_velocity_ENU_all[start_idx:end_idx]
    
    # Convert ECEF positions and velocities to local ENU tangent plane frame
    # and construct batched tensors of shape [current_batch_size, 1, 3]
    ut_loc_tensor = tf.constant(ut_loc_batch[:, np.newaxis, :], dtype=tf.float32)
    
    # Satellite ENU position is same for all UEs in this batch
    bs_loc_ENU_tile = np.tile(bs_loc_ENU, (current_batch_size, 1))
    bs_loc_tensor = tf.constant(bs_loc_ENU_tile[:, np.newaxis, :], dtype=tf.float32)
    
    ut_orientations = tf.zeros([current_batch_size, 1, 3])
    bs_orientations = tf.zeros([current_batch_size, 1, 3])
    
    ut_velocities_tensor = tf.constant(ut_vel_batch[:, np.newaxis, :], dtype=tf.float32)
    
    v_sat_ENU_tile = np.tile(v_sat_ENU, (current_batch_size, 1))
    bs_velocities_tensor = tf.constant(v_sat_ENU_tile[:, np.newaxis, :], dtype=tf.float32)
    
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
    H_full_ori = ofdm_channel()
    h_extracted_ori_ = remove_nulled(H_full_ori)
    H_extracted_ori = h_extracted_ori_[:, 0, 0, 0, 0, :, :].numpy()  # shape [current_batch_size, 14, 132]
    
    # ------------------ Part 2: Precompensated Channel ------------------
    # Precompensate satellite Doppler by setting satellite velocity to zero
    channel_model.set_topology(ut_loc_tensor, bs_loc_tensor, ut_orientations, bs_orientations,
                               ut_velocities_tensor, tf.zeros_like(bs_velocities_tensor), in_state, los=True)
    channel_model._scenario._doppler_mode = 'precompensated'
    tf.random.set_seed(iteration_seed)
    path_coefficients_comp, path_delays_comp = channel_model(num_time_steps, sampling_frequency)
    
    tf.random.set_seed(iteration_seed)
    H_full_comp = ofdm_channel()
    h_extracted_comp_ = remove_nulled(H_full_comp)
    H_extracted_comp = h_extracted_comp_[:, 0, 0, 0, 0, :, :].numpy()  # shape [current_batch_size, 14, 132]
    
    # ------------------ Part 3: Channel Estimation & Interpolation ------------------
    np.random.seed(iteration_seed)
    tx_grid = np.random.choice(qpsk_symbols, size=[current_batch_size, 14, 132])
    
    np.random.seed(iteration_seed + 10)
    noise_pattern = (np.random.randn(*H_extracted_ori.shape) + 1j * np.random.randn(*H_extracted_ori.shape))
    
    # Full Doppler LS estimation (Original Uncompensated)
    rx_signal_clean_ori = tx_grid * H_extracted_ori
    sig_power_ori = np.mean(np.abs(rx_signal_clean_ori) ** 2, axis=(1, 2), keepdims=True)
    noise_var_ori = sig_power_ori / SNR_linear
    noise_realized_ori = noise_pattern * np.sqrt(noise_var_ori / 2.0)
    rx_grid_ori = rx_signal_clean_ori + noise_realized_ori
    
    h_LS_ori = rx_grid_ori / tx_grid
    h_LS_pilots_ori = h_LS_ori[:, pilot_symbols, pilot_subcarriers]
    
    # Precompensated LS estimation
    rx_signal_clean_comp = tx_grid * H_extracted_comp
    sig_power_comp = np.mean(np.abs(rx_signal_clean_comp) ** 2, axis=(1, 2), keepdims=True)
    noise_var_comp = sig_power_comp / SNR_linear
    noise_realized_comp = noise_pattern * np.sqrt(noise_var_comp / 2.0)
    rx_grid_comp = rx_signal_clean_comp + noise_realized_comp
    
    h_LS_comp = rx_grid_comp / tx_grid
    h_LS_pilots_comp = h_LS_comp[:, pilot_symbols, pilot_subcarriers]
    
    # Interpolation
    h_interp_ori_batch = []
    h_interp_comp_batch = []
    for idx_in_batch in range(current_batch_size):
        h_interp_o = interpolate_channel(rx_grid_ori[idx_in_batch], tx_grid[idx_in_batch], pilot_mask)
        h_interp_c = interpolate_channel(rx_grid_comp[idx_in_batch], tx_grid[idx_in_batch], pilot_mask)
        h_interp_ori_batch.append(h_interp_o)
        h_interp_comp_batch.append(h_interp_c)
        
    h_interp_ori_batch = np.stack(h_interp_ori_batch, axis=0)
    h_interp_comp_batch = np.stack(h_interp_comp_batch, axis=0)
    
    nmse_ls_list = []
    nmse_ls_pilot_list = []
    nmse_li_list = []
    nmse_li_ori_list = []
    ssim_ls_list = []
    ssim_li_list = []
    ssim_li_ori_list = []
    
    nmse_ls_batch = []
    nmse_ls_pilot_batch = []
    nmse_li_batch = []
    nmse_li_ori_batch = []
    ssim_ls_batch = []
    ssim_li_batch = []
    ssim_li_ori_batch = []
    
    for idx_in_batch in range(current_batch_size):
        h_ext_c = H_extracted_comp[idx_in_batch]
        h_ext_o = H_extracted_ori[idx_in_batch]
        
        # LS estimated full grid (un-interpolated)
        h_ls_c = h_LS_comp[idx_in_batch]
        
        # NMSE LS (compensated)
        n_ls = np.sum(np.abs(h_ext_c - h_ls_c)**2) / np.sum(np.abs(h_ext_c)**2)
        nmse_ls_batch.append(n_ls)
        
        # NMSE LS pilots (only at pilot locations)
        h_eff_pilots = h_ext_c[pilot_symbols, pilot_subcarriers]
        h_ls_pilots = h_LS_pilots_comp[idx_in_batch]
        n_ls_p = np.sum(np.abs(h_eff_pilots - h_ls_pilots)**2) / np.sum(np.abs(h_eff_pilots)**2)
        nmse_ls_pilot_batch.append(n_ls_p)
        
        # NMSE LI (interpolated compensated)
        h_interp_c = h_interp_comp_batch[idx_in_batch]
        n_li = np.sum(np.abs(h_ext_c - h_interp_c)**2) / np.sum(np.abs(h_ext_c)**2)
        nmse_li_batch.append(n_li)
        
        # NMSE LI ori (interpolated uncompensated)
        h_interp_o = h_interp_ori_batch[idx_in_batch]
        n_li_ori = np.sum(np.abs(h_ext_o - h_interp_o)**2) / np.sum(np.abs(h_ext_o)**2)
        nmse_li_ori_batch.append(n_li_ori)
        
        # SSIM values
        s_ls = compute_complex_ssim(h_ext_c, h_ls_c)
        s_li = compute_complex_ssim(h_ext_c, h_interp_c)
        s_li_ori = compute_complex_ssim(h_ext_o, h_interp_o)
        
        ssim_ls_batch.append(s_ls)
        ssim_li_batch.append(s_li)
        ssim_li_ori_batch.append(s_li_ori)
        
    nmse_ls_list.extend(nmse_ls_batch)
    nmse_ls_pilot_list.extend(nmse_ls_pilot_batch)
    nmse_li_list.extend(nmse_li_batch)
    nmse_li_ori_list.extend(nmse_li_ori_batch)
    ssim_ls_list.extend(ssim_ls_batch)
    ssim_li_list.extend(ssim_li_batch)
    ssim_li_ori_list.extend(ssim_li_ori_batch)
    
    # Accumulate batch results
    H_extracted_ori_list.append(H_extracted_ori)
    H_extracted_comp_list.append(H_extracted_comp)
    H_LS_ori_list.append(h_LS_pilots_ori)
    H_LS_comp_list.append(h_LS_pilots_comp)
    H_interp_ori_list.append(h_interp_ori_batch)
    H_interp_comp_list.append(h_interp_comp_batch)
    
    # 4. Plots ONLY for the very first batch, first sample
    if b == 0:
        print("Plotting geometry and channel visualizations once for the first sample...")
        
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
                r_ue_ECEF_p = r_ue_ECEF_all[0] + ue_vel_ECEF_all[0] * tp
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
            
            # Calculate satellite state at t = 0.0 (peak zenith overhead 90 deg) for 3D geometry plotting
            r_sat_ECEF_plot0, v_sat_ECEF_plot0 = get_satellite_state_ecef(
                0.0, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E
            )

            fig_ecef = plt.figure(figsize=(10, 8), facecolor='white')
            ax_ecef = fig_ecef.add_subplot(111, projection='3d')
            ax_ecef.set_facecolor('white')
            XS, YS, ZS = get_local_earth_patch(lambda_UE, phi_UE, a_wgs84, delta_deg=12.0, num_pts=30)
            ax_ecef.plot_surface(XS, YS, ZS, color=(0.85, 0.90, 0.95), edgecolor=(0.70, 0.75, 0.80), alpha=0.5, linewidth=0.25)
            ax_ecef.plot(sat_path_ECEF[0], sat_path_ECEF[1], sat_path_ECEF[2], '--', color=(0.85, 0.60, 0.05), linewidth=1.5, label='LEO Orbit (ECEF)')
            ax_ecef.plot(ue_path_ECEF[0], ue_path_ECEF[1], ue_path_ECEF[2], '-', color=(0.85, 0.20, 0.20), linewidth=2, label='UE 0 Path')
            
            # Draw Beam Footprint and Beam Center
            ax_ecef.plot(circle_ECEF[0], circle_ECEF[1], circle_ECEF[2], color='purple', linewidth=2, label='Beam Footprint (15 km radius)')
            ax_ecef.scatter(r_ue_ECEF_0[0], r_ue_ECEF_0[1], r_ue_ECEF_0[2], color='purple', marker='X', s=120, edgecolor='w', label='Beam Center')
            
            # Plot Satellite at t = 0.0 (Peak Zenith Overhead)
            ax_ecef.scatter(r_sat_ECEF_plot0[0], r_sat_ECEF_plot0[1], r_sat_ECEF_plot0[2], color=(0.85, 0.60, 0.05), s=150, edgecolor='w', label='LEO Satellite (t = 0 s, 90° Zenith)')
            ax_ecef.scatter(r_ue_ECEF_all[0, 0], r_ue_ECEF_all[0, 1], r_ue_ECEF_all[0, 2], color=(0.85, 0.20, 0.20), s=100, edgecolor='w', label='UE 0')
            ax_ecef.plot([r_ue_ECEF_all[0, 0], r_sat_ECEF_plot0[0]], 
                         [r_ue_ECEF_all[0, 1], r_sat_ECEF_plot0[1]], 
                         [r_ue_ECEF_all[0, 2], r_sat_ECEF_plot0[2]], 
                         '--', color=(0.6, 0.6, 0.6), linewidth=1.2, label='Line of Sight (t = 0 s)')
            
            # Draw Velocity Vectors (scaled for WGS-84 scale)
            vel_scale_sat = 1.0e5 / 7500 * 5
            vel_scale_ue = 1.0e5 / 50 * 3.5
            v_sat_scaled = v_sat_ECEF_plot0 * vel_scale_sat
            v_ue_scaled = ue_vel_ECEF_all[0] * vel_scale_ue

            ax_ecef.quiver(r_sat_ECEF_plot0[0], r_sat_ECEF_plot0[1], r_sat_ECEF_plot0[2],
                           v_sat_scaled[0], v_sat_scaled[1], v_sat_scaled[2],
                           color=(0.10, 0.65, 0.35), linewidth=1.5, arrow_length_ratio=0.25, label='Satellite Velocity Vector (Scaled)')
            ax_ecef.quiver(r_ue_ECEF_all[0, 0], r_ue_ECEF_all[0, 1], r_ue_ECEF_all[0, 2],
                           v_ue_scaled[0], v_ue_scaled[1], v_ue_scaled[2],
                           color=(0.85, 0.40, 0.05), linewidth=2.0, arrow_length_ratio=0.3, label='UE 0 Velocity Vector (Scaled)')

            zoom_width = 1.2e6
            ax_ecef.set_xlim(r_ue_ECEF_0[0] - zoom_width, r_ue_ECEF_0[0] + zoom_width)
            ax_ecef.set_ylim(r_ue_ECEF_0[1] - zoom_width, r_ue_ECEF_0[1] + zoom_width)
            ax_ecef.set_zlim(r_ue_ECEF_0[2] - zoom_width, r_ue_ECEF_0[2] + zoom_width)
            ax_ecef.set_xlabel('ECEF X (meters)')
            ax_ecef.set_ylabel('ECEF Y (meters)')
            ax_ecef.set_zlabel('ECEF Z (meters)')
            ax_ecef.set_title("3D Geometry Snapshot (ECEF Coordinates)\nPeak Zenith Overhead (t = 0 s, ~90° Elevation)")
            ax_ecef.legend(facecolor='white')
            plt.tight_layout()
            ecef_plot_filename = os.path.join(output_dir, "geometry_ecef.pdf")
            plt.savefig(ecef_plot_filename, format='pdf', bbox_inches='tight', pad_inches=0.02)
            plt.close('all')
        except Exception as ex:
            print(f"Warning: Could not save ECEF plot. Error: {ex}")
            
        # Plot Local Tangent Plane ENU Topology for UE 0 at t = 0.0 (Peak Zenith Overhead)
        try:
            bs_loc_ENU_plot0 = ecef_to_enu(r_sat_ECEF_plot0, r_ue_ECEF_0, lambda_UE, phi_UE)
            v_sat_ENU_plot0 = ecef_to_enu(v_sat_ECEF_plot0, np.zeros(3), lambda_UE, phi_UE)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(ut_loc_batch[0, 0], ut_loc_batch[0, 1], ut_loc_batch[0, 2], color='red', s=100, label='User Equipment (UE 0)')
            
            # Draw Beam Footprint circle and Beam Center
            ax.plot(circle_ENU[0], circle_ENU[1], np.zeros_like(circle_ENU[0]), color='purple', linewidth=2, linestyle='-', label='Beam Footprint (15 km)')
            ax.scatter(0, 0, 0, color='purple', marker='X', s=120, edgecolor='w', label='Beam Center')
            
            ax.scatter(bs_loc_ENU_plot0[0], bs_loc_ENU_plot0[1], bs_loc_ENU_plot0[2], color='blue', s=200, label='LEO Satellite (t = 0 s, 90° Zenith)')
            ax.plot([ut_loc_batch[0, 0], bs_loc_ENU_plot0[0]], 
                    [ut_loc_batch[0, 1], bs_loc_ENU_plot0[1]], 
                    [ut_loc_batch[0, 2], bs_loc_ENU_plot0[2]], 
                    color='gray', linestyle='--', alpha=0.7, label='Line of Sight (t = 0 s)')
            
            grid_range = 25000.0
            x_grid, y_grid = np.meshgrid(np.linspace(-grid_range, grid_range, 10), np.linspace(-grid_range, grid_range, 10))
            z_grid = np.zeros_like(x_grid)
            ax.plot_wireframe(x_grid, y_grid, z_grid, color='green', alpha=0.15, rstride=1, cstride=1)
            
            scale_ue_vel = 100.0  
            ax.quiver(ut_loc_batch[0, 0], ut_loc_batch[0, 1], ut_loc_batch[0, 2],
                      ut_vel_batch[0, 0]*scale_ue_vel, ut_vel_batch[0, 1]*scale_ue_vel, ut_vel_batch[0, 2]*scale_ue_vel,
                      color='orange', linewidth=2, label='UE 0 Velocity (Scaled)')
            
            scale_sat_vel = 5.0
            ax.quiver(bs_loc_ENU_plot0[0], bs_loc_ENU_plot0[1], bs_loc_ENU_plot0[2],
                      v_sat_ENU_plot0[0]*scale_sat_vel, v_sat_ENU_plot0[1]*scale_sat_vel, v_sat_ENU_plot0[2]*scale_sat_vel,
                      color='cyan', linewidth=2, label='Satellite Velocity (Scaled)')
            
            ax.set_xlabel('East (meters)')
            ax.set_ylabel('North (meters)')
            ax.set_zlabel('Up (meters)')
            ax.set_title("3D Geometry (Local Tangent Plane ENU)\nPeak Zenith Overhead (t = 0 s, ~90° Elevation)")
            ax.legend()
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, "geometry_plot.pdf")
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight', pad_inches=0.02)
            plt.close('all')
        except Exception as ex:
            print(f"Warning: Could not save ENU plot. Error: {ex}")
            
        # Plot Channels (Transposed, viridis colormap, cropped)
        try:
            real_full_t = np.real(H_extracted_ori[0]).T
            real_comp_t = np.real(H_extracted_comp[0]).T
            
            fig0, ax0 = plt.subplots(figsize=(8, 5), facecolor='white')
            im0 = ax0.imshow(real_full_t, aspect='auto', cmap='viridis', origin='lower')
            ax0.set_title(f"Original Perfect Channel (Full Doppler): Real Part of Channel 0 ({scenario.upper()})\nElevation: {elevation_angles_all[0]:.2f}°")
            ax0.set_xlabel("OFDM Symbol Index")
            ax0.set_ylabel("Subcarrier Index")
            fig0.colorbar(im0, ax=ax0)
            plt.tight_layout()
            channel_real_ori_filename = os.path.join(output_dir, f"channel_real_ori_{scenario}.pdf")
            plt.savefig(channel_real_ori_filename, format='pdf', bbox_inches='tight', pad_inches=0.01)
            plt.close(fig0)
            
            fig1, ax1 = plt.subplots(figsize=(8, 5), facecolor='white')
            im1 = ax1.imshow(real_comp_t, aspect='auto', cmap='viridis', origin='lower')
            ax1.set_title(f"Effective Compensated Channel (Precompensated Satellite Doppler): Real Part of Channel 0 ({scenario.upper()})\nElevation: {elevation_angles_all[0]:.2f}°")
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
H_extracted_ori_all = np.concatenate(H_extracted_ori_list, axis=0)
H_extracted_comp_all = np.concatenate(H_extracted_comp_list, axis=0)
H_LS_ori_all = np.concatenate(H_LS_ori_list, axis=0)
H_LS_comp_all = np.concatenate(H_LS_comp_list, axis=0)
H_interp_ori_all = np.concatenate(H_interp_ori_list, axis=0)
H_interp_comp_all = np.concatenate(H_interp_comp_list, axis=0)

# Convert channel grids to MATLAB shape [14, 132, N_samples]
H_perfect = np.transpose(H_extracted_comp_all, (1, 2, 0))        # [14, 132, N_samples] (Compensated)
H_perfect_ori = np.transpose(H_extracted_ori_all, (1, 2, 0))    # [14, 132, N_samples] (Uncompensated)
H_prac = np.array([], dtype=np.complex64)                       # Empty matrix to save storage space
H_li = np.transpose(H_interp_comp_all, (1, 2, 0))               # [14, 132, N_samples] (LS+LI on Compensated)
H_li_ori = np.transpose(H_interp_ori_all, (1, 2, 0))             # [14, 132, N_samples] (LS+LI on Uncompensated)
H_ls_pilots = np.transpose(H_LS_comp_all, (1, 0))               # [numPilots, N_samples] (Compensated)
H_ls_pilots_ori = np.transpose(H_LS_ori_all, (1, 0))             # [numPilots, N_samples] (Uncompensated)

# Pilot positions (MATLAB style: 1-indexed)
pilot_rows = pilot_subcarriers + 1
pilot_cols = pilot_symbols + 1

# Convert NMSE lists to arrays, and average SSIM lists to scalars
nmse_ls = np.array(nmse_ls_list)
nmse_ls_pilot = np.array(nmse_ls_pilot_list)
nmse_li = np.array(nmse_li_list)
nmse_li_ori = np.array(nmse_li_ori_list)
ssim_ls = float(np.mean(ssim_ls_list)) if len(ssim_ls_list) > 0 else 0.0
ssim_li = float(np.mean(ssim_li_list)) if len(ssim_li_list) > 0 else 0.0
ssim_li_ori = float(np.mean(ssim_li_ori_list)) if len(ssim_li_ori_list) > 0 else 0.0

# Calculate nominal common Doppler shift at the beam center (ENU [0,0,0])
lambda_0 = SPEED_OF_LIGHT / carrier_frequency
v_los_bc = -bs_loc_ENU
u_los_bc = v_los_bc / np.linalg.norm(v_los_bc)
doppler_sat_bc = np.dot(v_sat_ENU, u_los_bc) / lambda_0
print(f"Calculated Doppler shift at beam center: {doppler_sat_bc:.2f} Hz")

# Save .mat file
mat_filename = os.path.join(output_dir, f"channel_{scenario}_randomizedUE.mat")
mat_data = {
    # Main requested MATLAB matrices
    "H_perfect": H_perfect,
    "H_perfect_ori": H_perfect_ori,
    "H_prac": H_prac,
    "H_li": H_li,
    "H_li_ori": H_li_ori,
    "H_ls_pilots": H_ls_pilots,
    "H_ls_pilots_ori": H_ls_pilots_ori,
    "pilot_rows": pilot_rows,
    "pilot_cols": pilot_cols,
    "nmse_ls": nmse_ls,
    "nmse_ls_pilot": nmse_ls_pilot,
    "nmse_li": nmse_li,
    "nmse_li_ori": nmse_li_ori,
    "ssim_ls": ssim_ls,
    "ssim_li": ssim_li,
    "ssim_li_ori": ssim_li_ori,
    "doppler_sat_bc": doppler_sat_bc,
    
    # Metadata and other parameters
    "pilot_symbols": pilot_symbols + 1,       
    "pilot_subcarriers": pilot_subcarriers + 1, 
    "ut_loc_ENU": ut_loc_ENU_all,       
    "ut_velocity_ENU": ut_velocity_ENU_all, 
    "bs_loc_ENU": bs_loc_ENU,
    "bs_velocity_ENU": v_sat_ENU,
    "sat_speed": float(sat_speed)
}
savemat(mat_filename, mat_data)
print(f"Successfully saved batched randomized channel data of {N_samples} samples to {mat_filename}")

# Save scenario parameters documentation
delay_spreads_ns = np.array(delay_spreads_all) * 1e9
avg_delay_spread_ns = np.mean(delay_spreads_ns) if len(delay_spreads_ns) > 0 else 0.0
min_delay_spread_ns = np.min(delay_spreads_ns) if len(delay_spreads_ns) > 0 else 0.0
max_delay_spread_ns = np.max(delay_spreads_ns) if len(delay_spreads_ns) > 0 else 0.0

md_filename = os.path.join(output_dir, f"readme_{scenario}_randomizedUE.md")

if delay_spread_ns_custom is not None:
    ds_val_str = f"{delay_spread_ns_custom:.1f} ns (Custom Overridden, Fixed)" if fix_delay_spread else f"{delay_spread_ns_custom:.1f} ns (Custom Overridden, Log-Normal)"
else:
    ds_val_str = "Standard 3GPP TR 38.811"

target_elev_str = f"{target_elevation_angle:.1f}°" if target_elevation_angle is not None else "90.0° (Peak Zenith Overhead)"

md_content = f"""# Channel & Geometry Generation Settings - {scenario.upper()} (Randomized UE)

- **Scenario Type**: {scenario.upper()} (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: {carrier_frequency / 1e9:.2f} GHz
- **Link Direction**: {direction}
- **Satellite (LEO) Height**: {satellite_height / 1000:.0f} km
- **Configured Target Elevation Angle**: {target_elev_str}
- **Nominal Beam-Center Elevation Angle**: {elevation_angle_nom:.2f}° (Snapshot time t = {t_snapshot:.1f} s)
- **Subcarrier Spacing (SCS)**: {SCS / 1e3:.0f} kHz
- **FFT Size**: {nFFT}
- **Active Subcarriers**: 132 (out of {nFFT})
- **SNR (for LS estimation)**: {SNR_dB} dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 11]
- **Pilot Density**: {"Dense (264 pilots)" if dense_pilot else "Sparse (88 pilots, subcarrier mod 6 = 0 or 1 on symbols 2 and 11)"}
- **Total Samples Generated**: {N_samples}
- **Target Delay Spread Configuration**: {ds_val_str}
- **Average RMS Delay Spread (Realized)**: {avg_delay_spread_ns:.2f} ns (Range: [{min_delay_spread_ns:.2f}, {max_delay_spread_ns:.2f}] ns)

## Satellite Orbital Pass & Elevation Angle Timeline
- **Pass Start (t_start = {t_min:.1f} s)**: Elevation = {elev_at_tmin:.2f}° (Horizon Rise)
- **Peak Zenith (t_peak = 0.0 s)**: Elevation = {elev_at_tpeak:.2f}° (Overhead Peak)
- **Snapshot Point (t_snap = {t_snapshot:.1f} s)**: Elevation = {elevation_angle_nom:.2f}° (Single Position Generated)
- **Pass End (t_end = {t_max:.1f} s)**: Elevation = {elev_at_tmax:.2f}° (Horizon Set)

## Spatial Elevation Variation Across 15km Beam Footprint ({N_samples} UEs)
- **UE Farthest from Satellite (Min Elevation)**: {np.min(elevation_angles_all):.2f}°
- **UE Closest to Satellite (Max Elevation)**: {np.max(elevation_angles_all):.2f}°
- **Average Across All UEs (Mean Elevation)**: {np.mean(elevation_angles_all):.2f}°

## Satellite (LEO) Settings (Fixed Snapshot)
- **Temporal State**: Single snapshot at orbital time $t = {t_snapshot:.1f}$ seconds
- **Satellite Position (ENU)**: Fixed at [{bs_loc_ENU[0]:.2f}, {bs_loc_ENU[1]:.2f}, {bs_loc_ENU[2]:.2f}] meters
- **Satellite Velocity Vector (ENU)**: Fixed at [{v_sat_ENU[0]:.2f}, {v_sat_ENU[1]:.2f}, {v_sat_ENU[2]:.2f}] m/s (Speed: {sat_speed:.2f} m/s)

## Beam Boresight & Footprint Settings
- **Beam Center (ECEF)**: [{r_ue_ECEF_0[0]:.2f}, {r_ue_ECEF_0[1]:.2f}, {r_ue_ECEF_0[2]:.2f}] meters
- **Beam Center (ENU)**: [0.00, 0.00, 0.00] meters (Origin of local tangent plane)
- **Beam Footprint Radius**: {r_beam / 1000:.1f} km

## UE Randomization Settings
- **Generation Method**: Randomized UE Positions and Velocities (GPU Mini-Batched)
- **Position Area (ENU)**: 
  * Shape: Uniformly distributed inside a circle of radius {r_ue_max/1000:.2f} km around the beam center
  * Height (Z): {z_val} meters above ground
- **Velocity (ENU)**:
  * Speed Range: [{v_min}, {v_max}] m/s
  * Heading (Direction): Randomized uniformly over [0, 360] degrees (full direction randomization across all generated samples)
"""

with open(md_filename, "w") as f:
    f.write(md_content)
print(f"Saved scenario parameters documentation to {md_filename}")

