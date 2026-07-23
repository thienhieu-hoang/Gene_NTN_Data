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
from OpenNTN import DenseUrban, Urban, SubUrban
from helper import ecef_to_enu, get_satellite_state_ecef, get_local_earth_patch, save_simulation_readme
from OpenNTN.utils import compute_satellite_speed
from OpenNTN.utils import cir_to_time_channel, time_lag_discrete_time_channel
from OpenNTN.utils import compute_stallite_doppler as compute_stallite_doppler
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.ofdm import RemoveNulledSubcarriers

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

# Calculate coordinates at snapshot t = 0
t = 0.0
r_sat_ECEF, v_sat_ECEF = get_satellite_state_ecef(
    t, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E
)

r_ue_ECEF = r_ue_ECEF_0 + ue_vel_ECEF * t
r_ue_ECEF = r_ue_ECEF * (np.linalg.norm(r_ue_ECEF_0) / np.linalg.norm(r_ue_ECEF))

# Calculate exact physical Elevation Angle and Slant Range
v_los = r_sat_ECEF - r_ue_ECEF
slant_range = np.linalg.norm(v_los)
u_normal = r_ue_ECEF / np.linalg.norm(r_ue_ECEF)
u_los = v_los / slant_range
elev_rad = np.arcsin(np.dot(u_normal, u_los))
elevation_angle = float(np.degrees(elev_rad))

# Convert ECEF positions and velocities to local Cartesian Oxyz (ENU) frame
ut_loc_ENU = ecef_to_enu(r_ue_ECEF, r_ue_ECEF_0, lambda_UE, phi_UE)
ut_loc_ENU[2] = 1.5  # Set vertical antenna height to standard 1.5m above GCS ground

bs_loc_ENU = ecef_to_enu(r_sat_ECEF, r_ue_ECEF_0, lambda_UE, phi_UE)
v_sat_ENU = ecef_to_enu(v_sat_ECEF, np.zeros(3), lambda_UE, phi_UE)
sat_speed = np.linalg.norm(v_sat_ECEF)

# =========================================================================
# 2. SYSTEM SIMULATION CONFIGURATION
# =========================================================================
scenario = "dur"            # dur (Dense Urban), sur (SubUrban), urb (Urban)
carrier_frequency = 27e9     # DL carrier frequency (Hz)
direction = "downlink"
num_ut = 1
batch_size = 1               # 1 batch for 1 snapshot
N_samples = 1

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
num_time_steps = 14
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

# Calculate subcarrier frequencies
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)

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

# Initialize accumulators
H_eff_all = []
H_LS_est_all = []
H_interp_all = []
delay_spreads_all = []

SNR_dB = 20.0
SNR_linear = 10.0 ** (SNR_dB / 10.0)
qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / np.sqrt(2)
pilot_mask = tf.squeeze(rg.pilot_pattern.mask).numpy()

# =========================================================================
# 4. NTN CHANNEL & TOPOLOGY SETUP
# =========================================================================
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

# Override clusters based on elevation angle (rounded to nearest 10 degrees)
rounded_elev = int(round(elevation_angle / 10.0) * 10)
rounded_elev = max(10, min(90, rounded_elev))
channel_model._scenario._params_nlos[f"numClusters_{rounded_elev}"] = 3

# Pack local ENU coordinates into tensors
ut_loc_tensor = tf.constant([[ut_loc_ENU]], dtype=tf.float32)         # Shape [1, 1, 3]
bs_loc_tensor = tf.constant([[bs_loc_ENU]], dtype=tf.float32)         # Shape [1, 1, 3]
ut_orientations = tf.zeros([1, 1, 3])                                # Shape [1, 1, 3]
bs_orientations = tf.zeros([1, 1, 3])                                # Shape [1, 1, 3]
ut_velocities_tensor = tf.constant([[v_UE_ENU]], dtype=tf.float32)   # Shape [1, 1, 3]
in_state = tf.constant([[False]], dtype=tf.bool)                     # Shape [1, 1]

# Apply custom ENU geometry to the topology
topology_data = (ut_loc_tensor, bs_loc_tensor, ut_orientations, bs_orientations, ut_velocities_tensor, in_state)
channel_model.set_topology(*topology_data, los=True)

# Generate channel path coefficients and delays
num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)

# Calculate RMS delay spread for this drop
coefs_np = np.abs(path_coefficients.numpy())**2
delays_np = path_delays.numpy()
for b in range(batch_size):
    p_b = np.mean(coefs_np[b], axis=(0, 1, 2, 3, 5))
    tau_b = delays_np[b, 0, 0, :]
    sum_p = np.sum(p_b)
    if sum_p > 0:
        mean_tau = np.sum(p_b * tau_b) / sum_p
        rms_ds = np.sqrt(np.sum(p_b * (tau_b - mean_tau)**2) / sum_p)
        delay_spreads_all.append(rms_ds)

# OFDM Channel Generation
ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid=rg)
h_freq = ofdm_channel() # shape [1, 1, 1, 1, 2, 14, 256]

remove_nulled = RemoveNulledSubcarriers(rg)
h_eff = remove_nulled(h_freq)   # shape [1, 1, 1, 1, 2, 14, 132]

# Extract SISO channel for the first BS antenna to UT antenna
h_eff_siso = h_eff[:, 0, 0, 0, 0, :, :].numpy() # shape [1, 14, 132]
H_eff_all.append(h_eff_siso)

# Generate QPSK tx_grid
tx_grid = np.random.choice(qpsk_symbols, size=[batch_size, 14, 132])

# Pass through the fading channel and add noise
rx_signal_clean = tx_grid * h_eff_siso
sig_power = np.mean(np.abs(rx_signal_clean) ** 2)
noise_var = sig_power / SNR_linear
noise = (np.random.randn(*h_eff_siso.shape) + 1j * np.random.randn(*h_eff_siso.shape)) * np.sqrt(noise_var / 2.0)
rx_grid = rx_signal_clean + noise

# Least-Squares estimation
h_LS = rx_grid / tx_grid # shape [1, 14, 132]
pilot_symbols, pilot_subcarriers = np.where(pilot_mask)
h_LS_pilots = h_LS[:, pilot_symbols, pilot_subcarriers]
H_LS_est_all.append(h_LS_pilots)

# Linear channel interpolation
h_interp_batch = []
for b in range(batch_size):
    h_interp_b = interpolate_channel(rx_grid[b], tx_grid[b], pilot_mask)
    h_interp_batch.append(h_interp_b)
h_interp_batch = np.stack(h_interp_batch, axis=0) # [1, 14, 132]
H_interp_all.append(h_interp_batch)

# Concatenate results
H_eff_total = np.concatenate(H_eff_all, axis=0)      # [1, 14, 132]
H_LS_total = np.concatenate(H_LS_est_all, axis=0)    # [1, num_pilots]
H_interp_total = np.concatenate(H_interp_all, axis=0) # [1, 14, 132]

# Set up dynamically named results directory
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_name = f"{scenario.upper()}_{int(carrier_frequency/1e9)}G_{int(satellite_height/1000)}km_{int(ue_speed)}ms_{int(SNR_dB)}dB"
output_dir = os.path.join(script_dir, "results", dir_name)
os.makedirs(output_dir, exist_ok=True)

# Save .mat file containing channel estimates, coordinates, and velocities
mat_filename = os.path.join(output_dir, f"channel_{scenario}.mat")
mat_data = {
    "H_eff": H_eff_total,
    "H_LS_values": H_LS_total,
    "pilot_symbols": pilot_symbols + 1,       # Convert to 1-indexed for MATLAB
    "pilot_subcarriers": pilot_subcarriers + 1, # Convert to 1-indexed for MATLAB
    "H_interp": H_interp_total,
    "ut_loc_ENU": ut_loc_ENU,           # UE local ENU position [x, y, z] in meters
    "bs_loc_ENU": bs_loc_ENU,           # SAT local ENU position [x, y, z] in meters
    "ut_velocity_ENU": v_UE_ENU,        # UE velocity vector [vx, vy, vz] in m/s
    "bs_velocity_ENU": v_sat_ENU,       # SAT velocity vector [vx, vy, vz] in m/s
    "sat_speed": float(sat_speed),      # SAT scalar speed in m/s
    "elevation_angle": elevation_angle, # Calculated physical elevation angle (deg)
    "slant_range": slant_range          # Calculated slant range (m)
}
savemat(mat_filename, mat_data)
print(f"Saved channel and geometry simulation data to {mat_filename}")

# =========================================================================
# 5. GEOMETRY SNAPSHOT PLOTTING (ECEF Coordinate System)
# =========================================================================
try:
    # Generate orbit and UE tracks in ECEF for plotting (+/- 150 seconds)
    t_path = np.linspace(-150, 150, 100)
    sat_path_ECEF = []
    ue_path_ECEF = []
    for tp in t_path:
        r_sat_ECEF_p, _ = get_satellite_state_ecef(
            tp, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E
        )
        sat_path_ECEF.append(r_sat_ECEF_p)
        
        r_ue_ECEF_p = r_ue_ECEF_0 + ue_vel_ECEF * tp
        r_ue_ECEF_p = r_ue_ECEF_p * (np.linalg.norm(r_ue_ECEF_0) / np.linalg.norm(r_ue_ECEF_p))
        ue_path_ECEF.append(r_ue_ECEF_p)
        
    sat_path_ECEF = np.array(sat_path_ECEF).T
    ue_path_ECEF = np.array(ue_path_ECEF).T

    # Create ECEF 3D Plot
    fig_ecef = plt.figure(figsize=(10, 8), facecolor='white')
    ax_ecef = fig_ecef.add_subplot(111, projection='3d')
    ax_ecef.set_facecolor('white')

    # Draw Earth surface patch (rather than full sphere)
    # centered around the UE longitude (lambda_UE) and latitude (phi_UE)
    XS, YS, ZS = get_local_earth_patch(lambda_UE, phi_UE, a_wgs84, delta_deg=12.0, num_pts=30)
            
    ax_ecef.plot_surface(XS, YS, ZS, color=(0.85, 0.90, 0.95), edgecolor=(0.70, 0.75, 0.80), alpha=0.5, linewidth=0.25)

    # Plot Orbit Path and UE Path
    ax_ecef.plot(sat_path_ECEF[0], sat_path_ECEF[1], sat_path_ECEF[2], '--', color=(0.85, 0.60, 0.05), linewidth=1.5, label='LEO Orbit (ECEF)')
    ax_ecef.plot(ue_path_ECEF[0], ue_path_ECEF[1], ue_path_ECEF[2], '-', color=(0.85, 0.20, 0.20), linewidth=2, label='UE Path')

    # Plot current position at t=0
    ax_ecef.scatter(r_sat_ECEF[0], r_sat_ECEF[1], r_sat_ECEF[2], color=(0.85, 0.60, 0.05), s=150, edgecolor='w', label='LEO Satellite')
    ax_ecef.scatter(r_ue_ECEF[0], r_ue_ECEF[1], r_ue_ECEF[2], color=(0.85, 0.20, 0.20), s=100, edgecolor='w', label='UE')

    # Draw Line of Sight
    ax_ecef.plot([r_ue_ECEF[0], r_sat_ECEF[0]], 
                 [r_ue_ECEF[1], r_sat_ECEF[1]], 
                 [r_ue_ECEF[2], r_sat_ECEF[2]], 
                 '--', color=(0.6, 0.6, 0.6), linewidth=1.2, label='Line of Sight (LoS)')

    # Draw Velocity Vectors (scaled for WGS-84 scale)
    vel_scale_sat = 1.0e5 / 7500 * 5
    vel_scale_ue = 1.0e5 / 50 * 3.5
    v_sat_scaled = v_sat_ECEF * vel_scale_sat
    v_ue_scaled = ue_vel_ECEF * vel_scale_ue

    ax_ecef.quiver(r_sat_ECEF[0], r_sat_ECEF[1], r_sat_ECEF[2],
                   v_sat_scaled[0], v_sat_scaled[1], v_sat_scaled[2],
                   color=(0.10, 0.65, 0.35), linewidth=1.5, arrow_length_ratio=0.25, label='Satellite Velocity Vector (Scaled)')
    ax_ecef.quiver(r_ue_ECEF[0], r_ue_ECEF[1], r_ue_ECEF[2],
                   v_ue_scaled[0], v_ue_scaled[1], v_ue_scaled[2],
                   color=(0.85, 0.40, 0.05), linewidth=2.0, arrow_length_ratio=0.3, label='UE Velocity Vector (Scaled)')

    # Zoom around the UE position
    zoom_width = 1.2e6
    ax_ecef.set_xlim(r_ue_ECEF_0[0] - zoom_width, r_ue_ECEF_0[0] + zoom_width)
    ax_ecef.set_ylim(r_ue_ECEF_0[1] - zoom_width, r_ue_ECEF_0[1] + zoom_width)
    ax_ecef.set_zlim(r_ue_ECEF_0[2] - zoom_width, r_ue_ECEF_0[2] + zoom_width)

    ax_ecef.set_xlabel('ECEF X (meters)')
    ax_ecef.set_ylabel('ECEF Y (meters)')
    ax_ecef.set_zlabel('ECEF Z (meters)')
    ax_ecef.set_title(f"3D NTN Geometry Snapshot (ECEF Coordinates)\nElevation: {elevation_angle:.2f}° | Range: {slant_range/1000:.2f} km")
    ax_ecef.legend(facecolor='white')
    plt.tight_layout()

    ecef_plot_filename = os.path.join(output_dir, "geometry_ecef.pdf")
    plt.savefig(ecef_plot_filename, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close('all')
    print(f"Saved ECEF 3D geometry plot to {ecef_plot_filename}")

except Exception as e:
    print(f"Warning: Could not save ECEF geometry plot. Error: {e}")

# =========================================================================
# 6. GEOMETRY SNAPSHOT PLOTTING (Local Oxyz ENU)
# =========================================================================
try:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # Plot the UE as a point
    ax.scatter(ut_loc_ENU[0], ut_loc_ENU[1], ut_loc_ENU[2], color='red', s=100, label='User Equipment (UE)')

    # Plot the Satellite as a point
    ax.scatter(bs_loc_ENU[0], bs_loc_ENU[1], bs_loc_ENU[2], color='blue', s=200, label='LEO Satellite')

    # Draw the line of sight
    ax.plot([ut_loc_ENU[0], bs_loc_ENU[0]], 
            [ut_loc_ENU[1], bs_loc_ENU[1]], 
            [ut_loc_ENU[2], bs_loc_ENU[2]], 
            color='gray', linestyle='--', alpha=0.7, label='Line of Sight (LoS)')

    # Draw the ground plane grid at z = 0
    grid_range = 25000.0  # 25 km range
    x_grid, y_grid = np.meshgrid(np.linspace(-grid_range, grid_range, 10), 
                                 np.linspace(-grid_range, grid_range, 10))
    z_grid = np.zeros_like(x_grid)
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='green', alpha=0.15, rstride=1, cstride=1)

    # Draw UE Velocity vector (scaled up for visibility)
    scale_ue_vel = 100.0  
    ax.quiver(ut_loc_ENU[0], ut_loc_ENU[1], ut_loc_ENU[2],
              v_UE_ENU[0]*scale_ue_vel, v_UE_ENU[1]*scale_ue_vel, v_UE_ENU[2]*scale_ue_vel,
              color='orange', linewidth=2, label='UE Velocity Vector (Scaled)')

    # Draw Satellite Velocity vector (scaled up for visibility)
    scale_sat_vel = 5.0
    ax.quiver(bs_loc_ENU[0], bs_loc_ENU[1], bs_loc_ENU[2],
              v_sat_ENU[0]*scale_sat_vel, v_sat_ENU[1]*scale_sat_vel, v_sat_ENU[2]*scale_sat_vel,
              color='cyan', linewidth=2, label='Satellite Velocity Vector (Scaled)')

    ax.set_xlabel('East (meters)')
    ax.set_ylabel('North (meters)')
    ax.set_zlabel('Up (meters)')
    ax.set_title(f"3D Geometry Snapshot (Local Tangent Plane ENU)\nElevation: {elevation_angle:.2f}° | Range: {slant_range/1000:.2f} km")
    ax.legend()
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, "geometry_plot.pdf")
    plt.savefig(plot_filename, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close('all')
    print(f"Saved custom local tangent plane geometry plot to {plot_filename}")
except Exception as e:
    print(f"Warning: Could not save ENU geometry plot. Error: {e}")

# Save scenario parameters and delay spread documentation
avg_delay_spread_ns = np.mean(delay_spreads_all) * 1e9 if len(delay_spreads_all) > 0 else 0.0
md_filename = os.path.join(output_dir, f"readme_{scenario}.md")
save_simulation_readme(
    md_filename, scenario, carrier_frequency, direction, elevation_angle,
    satellite_height, ue_speed, SCS, nFFT, SNR_dB, N_samples, avg_delay_spread_ns
)
