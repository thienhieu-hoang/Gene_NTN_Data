# =========================================================================
# LEO Satellite and UE 3D Orbit & Doppler Visualization Script (Python Edition)
# =========================================================================
# Antigravity Coding Assistant - Google DeepMind
# =========================================================================

import os
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

# =========================================================================
# 1. CONFIGURATION PARAMETERS
# =========================================================================

# --- Key System Settings (Configurable) ---
f_c = 2.0e9                  # Carrier frequency (Hz) (e.g., 2.0 GHz S-band)

# --- Satellite Orbital Parameters ---
h_s = 600e3                  # Orbit altitude (m) (600 km LEO)
inclination_deg = 55.0       # Orbit inclination (degrees)
# Note: RAAN (Omega) and initial argument of latitude (u_0) will be dynamically 
# calculated to align the orbit pass directly over the UE.

# --- Beam Boresight & Footprint Parameters ---
beamwidth_deg = 3.5          # Half-Power Beamwidth (HPBW) (degrees)
# Beam pointing mode:
#   'fixed'    - Boresight points at a fixed coordinate on the ground (UE's initial position)
#   'nadir'    - Boresight points straight down (sub-satellite point)
#   'tracking' - Boresight dynamically steers to track the moving UE
beamMode = 'fixed' 

# --- User Equipment (UE) Parameters ---
phi_UE_deg = 37.7749         # UE Initial Latitude (degrees, e.g., San Francisco)
lambda_UE_deg = -122.4194    # UE Initial Longitude (degrees)
h_UE = 100.0                 # UE Altitude above ellipsoid (m)
v_UE_ground = 50.0           # UE speed along the ground (m/s, ~180 km/h)
heading_deg = 45.0           # UE heading azimuth (degrees, 0 = North, 90 = East)

# --- Visualization & Time Controls ---
zoomView = True              # True: Zoomed in on UE region (recommended), False: Global Earth view
time_step = 1.0              # Time step for simulation (s)
time_duration = 300          # Simulation window: [-t_duration/2, +t_duration/2] (s)

# --- Velocity Vector Scaling for Plotting (in meters per m/s) ---
vel_scale_sat = 1.0e5 / 7500 * 5   # Scales satellite arrow to ~500 km
vel_scale_ue = 1.0e5 / 50 * 3.5    # Scales UE arrow to ~350 km
vel_scale_bc = 1.0e5 / 7000 * 4    # Scales beam center arrow to ~400 km

# --- Environmental & Physical Constants (Static) ---
c = 299792458                # Speed of light (m/s)
omega_E = 7.292115e-5        # Earth's rotation rate (rad/s)
mu = 3.986004418e14          # Earth's gravitational parameter (m^3/s^2)

# --- WGS-84 Ellipsoid Parameters ---
a = 6378137.0                # Semi-major axis (m)
e2 = 6.69437999e-3           # First eccentricity squared

# =========================================================================
# 2. GEOMETRIC PREPARATION & ORBIT ALIGNMENT
# =========================================================================

# Convert angles to radians
inclination = np.deg2rad(inclination_deg)
phi_UE = np.deg2rad(phi_UE_deg)
lambda_UE = np.deg2rad(lambda_UE_deg)
heading = np.deg2rad(heading_deg)
theta_b = np.deg2rad(beamwidth_deg / 2)  # Half-beamwidth

# Orbit radius & speed
r_orbit = a + h_s
omega_s = np.sqrt(mu / r_orbit**3)
v_sat_orbit = np.sqrt(mu / r_orbit)

# Calculate UE's initial ECEF position
N_phi_0 = a / np.sqrt(1.0 - e2 * np.sin(phi_UE)**2)
r_ue_ECEF_0 = np.array([
    (N_phi_0 + h_UE) * np.cos(phi_UE) * np.cos(lambda_UE),
    (N_phi_0 + h_UE) * np.cos(phi_UE) * np.sin(lambda_UE),
    (N_phi_0 * (1.0 - e2) + h_UE) * np.sin(phi_UE)
])

# Calculate UE's velocity vector in local ENU frame
v_UE_ENU = np.array([v_UE_ground * np.sin(heading), v_UE_ground * np.cos(heading), 0.0])

# Rotation matrix from local ENU to ECEF at UE's initial location
R_ENU2ECEF = np.array([
    [-np.sin(lambda_UE), -np.sin(phi_UE) * np.cos(lambda_UE), np.cos(phi_UE) * np.cos(lambda_UE)],
    [ np.cos(lambda_UE), -np.sin(phi_UE) * np.sin(lambda_UE), np.cos(phi_UE) * np.sin(lambda_UE)],
    [ 0.0,                np.cos(phi_UE),                 np.sin(phi_UE)]
])

# UE velocity vector in ECEF frame
ue_vel_ECEF = R_ENU2ECEF @ v_UE_ENU

# --- Analytical Orbit Alignment (closest approach at t_mid) ---
# Find orbit argument of latitude (u_mid) when satellite is at UE's latitude
if inclination >= abs(phi_UE):
    u_mid = np.arcsin(np.sin(phi_UE) / np.sin(inclination))
else:
    u_mid = np.sign(phi_UE) * np.pi / 2.0

# Find RAAN (Omega) to align satellite longitude with UE longitude at t_mid
Omega_RAAN = lambda_UE - np.arctan2(np.sin(u_mid) * np.cos(inclination), np.cos(u_mid))

# Define time grid
t_mid = 0.0
time_grid = np.arange(-time_duration/2, time_duration/2 + time_step, time_step)
N_steps = len(time_grid)

# =========================================================================
# 3. TIME-VARYING DYNAMICS PRECOMPUTATION
# =========================================================================

sat_ECEF_all = np.zeros((3, N_steps))
sat_vel_ECEF_all = np.zeros((3, N_steps))
ue_ECEF_all = np.zeros((3, N_steps))
bc_ECEF_all = np.zeros((3, N_steps))
bc_vel_ECEF_all = np.zeros((3, N_steps))

slant_range_all = np.zeros(N_steps)
elev_all = np.zeros(N_steps)
doppler_all = np.zeros(N_steps)
doppler_beam_all = np.zeros(N_steps)

for k, t in enumerate(time_grid):
    # Earth rotation angle (GST) at time t
    theta_G = omega_E * t
    R_z = np.array([
        [ np.cos(theta_G), np.sin(theta_G), 0.0],
        [-np.sin(theta_G), np.cos(theta_G), 0.0],
        [ 0.0,             0.0,             1.0]
    ])
    
    # Satellite position & velocity in ECI
    u_t = omega_s * (t - t_mid) + u_mid
    
    r_sat_ECI = np.array([
        r_orbit * (np.cos(u_t)*np.cos(Omega_RAAN) - np.sin(u_t)*np.sin(Omega_RAAN)*np.cos(inclination)),
        r_orbit * (np.cos(u_t)*np.sin(Omega_RAAN) + np.sin(u_t)*np.cos(Omega_RAAN)*np.cos(inclination)),
        r_orbit * np.sin(u_t)*np.sin(inclination)
    ])

    v_sat_ECI = np.array([
        v_sat_orbit * (-np.sin(u_t)*np.cos(Omega_RAAN) - np.cos(u_t)*np.sin(Omega_RAAN)*np.cos(inclination)),
        v_sat_orbit * (-np.sin(u_t)*np.sin(Omega_RAAN) + np.cos(u_t)*np.cos(Omega_RAAN)*np.cos(inclination)),
        v_sat_orbit * np.cos(u_t)*np.sin(inclination)
    ])
    
    # Convert Satellite position and velocity to ECEF (accounts for Coriolis term)
    r_sat_ECEF = R_z @ r_sat_ECI
    
    omega_cross_r = np.array([
        -omega_E * r_sat_ECI[1],
         omega_E * r_sat_ECI[0],
         0.0
    ])
    v_sat_ECEF = R_z @ (v_sat_ECI - omega_cross_r)
    
    sat_ECEF_all[:, k] = r_sat_ECEF
    sat_vel_ECEF_all[:, k] = v_sat_ECEF
    
    # UE Position in ECEF (maintain exact altitude)
    r_ue_ECEF = r_ue_ECEF_0 + ue_vel_ECEF * t
    r_ue_ECEF = r_ue_ECEF * (np.linalg.norm(r_ue_ECEF_0) / np.linalg.norm(r_ue_ECEF))
    ue_ECEF_all[:, k] = r_ue_ECEF
    
    # Slant Range and Elevation Angle from UE to Satellite
    v_los = r_sat_ECEF - r_ue_ECEF
    slant_range = np.linalg.norm(v_los)
    slant_range_all[k] = slant_range
    
    u_normal = r_ue_ECEF / np.linalg.norm(r_ue_ECEF)
    u_los = v_los / slant_range
    elev_rad = np.arcsin(np.dot(u_normal, u_los))
    elev_all[k] = np.degrees(elev_rad)
    
    # Doppler Shift (UE-specific)
    v_rel_ECEF = v_sat_ECEF - ue_vel_ECEF
    doppler_all[k] = - (np.dot(v_rel_ECEF, u_los) / c) * f_c
    
    # Beam Center Position & Velocity in ECEF
    if beamMode == 'fixed':
        r_bc = r_ue_ECEF_0
        v_bc = np.array([0.0, 0.0, 0.0])
    elif beamMode == 'nadir':
        r_bc = a * (r_sat_ECEF / np.linalg.norm(r_sat_ECEF))
        v_bc = (a / np.linalg.norm(r_sat_ECEF)) * v_sat_ECEF
    elif beamMode == 'tracking':
        r_bc = r_ue_ECEF
        v_bc = ue_vel_ECEF
        
    bc_ECEF_all[:, k] = r_bc
    bc_vel_ECEF_all[:, k] = v_bc
    
    # Beam-level Doppler Shift (to beam center)
    v_los_beam = r_sat_ECEF - r_bc
    slant_range_beam = np.linalg.norm(v_los_beam)
    u_los_beam = v_los_beam / slant_range_beam
    v_rel_beam = v_sat_ECEF - v_bc
    doppler_beam_all[k] = - (np.dot(v_rel_beam, u_los_beam) / c) * f_c

# Residual Doppler shift
doppler_residual_all = doppler_all - doppler_beam_all

# =========================================================================
# 4. GRAPHICS INITIALIZATION
# =========================================================================

# Color palette definition (clean white theme)
earth_color = (0.85, 0.90, 0.95)
grid_color = (0.70, 0.75, 0.80)
gold = (0.85, 0.60, 0.05)
red = (0.85, 0.20, 0.20)
cyan = (0.00, 0.55, 0.65)
light_cyan = (0.00, 0.60, 0.75)
emerald = (0.10, 0.65, 0.35)
orange = (0.85, 0.40, 0.05)

# Setup Figure 1: 3D Visualization
fig1 = plt.figure(num='3D NTN Geometry Simulation', figsize=(8.5, 8.0), facecolor='white')
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_facecolor('white')

# Draw Earth sphere (Optimized: 20x20 mesh for 5x faster rendering)
u_sp = np.linspace(0, 2 * np.pi, 20)
v_sp = np.linspace(0, np.pi, 20)
XS = a * np.outer(np.cos(u_sp), np.sin(v_sp))
YS = a * np.outer(np.sin(u_sp), np.sin(v_sp))
ZS = a * np.outer(np.ones(np.size(u_sp)), np.cos(v_sp))
surf_earth = ax1.plot_surface(XS, YS, ZS, color=earth_color, edgecolor=grid_color, alpha=0.6, linewidth=0.25)

# Draw Equator & Prime Meridian
t_pm = np.linspace(0, 2 * np.pi, 100)
ax1.plot(a * np.cos(t_pm), a * np.sin(t_pm), np.zeros_like(t_pm), color=(0.3, 0.4, 0.5), linewidth=1.0)
ax1.plot(a * np.cos(t_pm), np.zeros_like(t_pm), a * np.sin(t_pm), color=(0.3, 0.4, 0.5), linewidth=0.5)

# Plot static paths
plot_orbit, = ax1.plot(sat_ECEF_all[0, :], sat_ECEF_all[1, :], sat_ECEF_all[2, :], 
                       '--', color=gold, alpha=0.5, linewidth=1.5, label='LEO Orbit (ECEF)')
plot_ue_path, = ax1.plot(ue_ECEF_all[0, :], ue_ECEF_all[1, :], ue_ECEF_all[2, :], 
                         '-', color=red, linewidth=2, label='UE Path')

# Dynamic placeholders
h_sat, = ax1.plot([np.nan], [np.nan], [np.nan], 'o', markersize=10, markerfacecolor=gold, markeredgecolor='w', label='LEO Satellite')
h_ue, = ax1.plot([np.nan], [np.nan], [np.nan], '^', markersize=8, markerfacecolor=red, markeredgecolor='w', label='UE')
h_bc, = ax1.plot([np.nan], [np.nan], [np.nan], 'x', markersize=10, markeredgewidth=2.5, color=light_cyan, label='Beam Center')

h_los, = ax1.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], '--', color=(0.8, 0.8, 0.8), linewidth=1.2, label='UE-Sat Line-of-Sight')
h_boresight, = ax1.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], ':', color=cyan, linewidth=1.8, label='Beam Boresight')

# Footprint collection
h_footprint = Poly3DCollection([], facecolors=cyan, edgecolors=cyan, alpha=0.3, linewidths=2, label='Beam Footprint')
ax1.add_collection3d(h_footprint)

# Axis styling
ax1.set_xlabel('ECEF X (meters)', color='black')
ax1.set_ylabel('ECEF Y (meters)', color='black')
ax1.set_zlabel('ECEF Z (meters)', color='black')
ax1.tick_params(colors='black')
ax1.grid(True)

# Apply view constraints
if zoomView:
    zoom_width = 1.2e6
    ax1.set_xlim(r_ue_ECEF_0[0] - zoom_width, r_ue_ECEF_0[0] + zoom_width)
    ax1.set_ylim(r_ue_ECEF_0[1] - zoom_width, r_ue_ECEF_0[1] + zoom_width)
    ax1.set_zlim(r_ue_ECEF_0[2] - zoom_width, r_ue_ECEF_0[2] + zoom_width)
else:
    ax1.set_xlim(-1.3 * r_orbit, 1.3 * r_orbit)
    ax1.set_ylim(-1.3 * r_orbit, 1.3 * r_orbit)
    ax1.set_zlim(-1.3 * r_orbit, 1.3 * r_orbit)

ax1.legend(facecolor='white', edgecolor=grid_color)

# Setup Figure 2: Doppler & Elevation
fig2 = plt.figure(num='NTN Doppler & Elevation Dynamics', figsize=(8.0, 8.0), facecolor='white')
ax2_1 = fig2.add_subplot(211)
ax2_1.set_facecolor('white')
ax2_1.grid(True, color=grid_color, linestyle='--', alpha=0.5)

# Plot curves
ax2_1.plot(time_grid, doppler_all / 1000.0, color=red, linewidth=2.0, label='UE Doppler (Uncompensated)')
ax2_1.plot(time_grid, doppler_beam_all / 1000.0, color=light_cyan, linewidth=1.5, label='Beam-Center Doppler')
ax2_1.plot(time_grid, doppler_residual_all / 1000.0, ':', color=orange, linewidth=2.0, label='Residual UE Doppler (Compensated)')
ax2_1.set_xlabel('Time Relative to Closest Approach (seconds)')
ax2_1.set_ylabel('Doppler Shift (kHz)')
ax2_1.set_title('Doppler Shift Dynamics (S-Curve & Pre-compensation)')
ax2_1.legend(facecolor='white', edgecolor=grid_color, loc='lower left')
h_doppler_cursor = ax2_1.axvline(time_grid[0], color='k', linewidth=1.5, linestyle='--')

ax2_2 = fig2.add_subplot(212)
ax2_2.set_facecolor('white')
ax2_2.grid(True, color=grid_color, linestyle='--', alpha=0.5)

ax2_2.plot(time_grid, elev_all, color=cyan, linewidth=2.0)
ax2_2.set_xlabel('Time Relative to Closest Approach (seconds)')
ax2_2.set_ylabel('Elevation Angle (degrees)')
ax2_2.set_title('Satellite Elevation Angle from UE perspective')
ax2_2.set_ylim(0, 95)
h_elev_cursor = ax2_2.axvline(time_grid[0], color='k', linewidth=1.5, linestyle='--')

# =========================================================================
# 5. SIMULATION ANIMATION LOOP
# =========================================================================

gif_frames = 15
gif_indices = np.round(np.linspace(0, N_steps - 1, gif_frames)).astype(int)
script_dir = os.path.dirname(os.path.abspath(__file__))

pdf_path_3d = os.path.join(script_dir, 'ntn_geometry_3d_snapshot.pdf')
pdf_path_plots = os.path.join(script_dir, 'ntn_doppler_elevation_snapshot.pdf')
gif_path_3d = os.path.join(script_dir, 'ntn_geometry_3d_simulation.gif')
gif_path_plots = os.path.join(script_dir, 'ntn_doppler_elevation_simulation.gif')

frames_3d = []
frames_plots = []

# Quiver placeholders
q_sat = None
q_ue = None
q_bc = None

# Optimization: Only redraw and pause every N steps to speed up the animation
# (e.g. every 5 seconds). S-curves will still be plotted at 1-second resolution.
anim_skip = 5 

plt.ion()
plt.show()

print("Running optimized simulation...")

for k in range(N_steps):
    t = time_grid[k]
    
    # Render and capture snapshot at exactly t=0 (regardless of skip)
    is_closest_approach = (t == 0.0)
    # Render if it matches the skip interval or is the closest approach or is a GIF frame
    should_render = (k % anim_skip == 0) or is_closest_approach or (k in gif_indices)
    
    if not should_render:
        continue
        
    r_sat = sat_ECEF_all[:, k]
    v_sat = sat_vel_ECEF_all[:, k]
    r_ue = ue_ECEF_all[:, k]
    v_ue = ue_vel_ECEF
    r_bc = bc_ECEF_all[:, k]
    v_bc = bc_vel_ECEF_all[:, k]
    
    # Update marker points
    h_sat.set_data([r_sat[0]], [r_sat[1]])
    h_sat.set_3d_properties([r_sat[2]])
    
    h_ue.set_data([r_ue[0]], [r_ue[1]])
    h_ue.set_3d_properties([r_ue[2]])
    
    h_bc.set_data([r_bc[0]], [r_bc[1]])
    h_bc.set_3d_properties([r_bc[2]])
    
    # Update line vectors
    h_los.set_data([r_ue[0], r_sat[0]], [r_ue[1], r_sat[1]])
    h_los.set_3d_properties([r_ue[2], r_sat[2]])
    
    h_boresight.set_data([r_sat[0], r_bc[0]], [r_sat[1], r_bc[1]])
    h_boresight.set_3d_properties([r_sat[2], r_bc[2]])
    
    # --- Physical Beam Footprint Ray-Sphere Intersection ---
    v_boresight = r_bc - r_sat
    u_b = v_boresight / np.linalg.norm(v_boresight)
    
    if abs(u_b[2]) < 0.9:
        ref_vec = np.array([0.0, 0.0, 1.0])
    else:
        ref_vec = np.array([1.0, 0.0, 0.0])
        
    u_x = np.cross(u_b, ref_vec)
    u_x = u_x / np.linalg.norm(u_x)
    u_y = np.cross(u_b, u_x)
    u_y = u_y / np.linalg.norm(u_y)
    
    phi_az = np.linspace(0, 2*np.pi, 72)
    N_pts = len(phi_az)
    v_unit = (np.cos(theta_b) * u_b[:, np.newaxis] + 
              np.sin(theta_b) * (u_x[:, np.newaxis] * np.cos(phi_az) + u_y[:, np.newaxis] * np.sin(phi_az)))
              
    B_pts = np.sum(r_sat[:, np.newaxis] * v_unit, axis=0)
    C_pts = np.sum(r_sat**2) - a**2
    disc = B_pts**2 - C_pts
    
    r_footprint_pts = np.nan * np.ones((3, N_pts))
    valid = (disc >= 0) & (B_pts < 0)
    d_intersect = -B_pts[valid] - np.sqrt(disc[valid])
    r_footprint_pts[:, valid] = r_sat[:, np.newaxis] + v_unit[:, valid] * d_intersect
    
    # Elevate slightly to avoid z-fighting with Earth mesh
    r_footprint_plot = r_footprint_pts * 1.0006
    
    # Update footprint patch
    h_footprint.set_verts([r_footprint_plot.T])
    
    # --- Check if UE is inside the physical footprint cone ---
    v_sat2ue = r_ue - r_sat
    u_sat2ue = v_sat2ue / np.linalg.norm(v_sat2ue)
    cos_angle = np.dot(u_sat2ue, u_b)
    is_inside = cos_angle >= np.cos(theta_b)
    
    if is_inside:
        inside_str = 'YES'
        h_footprint.set_facecolor(cyan + (0.3,))
        h_footprint.set_edgecolor(cyan)
    else:
        inside_str = 'NO'
        h_footprint.set_facecolor((0.4, 0.4, 0.4, 0.3))
        h_footprint.set_edgecolor((0.5, 0.5, 0.5))
        
    # --- Update Velocity Vectors (Remove and redraw is most robust in 3D matplotlib) ---
    v_sat_scaled = v_sat * vel_scale_sat
    v_ue_scaled = v_ue * vel_scale_ue
    v_bc_scaled = v_bc * vel_scale_bc
    
    if q_sat is not None: q_sat.remove()
    if q_ue is not None: q_ue.remove()
    if q_bc is not None: q_bc.remove()
    
    q_sat = ax1.quiver(r_sat[0], r_sat[1], r_sat[2], v_sat_scaled[0], v_sat_scaled[1], v_sat_scaled[2], 
                       color=emerald, linewidth=1.5, arrow_length_ratio=0.25, label='Sat Velocity')
    q_ue = ax1.quiver(r_ue[0], r_ue[1], r_ue[2], v_ue_scaled[0], v_ue_scaled[1], v_ue_scaled[2], 
                      color=orange, linewidth=2.0, arrow_length_ratio=0.3, label='UE Velocity')
    q_bc = ax1.quiver(r_bc[0], r_bc[1], r_bc[2], v_bc_scaled[0], v_bc_scaled[1], v_bc_scaled[2], 
                      color=light_cyan, linewidth=1.2, arrow_length_ratio=0.25, label='Beam Center Velocity')

    # Update Title
    title_str = (f"LEO Satellite Pass - Time: {t:+.1f} s (BMode: {beamMode})\n"
                 f"Elevation: {elev_all[k]:.1f}° | Range: {slant_range_all[k]/1000:.1f} km | "
                 f"Doppler: {doppler_all[k]/1000:+.2f} kHz\n"
                 f"UE Inside Beam: {inside_str}")
    ax1.set_title(title_str, color='black', fontsize=11)
    
    # Update Subplot sweep cursors
    h_doppler_cursor.set_xdata([t, t])
    h_elev_cursor.set_xdata([t, t])
    
    # --- Capture snapshots at t = 0 (closest approach) ---
    if is_closest_approach:
        fig1.savefig(pdf_path_3d, format='pdf', bbox_inches='tight')
        fig2.savefig(pdf_path_plots, format='pdf', bbox_inches='tight')
        print(f"Saved cropped PDF snapshots to:\n  - {pdf_path_3d}\n  - {pdf_path_plots}")
        
    # Draw graphics
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    
    # --- Capture GIF frames ---
    if k in gif_indices:
        # Capture 3D plot
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
        buf1.seek(0)
        img1 = Image.open(buf1)
        img1.load()
        frames_3d.append(img1)
        buf1.close()
        
        # Capture Doppler/Elevation plots
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        img2 = Image.open(buf2)
        img2.load()
        frames_plots.append(img2)
        buf2.close()

    plt.pause(0.001)

plt.ioff()

# Save captured GIF frames
if frames_3d:
    frames_3d[0].save(gif_path_3d, save_all=True, append_images=frames_3d[1:], duration=400, loop=0)
if frames_plots:
    frames_plots[0].save(gif_path_plots, save_all=True, append_images=frames_plots[1:], duration=400, loop=0)
    
print(f"Saved animation GIFs to:\n  - {gif_path_3d}\n  - {gif_path_plots}")
print("Simulation complete!")
