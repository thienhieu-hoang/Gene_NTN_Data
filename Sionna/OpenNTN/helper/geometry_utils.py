import numpy as np

# =========================================================================
# LEO NTN Geometry & Coordinate Transformation Utilities
# =========================================================================

def ecef_to_enu(r_ecef, r_ref_ecef, lambda_ref, phi_ref):
    """
    Convert ECEF coordinates to the Local East-North-Up (ENU) tangent plane frame.
    
    Parameters:
    -----------
    r_ecef : numpy.ndarray of shape (3,) or (3, N)
        ECEF coordinates to convert.
    r_ref_ecef : numpy.ndarray of shape (3,)
        ECEF coordinates of the reference origin point.
    lambda_ref : float
        Reference longitude in radians.
    phi_ref : float
        Reference latitude in radians.
        
    Returns:
    --------
    r_enu : numpy.ndarray of shape (3,) or (3, N)
        Converted ENU coordinates in meters.
    """
    R_ENU2ECEF = np.array([
        [-np.sin(lambda_ref), -np.sin(phi_ref) * np.cos(lambda_ref), np.cos(phi_ref) * np.cos(lambda_ref)],
        [ np.cos(lambda_ref), -np.sin(phi_ref) * np.sin(lambda_ref), np.cos(phi_ref) * np.sin(lambda_ref)],
        [ 0.0,                np.cos(phi_ref),                 np.sin(phi_ref)]
    ])
    R_ECEF2ENU = R_ENU2ECEF.T
    
    if len(r_ecef.shape) == 1:
        return R_ECEF2ENU @ (r_ecef - r_ref_ecef)
    else:
        return R_ECEF2ENU @ (r_ecef - r_ref_ecef[:, np.newaxis])


def get_satellite_state_ecef(t, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E):
    """
    Compute the satellite's ECEF 3D position and velocity vectors at a given time.
    
    Parameters:
    -----------
    t : float
        Time relative to closest approach (seconds).
    omega_s : float
        Satellite mean motion (rad/s).
    u_mid : float
        Argument of latitude at closest approach (radians).
    Omega_RAAN : float
        Right Ascension of Ascending Node (radians).
    inclination : float
        Orbital inclination (radians).
    r_orbit : float
        Orbital radius (meters).
    v_sat_orbit : float
        Satellite orbital speed in ECI (m/s).
    omega_E : float
        Earth's rotation rate (rad/s).
        
    Returns:
    --------
    r_sat_ECEF : numpy.ndarray of shape (3,)
        Satellite ECEF position vector.
    v_sat_ECEF : numpy.ndarray of shape (3,)
        Satellite ECEF velocity vector.
    """
    # Earth rotation angle at time t
    theta_G = omega_E * t
    R_z = np.array([
        [ np.cos(theta_G), np.sin(theta_G), 0.0],
        [-np.sin(theta_G), np.cos(theta_G), 0.0],
        [ 0.0,             0.0,             1.0]
    ])
    
    # Satellite position and velocity in ECI
    u_t = omega_s * t + u_mid
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
    
    # Convert ECI to ECEF
    r_sat_ECEF = R_z @ r_sat_ECI
    
    # Account for Coriolis effect in relative velocity
    omega_cross_r = np.array([
        -omega_E * r_sat_ECI[1],
         omega_E * r_sat_ECI[0],
         0.0
    ])
    v_sat_ECEF = R_z @ (v_sat_ECI - omega_cross_r)
    
    return r_sat_ECEF, v_sat_ECEF


def get_local_earth_patch(lambda_ref, phi_ref, a_earth, delta_deg=12.0, num_pts=30):
    """
    Generate the 3D surface mesh (X, Y, Z coordinate matrices) of a local Earth ground patch
    centered at a specific longitude and latitude.
    
    Parameters:
    -----------
    lambda_ref : float
        Center longitude (radians).
    phi_ref : float
        Center latitude (radians).
    a_earth : float
        Earth's radius or semi-major axis (meters).
    delta_deg : float
        Half-span of the patch (degrees).
    num_pts : int
        Grid resolution.
        
    Returns:
    --------
    XS, YS, ZS : numpy.ndarray of shape (num_pts, num_pts)
        ECEF X, Y, Z mesh matrices of the Earth ground surface patch.
    """
    lon_range = np.linspace(lambda_ref - np.deg2rad(delta_deg), lambda_ref + np.deg2rad(delta_deg), num_pts)
    lat_range = np.linspace(phi_ref - np.deg2rad(delta_deg), phi_ref + np.deg2rad(delta_deg), num_pts)
    
    XS = np.zeros((len(lat_range), len(lon_range)))
    YS = np.zeros((len(lat_range), len(lon_range)))
    ZS = np.zeros((len(lat_range), len(lon_range)))
    
    for idx_lat, lat in enumerate(lat_range):
        for idx_lon, lon in enumerate(lon_range):
            XS[idx_lat, idx_lon] = a_earth * np.cos(lat) * np.cos(lon)
            YS[idx_lat, idx_lon] = a_earth * np.sin(lon) * np.cos(lat)
            ZS[idx_lat, idx_lon] = a_earth * np.sin(lat)
            
    return XS, YS, ZS


def save_simulation_readme(filename, scenario, carrier_frequency, direction, elevation_angle, 
                            satellite_height, ue_speed, scs, n_fft, snr_db, n_samples, avg_delay_spread_ns):
    """
    Save channel simulation configuration parameters and results into a markdown documentation file.
    """
    md_content = f"""# Channel & Geometry Generation Settings - {scenario.upper()}

- **Scenario Type**: {scenario.upper()} (dur = Dense Urban, sur = SubUrban, urb = Urban)
- **Carrier Frequency**: {carrier_frequency / 1e9:.2f} GHz
- **Link Direction**: {direction}
- **Satellite Elevation Angle**: {elevation_angle:.2f} degrees
- **Satellite (LEO) Height**: {satellite_height / 1000:.0f} km
- **UE Ground Speed**: {ue_speed} m/s
- **Subcarrier Spacing (SCS)**: {scs / 1e3:.0f} kHz
- **FFT Size**: {n_fft}
- **Active Subcarriers**: 132 (out of {n_fft})
- **SNR (for LS estimation)**: {snr_db} dB
- **Total OFDM Symbols**: 14
- **Pilot Symbols (0-indexed)**: [2, 7, 11]
- **Total Samples Generated**: {n_samples}
- **Average RMS Delay Spread**: {avg_delay_spread_ns:.2f} ns
"""
    with open(filename, "w") as f:
        f.write(md_content)
    print(f"Saved scenario parameters documentation to {filename}")

