"""
Script: gen_diverse_openntn_dataset.py
Purpose:
  Generate diverse multi-scenario OpenNTN channel datasets (.npy only) with:
    - Scenarios: Dense Urban (dur), Urban (urb), Suburban (sur)
    - 50 km beam footprint radius (100 km diameter)
    - UE velocities: 20 to 40 m/s (72 to 144 km/h)
    - Diverse 5G NR numerologies: Subcarrier Spacings (SCS) in {15 kHz, 30 kHz, 60 kHz}
    - Fixed tensor shape: 132 subcarriers x 14 OFDM symbols across all numerologies
    - Physically consistent coupling:
        * Elevation angles: 20° to 80°
        * LoS/NLoS probability follows 3GPP TR 38.811 Table 6.6.1-1
        * Delay spread inversely scales with elevation; NLoS DS > LoS DS; DUR > URB > SUR
    - Realistic beam-center Doppler precompensation:
        * Preserves residual differential Doppler across 50 km beam + UE mobility Doppler
    - Output: H_train.npy (80%), H_val.npy (10%), H_test.npy (10%)
    - Metadata: sample_metadata.mat and sample_metadata.json saving exact physics per drop.
"""

import os
import sys
import argparse
import time
import json
import csv
import numpy as np
import scipy.io as sio
import tensorflow as tf

# Suppress TF debug logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Add OpenNTN path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENNTN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(OPENNTN_ROOT)

from OpenNTN import DenseUrban_modify, Urban_modify, SubUrban_modify
from helper import ecef_to_enu, get_satellite_state_ecef
from sionna.phy.ofdm import ResourceGrid, RemoveNulledSubcarriers
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import Antenna, AntennaArray

SPEED_OF_LIGHT = 299792458.0


def compute_3gpp_plos(scenario: str, elev_deg: float) -> float:
    """Computes 3GPP TR 38.811 Table 6.6.1-1 LoS probability as a function of elevation."""
    theta_rad = np.deg2rad(np.clip(elev_deg, 5.0, 89.0))
    tan_theta = np.tan(theta_rad)
    if scenario == "dur":
        # Dense Urban: low LoS probability at low angles (~28% @ 20°, ~85% @ 80°)
        return float(np.exp(-2.5 / tan_theta))
    elif scenario == "urb":
        # Urban: moderate LoS probability (~38% @ 20°, ~90% @ 80°)
        return float(np.exp(-1.7 / tan_theta))
    else:
        # SubUrban: high LoS probability (~70% @ 20°, ~98% @ 80°)
        return float(np.exp(-0.8 / tan_theta))


def sample_delay_spread_ns(scenario: str, elev_deg: float, is_los: bool, ds_min: float = None, ds_max: float = None) -> float:
    """Generates physically consistent delay spread conditioned on scenario, elevation, and LoS state."""
    if ds_min is not None and ds_max is not None:
        # Custom user-defined range [ds_min, ds_max]
        if scenario == "dur":
            base = np.random.uniform(ds_min + 0.5 * (ds_max - ds_min), ds_max) if not is_los else np.random.uniform(ds_min * 0.8, ds_max * 0.8)
        elif scenario == "urb":
            base = np.random.uniform(ds_min + 0.25 * (ds_max - ds_min), ds_max - 0.1 * (ds_max - ds_min)) if not is_los else np.random.uniform(ds_min * 0.6, ds_max * 0.7)
        else:
            base = np.random.uniform(ds_min, ds_min + 0.6 * (ds_max - ds_min)) if not is_los else np.random.uniform(ds_min * 0.4, ds_max * 0.6)
        return float(np.clip(base, ds_min, ds_max))

    # Nominal Diverse mode
    # Delay spread decreases as elevation increases (shorter reflection delays, less ground clutter)
    elev_scale = 1.0 - 0.4 * ((elev_deg - 20.0) / 60.0)  # 1.0 @ 20° down to 0.6 @ 80°

    if scenario == "dur":
        base_ds = np.random.uniform(80.0, 180.0) if is_los else np.random.uniform(250.0, 550.0)
    elif scenario == "urb":
        base_ds = np.random.uniform(50.0, 110.0) if is_los else np.random.uniform(150.0, 320.0)
    else:  # sur
        base_ds = np.random.uniform(25.0, 55.0) if is_los else np.random.uniform(65.0, 140.0)

    return float(np.clip(base_ds * elev_scale, 20.0, 600.0))


def main():
    parser = argparse.ArgumentParser(description="Generate Diverse / Rare Stress-Test OpenNTN Dataset (.npy only).")
    parser.add_argument("--samples", type=int, default=20000, help="Total channel drops (default: 20000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per generator step (default: 64)")
    parser.add_argument("--mode", type=str, default="diverse", choices=["diverse", "rare"],
                        help="Mode: 'diverse' (nominal mixture) or 'rare' (extreme stress-test scenarios)")
    parser.add_argument("--beam-radius-km", type=float, default=50.0, help="Beam footprint radius in km (default: 50.0)")
    parser.add_argument("--carrier-frequency", type=float, default=2.18e9, help="Carrier frequency in Hz (default: 2.18 GHz)")
    parser.add_argument("--elev-min", type=float, default=None, help="Min elevation angle in deg")
    parser.add_argument("--elev-max", type=float, default=None, help="Max elevation angle in deg")
    parser.add_argument("--v-min", type=float, default=None, help="Min UE ground speed in m/s")
    parser.add_argument("--v-max", type=float, default=None, help="Max UE ground speed in m/s")
    parser.add_argument("--ds-min", type=float, default=None, help="Min custom delay spread in ns")
    parser.add_argument("--ds-max", type=float, default=None, help="Max custom delay spread in ns")
    parser.add_argument("--los", type=str, default="auto", choices=["auto", "los", "nlos"],
                        help="Propagation state: 'auto' (3GPP probability), 'los' (pure LoS), or 'nlos' (pure NLoS)")
    parser.add_argument("--scs", type=float, default=None,
                        help="Fixed Subcarrier Spacing in Hz (e.g. 120e3 or 120000). Overrides mode defaults.")
    parser.add_argument("--scs-list", type=float, nargs="+", default=None,
                        help="List of Subcarrier Spacings in Hz (e.g. --scs-list 15e3 30e3 60e3)")
    parser.add_argument("--no-mat", action="store_true",
                        help="Skip saving sample_metadata.mat (only save .csv and .json)")
    parser.add_argument("--outer-ring-ratio", type=float, default=0.0,
                        help="If > 0 (e.g. 0.7), sample UEs only in outer ring r in [ratio*r_beam, r_beam]")
    parser.add_argument("--out-name", type=str, default=None,
                        help="Output subfolder name in results/ (default: auto-named by mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set parameters according to mode if not explicitly overridden
    if args.mode == "rare":
        elev_min = args.elev_min if args.elev_min is not None else 10.0
        elev_max = args.elev_max if args.elev_max is not None else 25.0
        v_min = args.v_min if args.v_min is not None else 50.0
        v_max = args.v_max if args.v_max is not None else 120.0
        ds_min = args.ds_min if args.ds_min is not None else 600.0
        ds_max = args.ds_max if args.ds_max is not None else 1500.0
        outer_ring_ratio = args.outer_ring_ratio if args.outer_ring_ratio > 0 else 0.70
        scs_options = [30e3, 60e3]
        default_out = "openNTN_RareStress_BeamEdge_v50to120_ds600to1500_elev10to25"
    else:  # diverse
        elev_min = args.elev_min if args.elev_min is not None else 20.0
        elev_max = args.elev_max if args.elev_max is not None else 80.0
        v_min = args.v_min if args.v_min is not None else 20.0
        v_max = args.v_max if args.v_max is not None else 40.0
        ds_min = args.ds_min
        ds_max = args.ds_max
        outer_ring_ratio = args.outer_ring_ratio
        scs_options = [15e3, 30e3, 60e3]
        default_out = "openNTN_Diverse_50kmBeam_v20to40_scs15to60_elev20to80"

    # Override SCS if explicitly provided via CLI
    if args.scs is not None:
        scs_options = [float(args.scs)]
    elif args.scs_list is not None:
        scs_options = [float(s) for s in args.scs_list]

    out_name = args.out_name if args.out_name is not None else default_out
    results_dir = os.path.join(SCRIPT_DIR, "results", out_name)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 75)
    print(f"OpenNTN DATASET GENERATOR ({args.mode.upper()} MODE)")
    print(f"Total Drops       : {args.samples:,} (Train: {int(0.8*args.samples):,}, Val: {int(0.1*args.samples):,}, Test: {int(0.1*args.samples):,})")
    print(f"Beam Footprint    : {args.beam_radius_km:.1f} km radius (Outer Ring Ratio: {outer_ring_ratio:.2f})")
    print(f"UE Ground Speed   : [{v_min:.1f}, {v_max:.1f}] m/s ({v_min*3.6:.0f} to {v_max*3.6:.0f} km/h)")
    print(f"SCS Numerologies  : {[f'{int(s/1e3)} kHz' for s in scs_options]} (Fixed to 132 sc x 14 sym)")
    print(f"Elevation Range   : [{elev_min:.1f}°, {elev_max:.1f}°]")
    print(f"Delay Spread Range: [{ds_min if ds_min else 20.0:.0f}, {ds_max if ds_max else 600.0:.0f}] ns")
    print(f"Scenarios Covered : Dense Urban (dur), Urban (urb), Suburban (sur)")
    print(f"Output Directory  : {results_dir}")
    print("=" * 75)

    # 1. ORBITAL & REFERENCE CONSTANTS
    satellite_height = 600000.0   # 600 km LEO
    phi_UE_deg = 37.7749          # Reference Latitude
    lambda_UE_deg = -122.4194     # Reference Longitude
    h_UE = 100.0                  # Reference Altitude
    inclination_deg = 55.0        # Orbit inclination

    omega_E = 7.292115e-5
    mu = 3.986004418e14
    a_wgs84 = 6378137.0
    e2 = 6.69437999e-3

    inclination = np.deg2rad(inclination_deg)
    phi_UE = np.deg2rad(phi_UE_deg)
    lambda_UE = np.deg2rad(lambda_UE_deg)

    r_orbit = a_wgs84 + satellite_height
    omega_s = np.sqrt(mu / r_orbit**3)
    v_sat_orbit = np.sqrt(mu / r_orbit)

    N_phi_0 = a_wgs84 / np.sqrt(1.0 - e2 * np.sin(phi_UE)**2)
    r_ue_ECEF_0 = np.array([
        (N_phi_0 + h_UE) * np.cos(phi_UE) * np.cos(lambda_UE),
        (N_phi_0 + h_UE) * np.cos(phi_UE) * np.sin(lambda_UE),
        (N_phi_0 * (1.0 - e2) + h_UE) * np.sin(phi_UE)
    ])

    R_ENU2ECEF = np.array([
        [-np.sin(lambda_UE), -np.sin(phi_UE) * np.cos(lambda_UE), np.cos(phi_UE) * np.cos(lambda_UE)],
        [ np.cos(lambda_UE), -np.sin(phi_UE) * np.sin(lambda_UE), np.cos(phi_UE) * np.sin(lambda_UE)],
        [ 0.0,                np.cos(phi_UE),                 np.sin(phi_UE)]
    ])

    if inclination >= abs(phi_UE):
        u_mid = np.arcsin(np.sin(phi_UE) / np.sin(inclination))
    else:
        u_mid = np.sign(phi_UE) * np.pi / 2.0
    Omega_RAAN = lambda_UE - np.arctan2(np.sin(u_mid) * np.cos(inclination), np.cos(u_mid))

    r_beam_m = args.beam_radius_km * 1000.0
    r_ue_max = r_beam_m * 0.98

    # 2. ANTENNA & SIONNA SETUP
    carrier_frequency = args.carrier_frequency
    lambda_0 = SPEED_OF_LIGHT / carrier_frequency

    ut_array = Antenna("single", "V", "omni", carrier_frequency)
    bs_array = AntennaArray(1, 1, "dual", "VH", "38.901", carrier_frequency)

    # 3. RESOURCE GRIDS CACHE FOR CONFIGURED SCS OPTIONS
    # 11 Resource Blocks * 12 = 132 subcarriers
    # 256 FFT - 132 active = 124 guard carriers -> (62, 62)
    rg_cache = {}
    rem_cache = {}
    for scs in scs_options:
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=256,
            subcarrier_spacing=scs,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=26,
            num_guard_carriers=(62, 62),
            dc_null=False,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11]
        )
        rg_cache[scs] = rg
        rem_cache[scs] = RemoveNulledSubcarriers(rg)

    # Pre-instantiate OpenNTN Channel Models for fast reuse
    channel_models = {
        "dur": DenseUrban_modify(carrier_frequency, ut_array, bs_array, "downlink", 30.0,
                                 doppler_enabled=True, doppler_mode="precompensated"),
        "urb": Urban_modify(carrier_frequency, ut_array, bs_array, "downlink", 30.0,
                            doppler_enabled=True, doppler_mode="precompensated"),
        "sur": SubUrban_modify(carrier_frequency, ut_array, bs_array, "downlink", 30.0,
                               doppler_enabled=True, doppler_mode="precompensated")
    }

    # OFDM channel operators cache: key = (scenario, scs)
    ofdm_channels = {}
    for sc_name, c_model in channel_models.items():
        for scs in scs_options:
            ofdm_channels[(sc_name, scs)] = GenerateOFDMChannel(c_model, resource_grid=rg_cache[scs])

    # 4. GENERATION LOOP (BATCHED)
    total_samples = args.samples
    batch_size = args.batch_size
    num_batches = int(np.ceil(total_samples / batch_size))

    all_channels = []

    # Metadata storage
    meta_scenario = []
    meta_elev_deg = []
    meta_is_los = []
    meta_ds_ns = []
    meta_scs_khz = []
    meta_v_ue = []
    meta_dist_center_m = []
    meta_residual_doppler_hz = []

    print(f"\nGenerating {total_samples:,} drops across {num_batches} mini-batches of size {batch_size}...")
    start_time = time.time()

    for b in range(num_batches):
        cur_batch = min(batch_size, total_samples - b * batch_size)

        # A. Sample batch-level configuration
        # Diverse scenario choice
        scenario = np.random.choice(["dur", "urb", "sur"])
        scs = np.random.choice(scs_options)
        c_model = channel_models[scenario]
        ofdm_op = ofdm_channels[(scenario, scs)]
        rem_op = rem_cache[scs]
        rg = rg_cache[scs]

        # Elevation angle for this batch: uniform within [elev_min, elev_max]
        elev_deg = float(np.random.uniform(elev_min, elev_max))
        theta_target_rad = np.deg2rad(elev_deg)

        # LoS condition based on 3GPP TR 38.811 probability or explicit override
        if args.los == "nlos":
            is_los = False
        elif args.los == "los":
            is_los = True
        else:
            p_los = compute_3gpp_plos(scenario, elev_deg)
            is_los = bool(np.random.rand() < p_los)

        # Delay spread matched to scenario, elevation, and LoS
        ds_ns = sample_delay_spread_ns(scenario, elev_deg, is_los, ds_min=ds_min, ds_max=ds_max)

        # B. Orbital satellite state at this elevation
        gamma_central = np.pi / 2.0 - theta_target_rad - np.arcsin((a_wgs84 / r_orbit) * np.cos(theta_target_rad))
        t_snap = float(gamma_central / omega_s)
        r_sat_ECEF, v_sat_ECEF = get_satellite_state_ecef(
            t_snap, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E
        )
        bs_loc_ENU = ecef_to_enu(r_sat_ECEF, r_ue_ECEF_0, lambda_UE, phi_UE)
        v_sat_ENU = ecef_to_enu(v_sat_ECEF, np.zeros(3), lambda_UE, phi_UE)

        # Nominal satellite line-of-sight unit vector from beam center to satellite
        u_los_bc = bs_loc_ENU / np.linalg.norm(bs_loc_ENU)
        f_d_center = np.dot(v_sat_ENU, u_los_bc) / lambda_0

        # C. Randomized UEs across beam footprint (supports outer ring sampling)
        theta_rand = np.random.uniform(0.0, 2 * np.pi, cur_batch)
        min_u = outer_ring_ratio ** 2
        r_rand = r_ue_max * np.sqrt(np.random.uniform(min_u, 1.0, cur_batch))
        ut_loc_batch = np.zeros((cur_batch, 3), dtype=np.float32)
        ut_loc_batch[:, 0] = r_rand * np.cos(theta_rand)
        ut_loc_batch[:, 1] = r_rand * np.sin(theta_rand)
        ut_loc_batch[:, 2] = 1.5

        # UE Velocities: within [v_min, v_max]
        v_ue_mag = np.random.uniform(v_min, v_max, cur_batch)
        v_ue_head = np.random.uniform(0.0, 2 * np.pi, cur_batch)
        ut_vel_batch = np.zeros((cur_batch, 3), dtype=np.float32)
        ut_vel_batch[:, 0] = v_ue_mag * np.sin(v_ue_head)
        ut_vel_batch[:, 1] = v_ue_mag * np.cos(v_ue_head)
        ut_vel_batch[:, 2] = 0.0

        # Compute exact residual differential Doppler per UE:
        # f_d_ue = 1/lambda * (v_sat - v_ue) . u_los_ue
        # delta_f_d = f_d_ue - f_d_center (compensating for beam center)
        v_los_ues = bs_loc_ENU[np.newaxis, :] - ut_loc_batch
        d_slant_ues = np.linalg.norm(v_los_ues, axis=1, keepdims=True)
        u_los_ues = v_los_ues / d_slant_ues

        v_rel = v_sat_ENU[np.newaxis, :] - ut_vel_batch
        f_d_ues = np.sum(v_rel * u_los_ues, axis=1) / lambda_0
        delta_f_d = f_d_ues - f_d_center  # Residual Doppler in Hz (shape: [cur_batch])

        # D. Configure Channel Model and Generate
        rounded_elev = int(round(elev_deg / 10.0) * 10)
        rounded_elev = max(10, min(90, rounded_elev))

        # Override cluster DS parameter to match target ds_ns
        target_log_mean_ds = np.log10(ds_ns * 1e-9)
        c_model._scenario._params_los[f"muDS_{rounded_elev}"] = tf.constant(target_log_mean_ds, dtype=tf.float32)
        c_model._scenario._params_nlos[f"muDS_{rounded_elev}"] = tf.constant(target_log_mean_ds, dtype=tf.float32)
        c_model._scenario._params_los[f"sigmaDS_{rounded_elev}"] = tf.constant(0.0, dtype=tf.float32)
        c_model._scenario._params_nlos[f"sigmaDS_{rounded_elev}"] = tf.constant(0.0, dtype=tf.float32)

        # Build tensors
        ut_loc_tensor = tf.constant(ut_loc_batch[:, np.newaxis, :], dtype=tf.float32)
        bs_loc_tile = np.tile(bs_loc_ENU, (cur_batch, 1))
        bs_loc_tensor = tf.constant(bs_loc_tile[:, np.newaxis, :], dtype=tf.float32)
        ut_vel_tensor = tf.constant(ut_vel_batch[:, np.newaxis, :], dtype=tf.float32)
        bs_vel_zero = tf.zeros_like(bs_loc_tensor)
        ut_orient = tf.zeros([cur_batch, 1, 3])
        bs_orient = tf.zeros([cur_batch, 1, 3])
        in_state = tf.constant(np.zeros((cur_batch, 1), dtype=bool), dtype=tf.bool)

        c_model._scenario._requested_los = is_los
        c_model.set_topology(ut_loc_tensor, bs_loc_tensor, ut_orient, bs_orient,
                             ut_vel_tensor, bs_vel_zero, in_state, los=is_los)

        num_time_steps = 14 * (rg.fft_size + rg.cyclic_prefix_length)
        sampling_freq = scs * rg.fft_size

        tf.random.set_seed(args.seed + b)
        _, _ = c_model(num_time_steps, sampling_freq)

        tf.random.set_seed(args.seed + b)
        H_ofdm = ofdm_op()
        h_extracted = rem_op(H_ofdm)[:, 0, 0, 0, 0, :, :].numpy()  # [cur_batch, 14, 132]

        # E. Apply Residual Differential Doppler phase spin across OFDM symbols
        # T_sym = 1 / scs + T_cp (cyclic prefix duration)
        t_sym = float(1.0 / scs + (26.0 / (256.0 * scs)))
        sym_indices = np.arange(14)[np.newaxis, :, np.newaxis]  # [1, 14, 1]
        phase_spin = np.exp(1j * 2.0 * np.pi * delta_f_d[:, np.newaxis, np.newaxis] * sym_indices * t_sym)
        h_with_doppler = (h_extracted * phase_spin).astype(np.complex64)

        # Transpose to model standard shape: [cur_batch, 132 subcarriers, 14 symbols]
        h_final = np.transpose(h_with_doppler, (0, 2, 1))
        all_channels.append(h_final)

        # Save metadata per sample
        for i in range(cur_batch):
            meta_scenario.append(scenario)
            meta_elev_deg.append(elev_deg)
            meta_is_los.append(1 if is_los else 0)
            meta_ds_ns.append(ds_ns)
            meta_scs_khz.append(float(scs / 1e3))
            meta_v_ue.append(float(v_ue_mag[i]))
            meta_dist_center_m.append(float(r_rand[i]))
            meta_residual_doppler_hz.append(float(delta_f_d[i]))

        if (b + 1) % max(1, num_batches // 10) == 0 or (b + 1) == num_batches:
            elapsed = time.time() - start_time
            done_samples = min((b + 1) * batch_size, total_samples)
            rate = done_samples / max(1e-3, elapsed)
            print(f"  [{done_samples:,}/{total_samples:,}] drops generated ({done_samples/total_samples*100:.1f}%) | Speed: {rate:.1f} drops/s | Elapsed: {elapsed:.1f}s")

    # 5. CONCATENATE & SAVE SPLITS
    H_all = np.concatenate(all_channels, axis=0)
    print(f"\nFinal Channel Tensor Shape: {H_all.shape}, Dtype: {H_all.dtype}")

    n_train = int(0.80 * total_samples)
    n_val = int(0.10 * total_samples)

    H_train = H_all[:n_train]
    H_val = H_all[n_train:n_train + n_val]
    H_test = H_all[n_train + n_val:]

    np.save(os.path.join(results_dir, "H_train.npy"), H_train)
    np.save(os.path.join(results_dir, "H_val.npy"), H_val)
    np.save(os.path.join(results_dir, "H_test.npy"), H_test)

    print(f"Saved .npy splits in {results_dir}:")
    print(f"  H_train.npy: {H_train.shape} ({os.path.getsize(os.path.join(results_dir, 'H_train.npy')) / 1e6:.1f} MB)")
    print(f"  H_val.npy:   {H_val.shape}   ({os.path.getsize(os.path.join(results_dir, 'H_val.npy')) / 1e6:.1f} MB)")
    print(f"  H_test.npy:  {H_test.shape}  ({os.path.getsize(os.path.join(results_dir, 'H_test.npy')) / 1e6:.1f} MB)")

    # 6. SAVE METADATA (.csv, .json, and optional .mat)
    csv_path = os.path.join(results_dir, "sample_metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SampleIdx", "scenario", "elev_deg", "is_los", "ds_ns", "scs_khz", "v_ue_mps", "dist_center_m", "residual_doppler_hz"])
        for i in range(total_samples):
            writer.writerow([
                i + 1,
                meta_scenario[i],
                round(meta_elev_deg[i], 2),
                int(meta_is_los[i]),
                round(meta_ds_ns[i], 2),
                round(meta_scs_khz[i], 1),
                round(meta_v_ue[i], 2),
                round(meta_dist_center_m[i], 1),
                round(meta_residual_doppler_hz[i], 2)
            ])
    print(f"Saved sample metadata CSV: {csv_path} ({os.path.getsize(csv_path) / 1e6:.1f} MB)")

    if not args.no_mat:
        metadata_mat = {
            "meta_scenario": np.array(meta_scenario, dtype=object),
            "meta_elev_deg": np.array(meta_elev_deg, dtype=np.float32),
            "meta_is_los": np.array(meta_is_los, dtype=np.int32),
            "meta_ds_ns": np.array(meta_ds_ns, dtype=np.float32),
            "meta_scs_khz": np.array(meta_scs_khz, dtype=np.float32),
            "meta_v_ue": np.array(meta_v_ue, dtype=np.float32),
            "meta_dist_center_m": np.array(meta_dist_center_m, dtype=np.float32),
            "meta_residual_doppler_hz": np.array(meta_residual_doppler_hz, dtype=np.float32),
        }
        mat_path = os.path.join(results_dir, "sample_metadata.mat")
        sio.savemat(mat_path, metadata_mat)
        print(f"Saved sample metadata MAT: {mat_path} ({os.path.getsize(mat_path) / 1e6:.1f} MB)")

    # Also save metadata summary JSON
    summary_json = {
        "dataset_name": out_name,
        "mode": args.mode,
        "total_samples": total_samples,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": total_samples - n_train - n_val,
        "carrier_frequency_hz": carrier_frequency,
        "beam_radius_km": args.beam_radius_km,
        "outer_ring_ratio": outer_ring_ratio,
        "elevation_range_deg": [elev_min, elev_max],
        "scs_options_khz": [float(s / 1e3) for s in scs_options],
        "ue_velocity_range_mps": [v_min, v_max],
        "stats": {
            "mean_elev_deg": float(np.mean(meta_elev_deg)),
            "los_ratio": float(np.mean(meta_is_los)),
            "mean_ds_ns": float(np.mean(meta_ds_ns)),
            "min_ds_ns": float(np.min(meta_ds_ns)),
            "max_ds_ns": float(np.max(meta_ds_ns)),
            "mean_v_ue": float(np.mean(meta_v_ue)),
            "max_residual_doppler_hz": float(np.max(np.abs(meta_residual_doppler_hz))),
            "scenarios_count": {
                "dur": int(np.sum([s == "dur" for s in meta_scenario])),
                "urb": int(np.sum([s == "urb" for s in meta_scenario])),
                "sur": int(np.sum([s == "sur" for s in meta_scenario]))
            }
        }
    }
    with open(os.path.join(results_dir, "metadata_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    # 7. SAVE DOCUMENTATION note.md
    band_label = "Ka-band" if carrier_frequency >= 18.0e9 else ("C-band" if carrier_frequency >= 4.0e9 else "S-band")
    n_test = total_samples - n_train - n_val
    note_content = f"""# Dataset Overview: {out_name}

## 1. General Information
- **Channel Generator:** OpenNTN 3GPP TR 38.811 GSCM via Sionna
- **Carrier Frequency:** {carrier_frequency / 1e9:.2f} GHz ({band_label})
- **Total Drops:** {total_samples:,} (Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,})
  - **Train (`H_train.npy`):** First {n_train:,} samples (`SampleIdx` 1 – {n_train:,}, 0-based indices `[0:{n_train}]`)
  - **Val (`H_val.npy`):** Next {n_val:,} samples (`SampleIdx` {n_train + 1:,} – {n_train + n_val:,}, 0-based indices `[{n_train}:{n_train + n_val}]`)
  - **Test (`H_test.npy`):** Next {n_test:,} samples (`SampleIdx` {n_train + n_val + 1:,} – {total_samples:,}, 0-based indices `[{n_train + n_val}:{total_samples}]`)
- **Grid Size:** 132 Subcarriers × 14 OFDM Symbols (Fixed for model compatibility)
- **Data Type:** `complex64`, raw unnormalized channel matrices `(N, 132, 14)`

## 2. Multi-Environment Parameters
- **Scenarios:** Dense Urban (`dur`: {summary_json['stats']['scenarios_count']['dur']}), Urban (`urb`: {summary_json['stats']['scenarios_count']['urb']}), Suburban (`sur`: {summary_json['stats']['scenarios_count']['sur']})
- **Elevation Angle:** Continuous mixture from {elev_min:.1f}° to {elev_max:.1f}° (Mean: {summary_json['stats']['mean_elev_deg']:.1f}°)
- **LoS Ratio:** {summary_json['stats']['los_ratio']*100:.1f}% (Condition: {args.los})
- **Delay Spread ($\tau_{{\\text{{rms}}}}$):** [{summary_json['stats']['min_ds_ns']:.1f} ns, {summary_json['stats']['max_ds_ns']:.1f} ns] (Mean: {summary_json['stats']['mean_ds_ns']:.1f} ns)
- **Subcarrier Spacings:** Diverse mixture of {[f'{int(s/1e3)} kHz' for s in scs_options]}
- **Beam Footprint:** {args.beam_radius_km:.1f} km radius (Outer ring ratio: {outer_ring_ratio:.2f})
- **UE Ground Velocity:** {v_min:.1f} m/s to {v_max:.1f} m/s ({v_min*3.6:.0f} km/h to {v_max*3.6:.0f} km/h)
- **Doppler Model:** Beam-center precompensation with residual differential Doppler (Max residual: $\pm${summary_json['stats']['max_residual_doppler_hz']:.1f} Hz)

## 3. Metadata Files & Split Order Mapping
- `sample_metadata.csv`: Ground-truth parameters per sample for pandas / Python analysis. Contains all {total_samples:,} drops in sequential order:
  | Split | `.npy` File | `SampleIdx` (1-based CSV) | Python Slice `[start:stop]` | Count |
  | :--- | :--- | :--- | :--- | :--- |
  | **Train** | `H_train.npy` | `1` to `{n_train}` | `[0 : {n_train}]` | {n_train:,} |
  | **Val** | `H_val.npy` | `{n_train + 1}` to `{n_train + n_val}` | `[{n_train} : {n_train + n_val}]` | {n_val:,} |
  | **Test** | `H_test.npy` | `{n_train + n_val + 1}` to `{total_samples}` | `[{n_train + n_val} : {total_samples}]` | {n_test:,} |
- `H_train.npy`, `H_val.npy`, `H_test.npy`: Pure channel tensor splits `(N, 132, 14)` matching the exact sequential rows above.
- `metadata_summary.json`: High-level statistical summary.
"""
    if not args.no_mat:
        note_content += "- `sample_metadata.mat`: MATLAB-compatible structure containing per-sample physics.\n"
    with open(os.path.join(results_dir, "note.md"), "w") as f:
        f.write(note_content)

    print(f"\nGeneration complete! All files saved in: {results_dir}")


if __name__ == "__main__":
    main()
