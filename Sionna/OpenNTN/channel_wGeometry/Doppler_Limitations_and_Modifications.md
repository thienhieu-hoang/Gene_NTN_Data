# OpenNTN Doppler Modelling -- Limitations and Required Modifications

> **Purpose**: Documents the Doppler decomposition assumptions in the original OpenNTN
> library, the gap between the model and physical reality, and the exact code changes
> needed to implement (1) the true sat-UE direct Doppler and (2) the correct residual
> Doppler after satellite precompensation.

---

## 1. Background -- How Doppler Decomposes in NTN

In a non-terrestrial network (NTN), the total Doppler shift on any propagation path has two
distinct physical origins:

`
f_D,total = f_D(sat to beam-center) + f_D(beam-center to UE)
`

Expressed as projections of velocity onto the path unit vector r_hat:

`
f_D,total = (1/lambda_0) * [ r_hat_dep . v_sat  +  r_hat_arr . v_UE ]
`

| Term | Physical meaning | Dominant magnitude (LEO 600 km, 27 GHz) |
|---|---|---|
| r_hat_dep . v_sat / lambda_0 | Satellite motion along departure direction | ~+-200-400 kHz |
| r_hat_arr . v_UE  / lambda_0 | UE motion along arrival direction | ~+-5 Hz (50 m/s UE) |

Because the two components differ by several orders of magnitude, **the satellite bulk
Doppler is pre-compensated on the satellite/gateway** before the signal reaches the UE.
The UE only has to handle the residual.

**Idealized precompensation removes the LoS beam-center projection:**

`
f_D,residual = f_D,total  -  f_D(sat->bc)
             = r_hat_arr . v_UE / lambda_0
             + (r_hat_dep,cluster - r_hat_LoS) . v_sat / lambda_0   <- per-cluster sat residual
`

The per-cluster sat residual exists because scatter paths leave the satellite at
**slightly different angles** than the direct LoS path. For NTN at 600 km altitude,
this residual is small but non-zero (~10-50 Hz).

---

## 2. What the Original OpenNTN Library Does

### 2.1 Relevant Files

| File | Role |
|---|---|
| OpenNTN/channel_coefficients.py | Topology class + _step_11_doppler_matrix() |
| OpenNTN/system_level_channel.py | Builds Topology, wires scenario to CIR sampler |
| OpenNTN/system_level_scenario.py | Stores doppler_enabled flag, ut_velocities |
| channel_wGeometry/gen_channel_wGeometry.py | Entry-point, geometry, calls channel model |

### 2.2 The Doppler Computation (Step 11)

**File**: OpenNTN/channel_coefficients.py, function _step_11_doppler_matrix() (~line 546)

`python
# Term 1: UE Doppler -- CORRECT: per-cluster, uses scatter arrival angles (AoA, ZoA)
r_hat_ut = _unit_sphere_vector(zoa, aoa)         # shape: [batch, tx, rx, cluster, ray, 3, 1]
exponent  = 2pi/lambda_0 * dot(r_hat_ut, v_UE) * t   # per ray and cluster  (OK)

# Term 2: Satellite Doppler -- only added when bs_height >= 600 km AND doppler_enabled=True
if tf.logical_and(tf.less_equal(600000.0, topology.bs_height), topology.doppler_enabled):
    max_sat_speed = topology.sat_speed             # ONE scalar per batch
    rotation_for_time = 2pi/lambda_0 * max_sat_speed * t  # broadcast to ALL clusters  (PROBLEM)
    exponent = exponent + rotation_for_time
`

### 2.3 How sat_speed is Built (in Topology.__init__)

**File**: OpenNTN/channel_coefficients.py, Topology.__init__() (~line 99)

`python
sat_speed = compute_satellite_speed(bs_height)        # scalar |v_sat| from orbital mechanics
max_sat   = cos(elevation_angle) * sat_speed          # projected onto LoS elevation angle
random_th = tf.random.uniform([batch_size], 0, 2*pi)  # RANDOM orbit direction per batch (PROBLEM)
self.sat_speed = cos(random_th) * max_sat             # random scalar, one value per batch
`

### 2.4 Topology Construction -- Satellite 3D Velocity Never Passed

**File**: OpenNTN/system_level_channel.py (~line 202)

`python
topology = Topology(
    velocities    = self._scenario.ut_velocities,   # UE 3D velocity [batch, num_ut, 3]  (OK)
    # -- satellite 3D velocity is NEVER passed ------  (PROBLEM)
    bs_height     = self._scenario._bs_loc[:,:,2][0],
    elevation_angle = self._scenario.elevation_angle,
    doppler_enabled = self._scenario.doppler_enabled,
    ...
)
`

---

## 3. Limitations of the Original Model

### Limitation 1 -- Satellite Doppler is a Uniform Scalar (Not Per-Cluster)

The library adds the **same Doppler shift** to every scatter cluster. In reality, each
cluster departs the satellite at a different angle (od_cluster, zod_cluster), giving
a different contribution per cluster:

`
Library:  f_D,sat   =  v_sat_projected / lambda_0          (same for ALL clusters)
Reality:  f_D,sat_l =  r_hat_dep_l . v_sat / lambda_0      (different per cluster l)
`

The scatter departure angles (od, zod) per cluster are already present in the 
ays
object at this point in the code but are never used for the satellite Doppler term.

### Limitation 2 -- Satellite Direction is Randomized, Not From Real Geometry

The actual 3D satellite velocity _sat_ENU is computed correctly in
gen_channel_wGeometry.py from ECEF orbital mechanics but is **never wired** into the
Topology or ChannelCoefficientsGenerator. Instead, the library applies a **random
direction cosine** per batch, making the satellite Doppler a statistical approximation
over all possible orbit directions -- not deterministic from the actual geometry.

### Limitation 3 -- doppler_enabled=False is an Oversimplified Precompensation

Setting doppler_enabled=False removes the **entire** satellite Doppler term uniformly.
This is not a physically accurate precompensation because:

1. It removes the same scalar from all clusters, whereas true precompensation removes only
   the **LoS (beam-center) component** of the satellite velocity.
2. The **per-cluster sat residual** -- the difference due to each cluster departure angle
   vs. the LoS direction -- is never computed or retained.

The correct post-precompensation residual Doppler per cluster is:

`
f_D,residual_l = r_hat_arr_l . v_UE / lambda_0
               + (r_hat_dep_l - r_hat_LoS) . v_sat / lambda_0
                  ^^^^^^ this second term is absent in doppler_enabled=False ^^^^^^
`

### Limitation 4 -- True Sat-UE Doppler (Per-Cluster, Real Geometry) Not Available

There is no mode in the current library that models the full sat-UE Doppler using
the actual 3D geometry for both ends, per cluster:

| doppler_enabled | What is modelled |
|---|---|
| True  | UE Doppler (per-cluster) + sat Doppler (uniform scalar, random direction) |
| False | UE Doppler only (per-cluster) -- misnamed as precompensated |
| not available | True sat-UE per-cluster Doppler using real geometry |
| not available | Correct residual: UE + per-cluster sat residual after LoS precomp |

---

## 4. Required Code Modifications

The following changes implement three modes selectable via a new doppler_mode parameter:

- ull           -- True sat-UE Doppler, per-cluster, real 3D geometry
- precompensated -- Correct residual: UE + per-cluster sat angular-spread residual
- ue_only        -- UE Doppler only (equivalent to current doppler_enabled=False)

### 4.1 OpenNTN/channel_coefficients.py -- Topology class

Add s_velocities and doppler_mode; remove the random sat_speed approximation.

**Original signature:**
`python
def __init__(self, velocities, moving_end, los_aoa, los_aod, los_zoa, los_zod,
             los, distance_3d, tx_orientations, rx_orientations,
             bs_height, elevation_angle, doppler_enabled):
`

**Modified signature:**
`python
def __init__(self, velocities,
             bs_velocities,               # NEW: SAT 3D velocity [batch, num_bs, 3]
             moving_end, los_aoa, los_aod, los_zoa, los_zod,
             los, distance_3d, tx_orientations, rx_orientations,
             bs_height, elevation_angle, doppler_enabled,
             doppler_mode='full'):        # NEW: 'full' | 'precompensated' | 'ue_only'

    # Keep all existing assignments ...
    self.bs_velocities = bs_velocities   # ADD
    self.doppler_mode  = doppler_mode    # ADD

    # REMOVE the random sat_speed block (lines ~99-103):
    # sat_speed = compute_satellite_speed(bs_height)
    # max_sat_speed_for_elevation_angle = cos(elevation_angle*(PI/180)) * sat_speed
    # random_direction_per_batch = tf.random.uniform(...)
    # self.sat_speed = cos(random_direction_per_batch) * max_sat_speed_for_elevation_angle
`

---

### 4.2 OpenNTN/channel_coefficients.py -- _step_11_doppler_matrix()

Replace the scalar uniform sat block with per-cluster vector projection.

`python
def _step_11_doppler_matrix(self, topology, aoa, zoa, aod, zod, t):

    lambda_0 = self._lambda_0

    # --- UE Doppler: per-cluster, using scatter arrival angles (unchanged) ---
    v_uts_bar = topology.velocities
    v_uts_bar = tf.expand_dims(v_uts_bar, axis=-1)
    if topology.moving_end == 'rx':
        v_uts_bar = tf.expand_dims(v_uts_bar, 1)
        r_hat_ut  = self._unit_sphere_vector(zoa, aoa)
    elif topology.moving_end == 'tx':
        v_uts_bar = tf.expand_dims(v_uts_bar, 2)
        r_hat_ut  = self._unit_sphere_vector(zod, aod)
    v_uts_bar = tf.expand_dims(tf.expand_dims(v_uts_bar, -3), -3)
    exponent  = 2*PI/lambda_0 * tf.reduce_sum(r_hat_ut * v_uts_bar, -2) * t

    # --- Satellite Doppler: per-cluster, using departure angles (NEW) ---------
    is_ntn = tf.less_equal(600000.0, topology.bs_height)
    if is_ntn and topology.doppler_mode in ('full', 'precompensated'):

        # Build sat velocity tensor broadcastable over [batch, tx, rx, cluster, ray]
        v_sat_bar = topology.bs_velocities                 # [batch, num_bs, 3]
        v_sat_bar = tf.expand_dims(v_sat_bar, axis=-1)     # [batch, num_bs, 3, 1]
        v_sat_bar = tf.expand_dims(v_sat_bar, 2)
        v_sat_bar = tf.expand_dims(tf.expand_dims(v_sat_bar, -3), -3)

        # Per-cluster sat Doppler using AoD / ZoD (departure angles per cluster)
        r_hat_sat_cl = self._unit_sphere_vector(zod, aod)
        exp_sat_cl   = 2*PI/lambda_0 * tf.reduce_sum(r_hat_sat_cl * v_sat_bar, -2) * t

        if topology.doppler_mode == 'full':
            # Mode A: add full per-cluster sat Doppler (no precompensation)
            exponent = exponent + exp_sat_cl

        elif topology.doppler_mode == 'precompensated':
            # Mode B: per-cluster sat Doppler MINUS LoS direction (beam-center)
            los_zod_exp = tf.expand_dims(tf.expand_dims(topology.los_zod, -1), -1)
            los_aod_exp = tf.expand_dims(tf.expand_dims(topology.los_aod, -1), -1)
            r_hat_los   = self._unit_sphere_vector(los_zod_exp, los_aod_exp)
            exp_sat_los = 2*PI/lambda_0 * tf.reduce_sum(r_hat_los * v_sat_bar, -2) * t
            # Residual = per-cluster minus LoS (angular-spread residual only)
            exponent = exponent + (exp_sat_cl - exp_sat_los)

        # doppler_mode == 'ue_only': no satellite term added

    h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
    return h_doppler
`

---

### 4.3 OpenNTN/system_level_channel.py -- Topology Construction

Pass s_velocities and doppler_mode when building the Topology object.

`python
topology = Topology(
    velocities      = self._scenario.ut_velocities,     # UE 3D velocity [batch, num_ut, 3]
    bs_velocities   = self._scenario.bs_velocities,     # SAT 3D velocity [batch, num_bs, 3]  NEW
    moving_end      = moving_end,
    los_aoa         = deg_2_rad(self._scenario.los_aoa),
    los_aod         = deg_2_rad(self._scenario.los_aod),
    los_zoa         = deg_2_rad(self._scenario.los_zoa),
    los_zod         = deg_2_rad(self._scenario.los_zod),
    los             = self._scenario.los,
    distance_3d     = self._scenario.distance_3d,
    tx_orientations = tx_orientations,
    rx_orientations = rx_orientations,
    bs_height       = self._scenario._bs_loc[:,:,2][0],
    elevation_angle = self._scenario.elevation_angle,
    doppler_enabled = self._scenario.doppler_enabled,
    doppler_mode    = self._scenario.doppler_mode,      # NEW
)
`

---

### 4.4 OpenNTN/system_level_scenario.py -- Store bs_velocities and doppler_mode

In __init__(), add the new parameter and initialise the stored satellite velocity:

`python
def __init__(self, carrier_frequency, ut_array, bs_array, direction,
             elevation_angle, enable_pathloss=True, enable_shadow_fading=True,
             doppler_enabled=True,
             doppler_mode='full'):          # NEW parameter
    ...
    self._doppler_mode  = doppler_mode      # ADD
    self._bs_velocities = None              # ADD: will be set via set_topology()
`

In set_topology(), add s_velocities argument:

`python
def set_topology(self, ut_loc, bs_loc, ut_orientations, bs_orientations,
                 ut_velocities, bs_velocities, in_state, los=None):   # ADD bs_velocities
    ...
    self._bs_velocities = bs_velocities     # ADD
`

Add properties:

`python
@property
def bs_velocities(self):
    return self._bs_velocities

@property
def doppler_mode(self):
    return self._doppler_mode
`

---

### 4.5 channel_wGeometry/gen_channel_wGeometry.py -- Wire in Satellite Velocity

The satellite velocity _sat_ENU is already computed at ~line 115. Wire it in:

`python
# Build channel model with the desired Doppler mode
channel_model = channel_class(
    carrier_frequency = carrier_frequency,
    ut_array          = ut_array,
    bs_array          = bs_array,
    direction         = direction,
    elevation_angle   = elevation_angle,
    doppler_enabled   = True,
    doppler_mode      = 'precompensated',   # choose: 'full' | 'precompensated' | 'ue_only'
)

# Pack both UE and satellite velocities
ut_velocities_tensor = tf.constant([[v_UE_ENU]], dtype=tf.float32)    # [1, 1, 3]
bs_velocities_tensor = tf.constant([[v_sat_ENU]], dtype=tf.float32)   # [1, 1, 3]  -- NEW

topology_data = (
    ut_loc_tensor,
    bs_loc_tensor,
    ut_orientations,
    bs_orientations,
    ut_velocities_tensor,
    bs_velocities_tensor,     # NEW: satellite velocity
    in_state
)
channel_model.set_topology(*topology_data, los=True)
`

---

## 5. Mode Summary After Modification

| doppler_enabled | doppler_mode    | Physical meaning |
|---|---|---|
| True  | full            | True sat-UE Doppler: per-cluster sat + per-ray UE, real 3D geometry |
| True  | precompensated  | Correct residual: UE Doppler + per-cluster sat angular-spread residual |
| True  | ue_only         | UE Doppler only (equivalent to original doppler_enabled=False) |
| False | (any)           | UE Doppler only -- backward-compatible with original behaviour |

---

## 6. Impact Magnitude

For LEO at 600 km, 27 GHz carrier, 3GPP NTN cluster angular spread:

| Quantity | Approximate value |
|---|---|
| Satellite orbital speed, abs(v_sat) | ~7,600 m/s |
| Bulk sat Doppler (LoS, elevation 45 deg) | ~350 kHz |
| Per-cluster sat residual after LoS precomp | ~10-100 Hz (depends on scatter angle spread) |
| UE Doppler (50 m/s UE speed, 27 GHz) | ~4.5 Hz |

The per-cluster sat residual (~10-100 Hz) is **comparable to or larger than the UE Doppler
(~4.5 Hz)**. For channel estimation research comparing model fidelity against reality,
ignoring this term introduces a systematic bias in the residual Doppler characterization.

---

## 7. File Change Summary

`
OpenNTN/
|-- channel_coefficients.py     Modify: Topology.__init__ -- add bs_velocities, doppler_mode
|                               Modify: _step_11_doppler_matrix -- per-cluster sat Doppler
|-- system_level_channel.py     Modify: Topology construction -- pass bs_velocities, doppler_mode
+-- system_level_scenario.py    Modify: set_topology -- accept bs_velocities
                                Add:    bs_velocities and doppler_mode properties

channel_wGeometry/
+-- gen_channel_wGeometry.py    Modify: pass v_sat_ENU as bs_velocities_tensor
                                Modify: set doppler_mode on channel_model
`

---

*Last updated: 2026-07-23 -- Based on OpenNTN library code analysis against*
*3GPP TR 38.811 Section 6.8.1 and TR 38.901 Section 7.5 (Step 11 Doppler).*
