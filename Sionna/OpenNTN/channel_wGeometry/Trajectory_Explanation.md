# Trajectory Modeling for LEO Satellite and UE in NTN

This document explains the physical modeling, parameters, and coordinates of the User Equipment (UE) and Low Earth Orbit (LEO) satellite trajectories used in the `channel_wGeometry` simulation scripts.

---

## 1. UE (User Equipment) Trajectory

The UE is located on the ground and moves along a straight line on the Earth's surface (local tangent plane).

### A. Configuration Parameters
* `phi_UE_deg` / `lambda_UE_deg`: The initial latitude and longitude of the UE (e.g., San Francisco: `37.7749° N`, `-122.4194° E`).
* `h_UE`: Altitude of the UE (meters above the ellipsoid).
* `ue_speed`: The ground speed of the UE (meters per second, e.g., `50.0 m/s` or `180 km/h`).
* `heading_deg`: The azimuth movement direction, measured in degrees clockwise from North ($0^\circ$).

### B. Heading Azimuth Definition
The heading direction is defined as:
* **$0^\circ$**: Due North (increasing latitude)
* **$90^\circ$**: Due East (increasing longitude)
* **$180^\circ$**: Due South (decreasing latitude)
* **$270^\circ$**: Due West (decreasing longitude)
* **$45^\circ$**: Northeast (increasing both East and North equally)

### C. Mathematical Trajectory
1. The heading is converted to local **East-North-Up (ENU)** velocity components:
   $$\vec{v}_{\text{UE, ENU}} = [v_{\text{speed}} \sin(\theta_{\text{heading}}), \  v_{\text{speed}} \cos(\theta_{\text{heading}}), \  0.0]$$
2. It is rotated into the global **ECEF** frame:
   $$\vec{v}_{\text{UE, ECEF}} = \mathbf{R}_{\text{ENU}\to\text{ECEF}} \cdot \vec{v}_{\text{UE, ENU}}$$
3. The UE's position at any time $t$ relative to the start is:
   $$\vec{r}_{\text{UE, ECEF}}(t) = \vec{r}_{\text{UE, ECEF}}(0) + \vec{v}_{\text{UE, ECEF}} \cdot t$$
   *(The vector is normalized to maintain a constant WGS-84 altitude).*

---

## 2. Satellite (LEO) Trajectory

The satellite moves in a circular Keplerian orbit around the Earth, while the Earth rotates underneath it.

### A. Configuration Parameters
* `satellite_height`: Orbit altitude above the Earth's surface (e.g., `600,000 m` or `600 km`). This determines:
  * **Orbital radius**: $r_{\text{orbit}} = R_{\text{Earth}} + h_{\text{satellite}}$
  * **Orbital speed**: $v_{\text{orbit}} = \sqrt{\mu / r_{\text{orbit}}} \approx 7.5\text{ km/s}$
  * **Angular speed**: $\omega_s = \sqrt{\mu / r_{\text{orbit}}^3}$
* `inclination_deg`: The tilt of the orbit plane relative to the Earth's equator (e.g., `55.0°`).

### B. Trajectory Alignment (Zenith Pass at $t=0$)
To ensure the simulation simulates a valid satellite pass directly over the UE, the orbit is aligned dynamically at $t=0$ (closest approach):
* **`u_mid` (Argument of Latitude)**: Solves for the angle along the orbit circle where the satellite reaches the latitude of the UE.
  $$u_{\text{mid}} = \arcsin\left(\frac{\sin(\phi_{\text{UE}})}{\sin(i)}\right)$$
* **`Omega_RAAN` (RAAN)**: Orients the orbit plane around the Earth's polar Z-axis so that the satellite's longitude matches the UE's longitude at $t=0$.

At $t=0$, the satellite is **directly at the Zenith ($90^\circ$ elevation angle)** above the UE.

### C. Orbital Motion Direction (Ascending NE Pass)
As time $t$ increases from $t=0$, the satellite travels forward along its inclined circular path:
1. **Latitude direction**: Since $u_{\text{mid}}$ is in the first quadrant ($\approx 48.4^\circ$ for $37.77^\circ\text{ N}$ lat), the satellite is ascending (climbing from the Equator toward its highest latitude point at $55^\circ\text{ N}$). It moves **North**.
2. **Longitude direction**: The orbital motion combines with the Earth's rotation, resulting in a ground track moving **East**.
3. **Up direction**: Because $t=0$ is the point of closest approach (highest altitude above the UE), the vertical speed component relative to the UE is exactly zero. The satellite's velocity vector is completely parallel to the ground horizon.

Thus, at the snapshot ($t=0$), the satellite is directly overhead, moving **horizontally toward the Northeast (NE)**.

### D. Time Integration (Earth Rotation)
As time progresses, the Earth rotates under the satellite's orbit at $\omega_E = 7.292115 \times 10^{-5}\text{ rad/s}$. The satellite position in the Earth-Centered Earth-Fixed (ECEF) frame is:
$$\vec{r}_{\text{sat, ECEF}}(t) = \mathbf{R}_z(\omega_E \cdot t) \cdot \vec{r}_{\text{sat, ECI}}(t)$$
where $\vec{r}_{\text{sat, ECI}}(t)$ is the inertial orbital position vector.

### E. Understanding the Argument of Latitude ($u$) and Heading Directions
The **Argument of Latitude ($u$)** defines the satellite's position along the orbit circle, starting from the **Ascending Node** (Equator crossing going North). 

To understand which direction the satellite is traveling relative to the ground, we convert this position angle $u$ to a compass **Azimuth angle ($Az$)**.

#### 1. What is the Azimuth Angle ($Az$)?
The Azimuth is the compass heading direction of the satellite's velocity vector, measured in degrees clockwise from North:
* **$0^\circ$**: Heading due **North**
* **$90^\circ$**: Heading due **East**
* **$180^\circ$**: Heading due **South**
* **$270^\circ$**: Heading due **West**

#### 2. The Python Parameter: `u_mid`
In the code script [gen_channel_v2_wGeometry_straightDoppler.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Gene_NTN_Data/Sionna/OpenNTN/channel_wGeometry/gen_channel_v2_wGeometry_straightDoppler.py#L89), the parameter that defines the position of the satellite at the snapshot time ($t=0$) is **`u_mid`**. By changing `u_mid`, you directly control the satellite's location and its direction of travel at $t=0$.

#### 3. How to Set Heading Directions in the Python Code
For a circular orbit tilted at inclination $i = 55.0^\circ$, here is how different values of `u_mid` map to ground heading directions, and how you should configure them:

* **To set heading to North-Northeast (NNE) — Azimuth $\approx 35^\circ$ (Ascending Pass)**:
  * **What to set in Python**:
    ```python
    u_mid = np.arcsin(np.sin(phi_UE) / np.sin(inclination))
    ```
  * **Meaning**: The satellite crosses the latitude of the UE while heading Northwards and Eastwards (climbing up the globe).

* **To set heading to South-Southeast (SSE) — Azimuth $\approx 145^\circ$ (Descending Pass)**:
  * **What to set in Python**:
    ```python
    u_mid = np.pi - np.arcsin(np.sin(phi_UE) / np.sin(inclination))
    ```
  * **Meaning**: The satellite crosses the latitude of the UE while heading Southwards and Eastwards (falling down the globe).

* **To set heading to Exactly East (E) — Azimuth $= 90^\circ$ (Northernmost Peak)**:
  * **What to set in Python**:
    ```python
    # Note: Requires the UE latitude phi_UE to equal the inclination (55 degrees N)
    u_mid = np.pi / 2  # (90 degrees)
    ```
  * **Meaning**: The satellite is at the peak of its latitude curve. For a split second, it has stopped moving North and is heading directly East.

* **To set heading to Exactly East (E) — Azimuth $= 90^\circ$ (Southernmost Peak)**:
  * **What to set in Python**:
    ```python
    # Note: Requires the UE latitude phi_UE to equal -inclination (55 degrees S)
    u_mid = 3 * np.pi / 2  # (270 degrees)
    ```
  * **Meaning**: The satellite is at the bottom of its latitude curve, heading directly East.




---

## 3. How to Configure the Satellite Trajectory in the Code

You can customize the orbit, alignment, and pass trajectory by modifying specific lines in `gen_channel_v2_wGeometry_straightDoppler.py`:

### A. Orbit Altitude & Speed
* **To change the altitude**: Modify `satellite_height` around **Line 42** (in meters, e.g., `satellite_height = 600000.0` for 600 km).
* Changing this parameter automatically updates the orbital radius (`r_orbit`), the orbital angular rate (`omega_s`), and the linear satellite velocity (`v_sat_orbit`) via the Keplerian equations at **Lines 60-62**.

### B. Orbital Inclination (Tilted Plane)
* **To change the tilt**: Modify `inclination_deg` around **Line 43** (in degrees, e.g., `inclination_deg = 55.0`).
  * `0.0°`: Equatorial orbit (satellite travels directly West-to-East over the Equator).
  * `90.0°`: Polar orbit (satellite travels from South-to-North over the poles).
  * `180.0°`: Retrograde equatorial orbit (satellite travels directly East-to-West).

### C. Ascending vs. Descending Node Passes
By default, the satellite crosses the UE on an **ascending pass** (moving Southwest to Northeast).
* **To switch to a descending pass** (moving Northwest to Southeast):
  Modify the `u_mid` logic at **Lines 87–90**.
  * **Original (Ascending)**:
    ```python
    u_mid = np.arcsin(np.sin(phi_UE) / np.sin(inclination))
    ```
  * **Modified (Descending)**:
    ```python
    u_mid = np.pi - np.arcsin(np.sin(phi_UE) / np.sin(inclination))
    ```

### D. Simulating Non-Overhead Passes (Elevation Angle Offset)
By default, the RAAN (`Omega_RAAN`) is calculated so that the orbit plane passes directly overhead ($90^\circ$ peak elevation).
* **To simulate a side-pass** (where the satellite passes East or West of the UE, lowering the peak elevation angle):
  Add a longitude offset to `Omega_RAAN` at **Line 91**:
  * **Offset East**:
    ```python
    Omega_RAAN = (lambda_UE - np.arctan2(np.sin(u_mid) * np.cos(inclination), np.cos(u_mid))) + np.deg2rad(5.0)  # Offset by 5 degrees East
    ```
  * Adjusting this offset dynamically changes the maximum elevation angle and the slant range calculated at **Lines 102–108**.

