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
   $$\vec{v}_{\text{UE, ENU}} = [v_{\text{speed}} \sin(\theta_{\text{heading}}), \  v_{\text{speed}} \cos(\theta_{\text{heading}}), \  0.0]^T$$
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
  * **Orbital speed**: $v_{\text{orbit}} = \sqrt{\mu / r_{\text{orbit}}} \approx 7.56\text{ km/s}$
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

---

## 3. Global ECEF to Local Tangent Frame (ENU) Transformation

The primary global coordinate system used to derive orbital mechanics and global positioning is the **Earth-Centered Earth-Fixed (ECEF)** frame $(X, Y, Z)$. However, for wireless channel coefficient generation and ray-tracing (angle of arrival/departure), coordinates and velocity vectors are transformed into the **local East-North-Up (ENU)** Cartesian coordinate system $(x, y, z)$ anchored at the initial UE location $(\phi_{\text{UE}}, \lambda_{\text{UE}}, h_{\text{UE}})$.

### A. Rotation Matrix $\mathbf{R}_{\text{ECEF}\to\text{ENU}}$
The rotation matrix from ECEF to the local ENU frame at initial latitude $\phi_{\text{UE}}$ and longitude $\lambda_{\text{UE}}$ is:

$$\mathbf{R}_{\text{ECEF}\to\text{ENU}} = \begin{bmatrix} 
-\sin(\lambda_{\text{UE}}) & \cos(\lambda_{\text{UE}}) & 0 \\ 
-\sin(\phi_{\text{UE}})\cos(\lambda_{\text{UE}}) & -\sin(\phi_{\text{UE}})\sin(\lambda_{\text{UE}}) & \cos(\phi_{\text{UE}}) \\ 
\cos(\phi_{\text{UE}})\cos(\lambda_{\text{UE}}) & \cos(\phi_{\text{UE}})\sin(\lambda_{\text{UE}}) & \sin(\phi_{\text{UE}}) 
\end{bmatrix}$$

### B. Position Transformation
Given a 3D position vector in ECEF $\vec{r}_{\text{ECEF}}$, its local ENU position relative to the initial UE anchor $\vec{r}_{\text{UE, ECEF}}(0)$ is:

$$\vec{p}_{\text{ENU}} = \mathbf{R}_{\text{ECEF}\to\text{ENU}} \cdot \left( \vec{r}_{\text{ECEF}} - \vec{r}_{\text{UE, ECEF}}(0) \right)$$

### C. Velocity Vector Transformation
Given a 3D velocity vector in ECEF $\vec{v}_{\text{ECEF}}$ (for either the satellite or the UE), its 3D velocity vector in the local ENU frame is:

$$\vec{v}_{\text{ENU}} = \mathbf{R}_{\text{ECEF}\to\text{ENU}} \cdot \vec{v}_{\text{ECEF}}$$

---

## 4. How to Configure the Satellite Trajectory in the Code

You can customize the orbit, alignment, and pass trajectory by modifying specific lines in `gen_channel_v2_wGeometry_straightDoppler.py`:

### A. Orbit Altitude & Speed
* **To change the altitude**: Modify `satellite_height` around **Line 42** (in meters, e.g., `satellite_height = 600000.0` for 600 km).
* Changing this parameter automatically updates the orbital radius (`r_orbit`), the orbital angular rate (`omega_s`), and the linear satellite velocity (`v_sat_orbit`) via Keplerian equations.

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
