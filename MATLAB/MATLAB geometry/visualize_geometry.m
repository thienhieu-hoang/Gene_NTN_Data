% VISUALIZE_GEOMETRY Non-Terrestrial Network (NTN) Geometry & Doppler Visualization
%
% This script simulates and visualizes a Low Earth Orbit (LEO) satellite pass
% over a ground User Equipment (UE) in a Non-Terrestrial Network (NTN). It models:
%   1. WGS-84 Earth coordinate systems (ECI, ECEF, Local ENU).
%   2. LEO Keplerian circular orbit trajectory and velocities.
%   3. UE ground motion and ECEF velocity transformations.
%   4. Beam pointing modes: 'fixed' cell, 'nadir' pointing, or 'tracking' the UE.
%   5. Ray-sphere intersection for the physical beam footprint on the Earth's surface.
%   6. Slant range, elevation angle, and Doppler shift dynamics (the S-curve).
%   7. Beam-level pre-compensation and residual Doppler.
%
% Reference: Geometry.md (Section 1 to 6)
%
% Antigravity Coding Assistant - Google DeepMind

clear; clc; close all;

%% =========================================================================
%% 1. CONFIGURATION PARAMETERS
%% =========================================================================

% --- Key System Settings (Configurable) ---
f_c = 2.0e9;                 % Carrier frequency (Hz) (e.g., 2.0 GHz S-band)

% --- Satellite Orbital Parameters ---
h_s = 600e3;                 % Orbit altitude (m) (600 km LEO)
inclination_deg = 55.0;      % Orbit inclination (degrees)
% Note: RAAN (Omega) and initial argument of latitude (u_0) will be dynamically 
% calculated to align the orbit pass directly over the UE.

% --- Beam Boresight & Footprint Parameters ---
beamwidth_deg = 3.5;         % Half-Power Beamwidth (HPBW) (degrees)
% Beam pointing mode:
%   'fixed'    - Boresight points at a fixed coordinate on the ground (UE's initial position)
%   'nadir'    - Boresight points straight down (sub-satellite point)
%   'tracking' - Boresight dynamically steers to track the moving UE
beamMode = 'fixed'; 

% --- User Equipment (UE) Parameters ---
phi_UE_deg = 37.7749;        % UE Initial Latitude (degrees, e.g., San Francisco)
lambda_UE_deg = -122.4194;   % UE Initial Longitude (degrees)
h_UE = 100.0;                % UE Altitude above ellipsoid (m)
v_UE_ground = 20.0;          % UE speed along the ground (m/s, ~72 km/h)
heading_deg = 45.0;          % UE heading azimuth (degrees, 0 = North, 90 = East)

% --- Visualization & Time Controls ---
zoomView = true;             % true: Zoomed in on UE region (recommended), false: Global Earth view
time_step = 1.0;             % Time step for simulation (s)
time_duration = 300;         % Simulation window: [-t_duration/2, +t_duration/2] (s)

% --- Velocity Vector Scaling for Plotting (in meters per m/s) ---
vel_scale_sat = 1.0e5 / 7500 * 5;  % Scales satellite arrow to ~500 km
vel_scale_ue = 1.0e5 / 50 * 3.5;   % Scales UE arrow to ~350 km
vel_scale_bc = 1.0e5 / 7000 * 4;   % Scales beam center arrow to ~400 km

% --- Environmental & Physical Constants (Static) ---
c = 299792458;               % Speed of light (m/s)
omega_E = 7.292115e-5;       % Earth's rotation rate (rad/s)
mu = 3.986004418e14;         % Earth's gravitational parameter (m^3/s^2)

% --- WGS-84 Ellipsoid Parameters ---
a = 6378137.0;               % Semi-major axis (m)
e2 = 6.69437999e-3;          % First eccentricity squared

%% =========================================================================
%% 2. GEOMETRIC PREPARATION & ORBIT ALIGNMENT
%% =========================================================================

% Convert angles to radians
inclination = inclination_deg * pi/180;
phi_UE = phi_UE_deg * pi/180;
lambda_UE = lambda_UE_deg * pi/180;
heading = heading_deg * pi/180;
theta_b = (beamwidth_deg / 2) * pi/180; % Half-beamwidth

% Orbit radius & angular/linear speeds
r = a + h_s;
omega_s = sqrt(mu / r^3);
v_sat_orbit = sqrt(mu / r);

% Calculate UE's initial ECEF position
N_phi_0 = a / sqrt(1 - e2 * sin(phi_UE)^2);
r_ue_ECEF_0 = [ ...
    (N_phi_0 + h_UE) * cos(phi_UE) * cos(lambda_UE); ...
    (N_phi_0 + h_UE) * cos(phi_UE) * sin(lambda_UE); ...
    (N_phi_0 * (1 - e2) + h_UE) * sin(phi_UE) ...
];

% Calculate UE's velocity vector in local ENU frame
v_UE_ENU = [v_UE_ground * sin(heading); v_UE_ground * cos(heading); 0];

% Rotation matrix from local ENU to ECEF at UE's initial location
R_ENU2ECEF = [ ...
    -sin(lambda_UE), -sin(phi_UE)*cos(lambda_UE), cos(phi_UE)*cos(lambda_UE); ...
     cos(lambda_UE), -sin(phi_UE)*sin(lambda_UE), cos(phi_UE)*sin(lambda_UE); ...
     0,               cos(phi_UE),                sin(phi_UE) ...
];

% UE velocity vector in ECEF frame
ue_vel_ECEF = R_ENU2ECEF * v_UE_ENU;

% --- Analytical Orbit Alignment (closest approach at t_mid) ---
% Find orbit argument of latitude (u_mid) when satellite is at UE's latitude
if inclination >= abs(phi_UE)
    u_mid = asin(sin(phi_UE) / sin(inclination));
else
    u_mid = sign(phi_UE) * pi/2; % Max latitude reachable
end

% Find RAAN (Omega) to align satellite longitude with UE longitude at t_mid
Omega_RAAN = lambda_UE - atan2(sin(u_mid)*cos(inclination), cos(u_mid));

% Define time grid (centered around closest approach)
t_mid = 0;
time_grid = -time_duration/2 : time_step : time_duration/2;
N_steps = length(time_grid);

%% =========================================================================
%% 3. TIME-VARYING DYNAMICS PRECOMPUTATION
%% =========================================================================

sat_ECEF_all = zeros(3, N_steps);
sat_vel_ECEF_all = zeros(3, N_steps);
ue_ECEF_all = zeros(3, N_steps);
bc_ECEF_all = zeros(3, N_steps);
bc_vel_ECEF_all = zeros(3, N_steps);

slant_range_all = zeros(1, N_steps);
elev_all = zeros(1, N_steps);
doppler_all = zeros(1, N_steps);
doppler_beam_all = zeros(1, N_steps);

for k = 1:N_steps
    t = time_grid(k);
    
    % Earth rotation angle (GST) at time t (assume GST(0) = 0)
    theta_G = omega_E * t;
    R_z = [ ...
         cos(theta_G), sin(theta_G), 0; ...
        -sin(theta_G), cos(theta_G), 0; ...
         0,            0,            1 ...
    ];
    
    % Satellite position & velocity in ECI
    u_t = omega_s * (t - t_mid) + u_mid;
    
    r_sat_ECI = [ ...
        r * (cos(u_t)*cos(Omega_RAAN) - sin(u_t)*sin(Omega_RAAN)*cos(inclination)); ...
        r * (cos(u_t)*sin(Omega_RAAN) + sin(u_t)*cos(Omega_RAAN)*cos(inclination)); ...
        r * sin(u_t)*sin(inclination) ...
    ];

    v_sat_ECI = [ ...
        v_sat_orbit * (-sin(u_t)*cos(Omega_RAAN) - cos(u_t)*sin(Omega_RAAN)*cos(inclination)); ...
        v_sat_orbit * (-sin(u_t)*sin(Omega_RAAN) + cos(u_t)*cos(Omega_RAAN)*cos(inclination)); ...
        v_sat_orbit * cos(u_t)*sin(inclination) ...
    ];
    
    % Convert Satellite position and velocity to ECEF (accounts for Coriolis term)
    r_sat_ECEF = R_z * r_sat_ECI;
    
    omega_cross_r = [ ...
        -omega_E * r_sat_ECI(2); ...
         omega_E * r_sat_ECI(1); ...
         0 ...
    ];
    v_sat_ECEF = R_z * (v_sat_ECI - omega_cross_r);
    
    sat_ECEF_all(:, k) = r_sat_ECEF;
    sat_vel_ECEF_all(:, k) = v_sat_ECEF;
    
    % UE Position in ECEF (linearized motion on tangent plane, projected back to surface)
    r_ue_ECEF = r_ue_ECEF_0 + ue_vel_ECEF * t;
    r_ue_ECEF = r_ue_ECEF * (norm(r_ue_ECEF_0) / norm(r_ue_ECEF)); % Maintain exact altitude
    ue_ECEF_all(:, k) = r_ue_ECEF;
    
    % Slant Range and Elevation Angle from UE to Satellite
    v_los = r_sat_ECEF - r_ue_ECEF;
    slant_range = norm(v_los);
    slant_range_all(k) = slant_range;
    
    u_normal = r_ue_ECEF / norm(r_ue_ECEF);
    u_los = v_los / slant_range;
    elev_rad = asin(dot(u_normal, u_los));
    elev_all(k) = elev_rad * 180/pi;
    
    % Doppler Shift (UE-specific)
    v_rel_ECEF = v_sat_ECEF - ue_vel_ECEF;
    doppler_all(k) = - (dot(v_rel_ECEF, u_los) / c) * f_c;
    
    % Beam Center Position & Velocity in ECEF
    switch beamMode
        case 'fixed'
            r_bc = r_ue_ECEF_0;
            v_bc = [0; 0; 0];
        case 'nadir'
            r_bc = a * (r_sat_ECEF / norm(r_sat_ECEF));
            v_bc = (a / norm(r_sat_ECEF)) * v_sat_ECEF;
        case 'tracking'
            r_bc = r_ue_ECEF;
            v_bc = ue_vel_ECEF;
    end
    bc_ECEF_all(:, k) = r_bc;
    bc_vel_ECEF_all(:, k) = v_bc;
    
    % Beam-level Doppler Shift (to beam center)
    v_los_beam = r_sat_ECEF - r_bc;
    slant_range_beam = norm(v_los_beam);
    u_los_beam = v_los_beam / slant_range_beam;
    v_rel_beam = v_sat_ECEF - v_bc;
    doppler_beam_all(k) = - (dot(v_rel_beam, u_los_beam) / c) * f_c;
end

% Residual Doppler shift (difference after beam-level pre-compensation)
doppler_residual_all = doppler_all - doppler_beam_all;

%% =========================================================================
%% 4. GRAPHICS INITIALIZATION
%% =========================================================================

% Color palette definition (slate dark theme)
% Color palette definition (clean white theme)
bg_color = [1, 1, 1];               % White background
earth_color = [0.85, 0.90, 0.95];   % Light blue-gray Earth sphere color
grid_color = [0.70, 0.75, 0.80];    % Soft gray grid line color
gold = [0.85, 0.60, 0.05];          % Darker gold for satellite visibility
red = [0.85, 0.20, 0.20];           % UE color
cyan = [0.00, 0.55, 0.65];          % Deeper cyan for beam footprint visibility
light_cyan = [0.00, 0.60, 0.75];    % Beam center color
emerald = [0.10, 0.65, 0.35];       % Satellite velocity color (emerald)
orange = [0.85, 0.40, 0.05];        % UE velocity color (orange)

% Create main figures
fig1 = figure('Name', '3D NTN Geometry Simulation', 'Position', [100, 100, 800, 750], 'Color', bg_color);
ax1 = axes('Parent', fig1, 'Color', bg_color, 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
grid(ax1, 'on'); hold(ax1, 'on'); view(ax1, 3);
axis(ax1, 'equal');

% Draw Earth sphere
[XS, YS, ZS] = sphere(45);
surf_earth = surf(ax1, XS*a, YS*a, ZS*a, ...
    'FaceColor', earth_color, 'EdgeColor', grid_color, 'FaceAlpha', 0.85, 'EdgeAlpha', 0.25);

% Draw Equator & Prime Meridian for coordinate context
t_pm = linspace(0, 2*pi, 100);
plot3(ax1, a*cos(t_pm), a*sin(t_pm), zeros(size(t_pm)), 'Color', [0.3 0.4 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');
plot3(ax1, a*cos(t_pm), zeros(size(t_pm)), a*sin(t_pm), 'Color', [0.3 0.4 0.5], 'LineWidth', 0.5, 'HandleVisibility', 'off');

% Plot Satellite Orbit Path in ECEF
plot_orbit = plot3(ax1, sat_ECEF_all(1,:), sat_ECEF_all(2,:), sat_ECEF_all(3,:), ...
    '--', 'Color', [gold 0.5], 'LineWidth', 1.5, 'DisplayName', 'LEO Orbit (ECEF)');

% Plot UE Ground Path (magnified for visibility if zoomView is false)
plot_ue_path = plot3(ax1, ue_ECEF_all(1,:), ue_ECEF_all(2,:), ue_ECEF_all(3,:), ...
    '-', 'Color', red, 'LineWidth', 2, 'DisplayName', 'UE Path');

% Dynamic element placeholders
h_sat = plot3(ax1, NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerFaceColor', gold, 'MarkerEdgeColor', 'w', 'DisplayName', 'LEO Satellite');
h_ue = plot3(ax1, NaN, NaN, NaN, '^', 'MarkerSize', 8, 'MarkerFaceColor', red, 'MarkerEdgeColor', 'w', 'DisplayName', 'UE');
h_bc = plot3(ax1, NaN, NaN, NaN, 'x', 'MarkerSize', 10, 'LineWidth', 2.5, 'Color', light_cyan, 'DisplayName', 'Beam Center');

h_los = plot3(ax1, NaN, NaN, NaN, '--', 'Color', [0.8 0.8 0.8], 'LineWidth', 1.2, 'DisplayName', 'UE-Sat Line-of-Sight');
h_boresight = plot3(ax1, NaN, NaN, NaN, ':', 'Color', cyan, 'LineWidth', 1.8, 'DisplayName', 'Beam Boresight');
h_footprint = patch(ax1, NaN, NaN, NaN, cyan, 'FaceAlpha', 0.3, 'EdgeColor', cyan, 'LineWidth', 2, 'DisplayName', 'Beam Footprint');

% Velocity Vector Quivers (0 disables auto-scaling, allowing explicit scale)
h_sat_vel = quiver3(ax1, NaN, NaN, NaN, NaN, NaN, NaN, 0, 'Color', emerald, 'LineWidth', 2, 'MaxHeadSize', 0.4, 'DisplayName', 'Sat Velocity (Scaled)');
h_ue_vel = quiver3(ax1, NaN, NaN, NaN, NaN, NaN, NaN, 0, 'Color', orange, 'LineWidth', 2.5, 'MaxHeadSize', 0.5, 'DisplayName', 'UE Velocity (Scaled)');
h_bc_vel = quiver3(ax1, NaN, NaN, NaN, NaN, NaN, NaN, 0, 'Color', light_cyan, 'LineWidth', 1.5, 'MaxHeadSize', 0.4, 'DisplayName', 'Beam Center Vel (Scaled)');

% Visual styling of axes
xlabel(ax1, 'ECEF X (meters)'); ylabel(ax1, 'ECEF Y (meters)'); zlabel(ax1, 'ECEF Z (meters)');
title(ax1, 'Satellite NTN 3D Geometric Pass', 'Color', 'k', 'FontSize', 12);
colormap(ax1, 'cool');
shading(ax1, 'interp');
light('Parent', ax1, 'Position', [1.5*a, 1.5*a, 1.5*a], 'Style', 'local');
lighting(ax1, 'gouraud');

% Apply view constraints
if zoomView
    % Zoom tightly on the area around the UE
    zoom_width = 1.2e6; % 1200 km field of view
    xlim(ax1, [r_ue_ECEF_0(1) - zoom_width, r_ue_ECEF_0(1) + zoom_width]);
    ylim(ax1, [r_ue_ECEF_0(2) - zoom_width, r_ue_ECEF_0(2) + zoom_width]);
    zlim(ax1, [r_ue_ECEF_0(3) - zoom_width, r_ue_ECEF_0(3) + zoom_width]);
else
    % Global Earth view
    xlim(ax1, [-1.3*r, 1.3*r]);
    ylim(ax1, [-1.3*r, 1.3*r]);
    zlim(ax1, [-1.3*r, 1.3*r]);
end
legend(ax1, 'TextColor', 'k', 'Color', 'w', 'EdgeColor', grid_color, 'Location', 'northeast');

% Set up Doppler dynamics figure (S-curve)
fig2 = figure('Name', 'NTN Doppler & Elevation Dynamics', 'Position', [910, 100, 750, 750], 'Color', bg_color);

% Subplot 1: Doppler dynamics (S-Curve)
subplot(2,1,1);
plot(time_grid, doppler_all / 1000, 'Color', red, 'LineWidth', 2.0); hold on;
plot(time_grid, doppler_beam_all / 1000, 'Color', light_cyan, 'LineWidth', 1.5);
plot(time_grid, doppler_residual_all / 1000, ':', 'Color', orange, 'LineWidth', 2.0);
grid on;
set(gca, 'Color', bg_color, 'XColor', 'k', 'YColor', 'k', 'GridColor', grid_color);
xlabel('Time Relative to Closest Approach (seconds)');
ylabel('Doppler Shift (kHz)');
title('Doppler Shift Dynamics (S-Curve & Pre-compensation)');
legend('UE Doppler (Uncompensated)', 'Beam-Center Doppler', 'Residual UE Doppler (Compensated)', ...
       'TextColor', 'k', 'Color', 'w', 'EdgeColor', grid_color, 'Location', 'southwest');
h_doppler_cursor = line([time_grid(1), time_grid(1)], ylim, 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--');

% Subplot 2: Elevation angle
subplot(2,1,2);
plot(time_grid, elev_all, 'Color', cyan, 'LineWidth', 2.0); hold on;
grid on;
set(gca, 'Color', bg_color, 'XColor', 'k', 'YColor', 'k', 'GridColor', grid_color);
xlabel('Time Relative to Closest Approach (seconds)');
ylabel('Elevation Angle (degrees)');
title('Satellite Elevation Angle from UE perspective');
ylim([0, 95]);
h_elev_cursor = line([time_grid(1), time_grid(1)], [0, 95], 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--');

%% =========================================================================
%% 5. SIMULATION ANIMATION LOOP
%% =========================================================================

% --- GIF Export Configuration ---
gif_frames = 15;
gif_indices = round(linspace(1, N_steps, gif_frames));
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
gif_path_3d = fullfile(script_dir, 'ntn_geometry_3d_simulation.gif');
gif_path_plots = fullfile(script_dir, 'ntn_doppler_elevation_simulation.gif');

% Ensure the 3D figure is active
figure(fig1);
for k = 1:N_steps
    % Extract current positions/velocities
    r_sat = sat_ECEF_all(:, k);
    v_sat = sat_vel_ECEF_all(:, k);
    r_ue = ue_ECEF_all(:, k);
    v_ue = ue_vel_ECEF;
    r_bc = bc_ECEF_all(:, k);
    v_bc = bc_vel_ECEF_all(:, k);
    
    % Update basic markers
    set(h_sat, 'XData', r_sat(1), 'YData', r_sat(2), 'ZData', r_sat(3));
    set(h_ue, 'XData', r_ue(1), 'YData', r_ue(2), 'ZData', r_ue(3));
    set(h_bc, 'XData', r_bc(1), 'YData', r_bc(2), 'ZData', r_bc(3));
    
    % Update link lines
    set(h_los, 'XData', [r_ue(1), r_sat(1)], 'YData', [r_ue(2), r_sat(2)], 'ZData', [r_ue(3), r_sat(3)]);
    set(h_boresight, 'XData', [r_sat(1), r_bc(1)], 'YData', [r_sat(2), r_bc(2)], 'ZData', [r_sat(3), r_bc(3)]);
    
    % --- Physical Beam Footprint Ray-Sphere Intersection ---
    v_boresight = r_bc - r_sat;
    u_b = v_boresight / norm(v_boresight);
    
    % Determine local coordinate basis perpendicular to boresight
    if abs(u_b(3)) < 0.9
        ref_vec = [0; 0; 1];
    else
        ref_vec = [1; 0; 0];
    end
    u_x = cross(u_b, ref_vec);
    u_x = u_x / norm(u_x);
    u_y = cross(u_b, u_x);
    u_y = u_y / norm(u_y);
    
    % Generate ring of cone direction vectors
    phi_az = linspace(0, 2*pi, 72); % 72 azimuth points
    N_pts = length(phi_az);
    v_unit = cos(theta_b) * repmat(u_b, 1, N_pts) + sin(theta_b) * (u_x * cos(phi_az) + u_y * sin(phi_az));
    
    % Solve quadratic ray-sphere intersection
    B_pts = sum(r_sat .* v_unit, 1);
    C_pts = sum(r_sat.^2) - a^2;
    disc = B_pts.^2 - C_pts;
    
    r_footprint_pts = NaN(3, N_pts);
    valid = (disc >= 0) & (B_pts < 0);
    d_intersect = -B_pts(valid) - sqrt(disc(valid));
    r_footprint_pts(:, valid) = repmat(r_sat, 1, sum(valid)) + v_unit(:, valid) .* d_intersect;
    
    % Apply scale factor to footprint points to float slightly above Earth surface
    r_footprint_plot = r_footprint_pts * 1.0006; 
    
    % Update footprint patch
    set(h_footprint, 'XData', r_footprint_plot(1, :), 'YData', r_footprint_plot(2, :), 'ZData', r_footprint_plot(3, :));
    
    % --- Update Velocity Vectors ---
    v_sat_scaled = v_sat * vel_scale_sat;
    v_ue_scaled = v_ue * vel_scale_ue;
    v_bc_scaled = v_bc * vel_scale_bc;
    
    set(h_sat_vel, 'XData', r_sat(1), 'YData', r_sat(2), 'ZData', r_sat(3), ...
                   'UData', v_sat_scaled(1), 'VData', v_sat_scaled(2), 'WData', v_sat_scaled(3));
    set(h_ue_vel, 'XData', r_ue(1), 'YData', r_ue(2), 'ZData', r_ue(3), ...
                  'UData', v_ue_scaled(1), 'VData', v_ue_scaled(2), 'WData', v_ue_scaled(3));
    set(h_bc_vel, 'XData', r_bc(1), 'YData', r_bc(2), 'ZData', r_bc(3), ...
                  'UData', v_bc_scaled(1), 'VData', v_bc_scaled(2), 'WData', v_bc_scaled(3));
                  
    % --- Check if UE is inside the physical beam footprint cone ---
    v_sat2ue = r_ue - r_sat;
    u_sat2ue = v_sat2ue / norm(v_sat2ue);
    cos_angle = dot(u_sat2ue, u_b);
    is_inside = cos_angle >= cos(theta_b);
    
    if is_inside
        inside_str = 'YES';
        set(h_footprint, 'FaceColor', cyan, 'EdgeColor', cyan);
    else
        inside_str = 'NO';
        set(h_footprint, 'FaceColor', [0.4 0.4 0.4], 'EdgeColor', [0.5 0.5 0.5]);
    end
    
    % --- Update Titles & Text Metrics ---
    title_str = sprintf('LEO Satellite Pass - Time: %+.1f s (BMode: %s)\nElevation: %.1f^{\\circ} | Range: %.1f km | Doppler: %+.2f kHz\nUE Inside Beam: %s', ...
        time_grid(k), beamMode, elev_all(k), slant_range_all(k)/1000, doppler_all(k)/1000, inside_str);
    title(ax1, title_str, 'Color', 'k', 'FontSize', 11);
    
    % --- Update Sweep Cursor on Doppler/Elevation Plots ---
    set(h_doppler_cursor, 'XData', [time_grid(k), time_grid(k)]);
    set(h_elev_cursor, 'XData', [time_grid(k), time_grid(k)]);
    
    % --- Capture Snapshot at Closest Approach (t = 0) ---
    if time_grid(k) == 0
        script_dir = fileparts(mfilename('fullpath'));
        if isempty(script_dir)
            script_dir = pwd;
        end
        pdf_path_3d = fullfile(script_dir, 'ntn_geometry_3d_snapshot.pdf');
        pdf_path_plots = fullfile(script_dir, 'ntn_doppler_elevation_snapshot.pdf');
        
        % Save cropped vector PDF of both figures
        exportgraphics(fig1, pdf_path_3d, 'ContentType', 'vector');
        exportgraphics(fig2, pdf_path_plots, 'ContentType', 'vector');
        fprintf('Saved cropped PDF snapshots to:\n  - %s\n  - %s\n', pdf_path_3d, pdf_path_plots);
    end
    
    % Render and brief pause
    drawnow;
    
    % --- Capture GIF Frames ---
    if ismember(k, gif_indices)
        frame1 = getframe(fig1);
        im1 = frame2im(frame1);
        [imind1, cm1] = rgb2ind(im1, 256);
        
        frame2 = getframe(fig2);
        im2 = frame2im(frame2);
        [imind2, cm2] = rgb2ind(im2, 256);
        
        if k == gif_indices(1)
            % Clean existing files if they exist
            if exist(gif_path_3d, 'file'), delete(gif_path_3d); end
            if exist(gif_path_plots, 'file'), delete(gif_path_plots); end
            
            imwrite(imind1, cm1, gif_path_3d, 'gif', 'Loopcount', inf, 'DelayTime', 0.4);
            imwrite(imind2, cm2, gif_path_plots, 'gif', 'Loopcount', inf, 'DelayTime', 0.4);
            fprintf('Started GIF generation...\n');
        else
            imwrite(imind1, cm1, gif_path_3d, 'gif', 'WriteMode', 'append', 'DelayTime', 0.4);
            imwrite(imind2, cm2, gif_path_plots, 'gif', 'WriteMode', 'append', 'DelayTime', 0.4);
        end
    end
    
    pause(0.01);
end

fprintf('Simulation complete! You can interact with the 3D view using the rotate tool in MATLAB.\n');
