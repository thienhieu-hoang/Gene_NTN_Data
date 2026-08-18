%{
========================================================================================
             Target Domain Effective Channel Generator & Visualizer
========================================================================================
OVERVIEW:
  This script generates and visualizes effective 2D Doppler-compensated OFDM 
  channel matrices H_eff [nSubcarriers x nSymbols] for multiple UE samples 
  in the target NTN system domain.
  
  Purely for visualization purposes — plots figures directly without saving files.
========================================================================================
%}

if exist('mfilename', 'builtin') && ~isempty(mfilename('fullpath'))
    script_dir = fileparts(mfilename('fullpath'));
    if ~isempty(script_dir) && exist(script_dir, 'dir')
        cd(script_dir);
    end
end
addpath('..\..\helper\');
addpath('..\');

%% 1. Target Domain System & Simulation Parameters
numUE = 6;                          % Number of UE samples to generate & plot
r_beam = 15000.0;                   % 15 km beam footprint radius
r_ue_max = 14500.0;                 % 14.5 km max UE offset inside beam

simParameters.CarrierFrequency = 2.18e9;   % S-band (2.18 GHz)
simParameters.SatelliteAltitude = 600000;   % 600 km LEO orbit altitude
simParameters.ElevationAngle = 70;          % Target elevation angle (70°)
simParameters.MobileSpeed = 30;             % UE mobile speed (m/s)
simParameters.MobileAltitude = 1.5;         % Mobile antenna height (m)

carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 30;             % 30 kHz SCS
carrier.NSizeGrid = 11;                     % 11 RBs (132 subcarriers)
carrier.CyclicPrefix = 'Normal';            % Normal CP

channel = nrTDLChannel;                     % 3GPP NTN Small-scale channel
channel.DelayProfile = 'NTN-TDL-A';
channel.DelaySpread = 100e-9;              % 100 ns delay spread
channel.NumTransmitAntennas = 1;
channel.NumReceiveAntennas = 1;
channel.MaximumDopplerShift = 0;           % Doppler governed by exact orbital kinematics
channel.RandomStream = "mt19937ar with seed";

waveformInfo = nrOFDMInfo(carrier);
channel.SampleRate = waveformInfo.SampleRate;
c = physconst("lightspeed");
lambda = c / simParameters.CarrierFrequency;

%% 2. Satellite & Beam Center Orbit Geometry
phi_UE_deg = 37.7749;         % Beam Center Latitude
lambda_UE_deg = -122.4194;    % Beam Center Longitude
h_UE = 100.0;                 % Ground altitude (m)
inclination_deg = 55.0;       % Orbit inclination (degrees)

a_wgs84 = 6378137.0;          % Earth semi-major axis (m)
e2 = 6.69437999e-3;           % First eccentricity squared
mu = 3.986004418e14;          % Gravitational parameter (m^3/s^2)
omega_E = 7.292115e-5;        % Earth rotation rate (rad/s)

inclination = deg2rad(inclination_deg);
phi_UE = deg2rad(phi_UE_deg);
lambda_UE = deg2rad(lambda_UE_deg);

r_orbit = a_wgs84 + simParameters.SatelliteAltitude;
omega_s = sqrt(mu / r_orbit^3);
v_sat_orbit = sqrt(mu / r_orbit);

N_phi_0 = a_wgs84 / sqrt(1.0 - e2 * sin(phi_UE)^2);
r_ue_ECEF_0 = [ ...
    (N_phi_0 + h_UE) * cos(phi_UE) * cos(lambda_UE); ...
    (N_phi_0 + h_UE) * cos(phi_UE) * sin(lambda_UE); ...
    (N_phi_0 * (1.0 - e2) + h_UE) * sin(phi_UE) ...
];

R_ENU2ECEF = [ ...
    -sin(lambda_UE), -sin(phi_UE)*cos(lambda_UE), cos(phi_UE)*cos(lambda_UE); ...
     cos(lambda_UE), -sin(phi_UE)*sin(lambda_UE), cos(phi_UE)*sin(lambda_UE); ...
     0,               cos(phi_UE),                sin(phi_UE) ...
];

if inclination >= abs(phi_UE)
    u_mid = asin(sin(phi_UE) / sin(inclination));
else
    u_mid = sign(phi_UE) * pi / 2.0;
end
Omega_RAAN = lambda_UE - atan2(sin(u_mid) * cos(inclination), cos(u_mid));

if simParameters.ElevationAngle < 89.9
    theta_target_rad = deg2rad(simParameters.ElevationAngle);
    gamma_central = pi/2.0 - theta_target_rad - asin((a_wgs84 / r_orbit) * cos(theta_target_rad));
    t_snapshot = gamma_central / omega_s;
else
    t_snapshot = 0.0;
end

[r_sat_ECEF, v_sat_ECEF] = get_satellite_state_ecef_local( ...
    t_snapshot, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E);

v_los_bc = r_sat_ECEF - r_ue_ECEF_0;
slant_range_bc = norm(v_los_bc);
u_los_bc = v_los_bc / slant_range_bc;
satelliteDopplerShift_bc = dot(v_sat_ECEF, u_los_bc) / lambda;

%% 3. Generate Randomized UE Positions & Doppler Shifts
ut_loc_ENU_all     = zeros(3, numUE);
r_ue_ECEF_all      = zeros(3, numUE);
slant_ranges       = zeros(1, numUE);
elevation_angles   = zeros(1, numUE);
pl_dB_all          = zeros(1, numUE);
doppler_shifts_all = zeros(1, numUE);

for i = 1:numUE
    theta_rand = 2.0 * pi * rand();
    r_rand = r_ue_max * sqrt(rand());
    
    ut_loc_ENU = [r_rand * cos(theta_rand); r_rand * sin(theta_rand); simParameters.MobileAltitude];
    ut_loc_ENU_all(:, i) = ut_loc_ENU;
    
    r_ue_ECEF_i = r_ue_ECEF_0 + R_ENU2ECEF * ut_loc_ENU;
    r_ue_ECEF_all(:, i) = r_ue_ECEF_i;
    
    v_los_i = r_sat_ECEF - r_ue_ECEF_i;
    d_i = norm(v_los_i);
    slant_ranges(i) = d_i;
    
    u_normal_i = r_ue_ECEF_i / norm(r_ue_ECEF_i);
    u_los_i = v_los_i / d_i;
    elev_rad_i = asin(dot(u_normal_i, u_los_i));
    elevation_angles(i) = rad2deg(elev_rad_i);
    
    pl_dB_all(i) = fspl(d_i, lambda);
    doppler_shifts_all(i) = dot(v_sat_ECEF, u_los_i) / lambda;
end

%% 4. Probing Waveform Setup
chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays * channel.SampleRate)) + chInfo.ChannelFilterDelay;
txGrid_ones = ones(carrier.NSizeGrid * 12, carrier.SymbolsPerSlot);
[txWaveform1, ~] = nrOFDMModulate(carrier, txGrid_ones);
txWaveform1 = [txWaveform1; zeros(maxChDelay, size(txWaveform1, 2))];

% Precompensate probing waveform with beam center Doppler shift
t_vec = (0:size(txWaveform1, 1)-1)' / channel.SampleRate;
txWaveform2 = txWaveform1 .* exp(1j * 2 * pi * (-satelliteDopplerShift_bc) * t_vec);

%% 5. Generate Target Domain Effective Channels
fprintf('Generating target domain effective channels for %d UEs...\n', numUE);
nSubcarriers = carrier.NSizeGrid * 12;
nSymbols = carrier.SymbolsPerSlot;

H_eff_all = zeros(nSubcarriers, nSymbols, numUE);
H_ori_all = zeros(nSubcarriers, nSymbols, numUE);

for idxUE = 1:numUE
    channel.SatelliteDopplerShift = doppler_shifts_all(idxUE);
    
    if channel.RandomStream == "Global stream"
        reset(channel);
    elseif channel.RandomStream == "mt19937ar with seed"
        release(channel);
        channel.Seed = idxUE +2;
    end
    
    [rxWaveform2, pathGains] = channel(txWaveform2);
    pathFilters = getPathFilters(channel);
    offset = nrPerfectTimingEstimate(pathGains, pathFilters);
    rxGrid2 = nrOFDMDemodulate(carrier, rxWaveform2(1+offset:end, :));
    hEstPerfect2 = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset);
    
    H_ori_all(:, :, idxUE) = hEstPerfect2 * db2mag(-pl_dB_all(idxUE));
    H_eff_all(:, :, idxUE) = rxGrid2 * db2mag(-pl_dB_all(idxUE));
end

%% 6. Plot All Generated Target Domain Effective Channels
fprintf('Plotting all %d effective channel grids...\n', numUE);

% Grid layout calculation for subplotting
nCols = min(3, numUE);
nRows = ceil(numUE / nCols);

% --- Figure 1: Real Part of Effective Channels ---
fig1 = figure('Name', 'Effective Channels (Real Part)', 'Position', [50, 50, 400*nCols, 320*nRows]);
tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');

for idxUE = 1:numUE
    nexttile;
    imagesc(real(H_eff_all(:, :, idxUE)));
    colorbar;
    set(gca, 'FontSize', 10);
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    title(sprintf('UE %d: Real(H_{eff})\nElev: %.1f°, Dop: %.0f Hz', ...
        idxUE, elevation_angles(idxUE), doppler_shifts_all(idxUE)), 'FontSize', 11);
end
sgtitle(sprintf('Target Domain Effective Channels (Real Part) - %s (%.0f ns, %.2f GHz)', ...
    channel.DelayProfile, channel.DelaySpread*1e9, simParameters.CarrierFrequency/1e9), 'FontSize', 14, 'FontWeight', 'bold');

% --- Figure 2: Magnitude (dB) of Effective Channels ---
fig2 = figure('Name', 'Effective Channels (Magnitude dB)', 'Position', [100, 100, 400*nCols, 320*nRows]);
tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');

for idxUE = 1:numUE
    nexttile;
    imagesc(20*log10(abs(H_eff_all(:, :, idxUE))));
    colorbar;
    set(gca, 'FontSize', 10);
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    title(sprintf('UE %d: |H_{eff}| (dB)\nElev: %.1f°, Range: %.1f km', ...
        idxUE, elevation_angles(idxUE), slant_ranges(idxUE)/1000), 'FontSize', 11);
end
sgtitle(sprintf('Target Domain Effective Channels (Magnitude dB) - %s (%.0f ns, %.2f GHz)', ...
    channel.DelayProfile, channel.DelaySpread*1e9, simParameters.CarrierFrequency/1e9), 'FontSize', 14, 'FontWeight', 'bold');

fprintf('Done! Figures displayed on screen.\n');

%% Helper Function
function [r_sat_ECEF, v_sat_ECEF] = get_satellite_state_ecef_local(t, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E)
    theta_G = omega_E * t;
    R_z = [ cos(theta_G), sin(theta_G), 0;
           -sin(theta_G), cos(theta_G), 0;
            0,            0,            1 ];
        
    u_t = omega_s * t + u_mid;
    r_sat_ECI = [ r_orbit * (cos(u_t)*cos(Omega_RAAN) - sin(u_t)*sin(Omega_RAAN)*cos(inclination)); ...
                  r_orbit * (cos(u_t)*sin(Omega_RAAN) + sin(u_t)*cos(Omega_RAAN)*cos(inclination)); ...
                  r_orbit * sin(u_t)*sin(inclination) ];
              
    v_sat_ECI = [ v_sat_orbit * (-sin(u_t)*cos(Omega_RAAN) - cos(u_t)*sin(Omega_RAAN)*cos(inclination)); ...
                  v_sat_orbit * (-sin(u_t)*sin(Omega_RAAN) + cos(u_t)*cos(Omega_RAAN)*cos(inclination)); ...
                  v_sat_orbit * cos(u_t)*sin(inclination) ];
              
    r_sat_ECEF = R_z * r_sat_ECI;
    
    omega_cross_r = [ -omega_E * r_sat_ECI(2); ...
                       omega_E * r_sat_ECI(1); ...
                       0 ];
                   
    v_sat_ECEF = R_z * (v_sat_ECI - omega_cross_r);
end
