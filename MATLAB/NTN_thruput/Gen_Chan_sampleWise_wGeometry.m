cd(fileparts(matlab.desktop.editor.getActiveFilename))
addpath('..\helper\')

%% 1. Carrier and Simulation Setup
carrier = nrCarrierConfig;
channel = nrTDLChannel; % small-scale coefficient of the channel

% ========================================================
SNRdB = -5;
numUE = 1024;
r_beam = 15000.0;    % 15 km beam footprint radius
r_ue_max = 14500.0;  % 14.5 km max UE offset inside beam

% simParameters.CarrierFrequency = 20e9; % Ka band (20 GHz)                 % Carrier frequency (in Hz)
% simParameters.SatelliteAltitude = 1000000; % 1000 km altitude
% carrier.SubcarrierSpacing = 120;  % 15, 30, 60, 120, 240 (kHz)
% channel.DelaySpread = 100e-9;
simParameters.CarrierFrequency = 2.18e9; % S band                      % Carrier frequency (in Hz)
simParameters.SatelliteAltitude = 600000; % 600 km altitude
carrier.SubcarrierSpacing = 30;  % 15, 30, 60, 120, 240 (kHz)
channel.DelaySpread = 20e-9;

simParameters.ElevationAngle = 50; % Nominal elevation angle (in degrees)
simParameters.MobileSpeed = 30;    % Speed of mobile terminal (in m/s)
simParameters.MobileAltitude = 1.5; % Mobile antenna height above ground (in m)

carrier.NSizeGrid = 11;          % Bandwidth in resource blocks (132 subcarriers)
carrier.CyclicPrefix = 'Normal'; % 'Normal' or 'Extended'

channel.DelayProfile = 'NTN-TDL-A';

% ==========================================================

waveformInfo = nrOFDMInfo(carrier);
simParameters.SampleRate = waveformInfo.SampleRate;
c = physconst("lightspeed");
lambda = c / simParameters.CarrierFrequency;

%% 2. Geometry & Position Generation (UE & Satellite)
% Reference UE / Beam Center Geodetic Coordinates (San Francisco reference)
phi_UE_deg = 37.7749;         % Beam Center Latitude (degrees)
lambda_UE_deg = -122.4194;    % Beam Center Longitude (degrees)
h_UE = 100.0;                 % Ground altitude (m)
inclination_deg = 55.0;       % Orbit inclination (degrees)

% WGS-84 Ellipsoid & Gravitational Constants
a_wgs84 = 6378137.0;          % Earth semi-major axis (m)
e2 = 6.69437999e-3;           % First eccentricity squared
mu = 3.986004418e14;          % Gravitational parameter (m^3/s^2)
omega_E = 7.292115e-5;        % Earth rotation rate (rad/s)

% Convert to radians
inclination = deg2rad(inclination_deg);
phi_UE = deg2rad(phi_UE_deg);
lambda_UE = deg2rad(lambda_UE_deg);

% Satellite Orbit Parameters
r_orbit = a_wgs84 + simParameters.SatelliteAltitude;
omega_s = sqrt(mu / r_orbit^3);
v_sat_orbit = sqrt(mu / r_orbit);

% Reference UE / Beam Center ECEF Position
N_phi_0 = a_wgs84 / sqrt(1.0 - e2 * sin(phi_UE)^2);
r_ue_ECEF_0 = [ ...
    (N_phi_0 + h_UE) * cos(phi_UE) * cos(lambda_UE); ...
    (N_phi_0 + h_UE) * cos(phi_UE) * sin(lambda_UE); ...
    (N_phi_0 * (1.0 - e2) + h_UE) * sin(phi_UE) ...
];

% ENU to ECEF Rotation Matrix at Reference Location
R_ENU2ECEF = [ ...
    -sin(lambda_UE), -sin(phi_UE)*cos(lambda_UE), cos(phi_UE)*cos(lambda_UE); ...
     cos(lambda_UE), -sin(phi_UE)*sin(lambda_UE), cos(phi_UE)*sin(lambda_UE); ...
     0,               cos(phi_UE),                sin(phi_UE) ...
];
R_ECEF2ENU = R_ENU2ECEF';

% Orbit Alignment (Closest approach at t=0)
if inclination >= abs(phi_UE)
    u_mid = asin(sin(phi_UE) / sin(inclination));
else
    u_mid = sign(phi_UE) * pi / 2.0;
end
Omega_RAAN = lambda_UE - atan2(sin(u_mid) * cos(inclination), cos(u_mid));

% Calculate Snapshot Time t_snapshot for Target Nominal Elevation Angle
if simParameters.ElevationAngle < 89.9
    theta_target_rad = deg2rad(simParameters.ElevationAngle);
    gamma_central = pi/2.0 - theta_target_rad - asin((a_wgs84 / r_orbit) * cos(theta_target_rad));
    t_snapshot = gamma_central / omega_s;
else
    t_snapshot = 0.0;
end

% Compute Satellite State (Position & Velocity) in ECEF at snapshot time
[r_sat_ECEF, v_sat_ECEF] = get_satellite_state_ecef_local( ...
    t_snapshot, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E);

% Convert Satellite Position to Local ENU relative to Beam Center
bs_loc_ENU = R_ECEF2ENU * (r_sat_ECEF - r_ue_ECEF_0);
v_sat_ENU  = R_ECEF2ENU * v_sat_ECEF;

% Compute Nominal Slant Range & Doppler at Beam Center
v_los_bc = r_sat_ECEF - r_ue_ECEF_0;
slant_range_bc = norm(v_los_bc);
u_los_bc = v_los_bc / slant_range_bc;
satelliteDopplerShift_bc = dot(v_sat_ECEF, u_los_bc) / lambda;

%% 3. Generate Randomized UE Positions Inside Beam Footprint

rng(42); % Reproducible position generation
ut_loc_ENU_all     = zeros(3, numUE);
r_ue_ECEF_all      = zeros(3, numUE);
slant_ranges       = zeros(1, numUE);
elevation_angles   = zeros(1, numUE);
pl_dB_all          = zeros(1, numUE);
doppler_shifts_all = zeros(1, numUE);

for i = 1:numUE
    theta_rand = 2.0 * pi * rand();
    r_rand = r_ue_max * sqrt(rand());
    
    % Ensure 1st sample offset >= 5 km from beam center
    if i == 1
        while (r_rand < 5000.0)
            r_rand = r_ue_max * sqrt(rand());
        end
    end
    
    ut_loc_ENU = [r_rand * cos(theta_rand); r_rand * sin(theta_rand); simParameters.MobileAltitude];
    ut_loc_ENU_all(:, i) = ut_loc_ENU;
    
    % Convert UE ENU to ECEF
    r_ue_ECEF_i = r_ue_ECEF_0 + R_ENU2ECEF * ut_loc_ENU;
    r_ue_ECEF_all(:, i) = r_ue_ECEF_i;
    
    % Geometry calculations (LOS vector, Slant Range, Elevation Angle)
    v_los_i = r_sat_ECEF - r_ue_ECEF_i;
    d_i = norm(v_los_i);
    slant_ranges(i) = d_i;
    
    u_normal_i = r_ue_ECEF_i / norm(r_ue_ECEF_i);
    u_los_i = v_los_i / d_i;
    elev_rad_i = asin(dot(u_normal_i, u_los_i));
    elevation_angles(i) = rad2deg(elev_rad_i);
    
    % Free-Space Path Loss (FSPL) for UE i
    pl_dB_all(i) = fspl(d_i, lambda);
    
    % Satellite Doppler shift for UE i
    doppler_shifts_all(i) = dot(v_sat_ECEF, u_los_i) / lambda;
end

%% 4. Channel & Antenna Configuration
nTxAnts = 1;  % Number of transmit antennas
nRxAnts = 1;  % Number of receive antennas
nLayers = min(nTxAnts,nRxAnts);

channel.NumTransmitAntennas = nTxAnts;
channel.NumReceiveAntennas = nRxAnts;
channel.MaximumDopplerShift = simParameters.MobileSpeed * simParameters.CarrierFrequency / c;

ofdmInfo = nrOFDMInfo(carrier);
channel.SampleRate = ofdmInfo.SampleRate;
channel.RandomStream = "mt19937ar with seed";

%% 5. PDSCH Pilot Grid Settings
pdsch = nrPDSCHConfig;
pdsch.PRBSet = 0:carrier.NSizeGrid-1; 
pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot];
pdsch.MappingType = "A";
pdsch.NID = carrier.NCellID;
pdsch.RNTI = 1;
pdsch.VRBToPRBInterleaving = 0;
pdsch.VRBBundleSize = 4;
pdsch.NumLayers = 1;
pdsch.Modulation = "16QAM";

pdsch.DMRS.DMRSPortSet = [];
pdsch.DMRS.DMRSTypeAPosition = 2;
pdsch.DMRS.DMRSLength = 1;
pdsch.DMRS.DMRSAdditionalPosition = 1;
pdsch.DMRS.DMRSConfigurationType = 2;
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
pdsch.DMRS.NIDNSCID = 1;
pdsch.DMRS.NSCID = 0;

[refPDSCHIndices, refPDSCHIndicesInfo] = nrPDSCHIndices(carrier, pdsch);
refDMRSSymbols = nrPDSCHDMRS(carrier, pdsch);
refDMRSIndices = nrPDSCHDMRSIndices(carrier, pdsch);

pilot_indices = refDMRSIndices;
[pilot_rows, pilot_cols] = ind2sub([carrier.NSizeGrid*12, 14], refDMRSIndices);
numPilots = length(refDMRSIndices);

txGrid_pilot = zeros(carrier.NSizeGrid*12, 14);
txGrid_pilot(refDMRSIndices) = refDMRSSymbols;

% Probe grid for perfect channel reference
chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays * channel.SampleRate)) + chInfo.ChannelFilterDelay;
txGrid = nrResourceGrid(carrier, nTxAnts);
txGrid1 = ones(size(txGrid));
[txWaveform1, ~] = nrOFDMModulate(carrier, txGrid1);
txWaveform1 = [txWaveform1; zeros(maxChDelay, size(txWaveform1, 2))];

% Precompensate using beam-center satellite Doppler shift
t_vec = (0:size(txWaveform1,1)-1)' / simParameters.SampleRate;
txWaveform2 = txWaveform1 .* exp(1j * 2 * pi * (-satelliteDopplerShift_bc) * t_vec);

%% 6. Output Folder Definition
profile_str = erase(channel.DelayProfile, 'NTN-TDL-');
ds_str = num2str(channel.DelaySpread * 1e9);
fc_str = num2str(simParameters.CarrierFrequency / 1e9);
alt_str = [num2str(simParameters.SatelliteAltitude / 1000), 'km'];
scs_str = [num2str(carrier.SubcarrierSpacing), 'kHz'];

base_folder = sprintf('sampleWiseDoppler_wGeometry_%s%s_%se9_%s_%s', ...
    profile_str, ds_str, fc_str, alt_str, scs_str);


H_prac = zeros(14, carrier.NSizeGrid*12, numUE);
H_li = zeros(size(H_prac));
H_ls_pilots = zeros(numPilots, numUE);
H_perfect = zeros(size(H_prac));
H_perfect_ori = zeros(size(H_prac));

%% 7. Channel Simulation Loop Over UEs
for snr_idx = 1:length(SNRdB)
    nmse_prac     = 0;
    nmse_ls       = 0;
    nmse_ls_pilot = 0;
    nmse_li       = 0;
    
    for idxUE = 1:numUE
        % Apply UE-specific Doppler shift residual
        channel.SatelliteDopplerShift = satelliteDopplerShift_bc; 
        
        if channel.RandomStream == "Global stream"
            reset(channel);
        elseif channel.RandomStream == "mt19937ar with seed"
            release(channel);
            channel.Seed = idxUE;
        end
        
        [rxWaveform2, pathGains] = channel(txWaveform2);
        pathFilters = getPathFilters(channel);
        offset = nrPerfectTimingEstimate(pathGains, pathFilters);
        rxGrid2 = nrOFDMDemodulate(carrier, rxWaveform2(1+offset:end, :));
        hEstPerfect2 = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset);
        
        % Multiply normalized small-scale fading by position-based linear path loss factor
        pl_i_dB = pl_dB_all(idxUE);
        H_perfect_n_before = hEstPerfect2 * db2mag(-pl_i_dB);
        H_perfect_n        = rxGrid2 * db2mag(-pl_i_dB); % H_eff
        
        % Simplified transmission
        rxGrid_pilot = txGrid_pilot .* H_perfect_n;
        
        % Add Noise
        SNR = 10^(SNRdB(snr_idx) / 10);
        sigPower = mean(abs(rxGrid_pilot(:)).^2, 'all');
        noisePower = sigPower / SNR;
        noise = sqrt(noisePower / 2) * (randn(size(rxGrid_pilot)) + 1j * randn(size(rxGrid_pilot)));
        rxGrid_pilot_noisy = rxGrid_pilot + noise;
        
        % LS + Linear Interpolation
        [H_equalized_n, H_linear_n] = Lin_Interpolate(rxGrid_pilot_noisy, refDMRSIndices, refDMRSSymbols);
        [H_practical_n, noiseEst] = nrChannelEstimate(carrier, rxGrid_pilot_noisy, ...
                refDMRSIndices, refDMRSSymbols, 'CDMLengths', pdsch.DMRS.CDMLengths);
        
        % Store Channels
        H_prac(:,:,idxUE) = H_practical_n.';
        H_li(:,:,idxUE) = H_linear_n.';
        H_ls_pilots(:,idxUE) = H_equalized_n(refDMRSIndices);
        H_perfect(:,:,idxUE) = H_perfect_n.';
        H_perfect_ori(:,:,idxUE) = H_perfect_n_before.';
        
        % Full grid NMSEs
        nmse_prac_n = sum(abs(H_practical_n - H_perfect_n).^2, 'all') / sum(abs(H_perfect_n).^2, 'all');
        nmse_ls_n   = sum(abs(H_equalized_n - H_perfect_n).^2, 'all') / sum(abs(H_perfect_n).^2, 'all');
        nmse_li_n   = sum(abs(H_linear_n - H_perfect_n).^2, 'all') / sum(abs(H_perfect_n).^2, 'all');
        nmse_ls_pilot_n = sum(abs(H_equalized_n(refDMRSIndices) - H_perfect_n(refDMRSIndices)).^2, 'all') / ...
                          sum(abs(H_perfect_n(refDMRSIndices)).^2, 'all');
        
        nmse_prac     = nmse_prac + nmse_prac_n;
        nmse_ls       = nmse_ls + nmse_ls_n;
        nmse_ls_pilot = nmse_ls_pilot + nmse_ls_pilot_n;
        nmse_li       = nmse_li + nmse_li_n;
    end
    
    nmse_prac     = nmse_prac / numUE;
    nmse_ls       = nmse_ls / numUE;
    nmse_ls_pilot = nmse_ls_pilot / numUE;
    nmse_li       = nmse_li / numUE;

    save_folder = fullfile(base_folder, ['SNR_', num2str(SNRdB(snr_idx)), 'dB']);  
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    
    % Save .mat dataset including full geometry information
    save(fullfile(save_folder, 'matlabNTN.mat'), ...
        'H_perfect', ...
        'H_perfect_ori', ...
        'H_prac', ...
        'H_li', ...
        'H_ls_pilots', ...
        'pilot_indices', ...
        'pilot_rows', ...
        'pilot_cols', ...
        'nmse_prac', ...
        'nmse_ls', ...
        'nmse_ls_pilot', ...
        'nmse_li', ...
        'ut_loc_ENU_all', ...
        'r_ue_ECEF_all', ...
        'r_sat_ECEF', ...
        'v_sat_ECEF', ...
        'bs_loc_ENU', ...
        'v_sat_ENU', ...
        'slant_ranges', ...
        'elevation_angles', ...
        'pl_dB_all', ...
        'doppler_shifts_all', ...
        'satelliteDopplerShift_bc', ...
        '-v7.3');

    %% 8. Save Markdown Report with Parameters & Geometry Metadata
    md_path = fullfile(save_folder, 'simulation_parameters.md');
    fid = fopen(md_path, 'w');
    if fid ~= -1
        fprintf(fid, '# NTN Simulation Parameters (Position-Based Geometry)\n\n');
        
        fprintf(fid, '## Orbit & Satellite Parameters\n');
        fprintf(fid, '- **Carrier Frequency ($f_c$):** %.2f GHz (%g Hz)\n', simParameters.CarrierFrequency / 1e9, simParameters.CarrierFrequency);
        fprintf(fid, '- **Satellite Altitude:** %.2f km (%g m)\n', simParameters.SatelliteAltitude / 1000, simParameters.SatelliteAltitude);
        fprintf(fid, '- **Nominal Elevation Angle:** %.2f°\n', simParameters.ElevationAngle);
        fprintf(fid, '- **Beam Center Doppler Shift:** %.2f Hz\n', satelliteDopplerShift_bc);
        fprintf(fid, '- **Satellite Speed:** %.2f m/s\n\n', norm(v_sat_ECEF));
        
        fprintf(fid, '## Geometry & Beam Footprint\n');
        fprintf(fid, '- **Beam Footprint Radius:** %.2f km\n', r_beam / 1000);
        fprintf(fid, '- **Max UE Radius:** %.2f km\n', r_ue_max / 1000);
        fprintf(fid, '- **Beam Center Reference Location:** Lat %.4f°, Lon %.4f°\n', phi_UE_deg, lambda_UE_deg);
        fprintf(fid, '- **Elevation Angle Range:** Min %.2f°, Max %.2f°, Mean %.2f°\n', min(elevation_angles), max(elevation_angles), mean(elevation_angles));
        fprintf(fid, '- **Slant Range Range:** Min %.2f km, Max %.2f km\n', min(slant_ranges)/1000, max(slant_ranges)/1000);
        fprintf(fid, '- **Path Loss Range:** Min %.2f dB, Max %.2f dB\n\n', min(pl_dB_all), max(pl_dB_all));
        
        fprintf(fid, '## Mobile Terminal (UE) Parameters\n');
        fprintf(fid, '- **Mobile Speed:** %.2f m/s (%.2f km/h)\n', simParameters.MobileSpeed, simParameters.MobileSpeed * 3.6);
        fprintf(fid, '- **Mobile Antenna Height:** %.2f m\n', simParameters.MobileAltitude);
        fprintf(fid, '- **Number of UEs ($N_{\\text{UE}}$):** %d\n\n', numUE);
        
        fprintf(fid, '## 5G NR Carrier & OFDM Grid\n');
        fprintf(fid, '- **Resource Blocks ($N_{\\text{grid}}$):** %d RBs (%d Subcarriers)\n', carrier.NSizeGrid, carrier.NSizeGrid * 12);
        fprintf(fid, '- **Subcarrier Spacing (SCS):** %d kHz\n', carrier.SubcarrierSpacing);
        fprintf(fid, '- **Cyclic Prefix:** %s\n', carrier.CyclicPrefix);
        fprintf(fid, '- **Sample Rate:** %.2f MHz (%g Hz)\n\n', simParameters.SampleRate / 1e6, simParameters.SampleRate);
        
        fprintf(fid, '## Channel Model & Propagation\n');
        fprintf(fid, '- **Delay Profile:** %s\n', channel.DelayProfile);
        fprintf(fid, '- **Delay Spread:** %.2f ns (%g s)\n', channel.DelaySpread * 1e9, channel.DelaySpread);
        fprintf(fid, '- **Max Mobile Doppler Shift:** %.2f Hz\n', channel.MaximumDopplerShift);
        fprintf(fid, '- **Antenna Configuration:** %dx%d (Tx x Rx)\n\n', nTxAnts, nRxAnts);
        
        fprintf(fid, '## Signal & SNR Configuration\n');
        fprintf(fid, '- **SNR:** %g dB\n', SNRdB(snr_idx));
        fprintf(fid, '- **PDSCH Modulation:** %s\n', string(pdsch.Modulation));
        fprintf(fid, '- **Pilot Symbols:** DM-RS (Type %d, Position %d)\n', pdsch.DMRS.DMRSConfigurationType, pdsch.DMRS.DMRSTypeAPosition);
        fclose(fid);
    end

    %% 9. Visualization PDF Figures
    H_ls_grid_1 = zeros(carrier.NSizeGrid*12, 14);
    H_ls_grid_1(refDMRSIndices) = H_ls_pilots(:, 1);

    ch_struct = struct( ...
        'H_perfect_ori', H_perfect_ori(:,:,1).', ...
        'H_perfect',     H_perfect(:,:,1).', ...
        'H_prac',        H_prac(:,:,1).', ...
        'H_li',          H_li(:,:,1).', ...
        'H_ls',          H_ls_grid_1.' ...
    );
    
    ch_names = fieldnames(ch_struct);
    for k = 1:length(ch_names)
        ch_name = ch_names{k};
        fig = figure('Visible', 'off');
        imagesc(real(ch_struct.(ch_name)));
        colorbar;
        set(gca, 'FontSize', 20);
        xlabel('Symbol', 'FontSize', 25);
        ylabel('Subcarrier', 'FontSize', 25);
        
        pdf_path = fullfile(save_folder, [ch_name, '.pdf']);
        try
            exportgraphics(fig, pdf_path, 'ContentType', 'vector');
        catch
            saveas(fig, pdf_path);
        end

        if strcmp(ch_name, 'H_perfect_ori')
            alias_path = fullfile(save_folder, 'before.pdf');
            try
                exportgraphics(fig, alias_path, 'ContentType', 'vector');
            catch
                saveas(fig, alias_path);
            end
        elseif strcmp(ch_name, 'H_perfect')
            alias_path = fullfile(save_folder, 'after.pdf');
            try
                exportgraphics(fig, alias_path, 'ContentType', 'vector');
            catch
                saveas(fig, alias_path);
            end
        end
        close(fig);
    end
end

%% Helper Function: Compute Satellite ECEF Position & Velocity
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
