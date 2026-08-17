%{
========================================================================================
                      NTN OFDM Channel Estimation & BER Comparison (Grid Input ONNX Models)
========================================================================================
OVERVIEW:
  This script evaluates and compares the Bit Error Rate (BER) and Normalized Mean
  Squared Error (NMSE) performance of four 5G NTN (Non-Terrestrial Networks) 
  OFDM channel estimation techniques:
    1. LS + Linear Interpolation (LS+LI): Benchmark 2D linear interpolation from pilots.
    2. Deep Learning Sparse LS (H_ls_cnn): Inferring a pretrained ONNX CNN model using sparse
       Least-Squares estimates at DM-RS pilot positions as input.
    3. Deep Learning LI (H_li_cnn): Inferring a pretrained ONNX CNN model using 2D linearly
       interpolated channel grid as input.
    4. LS + Perfect MMSE (MMSE Benchmark): Ideal Minimum Mean Squared Error estimator
       utilizing the true empirical channel covariance matrix across all UE samples.

SYSTEM GEOMETRY & SIMULATION FLOW:
  1. Satellite & Orbit Geometry Initialization:
     - Defines fixed LEO orbit parameters (600 km altitude, 55° inclination, Earth WGS84 model).
     - Sets fixed beam center location on Earth surface (Lat 37.7749°, Lon -122.4194°).
     - Positions satellite at snapshot time corresponding to target nominal elevation angle
       (70° in this setup) and computes satellite velocity vector v_sat_ECEF.
     - Computes nominal satellite-to-beam-center Doppler shift (satelliteDopplerShift_bc).

  2. UE Generation & Small-Scale Channel Formulation (Loop over numUE samples):
     - Randomly places UEs within 15 km beam footprint radius with velocity & height.
     - Computes exact Sat-UE Doppler shift (doppler_shifts_all) and Free-Space Path Loss (FSPL).
     - Applies 3GPP NTN-TDL-A channel model with fixed delay spread (100 ns).
     - Formulates the EFFECTIVE channel grid (H_perfect_n):
       * The transmitted OFDM probing waveform is precompensated using beam-center Doppler (-Doppler_bc).
       * The precompensated waveform passes through time-domain nrTDLChannel with exact UE Doppler.
       * Demodulated rxGrid yields effective channel grid H_perfect_n affecting original txGrid.
       * (Also stores uncompensated original channel H_perfect_ori_n for reference).
     - Calculates empirical channel covariance matrix R_hh across all UEs for the MMSE estimator.

  3. Transmission & Channel Estimation Loop (per UE):
     - Transmission Model:
       * Generates random PDSCH data bits, modulates to 16QAM, maps PDSCH & DM-RS pilots to txGrid.
       * Grid-level transmission: rxGrid = txGrid .* H_perfect_n + noise (frequency-domain simulation).
     - Channel Estimation:
       * Obtains sparse Least-Squares estimates at pilot locations: y_p = rxGrid(dmrs) ./ dmrsSymbols.
       * Approach A (LS+LI): 2D Linear interpolation across subcarriers and symbols.
       * Approach B (LS+MMSE): W_MMSE = R_h_hp * (R_hp_hp + noisePower*I)^(-1) using perfect R_hh covariance.
       * Approach C (H_ls_cnn): Feeds sparse LS pilot grid to pretrained ONNX CNN model.
       * Approach D (H_li_cnn): Feeds 2D linearly interpolated grid to pretrained ONNX CNN model.
     - Equalization & Demodulation:
       * Performs per-RE Zero-Forcing / MMSE equalization: rxData = rxGrid .* conj(H_est) / (|H_est|^2 + N0).
       * Demodulates 16QAM to LLRs, makes hard bit decisions, and calculates BER & NMSE per UE.

  4. Results & Reporting:
     - Summarizes mean BER and NMSE for all 4 estimation techniques.
     - Exports MAT performance data, complex 2D channel matrices, vector PDF heatmaps for Sample 1,
       network architecture visualization graphs, and markdown summary report in 'inference_result/' folder structure.
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

%% 1. Carrier & Simulation Parameters Setup
carrier = nrCarrierConfig;
channel = nrTDLChannel; % Small-scale fading channel

SNRdB = -5;                      % Signal-to-Noise Ratio (dB)
numUE = 10;                    % Number of UEs (samples) for simulation
model_checkpoint_choice = 'best'; % Choice of model checkpoint: 'best' (best_model.onnx) or 'final' (final_model.onnx)
r_beam = 15000.0;                % 15 km beam footprint radius
r_ue_max = 14500.0;              % 14.5 km max UE offset inside beam

simParameters.CarrierFrequency = 2.18e9;   % S-band (2.18 GHz)
simParameters.SatelliteAltitude = 600000;   % 600 km LEO orbit altitude
carrier.SubcarrierSpacing = 30;             % 30 kHz SCS
channel.DelaySpread = 100e-9;              % Delay spread

simParameters.ElevationAngle = 70;          % Chosen nominal elevation angle (in degrees)
simParameters.MobileSpeed = 30;             % Speed of mobile terminal (m/s)
simParameters.MobileAltitude = 1.5;         % Mobile antenna height (m)

carrier.NSizeGrid = 11;                     % Bandwidth in resource blocks (132 subcarriers)
carrier.CyclicPrefix = 'Normal';            % Normal cyclic prefix

channel.DelayProfile = 'NTN-TDL-A';

waveformInfo = nrOFDMInfo(carrier);
simParameters.SampleRate = waveformInfo.SampleRate;
c = physconst("lightspeed");
lambda = c / simParameters.CarrierFrequency;

%% 2. Satellite & Beam Center Geometry Setup
phi_UE_deg = 37.7749;         % Beam Center Latitude (degrees)
lambda_UE_deg = -122.4194;    % Beam Center Longitude (degrees)
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

%% 3. Generate Randomized UE Positions and UE-Sat Doppler
ut_loc_ENU_all     = zeros(3, numUE);
r_ue_ECEF_all      = zeros(3, numUE);
slant_ranges       = zeros(1, numUE);
elevation_angles   = zeros(1, numUE);
pl_dB_all          = zeros(1, numUE);
doppler_shifts_all = zeros(1, numUE);

for i = 1:numUE
    theta_rand = 2.0 * pi * rand();
    r_rand = r_ue_max * sqrt(rand());
    
    if i == 1
        while (r_rand < 5000.0)
            r_rand = r_ue_max * sqrt(rand());
        end
    end
    
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

%% 4. PDSCH and DM-RS Pilot Grid Settings
nTxAnts = 1;
nRxAnts = 1;
channel.NumTransmitAntennas = nTxAnts;
channel.NumReceiveAntennas = nRxAnts;
% Set MaximumDopplerShift to 0 so Doppler is governed purely by exact joint Sat-UE SatelliteDopplerShift
channel.MaximumDopplerShift = 0; 
channel.SampleRate = waveformInfo.SampleRate;
channel.RandomStream = "mt19937ar with seed";

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

[pdschIndices, pdschIndicesInfo] = nrPDSCHIndices(carrier, pdsch);
dmrsSymbols = nrPDSCHDMRS(carrier, pdsch);
dmrsIndices = nrPDSCHDMRSIndices(carrier, pdsch);
numPilots = length(dmrsIndices);

% Probing waveform for perfect channel reference grid generation
chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays * channel.SampleRate)) + chInfo.ChannelFilterDelay;
txGrid_ones = ones(carrier.NSizeGrid * 12, carrier.SymbolsPerSlot);
[txWaveform1, ~] = nrOFDMModulate(carrier, txGrid_ones);
txWaveform1 = [txWaveform1; zeros(maxChDelay, size(txWaveform1, 2))];

% Precompensate probing waveform using beam-center Doppler shift
t_vec = (0:size(txWaveform1, 1)-1)' / simParameters.SampleRate;
txWaveform2 = txWaveform1 .* exp(1j * 2 * pi * (-satelliteDopplerShift_bc) * t_vec);

%% 4b. Load Trained ONNX CNN Models for LS and LI (single_source_trained_model)
source_model_folder_name = 'Clipped_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps';
source_model_dir = fullfile(script_dir, 'single_source_trained_model', source_model_folder_name);

snr_str = num2str(SNRdB);

% Determine target ONNX filenames based on model_checkpoint_choice ('best' vs 'final')
if strcmpi(model_checkpoint_choice, 'final')
    target_onnx_names = {'final_model.onnx', 'final.onnx', 'final_net.onnx'};
else
    target_onnx_names = {'best_model.onnx', 'best.onnx', 'best_net.onnx'};
end

onnx_ls_path = find_onnx_model_path(source_model_dir, ['LS_', snr_str], target_onnx_names, '*ls*');
onnx_li_path = find_onnx_model_path(source_model_dir, ['LI_', snr_str], target_onnx_names, '*li*');

fprintf('Loading ONNX model checkpoint choice: "%s" for dataset "%s" (SNR = %d dB)\n', ...
    model_checkpoint_choice, source_model_folder_name, SNRdB);

net_ls = load_onnx_model_helper(onnx_ls_path);
net_li = load_onnx_model_helper(onnx_li_path);

%% 5. Generate Effective Channels for All UEs & Compute Correlation Matrix
fprintf('Generating effective channel grids for %d UEs...\n', numUE);
nSubcarriers = carrier.NSizeGrid * 12;
nSymbols = carrier.SymbolsPerSlot;
nREs = nSubcarriers * nSymbols;

H_perfect_all     = zeros(nSubcarriers, nSymbols, numUE);
H_perfect_ori_all = zeros(nSubcarriers, nSymbols, numUE);

for idxUE = 1:numUE
    channel.SatelliteDopplerShift = doppler_shifts_all(idxUE);
    
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
    
    H_perfect_ori_n = hEstPerfect2 * db2mag(-pl_dB_all(idxUE));
    H_perfect_n     = rxGrid2 * db2mag(-pl_dB_all(idxUE));
    
    H_perfect_all(:, :, idxUE)     = H_perfect_n;
    H_perfect_ori_all(:, :, idxUE) = H_perfect_ori_n;
end

% Compute perfect channel correlation matrix across UEs
h_vec_all = reshape(H_perfect_all, nREs, numUE); % [nREs x numUE]
R_hh = (h_vec_all * h_vec_all') / numUE;          % Full grid covariance matrix

R_h_hp  = R_hh(:, dmrsIndices);                  % [nREs x numPilots]
R_hp_hp = R_hh(dmrsIndices, dmrsIndices);        % [numPilots x numPilots]

%% 6. Data Transmission, Channel Estimation & BER Calculation Loop
ber_LI     = zeros(1, numUE);
ber_MMSE   = zeros(1, numUE);
ber_ls_cnn = zeros(1, numUE);
ber_li_cnn = zeros(1, numUE);

nmse_LI     = zeros(1, numUE);
nmse_MMSE   = zeros(1, numUE);
nmse_ls_cnn = zeros(1, numUE);
nmse_li_cnn = zeros(1, numUE);

SNR = 10^(SNRdB / 10);

fprintf('Running transmission, channel estimation (LS+LI, LS+MMSE, H_ls_cnn ONNX, H_li_cnn ONNX), and BER evaluation at SNR = %d dB...\n', SNRdB);

for idxUE = 1:numUE
    H_perfect_n = H_perfect_all(:, :, idxUE);
    
    % 1. Generate random tx bit sequence for PDSCH
    txBits = randi([0 1], pdschIndicesInfo.G, 1);
    
    % 2. Modulate PDSCH data bits
    pdschSymbols = nrPDSCH(carrier, pdsch, txBits);
    
    % 3. Map PDSCH data symbols and DM-RS pilot symbols to txGrid
    txGrid = nrResourceGrid(carrier, nTxAnts);
    txGrid(pdschIndices) = pdschSymbols;
    txGrid(dmrsIndices)  = dmrsSymbols;
    
    % 4. Transmit over effective Doppler-compensated channel & add noise
    rxGrid_clean = txGrid .* H_perfect_n;
    
    sigPower = mean(abs(rxGrid_clean(pdschIndices)).^2, 'all');
    noisePower = sigPower / SNR;
    noise = sqrt(noisePower / 2) * (randn(size(rxGrid_clean)) + 1j * randn(size(rxGrid_clean)));
    
    rxGrid = rxGrid_clean + noise;
    
    % ---------------------------------------------------------------------
    % Approach A: LS + Linear Interpolation (LS+LI Benchmark)
    % ---------------------------------------------------------------------
    [~, H_est_LI] = Lin_Interpolate(rxGrid, dmrsIndices, dmrsSymbols);
    
    rxDataSymbols_LI = rxGrid(pdschIndices) .* conj(H_est_LI(pdschIndices)) ./ ...
        (abs(H_est_LI(pdschIndices)).^2 + noisePower);
    rxLLRs_LI = nrSymbolDemodulate(rxDataSymbols_LI, pdsch.Modulation, noisePower);
    rxBits_LI = double(rxLLRs_LI < 0);
    ber_LI(idxUE)  = mean(txBits ~= rxBits_LI);
    nmse_LI(idxUE) = sum(abs(H_est_LI - H_perfect_n).^2, 'all') / sum(abs(H_perfect_n).^2, 'all');
    
    % ---------------------------------------------------------------------
    % Approach B: LS + MMSE Channel Estimation (MMSE Benchmark)
    % ---------------------------------------------------------------------
    y_p = rxGrid(dmrsIndices) ./ dmrsSymbols; % LS pilot estimates
    
    W_MMSE = R_h_hp / (R_hp_hp + noisePower * eye(numPilots));
    h_est_MMSE_vec = W_MMSE * y_p;
    H_est_MMSE = reshape(h_est_MMSE_vec, [nSubcarriers, nSymbols]);
    
    rxDataSymbols_MMSE = rxGrid(pdschIndices) .* conj(H_est_MMSE(pdschIndices)) ./ ...
        (abs(H_est_MMSE(pdschIndices)).^2 + noisePower);
    rxLLRs_MMSE = nrSymbolDemodulate(rxDataSymbols_MMSE, pdsch.Modulation, noisePower);
    rxBits_MMSE = double(rxLLRs_MMSE < 0);
    ber_MMSE(idxUE)  = mean(txBits ~= rxBits_MMSE);
    nmse_MMSE(idxUE) = sum(abs(H_est_MMSE - H_perfect_n).^2, 'all') / sum(abs(H_perfect_n).^2, 'all');
    
    % ---------------------------------------------------------------------
    % Approach C: CNN Model with Sparse LS Pilot Grid Input (H_ls_cnn)
    % ---------------------------------------------------------------------
    if ~isempty(net_ls)
        H_ls_grid = zeros(nSubcarriers, nSymbols);
        H_ls_grid(dmrsIndices) = y_p;
        
        H_est_ls_cnn = predict_cnn_channel(net_ls, H_ls_grid);
        rxDataSymbols_ls_cnn = rxGrid(pdschIndices) .* conj(H_est_ls_cnn(pdschIndices)) ./ ...
            (abs(H_est_ls_cnn(pdschIndices)).^2 + noisePower);
        rxLLRs_ls_cnn = nrSymbolDemodulate(rxDataSymbols_ls_cnn, pdsch.Modulation, noisePower);
        rxBits_ls_cnn = double(rxLLRs_ls_cnn < 0);
        ber_ls_cnn(idxUE)  = mean(txBits ~= rxBits_ls_cnn);
        nmse_ls_cnn(idxUE) = sum(abs(H_est_ls_cnn - H_perfect_n).^2, 'all') / sum(abs(H_perfect_n).^2, 'all');
        if idxUE == 1
            H_ls_sample1     = H_ls_grid;
            H_ls_cnn_sample1 = H_est_ls_cnn;
        end
    end

    % ---------------------------------------------------------------------
    % Approach D: CNN Model with Linear Interpolation Input (H_li_cnn)
    % ---------------------------------------------------------------------
    if ~isempty(net_li)
        H_est_li_cnn = predict_cnn_channel(net_li, H_est_LI);
        rxDataSymbols_li_cnn = rxGrid(pdschIndices) .* conj(H_est_li_cnn(pdschIndices)) ./ ...
            (abs(H_est_li_cnn(pdschIndices)).^2 + noisePower);
        rxLLRs_li_cnn = nrSymbolDemodulate(rxDataSymbols_li_cnn, pdsch.Modulation, noisePower);
        rxBits_li_cnn = double(rxLLRs_li_cnn < 0);
        ber_li_cnn(idxUE)  = mean(txBits ~= rxBits_li_cnn);
        nmse_li_cnn(idxUE) = sum(abs(H_est_li_cnn - H_perfect_n).^2, 'all') / sum(abs(H_perfect_n).^2, 'all');
        if idxUE == 1
            H_li_sample1     = H_est_LI;
            H_li_cnn_sample1 = H_est_li_cnn;
        end
    else
        if idxUE == 1
            H_li_sample1 = H_est_LI;
        end
    end
end

%% 7. Summary & Visualizations Organized in inference_result Folder
source_tag = 'DUR100ns_2p18G';

profile_str = erase(channel.DelayProfile, 'NTN-TDL-');
ds_str = [num2str(channel.DelaySpread * 1e9), 'ns'];
fc_str = [num2str(simParameters.CarrierFrequency / 1e9), 'G'];
fc_str = strrep(fc_str, '.', 'p');

target_tag = sprintf('%s%s_%s', profile_str, ds_str, fc_str);
domain_folder_name = sprintf('%s__%s', source_tag, target_tag);

inference_base_dir = fullfile(script_dir, 'inference_result');
domain_dir = fullfile(inference_base_dir, domain_folder_name);
if ~exist(domain_dir, 'dir')
    mkdir(domain_dir);
end

% 1. Copy source_dataset.md from source model folder
src_readme_path = fullfile(source_model_dir, 'readme_dur_randomizedUE.md');
dst_source_md   = fullfile(domain_dir, 'source_dataset.md');
if exist(src_readme_path, 'file')
    copyfile(src_readme_path, dst_source_md);
end

% 2. Create target_dataset.md noting current target system settings
dst_target_md = fullfile(domain_dir, 'target_dataset.md');
fid_target = fopen(dst_target_md, 'w');
if fid_target ~= -1
    fprintf(fid_target, '# Channel & Geometry Settings - Target System Domain\n\n');
    fprintf(fid_target, '## Channel Model & Propagation Settings\n');
    fprintf(fid_target, '- **Delay Profile:** %s\n', channel.DelayProfile);
    fprintf(fid_target, '- **Delay Spread:** %.0f ns (%.2e s)\n', channel.DelaySpread * 1e9, channel.DelaySpread);
    fprintf(fid_target, '- **Carrier Frequency ($f_c$):** %.2f GHz (%g Hz)\n', simParameters.CarrierFrequency / 1e9, simParameters.CarrierFrequency);
    fprintf(fid_target, '- **Maximum Mobile Doppler Shift:** %.2f Hz (Speed: %.1f m/s)\n\n', channel.MaximumDopplerShift, simParameters.MobileSpeed);
    
    fprintf(fid_target, '## Satellite & Orbit Settings\n');
    fprintf(fid_target, '- **Satellite Altitude:** %.0f km (%g m)\n', simParameters.SatelliteAltitude / 1000, simParameters.SatelliteAltitude);
    fprintf(fid_target, '- **Nominal Elevation Angle:** %g°\n', simParameters.ElevationAngle);
    fprintf(fid_target, '- **Beam Center Satellite Doppler Shift:** %.2f Hz\n\n', satelliteDopplerShift_bc);
    
    fprintf(fid_target, '## Beam & UE Footprint Settings\n');
    fprintf(fid_target, '- **Beam Radius:** %.1f km\n', r_beam / 1000);
    fprintf(fid_target, '- **Max UE Offset Radius:** %.1f km\n', r_ue_max / 1000);
    fprintf(fid_target, '- **Number of UEs ($N_{\\text{UE}}$):** %d\n\n', numUE);
    
    fprintf(fid_target, '## OFDM Carrier Configuration\n');
    fprintf(fid_target, '- **Resource Blocks ($N_{\\text{grid}}$):** %d RBs (%d Subcarriers)\n', carrier.NSizeGrid, carrier.NSizeGrid * 12);
    fprintf(fid_target, '- **Subcarrier Spacing (SCS):** %d kHz\n', carrier.SubcarrierSpacing);
    fprintf(fid_target, '- **Cyclic Prefix:** %s\n', carrier.CyclicPrefix);
    fprintf(fid_target, '- **PDSCH Modulation:** %s\n', string(pdsch.Modulation));
    fclose(fid_target);
end

% 3. Create SNR folder inside domain directory (e.g. SNR_-5)
snr_folder_name = sprintf('SNR_%d', SNRdB);
result_folder   = fullfile(domain_dir, source_model_folder_name, snr_folder_name);
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

fprintf('\n=========================================================\n');
fprintf('SIMULATION RESULTS (SNR = %d dB, Elevation = %d°, numUE = %d)\n', SNRdB, simParameters.ElevationAngle, numUE);
fprintf('=========================================================\n');
fprintf('1. LS + Linear Interpolation (LI):\n');
fprintf('   - Mean BER : %.6f\n', mean(ber_LI));
fprintf('   - Mean NMSE: %.2f dB (%.6f)\n', 10*log10(mean(nmse_LI)), mean(nmse_LI));
fprintf('2. LS + MMSE Benchmark (Perfect Channel Correlation):\n');
fprintf('   - Mean BER : %.6f\n', mean(ber_MMSE));
fprintf('   - Mean NMSE: %.2f dB (%.6f)\n', 10*log10(mean(nmse_MMSE)), mean(nmse_MMSE));

method_idx = 3;
if ~isempty(net_ls)
    fprintf('%d. H_ls_cnn (CNN with Sparse LS Input):\n', method_idx);
    fprintf('   - Mean BER : %.6f\n', mean(ber_ls_cnn));
    fprintf('   - Mean NMSE: %.2f dB (%.6f)\n', 10*log10(mean(nmse_ls_cnn)), mean(nmse_ls_cnn));
    method_idx = method_idx + 1;
end
if ~isempty(net_li)
    fprintf('%d. H_li_cnn (CNN with Linear Interpolation Input):\n', method_idx);
    fprintf('   - Mean BER : %.6f\n', mean(ber_li_cnn));
    fprintf('   - Mean NMSE: %.2f dB (%.6f)\n', 10*log10(mean(nmse_li_cnn)), mean(nmse_li_cnn));
end
fprintf('=========================================================\n');

% Save numerical performance MAT file inside result/ folder
mat_file = fullfile(result_folder, 'BER_performance_results.mat');
save_vars = {'ber_LI', 'ber_MMSE', 'nmse_LI', 'nmse_MMSE', 'SNRdB', 'numUE', 'simParameters', 'carrier', 'pdsch'};
if ~isempty(net_ls)
    save_vars = [save_vars, {'ber_ls_cnn', 'nmse_ls_cnn'}];
end
if ~isempty(net_li)
    save_vars = [save_vars, {'ber_li_cnn', 'nmse_li_cnn'}];
end
save(mat_file, save_vars{:});

%% Plot & Save Channel Grid Vector PDF Heatmaps for Sample 1
sample_idx = 1;

sample_mat_file = fullfile(result_folder, 'sample1_channel_grids.mat');
H_perfect_ori_sample1 = H_perfect_ori_all(:, :, sample_idx);
H_perfect_eff_sample1 = H_perfect_all(:, :, sample_idx);

sample_grid_vars = {'H_perfect_ori_sample1', 'H_perfect_eff_sample1', 'H_li_sample1', 'SNRdB', 'simParameters', 'carrier'};
if ~isempty(net_ls)
    sample_grid_vars = [sample_grid_vars, {'H_ls_sample1', 'H_ls_cnn_sample1'}];
end
if ~isempty(net_li)
    sample_grid_vars = [sample_grid_vars, {'H_li_cnn_sample1'}];
end
save(sample_mat_file, sample_grid_vars{:});
fprintf('Complex 2D channel grids for Sample 1 saved in: %s\n', sample_mat_file);

% 1. Perfect Original Channel PDF (Sample 1)
fig = figure('Visible', 'off', 'Position', [100, 100, 750, 600]);
imagesc(real(H_perfect_ori_all(:, :, sample_idx)));
colorbar; set(gca, 'FontSize', 16);
xlabel('Symbol', 'FontSize', 18); ylabel('Subcarrier', 'FontSize', 18);
title('Perfect Original Channel (Real Part) - Sample 1', 'FontSize', 16);
pdf_ori = fullfile(result_folder, 'H_perfect_ori_sample1.pdf');
try exportgraphics(fig, pdf_ori, 'ContentType', 'vector'); catch saveas(fig, pdf_ori); end
close(fig);

% 2. Effective Doppler-Compensated Channel PDF (Sample 1)
fig = figure('Visible', 'off', 'Position', [100, 100, 750, 600]);
imagesc(real(H_perfect_all(:, :, sample_idx)));
colorbar; set(gca, 'FontSize', 16);
xlabel('Symbol', 'FontSize', 18); ylabel('Subcarrier', 'FontSize', 18);
title('Effective Compensated Channel (Real Part) - Sample 1', 'FontSize', 16);
pdf_eff = fullfile(result_folder, 'H_perfect_eff_sample1.pdf');
try exportgraphics(fig, pdf_eff, 'ContentType', 'vector'); catch saveas(fig, pdf_eff); end
close(fig);

% 3. Sparse LS Estimated Channel PDF (Sample 1)
if ~isempty(net_ls) && exist('H_ls_sample1', 'var')
    fig = figure('Visible', 'off', 'Position', [100, 100, 750, 600]);
    imagesc(real(H_ls_sample1));
    colorbar; set(gca, 'FontSize', 16);
    xlabel('Symbol', 'FontSize', 18); ylabel('Subcarrier', 'FontSize', 18);
    title('Sparse LS Estimated Channel H_{ls} (Sample 1)', 'FontSize', 16);
    pdf_ls = fullfile(result_folder, 'H_ls_sample1.pdf');
    try exportgraphics(fig, pdf_ls, 'ContentType', 'vector'); catch saveas(fig, pdf_ls); end
    close(fig);
end

% 4. Linear Interpolation (LI) Estimated Channel PDF (Sample 1)
fig = figure('Visible', 'off', 'Position', [100, 100, 750, 600]);
imagesc(real(H_li_sample1));
colorbar; set(gca, 'FontSize', 16);
xlabel('Symbol', 'FontSize', 18); ylabel('Subcarrier', 'FontSize', 18);
title('Linear Interpolation Estimated Channel H_{li} (Sample 1)', 'FontSize', 16);
pdf_li = fullfile(result_folder, 'H_li_sample1.pdf');
try exportgraphics(fig, pdf_li, 'ContentType', 'vector'); catch saveas(fig, pdf_li); end
close(fig);

% 5. H_ls_cnn Model Estimated Channel PDF (Sample 1)
if ~isempty(net_ls) && exist('H_ls_cnn_sample1', 'var')
    fig = figure('Visible', 'off', 'Position', [100, 100, 750, 600]);
    imagesc(real(H_ls_cnn_sample1));
    colorbar; set(gca, 'FontSize', 16);
    xlabel('Symbol', 'FontSize', 18); ylabel('Subcarrier', 'FontSize', 18);
    title('H_{ls} CNN Model Estimated Channel (Sample 1)', 'FontSize', 16);
    pdf_ls_cnn = fullfile(result_folder, 'H_ls_cnn_sample1.pdf');
    try exportgraphics(fig, pdf_ls_cnn, 'ContentType', 'vector'); catch saveas(fig, pdf_ls_cnn); end
    close(fig);
end

% 6. H_li_cnn Model Estimated Channel PDF (Sample 1)
if ~isempty(net_li) && exist('H_li_cnn_sample1', 'var')
    fig = figure('Visible', 'off', 'Position', [100, 100, 750, 600]);
    imagesc(real(H_li_cnn_sample1));
    colorbar; set(gca, 'FontSize', 16);
    xlabel('Symbol', 'FontSize', 18); ylabel('Subcarrier', 'FontSize', 18);
    title('H_{li} CNN Model Estimated Channel (Sample 1)', 'FontSize', 16);
    pdf_li_cnn = fullfile(result_folder, 'H_li_cnn_sample1.pdf');
    try exportgraphics(fig, pdf_li_cnn, 'ContentType', 'vector'); catch saveas(fig, pdf_li_cnn); end
    close(fig);
end

% 7. Network Architecture Visualization Graphs (Interactive GUI + PDF/PNG Export)
if ~isempty(net_ls)
    visualize_network_model(net_ls, 'H_ls_cnn', result_folder);
end
if ~isempty(net_li)
    visualize_network_model(net_li, 'H_li_cnn', result_folder);
end

% Save Markdown report file inside result/ folder
md_path = fullfile(result_folder, 'simulation_results.md');
fid = fopen(md_path, 'w');
if fid ~= -1
    fprintf(fid, '# NTN Channel Estimation & BER Simulation Results (ONNX Grid Models)\n\n');
    fprintf(fid, '```text\n');
    fprintf(fid, '=========================================================\n');
    fprintf(fid, 'SIMULATION RESULTS (SNR = %d dB, Elevation = %d°, numUE = %d)\n', SNRdB, simParameters.ElevationAngle, numUE);
    fprintf(fid, '=========================================================\n');
    fprintf(fid, '1. LS + Linear Interpolation (LI):\n');
    fprintf(fid, '   - Mean BER : %.6f\n', mean(ber_LI));
    fprintf(fid, '   - Mean NMSE: %.2f dB (%.6f)\n', 10*log10(mean(nmse_LI)), mean(nmse_LI));
    fprintf(fid, '2. LS + MMSE Benchmark (Perfect Channel Correlation):\n');
    fprintf(fid, '   - Mean BER : %.6f\n', mean(ber_MMSE));
    fprintf(fid, '   - Mean NMSE: %.2f dB (%.6f)\n', 10*log10(mean(nmse_MMSE)), mean(nmse_MMSE));
    
    m_idx = 3;
    if ~isempty(net_ls)
        fprintf(fid, '%d. H_ls_cnn (CNN with Sparse LS Pilot Input):\n', m_idx);
        fprintf(fid, '   - Mean BER : %.6f\n', mean(ber_ls_cnn));
        fprintf(fid, '   - Mean NMSE: %.2f dB (%.6f)\n', 10*log10(mean(nmse_ls_cnn)), mean(nmse_ls_cnn));
        m_idx = m_idx + 1;
    end
    if ~isempty(net_li)
        fprintf(fid, '%d. H_li_cnn (CNN with Linear Interpolation Input):\n', m_idx);
        fprintf(fid, '   - Mean BER : %.6f\n', mean(ber_li_cnn));
        fprintf(fid, '   - Mean NMSE: %.2f dB (%.6f)\n', 10*log10(mean(nmse_li_cnn)), mean(nmse_li_cnn));
    end
    fprintf(fid, '=========================================================\n');
    fprintf(fid, '```\n\n');
    
    fprintf(fid, '## Simulation Configuration & System Parameters\n');
    fprintf(fid, '- **Signal-to-Noise Ratio (SNR):** %d dB\n', SNRdB);
    fprintf(fid, '- **Satellite Elevation Angle:** %d°\n', simParameters.ElevationAngle);
    fprintf(fid, '- **Number of UEs ($N_{\\text{UE}}$):** %d\n', numUE);
    fprintf(fid, '- **Carrier Frequency ($f_c$):** %.2f GHz\n', simParameters.CarrierFrequency / 1e9);
    fprintf(fid, '- **Satellite Altitude:** %.0f km\n', simParameters.SatelliteAltitude / 1000);
    fprintf(fid, '- **Subcarrier Spacing (SCS):** %d kHz\n', carrier.SubcarrierSpacing);
    fprintf(fid, '- **Resource Blocks ($N_{\\text{grid}}$):** %d RBs (%d Subcarriers)\n', carrier.NSizeGrid, carrier.NSizeGrid * 12);
    fprintf(fid, '- **Channel Model:** %s (Delay Spread: %.0f ns)\n', channel.DelayProfile, channel.DelaySpread * 1e9);
    fprintf(fid, '- **PDSCH Modulation:** %s\n', string(pdsch.Modulation));
    fprintf(fid, '- **Beam Center Satellite Doppler Shift:** %.2f Hz\n\n', satelliteDopplerShift_bc);
    
    fprintf(fid, '## Performance Comparison Summary\n');
    fprintf(fid, '| Estimation Approach | Mean BER | Mean NMSE (dB) | Mean NMSE (Linear) |\n');
    fprintf(fid, '| :--- | :---: | :---: | :---: |\n');
    fprintf(fid, '| **LS + Linear Interpolation (LI)** | `%.6f` | `%.2f dB` | `%.6f` |\n', mean(ber_LI), 10*log10(mean(nmse_LI)), mean(nmse_LI));
    fprintf(fid, '| **LS + MMSE Benchmark (Perfect Correlation)** | `%.6f` | `%.2f dB` | `%.6f` |\n', mean(ber_MMSE), 10*log10(mean(nmse_MMSE)), mean(nmse_MMSE));
    if ~isempty(net_ls)
        fprintf(fid, '| **H_ls_cnn (CNN with Sparse LS Input)** | `%.6f` | `%.2f dB` | `%.6f` |\n', mean(ber_ls_cnn), 10*log10(mean(nmse_ls_cnn)), mean(nmse_ls_cnn));
    end
    if ~isempty(net_li)
        fprintf(fid, '| **H_li_cnn (CNN with LI Input)** | `%.6f` | `%.2f dB` | `%.6f` |\n', mean(ber_li_cnn), 10*log10(mean(nmse_li_cnn)), mean(nmse_li_cnn));
    end
    
    fprintf(fid, '\n## Variables Saved in MAT File (`BER_performance_results.mat`)\n');
    fprintf(fid, '- `ber_LI`: `[1 x numUE]` double array — Bit Error Rate per UE sample using LS + Linear Interpolation.\n');
    fprintf(fid, '- `ber_MMSE`: `[1 x numUE]` double array — Bit Error Rate per UE sample using LS + MMSE Benchmark.\n');
    if ~isempty(net_ls)
        fprintf(fid, '- `ber_ls_cnn`: `[1 x numUE]` double array — Bit Error Rate per UE sample using CNN model with sparse LS input (`H_ls_cnn`).\n');
    end
    if ~isempty(net_li)
        fprintf(fid, '- `ber_li_cnn`: `[1 x numUE]` double array — Bit Error Rate per UE sample using CNN model with linear interpolation input (`H_li_cnn`).\n');
    end
    fprintf(fid, '- `nmse_LI`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for LS + Linear Interpolation.\n');
    fprintf(fid, '- `nmse_MMSE`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for LS + MMSE Benchmark.\n');
    if ~isempty(net_ls)
        fprintf(fid, '- `nmse_ls_cnn`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for `H_ls_cnn`.\n');
    end
    if ~isempty(net_li)
        fprintf(fid, '- `nmse_li_cnn`: `[1 x numUE]` double array — Normalized Mean Squared Error per UE sample (linear scale) for `H_li_cnn`.\n');
    end
    fprintf(fid, '- `SNRdB`: Scalar SNR value in dB.\n');
    fprintf(fid, '- `numUE`: Scalar number of evaluated UE samples.\n');
    fprintf(fid, '- `simParameters`: Struct containing satellite orbit, geometry, and system parameters.\n');
    fprintf(fid, '- `pdsch`: 5G NR PDSCH configuration object (`nrPDSCHConfig`).\n\n');

    fprintf(fid, '## Complex 2D Channel Grids MAT File (`sample1_channel_grids.mat`)\n');
    fprintf(fid, 'Contains full complex 2D channel matrices `[nSubcarriers x nSymbols]` for Sample 1:\n');
    fprintf(fid, '- `H_perfect_ori_sample1`: Complex 2D array — Original uncompensated channel.\n');
    fprintf(fid, '- `H_perfect_eff_sample1`: Complex 2D array — Effective Doppler-compensated channel.\n');
    if ~isempty(net_ls)
        fprintf(fid, '- `H_ls_sample1`: Complex 2D array — Sparse LS pilot estimated channel grid.\n');
    end
    fprintf(fid, '- `H_li_sample1`: Complex 2D array — Linear Interpolation estimated channel grid.\n');
    if ~isempty(net_ls)
        fprintf(fid, '- `H_ls_cnn_sample1`: Complex 2D array — `H_ls_cnn` model estimated channel grid.\n');
    end
    if ~isempty(net_li)
        fprintf(fid, '- `H_li_cnn_sample1`: Complex 2D array — `H_li_cnn` model estimated channel grid.\n');
    end
    fprintf(fid, '\n');

    fclose(fid);
end

fprintf('Results and Markdown report saved successfully in: %s\n', result_folder);

%% =========================================================================
%% HELPER FUNCTIONS FOR GEOMETRY, ONNX CNN INFERENCE & VISUALIZATION
%% =========================================================================
function visualize_network_model(net, model_name, save_dir)
% VISUALIZE_NETWORK_MODEL Visualizes and exports network architecture graph
%   - Opens MATLAB interactive Network Analyzer GUI (analyzeNetwork)
%   - Plots network DAG / LayerGraph graph in a MATLAB figure
%   - Exports high-resolution vector PDF and PNG images to save_dir

    if isempty(net)
        return;
    end

    try
        % 1. Interactive Network Analyzer GUI (when running in MATLAB desktop)
        if usejava('desktop')
            try
                analyzeNetwork(net);
                fprintf('Opened interactive Network Analyzer for %s.\n', model_name);
            catch ME_gui
                fprintf('Note: analyzeNetwork GUI not launched: %s\n', ME_gui.message);
            end
        end

        % 2. Plot Architecture DAG Graph in a Figure Window
        fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 900], 'Name', ['Architecture - ', model_name]);
        
        if isa(net, 'DAGNetwork') || isa(net, 'SeriesNetwork') || isa(net, 'dlnetwork')
            plot(net);
        elseif isprop(net, 'Layers') || isprop(net, 'Nodes')
            try
                plot(layerGraph(net));
            catch
                plot(net);
            end
        else
            close(fig);
            return;
        end
        
        title(sprintf('Model Architecture Graph: %s', strrep(model_name, '_', '\_')), 'FontSize', 16);
        grid on;
        
        % Export architecture plot image
        if nargin >= 3 && ~isempty(save_dir) && exist(save_dir, 'dir')
            pdf_path = fullfile(save_dir, sprintf('%s_architecture_graph.pdf', model_name));
            png_path = fullfile(save_dir, sprintf('%s_architecture_graph.png', model_name));
            try
                exportgraphics(fig, pdf_path, 'ContentType', 'vector');
                exportgraphics(fig, png_path, 'Resolution', 300);
                fprintf('Saved architecture visualization graph to: %s\n', pdf_path);
            catch
                try saveas(fig, png_path); catch; end
            end
        end
        close(fig);
    catch
    end
end

function H_pred = predict_cnn_channel(net, H_in)
    nSubc = size(H_in, 1); % 132
    nSymb = size(H_in, 2); % 14
    
    real_grid = single(real(H_in));
    imag_grid = single(imag(H_in));
    
    min_real = min(real_grid(:)); max_real = max(real_grid(:)); range_real = max(max_real - min_real, 1e-8);
    min_imag = min(imag_grid(:)); max_imag = max(imag_grid(:)); range_imag = max(max_imag - min_imag, 1e-8);
    
    real_scaled = 2 * (real_grid - min_real) / range_real - 1;
    imag_scaled = 2 * (imag_grid - min_imag) / range_imag - 1;
    
    try
        if isa(net, 'dlnetwork')
            if ~net.Initialized
                net = initialize_dlnetwork_if_needed(net);
            end
            
            % Check InputDataFormats if available or try BCSS / BSSC
            fmt = 'BCSS';
            if isprop(net, 'InputDataFormats') && ~isempty(net.InputDataFormats)
                fmt = net.InputDataFormats{1};
            end
            
            if strcmp(fmt, 'BSSC')
                x_input = zeros(1, nSubc, nSymb, 2, 'single');
                x_input(1, :, :, 1) = real_scaled;
                x_input(1, :, :, 2) = imag_scaled;
                dlX = dlarray(x_input, 'BSSC');
                dlOut = predict(net, dlX);
                out = extractdata(dlOut);
                if ndims(out) == 4
                    out_real = squeeze(out(1, :, :, 1));
                    out_imag = squeeze(out(1, :, :, 2));
                else
                    out_real = squeeze(out(:, :, 1));
                    out_imag = squeeze(out(:, :, 2));
                end
            else % Default BCSS: [Batch(1), Channel(2), Spatial1(132), Spatial2(14)]
                x_input = zeros(1, 2, nSubc, nSymb, 'single');
                x_input(1, 1, :, :) = real_scaled;
                x_input(1, 2, :, :) = imag_scaled;
                dlX = dlarray(x_input, 'BCSS');
                dlOut = predict(net, dlX);
                out = extractdata(dlOut);
                if ndims(out) == 4
                    out_real = squeeze(out(1, 1, :, :));
                    out_imag = squeeze(out(1, 2, :, :));
                else
                    out_real = squeeze(out(1, :, :));
                    out_imag = squeeze(out(2, :, :));
                end
            end
        else
            x_input = zeros(1, 2, nSubc, nSymb, 'single');
            x_input(1, 1, :, :) = real_scaled;
            x_input(1, 2, :, :) = imag_scaled;
            out = predict(net, x_input);
            out_real = squeeze(out(1, 1, :, :));
            out_imag = squeeze(out(1, 2, :, :));
        end
        
        if max(abs(out_real(:))) <= 2.0
            unscaled_real = (out_real + 1) / 2 * range_real + min_real;
            unscaled_imag = (out_imag + 1) / 2 * range_imag + min_imag;
        else
            unscaled_real = out_real;
            unscaled_imag = out_imag;
        end
        H_pred = double(unscaled_real + 1j * unscaled_imag);
    catch ME
        warning('Failed to infer CNN channel model: %s', ME.message);
        H_pred = double(H_in);
    end
end

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

function onnx_path = find_onnx_model_path(base_dir, snr_subfolder, target_names, search_keyword)
    onnx_path = '';
    results_dir = fullfile(base_dir, snr_subfolder, 'results');
    
    for k = 1:length(target_names)
        candidate = fullfile(results_dir, target_names{k});
        if exist(candidate, 'file')
            onnx_path = candidate;
            return;
        end
    end
    
    for k = 1:length(target_names)
        file_list = dir(fullfile(base_dir, '**', search_keyword, target_names{k}));
        if ~isempty(file_list)
            onnx_path = fullfile(file_list(1).folder, file_list(1).name);
            return;
        end
    end
    
    file_list = dir(fullfile(base_dir, '**', search_keyword, '*.onnx'));
    if ~isempty(file_list)
        onnx_path = fullfile(file_list(1).folder, file_list(1).name);
    end
end

function net = load_onnx_model_helper(onnx_path)
% LOAD_ONNX_MODEL_HELPER Imports an ONNX model directly into a MATLAB network object
    net = [];
    if isempty(onnx_path) || ~exist(onnx_path, 'file')
        if ~isempty(onnx_path)
            fprintf('ONNX model file not found at: %s\n', onnx_path);
        end
        return;
    end
    
    fprintf('Loading ONNX model: %s\n', onnx_path);
    [~, onnx_base_name, ~] = fileparts(onnx_path);
    clean_ns_name = ['onnx_pkg_', regexprep(onnx_base_name, '[^a-zA-Z0-9_]', '_')];
    
    % Valid MATLAB ONNX input formats for 2D spatial CNNs
    formats_to_try = {'BCSS', 'BSSC'};
    for f = 1:length(formats_to_try)
        fmt = formats_to_try{f};
        try
            net = importNetworkFromONNX(onnx_path, ...
                'InputDataFormats', fmt, ...
                'Namespace', clean_ns_name);
            net = initialize_dlnetwork_if_needed(net);
            if ~isempty(net) && isa(net, 'dlnetwork') && net.Initialized
                fprintf('  -> Successfully imported ONNX network via InputDataFormats: %s.\n', fmt);
                return;
            end
        catch
        end
    end
    
    % Fallback to automatic import if explicit formats fail
    try
        net = importNetworkFromONNX(onnx_path, ...
            'Namespace', clean_ns_name);
        net = initialize_dlnetwork_if_needed(net);
        if ~isempty(net) && isa(net, 'dlnetwork') && net.Initialized
            fprintf('  -> Successfully imported ONNX network via automatic shape detection.\n');
            return;
        end
    catch ME
        warning('Failed to import ONNX model via importNetworkFromONNX "%s": %s', onnx_path, ME.message);
    end
end

function net = initialize_dlnetwork_if_needed(net)
    if isa(net, 'dlnetwork') && ~net.Initialized
        dummy_cases = {
            'BCSS', [1, 2, 132, 14], 'BCSS';
            'BSSC', [1, 132, 14, 2], 'BSSC';
            'SSC',  [132, 14, 2],    'SSC';
        };
        
        for d = 1:size(dummy_cases, 1)
            fmt_str  = dummy_cases{d, 1};
            dummy_sz = dummy_cases{d, 2};
            dummy_fmt = dummy_cases{d, 3};
            try
                dummy_x = dlarray(zeros(dummy_sz, 'single'), dummy_fmt);
                
                lastwarn('');
                net_init = initialize(net, dummy_x);
                [warnMsg, ~] = lastwarn;
                
                % Ensure initialization produced no warnings and set Initialized to true
                if net_init.Initialized && isempty(warnMsg)
                    net = net_init;
                    return;
                end
            catch
            end
        end
        if ~net.Initialized
            error('Failed to initialize dlnetwork: network remains uninitialized or produced initialization warnings.');
        end
    end
end
