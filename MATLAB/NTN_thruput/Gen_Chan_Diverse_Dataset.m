function Gen_Chan_Diverse_Dataset(varargin)
% GEN_CHAN_DIVERSE_DATASET Generates 20,000 diverse NTN channel samples (.npy)
%
% Parameter distributions are physically coupled according to 3GPP TR 38.811:
%   - Elevation Angle: 30° to 80°
%   - Channel Profile: NTN-TDL-A, B, C, D (LOS probability scales with elevation)
%   - Delay Spread: 30 ns to 500 ns (LOS has lower delay spread 30-150ns, NLOS has 80-500ns)
%   - Rician K-factor: 4 dB to 18 dB for LOS profiles (increases with elevation)
%   - UE Velocity: 10 m/s to 30 m/s (36 to 108 km/h)
%   - UE Position: Randomized across 15 km beam footprint
%
% Output files in generatedChannel_Results/Diverse_2p18e9_600km_30to80deg_30kHz/:
%   - H_train.npy (80% = 16,000 samples)
%   - H_val.npy   (10% = 2,000 samples)
%   - H_test.npy  (10% = 2,000 samples)
%   - note.md (complete documentation)
%   - sample_metadata.mat (per-sample ground-truth parameters)

p = inputParser;
addParameter(p, 'NumUE', 20000, @isnumeric);
addParameter(p, 'CarrierFrequency', 2.18e9, @isnumeric);
addParameter(p, 'SatelliteAltitude', 600000, @isnumeric);
addParameter(p, 'SubcarrierSpacing', 30, @isnumeric);
addParameter(p, 'FolderName', 'Diverse_2p18e9_600km_30to80deg_30kHz', @ischar);
parse(p, varargin{:});
opt = p.Results;

numUE = opt.NumUE;
fc = opt.CarrierFrequency;
alt = opt.SatelliteAltitude;
scs = opt.SubcarrierSpacing;

if exist('mfilename', 'builtin') && ~isempty(mfilename('fullpath'))
    script_dir = fileparts(mfilename('fullpath'));
    if ~isempty(script_dir) && exist(script_dir, 'dir')
        cd(script_dir);
    end
end
addpath('..\helper\');

base_folder = fullfile('generatedChannel_Results', opt.FolderName);
if ~exist(base_folder, 'dir')
    mkdir(base_folder);
end

fprintf('====================================================================\n');
fprintf('  Generating Diverse NTN Channel Dataset (%d samples)\n', numUE);
fprintf('  Target Directory: %s\n', base_folder);
fprintf('====================================================================\n\n');

%% 1. 5G NR Carrier Configuration
carrier = nrCarrierConfig;
carrier.NSizeGrid = 11;          % 11 RBs (132 subcarriers)
carrier.SubcarrierSpacing = scs; % 30 kHz
carrier.CyclicPrefix = 'Normal';

waveformInfo = nrOFDMInfo(carrier);
sampleRate = waveformInfo.SampleRate;
c = physconst('lightspeed');
lambda = c / fc;

% Reference probe grid for perfect channel extraction
txGrid = nrResourceGrid(carrier, 1);
txGrid1 = ones(size(txGrid));
[txWaveform1, ~] = nrOFDMModulate(carrier, txGrid1);

% Pad waveform to absorb delay spread & filter transients
pad_len = 128;
txWaveform1 = [txWaveform1; zeros(pad_len, size(txWaveform1, 2))];
t_vec = (0:size(txWaveform1, 1)-1)' / sampleRate;

%% 2. Satellite Orbital Geometry Parameters
phi_UE_deg = 37.7749;       % San Francisco reference latitude
lambda_UE_deg = -122.4194;  % Longitude
h_UE = 100.0;               % Ground elevation (m)
inclination_deg = 55.0;     % LEO inclination
mobileAltitude = 1.5;       % UE antenna height (m)
r_ue_max = 14500.0;         % 14.5 km beam radius

a_wgs84 = 6378137.0;
e2 = 6.69437999e-3;
mu = 3.986004418e14;
omega_E = 7.292115e-5;

inclination = deg2rad(inclination_deg);
phi_UE = deg2rad(phi_UE_deg);
lambda_UE = deg2rad(lambda_UE_deg);

r_orbit = a_wgs84 + alt;
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

%% 3. Channel Model Initialization
warning('off', 'MATLAB:system:nonRelevantProperty');
channel = nrTDLChannel;
channel.NumTransmitAntennas = 1;
channel.NumReceiveAntennas = 1;
channel.SampleRate = sampleRate;
channel.RandomStream = 'mt19937ar with seed';

%% 4. Preallocate Metadata & Output Arrays
H_perfect = zeros(14, carrier.NSizeGrid * 12, numUE, 'like', 1j);

meta_elev_deg  = zeros(numUE, 1);
meta_v_ue      = zeros(numUE, 1);
meta_ds_ns     = zeros(numUE, 1);
meta_profile   = cell(numUE, 1);
meta_is_los    = false(numUE, 1);
meta_k_factor  = zeros(numUE, 1);
meta_doppler   = zeros(numUE, 1);
meta_pathloss  = zeros(numUE, 1);

rng(42, 'twister'); % Seed for reproducible diverse parameter generation

%% 5. Sample Generation Loop
t_start = tic;
print_step = max(500, floor(numUE / 20));

for idxUE = 1:numUE
    % -------------------------------------------------------------
    % A. Sample Elevation Angle (30° to 80°)
    % -------------------------------------------------------------
    elev_deg = 30.0 + 50.0 * rand();
    meta_elev_deg(idxUE) = elev_deg;
    
    % -------------------------------------------------------------
    % B. Physically Reasonable LOS vs NLOS Probability (3GPP TR 38.811)
    % At 30°: ~30% LOS, 70% NLOS; At 80°: ~85% LOS, 15% NLOS
    % -------------------------------------------------------------
    p_los = 0.30 + 0.55 * ((elev_deg - 30.0) / 50.0);
    is_los = (rand() < p_los);
    meta_is_los(idxUE) = is_los;
    
    if is_los
        % LOS Profiles: predominantly NTN-TDL-C (80%) or NTN-TDL-D (20%)
        if rand() < 0.80
            prof = 'NTN-TDL-C';
        else
            prof = 'NTN-TDL-D';
        end
        % LOS Delay Spread: 30 ns to 150 ns (decreases as elevation increases)
        base_ds = 30e-9 + (80.0 - elev_deg) / 50.0 * 80e-9; % 30 ns (at 80°) to 110 ns (at 30°)
        ds = max(30e-9, min(150e-9, base_ds + (rand() - 0.5) * 40e-9));
        
        % Rician K-factor: 4 dB to 18 dB (higher elevation -> stronger direct path)
        base_k = 5.0 + 10.0 * ((elev_deg - 30.0) / 50.0);
        k_dB = max(4.0, min(18.0, base_k + (rand() - 0.5) * 4.0));
    else
        % NLOS Profiles: predominantly NTN-TDL-A (80%) or NTN-TDL-B (20%)
        if rand() < 0.80
            prof = 'NTN-TDL-A';
        else
            prof = 'NTN-TDL-B';
        end
        % NLOS Delay Spread: 80 ns to 500 ns (higher at lower elevation)
        base_ds = 100e-9 + (80.0 - elev_deg) / 50.0 * 350e-9; % 100 ns (at 80°) to 450 ns (at 30°)
        ds = max(80e-9, min(500e-9, base_ds + (rand() - 0.5) * 80e-9));
        k_dB = -Inf; % Rayleigh
    end
    
    meta_profile{idxUE} = prof;
    meta_ds_ns(idxUE) = ds * 1e9;
    meta_k_factor(idxUE) = k_dB;
    
    % -------------------------------------------------------------
    % C. Sample UE Velocity (10 to 30 m/s = 36 to 108 km/h)
    % -------------------------------------------------------------
    v_ue = 10.0 + 20.0 * rand();
    meta_v_ue(idxUE) = v_ue;
    max_doppler_ue = v_ue * fc / c;
    
    % -------------------------------------------------------------
    % D. Geometry & Orbit Calculation for Sample Elevation
    % -------------------------------------------------------------
    if elev_deg < 89.9
        theta_target_rad = deg2rad(elev_deg);
        gamma_central = pi/2.0 - theta_target_rad - asin((a_wgs84 / r_orbit) * cos(theta_target_rad));
        t_snapshot = gamma_central / omega_s;
    else
        t_snapshot = 0.0;
    end
    
    [r_sat_ECEF, v_sat_ECEF] = get_satellite_state_ecef_local( ...
        t_snapshot, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E);
    
    % Beam center slant range and Doppler shift
    v_los_bc = r_sat_ECEF - r_ue_ECEF_0;
    u_los_bc = v_los_bc / norm(v_los_bc);
    sat_doppler_bc = dot(v_sat_ECEF, u_los_bc) / lambda;
    
    % Individual UE position inside 15 km footprint
    theta_rand = 2.0 * pi * rand();
    r_rand = r_ue_max * sqrt(rand());
    ut_loc_ENU = [r_rand * cos(theta_rand); r_rand * sin(theta_rand); mobileAltitude];
    r_ue_ECEF_i = r_ue_ECEF_0 + R_ENU2ECEF * ut_loc_ENU;
    
    v_los_i = r_sat_ECEF - r_ue_ECEF_i;
    d_i = norm(v_los_i);
    u_los_i = v_los_i / d_i;
    pl_i_dB = fspl(d_i, lambda);
    meta_pathloss(idxUE) = pl_i_dB;
    
    % Satellite Doppler shift for this UE
    sat_doppler_ue = dot(v_sat_ECEF, u_los_i) / lambda;
    meta_doppler(idxUE) = sat_doppler_ue;
    
    % -------------------------------------------------------------
    % E. Configure nrTDLChannel
    % -------------------------------------------------------------
    release(channel);
    channel.DelayProfile = prof;
    channel.DelaySpread = ds;
    channel.MaximumDopplerShift = max_doppler_ue;
    channel.SatelliteDopplerShift = sat_doppler_bc;
    channel.Seed = idxUE + 20 * numUE;
    
    if is_los
        channel.KFactorScaling = true;
        channel.KFactor = k_dB;
    else
        channel.KFactorScaling = false;
    end
    
    % Pre-compensate beam-center Doppler on transmitted waveform
    txWaveform2 = txWaveform1 .* exp(1j * 2 * pi * (-sat_doppler_bc) * t_vec);
    
    % -------------------------------------------------------------
    % F. Filter Waveform & Demodulate
    % -------------------------------------------------------------
    [rxWaveform2, pathGains] = channel(txWaveform2);
    pathFilters = getPathFilters(channel);
    offset = nrPerfectTimingEstimate(pathGains, pathFilters);
    rxGrid2 = nrOFDMDemodulate(carrier, rxWaveform2(1+offset:end, :));
    
    % Apply linear path loss
    H_perfect_n = rxGrid2 * db2mag(-pl_i_dB);
    H_perfect(:,:,idxUE) = H_perfect_n.';
    
    % Progress display
    if mod(idxUE, print_step) == 0 || idxUE == numUE
        elapsed = toc(t_start);
        rate = idxUE / elapsed;
        eta_sec = (numUE - idxUE) / max(1e-3, rate);
        fprintf('  [%5d / %5d] (%5.1f%%) | Elev: %4.1f° | %-9s | DS: %5.1f ns | v: %4.1f m/s | ETA: %.1f min\n', ...
            idxUE, numUE, idxUE/numUE*100, elev_deg, prof, ds*1e9, v_ue, eta_sec/60);
    end
end

elapsed_total = toc(t_start);
fprintf('\n>>> Channel generation completed in %.2f min (%.2f ms/sample)\n', ...
    elapsed_total / 60, (elapsed_total / numUE) * 1000);

%% 6. Save NPY Dataset Splits (Train 80%, Val 10%, Test 10%)
fprintf('\n--- Saving .npy dataset splits (H_train, H_val, H_test) ---\n');

n_train = floor(0.80 * numUE);
n_val   = floor(0.10 * numUE);

H_train = H_perfect(:, :, 1:n_train);
H_val   = H_perfect(:, :, n_train+1:n_train+n_val);
H_test  = H_perfect(:, :, n_train+n_val+1:end);

saveNPY_channel(fullfile(base_folder, 'H_train.npy'), H_train, false);
saveNPY_channel(fullfile(base_folder, 'H_val.npy'), H_val, false);
saveNPY_channel(fullfile(base_folder, 'H_test.npy'), H_test, false);

%% 7. Save Sample Metadata (.mat) for Full Tracking
save(fullfile(base_folder, 'sample_metadata.mat'), ...
    'meta_elev_deg', 'meta_v_ue', 'meta_ds_ns', 'meta_profile', ...
    'meta_is_los', 'meta_k_factor', 'meta_doppler', 'meta_pathloss', ...
    '-v7.3');
fprintf('Saved sample metadata: %s\n', fullfile(base_folder, 'sample_metadata.mat'));

%% 8. Save Comprehensive note.md
save_diverse_note_md(base_folder, opt, meta_elev_deg, meta_v_ue, meta_ds_ns, ...
    meta_profile, meta_is_los, meta_k_factor, size(H_train, 3), size(H_val, 3), size(H_test, 3));

fprintf('\n====================================================================\n');
fprintf('  Diverse Dataset successfully created in:\n  %s\n', base_folder);
fprintf('====================================================================\n');
end

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

function [r_sat_ECEF, v_sat_ECEF] = get_satellite_state_ecef_local(t, omega_s, u_mid, Omega_RAAN, inclination, r_orbit, v_sat_orbit, omega_E)
    theta_G = omega_E * t;
    u_t = omega_s * t + u_mid;
    
    r_sat_ECI = [ r_orbit * (cos(u_t)*cos(Omega_RAAN) - sin(u_t)*sin(Omega_RAAN)*cos(inclination)); ...
                  r_orbit * (cos(u_t)*sin(Omega_RAAN) + sin(u_t)*cos(Omega_RAAN)*cos(inclination)); ...
                  r_orbit * sin(u_t)*sin(inclination) ];
              
    v_sat_ECI = [ v_sat_orbit * (-sin(u_t)*cos(Omega_RAAN) - cos(u_t)*sin(Omega_RAAN)*cos(inclination)); ...
                  v_sat_orbit * (-sin(u_t)*sin(Omega_RAAN) + cos(u_t)*cos(Omega_RAAN)*cos(inclination)); ...
                  v_sat_orbit * cos(u_t)*sin(inclination) ];

    R_z = [ cos(theta_G), sin(theta_G), 0;
           -sin(theta_G), cos(theta_G), 0;
            0,            0,            1 ];
        
    r_sat_ECEF = R_z * r_sat_ECI;
    
    omega_vec = [0; 0; omega_E];
    v_sat_ECEF = R_z * (v_sat_ECI - cross(omega_vec, r_sat_ECI));
end

function saveNPY_channel(filename, H, normalize_rms)
    if nargin < 3
        normalize_rms = false;
    end

    sz = size(H);
    if numel(sz) == 2
        n_sym = sz(1);
        n_sc = sz(2);
        n_samples = 1;
    else
        n_sym = sz(1);
        n_sc = sz(2);
        n_samples = sz(3);
    end

    if normalize_rms
        for i = 1:n_samples
            slot_pwr = mean(abs(H(:,:,i)).^2, 'all');
            if slot_pwr > 0
                H(:,:,i) = H(:,:,i) / sqrt(slot_pwr);
            end
        end
    end

    header_dict = sprintf('{''descr'': ''<c8'', ''fortran_order'': False, ''shape'': (%d, %d, %d), }', ...
        n_samples, n_sc, n_sym);

    prefix_len = 10;
    target_total_header = 128;
    while target_total_header < prefix_len + length(header_dict) + 1
        target_total_header = target_total_header + 64;
    end

    pad_len = target_total_header - prefix_len - length(header_dict) - 1;
    header_str = [header_dict, repmat(' ', 1, pad_len), newline];
    header_len_uint16 = uint16(length(header_str));

    num_elem = numel(H);
    interleaved = zeros(2, num_elem, 'single');
    interleaved(1, :) = single(real(H(:)));
    interleaved(2, :) = single(imag(H(:)));

    folder_out = fileparts(filename);
    if ~isempty(folder_out) && ~exist(folder_out, 'dir')
        mkdir(folder_out);
    end

    fid = fopen(filename, 'wb');
    if fid == -1
        error('Could not open file for writing: %s', filename);
    end

    fwrite(fid, uint8([147, 'NUMPY']), 'uint8');
    fwrite(fid, uint8([1, 0]), 'uint8');
    fwrite(fid, header_len_uint16, 'uint16');
    fwrite(fid, uint8(header_str), 'uint8');
    fwrite(fid, interleaved, 'single');
    fclose(fid);
    fprintf('Saved NPY channel: %s [shape=(%d, %d, %d), complex64]\n', ...
        filename, n_samples, n_sc, n_sym);
end

function save_diverse_note_md(folder, opt, elev, v_ue, ds, profs, is_los, k_fac, n_tr, n_va, n_te)
    md_file = fullfile(folder, 'note.md');
    fid = fopen(md_file, 'w');
    if fid == -1
        return;
    end
    
    total_samples = n_tr + n_va + n_te;
    los_pct = mean(is_los) * 100;
    nlos_pct = 100 - los_pct;
    
    fprintf(fid, '# Dataset Overview: Diverse NTN Channel (%.2f GHz, %d km, 30°–80°, 30 kHz)\n\n', ...
        opt.CarrierFrequency / 1e9, round(opt.SatelliteAltitude / 1000));
    
    fprintf(fid, '## General Information\n\n');
    fprintf(fid, '- **Dataset Type:** Physically Grounded Diverse NTN Channel Dataset (3GPP TR 38.811)\n');
    fprintf(fid, '- **Data Generator:** MATLAB 5G Toolbox (`Gen_Chan_Diverse_Dataset.m`)\n');
    fprintf(fid, '- **Total Samples:** %d drops (Train: %d, Val: %d, Test: %d)\n', ...
        total_samples, n_tr, n_va, n_te);
    fprintf(fid, '- **Available Files:** `H_train.npy` (%d), `H_val.npy` (%d), `H_test.npy` (%d)\n', ...
        n_tr, n_va, n_te);
    fprintf(fid, '- **Data Type:** `complex64`, shape `(N, 132, 14)` (132 subcarriers × 14 OFDM symbols)\n');
    fprintf(fid, '- **Power Normalization:** Raw unnormalized physical channel scale preserved\n\n');
    
    fprintf(fid, '---\n\n');
    fprintf(fid, '## Parameter Distributions & Diversity Summary\n\n');
    fprintf(fid, '| Parameter | Tested Range | Distribution / Modeling |\n');
    fprintf(fid, '| :--- | :--- | :--- |\n');
    fprintf(fid, '| **Elevation Angle ($\\theta$)** | [%.1f°, %.1f°] (Mean: %.1f°) | Uniform random across pass range |\n', ...
        min(elev), max(elev), mean(elev));
    fprintf(fid, '| **UE Velocity ($v_{\\text{UE}}$)** | [%.1f, %.1f] m/s (%.0f–%.0f km/h) | Uniform vehicular mobility range |\n', ...
        min(v_ue), max(v_ue), min(v_ue)*3.6, max(v_ue)*3.6);
    fprintf(fid, '| **Channel Profiles** | NTN-TDL-A, B, C, D | LOS: %.1f%% (`NTN-TDL-C/D`), NLOS: %.1f%% (`NTN-TDL-A/B`) |\n', ...
        los_pct, nlos_pct);
    fprintf(fid, '| **Delay Spread ($\\tau_{\\text{rms}}$)** | [%.1f, %.1f] ns (Mean: %.1f ns) | **Coupled to Elevation & LOS:**<br>• LOS: 30–150 ns (mean ~60 ns)<br>• NLOS: 80–500 ns (mean ~260 ns) |\n', ...
        min(ds), max(ds), mean(ds));
    
    k_los = k_fac(is_los);
    if ~isempty(k_los)
        fprintf(fid, '| **Rician $K$-factor (LOS)** | [%.1f, %.1f] dB (Mean: %.1f dB) | Scales monotonically with elevation angle (3GPP TR 38.811) |\n', ...
            min(k_los), max(k_los), mean(k_los));
    end
    fprintf(fid, '| **UE Spatial Distribution** | Uniform disk inside 15 km beam | $r \\le 14.5$ km beam footprint offset |\n');
    fprintf(fid, '| **Carrier Frequency ($f_c$)** | %.2f GHz | S-band NTN carrier |\n', opt.CarrierFrequency / 1e9);
    fprintf(fid, '| **Subcarrier Spacing (SCS)** | %d kHz | 5G NR Numerology $\\mu=1$ |\n', opt.SubcarrierSpacing);
    fprintf(fid, '| **Grid Bandwidth** | 11 RBs (132 subcarriers) | 5G NR slot with 14 symbols |\n');
    fprintf(fid, '| **Antenna Configuration** | 1 × 1 (SISO) | Single-input single-output |\n\n');
    
    fprintf(fid, '---\n\n');
    fprintf(fid, '## Physical Coupling Justification (3GPP TR 38.811)\n\n');
    fprintf(fid, '1. **Elevation-Dependent LOS Probability**: At higher elevation angles ($\\\\theta \\\\ge 60^\\\\circ$), obstacles and terrain clutter are avoided, yielding up to ~85%% LOS. At lower elevation ($\\\\theta \\\\le 35^\\\\circ$), building blockage increases NLOS up to ~70%%.\n');
    fprintf(fid, '2. **Reasonable Delay Spread Scaling**: In NTN, scattering occurs solely near the UE. When LOS is present, the direct path dominates, keeping $\\tau_{\\text{rms}} \\le 150$ ns. In NLOS, rich urban multipath stretches $\\tau_{\\text{rms}}$ up to 500 ns.\n');
    fprintf(fid, '3. **Rician $K$-factor Scaling**: Direct-to-multipath ratio increases as the satellite approaches zenith ($K$ rises from ~5 dB at 30° to ~16 dB at 80°).\n');
    
    fclose(fid);
    fprintf('Saved dataset documentation: %s\n', md_file);
end
