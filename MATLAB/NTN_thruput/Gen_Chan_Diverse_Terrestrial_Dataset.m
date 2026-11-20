function Gen_Chan_Diverse_Terrestrial_Dataset(varargin)
% GEN_CHAN_DIVERSE_TERRESTRIAL_DATASET Generates diverse Terrestrial 3GPP TR 38.901 channel samples (.npy)
%
% Generates realistic terrestrial channels (TDL-A, TDL-B, TDL-C, TDL-D) across a cellular geometry
% for domain adaptation benchmarks (evaluating NTN-trained models transferring to terrestrial).
%
% Physically coupled parameters according to 3GPP TR 38.901 (Urban Macrocell - UMa):
%   - BS Height: 25.0 m (standard rooftop/macro tower)
%   - UE Height: 1.5 m (handheld / vehicular user)
%   - Cell Radius: 20 m to 500 m (random spatial distribution within cell footprint)
%   - LOS Probability: 3GPP TR 38.901 UMa distance-dependent formula
%   - Channel Profiles: 
%       * NLOS (TDL-A: 75%, TDL-B: 25%) - Rayleigh fading
%       * LOS  (TDL-C: 75%, TDL-D: 25%) - Rician fading with K-factor
%   - Delay Spread: 
%       * LOS:  30 ns to 250 ns  (mean ~100 ns)
%       * NLOS: 150 ns to 800 ns (mean ~350 ns)
%   - Rician K-factor (LOS): 0 dB to 9 dB (decreases with distance)
%   - UE Velocity: 5 m/s to 35 m/s (18 km/h to 126 km/h vehicular mobility)
%   - Doppler Spectrum: Classical Jakes isotropic multipath spectrum (no satellite carrier shift)
%   - Path Loss: 3GPP TR 38.901 UMa model with log-normal shadow fading
%
% Output files in generatedChannel_Results/<FolderName>/:
%   - H_train.npy (80% = 16,000 samples for default 20k)
%   - H_val.npy   (10% = 2,000 samples)
%   - H_test.npy  (10% = 2,000 samples)
%   - note.md (complete documentation)
%   - sample_metadata.mat (per-sample ground-truth parameters)

p = inputParser;
addParameter(p, 'NumUE', 20000, @isnumeric);
addParameter(p, 'CarrierFrequency', 2.18e9, @isnumeric);
addParameter(p, 'SubcarrierSpacing', 30, @isnumeric);
addParameter(p, 'BSHeight', 25.0, @isnumeric);
addParameter(p, 'UEHeight', 1.5, @isnumeric);
addParameter(p, 'CellRadius', 500.0, @isnumeric);
addParameter(p, 'FolderName', 'Diverse_Terrestrial_2p18e9_UMa_30kHz', @ischar);
parse(p, varargin{:});
opt = p.Results;

numUE = opt.NumUE;
fc = opt.CarrierFrequency;
scs = opt.SubcarrierSpacing;
h_BS = opt.BSHeight;
h_UE = opt.UEHeight;
r_cell = opt.CellRadius;

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
fprintf('  Generating Diverse Terrestrial 3GPP TR 38.901 Dataset (%d samples)\n', numUE);
fprintf('  Scenario: UMa (BS = %.1f m, UE = %.1f m, Cell Radius = %.0f m)\n', h_BS, h_UE, r_cell);
fprintf('  Carrier: %.2f GHz | SCS: %d kHz\n', fc / 1e9, scs);
fprintf('  Target Directory: %s\n', base_folder);
fprintf('====================================================================\n\n');

%% 1. 5G NR Carrier Configuration (Identical numerology to NTN model)
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

%% 2. Channel Model Initialization (Terrestrial 3GPP TR 38.901)
warning('off', 'MATLAB:system:nonRelevantProperty');
channel = nrTDLChannel;
channel.NumTransmitAntennas = 1;
channel.NumReceiveAntennas = 1;
channel.SampleRate = sampleRate;
channel.RandomStream = 'mt19937ar with seed';

%% 3. Preallocate Metadata & Output Arrays
H_perfect = zeros(14, carrier.NSizeGrid * 12, numUE, 'like', 1j);

meta_d2d_m     = zeros(numUE, 1);
meta_d3d_m     = zeros(numUE, 1);
meta_v_ue      = zeros(numUE, 1);
meta_ds_ns     = zeros(numUE, 1);
meta_profile   = cell(numUE, 1);
meta_is_los    = false(numUE, 1);
meta_k_factor  = zeros(numUE, 1);
meta_doppler   = zeros(numUE, 1);
meta_pathloss  = zeros(numUE, 1);

rng(42, 'twister'); % Seed for reproducible parameter generation

%% 4. Sample Generation Loop
t_start = tic;
print_step = max(500, floor(numUE / 20));

for idxUE = 1:numUE
    % -------------------------------------------------------------
    % A. Sample UE Position in Cell (Uniform area distribution)
    % -------------------------------------------------------------
    r_min = 20.0; % minimum distance to BS (m)
    r_ue = sqrt(r_min^2 + (r_cell^2 - r_min^2) * rand());
    d_2d = r_ue;
    d_3d = sqrt(d_2d^2 + (h_BS - h_UE)^2);
    
    meta_d2d_m(idxUE) = d_2d;
    meta_d3d_m(idxUE) = d_3d;
    
    % -------------------------------------------------------------
    % B. 3GPP TR 38.901 UMa LOS Probability Formula
    % -------------------------------------------------------------
    if d_2d <= 18.0
        p_los = 1.0;
    else
        p_los = (18.0 / d_2d) + exp(-d_2d / 63.0) * (1.0 - 18.0 / d_2d);
    end
    is_los = (rand() < p_los);
    meta_is_los(idxUE) = is_los;
    
    % -------------------------------------------------------------
    % C. Channel Profile & Delay Spread Sampling
    % -------------------------------------------------------------
    if is_los
        % LOS Profiles: TDL-C (75%) or TDL-D (25%)
        if rand() < 0.75
            prof = 'TDL-C';
        else
            prof = 'TDL-D';
        end
        
        % LOS Delay Spread: 30 ns to 250 ns (increases slightly with distance)
        base_ds = 40e-9 + (d_2d / r_cell) * 120e-9; % 40 ns to 160 ns base
        ds = max(30e-9, min(250e-9, base_ds + (rand() - 0.5) * 60e-9));
        
        % Rician K-factor: 0 dB to 9 dB (decreases with distance)
        base_k = 9.0 - (d_2d / r_cell) * 7.0; % 9 dB at close range -> 2 dB at cell edge
        k_dB = max(0.0, min(12.0, base_k + (rand() - 0.5) * 4.0));
    else
        % NLOS Profiles: TDL-A (75%) or TDL-B (25%)
        if rand() < 0.75
            prof = 'TDL-A';
        else
            prof = 'TDL-B';
        end
        
        % NLOS Delay Spread: 150 ns to 800 ns (rich multipath scattering)
        base_ds = 200e-9 + (d_2d / r_cell) * 350e-9; % 200 ns to 550 ns base
        ds = max(150e-9, min(800e-9, base_ds + (rand() - 0.5) * 150e-9));
        k_dB = -Inf; % Rayleigh
    end
    
    meta_profile{idxUE} = prof;
    meta_ds_ns(idxUE) = ds * 1e9;
    meta_k_factor(idxUE) = k_dB;
    
    % -------------------------------------------------------------
    % D. UE Vehicular Mobility & Maximum Doppler Shift
    % -------------------------------------------------------------
    v_ue = 5.0 + 30.0 * rand(); % 5 to 35 m/s (18 to 126 km/h)
    meta_v_ue(idxUE) = v_ue;
    max_doppler_ue = v_ue * fc / c;
    meta_doppler(idxUE) = max_doppler_ue;
    
    % -------------------------------------------------------------
    % E. 3GPP TR 38.901 UMa Path Loss Model with Shadow Fading
    % -------------------------------------------------------------
    fc_GHz = fc / 1e9;
    if is_los
        % UMa LOS: PL = 28.0 + 22.0*log10(d_3D) + 20.0*log10(fc_GHz) + SF (sigma = 4 dB)
        pl_dB = 28.0 + 22.0 * log10(d_3d) + 20.0 * log10(fc_GHz) + 4.0 * randn();
    else
        % UMa NLOS: PL = 13.54 + 39.03*log10(d_3D) + 20.0*log10(fc_GHz) - 0.6*(h_UE - 1.5) + SF (sigma = 6 dB)
        pl_dB = 13.54 + 39.03 * log10(d_3d) + 20.0 * log10(fc_GHz) - 0.6 * (h_UE - 1.5) + 6.0 * randn();
    end
    meta_pathloss(idxUE) = pl_dB;
    
    % -------------------------------------------------------------
    % F. Configure nrTDLChannel (Terrestrial: No Satellite Doppler!)
    % -------------------------------------------------------------
    release(channel);
    channel.DelayProfile = prof;
    channel.DelaySpread = ds;
    channel.MaximumDopplerShift = max_doppler_ue;
    % Crucial: Terrestrial channel has no satellite orbital shift
    % channel.SatelliteDopplerShift is intentionally omitted / default 0
    channel.Seed = idxUE + 30 * numUE;
    
    if is_los
        channel.KFactorScaling = true;
        channel.KFactor = k_dB;
    else
        channel.KFactorScaling = false;
    end
    
    % -------------------------------------------------------------
    % G. Filter Waveform & Demodulate (No Satellite Pre-compensation)
    % -------------------------------------------------------------
    [rxWaveform, pathGains] = channel(txWaveform1);
    pathFilters = getPathFilters(channel);
    offset = nrPerfectTimingEstimate(pathGains, pathFilters);
    rxGrid = nrOFDMDemodulate(carrier, rxWaveform(1+offset:end, :));
    
    % Apply linear path loss
    H_perfect_n = rxGrid * db2mag(-pl_dB);
    H_perfect(:,:,idxUE) = H_perfect_n.';
    
    % Progress display
    if mod(idxUE, print_step) == 0 || idxUE == numUE
        elapsed = toc(t_start);
        rate = idxUE / elapsed;
        eta_sec = (numUE - idxUE) / max(1e-3, rate);
        fprintf('  [%5d / %5d] (%5.1f%%) | d_2d: %5.1f m | %-5s | DS: %5.1f ns | v: %4.1f m/s | fd: %5.1f Hz | ETA: %.1f min\n', ...
            idxUE, numUE, idxUE/numUE*100, d_2d, prof, ds*1e9, v_ue, max_doppler_ue, eta_sec/60);
    end
end

elapsed_total = toc(t_start);
fprintf('\n>>> Terrestrial channel generation completed in %.2f min (%.2f ms/sample)\n', ...
    elapsed_total / 60, (elapsed_total / numUE) * 1000);

%% 5. Save NPY Dataset Splits (Train 80%, Val 10%, Test 10%)
fprintf('\n--- Saving .npy dataset splits (H_train, H_val, H_test) ---\n');

n_train = floor(0.80 * numUE);
n_val   = floor(0.10 * numUE);

H_train = H_perfect(:, :, 1:n_train);
H_val   = H_perfect(:, :, n_train+1:n_train+n_val);
H_test  = H_perfect(:, :, n_train+n_val+1:end);

saveNPY_channel(fullfile(base_folder, 'H_train.npy'), H_train, false);
saveNPY_channel(fullfile(base_folder, 'H_val.npy'), H_val, false);
saveNPY_channel(fullfile(base_folder, 'H_test.npy'), H_test, false);

%% 6. Save Sample Metadata (.mat & .csv) for Distribution Analysis
save(fullfile(base_folder, 'sample_metadata.mat'), ...
    'meta_d2d_m', 'meta_d3d_m', 'meta_v_ue', 'meta_ds_ns', 'meta_profile', ...
    'meta_is_los', 'meta_k_factor', 'meta_doppler', 'meta_pathloss', ...
    '-v7.3');
fprintf('Saved sample metadata: %s\n', fullfile(base_folder, 'sample_metadata.mat'));

try
    T_meta = table((1:numUE)', meta_d2d_m, meta_d3d_m, meta_v_ue, meta_ds_ns, string(meta_profile), ...
        meta_is_los, meta_k_factor, meta_doppler, meta_pathloss, ...
        'VariableNames', {'SampleIdx', 'd2d_m', 'd3d_m', 'v_ue_mps', 'ds_ns', 'profile', 'is_los', 'k_factor_dB', 'doppler_Hz', 'pathloss_dB'});
    writetable(T_meta, fullfile(base_folder, 'sample_metadata.csv'));
    fprintf('Saved sample metadata CSV: %s\n', fullfile(base_folder, 'sample_metadata.csv'));
catch ME
    fprintf('Warning: could not write CSV metadata: %s\n', ME.message);
end

%% 7. Save Comprehensive note.md
save_terrestrial_note_md(base_folder, opt, meta_d2d_m, meta_v_ue, meta_ds_ns, ...
    meta_profile, meta_is_los, meta_k_factor, meta_doppler, size(H_train, 3), size(H_val, 3), size(H_test, 3));

fprintf('\n====================================================================\n');
fprintf('  Terrestrial Dataset successfully created in:\n  %s\n', base_folder);
fprintf('====================================================================\n');
end

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

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

function save_terrestrial_note_md(folder, opt, d2d, v_ue, ds, profs, is_los, k_fac, doppler, n_tr, n_va, n_te)
    md_file = fullfile(folder, 'note.md');
    fid = fopen(md_file, 'w');
    if fid == -1
        return;
    end
    
    total_samples = n_tr + n_va + n_te;
    los_pct = mean(is_los) * 100;
    nlos_pct = 100 - los_pct;
    
    fprintf(fid, '# Dataset Overview: Diverse Terrestrial 3GPP TR 38.901 Channel (%.2f GHz, UMa, 30 kHz)\n\n', ...
        opt.CarrierFrequency / 1e9);
    
    fprintf(fid, '## General Information\n\n');
    fprintf(fid, '- **Dataset Type:** Physically Grounded Terrestrial Channel Dataset (3GPP TR 38.901 UMa)\n');
    fprintf(fid, '- **Data Generator:** MATLAB 5G Toolbox (`Gen_Chan_Diverse_Terrestrial_Dataset.m`)\n');
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
    fprintf(fid, '| **Base Station Height ($h_{\\text{BS}}$)** | %.1f m | Standard Urban Macro (UMa) rooftop tower |\n', opt.BSHeight);
    fprintf(fid, '| **UE Height ($h_{\\text{UE}}$)** | %.1f m | Standard pedestrian / vehicular terminal height |\n', opt.UEHeight);
    fprintf(fid, '| **2D Cell Distance ($d_{\\text{2D}}$)** | [%.1f, %.1f] m (Mean: %.1f m) | Uniform random area distribution inside %.0f m cell radius |\n', ...
        min(d2d), max(d2d), mean(d2d), opt.CellRadius);
    fprintf(fid, '| **UE Velocity ($v_{\\text{UE}}$)** | [%.1f, %.1f] m/s (%.0f–%.0f km/h) | Uniform vehicular mobility range |\n', ...
        min(v_ue), max(v_ue), min(v_ue)*3.6, max(v_ue)*3.6);
    fprintf(fid, '| **Max Doppler Shift ($f_{D,\\max}$)** | [%.1f, %.1f] Hz (Mean: %.1f Hz) | Isotropic Jakes multipath spectrum (no satellite shift) |\n', ...
        min(doppler), max(doppler), mean(doppler));
    fprintf(fid, '| **Channel Profiles** | TDL-A, TDL-B, TDL-C, TDL-D | LOS: %.1f%% (`TDL-C/D`), NLOS: %.1f%% (`TDL-A/B`) |\n', ...
        los_pct, nlos_pct);
    fprintf(fid, '| **Delay Spread ($\\tau_{\\text{rms}}$)** | [%.1f, %.1f] ns (Mean: %.1f ns) | **Coupled to LOS/NLOS & Distance:**<br>• LOS: 30–250 ns (mean ~100 ns)<br>• NLOS: 150–800 ns (mean ~350 ns) |\n', ...
        min(ds), max(ds), mean(ds));
    
    k_los = k_fac(is_los);
    if ~isempty(k_los)
        fprintf(fid, '| **Rician $K$-factor (LOS)** | [%.1f, %.1f] dB (Mean: %.1f dB) | Distance-dependent direct-to-multipath ratio (3GPP TR 38.901) |\n', ...
            min(k_los), max(k_los), mean(k_los));
    end
    fprintf(fid, '| **Carrier Frequency ($f_c$)** | %.2f GHz | Matched to NTN carrier for direct adaptation analysis |\n', opt.CarrierFrequency / 1e9);
    fprintf(fid, '| **Subcarrier Spacing (SCS)** | %d kHz | 5G NR Numerology $\\mu=1$ |\n', opt.SubcarrierSpacing);
    fprintf(fid, '| **Grid Bandwidth** | 11 RBs (132 subcarriers) | 5G NR slot with 14 symbols |\n');
    fprintf(fid, '| **Antenna Configuration** | 1 × 1 (SISO) | Single-input single-output |\n\n');
    
    fprintf(fid, '---\n\n');
    fprintf(fid, '## Physical Justification & Domain Gap vs. NTN\n\n');
    fprintf(fid, '1. **No Satellite Doppler Pre-compensation**: In NTN, the gNB/UE pre-compensates for the massive orbital Doppler shift, leaving only slight residual Doppler. In terrestrial, the BS is stationary, and all Doppler is generated by user mobility with an isotropic (360°) Jakes arrival spectrum.\n');
    fprintf(fid, '2. **Higher Delay Spread (Stronger Frequency Selectivity)**: Due to rich terrestrial building clutter and multi-bounce scatterers, delay spread reaches 150–800 ns (vs 30–150 ns in NTN LOS), significantly reducing coherence bandwidth.\n');
    fprintf(fid, '3. **Predominance of NLOS Rayleigh Fading**: While NTN at medium-to-high elevations is dominated by strong LOS paths ($K \\approx 5-18$ dB), terrestrial UMa is predominantly NLOS (~70–80%% NLOS) with deep Rayleigh fading nulls across subcarriers.\n');
    
    fclose(fid);
    fprintf('Saved dataset documentation: %s\n', md_file);
end
