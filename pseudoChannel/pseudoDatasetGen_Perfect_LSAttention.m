%% Pseudo Dataset Generation for NTN Channel Estimation/Prediction (LS_Attention Version)
% =========================================================================
% WORKFLOW & PURPOSE:
% 1. Pseudo Ground Truth Generation (FDA):
%    - Fourier transfer is the result of translation of Perfect (source) -> LS_infer (target).
%    - Translates target domain style (Delay-Doppler low-frequency components of target H_LS_infer)
%      into clean source domain channels (source Perfect) using Delay-Doppler Fourier Domain
%      Adaptation (FDA) with window size (w_h = 13 x w_w in [3, 5]).
%    - Produces the pseudo ground-truth channel grid: H_perfect (14 x 132 x N).
%
% 2. Flexible Sample Alignment & Expansion:
%    - Allows generating arbitrary number of pseudo samples (num_pseudo_samples),
%      even larger than the available samples in Source (N_src) and Target (N_tgt):
%      * Phase 1 (1 to N_min): In-order 1-to-1 mapping (Source 1 -> Target 1, ..., N_min -> N_min).
%      * Phase 2 (N_min+1 to N_max): In-order for the larger dataset, while sampling
%        randomly from the smaller dataset.
%      * Phase 3 (> N_max): Random sampling from both Source and Target datasets.
%    - Saves explicit mapping indices: idx_map_source and idx_map_target into matlabNTN.mat
%      and sample_mapping.mat.
%
% 3. 5G NR Transmission, Noise Realization & Channel Estimation:
%    - Simulates pilot transmission across the generated pseudo channel grids.
%    - Injects complex AWGN calibrated to pilot signal power across target SNRs
%      (SNR_dB = -10:5:15 dB).
%    - Computes Least Squares (LS) channel estimates at pilot locations: H_ls_pilots.
%    - Performs 2D scattered linear interpolation and edge boundary cropping
%      to reconstruct the estimated channel grid: H_li (14 x 132 x N).
%
% 4. Metric Evaluation & Output Organization:
%    - Evaluates NMSE and complex SSIM metrics on both full-grid interpolated
%      channels (nmse_li, ssim_li) and pilot-position estimations (nmse_ls_pilot,
%      ssim_ls, ssim_li_pilot).
%    - Saves structured .mat files into nested folders: <outputFolder>/13x<w_w>/SNR_<X>dB/matlabNTN.mat.
%    - Generates markdown documentation (note.md) and copies notes from source
%      and target paths (note_source.md, note_target.md).
%
% NOTE ON PILOT CONFIGURATION:
%    - Pilot positions are extracted directly and dynamically from the TARGET domain dataset.
% =========================================================================

% =========================================================================
% Configuration: Source and Target Datasets
% =========================================================================
SourceDatasetPath = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\generatedChannel_Results\A100_2p18e9_600km_70deg_30kHz";
TargetDatasetPath = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\inferred_dataset\A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz\LS_Attention_standardize";

% Output directory
outputFolder = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\A100Perfect__DUR300LSAttention";
outpuFolder  = outputFolder; % backward compatibility alias

SNR_dB = -10:5:15;
w_window = [3, 5];

% Desired number of samples in generated pseudo dataset:
% - Set to [] (or -1) to automatically use min(N_src, N_tgt)
% - Can be set to any positive integer (e.g. 500, 1000, 1500, 2000), even > N_max
num_pseudo_samples = []; 

% Random seed for reproducible sample pairing across SNRs
rng_seed = 42;

% Preprocessing scaling mode for Fourier Domain Adaptation ("rms", "minmax", or "none"):
%   "rms"    - Normalizes both source and target to unit RMS power before FDA,
%              then descales the pseudo channel with target RMS power.
%   "minmax" - Scales Real/Imag parts to [-1, 1] before FDA,
%              then descales with target [min, max] range.
%   "none"   - Direct unscaled FDA mixing.
preprocessing_scale = "rms";

% Whether to clamp boundaries of target inferred channel (typically false for model inferences)
apply_crop_target = false;

if exist('mfilename', 'builtin') && ~isempty(mfilename('fullpath'))
    script_dir = fileparts(mfilename('fullpath'));
elseif exist('matlab.desktop.editor.getActiveFilename', 'builtin') && ~isempty(matlab.desktop.editor.getActiveFilename)
    script_dir = fileparts(matlab.desktop.editor.getActiveFilename);
else
    script_dir = pwd;
end

if ~isempty(script_dir) && exist(script_dir, 'dir')
    cd(script_dir);
end
addpath('..\helper\')

%% 1. Gen Tx grid DM-RS pilots from Target Dataset Configuration
fprintf('--- 1. Loading Target Pilot Configuration ---\n');
% Load a sample file from TargetDatasetPath to extract pilot_rows and pilot_cols
sample_target = load_snr_dataset(TargetDatasetPath, 10);
if isempty(sample_target)
    sample_target = load_snr_dataset(TargetDatasetPath, 0);
end
if isempty(sample_target)
    sample_target = load_snr_dataset(TargetDatasetPath);
end
if isempty(sample_target)
    error('Could not load any target dataset file from %s', TargetDatasetPath);
end

if ~isfield(sample_target, 'pilot_rows') || ~isfield(sample_target, 'pilot_cols')
    error('Target dataset must contain "pilot_rows" and "pilot_cols" fields.');
end

pilot_rows = double(sample_target.pilot_rows(:)); % subcarrier indices as column vector
pilot_cols = double(sample_target.pilot_cols(:)); % OFDM symbol indices as column vector

% Safeguard for 0-based indexing
if min(pilot_rows) == 0
    pilot_rows = pilot_rows + 1;
end
if min(pilot_cols) == 0
    pilot_cols = pilot_cols + 1;
end
numPilots = length(pilot_rows);

% Map pilots with value 1 to the pilot positions in txGrid (132 subcarriers x 14 symbols)
txGrid = zeros(132, 14);
pilot_indices = sub2ind(size(txGrid), pilot_rows, pilot_cols);
txGrid(pilot_indices) = 1;
pilot_symbols = txGrid(pilot_indices); % All 1s

fprintf('Loaded %d pilot positions from Target dataset.\n', numPilots);
fprintf('Pilot subcarrier range: [%d, %d], symbol range: [%d, %d]\n\n', ...
    min(pilot_rows), max(pilot_rows), min(pilot_cols), max(pilot_cols));

%% 2. Load Clean Source Ground Truth Channel
fprintf('--- 2. Loading Clean Source Ground Truth Channel ---\n');
source_domain = load_snr_dataset(SourceDatasetPath, 10);
if isempty(source_domain)
    source_domain = load_snr_dataset(SourceDatasetPath, 0);
end
if isempty(source_domain) || ~isfield(source_domain, 'H_perfect')
    error('Could not load source ground truth (H_perfect) from %s', SourceDatasetPath);
end

% Standardize source H_perfect dimensions to: (14 symbols x 132 subcarriers x N_src)
src_H_perf_all = source_domain.H_perfect;
if size(src_H_perf_all, 1) == 14 && size(src_H_perf_all, 2) == 132
    % Already (14 x 132 x N)
elseif size(src_H_perf_all, 2) == 132 && size(src_H_perf_all, 3) == 14
    src_H_perf_all = permute(src_H_perf_all, [3, 2, 1]); % (N, 132, 14) -> (14, 132, N)
elseif size(src_H_perf_all, 1) == 132 && size(src_H_perf_all, 2) == 14
    src_H_perf_all = permute(src_H_perf_all, [2, 1, 3]); % (132, 14, N) -> (14, 132, N)
end
N_src_total = size(src_H_perf_all, 3);
fprintf('Loaded Source H_perfect with size: %s (Total Samples: %d)\n\n', ...
    mat2str(size(src_H_perf_all)), N_src_total);

%% Create result folder and markdown notes
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Copy source note.md if available
src_note = fullfile(SourceDatasetPath, 'note.md');
if exist(src_note, 'file')
    copyfile(src_note, fullfile(outputFolder, 'note_source.md'));
    fprintf('Copied Source note.md -> %s\n', fullfile(outputFolder, 'note_source.md'));
end

% Copy target documentation if available (note.md or info.md)
tgt_note = fullfile(TargetDatasetPath, 'note.md');
if ~exist(tgt_note, 'file')
    tgt_note = fullfile(TargetDatasetPath, 'info.md');
end
if exist(tgt_note, 'file')
    copyfile(tgt_note, fullfile(outputFolder, 'note_target.md'));
    fprintf('Copied Target documentation -> %s\n', fullfile(outputFolder, 'note_target.md'));
end

% Generate note.md at output folder
pseudo_note_path = fullfile(outputFolder, 'note.md');
fid = fopen(pseudo_note_path, 'w');
if fid ~= -1
    fprintf(fid, '# Pseudo Dataset Overview: FDA Domain Adaptation (LS_Attention)\n\n');
    fprintf(fid, '## General Information\n\n');
    fprintf(fid, '- **Method:** Delay-Doppler Fourier Domain Adaptation (FDA) + 5G NR DM-RS Pilot & Noise Realization\n');
    fprintf(fid, '- **Fourier Transfer (FDA):** Translation of **Perfect (source) -> LS_infer (target)** (`src_H_perf` -> `tgt_H_infer`)\n');
    fprintf(fid, '- **Source Dataset Path:** `%s`\n', SourceDatasetPath);
    fprintf(fid, '- **Target Dataset Path:** `%s`\n', TargetDatasetPath);
    fprintf(fid, '- **Result Folder:** `%s`\n', outputFolder);
    fprintf(fid, '- **Generation Date:** %s\n\n', string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
    fprintf(fid, '> **Note on Fourier Transfer & Sampling Strategy:**\n');
    fprintf(fid, '> - FDA transfers low-frequency Delay-Doppler components from target domain `tgt_H_infer` onto clean source channel `src_H_perf`.\n');
    fprintf(fid, '> - When dataset sizes differ or when target pseudo sample count exceeds dataset sizes:\n');
    fprintf(fid, '>   * Samples `1` to `N_min`: Strict 1-to-1 sequential mapping (Source `i` -> Target `i`).\n');
    fprintf(fid, '>   * Samples `N_min+1` to `N_max`: Sequential mapping for the larger dataset, uniform random sampling from the smaller dataset.\n');
    fprintf(fid, '>   * Samples `> N_max`: Uniform random sampling from both Source and Target datasets.\n');
    fprintf(fid, '> - Corresponding index mappings are preserved in variables `idx_map_source` and `idx_map_target`.\n\n');
    fprintf(fid, '---\n\n');
    fprintf(fid, '## Configuration Summary\n\n');
    fprintf(fid, '| Parameter | Value |\n');
    fprintf(fid, '| :--- | :--- |\n');
    fprintf(fid, '| **Fourier Transfer (FDA)** | **Perfect (source)** $\\rightarrow$ **LS_infer (target)** (`src_H_perf` $\\rightarrow$ `tgt_H_infer`) |\n');
    fprintf(fid, '| **Source Domain** | NTN TDL-A (NLOS, 70° elevation, 100 ns delay spread, 30 kHz SCS) |\n');
    fprintf(fid, '| **Target Domain** | OpenNTN DUR NLOS (30° elevation, 300 ns delay spread, 30 kHz SCS) - LS_Attention Inferred |\n');
    fprintf(fid, '| **FDA Preprocessing Scaling** | `%s` (Source & Target normalized to unit RMS, descaled by Target RMS) |\n', preprocessing_scale);
    fprintf(fid, '| **FDA Window ($13 \\times w_w$)** | %s |\n', mat2str(w_window));
    fprintf(fid, '| **SNR Range** | %s dB |\n', mat2str(SNR_dB));
    fprintf(fid, '| **Requested Pseudo Samples** | %s |\n', mat2str(num_pseudo_samples));
    fprintf(fid, '| **Grid Dimensions** | 132 Subcarriers × 14 OFDM Symbols (11 RBs) |\n');
    fprintf(fid, '| **Pilot Configuration** | DM-RS Type 2 Port 1 (%d pilots per slot: symbols 3, 12; subcarriers 1-128) |\n', numPilots);
    fprintf(fid, '| **Saved Variables** | `H_perfect`, `H_li`, `H_ls_pilots`, `pilot_rows`, `pilot_cols`, `pilot_indices`, `idx_map_source`, `idx_map_target`, `nmse_li`, `nmse_ls_pilot`, `ssim_li`, `ssim_li_pilot`, `ssim_ls` |\n\n');
    fprintf(fid, '## Linked Reference Notes\n\n');
    fprintf(fid, '- [Source Dataset Note](note_source.md)\n');
    fprintf(fid, '- [Target Dataset Note](note_target.md)\n');
    fclose(fid);
    fprintf('Created note.md at %s\n\n', pseudo_note_path);
end

%% 3. Generation Loop (w_window x SNR_dB)
for w_w = w_window
    for snr_db = SNR_dB
        fprintf('====================================================\n');
        fprintf('Processing: w_window = %d, SNR = %d dB\n', w_w, snr_db);
        
        target_domain = load_snr_dataset(TargetDatasetPath, snr_db);
        if isempty(target_domain)
            warning('Target dataset not found for SNR=%d dB in %s. Skipping.', snr_db, TargetDatasetPath);
            continue;
        end

        % Adaptively extract target inferred channel (H_LS_infer, H_ls_infer, H_infer, or H_li)
        if isfield(target_domain, 'H_LS_infer')
            tgt_H_raw = double(target_domain.H_LS_infer);
            target_var_name = 'H_LS_infer';
        elseif isfield(target_domain, 'H_ls_infer')
            tgt_H_raw = double(target_domain.H_ls_infer);
            target_var_name = 'H_ls_infer';
        elseif isfield(target_domain, 'H_infer')
            tgt_H_raw = double(target_domain.H_infer);
            target_var_name = 'H_infer';
        elseif isfield(target_domain, 'H_li')
            tgt_H_raw = double(target_domain.H_li);
            target_var_name = 'H_li';
        else
            warning('Target dataset H_LS_infer / H_infer not found for SNR=%d dB. Skipping.', snr_db);
            continue;
        end

        % Standardize target inferred channel dimensions to: (14 symbols x 132 subcarriers x N_tgt)
        if size(tgt_H_raw, 1) == 14 && size(tgt_H_raw, 2) == 132
            tgt_H_infer_all = tgt_H_raw;
        elseif size(tgt_H_raw, 2) == 132 && size(tgt_H_raw, 3) == 14
            tgt_H_infer_all = permute(tgt_H_raw, [3, 2, 1]); % (N, 132, 14) -> (14, 132, N)
        elseif size(tgt_H_raw, 1) == 132 && size(tgt_H_raw, 2) == 14
            tgt_H_infer_all = permute(tgt_H_raw, [2, 1, 3]); % (132, 14, N) -> (14, 132, N)
        elseif size(tgt_H_raw, 2) == 14 && size(tgt_H_raw, 3) == 132
            tgt_H_infer_all = permute(tgt_H_raw, [2, 3, 1]); % (N, 14, 132) -> (14, 132, N)
        else
            error('Unexpected dimensions for target %s: %s', target_var_name, mat2str(size(tgt_H_raw)));
        end

        % Sample Count Determination and Alignment
        N_src = size(src_H_perf_all, 3);
        N_tgt = size(tgt_H_infer_all, 3);

        if isempty(num_pseudo_samples) || num_pseudo_samples <= 0
            nSamples = min(N_src, N_tgt);
        else
            nSamples = round(num_pseudo_samples);
        end

        % Seed RNG for reproducible sampling across SNR loops
        if ~isempty(rng_seed)
            rng(rng_seed);
        end

        N_min = min(N_src, N_tgt);
        N_max = max(N_src, N_tgt);

        idx_map_source = zeros(1, nSamples);
        idx_map_target = zeros(1, nSamples);

        % Phase 1: Samples 1 to min(nSamples, N_min) -> Sequential 1-to-1 mapping
        n_p1 = min(nSamples, N_min);
        idx_map_source(1:n_p1) = 1:n_p1;
        idx_map_target(1:n_p1) = 1:n_p1;

        % Phase 2: Samples N_min+1 to min(nSamples, N_max)
        % Larger dataset stays sequential; smaller dataset samples uniformly at random
        if nSamples > N_min
            n_p2_end = min(nSamples, N_max);
            p2_range = (N_min + 1) : n_p2_end;
            len_p2   = length(p2_range);
            if N_src >= N_tgt
                % Source is larger -> Source in-order, Target random
                idx_map_source(p2_range) = p2_range;
                idx_map_target(p2_range) = randi(N_tgt, 1, len_p2);
            else
                % Target is larger -> Target in-order, Source random
                idx_map_source(p2_range) = randi(N_src, 1, len_p2);
                idx_map_target(p2_range) = p2_range;
            end
        end

        % Phase 3: Samples N_max+1 to nSamples
        % Both datasets pick random samples uniformly with replacement
        if nSamples > N_max
            p3_range = (N_max + 1) : nSamples;
            len_p3   = length(p3_range);
            idx_map_source(p3_range) = randi(N_src, 1, len_p3);
            idx_map_target(p3_range) = randi(N_tgt, 1, len_p3);
        end

        fprintf('Dataset Alignment: Source (%d) | Target (%d) -> Generating %d pseudo samples\n', ...
            N_src, N_tgt, nSamples);
        fprintf('  - Phase 1 (1:%d): 1-to-1 sequential\n', n_p1);
        if nSamples > N_min
            fprintf('  - Phase 2 (%d:%d): Larger dataset sequential, smaller dataset random\n', ...
                N_min + 1, min(nSamples, N_max));
        end
        if nSamples > N_max
            fprintf('  - Phase 3 (%d:%d): Both datasets random sampling\n', ...
                N_max + 1, nSamples);
        end

        % Slice channels according to index mapping
        src_H_perf  = src_H_perf_all(:, :, idx_map_source);
        tgt_H_infer = tgt_H_infer_all(:, :, idx_map_target);

        % Optional preprocessing: crop extrapolation boundaries if requested
        if apply_crop_target
            for n = 1:nSamples
                tgt_H_infer(:, :, n) = crop_(tgt_H_infer(:, :, n), ...
                    [min(pilot_cols), max(pilot_cols)], ...
                    [min(pilot_rows), max(pilot_rows)]);
            end
        end

        % Step A: Generate pseudo label (H_pseudo) by Fourier translation in Delay-Doppler domain
        % Translates target style (Doppler/delay from H_LS_infer) into source channel
        pseudo_label = FTranslate_bulk(src_H_perf, tgt_H_infer, 13, w_w, preprocessing_scale); % 14 x 132 x nSamples
        H_perfect = pseudo_label;                                                               % Pseudo ground truth
        H_pseudo  = permute(pseudo_label, [2, 1, 3]);                                           % 132 x 14 x nSamples

        % Initialize output arrays
        H_li = zeros(size(pseudo_label));
        H_ls_pilots = zeros(numPilots, nSamples);

        nmse_pseudo_li = 0;
        nmse_pseudo_ls = 0;
        ssim_pseudo_li = 0;
        ssim_pseudo_li_pilot = 0;
        ssim_pseudo_ls = 0;

        % Step B: Transmission, Noise Injection, LS Estimation, and Linear Interpolation
        for n = 1:nSamples
            H_pseudo_n = H_pseudo(:, :, n); % 132 x 14
            
            % Transmit txGrid through pseudo channel
            Y_received_noNoise = txGrid .* H_pseudo_n;

            % Calculate Noise Variance based on pilot signal power
            Ps = mean(abs(Y_received_noNoise(pilot_indices)).^2);
            sigma2 = Ps / (10^(snr_db / 10));

            % Generate Complex AWGN
            Noise = sqrt(sigma2 / 2) * (randn(size(txGrid)) + 1i * randn(size(txGrid)));
            Y_received = Y_received_noNoise + Noise;

            % Step C: LS Estimation at pilot positions as a sequence
            % Since txGrid(pilot_indices) = 1, LS estimate is Y_received(pilot_indices) / 1
            H_ls_pilots(:, n) = Y_received(pilot_indices);

            % Step D: Linear Interpolation across the 2D grid
            [~, H_linear_n] = Lin_Interpolate(Y_received, pilot_indices, pilot_symbols);
            
            % Crop extrapolated edges beyond pilot boundaries
            H_linear_n_ = crop_(H_linear_n, [min(pilot_rows), max(pilot_rows)], [min(pilot_cols), max(pilot_cols)]);

            % Store interpolated channel (14 x 132)
            H_li(:, :, n) = permute(H_linear_n_, [2, 1, 3]);

            % Calculate NMSE for Linear Interpolation
            error_li = H_pseudo_n - H_linear_n_;
            true_power = sum(abs(H_pseudo_n).^2, 'all');
            nmse_pseudo_li = nmse_pseudo_li + (sum(abs(error_li).^2, 'all') / true_power);

            % Calculate NMSE for LS at pilot positions
            error_ls = H_ls_pilots(:, n) - H_pseudo_n(pilot_indices);
            pilot_power = sum(abs(H_pseudo_n(pilot_indices)).^2);
            nmse_pseudo_ls = nmse_pseudo_ls + (sum(abs(error_ls).^2) / pilot_power);

            % Calculate SSIM for Linear Interpolation (full 2D grid vs H_pseudo_n)
            ssim_pseudo_li = ssim_pseudo_li + compute_complex_ssim(H_pseudo_n, H_linear_n_);

            % Calculate SSIM for Linear Interpolation at pilot positions
            ssim_pseudo_li_pilot = ssim_pseudo_li_pilot + compute_complex_ssim(H_pseudo_n(pilot_indices), H_linear_n_(pilot_indices));

            % Calculate SSIM for LS at pilot positions (vs true pilot values)
            ssim_pseudo_ls = ssim_pseudo_ls + compute_complex_ssim(H_pseudo_n(pilot_indices), H_ls_pilots(:, n));
        end

        nmse_li = nmse_pseudo_li / nSamples;
        nmse_ls_pilot = nmse_pseudo_ls / nSamples;
        ssim_li = ssim_pseudo_li / nSamples;
        ssim_li_pilot = ssim_pseudo_li_pilot / nSamples;
        ssim_ls = ssim_pseudo_ls / nSamples;
        ssim_ls_pilot = ssim_ls;

        % Step E: Save dataset
        save_folder = fullfile(outputFolder, ['13x', num2str(w_w)], ['SNR_', num2str(snr_db), 'dB']);
        if ~exist(save_folder, 'dir')
            mkdir(save_folder);
        end

        output_mat = fullfile(save_folder, 'matlabNTN.mat');
        save(output_mat, ...
            'H_perfect', ...     % Ground truth channel: (14 x 132 x nSamples)
            'H_li', ...          % Linearly interpolated channel: (14 x 132 x nSamples)
            'H_ls_pilots', ...   % LS sequence at pilots: (numPilots x nSamples)
            'pilot_rows', ...    % 1-based subcarrier coordinates: (1 x numPilots)
            'pilot_cols', ...    % 1-based OFDM symbol coordinates: (1 x numPilots)
            'pilot_indices', ... % 1-based linear indices in 132x14 grid
            'idx_map_source', ...% Source sample indices mapped to pseudo samples: (1 x nSamples)
            'idx_map_target', ...% Target sample indices mapped to pseudo samples: (1 x nSamples)
            'nmse_li', ...       % Average NMSE of linear interpolation
            'nmse_ls_pilot', ... % Average NMSE of LS at pilots
            'ssim_li', ...       % Average SSIM of linear interpolation (full grid)
            'ssim_li_pilot', ... % Average SSIM of linear interpolation at pilots
            'ssim_ls', ...       % Average SSIM of LS at pilots
            'ssim_ls_pilot', ... % Average SSIM of LS at pilots (alias)
            '-v7.3');

        % Also save global mapping file at outputFolder root
        mapping_mat = fullfile(outputFolder, 'sample_mapping.mat');
        save(mapping_mat, 'idx_map_source', 'idx_map_target', 'N_src', 'N_tgt', 'nSamples', '-v7.3');

        fprintf('Saved: %s\n', output_mat);
        fprintf('  -> NMSE (LI): %.4f (%.2f dB) | NMSE (LS pilots): %.4f (%.2f dB)\n', ...
            nmse_li, 10*log10(nmse_li), nmse_ls_pilot, 10*log10(nmse_ls_pilot));
        fprintf('  -> SSIM (LI): %.4f | SSIM (LI pilots): %.4f | SSIM (LS pilots): %.4f\n', ...
            ssim_li, ssim_li_pilot, ssim_ls);

        % Step F: Plot and Save Channel Visualizations (4 pairs: Magnitude & Real)
        plot_channel_comparisons(save_folder, src_H_perf, tgt_H_infer, H_perfect, H_li, snr_db, w_w, idx_map_source, idx_map_target);
    end
end

fprintf('\nAll pseudo dataset generations finished successfully!\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function matData = load_snr_dataset(datasetPath, snr_db)
    % Adaptively loads .mat file (inferredChannel.mat, matlabNTN.mat, or channel_dur_randomizedUE.mat)
    % from LS_<snr>dB, SNR_<snr>dB, or <snr>dB subfolder
    matData = [];
    if nargin >= 2 && ~isempty(snr_db)
        candidates = {
            fullfile(datasetPath, ['LS_', num2str(snr_db), 'dB']), ...
            fullfile(datasetPath, ['LS_SNR_', num2str(snr_db), 'dB']), ...
            fullfile(datasetPath, ['SNR_', num2str(snr_db), 'dB']), ...
            fullfile(datasetPath, [num2str(snr_db), 'dB']), ...
            fullfile(datasetPath, ['LS_+', num2str(snr_db), 'dB']), ...
            fullfile(datasetPath, ['+', num2str(snr_db), 'dB'])
        };
    else
        candidates = {
            fullfile(datasetPath, 'LS_10dB'), fullfile(datasetPath, 'LS_0dB'), ...
            fullfile(datasetPath, 'SNR_10dB'), fullfile(datasetPath, '10dB'), ...
            fullfile(datasetPath, 'SNR_0dB'), fullfile(datasetPath, '0dB'), ...
            datasetPath
        };
    end

    for i = 1:length(candidates)
        fDir = candidates{i};
        if exist(fDir, 'dir')
            matFiles = {
                fullfile(fDir, 'inferredChannel.mat'), ...
                fullfile(fDir, 'matlabNTN.mat'), ...
                fullfile(fDir, 'channel_dur_randomizedUE.mat')
            };
            for j = 1:length(matFiles)
                if exist(matFiles{j}, 'file')
                    matData = load(matFiles{j});
                    return;
                end
            end
            % Fallback search any .mat file
            d = dir(fullfile(fDir, '*.mat'));
            if ~isempty(d)
                matData = load(fullfile(fDir, d(1).name));
                return;
            end
        end
    end

    % Direct search in datasetPath if not found in candidate subfolders
    if exist(datasetPath, 'dir')
        matFiles = {
            fullfile(datasetPath, 'inferredChannel.mat'), ...
            fullfile(datasetPath, 'matlabNTN.mat'), ...
            fullfile(datasetPath, 'channel_dur_randomizedUE.mat')
        };
        for j = 1:length(matFiles)
            if exist(matFiles{j}, 'file')
                matData = load(matFiles{j});
                return;
            end
        end
    end
end

function [amplitude_spectrum, phase_spectrum] = F_extract_DD(channel_grid)
% F_EXTRACT Converts Time-Freq channel to Delay-Doppler domain
% Input:  channel_grid: 132 (Subcarriers) x 14 (Symbols) Complex Matrix
% Output: amplitude_spectrum, phase_spectrum
    % 1. Frequency -> Delay (IFFT along dimension 1)
    grid_delay = ifft(channel_grid, [], 1);
    
    % 2. Time -> Doppler (FFT along dimension 2)
    grid_delay_doppler_raw = fft(grid_delay, [], 2);
    
    % 3. Shift both dimensions to center (Delay=0, Doppler=0 in middle)
    grid_dd_shifted = fftshift(grid_delay_doppler_raw);
    
    % 4. Extract Amplitude and Phase
    amplitude_spectrum = abs(grid_dd_shifted);
    phase_spectrum = angle(grid_dd_shifted);
end

function [mixed_img] = fda_mix_pixels(source_img, target_img, win_h_px, win_w_px)
% fda_mix_pixels blends target amplitude into center of source amplitude
    [h, w] = size(source_img);
    cy = floor(h / 2) + 1;
    cx = floor(w / 2) + 1;   

    r_h = floor(win_h_px / 2);
    r_w = floor(win_w_px / 2);

    mask = zeros(h, w);
    y_range = (cy - r_h) : (cy + r_h);
    x_range = (cx - r_w) : (cx + r_w);
    mask(y_range, x_range) = 1;
    
    mixed_img = (mask .* target_img) + ((1 - mask) .* source_img); 
end

function [tf_grid] = F_inverse_DD(complex_dd)
% F_INVERSE_DD Converts Delay-Doppler grid back to Time-Freq grid
    grid_unshifted = ifftshift(complex_dd);
    grid_delay_time = ifft(grid_unshifted, [], 2);
    tf_grid = fft(grid_delay_time, [], 1);
end

function translate_img = FTranslate_single(source_img, target_img, win_h_px, win_w_px)
    [source_amplitude_spectrum, source_phase_spectrum] = F_extract_DD(source_img);
    [target_amplitude_spectrum, ~] = F_extract_DD(target_img);

    translate_img_amp = fda_mix_pixels(source_amplitude_spectrum, target_amplitude_spectrum, win_h_px, win_w_px);
    translate_img_DD = translate_img_amp .* exp(1i * source_phase_spectrum);
    translate_img = F_inverse_DD(translate_img_DD);
end

function img_slice = crop_(img_slice, row_range, col_range)
% Clamps extrapolated boundary elements to min/max of inner pilot region
    ref_region = img_slice(row_range(1):row_range(2), col_range(1):col_range(2));
    
    r_min = min(real(ref_region), [], 'all');
    r_max = max(real(ref_region), [], 'all');
    i_min = min(imag(ref_region), [], 'all');
    i_max = max(imag(ref_region), [], 'all');
    
    [rows, cols] = size(img_slice);
    is_outside = true(rows, cols);
    is_outside(row_range(1):row_range(2), col_range(1):col_range(2)) = false;
    
    R = real(img_slice(is_outside));
    I = imag(img_slice(is_outside));
    
    R = max(min(R, r_max), r_min);
    I = max(min(I, i_max), i_min);
    
    img_slice(is_outside) = complex(R, I);
end

function translate_img = FTranslate_bulk(source_img, target_img, win_h_px, win_w_px, preprocessing_scale)
    if nargin < 5 || isempty(preprocessing_scale)
        preprocessing_scale = "none";
    end

    source_img = permute(source_img, [2, 1, 3]); % 132 x 14 x N
    target_img = permute(target_img, [2, 1, 3]); % 132 x 14 x N
    translate_img = zeros(size(target_img));

    for n = 1:size(target_img, 3)
        src_n = source_img(:, :, n);
        tgt_n = target_img(:, :, n);

        if strcmpi(preprocessing_scale, "rms")
            p_src = sqrt(mean(abs(src_n(:)).^2));
            p_tgt = sqrt(mean(abs(tgt_n(:)).^2));

            if p_src > 0
                src_n_norm = src_n / p_src;
            else
                src_n_norm = src_n;
            end

            if p_tgt > 0
                tgt_n_norm = tgt_n / p_tgt;
            else
                tgt_n_norm = tgt_n;
            end

            % FDA on normalized channels (both have RMS = 1.0)
            pseudo_norm = FTranslate_single(src_n_norm, tgt_n_norm, win_h_px, win_w_px);

            % Descale with Target RMS power to preserve Target physical power level
            translate_img(:, :, n) = pseudo_norm * p_tgt;

        elseif strcmpi(preprocessing_scale, "minmax") || strcmpi(preprocessing_scale, "min-max")
            % Min-Max scaling to [-1, 1] on Real and Imag parts separately
            r_min_src = min(real(src_n), [], 'all'); r_max_src = max(real(src_n), [], 'all');
            i_min_src = min(imag(src_n), [], 'all'); i_max_src = max(imag(src_n), [], 'all');
            dr_src = r_max_src - r_min_src; if dr_src == 0, dr_src = 1; end
            di_src = i_max_src - i_min_src; if di_src == 0, di_src = 1; end

            r_min_tgt = min(real(tgt_n), [], 'all'); r_max_tgt = max(real(tgt_n), [], 'all');
            i_min_tgt = min(imag(tgt_n), [], 'all'); i_max_tgt = max(imag(tgt_n), [], 'all');
            dr_tgt = r_max_tgt - r_min_tgt; if dr_tgt == 0, dr_tgt = 1; end
            di_tgt = i_max_tgt - i_min_tgt; if di_tgt == 0, di_tgt = 1; end

            src_norm_r = 2 * (real(src_n) - r_min_src) / dr_src - 1;
            src_norm_i = 2 * (imag(src_n) - i_min_src) / di_src - 1;
            src_n_norm = complex(src_norm_r, src_norm_i);

            tgt_norm_r = 2 * (real(tgt_n) - r_min_tgt) / dr_tgt - 1;
            tgt_norm_i = 2 * (imag(tgt_n) - i_min_tgt) / di_tgt - 1;
            tgt_n_norm = complex(tgt_norm_r, tgt_norm_i);

            % FDA on normalized channels in [-1, 1]
            pseudo_norm = FTranslate_single(src_n_norm, tgt_n_norm, win_h_px, win_w_px);

            % Descale with Target min/max range
            pseudo_r = (real(pseudo_norm) + 1) / 2 * dr_tgt + r_min_tgt;
            pseudo_i = (imag(pseudo_norm) + 1) / 2 * di_tgt + i_min_tgt;
            translate_img(:, :, n) = complex(pseudo_r, pseudo_i);
        else
            translate_img(:, :, n) = FTranslate_single(src_n, tgt_n, win_h_px, win_w_px);
        end
    end

    translate_img = permute(translate_img, [2, 1, 3]); % 14 x 132 x N
end

function s = compute_complex_ssim(h_true, h_est)
% COMPUTE_COMPLEX_SSIM Evaluates complex SSIM between true and estimated channel
% Evaluates SSIM on real and imaginary parts independently and averages them,
% matching the OpenNTN formulation.
    s_r = ssim_real_val(real(h_true), real(h_est));
    s_i = ssim_real_val(imag(h_true), imag(h_est));
    s = (s_r + s_i) / 2.0;
end

function s = ssim_real_val(x, y)
% SSIM_REAL_VAL 1D/2D SSIM for real arrays
    x = double(x(:));
    y = double(y(:));

    mu_x = mean(x);
    mu_y = mean(y);
    var_x = var(x, 1); % population variance
    var_y = var(y, 1);
    cov_xy = mean((x - mu_x) .* (y - mu_y));

    val_max = max(max(x), max(y));
    val_min = min(min(x), min(y));
    L = val_max - val_min;
    if L == 0
        L = 1.0;
    end

    C1 = (0.01 * L)^2;
    C2 = (0.03 * L)^2;

    num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2);
    den = (mu_x^2 + mu_y^2 + C1) * (var_x + var_y + C2);
    s = num / den;
end

function plot_channel_comparisons(save_folder, src_H_perf, tgt_H_infer, pseudo_H_perf, pseudo_H_li, snr_db, w_w, idx_src_map, idx_tgt_map)
% PLOT_CHANNEL_COMPARISONS Generates 4 pairs of figures (Magnitude & Real) per SNR subfolder.
% Each figure contains 4 subplots:
%   1. Source H_perfect (Original clean source ground truth)
%   2. Target H_LS_infer (Target inferred channel)
%   3. Pseudo H_perfect (Pseudo ground truth from FDA translation)
%   4. Pseudo H_li (Pseudo observation after transmission, noise & interpolation)

    nSamples = size(pseudo_H_perf, 3);
    if nSamples >= 4
        sample_indices = round(linspace(1, min(nSamples, 100), 4));
    else
        sample_indices = 1:nSamples;
    end

    for k = 1:length(sample_indices)
        idx = sample_indices(k);
        src_orig_idx = idx_src_map(idx);
        tgt_orig_idx = idx_tgt_map(idx);

        % Transpose from (14 symbols x 132 subcarriers) to (132 subcarriers x 14 symbols)
        % so Y-axis represents subcarrier and X-axis represents OFDM symbol
        h_src_perf    = src_H_perf(:, :, idx).';
        h_tgt_infer   = tgt_H_infer(:, :, idx).';
        h_pseudo_perf = pseudo_H_perf(:, :, idx).';
        h_pseudo_li   = pseudo_H_li(:, :, idx).';

        % --- Pair Part 1: Magnitude Figure (4 subplots) ---
        fig_mag = figure('Visible', 'off', 'Position', [100, 100, 1000, 750]);

        subplot(2, 2, 1);
        imagesc(1:14, 1:132, abs(h_src_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Source H_{perfect} (Pseudo #%d, Src #%d)', idx, src_orig_idx), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(2, 2, 2);
        imagesc(1:14, 1:132, abs(h_tgt_infer));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Target H_{LS\\_infer} (Pseudo #%d, Tgt #%d)', idx, tgt_orig_idx), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(2, 2, 3);
        imagesc(1:14, 1:132, abs(h_pseudo_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{perfect} (FDA 13x%d, Sample %d)', w_w, idx), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(2, 2, 4);
        imagesc(1:14, 1:132, abs(h_pseudo_li));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{li} (SNR %d dB, Sample %d)', snr_db, idx), 'FontSize', 10, 'FontWeight', 'bold');

        sgtitle(sprintf('Channel Magnitude |H| - Pseudo #%d [Src #%d + Tgt #%d] (Window 13x%d, SNR %d dB)', ...
            idx, src_orig_idx, tgt_orig_idx, w_w, snr_db), 'FontSize', 12, 'FontWeight', 'bold');

        mag_file = fullfile(save_folder, sprintf('sample_%d_magnitude.png', idx));
        saveas(fig_mag, mag_file);
        close(fig_mag);

        % --- Pair Part 2: Real Part Figure (4 subplots) ---
        fig_real = figure('Visible', 'off', 'Position', [100, 100, 1000, 750]);

        subplot(2, 2, 1);
        imagesc(1:14, 1:132, real(h_src_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Source H_{perfect} (Pseudo #%d, Src #%d)', idx, src_orig_idx), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(2, 2, 2);
        imagesc(1:14, 1:132, real(h_tgt_infer));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Target H_{LS\\_infer} (Pseudo #%d, Tgt #%d)', idx, tgt_orig_idx), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(2, 2, 3);
        imagesc(1:14, 1:132, real(h_pseudo_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{perfect} (FDA 13x%d, Sample %d)', w_w, idx), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(2, 2, 4);
        imagesc(1:14, 1:132, real(h_pseudo_li));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{li} (SNR %d dB, Sample %d)', snr_db, idx), 'FontSize', 10, 'FontWeight', 'bold');

        sgtitle(sprintf('Channel Real Part Re(H) - Pseudo #%d [Src #%d + Tgt #%d] (Window 13x%d, SNR %d dB)', ...
            idx, src_orig_idx, tgt_orig_idx, w_w, snr_db), 'FontSize', 12, 'FontWeight', 'bold');

        real_file = fullfile(save_folder, sprintf('sample_%d_real.png', idx));
        saveas(fig_real, real_file);
        close(fig_real);
    end
    fprintf('  -> Saved 4 pairs of visualization figures (Magnitude & Real) in: %s\n', save_folder);
end
