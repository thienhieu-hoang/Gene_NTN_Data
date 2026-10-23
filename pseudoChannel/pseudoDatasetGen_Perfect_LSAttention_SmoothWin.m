% =========================================================================
% pseudoDatasetGen_Perfect_LSAttention_SmoothWin.m
% =========================================================================
% Generates synthetic (pseudo) 5G NTN channel datasets using Delay-Doppler
% Fourier Domain Adaptation (FDA) with SMOOTH TAPERING WINDOWS.
%
% Output Organization:
%   Instead of writing to a separate root dataset, results are saved directly
%   inside the main dataset folder (e.g., A100Perfect__DUR300LSAttention_minmax)
%   in subfolders alongside the legacy 13x3 and 13x5 folders:
%     ├── 13x3/              <-- Legacy hard rectangular boxcar
%     ├── 13x5/              <-- Legacy hard rectangular boxcar
%     ├── 13x3_smooth1/      <-- Smooth Variant 1 (Tukey outward: [0.75, 0.25] / [0.50])
%     │    ├── window_info.md <-- Full window documentation, equations & matrix values
%     │    ├── SNR_-10dB/     <-- matlabNTN.mat + figures
%     │    └── ...
%     ├── 13x5_smooth1/
%     │    ├── window_info.md
%     │    └── ...
%     ├── 13x3_smooth2/      <-- Smooth Variant 2 (Optional wider gentle taper)
%     └── ...
%
% =========================================================================

clear; clc; close all;

%% ========================================================================
%% Configuration
%% ========================================================================

SourceDatasetPath = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\generatedChannel_Results\A100_2p18e9_600km_70deg_30kHz";
TargetDatasetPath = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\inferred_dataset\A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz\LS_Attention_standardize";

% Base output directory (same parent directory as legacy 13x3 and 13x5)
baseOutputFolder = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\A100Perfect__DUR300LSAttention";

SNR_dB   = -10:5:15;
w_window = [3, 5];

% Desired number of samples in generated pseudo dataset ([] = auto min(N_src, N_tgt))
num_pseudo_samples = [];

% Random seed for reproducible sample pairing across SNRs
rng_seed = 42;

% Preprocessing scaling mode ("minmax", "rms", "standardize", or "none")
preprocessing_scale = "minmax";

% -------------------------------------------------------------------------
% Smooth Window Variant Definitions
% -------------------------------------------------------------------------
smooth_configs = struct();

% --- Variant 1: "smooth1" (Standard Outward Tukey - Compact 1-element Doppler taper) ---
smooth_configs(1).tag         = "smooth1";
smooth_configs(1).type        = "tukey_outward";
smooth_configs(1).taper_h     = 2; % 2 Delay transition rows -> [0.75, 0.25]
smooth_configs(1).taper_w     = 1; % 1 Doppler transition col  -> [0.50]
smooth_configs(1).name        = "Tukey Outward (Compact Roll-Off)";
smooth_configs(1).description = "Keeps core 13xWw flat at 1.0 (100% Target). Adds 1-element Doppler taper [0.50] and 2-element Delay taper [0.75, 0.25].";

% --- Variant 2: "smooth2" (Gentle Outward Tukey - Wider 2-element Doppler taper) ---
smooth_configs(2).tag         = "smooth2";
smooth_configs(2).type        = "tukey_outward";
smooth_configs(2).taper_h     = 3; % 3 Delay transition rows -> [0.854, 0.500, 0.146]
smooth_configs(2).taper_w     = 2; % 2 Doppler transition cols -> [0.750, 0.250]
smooth_configs(2).name        = "Tukey Outward (Gentle / Wider Roll-Off)";
smooth_configs(2).description = "Keeps core 13xWw flat at 1.0. Adds wider gentle roll-off: 2 Doppler elements [0.75, 0.25] and 3 Delay elements [0.85, 0.50, 0.15].";

% --- Variant 3: "smooth3" (Flat-Top Super-Gaussian) ---
smooth_configs(3).tag         = "smooth3";
smooth_configs(3).type        = "supergaussian";
smooth_configs(3).taper_h     = 2;
smooth_configs(3).taper_w     = 1;
smooth_configs(3).name        = "Super-Gaussian (Flat-Top Ellipse, p=2)";
smooth_configs(3).description = "Elliptical flat-top super-Gaussian taper with continuous C^inf exponential roll-off.";

% Choose which smooth variants to generate in this run:
% (e.g. ["smooth1"], or ["smooth1", "smooth2"])
active_variants = ["smooth1"];

% Filter active configurations
all_tags = [smooth_configs.tag];
[~, active_idx] = ismember(active_variants, all_tags);
active_cfgs = smooth_configs(active_idx(active_idx > 0));

if isempty(active_cfgs)
    error('No matching configuration found for active_variants: %s', mat2str(active_variants));
end

% Construct output directory matching existing preprocessing tag (e.g. A100Perfect__DUR300LSAttention_minmax)
if isempty(preprocessing_scale) || strcmpi(preprocessing_scale, "none")
    outputFolder = baseOutputFolder;
else
    scale_tag = lower(strrep(preprocessing_scale, "-", ""));
    outputFolder = baseOutputFolder + "_" + scale_tag;
end

% Whether to clamp boundaries of target inferred channel
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

% Add path to MATLAB helper directory (contains Lin_Interpolate.m)
helper_path = fullfile(fileparts(script_dir), 'MATLAB', 'helper');
if exist(helper_path, 'dir')
    addpath(helper_path);
else
    addpath('..\MATLAB\helper\');
end

%% ========================================================================
%% 1. Load Target Pilot Configuration & Build txGrid
%% ========================================================================
fprintf('=================================================================\n');
fprintf('  FDA Pseudo Dataset Generation: Smooth Windowing\n');
fprintf('  Active Smooth Variants: %s\n', strjoin(active_variants, ', '));
fprintf('  Output Root: %s\n', outputFolder);
fprintf('  Scaling: %s\n', preprocessing_scale);
fprintf('=================================================================\n\n');

fprintf('--- 1. Loading Target Pilot Configuration ---\n');
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

pilot_rows = double(sample_target.pilot_rows(:));
pilot_cols = double(sample_target.pilot_cols(:));

% Safeguard for 0-based indexing
if min(pilot_rows) == 0, pilot_rows = pilot_rows + 1; end
if min(pilot_cols) == 0, pilot_cols = pilot_cols + 1; end
numPilots = length(pilot_rows);

% Map pilots with value 1 to pilot positions in txGrid (132 subcarriers x 14 symbols)
txGrid = zeros(132, 14);
pilot_indices = sub2ind(size(txGrid), pilot_rows, pilot_cols);
txGrid(pilot_indices) = 1;
pilot_symbols = txGrid(pilot_indices);

fprintf('Loaded %d pilot positions from Target dataset.\n', numPilots);
fprintf('Pilot subcarrier range: [%d, %d], symbol range: [%d, %d]\n\n', ...
    min(pilot_rows), max(pilot_rows), min(pilot_cols), max(pilot_cols));

%% ========================================================================
%% 2. Load Clean Source Ground Truth Channel
%% ========================================================================
fprintf('--- 2. Loading Clean Source Ground Truth Channel ---\n');
source_domain = load_snr_dataset(SourceDatasetPath, 10);
if isempty(source_domain)
    source_domain = load_snr_dataset(SourceDatasetPath, 0);
end
if isempty(source_domain) || ~isfield(source_domain, 'H_perfect')
    error('Could not load source ground truth (H_perfect) from %s', SourceDatasetPath);
end

src_H_perf_all = source_domain.H_perfect;
if size(src_H_perf_all, 1) == 14 && size(src_H_perf_all, 2) == 132
    % Already (14 x 132 x N)
elseif size(src_H_perf_all, 2) == 132 && size(src_H_perf_all, 3) == 14
    src_H_perf_all = permute(src_H_perf_all, [3, 2, 1]);
elseif size(src_H_perf_all, 1) == 132 && size(src_H_perf_all, 2) == 14
    src_H_perf_all = permute(src_H_perf_all, [2, 1, 3]);
end
N_src_total = size(src_H_perf_all, 3);
fprintf('Loaded Source H_perfect with size: %s (Total Samples: %d)\n\n', ...
    mat2str(size(src_H_perf_all)), N_src_total);

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% ========================================================================
%% 3. Main Generation Loop (Smooth Variants x Window Sizes x SNR_dB)
%% ========================================================================
for c_idx = 1:length(active_cfgs)
    cfg = active_cfgs(c_idx);
    fprintf('\n############################################################\n');
    fprintf('### Smooth Variant: %s (%s)\n', cfg.tag, cfg.name);
    fprintf('### Type: %s | Delay Taper: %d | Doppler Taper: %d\n', ...
        cfg.type, cfg.taper_h, cfg.taper_w);
    fprintf('############################################################\n');

    for w_w = w_window
        % Construct window folder name alongside 13x3 and 13x5 (e.g. 13x3_smooth1)
        win_folder_name = sprintf('13x%d_%s', w_w, cfg.tag);
        win_folder_path = fullfile(outputFolder, win_folder_name);
        if ~exist(win_folder_path, 'dir')
            mkdir(win_folder_path);
        end

        % Generate and save window_info.md inside this window folder
        write_window_info_md(win_folder_path, w_w, cfg, 132, 14);

        for snr_db = SNR_dB
            fprintf('\n------------------------------------------------------------\n');
            fprintf('Processing: %s | SNR: %d dB\n', win_folder_name, snr_db);
            fprintf('------------------------------------------------------------\n');

            % Load target domain channel for this SNR
            target_domain = load_snr_dataset(TargetDatasetPath, snr_db);
            if isempty(target_domain)
                warning('Target dataset for SNR=%d dB not found in %s. Skipping.', snr_db, TargetDatasetPath);
                continue;
            end

            % Extract target inferred channel
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
                warning('Target inferred channel not found for SNR=%d dB. Skipping.', snr_db);
                continue;
            end

            % Standardize target inferred channel dimensions to: (14 symbols x 132 subcarriers x N_tgt)
            if size(tgt_H_raw, 1) == 14 && size(tgt_H_raw, 2) == 132
                tgt_H_infer_all = tgt_H_raw;
            elseif size(tgt_H_raw, 2) == 132 && size(tgt_H_raw, 3) == 14
                tgt_H_infer_all = permute(tgt_H_raw, [3, 2, 1]);
            elseif size(tgt_H_raw, 1) == 132 && size(tgt_H_raw, 2) == 14
                tgt_H_infer_all = permute(tgt_H_raw, [2, 1, 3]);
            elseif size(tgt_H_raw, 2) == 14 && size(tgt_H_raw, 3) == 132
                tgt_H_infer_all = permute(tgt_H_raw, [2, 3, 1]);
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

            if ~isempty(rng_seed)
                rng(rng_seed);
            end

            N_min = min(N_src, N_tgt);
            N_max = max(N_src, N_tgt);

            idx_map_source = zeros(1, nSamples);
            idx_map_target = zeros(1, nSamples);

            n_p1 = min(nSamples, N_min);
            idx_map_source(1:n_p1) = 1:n_p1;
            idx_map_target(1:n_p1) = 1:n_p1;

            if nSamples > N_min
                n_p2_end = min(nSamples, N_max);
                p2_range = (N_min + 1) : n_p2_end;
                len_p2   = length(p2_range);
                if N_src >= N_tgt
                    idx_map_source(p2_range) = p2_range;
                    idx_map_target(p2_range) = randi(N_tgt, 1, len_p2);
                else
                    idx_map_source(p2_range) = randi(N_src, 1, len_p2);
                    idx_map_target(p2_range) = p2_range;
                end
            end

            if nSamples > N_max
                p3_range = (N_max + 1) : nSamples;
                len_p3   = length(p3_range);
                idx_map_source(p3_range) = randi(N_src, 1, len_p3);
                idx_map_target(p3_range) = randi(N_tgt, 1, len_p3);
            end

            fprintf('Dataset Alignment: Source (%d) | Target (%d) -> Generating %d pseudo samples\n', ...
                N_src, N_tgt, nSamples);

            src_H_perf  = src_H_perf_all(:, :, idx_map_source);
            tgt_H_infer = tgt_H_infer_all(:, :, idx_map_target);

            if apply_crop_target
                for n = 1:nSamples
                    tgt_H_infer(:, :, n) = crop_(tgt_H_infer(:, :, n), ...
                        [min(pilot_cols), max(pilot_cols)], ...
                        [min(pilot_rows), max(pilot_rows)]);
                end
            end

            % -------------------------------------------------------------
            % Step A: FDA Translation with SMOOTH WINDOW
            % -------------------------------------------------------------
            pseudo_label = FTranslate_bulk(src_H_perf, tgt_H_infer, 13, w_w, ...
                preprocessing_scale, cfg.type, cfg.taper_h, cfg.taper_w);
            H_perfect = pseudo_label;                         % Pseudo ground truth (14 x 132 x nSamples)
            H_pseudo  = permute(pseudo_label, [2, 1, 3]);     % (132 x 14 x nSamples)

            % Initialize output arrays
            H_li = zeros(size(pseudo_label));
            H_ls_pilots = zeros(numPilots, nSamples);

            nmse_pseudo_li = 0;
            nmse_pseudo_ls = 0;
            ssim_pseudo_li = 0;
            ssim_pseudo_li_pilot = 0;
            ssim_pseudo_ls = 0;

            % -------------------------------------------------------------
            % Step B: Transmission, Noise Injection, LS & Linear Interpolation
            % -------------------------------------------------------------
            for n = 1:nSamples
                H_pseudo_n = H_pseudo(:, :, n); % 132 x 14

                % Transmit txGrid through pseudo channel
                Y_received_noNoise = txGrid .* H_pseudo_n;

                % Noise variance based on pilot signal power
                Ps = mean(abs(Y_received_noNoise(pilot_indices)).^2);
                sigma2 = Ps / (10^(snr_db / 10));

                % Complex AWGN
                Noise = sqrt(sigma2 / 2) * (randn(size(txGrid)) + 1i * randn(size(txGrid)));
                Y_received = Y_received_noNoise + Noise;

                % LS Estimation at pilots
                H_ls_pilots(:, n) = Y_received(pilot_indices);

                % Linear Interpolation across 2D grid
                [~, H_linear_n] = Lin_Interpolate(Y_received, pilot_indices, pilot_symbols);

                % Crop extrapolated edges beyond pilot boundaries
                H_linear_n_ = crop_(H_linear_n, [min(pilot_rows), max(pilot_rows)], [min(pilot_cols), max(pilot_cols)]);

                % Store interpolated channel (14 x 132)
                H_li(:, :, n) = permute(H_linear_n_, [2, 1, 3]);

                % NMSE metrics
                error_li = H_pseudo_n - H_linear_n_;
                true_power = sum(abs(H_pseudo_n).^2, 'all');
                nmse_pseudo_li = nmse_pseudo_li + (sum(abs(error_li).^2, 'all') / true_power);

                error_ls = H_ls_pilots(:, n) - H_pseudo_n(pilot_indices);
                pilot_power = sum(abs(H_pseudo_n(pilot_indices)).^2);
                nmse_pseudo_ls = nmse_pseudo_ls + (sum(abs(error_ls).^2) / pilot_power);

                % SSIM metrics
                ssim_pseudo_li = ssim_pseudo_li + compute_complex_ssim(H_pseudo_n, H_linear_n_);
                ssim_pseudo_li_pilot = ssim_pseudo_li_pilot + compute_complex_ssim(H_pseudo_n(pilot_indices), H_linear_n_(pilot_indices));
                ssim_pseudo_ls = ssim_pseudo_ls + compute_complex_ssim(H_pseudo_n(pilot_indices), H_ls_pilots(:, n));
            end

            nmse_li = nmse_pseudo_li / nSamples;
            nmse_ls_pilot = nmse_pseudo_ls / nSamples;
            ssim_li = ssim_pseudo_li / nSamples;
            ssim_li_pilot = ssim_pseudo_li_pilot / nSamples;
            ssim_ls = ssim_pseudo_ls / nSamples;
            ssim_ls_pilot = ssim_ls;

            % -------------------------------------------------------------
            % Step C: Save Dataset in SNR Subfolder
            % -------------------------------------------------------------
            save_folder = fullfile(win_folder_path, ['SNR_', num2str(snr_db), 'dB']);
            if ~exist(save_folder, 'dir')
                mkdir(save_folder);
            end

            output_mat = fullfile(save_folder, 'matlabNTN.mat');
            save(output_mat, ...
                'H_perfect', ...     % Ground truth channel: (14 x 132 x nSamples)
                'H_li', ...          % Linearly interpolated channel: (14 x 132 x nSamples)
                'H_ls_pilots', ...   % LS sequence at pilots: (numPilots x nSamples)
                'pilot_rows', ...    % 1-based subcarrier coordinates
                'pilot_cols', ...    % 1-based OFDM symbol coordinates
                'pilot_indices', ... % 1-based linear indices
                'idx_map_source', ...% Source sample mapping
                'idx_map_target', ...% Target sample mapping
                'nmse_li', ...       % Average NMSE of linear interpolation
                'nmse_ls_pilot', ... % Average NMSE of LS at pilots
                'ssim_li', ...       % Average SSIM of linear interpolation (full grid)
                'ssim_li_pilot', ... % Average SSIM of linear interpolation at pilots
                'ssim_ls', ...       % Average SSIM of LS at pilots
                'ssim_ls_pilot', ... % Average SSIM of LS at pilots (alias)
                '-v7.3');

            fprintf('Saved: %s\n', output_mat);
            fprintf('  -> NMSE (LI): %.4f (%.2f dB) | NMSE (LS pilots): %.4f (%.2f dB)\n', ...
                nmse_li, 10*log10(nmse_li), nmse_ls_pilot, 10*log10(nmse_ls_pilot));
            fprintf('  -> SSIM (LI): %.4f | SSIM (LI pilots): %.4f | SSIM (LS pilots): %.4f\n', ...
                ssim_li, ssim_li_pilot, ssim_ls);

            % Step D: Visualization figures
            plot_channel_comparisons(save_folder, src_H_perf, tgt_H_infer, H_perfect, H_li, ...
                snr_db, w_w, idx_map_source, idx_map_target, cfg.tag);
        end
    end
end

fprintf('\nAll smooth-window pseudo dataset generations finished successfully!\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function write_window_info_md(target_folder, win_w_px, cfg, grid_h, grid_w)
% WRITE_WINDOW_INFO_MD Generates window_info.md inside the window folder (e.g. 13x3_smooth1)
    if nargin < 4, grid_h = 132; end
    if nargin < 5, grid_w = 14; end
    win_h_px = 13;

    info_path = fullfile(target_folder, 'window_info.md');
    fid = fopen(info_path, 'w');
    if fid == -1
        warning('Could not write window_info.md at: %s', info_path);
        return;
    end

    [mask, w_y, w_x] = fda_create_smooth_window(grid_h, grid_w, win_h_px, win_w_px, ...
        cfg.type, cfg.taper_h, cfg.taper_w);

    cy = floor(grid_h / 2) + 1; % 67
    cx = floor(grid_w / 2) + 1; % 8
    r_h = floor(win_h_px / 2);  % 6
    r_w = floor(win_w_px / 2);

    fprintf(fid, '# FDA Smooth Window Information: `13x%d_%s`\n\n', win_w_px, cfg.tag);
    fprintf(fid, '- **Window Variant:** `%s` (%s)\n', cfg.tag, cfg.name);
    fprintf(fid, '- **Base Core Window Size:** $13 \\times %d$ ($13$ Delay bins $\\times %d$ Doppler bins)\n', win_h_px, win_w_px);
    fprintf(fid, '- **Window Type:** `%s`\n', cfg.type);
    fprintf(fid, '- **Delay Transition ($taper\\_h$):** `%d` rows\n', cfg.taper_h);
    fprintf(fid, '- **Doppler Transition ($taper\\_w$):** `%d` columns\n', cfg.taper_w);
    fprintf(fid, '- **Description:** %s\n', cfg.description);
    fprintf(fid, '- **Generated On:** %s\n\n', string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));

    fprintf(fid, '---\n\n');
    fprintf(fid, '## Rationale: Eliminating the Boundary "Outside Ring"\n\n');
    fprintf(fid, 'In standard FDA with a hard rectangular boxcar window ($13 \\times %d$), an abrupt jump discontinuity occurs between the Target and Source amplitude spectra at the window boundaries.\n', win_w_px);
    fprintf(fid, 'When transforming back to the Time-Frequency domain via 2D inverse Fourier transform (`F_inverse_DD`), this frequency step discontinuity convolves with sinc sidelobes, causing **Gibbs ringing** and a visible rectangular boundary "ring".\n\n');
    fprintf(fid, 'The `%s` window maintains the central core strictly at **$1.0$** (preserving 100%% of the target domain dominant delay-Doppler energy) while softly tapering outward to **$0.0$** over smooth cosine transitions.\n\n', cfg.tag);

    fprintf(fid, '---\n\n');
    fprintf(fid, '## Exact Profile Weights\n\n');

    % Doppler Profile Breakdown
    fprintf(fid, '### 1. Doppler Profile $w_x$ (OFDM Symbols, dimension %d, center column %d)\n\n', grid_w, cx);
    fprintf(fid, '| Column Index | Distance from DC | Weight $w_x$ | Region |\n');
    fprintf(fid, '| :---: | :---: | :---: | :--- |\n');
    for x = max(1, cx - r_w - cfg.taper_w - 1) : min(grid_w, cx + r_w + cfg.taper_w + 1)
        dist = abs(x - cx);
        val = w_x(x);
        if dist <= r_w
            reg = "**Core (100% Target)**";
        elseif val > 0
            reg = "*Smooth Transition*";
        else
            reg = "Outer (100% Source)";
        end
        fprintf(fid, '| **Col %d** | $\\vert x - %d \\vert = %d$ | `%.4f` | %s |\n', x, cx, dist, val, reg);
    end
    fprintf(fid, '\n');

    % Delay Profile Breakdown
    fprintf(fid, '### 2. Delay Profile $w_y$ (Subcarriers, dimension %d, center row %d)\n\n', grid_h, cy);
    fprintf(fid, '| Row Index | Distance from DC | Weight $w_y$ | Region |\n');
    fprintf(fid, '| :---: | :---: | :---: | :--- |\n');
    for y = (cy - r_h - cfg.taper_h - 1) : (cy + r_h + cfg.taper_h + 1)
        if y < 1 || y > grid_h, continue; end
        dist = abs(y - cy);
        val = w_y(y);
        if dist <= r_h
            reg = "**Core (100% Target)**";
        elseif val > 0
            reg = "*Smooth Transition*";
        else
            reg = "Outer (100% Source)";
        end
        fprintf(fid, '| **Row %d** | $\\vert y - %d \\vert = %d$ | `%.4f` | %s |\n', y, cy, dist, val, reg);
    end
    fprintf(fid, '\n');

    % 2D Visual Slice around Center DC
    fprintf(fid, '### 3. Central 2D Mask Slice ($M(\\tau, \\nu)$ around DC)\n\n');
    fprintf(fid, 'Values represent the fraction of Target amplitude spectrum ($1.0$ = 100%% Target, $0.0$ = 100%% Source):\n\n');

    x_disp = max(1, cx - r_w - cfg.taper_w - 1) : min(grid_w, cx + r_w + cfg.taper_w + 1);
    y_disp = (cy - r_h - cfg.taper_h - 1) : (cy + r_h + cfg.taper_h + 1);

    % Markdown table header
    hdr = '| Row \\ Col |';
    sep = '| :---: |';
    for x = x_disp
        hdr = [hdr, sprintf(' **C%d** |', x)];
        sep = [sep, ' :---: |'];
    end
    fprintf(fid, '%s\n%s\n', hdr, sep);

    for y = y_disp
        row_str = sprintf('| **R%d** |', y);
        for x = x_disp
            v = mask(y, x);
            if v == 1.0
                row_str = [row_str, sprintf(' **%.2f** |', v)];
            elseif v > 0
                row_str = [row_str, sprintf(' *%.2f* |', v)];
            else
                row_str = [row_str, sprintf(' `%.2f` |', v)];
            end
        end
        fprintf(fid, '%s\n', row_str);
    end
    fprintf(fid, '\n');

    fprintf(fid, '---\n\n');
    fprintf(fid, '## Mathematical Formula\n\n');
    if strcmpi(cfg.type, "tukey_outward") || strcmpi(cfg.type, "tukey")
        fprintf(fid, '$$\\text{For outward step } k = d - r_{\\text{core}} \\in \\{1, \\dots, \\text{taper}\\}:\\quad w(k) = \\frac{1}{2} \\left(1 + \\cos\\left(\\frac{k \\pi}{\\text{taper} + 1}\\right)\\right)$$\n\n');
        fprintf(fid, '- Along Doppler ($taper\\_w = 1$): $k=1 \\implies \\frac{1}{2}(1 + \\cos(\\pi/2)) = \\mathbf{0.50}$.\n');
        fprintf(fid, '- Along Delay ($taper\\_h = 2$): $k=1 \\implies \\frac{1}{2}(1 + \\cos(\\pi/3)) = \\mathbf{0.75}$, $k=2 \\implies \\frac{1}{2}(1 + \\cos(2\\pi/3)) = \\mathbf{0.25}$.\n');
    elseif strcmpi(cfg.type, "supergaussian")
        fprintf(fid, '$$M(y, x) = \\exp\\left( - \\left[ \\left(\\frac{y - c_y}{r_h}\\right)^2 + \\left(\\frac{x - c_x}{r_w}\\right)^2 \\right]^2 \\right)$$\n\n');
    end

    fclose(fid);
    fprintf('  -> Wrote %s\n', info_path);
end

function [mask, w_y, w_x] = fda_create_smooth_window(grid_h, grid_w, win_h_px, win_w_px, window_type, taper_h, taper_w)
    if nargin < 5 || isempty(window_type), window_type = 'tukey_outward'; end
    if nargin < 6 || isempty(taper_h), taper_h = 2; end
    if nargin < 7 || isempty(taper_w), taper_w = 1; end

    cy = floor(grid_h / 2) + 1;
    cx = floor(grid_w / 2) + 1;

    r_h = floor(win_h_px / 2);
    r_w = floor(win_w_px / 2);

    window_type = lower(string(window_type));

    switch window_type
        case {"tukey_outward", "tukey", "raised_cosine"}
            w_y = zeros(grid_h, 1);
            for y = 1:grid_h
                d = abs(y - cy);
                if d <= r_h
                    w_y(y) = 1.0;
                elseif d <= (r_h + taper_h)
                    k = d - r_h;
                    w_y(y) = 0.5 * (1 + cos(pi * k / (taper_h + 1)));
                end
            end

            w_x = zeros(1, grid_w);
            for x = 1:grid_w
                d = abs(x - cx);
                if d <= r_w
                    w_x(x) = 1.0;
                elseif d <= (r_w + taper_w)
                    k = d - r_w;
                    w_x(x) = 0.5 * (1 + cos(pi * k / (taper_w + 1)));
                end
            end

            mask = w_y * w_x;

        case "tukey_inward"
            t_h = min(taper_h, max(1, r_h - 1));
            t_w = min(taper_w, max(1, r_w - 1));

            w_y = zeros(grid_h, 1);
            for y = 1:grid_h
                d = abs(y - cy);
                if d <= (r_h - t_h)
                    w_y(y) = 1.0;
                elseif d <= r_h
                    k = d - (r_h - t_h);
                    w_y(y) = 0.5 * (1 + cos(pi * k / (t_h + 1)));
                end
            end

            w_x = zeros(1, grid_w);
            for x = 1:grid_w
                d = abs(x - cx);
                if d <= (r_w - t_w)
                    w_x(x) = 1.0;
                elseif d <= r_w
                    k = d - (r_w - t_w);
                    w_x(x) = 0.5 * (1 + cos(pi * k / (t_w + 1)));
                end
            end

            mask = w_y * w_x;

        case {"supergaussian", "gaussian"}
            [X, Y] = meshgrid(1:grid_w, 1:grid_h);
            norm_dist = ((Y - cy) ./ max(1, r_h)).^2 + ((X - cx) ./ max(1, r_w)).^2;
            mask = exp(- (norm_dist .^ 2));
            mask(mask < 0.01) = 0;
            w_y = mask(:, cx);
            w_x = mask(cy, :);

        case "hard"
            mask = zeros(grid_h, grid_w);
            y_range = max(1, cy - r_h) : min(grid_h, cy + r_h);
            x_range = max(1, cx - r_w) : min(grid_w, cx + r_w);
            mask(y_range, x_range) = 1;
            w_y = zeros(grid_h, 1);
            w_y(y_range) = 1.0;
            w_x = zeros(1, grid_w);
            w_x(x_range) = 1.0;

        otherwise
            error('Unsupported window_type: "%s"', window_type);
    end
end

function [mixed_img] = fda_mix_pixels(source_img, target_img, win_h_px, win_w_px, window_type, taper_h, taper_w)
    if nargin < 5 || isempty(window_type), window_type = 'tukey_outward'; end
    if nargin < 6 || isempty(taper_h), taper_h = 2; end
    if nargin < 7 || isempty(taper_w), taper_w = 1; end

    [h, w] = size(source_img);
    mask = fda_create_smooth_window(h, w, win_h_px, win_w_px, window_type, taper_h, taper_w);
    mixed_img = (mask .* target_img) + ((1 - mask) .* source_img);
end

function [amplitude_spectrum, phase_spectrum] = F_extract_DD(channel_grid)
    grid_delay = ifft(channel_grid, [], 1);
    grid_delay_doppler_raw = fft(grid_delay, [], 2);
    grid_dd_shifted = fftshift(grid_delay_doppler_raw);
    amplitude_spectrum = abs(grid_dd_shifted);
    phase_spectrum = angle(grid_dd_shifted);
end

function [tf_grid] = F_inverse_DD(complex_dd)
    grid_unshifted = ifftshift(complex_dd);
    grid_delay_time = ifft(grid_unshifted, [], 2);
    tf_grid = fft(grid_delay_time, [], 1);
end

function translate_img = FTranslate_single(source_img, target_img, win_h_px, win_w_px, window_type, taper_h, taper_w)
    if nargin < 5, window_type = 'tukey_outward'; end
    if nargin < 6, taper_h = 2; end
    if nargin < 7, taper_w = 1; end

    [source_amplitude_spectrum, source_phase_spectrum] = F_extract_DD(source_img);
    [target_amplitude_spectrum, ~] = F_extract_DD(target_img);

    translate_img_amp = fda_mix_pixels(source_amplitude_spectrum, target_amplitude_spectrum, ...
        win_h_px, win_w_px, window_type, taper_h, taper_w);
    translate_img_DD = translate_img_amp .* exp(1i * source_phase_spectrum);
    translate_img = F_inverse_DD(translate_img_DD);
end

function translate_img = FTranslate_bulk(source_img, target_img, win_h_px, win_w_px, ...
    preprocessing_scale, window_type, taper_h, taper_w)
    if nargin < 5 || isempty(preprocessing_scale), preprocessing_scale = "none"; end
    if nargin < 6 || isempty(window_type), window_type = "tukey_outward"; end
    if nargin < 7 || isempty(taper_h), taper_h = 2; end
    if nargin < 8 || isempty(taper_w), taper_w = 1; end

    source_img = permute(source_img, [2, 1, 3]); % 132 x 14 x N
    target_img = permute(target_img, [2, 1, 3]); % 132 x 14 x N
    translate_img = zeros(size(target_img));

    for n = 1:size(target_img, 3)
        src_n = source_img(:, :, n);
        tgt_n = target_img(:, :, n);

        if strcmpi(preprocessing_scale, "rms")
            p_src = sqrt(mean(abs(src_n(:)).^2));
            p_tgt = sqrt(mean(abs(tgt_n(:)).^2));

            src_n_norm = src_n / max(p_src, eps);
            tgt_n_norm = tgt_n / max(p_tgt, eps);

            pseudo_norm = FTranslate_single(src_n_norm, tgt_n_norm, win_h_px, win_w_px, ...
                window_type, taper_h, taper_w);
            translate_img(:, :, n) = pseudo_norm * p_tgt;

        elseif strcmpi(preprocessing_scale, "minmax") || strcmpi(preprocessing_scale, "min-max")
            r_min_src = min(real(src_n), [], 'all'); r_max_src = max(real(src_n), [], 'all');
            i_min_src = min(imag(src_n), [], 'all'); i_max_src = max(imag(src_n), [], 'all');
            dr_src = max(r_max_src - r_min_src, eps);
            di_src = max(i_max_src - i_min_src, eps);

            r_min_tgt = min(real(tgt_n), [], 'all'); r_max_tgt = max(real(tgt_n), [], 'all');
            i_min_tgt = min(imag(tgt_n), [], 'all'); i_max_tgt = max(imag(tgt_n), [], 'all');
            dr_tgt = max(r_max_tgt - r_min_tgt, eps);
            di_tgt = max(i_max_tgt - i_min_tgt, eps);

            src_norm_r = 2 * (real(src_n) - r_min_src) / dr_src - 1;
            src_norm_i = 2 * (imag(src_n) - i_min_src) / di_src - 1;
            src_n_norm = complex(src_norm_r, src_norm_i);

            tgt_norm_r = 2 * (real(tgt_n) - r_min_tgt) / dr_tgt - 1;
            tgt_norm_i = 2 * (imag(tgt_n) - i_min_tgt) / di_tgt - 1;
            tgt_n_norm = complex(tgt_norm_r, tgt_norm_i);

            pseudo_norm = FTranslate_single(src_n_norm, tgt_n_norm, win_h_px, win_w_px, ...
                window_type, taper_h, taper_w);

            pseudo_r = (real(pseudo_norm) + 1) / 2 * dr_tgt + r_min_tgt;
            pseudo_i = (imag(pseudo_norm) + 1) / 2 * di_tgt + i_min_tgt;
            translate_img(:, :, n) = complex(pseudo_r, pseudo_i);

        elseif strcmpi(preprocessing_scale, "standardize") || strcmpi(preprocessing_scale, "std")
            mu_r_src = mean(real(src_n), 'all'); std_r_src = max(std(real(src_n), 0, 'all'), eps);
            mu_i_src = mean(imag(src_n), 'all'); std_i_src = max(std(imag(src_n), 0, 'all'), eps);

            mu_r_tgt = mean(real(tgt_n), 'all'); std_r_tgt = max(std(real(tgt_n), 0, 'all'), eps);
            mu_i_tgt = mean(imag(tgt_n), 'all'); std_i_tgt = max(std(imag(tgt_n), 0, 'all'), eps);

            src_norm_r = (real(src_n) - mu_r_src) / std_r_src;
            src_norm_i = (imag(src_n) - mu_i_src) / std_i_src;
            src_n_norm = complex(src_norm_r, src_norm_i);

            tgt_norm_r = (real(tgt_n) - mu_r_tgt) / std_r_tgt;
            tgt_norm_i = (imag(tgt_n) - mu_i_tgt) / std_i_tgt;
            tgt_n_norm = complex(tgt_norm_r, tgt_norm_i);

            pseudo_norm = FTranslate_single(src_n_norm, tgt_n_norm, win_h_px, win_w_px, ...
                window_type, taper_h, taper_w);

            pseudo_r = real(pseudo_norm) * std_r_tgt + mu_r_tgt;
            pseudo_i = imag(pseudo_norm) * std_i_tgt + mu_i_tgt;
            translate_img(:, :, n) = complex(pseudo_r, pseudo_i);
        else
            translate_img(:, :, n) = FTranslate_single(src_n, tgt_n, win_h_px, win_w_px, ...
                window_type, taper_h, taper_w);
        end
    end

    translate_img = permute(translate_img, [2, 1, 3]); % 14 x 132 x N
end

function img_slice = crop_(img_slice, row_range, col_range)
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

function s = compute_complex_ssim(h_true, h_est)
    r_true = real(h_true); r_est = real(h_est);
    i_true = imag(h_true); i_est = imag(h_est);
    ssim_r = ssim_scalar(r_true, r_est);
    ssim_i = ssim_scalar(i_true, i_est);
    s = (ssim_r + ssim_i) / 2;
end

function val = ssim_scalar(x, y)
    x = double(x(:)); y = double(y(:));
    ux = mean(x); uy = mean(y);
    vx = var(x, 1); vy = var(y, 1);
    vxy = mean((x - ux) .* (y - uy));
    L = max(max(x) - min(x), max(y) - min(y));
    if L == 0, L = 1; end
    c1 = (0.01 * L)^2;
    c2 = (0.03 * L)^2;
    val = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux^2 + uy^2 + c1) * (vx + vy + c2));
end

function matData = load_snr_dataset(datasetPath, snr_db)
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
            d = dir(fullfile(fDir, '*.mat'));
            if ~isempty(d)
                matData = load(fullfile(fDir, d(1).name));
                return;
            end
        end
    end

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

function plot_channel_comparisons(save_folder, src_H_perf, tgt_H_infer, pseudo_H_perf, pseudo_H_li, ...
    snr_db, w_w, idx_src_map, idx_tgt_map, win_tag)
    if nargin < 10, win_tag = "smooth"; end

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

        h_src_perf    = src_H_perf(:, :, idx).';
        h_tgt_infer   = tgt_H_infer(:, :, idx).';
        h_pseudo_perf = pseudo_H_perf(:, :, idx).';
        h_pseudo_li   = pseudo_H_li(:, :, idx).';

        % --- Part 1: Magnitude Figure ---
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
        title(sprintf('Pseudo H_{perfect} (FDA 13x%d %s)', w_w, win_tag), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(2, 2, 4);
        imagesc(1:14, 1:132, abs(h_pseudo_li));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{li} (SNR %d dB, Sample %d)', snr_db, idx), 'FontSize', 10, 'FontWeight', 'bold');

        sgtitle(sprintf('Channel Magnitude |H| - Pseudo #%d (FDA 13x%d %s, SNR %d dB)', ...
            idx, w_w, win_tag, snr_db), 'FontSize', 12, 'FontWeight', 'bold');

        mag_file = fullfile(save_folder, sprintf('sample_%d_magnitude.png', idx));
        saveas(fig_mag, mag_file);
        close(fig_mag);

        % --- Part 2: Real Part Figure ---
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
        title(sprintf('Pseudo H_{perfect} (FDA 13x%d %s)', w_w, win_tag), 'FontSize', 10, 'FontWeight', 'bold');

        subplot(2, 2, 4);
        imagesc(1:14, 1:132, real(h_pseudo_li));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{li} (SNR %d dB, Sample %d)', snr_db, idx), 'FontSize', 10, 'FontWeight', 'bold');

        sgtitle(sprintf('Channel Real Part Re(H) - Pseudo #%d (FDA 13x%d %s, SNR %d dB)', ...
            idx, w_w, win_tag, snr_db), 'FontSize', 12, 'FontWeight', 'bold');

        real_file = fullfile(save_folder, sprintf('sample_%d_real.png', idx));
        saveas(fig_real, real_file);
        close(fig_real);
    end
    fprintf('  -> Saved 4 pairs of visualization figures (Magnitude & Real) in: %s\n', save_folder);
end
