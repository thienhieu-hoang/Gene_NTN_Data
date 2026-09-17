%% Pseudo Dataset Generation for NTN Channel Estimation/Prediction (Version 2)
% =========================================================================
% WORKFLOW & PURPOSE:
% 1. Pseudo Ground Truth Generation (FDA):
%    - Translates target domain style (Delay-Doppler low-frequency components)
%      into clean source domain channels using Delay-Doppler Fourier Domain
%      Adaptation (FDA) with window size (w_h = 13 x w_w in [3, 5]).
%    - Produces the pseudo ground-truth channel grid: H_perfect (14 x 132 x N).
%
% 2. 5G NR Transmission, Noise Realization & Channel Estimation:
%    - Simulates pilot transmission across the generated pseudo channel grids.
%    - Injects complex AWGN calibrated to pilot signal power across target SNRs
%      (SNR_dB = -10:5:15 dB).
%    - Computes Least Squares (LS) channel estimates at pilot locations: H_ls_pilots.
%    - Performs 2D scattered linear interpolation and edge boundary cropping
%      to reconstruct the estimated channel grid: H_li (14 x 132 x N).
%
% 3. Metric Evaluation & Output Organization:
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
% ===========================================================================
SourceDatasetPath = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\generatedChannel_Results\A100_2p18e9_600km_70deg_30kHz";
TargetDatasetPath = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\Sionna\OpenNTN\channel_wGeometry\results\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps";

% Output directory
outputFolder = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\A100__DUR300";
outpuFolder  = outputFolder; % backward compatibility alias

SNR_dB = -10:5:15;
w_window = [3, 5];


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
    error('Could not load any target dataset file from %s', TargetDatasetPath);
end

if ~isfield(sample_target, 'pilot_rows') || ~isfield(sample_target, 'pilot_cols')
    error('Target dataset must contain "pilot_rows" and "pilot_cols" fields.');
end

pilot_rows = double(sample_target.pilot_rows(:)); % 1-based subcarrier indices (1 to 132) as column vector
pilot_cols = double(sample_target.pilot_cols(:)); % 1-based OFDM symbol indices (1 to 14) as column vector
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
fprintf('Loaded Source H_perfect with size: %s\n\n', mat2str(size(source_domain.H_perfect)));

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

% Copy target note.md if available
tgt_note = fullfile(TargetDatasetPath, 'note.md');
if exist(tgt_note, 'file')
    copyfile(tgt_note, fullfile(outputFolder, 'note_target.md'));
    fprintf('Copied Target note.md -> %s\n', fullfile(outputFolder, 'note_target.md'));
end

% Generate note.md at output folder
pseudo_note_path = fullfile(outputFolder, 'note.md');
fid = fopen(pseudo_note_path, 'w');
if fid ~= -1
    fprintf(fid, '# Pseudo Dataset Overview: FDA Domain Adaptation\n\n');
    fprintf(fid, '## General Information\n\n');
    fprintf(fid, '- **Method:** Delay-Doppler Fourier Domain Adaptation (FDA) + 5G NR DM-RS Pilot & Noise Realization\n');
    fprintf(fid, '- **Source Dataset Path:** `%s`\n', SourceDatasetPath);
    fprintf(fid, '- **Target Dataset Path:** `%s`\n', TargetDatasetPath);
    fprintf(fid, '- **Result Folder:** `%s`\n', outputFolder);
    fprintf(fid, '- **Generation Date:** %s\n\n', string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
    fprintf(fid, '---\n\n');
    fprintf(fid, '## Configuration Summary\n\n');
    fprintf(fid, '| Parameter | Value |\n');
    fprintf(fid, '| :--- | :--- |\n');
    fprintf(fid, '| **Source Domain** | NTN TDL-A (NLOS, 70° elevation, 100 ns delay spread, 30 kHz SCS) |\n');
    fprintf(fid, '| **Target Domain** | OpenNTN DUR NLOS (30° elevation, 300 ns delay spread, 30 kHz SCS) |\n');
    fprintf(fid, '| **FDA Window ($13 \\times w_w$)** | %s |\n', mat2str(w_window));
    fprintf(fid, '| **SNR Range** | %s dB |\n', mat2str(SNR_dB));
    fprintf(fid, '| **Grid Dimensions** | 132 Subcarriers × 14 OFDM Symbols (11 RBs) |\n');
    fprintf(fid, '| **Pilot Configuration** | DM-RS Type 2 Port 1 (%d pilots per slot: symbols 3, 12; subcarriers 1-128) |\n', numPilots);
    fprintf(fid, '| **Saved Variables** | `H_perfect`, `H_li`, `H_ls_pilots`, `pilot_rows`, `pilot_cols`, `pilot_indices`, `nmse_li`, `nmse_ls_pilot`, `ssim_li`, `ssim_li_pilot`, `ssim_ls` |\n\n');
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
        if isempty(target_domain) || ~isfield(target_domain, 'H_li')
            warning('Target dataset H_li not found for SNR=%d dB. Skipping.', snr_db);
            continue;
        end

        % Align sample count across source and target
        nSamples = min(size(source_domain.H_perfect, 3), size(target_domain.H_li, 3));
        src_H_perf = source_domain.H_perfect(:, :, 1:nSamples);
        tgt_H_li   = target_domain.H_li(:, :, 1:nSamples);

        % Step A: Generate pseudo label (H_pseudo) by Fourier translation in Delay-Doppler domain
        % Translates target style (Doppler/delay) into source channel
        pseudo_label_li = FTranslate_bulk(src_H_perf, tgt_H_li, 13, w_w); % 14 x 132 x nSamples
        H_perfect = pseudo_label_li;                                     % Pseudo ground truth
        H_pseudo  = permute(pseudo_label_li, [2, 1, 3]);                 % 132 x 14 x nSamples

        % Initialize output arrays
        H_li = zeros(size(pseudo_label_li));
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

        % Step F: Plot and Save Channel Visualizations (4 pairs: Magnitude & Real)
        plot_channel_comparisons(save_folder, src_H_perf, tgt_H_li, H_perfect, H_li, snr_db, w_w);
    end
end

fprintf('\nAll pseudo dataset generations finished successfully!\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function matData = load_snr_dataset(datasetPath, snr_db)
    % Adaptively loads .mat file (matlabNTN.mat or channel_dur_randomizedUE.mat)
    % from either SNR_<snr>dB or <snr>dB subfolder
    matData = [];
    if nargin >= 2 && ~isempty(snr_db)
        candidates = {
            fullfile(datasetPath, ['SNR_', num2str(snr_db), 'dB']), ...
            fullfile(datasetPath, [num2str(snr_db), 'dB'])
        };
    else
        candidates = {
            fullfile(datasetPath, 'SNR_10dB'), fullfile(datasetPath, '10dB'), ...
            fullfile(datasetPath, 'SNR_0dB'), fullfile(datasetPath, '0dB'), ...
            datasetPath
        };
    end

    for i = 1:length(candidates)
        fDir = candidates{i};
        if exist(fDir, 'dir')
            matFiles = {
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

function translate_img = FTranslate_bulk(source_img, target_img, win_h_px, win_w_px)
    source_img = permute(source_img, [2, 1, 3]); % 132 x 14 x N
    target_img = permute(target_img, [2, 1, 3]); % 132 x 14 x N
    translate_img = zeros(size(target_img));

    for n = 1:size(target_img, 3)
        target_img_slice = crop_(target_img(:, :, n), [1, 128], [3, 12]);
        translate_img(:, :, n) = FTranslate_single(source_img(:, :, n), target_img_slice, win_h_px, win_w_px);
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

function plot_channel_comparisons(save_folder, src_H_perf, tgt_H_li, pseudo_H_perf, pseudo_H_li, snr_db, w_w)
% PLOT_CHANNEL_COMPARISONS Generates 4 pairs of figures (Magnitude & Real) per SNR subfolder.
% Each figure contains 4 subplots:
%   1. Source H_perfect (Original clean source ground truth)
%   2. Target H_li (Original target linearly interpolated channel)
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

        % Transpose from (14 symbols x 132 subcarriers) to (132 subcarriers x 14 symbols)
        % so Y-axis represents subcarrier and X-axis represents OFDM symbol
        h_src_perf    = src_H_perf(:, :, idx).';
        h_tgt_li      = tgt_H_li(:, :, idx).';
        h_pseudo_perf = pseudo_H_perf(:, :, idx).';
        h_pseudo_li   = pseudo_H_li(:, :, idx).';

        % --- Pair Part 1: Magnitude Figure (4 subplots) ---
        fig_mag = figure('Visible', 'off', 'Position', [100, 100, 1000, 750]);

        subplot(2, 2, 1);
        imagesc(1:14, 1:132, abs(h_src_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Source H_{perfect} (Sample %d)', idx), 'FontSize', 11, 'FontWeight', 'bold');

        subplot(2, 2, 2);
        imagesc(1:14, 1:132, abs(h_tgt_li));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Target H_{li} (Sample %d)', idx), 'FontSize', 11, 'FontWeight', 'bold');

        subplot(2, 2, 3);
        imagesc(1:14, 1:132, abs(h_pseudo_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{perfect} (FDA 13x%d, Sample %d)', w_w, idx), 'FontSize', 11, 'FontWeight', 'bold');

        subplot(2, 2, 4);
        imagesc(1:14, 1:132, abs(h_pseudo_li));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{li} (SNR %d dB, Sample %d)', snr_db, idx), 'FontSize', 11, 'FontWeight', 'bold');

        sgtitle(sprintf('Channel Magnitude |H| Comparison - Sample %d (Window 13x%d, SNR %d dB)', idx, w_w, snr_db), ...
            'FontSize', 13, 'FontWeight', 'bold');

        mag_file = fullfile(save_folder, sprintf('sample_%d_magnitude.png', idx));
        saveas(fig_mag, mag_file);
        close(fig_mag);

        % --- Pair Part 2: Real Part Figure (4 subplots) ---
        fig_real = figure('Visible', 'off', 'Position', [100, 100, 1000, 750]);

        subplot(2, 2, 1);
        imagesc(1:14, 1:132, real(h_src_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Source H_{perfect} (Sample %d)', idx), 'FontSize', 11, 'FontWeight', 'bold');

        subplot(2, 2, 2);
        imagesc(1:14, 1:132, real(h_tgt_li));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Target H_{li} (Sample %d)', idx), 'FontSize', 11, 'FontWeight', 'bold');

        subplot(2, 2, 3);
        imagesc(1:14, 1:132, real(h_pseudo_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{perfect} (FDA 13x%d, Sample %d)', w_w, idx), 'FontSize', 11, 'FontWeight', 'bold');

        subplot(2, 2, 4);
        imagesc(1:14, 1:132, real(h_pseudo_li));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Pseudo H_{li} (SNR %d dB, Sample %d)', snr_db, idx), 'FontSize', 11, 'FontWeight', 'bold');

        sgtitle(sprintf('Channel Real Part Re(H) Comparison - Sample %d (Window 13x%d, SNR %d dB)', idx, w_w, snr_db), ...
            'FontSize', 13, 'FontWeight', 'bold');

        real_file = fullfile(save_folder, sprintf('sample_%d_real.png', idx));
        saveas(fig_real, real_file);
        close(fig_real);
    end
    fprintf('  -> Saved 4 pairs of visualization figures (Magnitude & Real) in: %s\n', save_folder);
end
