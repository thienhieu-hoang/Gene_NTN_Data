% plot_inferred_channel_comparisons.m
% Visualizes channel comparisons between True Ground Truth (H_perfect), 
% Model Inferred Channel (H_LS_infer), and Linear Interpolation Baseline (H_li)
% for each SNR subfolder in the inferred_dataset.
%
% Generates matching style visualizations (sample_%d_magnitude.png and sample_%d_real.png)
% for representative samples [1, 34, 67, 100].

clear; clc; close all;

inferredRootFolder = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\inferred_dataset\A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz\LS_Attention_standardize";

if ~exist(inferredRootFolder, 'dir')
    error('Directory does not exist: %s', inferredRootFolder);
end

% Locate all LS_* subdirectories
subDirs = dir(fullfile(inferredRootFolder, 'LS_*'));
subDirs = subDirs([subDirs.isdir]);

if isempty(subDirs)
    error('No LS_* subdirectories found in: %s', inferredRootFolder);
end

fprintf('Found %d SNR subdirectories in %s\n', length(subDirs), inferredRootFolder);

for d = 1:length(subDirs)
    folderName = subDirs(d).name;
    folderPath = fullfile(inferredRootFolder, folderName);
    matFile = fullfile(folderPath, 'inferredChannel.mat');
    
    if ~exist(matFile, 'file')
        fprintf('[Skip] %s does not contain inferredChannel.mat\n', folderName);
        continue;
    end
    
    fprintf('--> Processing folder: %s ...\n', folderName);
    
    % Extract SNR value from folder name (e.g., 'LS_0dB', 'LS_-10dB')
    snr_match = regexp(folderName, 'LS_([+-]?\d+)dB', 'tokens');
    if ~isempty(snr_match)
        snr_val = str2double(snr_match{1}{1});
    else
        snr_val = NaN;
    end
    
    % Load required variables
    data = load(matFile, 'H_perfect', 'H_LS_infer', 'H_li');
    
    if ~isfield(data, 'H_perfect') || ~isfield(data, 'H_LS_infer')
        warning('Missing H_perfect or H_LS_infer in %s', matFile);
        continue;
    end
    
    H_perf_all  = double(data.H_perfect);
    H_infer_all = double(data.H_LS_infer);
    has_li = isfield(data, 'H_li');
    if has_li
        H_li_all = double(data.H_li);
    end
    
    % Standardize dimensions to: (14 symbols x 132 subcarriers x N)
    H_perf_all  = standardize_dims(H_perf_all);
    H_infer_all = standardize_dims(H_infer_all);
    if has_li
        H_li_all = standardize_dims(H_li_all);
    end
    
    nSamples = size(H_perf_all, 3);
    if nSamples >= 4
        sample_indices = round(linspace(1, min(nSamples, 100), 4));
    else
        sample_indices = 1:nSamples;
    end
    
    % Generate figures for each sample index
    for k = 1:length(sample_indices)
        idx = sample_indices(k);
        
        % Transpose to (132 subcarriers x 14 symbols)
        h_perf  = H_perf_all(:, :, idx).';
        h_infer = H_infer_all(:, :, idx).';
        h_diff  = abs(h_infer - h_perf);
        
        if has_li
            h_li = H_li_all(:, :, idx).';
        end
        
        % Evaluate sample-level NMSE
        sample_nmse = sum(abs(h_infer - h_perf).^2, 'all') / sum(abs(h_perf).^2, 'all');
        sample_nmse_db = 10 * log10(sample_nmse);
        
        % -------------------------------------------------------------
        % 1. Magnitude Figure (2x2 subplots matching pseudo style)
        % -------------------------------------------------------------
        fig_mag = figure('Visible', 'off', 'Position', [100, 100, 1000, 750]);
        
        subplot(2, 2, 1);
        imagesc(1:14, 1:132, abs(h_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Target H_{perfect} (Ground Truth, Sample #%d)', idx), 'FontSize', 10, 'FontWeight', 'bold');
        
        subplot(2, 2, 2);
        imagesc(1:14, 1:132, abs(h_infer));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Model H_{LS\\_infer} (Inferred, Sample #%d)', idx), 'FontSize', 10, 'FontWeight', 'bold');
        
        if has_li
            subplot(2, 2, 3);
            imagesc(1:14, 1:132, abs(h_li));
            colorbar;
            xlabel('OFDM Symbol'); ylabel('Subcarrier');
            title(sprintf('Baseline H_{li} (Linear Interpolation, Sample #%d)', idx), 'FontSize', 10, 'FontWeight', 'bold');
            
            subplot(2, 2, 4);
            imagesc(1:14, 1:132, h_diff);
            colorbar;
            xlabel('OFDM Symbol'); ylabel('Subcarrier');
            title(sprintf('Magnitude Error |H_{infer} - H_{perfect}| (NMSE: %.2f dB)', sample_nmse_db), 'FontSize', 10, 'FontWeight', 'bold');
        else
            subplot(2, 2, 3);
            imagesc(1:14, 1:132, h_diff);
            colorbar;
            xlabel('OFDM Symbol'); ylabel('Subcarrier');
            title(sprintf('Magnitude Error |H_{infer} - H_{perfect}| (NMSE: %.2f dB)', sample_nmse_db), 'FontSize', 10, 'FontWeight', 'bold');
        end
        
        if ~isnan(snr_val)
            sgtitle(sprintf('Inferred Channel Magnitude |H| - Sample #%d (SNR %d dB, NMSE %.2f dB)', ...
                idx, snr_val, sample_nmse_db), 'FontSize', 12, 'FontWeight', 'bold');
        else
            sgtitle(sprintf('Inferred Channel Magnitude |H| - %s - Sample #%d (NMSE %.2f dB)', ...
                folderName, idx, sample_nmse_db), 'FontSize', 12, 'FontWeight', 'bold');
        end
        
        mag_file = fullfile(folderPath, sprintf('sample_%d_magnitude.png', idx));
        saveas(fig_mag, mag_file);
        close(fig_mag);
        
        % -------------------------------------------------------------
        % 2. Real Part Figure (2x2 subplots matching pseudo style)
        % -------------------------------------------------------------
        fig_real = figure('Visible', 'off', 'Position', [100, 100, 1000, 750]);
        
        subplot(2, 2, 1);
        imagesc(1:14, 1:132, real(h_perf));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Target H_{perfect} (Ground Truth, Sample #%d)', idx), 'FontSize', 10, 'FontWeight', 'bold');
        
        subplot(2, 2, 2);
        imagesc(1:14, 1:132, real(h_infer));
        colorbar;
        xlabel('OFDM Symbol'); ylabel('Subcarrier');
        title(sprintf('Model H_{LS\\_infer} (Inferred, Sample #%d)', idx), 'FontSize', 10, 'FontWeight', 'bold');
        
        if has_li
            subplot(2, 2, 3);
            imagesc(1:14, 1:132, real(h_li));
            colorbar;
            xlabel('OFDM Symbol'); ylabel('Subcarrier');
            title(sprintf('Baseline H_{li} (Linear Interpolation, Sample #%d)', idx), 'FontSize', 10, 'FontWeight', 'bold');
            
            subplot(2, 2, 4);
            imagesc(1:14, 1:132, real(h_infer - h_perf));
            colorbar;
            xlabel('OFDM Symbol'); ylabel('Subcarrier');
            title(sprintf('Real Error Re(H_{infer} - H_{perfect}) (Sample #%d)', idx), 'FontSize', 10, 'FontWeight', 'bold');
        else
            subplot(2, 2, 3);
            imagesc(1:14, 1:132, real(h_infer - h_perf));
            colorbar;
            xlabel('OFDM Symbol'); ylabel('Subcarrier');
            title(sprintf('Real Error Re(H_{infer} - H_{perfect}) (Sample #%d)', idx), 'FontSize', 10, 'FontWeight', 'bold');
        end
        
        if ~isnan(snr_val)
            sgtitle(sprintf('Inferred Channel Real Part Re(H) - Sample #%d (SNR %d dB, NMSE %.2f dB)', ...
                idx, snr_val, sample_nmse_db), 'FontSize', 12, 'FontWeight', 'bold');
        else
            sgtitle(sprintf('Inferred Channel Real Part Re(H) - %s - Sample #%d (NMSE %.2f dB)', ...
                folderName, idx, sample_nmse_db), 'FontSize', 12, 'FontWeight', 'bold');
        end
        
        real_file = fullfile(folderPath, sprintf('sample_%d_real.png', idx));
        saveas(fig_real, real_file);
        close(fig_real);
    end
    
    fprintf('    Saved 4 pairs of figures (Magnitude & Real) in: %s\n', folderPath);
end

fprintf('\nDone generating all inferred channel comparison plots!\n');

%% Helper function for dimension standardization
function H_out = standardize_dims(H_in)
    if size(H_in, 1) == 14 && size(H_in, 2) == 132
        H_out = H_in;
    elseif size(H_in, 2) == 132 && size(H_in, 3) == 14
        H_out = permute(H_in, [3, 2, 1]); % (N, 132, 14) -> (14, 132, N)
    elseif size(H_in, 1) == 132 && size(H_in, 2) == 14
        H_out = permute(H_in, [2, 1, 3]); % (132, 14, N) -> (14, 132, N)
    elseif size(H_in, 2) == 14 && size(H_in, 3) == 132
        H_out = permute(H_in, [2, 3, 1]); % (N, 14, 132) -> (14, 132, N)
    else
        error('Unexpected dimensions: %s', mat2str(size(H_in)));
    end
end
