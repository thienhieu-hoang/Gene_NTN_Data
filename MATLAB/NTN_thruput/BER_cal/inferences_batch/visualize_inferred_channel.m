%{
========================================================================================
                 Inferred Channel vs. Perfect Channel Visualization
========================================================================================
OVERVIEW:
  This script loads pre-inferred OFDM channel matrices from 'inferredChannel.mat'
  for a specified batch folder and SNR subfolder, adaptively extracts H_perfect and
  the inferred channel (H_LS_infer, H_LI_infer, or H_infer), and visualizes the
  real parts of the first 8 samples in 2 separate figures (2 rows x 4 columns each).
========================================================================================
%}

folder = 'DUR100__A100_2p18e9_600km_30kHz_LI_Grid';
SNR    = '10'; % SNR dB string (e.g., '10', '-5', 'SNR_10dB', or numeric 10)
num_samples = 8; % Number of samples to plot

if exist('mfilename', 'builtin') && ~isempty(mfilename('fullpath'))
    script_dir = fileparts(mfilename('fullpath'));
else
    script_dir = pwd;
end

% 1. Format SNR subfolder name
if isnumeric(SNR)
    snr_str = sprintf('%ddB', SNR);
else
    snr_str = char(SNR);
    if ~contains(snr_str, 'dB', 'IgnoreCase', true)
        snr_str = [snr_str 'dB'];
    end
end
if ~startsWith(snr_str, 'SNR_', 'IgnoreCase', true)
    snr_folder = ['SNR_' snr_str];
else
    snr_folder = snr_str;
end

% 2. Locate target inferredChannel.mat file
mat_path = fullfile(folder, snr_folder, 'inferredChannel.mat');
if ~exist(mat_path, 'file') && exist(fullfile(script_dir, folder, snr_folder, 'inferredChannel.mat'), 'file')
    mat_path = fullfile(script_dir, folder, snr_folder, 'inferredChannel.mat');
elseif ~exist(mat_path, 'file') && exist(fullfile(script_dir, '..', folder, snr_folder, 'inferredChannel.mat'), 'file')
    mat_path = fullfile(script_dir, '..', folder, snr_folder, 'inferredChannel.mat');
end

if ~exist(mat_path, 'file')
    error('File not found: %s\nPlease check folder name and SNR setting.', mat_path);
end

fprintf('Loading inferred channel file:\n  %s\n\n', mat_path);
data = load(mat_path);

% 3. Extract H_perfect
if isfield(data, 'H_perfect')
    H_perfect = double(data.H_perfect);
else
    error('Field H_perfect not found in: %s', mat_path);
end

% 4. Adaptively load inferred channel (H_LS_infer, H_ls_infer, H_LI_infer, H_li_infer, H_infer)
if isfield(data, 'H_LS_infer')
    H_infer = double(data.H_LS_infer);
    infer_label = 'H_{LS\_infer}';
elseif isfield(data, 'H_ls_infer')
    H_infer = double(data.H_ls_infer);
    infer_label = 'H_{ls\_infer}';
elseif isfield(data, 'H_LI_infer')
    H_infer = double(data.H_LI_infer);
    infer_label = 'H_{LI\_infer}';
elseif isfield(data, 'H_li_infer')
    H_infer = double(data.H_li_infer);
    infer_label = 'H_{li\_infer}';
elseif isfield(data, 'H_infer')
    H_infer = double(data.H_infer);
    infer_label = 'H_{infer}';
else
    error('No inferred channel field (H_LS_infer, H_LI_infer, or H_infer) found in:\n  %s', mat_path);
end

total_samples = size(H_perfect, 1);
N_plot = min(num_samples, total_samples);

fprintf('Loaded H_perfect shape: [%d x %d x %d]\n', size(H_perfect,1), size(H_perfect,2), size(H_perfect,3));
fprintf('Loaded %s shape:  [%d x %d x %d]\n', infer_label, size(H_infer,1), size(H_infer,2), size(H_infer,3));
fprintf('Plotting real parts of first %d sample(s) into 2 separate figures...\n\n', N_plot);

nRows = 2;
nCols = 4;

% =========================================================================
% FIGURE 1: Real Part of H_perfect (2 rows x 4 columns)
% =========================================================================
fig1 = figure('Name', sprintf('H_perfect Real Part (%s)', snr_folder), ...
    'Color', 'w', 'Position', [50, 50, 1400, 620]);

tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');

for s = 1:N_plot
    nexttile;
    grid_perfect = real(squeeze(H_perfect(s, :, :)));
    imagesc(grid_perfect);
    colorbar;
    set(gca, 'FontSize', 9);
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    title(sprintf('Sample %d: Real(H_{perfect})', s), 'FontSize', 11, 'FontWeight', 'bold');
end

sgtitle(sprintf('H_{perfect} Real Part (First %d Samples - %s, Folder: %s)', ...
    N_plot, snr_folder, strrep(folder, '_', '\_')), 'FontSize', 14, 'FontWeight', 'bold');

% =========================================================================
% FIGURE 2: Real Part of H_infer (2 rows x 4 columns)
% =========================================================================
fig2 = figure('Name', sprintf('%s Real Part (%s)', infer_label, snr_folder), ...
    'Color', 'w', 'Position', [100, 100, 1400, 620]);

tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');

for s = 1:N_plot
    nexttile;
    grid_infer = real(squeeze(H_infer(s, :, :)));
    imagesc(grid_infer);
    colorbar;
    set(gca, 'FontSize', 9);
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    title(sprintf('Sample %d: Real(%s)', s, infer_label), 'FontSize', 11, 'FontWeight', 'bold');
end

sgtitle(sprintf('%s Real Part (First %d Samples - %s, Folder: %s)', ...
    infer_label, N_plot, snr_folder, strrep(folder, '_', '\_')), 'FontSize', 14, 'FontWeight', 'bold');

fprintf('Visualization complete! 2 figures displayed.\n');
