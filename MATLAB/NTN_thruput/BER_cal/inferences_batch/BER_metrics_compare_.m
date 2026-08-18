%{
========================================================================================
    NTN OFDM Channel Equalization, BER, MSE, NMSE & SSIM Performance Evaluation
========================================================================================
OVERVIEW:
  This script evaluates the transmission, equalization, BER, MSE, NMSE, and SSIM
  performance of 5G NTN channel estimation techniques by loading pre-inferred channels
  from 'inferredChannel.mat' across SNR subfolders (SNR_-10dB, SNR_-5dB, ...):

    1. LS + Linear Interpolation (H_li): Standard 2D linear interpolation benchmark.
    2. Inferred Neural Network (H_LI_infer): Deep Learning model inference output.
    3. Perfect MMSE Benchmark (H_MMSE): Optimal MMSE estimator using H_ls_pilots
       and empirical covariance matrix R_hh derived from H_perfect.

USAGE:
  BER_metrics_compare()                   % Evaluates all batch folders in inferences_batch
  BER_metrics_compare('folder_name')       % Evaluates specific batch folder
========================================================================================
%}

folder = 'DUR100__A100_2p18e9_600km_30kHz_LSSequence';
labelname = 'LS+Attention Inferred';
BER_metrics_compare(folder, labelname)

function results = BER_metrics_compare(batch_folder, labelname)
    if exist('mfilename', 'builtin') && ~isempty(mfilename('fullpath'))
        script_dir = fileparts(mfilename('fullpath'));
    else
        script_dir = pwd;
    end

    if nargin < 1 || isempty(batch_folder)
        batch_folder = script_dir;
    end

    if nargin < 2 || isempty(labelname)
        labelname = 'LS+Attention Inferred';
    end

    % Add paths for helper functions (3 levels up) and BER_cal root (1 level up)
    addpath(fullfile(script_dir, '..', '..', '..', 'helper'));
    addpath(fullfile(script_dir, '..'));

    % If relative path provided, resolve relative to script_dir or pwd
    if ~exist(batch_folder, 'dir') && exist(fullfile(script_dir, batch_folder), 'dir')
        batch_folder = fullfile(script_dir, batch_folder);
    end

    % Check if batch_folder directly contains SNR subfolders with inferredChannel.mat
    direct_snr = discover_snr_folders(batch_folder);

    if isempty(direct_snr)
        % Search child directories for target batch folders
        dir_items = dir(batch_folder);
        batch_targets = {};
        for k = 1:length(dir_items)
            if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
                sub_path = fullfile(batch_folder, dir_items(k).name);
                if ~isempty(discover_snr_folders(sub_path))
                    batch_targets{end+1} = sub_path; %#ok<AGROW>
                end
            end
        end

        if isempty(batch_targets)
            error('No subfolders containing inferredChannel.mat found in:\n  %s', batch_folder);
        end

        fprintf('Found %d batch folder(s) to process.\n\n', length(batch_targets));
        results = cell(1, length(batch_targets));
        for t_idx = 1:length(batch_targets)
            fprintf('>>> Processing Batch [%d/%d]: %s <<<\n', t_idx, length(batch_targets), batch_targets{t_idx});
            results{t_idx} = process_single_batch(batch_targets{t_idx}, labelname);
        end
    else
        results = process_single_batch(batch_folder, labelname);
    end
end

%% Helper function to discover SNR subfolders in a target batch directory
function snr_subfolders = discover_snr_folders(target_dir)
    snr_subfolders = {};
    if ~exist(target_dir, 'dir')
        return;
    end
    dir_items = dir(target_dir);
    for k = 1:length(dir_items)
        if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
            subname = dir_items(k).name;
            mat_path = fullfile(target_dir, subname, 'inferredChannel.mat');
            if exist(mat_path, 'file')
                snr_subfolders{end+1} = fullfile(target_dir, subname); %#ok<AGROW>
            end
        end
    end
end

%% Primary evaluation function for a single batch folder
function save_struct = process_single_batch(batch_folder, labelname)
    if nargin < 2 || isempty(labelname)
        labelname = 'LS+Attention Inferred';
    end

    fprintf('========================================================================\n');
    fprintf('        NTN BER, MSE, NMSE & SSIM Evaluation from Inferred Channels     \n');
    fprintf('========================================================================\n');
    fprintf('Batch Folder: %s\n\n', batch_folder);

    % Discover & parse SNR subfolders
    dir_items = dir(batch_folder);
    snr_subfolders = {};
    snr_vals = [];

    for k = 1:length(dir_items)
        if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
            subname = dir_items(k).name;
            mat_path = fullfile(batch_folder, subname, 'inferredChannel.mat');
            if exist(mat_path, 'file')
                tokens = regexp(subname, 'SNR_(-?\d+(\.\d+)?)dB', 'tokens');
                if ~isempty(tokens)
                    snr_val = str2double(tokens{1}{1});
                else
                    mat_data = load(mat_path, 'SNRdB', 'snr');
                    if isfield(mat_data, 'SNRdB')
                        snr_val = double(mat_data.SNRdB);
                    elseif isfield(mat_data, 'snr')
                        snr_val = double(mat_data.snr);
                    else
                        continue;
                    end
                end
                snr_subfolders{end+1} = fullfile(batch_folder, subname); %#ok<AGROW>
                snr_vals(end+1) = snr_val; %#ok<AGROW>
            end
        end
    end

    if isempty(snr_subfolders)
        error('No subfolders containing inferredChannel.mat found in:\n  %s', batch_folder);
    end

    % Sort subfolders by SNR in ascending order
    [SNRdB_sorted, sort_idx] = sort(snr_vals);
    snr_subfolders = snr_subfolders(sort_idx);
    num_snr = length(SNRdB_sorted);

    fprintf('Discovered %d SNR subfolders (SNRdB = %s):\n', num_snr, mat2str(SNRdB_sorted));
    for i = 1:num_snr
        [~, sf_name] = fileparts(snr_subfolders{i});
        fprintf('  [%d] %s -> SNRdB = %.1f dB\n', i, sf_name, SNRdB_sorted(i));
    end
    fprintf('\n');

    % Initialize simulation parameters & OFDM carrier configuration
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing = 30;     % 30 kHz SCS
    carrier.NSizeGrid = 11;             % 11 RBs (132 subcarriers)
    carrier.CyclicPrefix = 'Normal';

    nSubcarriers = carrier.NSizeGrid * 12;
    nSymbols     = carrier.SymbolsPerSlot;

    pdsch = nrPDSCHConfig;
    pdsch.PRBSet = 0:carrier.NSizeGrid-1;
    pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot];
    pdsch.MappingType = 'A';
    pdsch.NID = carrier.NCellID;
    pdsch.RNTI = 1;
    pdsch.VRBToPRBInterleaving = 0;
    pdsch.VRBBundleSize = 4;
    pdsch.NumLayers = 1;
    pdsch.Modulation = '16QAM';

    pdsch.DMRS.DMRSPortSet = [];
    pdsch.DMRS.DMRSTypeAPosition = 2;
    pdsch.DMRS.DMRSLength = 1;
    pdsch.DMRS.DMRSAdditionalPosition = 1;
    pdsch.DMRS.DMRSConfigurationType = 2;
    pdsch.DMRS.NumCDMGroupsWithoutData = 1;
    pdsch.DMRS.NIDNSCID = 1;
    pdsch.DMRS.NSCID = 0;

    [pdschIndices, pdschInfo] = nrPDSCHIndices(carrier, pdsch);
    dmrsSymbols = nrPDSCHDMRS(carrier, pdsch);
    dmrsIndices = nrPDSCHDMRSIndices(carrier, pdsch);

    % Metric storage vectors across SNR points
    ber_li_arr       = zeros(1, num_snr);
    ber_infer_arr    = zeros(1, num_snr);
    ber_mmse_arr     = zeros(1, num_snr);

    mse_li_arr       = zeros(1, num_snr);
    mse_infer_arr    = zeros(1, num_snr);
    mse_mmse_arr     = zeros(1, num_snr);

    nmse_li_arr      = zeros(1, num_snr);
    nmse_infer_arr   = zeros(1, num_snr);
    nmse_mmse_arr    = zeros(1, num_snr);

    nmse_li_db_arr   = zeros(1, num_snr);
    nmse_infer_db_arr= zeros(1, num_snr);
    nmse_mmse_db_arr = zeros(1, num_snr);

    ssim_li_arr      = zeros(1, num_snr);
    ssim_infer_arr   = zeros(1, num_snr);
    ssim_mmse_arr    = zeros(1, num_snr);

    % Loop over SNR subfolders
    for s_idx = 1:num_snr
        snr_val = SNRdB_sorted(s_idx);
        mat_path = fullfile(snr_subfolders{s_idx}, 'inferredChannel.mat');
        data = load(mat_path);

        % Extract channels: [numUE x 132 x 14]
        H_perfect_raw = data.H_perfect;

        if isfield(data, 'H_li')
            H_li_raw = data.H_li;
        elseif isfield(data, 'H_LI')
            H_li_raw = data.H_LI;
        elseif isfield(data, 'H_ls')
            H_li_raw = data.H_ls;
        elseif isfield(data, 'H_LS')
            H_li_raw = data.H_LS;
        else
            H_li_raw = data.H_perfect;
        end

        % Adaptively load inferred channel (H_LS_infer, H_LI_infer, H_infer, etc.)
        if isfield(data, 'H_LS_infer')
            H_infer_raw = double(data.H_LS_infer);
        elseif isfield(data, 'H_ls_infer')
            H_infer_raw = double(data.H_ls_infer);
        elseif isfield(data, 'H_LI_infer')
            H_infer_raw = double(data.H_LI_infer);
        elseif isfield(data, 'H_li_infer')
            H_infer_raw = double(data.H_li_infer);
        elseif isfield(data, 'H_infer')
            H_infer_raw = double(data.H_infer);
        else
            error('No inferred channel field (H_LS_infer, H_LI_infer, or H_infer) found in:\n  %s', mat_path);
        end

        H_ls_pilots   = data.H_ls_pilots;
        dmrs_idx      = double(data.pilot_indices);

        numUE = size(H_perfect_raw, 1);

        % Convert from [numUE x 132 x 14] to cell array of [132 x 14] grids
        H_perfect_cell = cell(1, numUE);
        H_li_cell      = cell(1, numUE);
        H_infer_cell   = cell(1, numUE);

        for n = 1:numUE
            H_perfect_cell{n} = squeeze(H_perfect_raw(n, :, :));
            H_li_cell{n}      = squeeze(H_li_raw(n, :, :));
            H_infer_cell{n}   = squeeze(H_infer_raw(n, :, :));
        end

        % Build empirical covariance matrix R_hh across all UEs from H_perfect
        nREs = nSubcarriers * nSymbols;
        H_perf_matrix = zeros(nREs, numUE);
        for n = 1:numUE
            h_grid = H_perfect_cell{n};
            H_perf_matrix(:, n) = h_grid(:);
        end

        R_hh = (H_perf_matrix * H_perf_matrix') / numUE;
        R_h_hp = R_hh(:, dmrs_idx);
        R_hp_hp = R_hh(dmrs_idx, dmrs_idx);

        % Compute noise power for MMSE weight matrix
        sigPower = mean(abs(H_ls_pilots(:)).^2);
        snr_linear = 10^(snr_val / 10);
        noisePower = sigPower / snr_linear;

        W_MMSE = R_h_hp / (R_hp_hp + noisePower * eye(length(dmrs_idx)));

        % Formulate H_MMSE for all UEs
        H_mmse_cell = cell(1, numUE);
        for n = 1:numUE
            h_pilot_vec = H_ls_pilots(n, :).';
            h_mmse_vec  = W_MMSE * h_pilot_vec;
            H_mmse_cell{n} = reshape(h_mmse_vec, [nSubcarriers, nSymbols]);
        end

        % Initialize per-UE metric accumulators
        ber_li_ue    = zeros(1, numUE);
        ber_infer_ue = zeros(1, numUE);
        ber_mmse_ue  = zeros(1, numUE);

        mse_li_ue    = zeros(1, numUE);
        mse_infer_ue = zeros(1, numUE);
        mse_mmse_ue  = zeros(1, numUE);

        nmse_li_ue   = zeros(1, numUE);
        nmse_infer_ue= zeros(1, numUE);
        nmse_mmse_ue = zeros(1, numUE);

        ssim_li_ue   = zeros(1, numUE);
        ssim_infer_ue= zeros(1, numUE);
        ssim_mmse_ue = zeros(1, numUE);

        % Simulation Loop over UEs
        rng(100 + s_idx); % Fixed bit seed per SNR for reproducible payload

        for n = 1:numUE
            H_perf_n  = H_perfect_cell{n};
            H_li_n    = H_li_cell{n};
            H_infer_n = H_infer_cell{n};
            H_mmse_n  = H_mmse_cell{n};

            % Generate random PDSCH payload data bits
            txBits = randi([0 1], pdschInfo.G, 1);
            txSymbols = nrPDSCH(carrier, pdsch, txBits);

            % Build transmit OFDM grid
            txGrid = zeros(nSubcarriers, nSymbols);
            txGrid(pdschIndices) = txSymbols;
            txGrid(dmrsIndices)  = dmrsSymbols;

            % Receive Grid transmission over true channel H_perfect_n + AWGN
            rxGrid_clean = txGrid .* H_perf_n;
            sigPwr_grid  = mean(abs(rxGrid_clean(:)).^2);
            N0           = sigPwr_grid / snr_linear;
            noise_grid   = sqrt(N0 / 2) * (randn(size(rxGrid_clean)) + 1j * randn(size(rxGrid_clean)));
            rxGrid       = rxGrid_clean + noise_grid;

            % --- Equalization & Demodulation for H_li ---
            rxData_li = rxGrid(pdschIndices) .* conj(H_li_n(pdschIndices)) ./ (abs(H_li_n(pdschIndices)).^2 + N0);
            llr_li    = nrPDSCHDecode(carrier, pdsch, rxData_li, N0);
            rxBits_li = llr_li{1} < 0;
            ber_li_ue(n) = mean(txBits ~= rxBits_li);

            % --- Equalization & Demodulation for H_LI_infer ---
            rxData_infer = rxGrid(pdschIndices) .* conj(H_infer_n(pdschIndices)) ./ (abs(H_infer_n(pdschIndices)).^2 + N0);
            llr_infer    = nrPDSCHDecode(carrier, pdsch, rxData_infer, N0);
            rxBits_infer = llr_infer{1} < 0;
            ber_infer_ue(n) = mean(txBits ~= rxBits_infer);

            % --- Equalization & Demodulation for H_MMSE ---
            rxData_mmse = rxGrid(pdschIndices) .* conj(H_mmse_n(pdschIndices)) ./ (abs(H_mmse_n(pdschIndices)).^2 + N0);
            llr_mmse    = nrPDSCHDecode(carrier, pdsch, rxData_mmse, N0);
            rxBits_mmse = llr_mmse{1} < 0;
            ber_mmse_ue(n) = mean(txBits ~= rxBits_mmse);

            % --- Channel Estimation Metrics: MSE, NMSE, SSIM ---
            % MSE
            mse_li_ue(n)    = mean(abs(H_li_n(:) - H_perf_n(:)).^2);
            mse_infer_ue(n) = mean(abs(H_infer_n(:) - H_perf_n(:)).^2);
            mse_mmse_ue(n)  = mean(abs(H_mmse_n(:) - H_perf_n(:)).^2);

            % NMSE
            norm_gt = sum(abs(H_perf_n(:)).^2);
            nmse_li_ue(n)    = sum(abs(H_li_n(:) - H_perf_n(:)).^2) / norm_gt;
            nmse_infer_ue(n) = sum(abs(H_infer_n(:) - H_perf_n(:)).^2) / norm_gt;
            nmse_mmse_ue(n)  = sum(abs(H_mmse_n(:) - H_perf_n(:)).^2) / norm_gt;

            % SSIM (Normalized magnitude SSIM)
            max_val = max(abs(H_perf_n(:)));
            if max_val > 0
                img_gt    = abs(H_perf_n) / max_val;
                img_li    = abs(H_li_n) / max_val;
                img_infer = abs(H_infer_n) / max_val;
                img_mmse  = abs(H_mmse_n) / max_val;

                ssim_li_ue(n)    = compute_ssim_robust(img_li, img_gt);
                ssim_infer_ue(n) = compute_ssim_robust(img_infer, img_gt);
                ssim_mmse_ue(n)  = compute_ssim_robust(img_mmse, img_gt);
            end
        end

        % Store average metrics for current SNR
        ber_li_arr(s_idx)    = mean(ber_li_ue);
        ber_infer_arr(s_idx) = mean(ber_infer_ue);
        ber_mmse_arr(s_idx)  = mean(ber_mmse_ue);

        mse_li_arr(s_idx)    = mean(mse_li_ue);
        mse_infer_arr(s_idx) = mean(mse_infer_ue);
        mse_mmse_arr(s_idx)  = mean(mse_mmse_ue);

        nmse_li_arr(s_idx)    = mean(nmse_li_ue);
        nmse_infer_arr(s_idx) = mean(nmse_infer_ue);
        nmse_mmse_arr(s_idx)  = mean(nmse_mmse_ue);

        nmse_li_db_arr(s_idx)    = 10 * log10(mean(nmse_li_ue));
        nmse_infer_db_arr(s_idx) = 10 * log10(mean(nmse_infer_ue));
        nmse_mmse_db_arr(s_idx)  = 10 * log10(mean(nmse_mmse_ue));

        ssim_li_arr(s_idx)    = mean(ssim_li_ue);
        ssim_infer_arr(s_idx) = mean(ssim_infer_ue);
        ssim_mmse_arr(s_idx)  = mean(ssim_mmse_ue);

        % Save BER & Channel metrics inside current SNR subfolder
        eval_mat_file = fullfile(snr_subfolders{s_idx}, 'BER_performance_results.mat');
        save(eval_mat_file, ...
            'snr_val', 'ber_li_ue', 'ber_infer_ue', 'ber_mmse_ue', ...
            'mse_li_ue', 'mse_infer_ue', 'mse_mmse_ue', ...
            'nmse_li_ue', 'nmse_infer_ue', 'nmse_mmse_ue', ...
            'ssim_li_ue', 'ssim_infer_ue', 'ssim_mmse_ue');
    end

    % Print console summary table
    fprintf('\n========================================================================\n');
    fprintf('                      PERFORMANCE COMPARISON SUMMARY                    \n');
    fprintf('========================================================================\n');
    fprintf('%-8s | %-12s %-12s %-12s | %-12s %-12s %-12s\n', ...
        'SNR (dB)', 'BER (LI)', 'BER (Infer)', 'BER (MMSE)', 'NMSEdB (LI)', 'NMSEdB(Inf)', 'NMSEdB(MMSE)');
    fprintf('%s\n', repmat('-', 1, 95));
    for i = 1:num_snr
        fprintf('%-8.1f | %-12.6f %-12.6f %-12.6f | %-12.2f %-12.2f %-12.2f\n', ...
            SNRdB_sorted(i), ber_li_arr(i), ber_infer_arr(i), ber_mmse_arr(i), ...
            nmse_li_db_arr(i), nmse_infer_db_arr(i), nmse_mmse_db_arr(i));
    end
    fprintf('\n');

    fprintf('%-8s | %-12s %-12s %-12s | %-12s %-12s %-12s\n', ...
        'SNR (dB)', 'SSIM (LI)', 'SSIM (Infer)', 'SSIM (MMSE)', 'MSE (LI)', 'MSE (Infer)', 'MSE (MMSE)');
    fprintf('%s\n', repmat('-', 1, 95));
    for i = 1:num_snr
        fprintf('%-8.1f | %-12.4f %-12.4f %-12.4f | %-12.3e %-12.3e %-12.3e\n', ...
            SNRdB_sorted(i), ssim_li_arr(i), ssim_infer_arr(i), ssim_mmse_arr(i), ...
            mse_li_arr(i), mse_infer_arr(i), mse_mmse_arr(i));
    end
    fprintf('\n');

    % Plot & Export PDF Comparison Figures (Wrapped in try-catch to prevent permission lock errors)
    % --- Figure 1: BER Comparison vs SNR ---
    fig1 = figure('Name', 'BER Comparison', 'Color', 'w', 'Position', [100 100 700 500], 'Visible', 'off');
    hold on;
    h1 = semilogy(SNRdB_sorted, ber_li_arr, '--o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0.4470 0.7410]);
    h2 = semilogy(SNRdB_sorted, ber_infer_arr, '-^', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    h3 = semilogy(SNRdB_sorted, ber_mmse_arr, '-s', 'Color', [0 0 0], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0 0]);
    
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
    title('BER Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3], {'LS + Linear Interpolation', labelname, 'MMSE Benchmark'}, 'Location', 'southwest', 'FontSize', 10);
    hold off;

    ber_pdf_path = fullfile(batch_folder, 'BER_comparison.pdf');
    try exportgraphics(fig1, ber_pdf_path, 'ContentType', 'vector'); catch, try saveas(fig1, ber_pdf_path); catch; end; end
    fprintf('Saved BER figure: %s\n', ber_pdf_path);
    close(fig1);

    % --- Figure 2: NMSE Comparison (dB) vs SNR ---
    fig2 = figure('Name', 'NMSE Comparison', 'Color', 'w', 'Position', [150 150 700 500], 'Visible', 'off');
    hold on;
    h1 = plot(SNRdB_sorted, nmse_li_db_arr, '--o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0.4470 0.7410]);
    h2 = plot(SNRdB_sorted, nmse_infer_db_arr, '-^', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    h3 = plot(SNRdB_sorted, nmse_mmse_db_arr, '-s', 'Color', [0 0 0], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0 0]);
    
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Mean Squared Error (NMSE) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
    title('NMSE Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3], {'LS + Linear Interpolation', labelname, 'MMSE Benchmark'}, 'Location', 'northeast', 'FontSize', 10);
    hold off;

    nmse_pdf_path = fullfile(batch_folder, 'NMSE_comparison.pdf');
    try exportgraphics(fig2, nmse_pdf_path, 'ContentType', 'vector'); catch, try saveas(fig2, nmse_pdf_path); catch; end; end
    fprintf('Saved NMSE figure: %s\n', nmse_pdf_path);
    close(fig2);

    % --- Figure 3: SSIM Comparison vs SNR ---
    fig3 = figure('Name', 'SSIM Comparison', 'Color', 'w', 'Position', [200 200 700 500], 'Visible', 'off');
    hold on;
    h1 = plot(SNRdB_sorted, ssim_li_arr, '--o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0.4470 0.7410]);
    h2 = plot(SNRdB_sorted, ssim_infer_arr, '-^', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    h3 = plot(SNRdB_sorted, ssim_mmse_arr, '-s', 'Color', [0 0 0], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0 0]);
    
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Structural Similarity Index (SSIM)', 'FontSize', 12, 'FontWeight', 'bold');
    title('SSIM Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on;
    ylim([0 1.05]);
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3], {'LS + Linear Interpolation', labelname, 'MMSE Benchmark'}, 'Location', 'southeast', 'FontSize', 10);
    hold off;

    ssim_pdf_path = fullfile(batch_folder, 'SSIM_comparison.pdf');
    try exportgraphics(fig3, ssim_pdf_path, 'ContentType', 'vector'); catch, try saveas(fig3, ssim_pdf_path); catch; end; end
    fprintf('Saved SSIM figure: %s\n', ssim_pdf_path);
    close(fig3);

    % --- Figure 4: MSE Comparison vs SNR ---
    fig4 = figure('Name', 'MSE Comparison', 'Color', 'w', 'Position', [250 250 700 500], 'Visible', 'off');
    hold on;
    h1 = semilogy(SNRdB_sorted, mse_li_arr, '--o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0.4470 0.7410]);
    h2 = semilogy(SNRdB_sorted, mse_infer_arr, '-^', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    h3 = semilogy(SNRdB_sorted, mse_mmse_arr, '-s', 'Color', [0 0 0], 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0 0]);
    
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Mean Squared Error (MSE)', 'FontSize', 12, 'FontWeight', 'bold');
    title('MSE Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend([h1, h2, h3], {'LS + Linear Interpolation', labelname, 'MMSE Benchmark'}, 'Location', 'northeast', 'FontSize', 10);
    hold off;

    mse_pdf_path = fullfile(batch_folder, 'MSE_comparison.pdf');
    try exportgraphics(fig4, mse_pdf_path, 'ContentType', 'vector'); catch, try saveas(fig4, mse_pdf_path); catch; end; end
    fprintf('Saved MSE figure: %s\n', mse_pdf_path);
    close(fig4);

    % Save synthesized results into MAT file in batch folder
    save_struct = struct();
    save_struct.SNRdB           = SNRdB_sorted;
    save_struct.ber_li          = ber_li_arr;
    save_struct.ber_infer       = ber_infer_arr;
    save_struct.ber_mmse        = ber_mmse_arr;

    save_struct.mse_li          = mse_li_arr;
    save_struct.mse_infer       = mse_infer_arr;
    save_struct.mse_mmse       = mse_mmse_arr;

    save_struct.nmse_li         = nmse_li_arr;
    save_struct.nmse_infer      = nmse_infer_arr;
    save_struct.nmse_mmse       = nmse_mmse_arr;

    save_struct.nmse_li_dB      = nmse_li_db_arr;
    save_struct.nmse_infer_dB   = nmse_infer_db_arr;
    save_struct.nmse_mmse_dB    = nmse_mmse_db_arr;

    save_struct.ssim_li         = ssim_li_arr;
    save_struct.ssim_infer      = ssim_infer_arr;
    save_struct.ssim_mmse       = ssim_mmse_arr;

    mat_out_path = fullfile(batch_folder, 'synthesized_results.mat');
    save(mat_out_path, '-struct', 'save_struct');
    fprintf('Saved synthesized MAT file: %s\n', mat_out_path);

    % Save Markdown Report
    md_path = fullfile(batch_folder, 'simulation_results.md');
    fid = fopen(md_path, 'w');
    if fid ~= -1
        fprintf(fid, '# NTN Inferred Channel Equalization & Performance Summary\n\n');
        fprintf(fid, '## Target Batch Directory\n`%s`\n\n', batch_folder);
        
        fprintf(fid, '## BER Performance Table\n');
        fprintf(fid, '| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |\n');
        fprintf(fid, '|:---:|:---:|:---:|:---:|\n');
        for i = 1:num_snr
            fprintf(fid, '| %.1f | %.6f | %.6f | %.6f |\n', SNRdB_sorted(i), ber_li_arr(i), ber_infer_arr(i), ber_mmse_arr(i));
        end
        fprintf(fid, '\n');

        fprintf(fid, '## NMSE (dB) Performance Table\n');
        fprintf(fid, '| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |\n');
        fprintf(fid, '|:---:|:---:|:---:|:---:|\n');
        for i = 1:num_snr
            fprintf(fid, '| %.1f | %.2f dB | %.2f dB | %.2f dB |\n', SNRdB_sorted(i), nmse_li_db_arr(i), nmse_infer_db_arr(i), nmse_mmse_db_arr(i));
        end
        fprintf(fid, '\n');

        fprintf(fid, '## SSIM Performance Table\n');
        fprintf(fid, '| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |\n');
        fprintf(fid, '|:---:|:---:|:---:|:---:|\n');
        for i = 1:num_snr
            fprintf(fid, '| %.1f | %.4f | %.4f | %.4f |\n', SNRdB_sorted(i), ssim_li_arr(i), ssim_infer_arr(i), ssim_mmse_arr(i));
        end
        fclose(fid);
        fprintf('Saved Markdown report: %s\n', md_path);
    end

    fprintf('\nEvaluation completed successfully for %s!\n', batch_folder);
end

function val = compute_ssim_robust(x, y)
    try
        val = ssim(x, y);
    catch
        % Fallback SSIM calculation without Image Processing Toolbox
        K1 = 0.01; K2 = 0.03; L = 1.0;
        C1 = (K1 * L)^2; C2 = (K2 * L)^2;
        mu_x = mean(x(:)); mu_y = mean(y(:));
        var_x = var(x(:)); var_y = var(y(:));
        cov_xy = mean((x(:) - mu_x) .* (y(:) - mu_y));
        
        num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2);
        den = (mu_x^2 + mu_y^2 + C1) * (var_x + var_y + C2);
        val = num / den;
    end
end
