
folders = {'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\BER_cal\inferences_batch\DUR100__A100_2p18e9_600km_30kHz_LSSequence',
    'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\BER_cal\inferences_batch\DUR100__A100_2p18e9_600km_30kHz_LI_Grid'};
folder_labels = {'LS+Attention Inferred', 'LI+DnCNN Inferred'};
output_folder = 'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\BER_cal\inference_all\DUR100__A100_2p18e9_600km_30kHz';

plot_synthesized_comparison(folders, folder_labels, output_folder)

function results = plot_synthesized_comparison(folders, folder_labels, output_folder)
% PLOT_SYNTHESIZED_COMPARISON Load synthesized_results.mat from multiple folders
% and plot performance comparisons (MSE, NMSE, SSIM, BER) across approaches.
%
% Synthesizes and plots comparison curves for multiple inferred models alongside
% averaged benchmark curves (LI Benchmark Average and MMSE Benchmark Average).
%
% Usage:
%   plot_synthesized_comparison()
%   plot_synthesized_comparison(folders)
%   plot_synthesized_comparison(folders, folder_labels)
%   plot_synthesized_comparison(folders, folder_labels, output_folder)
%
% Inputs:
%   folders       - Cell array of folder paths containing synthesized_results.mat
%   folder_labels - Cell array of custom legend names for each folder's model
%                   (e.g., {'LS+Attention Inferred', 'LI+DnCNN Inferred'})
%   output_folder - (Optional) Directory to save export PDFs and comparison MAT file.
%
% Output:
%   results       - Struct with all synthesized data and averaged benchmarks.

    script_dir = fileparts(mfilename('fullpath'));

    % Default Folders Fallback
    if nargin < 1 || isempty(folders)
        folders = { ...
            fullfile(script_dir, 'inference_samples', 'DUR100ns_2p18G__A100ns_2p18G', ...
                'Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps'), ...
            fullfile(script_dir, 'inferences_batch', ...
                'DUR100__A100_2p18e9_600km_30kHz_LI_Grid') ...
        };
    end

    if ischar(folders) || isstring(folders)
        folders = {char(folders)};
    end

    num_folders = length(folders);

    % Default Folder Labels Fallback
    if nargin < 2 || isempty(folder_labels)
        folder_labels = {'LS+Attention Inferred', 'LI+DnCNN Inferred'};
    elseif ischar(folder_labels) || isstring(folder_labels)
        folder_labels = {char(folder_labels)};
    end

    % Fill missing labels if necessary
    for i = (length(folder_labels) + 1):num_folders
        [~, leaf_name] = fileparts(folders{i});
        folder_labels{i} = sprintf('Model %d (%s)', i, strrep(leaf_name, '_', ' '));
    end

    % Default Output Directory
    if nargin < 3 || isempty(output_folder)
        output_folder = fullfile(script_dir, 'synthesized_comparison_output');
    end

    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    safe_printf('========================================================================\n');
    safe_printf('           Synthesized Performance Results Comparison                   \n');
    safe_printf('========================================================================\n');
    safe_printf('Output Directory: %s\n\n', output_folder);
    safe_printf('Loading %d Synthesized Results Folders:\n', num_folders);
    for i = 1:num_folders
        safe_printf('  [%d] Label: "%s"\n      Path:  %s\n', i, folder_labels{i}, folders{i});
    end
    safe_printf('\n');

    % Data Structures Initialization
    models_data = cell(1, num_folders);
    SNRdB = [];

    % Track extracted metrics per folder
    m_mse     = cell(1, num_folders);
    m_nmse    = cell(1, num_folders);
    m_nmse_db = cell(1, num_folders);
    m_ssim    = cell(1, num_folders);
    m_ber     = cell(1, num_folders);

    li_mse     = cell(1, num_folders);
    li_nmse    = cell(1, num_folders);
    li_nmse_db = cell(1, num_folders);
    li_ssim    = cell(1, num_folders);
    li_ber     = cell(1, num_folders);

    mmse_mse     = cell(1, num_folders);
    mmse_nmse    = cell(1, num_folders);
    mmse_nmse_db = cell(1, num_folders);
    mmse_ssim    = cell(1, num_folders);
    mmse_ber     = cell(1, num_folders);

    % Load and Parse MAT files
    for i = 1:num_folders
        mat_path = fullfile(folders{i}, 'synthesized_results.mat');
        if ~exist(mat_path, 'file')
            % Fallback check if folders{i} is directly the .mat file
            if exist(folders{i}, 'file') && endsWith(folders{i}, '.mat')
                mat_path = folders{i};
            else
                error('File not found: %s\nPlease run lines_synthesize() on target folder first.', mat_path);
            end
        end

        data = load(mat_path);
        models_data{i} = data;

        % Retrieve SNR vector
        current_snr = [];
        if isfield(data, 'SNRdB')
            current_snr = double(data.SNRdB(:).');
        elseif isfield(data, 'snr')
            current_snr = double(data.snr(:).');
        end

        if isempty(SNRdB)
            SNRdB = current_snr;
        elseif ~isempty(current_snr) && ~isequal(SNRdB, current_snr)
            warning('SNRdB mismatch between folders! Aligning with first folder SNRdB.');
        end

        num_snr = length(SNRdB);

        % Extract Inferred Model Metrics
        m_mse{i}     = extract_field(data, {'mse_infer', 'mse_ls_atten', 'mse_ls_cnn', 'mse_li_cnn', 'mmse_test', 'mse_test'}, 'mse_', {'li', 'mmse', 'benchmark', 'input'}, num_snr);
        m_nmse{i}    = extract_field(data, {'nmse_infer', 'nmse_ls_atten', 'nmse_ls_cnn', 'nmse_li_cnn', 'nmse_test'}, 'nmse_', {'li', 'mmse', 'benchmark', 'input', 'db'}, num_snr);
        m_nmse_db{i} = extract_field(data, {'nmse_infer_dB', 'nmse_infer_db', 'nmse_ls_atten_dB', 'nmse_ls_atten_db', 'nmse_ls_cnn_dB', 'nmse_li_cnn_dB', 'nmse_test_db'}, 'nmse_', {'li', 'mmse', 'benchmark', 'input'}, num_snr, true);
        m_ssim{i}    = extract_field(data, {'ssim_infer', 'ssim_ls_atten', 'ssim_ls_cnn', 'ssim_li_cnn', 'ssim_test'}, 'ssim_', {'li', 'mmse', 'benchmark', 'input'}, num_snr);
        m_ber{i}     = extract_field(data, {'ber_infer', 'ber_ls_atten', 'ber_ls_cnn', 'ber_li_cnn', 'ber_test'}, 'ber_', {'li', 'mmse'}, num_snr);

        % Fill NMSE_dB if linear NMSE exists
        if all(isnan(m_nmse_db{i})) && ~all(isnan(m_nmse{i}))
            m_nmse_db{i} = 10 * log10(m_nmse{i});
        elseif all(isnan(m_nmse{i})) && ~all(isnan(m_nmse_db{i}))
            m_nmse{i} = 10.^(m_nmse_db{i} / 10);
        end

        % Extract LI Benchmark Metrics
        li_mse{i}     = extract_field(data, {'mse_li', 'mse_LI', 'mmse_li_benchmark_test', 'mmse_input_test'}, 'mse_li', {}, num_snr);
        li_nmse{i}    = extract_field(data, {'nmse_li', 'nmse_LI', 'nmse_li_benchmark_test', 'nmse_input_test'}, 'nmse_li', {'db'}, num_snr);
        li_nmse_db{i} = extract_field(data, {'nmse_li_dB', 'nmse_li_db', 'nmse_LI_dB', 'nmse_li_benchmark_test_db', 'nmse_input_test_db'}, 'nmse_li', {}, num_snr, true);
        li_ssim{i}    = extract_field(data, {'ssim_li', 'ssim_LI', 'ssim_li_benchmark_test', 'ssim_input_test'}, 'ssim_li', {}, num_snr);
        li_ber{i}     = extract_field(data, {'ber_li', 'ber_LI'}, 'ber_li', {}, num_snr);

        if all(isnan(li_nmse_db{i})) && ~all(isnan(li_nmse{i}))
            li_nmse_db{i} = 10 * log10(li_nmse{i});
        end

        % Extract MMSE Benchmark Metrics
        mmse_mse{i}     = extract_field(data, {'mse_mmse', 'mse_MMSE'}, 'mse_mmse', {}, num_snr);
        mmse_nmse{i}    = extract_field(data, {'nmse_mmse', 'nmse_MMSE'}, 'nmse_mmse', {'db'}, num_snr);
        mmse_nmse_db{i} = extract_field(data, {'nmse_mmse_dB', 'nmse_mmse_db', 'nmse_MMSE_dB'}, 'nmse_mmse', {}, num_snr, true);
        mmse_ssim{i}    = extract_field(data, {'ssim_mmse', 'ssim_MMSE'}, 'ssim_mmse', {}, num_snr);
        mmse_ber{i}     = extract_field(data, {'ber_mmse', 'ber_MMSE'}, 'ber_mmse', {}, num_snr);

        if all(isnan(mmse_nmse_db{i})) && ~all(isnan(mmse_nmse{i}))
            mmse_nmse_db{i} = 10 * log10(mmse_nmse{i});
        end
    end

    num_snr = length(SNRdB);

    % Compute Average Benchmarks Across All Folders
    li_mse_avg     = compute_benchmark_avg(li_mse, num_snr);
    li_nmse_avg    = compute_benchmark_avg(li_nmse, num_snr);
    li_nmse_db_avg = compute_benchmark_avg(li_nmse_db, num_snr);
    li_ssim_avg    = compute_benchmark_avg(li_ssim, num_snr);
    li_ber_avg     = compute_benchmark_avg(li_ber, num_snr);

    mmse_mse_avg     = compute_benchmark_avg(mmse_mse, num_snr);
    mmse_nmse_avg    = compute_benchmark_avg(mmse_nmse, num_snr);
    mmse_nmse_db_avg = compute_benchmark_avg(mmse_nmse_db, num_snr);
    mmse_ssim_avg    = compute_benchmark_avg(mmse_ssim, num_snr);
    mmse_ber_avg     = compute_benchmark_avg(mmse_ber, num_snr);

    % Print Console Summary Tables
    print_summary_table('MSE', SNRdB, m_mse, li_mse_avg, mmse_mse_avg, folder_labels);
    print_summary_table('NMSE (dB)', SNRdB, m_nmse_db, li_nmse_db_avg, mmse_nmse_db_avg, folder_labels);
    print_summary_table('SSIM', SNRdB, m_ssim, li_ssim_avg, mmse_ssim_avg, folder_labels);
    print_summary_table('BER', SNRdB, m_ber, li_ber_avg, mmse_ber_avg, folder_labels);

    % Define Color Palette & Styles for Inferred Models
    model_colors = {
        [0.8500 0.3250 0.0980], ... % Red-Orange
        [0.4660 0.6740 0.1880], ... % Green
        [0.4940 0.1840 0.5560], ... % Purple
        [0.9290 0.6940 0.1250], ... % Yellow-Orange
        [0.3010 0.7450 0.9330], ... % Cyan
        [0.6350 0.0780 0.1840]      % Dark Red
    };
    model_markers = {'^', 'v', 'd', '*', 'p', 'h'};

    % Benchmark Styles
    li_color   = [0 0.4470 0.7410]; % Blue
    li_style   = '--';
    li_marker  = 'o';

    mmse_color  = [0.15 0.15 0.15]; % Dark Grey / Black
    mmse_style  = '-.';
    mmse_marker = 's';

    % =========================================================================
    % FIGURE 1: MSE Comparison
    % =========================================================================
    fig1 = figure('Name', 'MSE Comparison', 'Color', 'w', 'Position', [100 100 750 520], 'Visible', 'off');
    hold on; h_lines = []; h_labels = {};

    % Plot Inferred Models
    for i = 1:num_folders
        if ~all(isnan(m_mse{i}))
            col = model_colors{mod(i-1, length(model_colors))+1};
            mrk = model_markers{mod(i-1, length(model_markers))+1};
            h = semilogy(SNRdB, m_mse{i}, '-', 'Color', col, 'LineWidth', 2.0, ...
                'Marker', mrk, 'MarkerSize', 8, 'MarkerFaceColor', col);
            h_lines = [h_lines; h]; %#ok<AGROW>
            h_labels{end+1} = folder_labels{i}; %#ok<AGROW>
        end
    end

    % Plot LI Benchmark Average
    if ~all(isnan(li_mse_avg))
        h = semilogy(SNRdB, li_mse_avg, li_style, 'Color', li_color, 'LineWidth', 1.8, ...
            'Marker', li_marker, 'MarkerSize', 7, 'MarkerFaceColor', li_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'LI Benchmark'; %#ok<AGROW>
    end

    % Plot MMSE Benchmark Average
    if ~all(isnan(mmse_mse_avg))
        h = semilogy(SNRdB, mmse_mse_avg, mmse_style, 'Color', mmse_color, 'LineWidth', 1.8, ...
            'Marker', mmse_marker, 'MarkerSize', 7, 'MarkerFaceColor', mmse_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'MMSE Benchmark '; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Mean Squared Error (MSE)', 'FontSize', 12, 'FontWeight', 'bold');
    title('MSE Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    if ~isempty(h_lines)
        legend(h_lines, h_labels, 'Location', 'northeast', 'FontSize', 10);
    end
    hold off;

    mse_pdf_path = fullfile(output_folder, 'MSE_comparison.pdf');
    try exportgraphics(fig1, mse_pdf_path, 'ContentType', 'vector'); catch, saveas(fig1, mse_pdf_path); end
    safe_printf('Saved MSE plot: %s\n', mse_pdf_path);
    close(fig1);

    % =========================================================================
    % FIGURE 2: NMSE Comparison (dB)
    % =========================================================================
    fig2 = figure('Name', 'NMSE Comparison (dB)', 'Color', 'w', 'Position', [150 150 750 520], 'Visible', 'off');
    hold on; h_lines = []; h_labels = {};

    % Plot Inferred Models
    for i = 1:num_folders
        if ~all(isnan(m_nmse_db{i}))
            col = model_colors{mod(i-1, length(model_colors))+1};
            mrk = model_markers{mod(i-1, length(model_markers))+1};
            h = plot(SNRdB, m_nmse_db{i}, '-', 'Color', col, 'LineWidth', 2.0, ...
                'Marker', mrk, 'MarkerSize', 8, 'MarkerFaceColor', col);
            h_lines = [h_lines; h]; %#ok<AGROW>
            h_labels{end+1} = folder_labels{i}; %#ok<AGROW>
        end
    end

    % Plot LI Benchmark Average
    if ~all(isnan(li_nmse_db_avg))
        h = plot(SNRdB, li_nmse_db_avg, li_style, 'Color', li_color, 'LineWidth', 1.8, ...
            'Marker', li_marker, 'MarkerSize', 7, 'MarkerFaceColor', li_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'LI Benchmark '; %#ok<AGROW>
    end

    % Plot MMSE Benchmark Average
    if ~all(isnan(mmse_nmse_db_avg))
        h = plot(SNRdB, mmse_nmse_db_avg, mmse_style, 'Color', mmse_color, 'LineWidth', 1.8, ...
            'Marker', mmse_marker, 'MarkerSize', 7, 'MarkerFaceColor', mmse_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'MMSE Benchmark '; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Mean Squared Error (NMSE) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
    title('NMSE Performance Comparison (dB)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    if ~isempty(h_lines)
        legend(h_lines, h_labels, 'Location', 'northeast', 'FontSize', 10);
    end
    hold off;

    nmse_pdf_path = fullfile(output_folder, 'NMSE_comparison.pdf');
    try exportgraphics(fig2, nmse_pdf_path, 'ContentType', 'vector'); catch, saveas(fig2, nmse_pdf_path); end
    safe_printf('Saved NMSE plot: %s\n', nmse_pdf_path);
    close(fig2);

    % =========================================================================
    % FIGURE 3: SSIM Comparison
    % =========================================================================
    fig3 = figure('Name', 'SSIM Comparison', 'Color', 'w', 'Position', [200 200 750 520], 'Visible', 'off');
    hold on; h_lines = []; h_labels = {};

    % Plot Inferred Models
    for i = 1:num_folders
        if ~all(isnan(m_ssim{i}))
            col = model_colors{mod(i-1, length(model_colors))+1};
            mrk = model_markers{mod(i-1, length(model_markers))+1};
            h = plot(SNRdB, m_ssim{i}, '-', 'Color', col, 'LineWidth', 2.0, ...
                'Marker', mrk, 'MarkerSize', 8, 'MarkerFaceColor', col);
            h_lines = [h_lines; h]; %#ok<AGROW>
            h_labels{end+1} = folder_labels{i}; %#ok<AGROW>
        end
    end

    % Plot LI Benchmark Average
    if ~all(isnan(li_ssim_avg))
        h = plot(SNRdB, li_ssim_avg, li_style, 'Color', li_color, 'LineWidth', 1.8, ...
            'Marker', li_marker, 'MarkerSize', 7, 'MarkerFaceColor', li_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'LI Benchmark '; %#ok<AGROW>
    end

    % Plot MMSE Benchmark Average
    if ~all(isnan(mmse_ssim_avg))
        h = plot(SNRdB, mmse_ssim_avg, mmse_style, 'Color', mmse_color, 'LineWidth', 1.8, ...
            'Marker', mmse_marker, 'MarkerSize', 7, 'MarkerFaceColor', mmse_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'MMSE Benchmark '; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Structural Similarity Index (SSIM)', 'FontSize', 12, 'FontWeight', 'bold');
    title('SSIM Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; ylim([0 1.05]); set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    if ~isempty(h_lines)
        legend(h_lines, h_labels, 'Location', 'southeast', 'FontSize', 10);
    end
    hold off;

    ssim_pdf_path = fullfile(output_folder, 'SSIM_comparison.pdf');
    try exportgraphics(fig3, ssim_pdf_path, 'ContentType', 'vector'); catch, saveas(fig3, ssim_pdf_path); end
    safe_printf('Saved SSIM plot: %s\n', ssim_pdf_path);
    close(fig3);

    % =========================================================================
    % FIGURE 4: BER Comparison
    % =========================================================================
    fig4 = figure('Name', 'BER Comparison', 'Color', 'w', 'Position', [250 250 750 520], 'Visible', 'off');
    hold on; h_lines = []; h_labels = {};

    % Plot Inferred Models
    for i = 1:num_folders
        if ~all(isnan(m_ber{i}))
            col = model_colors{mod(i-1, length(model_colors))+1};
            mrk = model_markers{mod(i-1, length(model_markers))+1};
            h = semilogy(SNRdB, m_ber{i}, '-', 'Color', col, 'LineWidth', 2.0, ...
                'Marker', mrk, 'MarkerSize', 8, 'MarkerFaceColor', col);
            h_lines = [h_lines; h]; %#ok<AGROW>
            h_labels{end+1} = folder_labels{i}; %#ok<AGROW>
        end
    end

    % Plot LI Benchmark Average
    if ~all(isnan(li_ber_avg))
        h = semilogy(SNRdB, li_ber_avg, li_style, 'Color', li_color, 'LineWidth', 1.8, ...
            'Marker', li_marker, 'MarkerSize', 7, 'MarkerFaceColor', li_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'LI Benchmark '; %#ok<AGROW>
    end

    % Plot MMSE Benchmark Average
    if ~all(isnan(mmse_ber_avg))
        h = semilogy(SNRdB, mmse_ber_avg, mmse_style, 'Color', mmse_color, 'LineWidth', 1.8, ...
            'Marker', mmse_marker, 'MarkerSize', 7, 'MarkerFaceColor', mmse_color);
        h_lines = [h_lines; h]; %#ok<AGROW>
        h_labels{end+1} = 'MMSE Benchmark '; %#ok<AGROW>
    end

    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
    title('BER Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on; set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    if ~isempty(h_lines)
        legend(h_lines, h_labels, 'Location', 'southwest', 'FontSize', 10);
    end
    hold off;

    ber_pdf_path = fullfile(output_folder, 'BER_comparison.pdf');
    try exportgraphics(fig4, ber_pdf_path, 'ContentType', 'vector'); catch, saveas(fig4, ber_pdf_path); end
    safe_printf('Saved BER plot: %s\n', ber_pdf_path);
    close(fig4);

    % Save Aggregated MAT File
    save_struct = struct();
    save_struct.SNRdB = SNRdB;
    save_struct.folder_labels = folder_labels;
    save_struct.folders = folders;

    for i = 1:num_folders
        clean_lbl = matlab.lang.makeValidName(folder_labels{i});
        save_struct.models.(clean_lbl).mse     = m_mse{i};
        save_struct.models.(clean_lbl).nmse    = m_nmse{i};
        save_struct.models.(clean_lbl).nmse_db = m_nmse_db{i};
        save_struct.models.(clean_lbl).ssim    = m_ssim{i};
        save_struct.models.(clean_lbl).ber     = m_ber{i};
    end

    save_struct.benchmarks.li.mse_avg     = li_mse_avg;
    save_struct.benchmarks.li.nmse_avg    = li_nmse_avg;
    save_struct.benchmarks.li.nmse_db_avg = li_nmse_db_avg;
    save_struct.benchmarks.li.ssim_avg    = li_ssim_avg;
    save_struct.benchmarks.li.ber_avg     = li_ber_avg;

    save_struct.benchmarks.mmse.mse_avg     = mmse_mse_avg;
    save_struct.benchmarks.mmse.nmse_avg    = mmse_nmse_avg;
    save_struct.benchmarks.mmse.nmse_db_avg = mmse_nmse_db_avg;
    save_struct.benchmarks.mmse.ssim_avg    = mmse_ssim_avg;
    save_struct.benchmarks.mmse.ber_avg     = mmse_ber_avg;

    mat_out_path = fullfile(output_folder, 'synthesized_comparison_results.mat');
    save(mat_out_path, '-struct', 'save_struct');
    safe_printf('Saved synthesized comparison MAT file: %s\n', mat_out_path);

    results = save_struct;
    safe_printf('\nSynthesized results comparison complete!\n');
end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function safe_printf(fmt, varargin)
    try
        fprintf(fmt, varargin{:});
    catch
        % Fallback for headless environments without active standard output stream
    end
end

function val = extract_field(data, explicit_keys, prefix, exclude_tokens, expected_len, is_db_field)
    if nargin < 6, is_db_field = false; end
    val = NaN(1, expected_len);

    % 1. Try explicit keys first
    for k = 1:length(explicit_keys)
        key = explicit_keys{k};
        if isfield(data, key)
            v = double(data.(key));
            if numel(v) == expected_len
                val = v(:).';
                return;
            end
        end
    end

    % 2. Search dynamic field names
    all_fields = fieldnames(data);
    for f = 1:length(all_fields)
        fn = all_fields{f};
        fn_lower = lower(fn);

        if startsWith(fn_lower, lower(prefix))
            % Check exclusions
            skip = false;
            for ex = 1:length(exclude_tokens)
                if contains(fn_lower, lower(exclude_tokens{ex}))
                    skip = true;
                    break;
                end
            end
            if skip, continue; end

            % Check dB condition matching
            if is_db_field && ~endsWith(fn_lower, 'db')
                continue;
            elseif ~is_db_field && endsWith(fn_lower, 'db')
                continue;
            end

            v = double(data.(fn));
            if numel(v) == expected_len
                val = v(:).';
                return;
            end
        end
    end
end

function avg_vec = compute_benchmark_avg(cell_data, expected_len)
    num_folders = length(cell_data);
    mat = NaN(num_folders, expected_len);
    for i = 1:num_folders
        if ~isempty(cell_data{i}) && numel(cell_data{i}) == expected_len
            mat(i, :) = cell_data{i};
        end
    end
    avg_vec = mean(mat, 1, 'omitnan');
end

function print_summary_table(metric_name, SNRdB, model_cell, li_avg, mmse_avg, folder_labels)
    num_folders = length(model_cell);
    num_snr = length(SNRdB);

    safe_printf('------------------------------------------------------------------------\n');
    safe_printf(' Summary Table: %s\n', metric_name);
    safe_printf('------------------------------------------------------------------------\n');
    safe_printf('%-10s', 'SNR (dB)');
    for i = 1:num_folders
        safe_printf('%-24s', truncate_str(folder_labels{i}, 23));
    end
    safe_printf('%-22s%-22s\n', 'LI Benchmark ', 'MMSE Benchmark ');
    safe_printf('%s\n', repmat('-', 1, 10 + 24*num_folders + 44));

    for s = 1:num_snr
        safe_printf('%-10.1f', SNRdB(s));
        for i = 1:num_folders
            val = model_cell{i}(s);
            if isnan(val)
                safe_printf('%-24s', 'N/A');
            elseif strcmp(metric_name, 'BER') || strcmp(metric_name, 'MSE')
                safe_printf('%-24.6e', val);
            else
                safe_printf('%-24.4f', val);
            end
        end

        % LI Avg
        if isnan(li_avg(s))
            safe_printf('%-22s', 'N/A');
        elseif strcmp(metric_name, 'BER') || strcmp(metric_name, 'MSE')
            safe_printf('%-22.6e', li_avg(s));
        else
            safe_printf('%-22.4f', li_avg(s));
        end

        % MMSE Avg
        if isnan(mmse_avg(s))
            safe_printf('%-22s', 'N/A');
        elseif strcmp(metric_name, 'BER') || strcmp(metric_name, 'MSE')
            safe_printf('%-22.6e', mmse_avg(s));
        else
            safe_printf('%-22.4f', mmse_avg(s));
        end
        safe_printf('\n');
    end
    safe_printf('\n');
end

function s_out = truncate_str(s_in, max_len)
    if length(s_in) > max_len
        s_out = [s_in(1:max_len-3) '...'];
    else
        s_out = s_in;
    end
end
