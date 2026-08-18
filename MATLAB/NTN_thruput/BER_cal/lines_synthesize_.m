target_folder = 'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\BER_cal\inferences_batch\DUR100__A100_2p18e9_600km_30kHz_LSSequence';

lines_synthesize(target_folder)

function results = lines_synthesize(target_folder)
% LINES_SYNTHESIZE Synthesize and plot performance comparison curves across SNR subfolders.
%
% to synthesize results of trained models (on Source domain) and of
% LS_Sequence inference
%
% Supports two modes based on target directory structure:
%   Mode 1 (Inference Results):
%     Subfolders: SNR_x containing BER_performance_results.mat
%     Exports: BER_comparison.pdf, NMSE_comparison.pdf, synthesized_results.mat
%
%   Mode 2 (Single Source Trained Model):
%     Subfolders: LS_x, LI_x, or SNR_x containing evaluation_results.mat
%     Saves MAT: All train, val, test vectors for mmse, nmse, ssim of model and benchmark/input
%     Exports PDFs: MMSE_comparison.pdf, NMSE_comparison.pdf, SSIM_comparison.pdf
%
% Usage:
%   lines_synthesize()                     % Synthesizes default example folder
%   lines_synthesize(target_folder)        % Synthesizes specified folder
%

    if nargin < 1 || isempty(target_folder)
        % Default folder fallback
        script_dir = fileparts(mfilename('fullpath'));
        default_folder = fullfile(script_dir, 'inference_result', ...
            'DUR100ns_2p18G__A100ns_2p18G', ...
            'Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps');
        
        if exist(default_folder, 'dir')
            target_folder = default_folder;
        else
            target_folder = uigetdir(script_dir, 'Select folder containing SNR or LS/LI subfolders');
            if isequal(target_folder, 0)
                error('No folder selected.');
            end
        end
    end

    fprintf('========================================================================\n');
    fprintf('                 NTN Performance Lines Synthesize                       \n');
    fprintf('========================================================================\n');
    fprintf('Target Folder: %s\n\n', target_folder);

    % Detect mode by checking for evaluation_results.mat vs BER_performance_results.mat
    is_eval_mode = false;
    dir_items = dir(target_folder);
    
    for k = 1:length(dir_items)
        if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
            subname = dir_items(k).name;
            eval_file1 = fullfile(target_folder, subname, 'results', 'evaluation_results.mat');
            eval_file2 = fullfile(target_folder, subname, 'evaluation_results.mat');
            if exist(eval_file1, 'file') || exist(eval_file2, 'file')
                is_eval_mode = true;
                break;
            end
        end
    end

    if is_eval_mode
        fprintf('[Mode Detected]: Single Source Trained Model (evaluation_results.mat)\n\n');
        results = synthesize_evaluation_results(target_folder);
    else
        fprintf('[Mode Detected]: BER & NMSE Inference Results (BER_performance_results.mat)\n\n');
        results = synthesize_ber_results(target_folder);
    end
end

%% ========================================================================
%% MODE 2: Single Source Trained Model Synthesis (evaluation_results.mat)
%% ========================================================================
function results = synthesize_evaluation_results(target_folder)
    dir_items = dir(target_folder);
    subfolders = {};
    mat_files = {};
    snr_vals = [];

    for k = 1:length(dir_items)
        if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
            subname = dir_items(k).name;
            p1 = fullfile(target_folder, subname, 'results', 'evaluation_results.mat');
            p2 = fullfile(target_folder, subname, 'evaluation_results.mat');
            
            target_mat = '';
            if exist(p1, 'file')
                target_mat = p1;
            elseif exist(p2, 'file')
                target_mat = p2;
            end

            if ~isempty(target_mat)
                mat_data = load(target_mat);
                if isfield(mat_data, 'snr')
                    snr_val = double(mat_data.snr);
                else
                    tokens = regexp(subname, '[-+]?\d+(\.\d+)?', 'tokens');
                    if ~isempty(tokens)
                        snr_val = str2double(tokens{1}{1});
                    else
                        continue;
                    end
                end
                subfolders{end+1} = fullfile(target_folder, subname); %#ok<AGROW>
                mat_files{end+1} = target_mat; %#ok<AGROW>
                snr_vals(end+1) = snr_val; %#ok<AGROW>
            end
        end
    end

    if isempty(subfolders)
        error('No subfolders containing evaluation_results.mat found in:\n  %s', target_folder);
    end

    % Sort subfolders by SNRdB in ascending order
    [SNRdB_sorted, sort_idx] = sort(snr_vals);
    subfolders = subfolders(sort_idx);
    mat_files = mat_files(sort_idx);
    num_snr = length(SNRdB_sorted);

    % Inspect first mat file to determine all metric fields
    sample_mat = load(mat_files{1});
    all_fields = fieldnames(sample_mat);

    save_struct = struct();
    save_struct.SNRdB = SNRdB_sorted;
    save_struct.snr   = SNRdB_sorted;

    % Approach & Benchmark Naming based on folder name conventions
    [~, folder_leaf] = fileparts(target_folder);
    if contains(folder_leaf, 'Attention', 'IgnoreCase', true)
        model_name = 'LS + Attention';
        benchmark_name = 'LI Benchmark';
    elseif contains(folder_leaf, 'Clipped', 'IgnoreCase', true)
        model_name = 'LI + CNN';
        benchmark_name = 'LI Input';
    else
        model_name = 'LS + CNN';
        benchmark_name = 'LS Input';
    end
    save_struct.model_name = model_name;
    save_struct.benchmark_name = benchmark_name;

    if isfield(sample_mat, 'input_type')
        if iscell(sample_mat.input_type)
            save_struct.input_type = sample_mat.input_type{1};
        else
            save_struct.input_type = char(sample_mat.input_type);
        end
    end
    if isfield(sample_mat, 'best_epoch')
        save_struct.best_epoch = double(sample_mat.best_epoch);
    end

    % Collect all metric fields (mmse_*, nmse_*, ssim_*)
    metric_fields = {};
    for f = 1:length(all_fields)
        fn = all_fields{f};
        if iscolumn_metric_field(fn)
            metric_fields{end+1} = fn; %#ok<AGROW>
            save_struct.(fn) = zeros(1, num_snr);
        end
    end

    % Aggregate metric vectors across sorted SNR subfolders
    for i = 1:num_snr
        data = load(mat_files{i});
        for m = 1:length(metric_fields)
            fn = metric_fields{m};
            if isfield(data, fn)
                save_struct.(fn)(i) = double(data.(fn));
            end
        end
    end

    % Display console summary table for Test set
    fprintf('Discovered %d SNR evaluation subfolders (SNR = %s):\n', num_snr, mat2str(SNRdB_sorted));
    fprintf('Approach: %s | Benchmark: %s\n\n', model_name, benchmark_name);
    
    % Find benchmark test keys dynamically (_li_benchmark_test or _input_test)
    bm_nmse_db_key = find_key(metric_fields, {'nmse_li_benchmark_test_db', 'nmse_input_test_db'});
    bm_ssim_key    = find_key(metric_fields, {'ssim_li_benchmark_test', 'ssim_input_test'});
    bm_mmse_key    = find_key(metric_fields, {'mmse_li_benchmark_test', 'mmse_input_test'});

    fprintf('Test Set Evaluation Results Summary:\n');
    fprintf('%-8s | %-14s %-14s | %-12s %-12s | %-12s %-12s\n', ...
        'SNR (dB)', [model_name ' NMSEdB'], [benchmark_name ' NMSEdB'], ...
        [model_name ' SSIM'], [benchmark_name ' SSIM'], ...
        [model_name ' MMSE'], [benchmark_name ' MMSE']);
    fprintf('%s\n', repmat('-', 1, 100));
    for i = 1:num_snr
        m_nmse_db = save_struct.nmse_test_db(i);
        b_nmse_db = get_val(save_struct, bm_nmse_db_key, i);
        m_ssim    = save_struct.ssim_test(i);
        b_ssim    = get_val(save_struct, bm_ssim_key, i);
        m_mmse    = save_struct.mmse_test(i);
        b_mmse    = get_val(save_struct, bm_mmse_key, i);

        fprintf('%-8.1f | %-14.2f %-14.2f | %-12.4f %-12.4f | %-12.3e %-12.3e\n', ...
            SNRdB_sorted(i), m_nmse_db, b_nmse_db, m_ssim, b_ssim, m_mmse, b_mmse);
    end
    fprintf('\n');

    % Save synthesized results MAT file inside target folder
    mat_out_path = fullfile(target_folder, 'synthesized_results.mat');
    save(mat_out_path, '-struct', 'save_struct');
    fprintf('Saved synthesized MAT file: %s\n', mat_out_path);

    % Plot 1: MMSE Comparison (Test set)
    fig1 = figure('Name', 'MMSE Comparison', 'Color', 'w', 'Position', [100 100 700 500], 'Visible', 'off');
    hold on;
    h1 = semilogy(SNRdB_sorted, save_struct.mmse_test, '-^', 'Color', [0.8500 0.3250 0.0980], ...
        'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    if ~isempty(bm_mmse_key)
        h2 = semilogy(SNRdB_sorted, save_struct.(bm_mmse_key), '--o', 'Color', [0 0.4470 0.7410], ...
            'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0.4470 0.7410]);
        legend([h1, h2], {model_name, benchmark_name}, 'Location', 'northeast', 'FontSize', 10);
    else
        legend(h1, {model_name}, 'Location', 'northeast', 'FontSize', 10);
    end
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Mean Squared Error (MMSE)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('MMSE Comparison (Test Set): %s vs. %s', model_name, benchmark_name), 'FontSize', 13, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    hold off;

    mmse_pdf_path = fullfile(target_folder, 'MMSE_comparison.pdf');
    try exportgraphics(fig1, mmse_pdf_path, 'ContentType', 'vector'); catch, saveas(fig1, mmse_pdf_path); end
    fprintf('Saved MMSE figure: %s\n', mmse_pdf_path);
    close(fig1);

    % Plot 2: NMSE Comparison (dB) (Test set)
    fig2 = figure('Name', 'NMSE Comparison', 'Color', 'w', 'Position', [150 150 700 500], 'Visible', 'off');
    hold on;
    h1 = plot(SNRdB_sorted, save_struct.nmse_test_db, '-^', 'Color', [0.8500 0.3250 0.0980], ...
        'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    if ~isempty(bm_nmse_db_key)
        h2 = plot(SNRdB_sorted, save_struct.(bm_nmse_db_key), '--o', 'Color', [0 0.4470 0.7410], ...
            'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0.4470 0.7410]);
        legend([h1, h2], {model_name, benchmark_name}, 'Location', 'northeast', 'FontSize', 10);
    else
        legend(h1, {model_name}, 'Location', 'northeast', 'FontSize', 10);
    end
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Mean Squared Error (NMSE) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('NMSE Comparison (Test Set): %s vs. %s', model_name, benchmark_name), 'FontSize', 13, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    hold off;

    nmse_pdf_path = fullfile(target_folder, 'NMSE_comparison.pdf');
    try exportgraphics(fig2, nmse_pdf_path, 'ContentType', 'vector'); catch, saveas(fig2, nmse_pdf_path); end
    fprintf('Saved NMSE figure: %s\n', nmse_pdf_path);
    close(fig2);

    % Plot 3: SSIM Comparison (Test set)
    fig3 = figure('Name', 'SSIM Comparison', 'Color', 'w', 'Position', [200 200 700 500], 'Visible', 'off');
    hold on;
    h1 = plot(SNRdB_sorted, save_struct.ssim_test, '-^', 'Color', [0.8500 0.3250 0.0980], ...
        'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    if ~isempty(bm_ssim_key)
        h2 = plot(SNRdB_sorted, save_struct.(bm_ssim_key), '--o', 'Color', [0 0.4470 0.7410], ...
            'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', [0 0.4470 0.7410]);
        legend([h1, h2], {model_name, benchmark_name}, 'Location', 'southeast', 'FontSize', 10);
    else
        legend(h1, {model_name}, 'Location', 'southeast', 'FontSize', 10);
    end
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Structural Similarity Index (SSIM)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('SSIM Comparison (Test Set): %s vs. %s', model_name, benchmark_name), 'FontSize', 13, 'FontWeight', 'bold');
    grid on; box on;
    ylim([0 1.05]);
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    hold off;

    ssim_pdf_path = fullfile(target_folder, 'SSIM_comparison.pdf');
    try exportgraphics(fig3, ssim_pdf_path, 'ContentType', 'vector'); catch, saveas(fig3, ssim_pdf_path); end
    fprintf('Saved SSIM figure: %s\n', ssim_pdf_path);
    close(fig3);

    results = save_struct;
    fprintf('Single source model synthesis complete!\n');
end

%% ========================================================================
%% MODE 1: BER & NMSE Inference Results Synthesis (BER_performance_results.mat)
%% ========================================================================
function results = synthesize_ber_results(target_folder)
    dir_items = dir(target_folder);
    snr_subfolders = {};
    snr_vals = [];
    
    for k = 1:length(dir_items)
        if dir_items(k).isdir && ~strcmp(dir_items(k).name, '.') && ~strcmp(dir_items(k).name, '..')
            folder_name = dir_items(k).name;
            mat_path = fullfile(target_folder, folder_name, 'BER_performance_results.mat');
            
            if exist(mat_path, 'file')
                mat_data = load(mat_path);
                if isfield(mat_data, 'SNRdB')
                    snr_val = double(mat_data.SNRdB);
                else
                    tokens = regexp(folder_name, 'SNR_(-?\d+(\.\d+)?)', 'tokens');
                    if ~isempty(tokens)
                        snr_val = str2double(tokens{1}{1});
                    else
                        continue;
                    end
                end
                snr_subfolders{end+1} = fullfile(target_folder, folder_name); %#ok<AGROW>
                snr_vals(end+1) = snr_val; %#ok<AGROW>
            end
        end
    end

    if isempty(snr_subfolders)
        error('No subfolders containing BER_performance_results.mat found in:\n  %s', target_folder);
    end

    [SNRdB_sorted, sort_idx] = sort(snr_vals);
    snr_subfolders = snr_subfolders(sort_idx);
    num_snr = length(SNRdB_sorted);

    fprintf('Discovered %d SNR subfolders (SNRdB = %s):\n', num_snr, mat2str(SNRdB_sorted));
    for i = 1:num_snr
        [~, sf_name] = fileparts(snr_subfolders{i});
        fprintf('  [%d] %s -> SNRdB = %.1f\n', i, sf_name, SNRdB_sorted(i));
    end
    fprintf('\n');

    sample_mat = load(fullfile(snr_subfolders{1}, 'BER_performance_results.mat'));
    mat_fields = fieldnames(sample_mat);
    
    ber_fields = {};
    nmse_fields = {};
    
    for f = 1:length(mat_fields)
        fname = mat_fields{f};
        if startsWith(fname, 'ber_') && ~strcmp(fname, 'ber_performance')
            ber_fields{end+1} = fname; %#ok<AGROW>
        elseif startsWith(fname, 'nmse_') && ~strcmp(fname, 'nmse_performance')
            nmse_fields{end+1} = fname; %#ok<AGROW>
        end
    end

    ber_data = struct();
    nmse_data = struct();
    nmse_db_data = struct();

    for b = 1:length(ber_fields)
        ber_data.(ber_fields{b}) = zeros(1, num_snr);
    end
    for n = 1:length(nmse_fields)
        nmse_data.(nmse_fields{n}) = zeros(1, num_snr);
        nmse_db_data.(nmse_fields{n}) = zeros(1, num_snr);
    end

    for i = 1:num_snr
        mat_path = fullfile(snr_subfolders{i}, 'BER_performance_results.mat');
        data = load(mat_path);
        
        for b = 1:length(ber_fields)
            fn = ber_fields{b};
            if isfield(data, fn)
                vals = double(data.(fn));
                ber_data.(fn)(i) = mean(vals(:), 'omitnan');
            end
        end
        
        for n = 1:length(nmse_fields)
            fn = nmse_fields{n};
            if isfield(data, fn)
                vals = double(data.(fn));
                mean_linear = mean(vals(:), 'omitnan');
                nmse_data.(fn)(i) = mean_linear;
                nmse_db_data.(fn)(i) = 10 * log10(mean_linear);
            end
        end
    end

    fprintf('Synthesized BER Performance Results:\n');
    fprintf('%-10s', 'SNR (dB)');
    for b = 1:length(ber_fields)
        fprintf('%-18s', ber_fields{b});
    end
    fprintf('\n%s\n', repmat('-', 1, 10 + 18*length(ber_fields)));
    for i = 1:num_snr
        fprintf('%-10.1f', SNRdB_sorted(i));
        for b = 1:length(ber_fields)
            fprintf('%-18.6e', ber_data.(ber_fields{b})(i));
        end
        fprintf('\n');
    end

    fprintf('\nSynthesized NMSE Performance Results (dB):\n');
    fprintf('%-10s', 'SNR (dB)');
    for n = 1:length(nmse_fields)
        fprintf('%-18s', nmse_fields{n});
    end
    fprintf('\n%s\n', repmat('-', 1, 10 + 18*length(nmse_fields)));
    for i = 1:num_snr
        fprintf('%-10.1f', SNRdB_sorted(i));
        for n = 1:length(nmse_fields)
            fprintf('%-18.2f', nmse_db_data.(nmse_fields{n})(i));
        end
        fprintf('\n');
    end
    fprintf('\n');

    % Figure 1: BER vs SNRdB
    fig1 = figure('Name', 'BER Comparison', 'Color', 'w', 'Position', [100 100 700 500], 'Visible', 'off');
    hold on;
    legend_entries_ber = {};
    h_ber = [];
    
    for b = 1:length(ber_fields)
        fn = ber_fields{b};
        [style, color, marker, label] = get_method_style(fn);
        h = semilogy(SNRdB_sorted, ber_data.(fn), [style marker], ...
            'Color', color, 'LineWidth', 1.8, 'MarkerSize', 7, ...
            'MarkerFaceColor', color);
        h_ber = [h_ber; h]; %#ok<AGROW>
        legend_entries_ber{end+1} = label; %#ok<AGROW>
    end
    
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
    title('BER Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend(h_ber, legend_entries_ber, 'Location', 'southwest', 'FontSize', 10);
    hold off;

    ber_pdf_path = fullfile(target_folder, 'BER_comparison.pdf');
    try exportgraphics(fig1, ber_pdf_path, 'ContentType', 'vector'); catch, saveas(fig1, ber_pdf_path); end
    fprintf('Saved BER comparison figure: %s\n', ber_pdf_path);
    close(fig1);

    % Figure 2: NMSE vs SNRdB
    fig2 = figure('Name', 'NMSE Comparison', 'Color', 'w', 'Position', [150 150 700 500], 'Visible', 'off');
    hold on;
    legend_entries_nmse = {};
    h_nmse = [];
    
    for n = 1:length(nmse_fields)
        fn = nmse_fields{n};
        [style, color, marker, label] = get_method_style(fn);
        h = plot(SNRdB_sorted, nmse_db_data.(fn), [style marker], ...
            'Color', color, 'LineWidth', 1.8, 'MarkerSize', 7, ...
            'MarkerFaceColor', color);
        h_nmse = [h_nmse; h]; %#ok<AGROW>
        legend_entries_nmse{end+1} = label; %#ok<AGROW>
    end
    
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Mean Squared Error (NMSE) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
    title('NMSE Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'YMinorGrid', 'on', 'FontSize', 11);
    legend(h_nmse, legend_entries_nmse, 'Location', 'northeast', 'FontSize', 10);
    hold off;

    nmse_pdf_path = fullfile(target_folder, 'NMSE_comparison.pdf');
    try exportgraphics(fig2, nmse_pdf_path, 'ContentType', 'vector'); catch, saveas(fig2, nmse_pdf_path); end
    fprintf('Saved NMSE comparison figure: %s\n', nmse_pdf_path);
    close(fig2);

    save_struct = struct();
    save_struct.SNRdB = SNRdB_sorted;
    save_struct.BER = ber_data;
    save_struct.NMSE = nmse_data;
    save_struct.NMSE_dB = nmse_db_data;
    
    for b = 1:length(ber_fields)
        fn = ber_fields{b};
        save_struct.(fn) = ber_data.(fn);
    end
    for n = 1:length(nmse_fields)
        fn = nmse_fields{n};
        save_struct.(fn) = nmse_data.(fn);
        save_struct.([fn '_dB']) = nmse_db_data.(fn);
    end
    
    mat_out_path = fullfile(target_folder, 'synthesized_results.mat');
    save(mat_out_path, '-struct', 'save_struct');
    fprintf('Saved synthesized MAT file: %s\n', mat_out_path);

    results = save_struct;
    fprintf('BER synthesis complete!\n');
end

%% Helper functions
function tf = iscolumn_metric_field(fn)
    valid_prefixes = {'mmse_', 'nmse_', 'ssim_'};
    tf = false;
    for p = 1:length(valid_prefixes)
        if startsWith(fn, valid_prefixes{p})
            tf = true;
            return;
        end
    end
end

function found_key = find_key(field_list, candidate_keys)
    found_key = '';
    for c = 1:length(candidate_keys)
        cand = candidate_keys{c};
        if ismember(cand, field_list)
            found_key = cand;
            return;
        end
    end
end

function v = get_val(s, key, idx)
    if ~isempty(key) && isfield(s, key) && length(s.(key)) >= idx
        v = s.(key)(idx);
    else
        v = NaN;
    end
end

function [style, color, marker, label] = get_method_style(fn)
    switch lower(fn)
        case {'ber_li', 'nmse_li'}
            style = '--'; color = [0 0.4470 0.7410]; marker = 'o';
            label = 'LS + Linear Interpolation';
        case {'ber_mmse', 'nmse_mmse'}
            style = '-'; color = [0 0 0]; marker = 's';
            label = 'LS + MMSE Benchmark';
        case {'ber_ls_atten', 'nmse_ls_atten'}
            style = '-'; color = [0.8500 0.3250 0.0980]; marker = '^';
            label = 'LS + Attention';
        case {'ber_ls_cnn', 'nmse_ls_cnn'}
            style = '-'; color = [0.4940 0.1840 0.5560]; marker = 'd';
            label = 'LS + ResNet DnCNN';
        case {'ber_li_cnn', 'nmse_li_cnn'}
            style = '-'; color = [0.4660 0.6740 0.1880]; marker = 'v';
            label = 'LI + ResNet DnCNN';
        otherwise
            style = '-'; color = [0.6350 0.0780 0.1840]; marker = 'x';
            clean_name = strrep(strrep(fn, 'ber_', ''), 'nmse_', '');
            label = strrep(clean_name, '_', '\_');
    end
end
