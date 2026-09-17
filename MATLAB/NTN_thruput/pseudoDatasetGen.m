%% Pseudo Dataset Generation for NTN Channel Estimation/Prediction
% 1. Generating the pseudo ground truth H_perfect via Fourier Domain Adaptation (FDA)
%    in the Delay-Doppler (DD) domain.
% 2. Generating the pseudo observation H_li via simulated 5G NR DM-RS pilot
%    transmission, AWGN noise injection, Least Squares (LS) estimation, and
%    2D linear interpolation.
%
% USAGE EXAMPLES:
%   1. In MATLAB Editor:
%          Click the "Run" button (or press F5). Runs with all default settings.
%
%   2. In MATLAB Command Window:
%          pseudoDatasetGen
%          pseudoDatasetGen('SNR_dB', [-10, 0, 5], 'w_window', 3)
%          pseudoDatasetGen('sourcePath', 'my_source.mat', 'targetFolder', 'my_target_dir')
%
%   3. From System Command Prompt (Windows cmd / PowerShell / Terminal):
%          matlab -batch "pseudoDatasetGen"
%          matlab -batch "pseudoDatasetGen('SNR_dB', [-15:5:5], 'w_window', [3, 5])"
%          matlab -batch "pseudoDatasetGen('--snr', '[-10, 0]', '--w_window', '3')"
%          matlab -nosplash -nodesktop -r "pseudoDatasetGen('SNR_dB', [-10, 0]); exit"

function pseudoDatasetGen(varargin)
    



    
    % -------------------------------------------------------------------------
    % 0. Working Directory & Path Setup
    % -------------------------------------------------------------------------
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

    helper_dir = fullfile(script_dir, '..', 'helper');
    if exist(helper_dir, 'dir')
        addpath(helper_dir);
    else
        addpath('..\helper\');
    end

    % -------------------------------------------------------------------------
    % 1. Input Parser Setup
    % -------------------------------------------------------------------------
    % Normalize arguments: strip leading '--' or '-' for CLI friendliness
    clean_args = varargin;
    for i = 1:2:length(clean_args)
        if ischar(clean_args{i}) || isstring(clean_args{i})
            arg_str = char(clean_args{i});
            clean_args{i} = regexprep(arg_str, '^--?', '');
        end
    end

    p = inputParser;
    p.FunctionName = 'pseudoDatasetGen';
    p.CaseSensitive = false;
    p.KeepUnmatched = true;

    % Define parameters and their defaults
    addParameter(p, 'sourcePath', fullfile(script_dir, 'generatedChannel_TDL_D_30_simple', 'SNR_10dB', 'matlabNTN.mat'));
    addParameter(p, 'targetFolder', fullfile(script_dir, 'generatedChannel_TDL_A_300_simple'));
    addParameter(p, 'outputFolder', fullfile(script_dir, 'pseudoChannelD30_A300'));
    addParameter(p, 'SNR_dB', -15:5:5);
    addParameter(p, 'snr', []); % alias for SNR_dB
    addParameter(p, 'w_window', [3, 5]);
    addParameter(p, 'h_window', 13);
    addParameter(p, 'scs', 30);
    addParameter(p, 'nSizeGrid', 11);

    parse(p, clean_args{:});

    % Extract and resolve parameters
    sourcePath = p.Results.sourcePath;
    targetFolder = p.Results.targetFolder;
    outputFolder = p.Results.outputFolder;
    
    if ~isempty(p.Results.snr)
        SNR_dB = parseNumeric(p.Results.snr);
    else
        SNR_dB = parseNumeric(p.Results.SNR_dB);
    end
    w_window = parseNumeric(p.Results.w_window);
    h_window = parseNumeric(p.Results.h_window);
    scs = parseNumeric(p.Results.scs);
    nSizeGrid = parseNumeric(p.Results.nSizeGrid);

    fprintf('====================================================\n');
    fprintf('           PSEUDO DATASET GENERATION                \n');
    fprintf('====================================================\n');
    fprintf(' Source File  : %s\n', sourcePath);
    fprintf(' Target Folder: %s\n', targetFolder);
    fprintf(' Output Folder: %s\n', outputFolder);
    fprintf(' SNR range    : %s dB\n', mat2str(SNR_dB));
    fprintf(' W Window(s)  : %s\n', mat2str(w_window));
    fprintf(' H Window     : %d\n', h_window);
    fprintf(' SCS / NGrid  : %d kHz / %d RBs\n', scs, nSizeGrid);
    fprintf('====================================================\n\n');

    % -------------------------------------------------------------------------
    % 2. 5G NR DM-RS Pilot Grid Configuration
    % -------------------------------------------------------------------------
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing = scs;
    carrier.CyclicPrefix = "Normal";
    carrier.NSizeGrid = nSizeGrid;
    carrier.NCellID = 1;

    pdsch = nrPDSCHConfig;
    pdsch.PRBSet = 0:carrier.NSizeGrid-1;
    pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot];
    pdsch.MappingType = "A";

    pdsch.DMRS.DMRSPortSet = []; % Auto-configure DM-RS ports
    pdsch.DMRS.DMRSTypeAPosition = 2;
    pdsch.DMRS.DMRSLength = 1;
    pdsch.DMRS.DMRSAdditionalPosition = 2;
    pdsch.DMRS.DMRSConfigurationType = 2;
    pdsch.DMRS.NumCDMGroupsWithoutData = 1;
    pdsch.DMRS.NIDNSCID = 1;
    pdsch.DMRS.NSCID = 0;

    refDMRSSymbols = nrPDSCHDMRS(carrier, pdsch);
    refDMRSIndices = nrPDSCHDMRSIndices(carrier, pdsch);

    nSubcarriers = carrier.NSizeGrid * 12;
    nSymbols = carrier.SymbolsPerSlot;
    txGrid = zeros(nSubcarriers, nSymbols);
    txGrid(refDMRSIndices) = refDMRSSymbols;

    % -------------------------------------------------------------------------
    % 3. Source Domain Data Loading
    % -------------------------------------------------------------------------
    if ~exist(sourcePath, 'file')
        error('Source file not found: %s', sourcePath);
    end
    fprintf('Loading source domain: %s ...\n', sourcePath);
    source_domain = load(sourcePath);

    % -------------------------------------------------------------------------
    % 4. Generation Loop (w_window x SNR_dB)
    % -------------------------------------------------------------------------
    for w_w = w_window
        for snr_db = SNR_dB
            target_path = fullfile(targetFolder, ['SNR_', num2str(snr_db), 'dB'], 'matlabNTN.mat');
            if ~exist(target_path, 'file')
                warning('Target file not found: %s. Skipping this SNR.', target_path);
                continue;
            end
            fprintf('Processing w_w=%d, SNR=%d dB ...\n', w_w, snr_db);
            target_domain = load(target_path);

            % Step 1: Fourier Domain Adaptation (FDA) in Delay-Doppler domain
            pseudo_label_li = FTranslate_bulk(source_domain.H_perfect, target_domain.H_li, h_window, w_w);
            H_perfect = pseudo_label_li; % 14 x 132 x N_samples
            H_pseudo = permute(pseudo_label_li, [2 1 3]); % 132 x 14 x N_samples

            H_li = zeros(size(pseudo_label_li));
            H_li_clip = zeros(size(pseudo_label_li));

            nmse_pseudo_li = 0;
            nSamples = size(pseudo_label_li, 3);

            for n = 1:nSamples
                H_pseudo_n = H_pseudo(:, :, n);
                Y_received_noNoise = txGrid .* H_pseudo_n;

                % Noise Variance
                Ps = mean(abs(Y_received_noNoise(refDMRSIndices)).^2);
                sigma2 = Ps / (10^(snr_db / 10));

                % Complex AWGN
                Noise = sqrt(sigma2 / 2) * (randn(size(txGrid)) + 1i * randn(size(txGrid)));
                Y_received = Y_received_noNoise + Noise;

                % Least Square + Linear Interpolation
                [~, H_linear_n] = Lin_Interpolate(Y_received, refDMRSIndices, refDMRSSymbols);
                H_linear_n_ = crop_(H_linear_n, [1, 128], [3, 12]); % Crop boundary extrapolation

                H_li(:, :, n) = permute(H_linear_n, [2 1 3]);
                H_li_clip(:, :, n) = permute(H_linear_n_, [2 1 3]);

                % Calculate NMSE
                error_matrix = H_pseudo_n - H_linear_n_;
                squared_error = sum(abs(error_matrix).^2, 'all');
                true_power = sum(abs(H_pseudo_n).^2, 'all');
                nmse_pseudo_li_n = squared_error / true_power;
                nmse_pseudo_li = nmse_pseudo_li + nmse_pseudo_li_n;
            end

            nmse_li = nmse_pseudo_li / nSamples;

            % Save results
            save_folder = fullfile(outputFolder, ['SNR_', num2str(snr_db), 'dB_', num2str(h_window), 'x', num2str(w_w)]);
            if ~exist(save_folder, 'dir')
                mkdir(save_folder);
            end
            output_file = fullfile(save_folder, 'matlabNTN.mat');
            save(output_file, 'H_perfect', 'H_li', 'H_li_clip', 'nmse_li', '-v7.3');
            fprintf('  Saved to: %s (Average NMSE: %.4f / %.2f dB)\n', output_file, nmse_li, 10*log10(nmse_li));
        end
    end

    fprintf('\nAll tasks finished successfully!\n');
end

% =========================================================================
% HELPER & LOCAL FUNCTIONS
% =========================================================================

function val = parseNumeric(val)
    % Converts char/string representations like '[-15:5:5]' or '[3, 5]' to numeric array
    if ischar(val) || isstring(val)
        val_str = strtrim(char(val));
        num_val = str2num(val_str); %#ok<ST2NM>
        if ~isempty(num_val)
            val = num_val;
        end
    end
end

function [amplitude_spectrum, phase_spectrum] = F_extract_DD(channel_grid)
% F_EXTRACT Converts Time-Freq channel to Delay-Doppler domain
% Input: 
%   channel_grid: 132 (Subcarriers) x 14 (Symbols) Complex Matrix
% Output:
%   amplitude_spectrum: Magnitude in Delay-Doppler (The "Style")
%   phase_spectrum: Phase in Delay-Doppler (The "Content")

    % 1. Transform Subcarriers (Frequency) -> Delay
    grid_delay = ifft(channel_grid, [], 1);
    
    % 2. Transform Symbols (Time) -> Doppler
    grid_delay_doppler_raw = fft(grid_delay, [], 2);
    
    % 3. Shift BOTH dimensions to Center
    grid_dd_shifted = fftshift(grid_delay_doppler_raw);
    
    % 4. Extract Amplitude and Phase
    amplitude_spectrum = abs(grid_dd_shifted);
    phase_spectrum = angle(grid_dd_shifted);
end

function [mixed_img] = fda_mix_pixels(source_img, target_img, win_h_px, win_w_px)
    % center: target
    % outer: source
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
    source_img = permute(source_img, [2, 1, 3]);
    target_img = permute(target_img, [2, 1, 3]);    
    translate_img = zeros(size(target_img));
    target_img_ = zeros(size(target_img));

    for n = 1:size(target_img, 3)
        target_img_(:, :, n) = crop_(target_img(:, :, n), [1, 128], [3, 12]);
        translate_img(:, :, n) = FTranslate_single(source_img(:, :, n), target_img_(:, :, n), win_h_px, win_w_px);
    end

    translate_img = permute(translate_img, [2, 1, 3]);
end
