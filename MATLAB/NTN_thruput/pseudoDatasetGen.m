%% Pseudo Dataset Generation for NTN Channel Estimation/Prediction
% 1. Generating the pseudo ground truth H_perfect via Fourier Domain Adaptation (FDA)
%    in the Delay-Doppler (DD) domain.
% 2. Generating the pseudo observation H_li via simulated 5G NR DM-RS pilot
%    transmission, AWGN noise injection, Least Squares (LS) estimation, and
%    2D linear interpolation.

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

%% 1. Gen Tx grid DM-RS pilots
% Set waveform type and PDSCH numerology (SCS and CP type)
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 30;
carrier.CyclicPrefix = "Normal";
% Bandwidth in number of RBs (11 RBs at 30 kHz SCS for 5 MHz bandwidth)
carrier.NSizeGrid = 11;
% Physical layer cell identity
carrier.NCellID = 1;

% DM-RS and antenna port configuration (TS 38.211 Section 7.4.1.1)
pdsch = nrPDSCHConfig;
% PDSCH PRB allocation
pdsch.PRBSet = 0:carrier.NSizeGrid-1;
% Starting symbol and number of symbols of each PDSCH allocation
pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot];
pdsch.MappingType = "A";

pdsch.DMRS.DMRSPortSet = []; % Use empty to auto-configure the DM-RS ports
pdsch.DMRS.DMRSTypeAPosition = 2;
pdsch.DMRS.DMRSLength = 1;
pdsch.DMRS.DMRSAdditionalPosition = 2;
pdsch.DMRS.DMRSConfigurationType = 2;
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
pdsch.DMRS.NIDNSCID = 1;
pdsch.DMRS.NSCID = 0;

refDMRSSymbols = nrPDSCHDMRS(carrier, pdsch);
refDMRSIndices = nrPDSCHDMRSIndices(carrier, pdsch);

txGrid = zeros(132, 14);
txGrid(refDMRSIndices) = refDMRSSymbols;

%% 2. Generation Loop (w_window x SNR_dB)
source_domain = load("generatedChannel_TDL_D_30_simple\SNR_10dB\matlabNTN.mat");
SNR_dB = -15:5:5;
w_window = [3, 5];

for w_w = w_window
    for snr_db = SNR_dB
        target_domain = load(['generatedChannel_TDL_A_300_simple\SNR_', num2str(snr_db), 'dB\matlabNTN.mat']);
        
        pseudo_label_li = FTranslate_bulk(source_domain.H_perfect, target_domain.H_li, 13, w_w); % 13x3 13x5
        H_perfect = pseudo_label_li; % 14 x 132 x 2048
        H_pseudo = permute(pseudo_label_li, [2 1 3]); 
        % H_perfect and H_pseudo already consider pathGains and small-scale factors

        H_li = zeros(size(pseudo_label_li));
        H_li_clip = zeros(size(pseudo_label_li));

        nmse_pseudo_li = 0;
        for n = 1:size(pseudo_label_li, 3)
            H_pseudo_n = H_pseudo(:, :, n);
            Y_received_noNoise = txGrid .* H_pseudo(:, :, n);
            
            % Calculate Noise Variance (Linear Scale)
            Ps = mean(abs(Y_received_noNoise(refDMRSIndices)).^2);
            % SNR = Ps / sigma^2  =>  sigma^2 = Ps / 10^(SNR_dB/10)
            sigma2 = Ps / (10^(snr_db / 10));
        
            % Generate Complex AWGN
            % We use sqrt(sigma2/2) because randn generates variance 1.
            % We need total variance sigma2 split between Real and Imag.
            Noise = sqrt(sigma2 / 2) * (randn(size(txGrid)) + 1i * randn(size(txGrid)));
            Y_received = Y_received_noNoise + Noise;
        
            % Least Square + Linear Interpolation
            [H_equalized_n, H_linear_n] = Lin_Interpolate(Y_received, refDMRSIndices, refDMRSSymbols);
            H_linear_n_ = crop_(H_linear_n, [1, 128], [3, 12]); % crop the extrapolation values 132 x 14
            
            H_li(:, :, n) = permute(H_linear_n, [2 1 3]); % 14 x 132 x 2048
            H_li_clip(:, :, n) = permute(H_linear_n_, [2 1 3]); % 14 x 132 x 2048
            
            % Calculate NMSE
            error_matrix = H_pseudo_n - H_linear_n_;
            squared_error = sum(abs(error_matrix).^2, 'all');
            true_power = sum(abs(H_pseudo_n).^2, 'all');
            nmse_pseudo_li_n = squared_error / true_power;
            nmse_pseudo_li = nmse_pseudo_li + nmse_pseudo_li_n;
        end
        nmse_li = nmse_pseudo_li / size(pseudo_label_li, 3);
        
        save_folder = ['pseudoChannelD30_A300/SNR_', num2str(snr_db), 'dB_', '13x', num2str(w_w)];  
        if ~exist(save_folder, 'dir')
            mkdir(save_folder);
        end
        save([save_folder, '/', 'matlabNTN.mat'], ...
                                        'H_perfect', ...
                                        'H_li', ... 
                                        'H_li_clip', ... 
                                        'nmse_li', ...
                                        '-v7.3');
        fprintf('Saved: %s/matlabNTN.mat (NMSE: %.4f)\n', save_folder, nmse_li);
    end
end
fprintf('\nPseudo dataset generation finished successfully!\n');

%% ========================================================================
%% LOCAL FUNCTIONS (Placed at the end of the script per MATLAB convention)
%% ========================================================================

function [amplitude_spectrum, phase_spectrum] = F_extract_DD(channel_grid)
% F_EXTRACT Converts Time-Freq channel to Delay-Doppler domain
% Input: 
%   channel_grid: 132 (Subcarriers) x 14 (Symbols) Complex Matrix
% Output:
%   amplitude_spectrum: Magnitude in Delay-Doppler (The "Style")
%   phase_spectrum: Phase in Delay-Doppler (The "Content")

    % 1. Transform Subcarriers (Frequency) -> Delay
    % We use IFFT along dimension 1 (Rows).
    % No shift needed here because delay is naturally 0 to Max.
    grid_delay = ifft(channel_grid, [], 1);
    
    % 2. Transform Symbols (Time) -> Doppler
    % We use FFT along dimension 2 (Columns).
    grid_delay_doppler_raw = fft(grid_delay, [], 2);
    
    % 3. Shift BOTH dimensions to Center
    % This moves (Delay=0, Doppler=0) to the middle of the matrix.
    grid_dd_shifted = fftshift(grid_delay_doppler_raw);
    
    % 4. Extract Amplitude and Phase
    amplitude_spectrum = abs(grid_dd_shifted);
    phase_spectrum = angle(grid_dd_shifted);
end

function [mixed_img] = fda_mix_pixels(source_img, target_img, win_h_px, win_w_px)
    % center: target
    % outer: source
    % Calculate Center Coordinates
    [h, w] = size(source_img);
    cy = floor(h / 2) + 1;
    cx = floor(w / 2) + 1;   

    % Determine Window Radius (Half-size)
    r_h = floor(win_h_px / 2);
    r_w = floor(win_w_px / 2);

    % Create the Mask
    mask = zeros(h, w);
    
    % Set the center region to 1 (use Target amplitude here)
    y_range = (cy - r_h) : (cy + r_h);
    x_range = (cx - r_w) : (cx + r_w);
    
    mask(y_range, x_range) = 1;
    
    % Mix the Amplitudes
    % If mask is 1, take Target. If mask is 0, take Source.
    mixed_img = (mask .* target_img) + ((1 - mask) .* source_img); 
end

function [tf_grid] = F_inverse_DD(complex_dd)
% F_INVERSE_DD Converts Delay-Doppler grid back to Time-Freq grid
    % 1. Unshift Both Dimensions
    grid_unshifted = ifftshift(complex_dd);
    
    % 2. Inverse Doppler -> Time (Dimension 2: Columns)
    grid_delay_time = ifft(grid_unshifted, [], 2);
    
    % 3. Inverse Delay -> Frequency (Dimension 1: Rows)
    tf_grid = fft(grid_delay_time, [], 1);
end

function translate_img = FTranslate_single(source_img, target_img, win_h_px, win_w_px)
    % center: target_img
    % outer: source_img
    % phase: source_img
    [source_amplitude_spectrum, source_phase_spectrum] = F_extract_DD(source_img);
    [target_amplitude_spectrum, ~] = F_extract_DD(target_img);

    translate_img_amp = fda_mix_pixels(source_amplitude_spectrum, target_amplitude_spectrum, win_h_px, win_w_px);

    translate_img_DD = translate_img_amp .* exp(1i * source_phase_spectrum);
    translate_img = F_inverse_DD(translate_img_DD);
end

function img_slice = crop_(img_slice, row_range, col_range)
    % img_slice is a 2D matrix (e.g., target_img(:,:,n))
    % row_range = [1, 128]
    % col_range = [3, 12]

    % 1. Extract the reference "inner" region
    ref_region = img_slice(row_range(1):row_range(2), col_range(1):col_range(2));
    
    % 2. Get global limits for Real and Imaginary parts from the reference
    r_min = min(real(ref_region), [], 'all');
    r_max = max(real(ref_region), [], 'all');
    
    i_min = min(imag(ref_region), [], 'all');
    i_max = max(imag(ref_region), [], 'all');
    
    % 3. Create a logical mask for the "outside" area
    [rows, cols] = size(img_slice);
    is_outside = true(rows, cols);
    
    % Set the "inside" area to 'false' so we don't modify it
    is_outside(row_range(1):row_range(2), col_range(1):col_range(2)) = false;
    
    % 4. Apply the limits only to the outside elements
    R = real(img_slice(is_outside));
    I = imag(img_slice(is_outside));
    
    % Constrain (Crop) the values
    R = max(min(R, r_max), r_min);
    I = max(min(I, i_max), i_min);
    
    % 5. Put the modified values back into the slice
    img_slice(is_outside) = complex(R, I);
end

function translate_img = FTranslate_bulk(source_img, target_img, win_h_px, win_w_px)
    % center: target_img
    % outer: source_img
    % phase: source_img

    source_img = permute(source_img, [2, 1, 3]);
    target_img = permute(target_img, [2, 1, 3]);    
    translate_img = zeros(size(target_img));
    target_img_ = zeros(size(target_img));

    for n = 1:size(target_img, 3)
        target_img_(:, :, n) = crop_(target_img(:, :, n), [1, 128], [3, 12]); % crop the extrapolated values (of the pilots)
        translate_img(:, :, n) = FTranslate_single(source_img(:, :, n), target_img_(:, :, n), win_h_px, win_w_px);
    end

    translate_img = permute(translate_img, [2, 1, 3]);
end
