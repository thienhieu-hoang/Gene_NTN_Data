function saveNPY_channel(filename, H, normalize_rms)
% SAVENPY_CHANNEL Save 3D complex channel matrix directly to NumPy .npy format.
%
% Inputs:
%   filename      - Full path to output .npy file (e.g. 'H_perfect.npy' or 'dataset_train.npy')
%   H             - Complex channel matrix of size [14, 132, numUE] (symbols x subcarriers x UEs)
%   normalize_rms - (Optional) Boolean. Defaults to false to preserve raw channel power.
%                   If set to true, normalizes each UE slot to unit power (E[|H|^2] = 1.0).
%
% Output in Python:
%   Array with shape (numUE, 132, 14), dtype=complex64, C-contiguous order.

if nargin < 3
    normalize_rms = false;
end

% Check dimensions
sz = size(H);
if numel(sz) == 2
    n_sym = sz(1);
    n_sc = sz(2);
    n_samples = 1;
else
    n_sym = sz(1);
    n_sc = sz(2);
    n_samples = sz(3);
end

% Optional sample-wise RMS normalization for pyntn compatibility
if normalize_rms
    for i = 1:n_samples
        slot_pwr = mean(abs(H(:,:,i)).^2, 'all');
        if slot_pwr > 0
            H(:,:,i) = H(:,:,i) / sqrt(slot_pwr);
        end
    end
end

% Build NPY v1.0 Header
% Python shape: (n_samples, n_sc, n_sym) -> e.g. (numUE, 132, 14)
header_dict = sprintf('{''descr'': ''<c8'', ''fortran_order'': False, ''shape'': (%d, %d, %d), }', ...
    n_samples, n_sc, n_sym);

% NPY format requires total prefix length (10 bytes prefix + header_len) to be multiple of 64 or 16
prefix_len = 10; % 6 bytes magic + 2 bytes version + 2 bytes header len
target_total_header = 128;
while target_total_header < prefix_len + length(header_dict) + 1
    target_total_header = target_total_header + 64;
end

pad_len = target_total_header - prefix_len - length(header_dict) - 1;
header_str = [header_dict, repmat(' ', 1, pad_len), newline];
header_len_uint16 = uint16(length(header_str));

% Interleave real and imaginary components as single-precision float32
num_elem = numel(H);
interleaved = zeros(2, num_elem, 'single');
interleaved(1, :) = single(real(H(:)));
interleaved(2, :) = single(imag(H(:)));

% Ensure parent folder exists
folder_out = fileparts(filename);
if ~isempty(folder_out) && ~exist(folder_out, 'dir')
    mkdir(folder_out);
end

% Write binary file
fid = fopen(filename, 'wb');
if fid == -1
    error('Could not open file for writing: %s', filename);
end

% 1. Magic string: \x93NUMPY
fwrite(fid, uint8([147, 'NUMPY']), 'uint8');
% 2. Version: 1.0
fwrite(fid, uint8([1, 0]), 'uint8');
% 3. Header length (uint16 little-endian)
fwrite(fid, header_len_uint16, 'uint16');
% 4. Header string (ASCII)
fwrite(fid, uint8(header_str), 'uint8');
% 5. Data payload (interleaved float32 complex64)
fwrite(fid, interleaved, 'single');

fclose(fid);
fprintf('Saved NPY channel: %s [shape=(%d, %d, %d), complex64]\n', ...
    filename, n_samples, n_sc, n_sym);
end
