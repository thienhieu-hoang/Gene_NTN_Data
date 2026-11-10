%% Convert all existing .mat datasets in generatedChannel_Results to .npy
% This script scans subfolders in generatedChannel_Results, loads H_perfect
% from matlabNTN.mat, applies sample-wise RMS normalization, and saves:
%   - H_train.npy (80%)
%   - H_val.npy   (10%)
%   - H_test.npy  (10%)

clc; clear;
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
cd(script_dir);

results_dir = fullfile(script_dir, 'generatedChannel_Results');
if ~exist(results_dir, 'dir')
    error('Directory not found: %s', results_dir);
end

scenarios = dir(results_dir);
scenarios = scenarios([scenarios.isdir] & ~ismember({scenarios.name}, {'.', '..'}));

fprintf('Found %d scenario folder(s) in %s\n\n', length(scenarios), results_dir);

for k = 1:length(scenarios)
    scenario_name = scenarios(k).name;
    scenario_path = fullfile(results_dir, scenario_name);
    fprintf('Processing: %s ...\n', scenario_name);
    
    % Find all SNR subfolders
    snr_dirs = dir(scenario_path);
    snr_dirs = snr_dirs([snr_dirs.isdir] & ~ismember({snr_dirs.name}, {'.', '..'}));
    
    H_loaded = [];
    
    % Loop over SNR subfolders to find matlabNTN.mat
    for s = 1:length(snr_dirs)
        snr_folder = fullfile(scenario_path, snr_dirs(s).name);
        mat_file = fullfile(snr_folder, 'matlabNTN.mat');
        
        if exist(mat_file, 'file')
            mat_data = load(mat_file, 'H_perfect');
            if isfield(mat_data, 'H_perfect')
                H_loaded = mat_data.H_perfect; % size: [14, 132, numUE]
                break;
            end
        end
    end
    
    if isempty(H_loaded)
        warning('No valid matlabNTN.mat found in %s', scenario_name);
        continue;
    end
    
    % Clean up any old H_perfect.npy or long-name files in scenario_path
    old_files = dir(fullfile(scenario_path, '*.npy'));
    for f = 1:length(old_files)
        delete(fullfile(scenario_path, old_files(f).name));
    end
    
    % Also clean up any per-SNR H_perfect.npy
    for s = 1:length(snr_dirs)
        snr_npy = fullfile(scenario_path, snr_dirs(s).name, 'H_perfect.npy');
        if exist(snr_npy, 'file')
            delete(snr_npy);
        end
    end
    
    % Generate 80% / 10% / 10% splits
    numUE = size(H_loaded, 3);
    n_train = floor(0.80 * numUE);
    n_val   = floor(0.10 * numUE);
    
    H_train = H_loaded(:, :, 1:n_train);
    H_val   = H_loaded(:, :, n_train+1:n_train+n_val);
    H_test  = H_loaded(:, :, n_train+n_val+1:end);
    
    saveNPY_channel(fullfile(scenario_path, 'H_train.npy'), H_train, true);
    saveNPY_channel(fullfile(scenario_path, 'H_val.npy'), H_val, true);
    saveNPY_channel(fullfile(scenario_path, 'H_test.npy'), H_test, true);
    
    fprintf('  Done: H_train=%d, H_val=%d, H_test=%d\n\n', ...
        size(H_train, 3), size(H_val, 3), size(H_test, 3));
end

fprintf('All existing datasets updated to H_train.npy, H_val.npy, H_test.npy!\n');
