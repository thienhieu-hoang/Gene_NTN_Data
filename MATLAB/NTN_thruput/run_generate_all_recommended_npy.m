%% Batch Generator for Recommended NTN Datasets (.npy only) - 20,000 Samples Each
% Generates all recommended benchmark datasets with NumUE = 20000 and saves only
% H_train.npy (16,000), H_val.npy (2,000), H_test.npy (2,000) in generatedChannel_Results/

clc; clear;
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
cd(script_dir);

num_samples_per_scenario = 20000;

scenarios = {
    % Profile,       DelaySpread, ElevationAngle, Description
    'NTN-TDL-C',     300e-9,      30,             'LoS Target Domain (pyntn target benchmark)';
    'NTN-TDL-A',     300e-9,      30,             'NLoS Target Domain (matches Sionna DUR300)';
    'NTN-TDL-A',     100e-9,      30,             'Elevation-only domain shift (70 deg -> 30 deg)';
    'NTN-TDL-A',     500e-9,      30,             'Dense urban stress test (high delay spread)';
    'NTN-TDL-C',      30e-9,      70,             'Rural / Open-area clear-sky LoS';
    'NTN-TDL-A',     100e-9,      70,             'Source Baseline Domain (A100 70deg)'
};

num_scenarios = size(scenarios, 1);
fprintf('====================================================================\n');
fprintf('Starting Batch Generation of %d NTN Datasets (20,000 samples each)\n', num_scenarios);
fprintf('====================================================================\n\n');

t_total_start = tic;

for i = 1:num_scenarios
    prof = scenarios{i, 1};
    ds   = scenarios{i, 2};
    elev = scenarios{i, 3};
    desc = scenarios{i, 4};
    
    fprintf('>>> [%d/%d] Generating: %s | DelaySpread: %.0f ns | Elev: %.0f deg\n', ...
        i, num_scenarios, prof, ds*1e9, elev);
    fprintf('    Role: %s\n', desc);
    
    t_start = tic;
    try
        Gen_Chan_sampleWise_wGeometry( ...
            'Profile', prof, ...
            'DelaySpread', ds, ...
            'ElevationAngle', elev, ...
            'NumUE', num_samples_per_scenario, ...
            'SaveOnlyNPY', true);
        fprintf('    [Done in %.2f s (%.2f min)]\n\n', toc(t_start), toc(t_start)/60);
    catch err
        fprintf('    [ERROR]: %s\n\n', err.message);
    end
end

fprintf('====================================================================\n');
fprintf('All %d Datasets Generated Successfully in %.2f min!\n', ...
    num_scenarios, toc(t_total_start)/60);
fprintf('====================================================================\n');
