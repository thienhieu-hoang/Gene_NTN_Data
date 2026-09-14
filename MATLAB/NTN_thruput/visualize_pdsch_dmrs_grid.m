%% =========================================================================
%  5G NR PDSCH & DM-RS Resource Grid Visualizer
%  Supports: Mapping Type A / B, DM-RS Config Type 1 / 2, Additional Positions
% =========================================================================
clear; clc; close all;

%% 1. Carrier Configuration
carrier = nrCarrierConfig;
carrier.NSizeGrid          = 11;        % Number of Resource Blocks (PRBs)
carrier.SubcarrierSpacing  = 30;        % Subcarrier spacing in kHz (15, 30, 60, 120)
carrier.CyclicPrefix       = 'Normal';  % 'Normal' (14 symbols) or 'Extended' (12 symbols)
carrier.NCellID            = 1;         % Physical layer Cell ID
carrier.NSlot              = 0;         % Slot number

%% 2. PDSCH and DM-RS Configuration
pdsch = nrPDSCHConfig;
pdsch.PRBSet              = 0:carrier.NSizeGrid-1; % Full carrier PRB allocation
pdsch.NumLayers           = 1;                     % Number of transmission layers
pdsch.Modulation          = '16QAM';

% --- [SETTING 1] Mapping Type: 'A' (Slot-based) or 'B' (Non-slot / Mini-slot) ---
pdsch.MappingType         = 'A';  % Choose 'A' or 'B'

% Symbol Allocation: [startSymbolIndex, numSymbols] (0-based indexing)
if strcmp(pdsch.MappingType, 'A')
    pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot]; % Type A typically spans whole slot [0, 14]
else
    pdsch.SymbolAllocation = [2, 10]; % Type B can start at any symbol, e.g., symbol 2 for 10 symbols
end

% --- [SETTING 2] DM-RS Configuration Type: 1 or 2 ---
pdsch.DMRS.DMRSConfigurationType = 1;  % 1: 6 subcarriers/PRB, 2: 4 subcarriers/PRB (paired)

% --- [SETTING 3] DM-RS Type A Position (Only used when MappingType = 'A') ---
pdsch.DMRS.DMRSTypeAPosition     = 2;  % 2 or 3 (OFDM symbol index within slot for 1st DM-RS)

% --- [SETTING 4] Additional DM-RS Settings ---
pdsch.DMRS.DMRSLength             = 1;  % 1 (Single symbol) or 2 (Double symbol DM-RS)
pdsch.DMRS.DMRSAdditionalPosition = 1;  % Additional positions: 0, 1, 2, or 3
pdsch.DMRS.NumCDMGroupsWithoutData = 3; % 1, 2, or 3 CDM groups reserved
pdsch.DMRS.NIDNSCID               = carrier.NCellID;
pdsch.DMRS.NSCID                  = 0;

%% 3. Generate Indices & Populate Resource Grid
% Create empty carrier resource grid: [Subcarriers x Symbols x Ports]
grid = nrResourceGrid(carrier);

% Generate 1-based linear indices for PDSCH data and DM-RS pilots
[pdschIndices, pdschInfo] = nrPDSCHIndices(carrier, pdsch);
dmrsIndices               = nrPDSCHDMRSIndices(carrier, pdsch);
dmrsSymbols               = nrPDSCHDMRS(carrier, pdsch);

% Numerical labels for visualization:
% 0 = Unused / Empty RE
% 1 = PDSCH Data RE
% 2 = DM-RS Pilot RE
visualGrid = zeros(size(grid, 1), size(grid, 2));
visualGrid(pdschIndices) = 1;
visualGrid(dmrsIndices)  = 2;

%% 4. Single Configuration Visualization
figure('Name', 'PDSCH & DM-RS Resource Grid', 'Color', 'w', 'Position', [150, 150, 950, 650]);
customColormap = [
    0.95 0.95 0.95;  % 0: Empty / Unallocated (Light Gray)
    0.20 0.50 0.85;  % 1: PDSCH Data REs (Blue)
    0.85 0.20 0.20   % 2: DM-RS Pilot REs (Red)
];

imagesc(0:carrier.SymbolsPerSlot-1, 0:(carrier.NSizeGrid*12 - 1), visualGrid);
colormap(customColormap);
caxis([0 2]);
axis xy;

% Grid formatting & labels
xlabel('OFDM Symbol Index (Time \rightarrow)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Subcarrier Index (Frequency \uparrow)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('PDSCH Grid: Mapping Type %s | DM-RS Config Type %d | AdditionalPos = %d | TypeAPos = %d', ...
    pdsch.MappingType, pdsch.DMRS.DMRSConfigurationType, pdsch.DMRS.DMRSAdditionalPosition, pdsch.DMRS.DMRSTypeAPosition), ...
    'FontSize', 13, 'FontWeight', 'bold');

% Draw PRB boundary lines (every 12 subcarriers)
hold on;
for prb = 1:carrier.NSizeGrid-1
    yline(prb*12 - 0.5, 'k--', 'LineWidth', 0.75, 'Alpha', 0.4);
end
hold off;

% Setup custom legend
hold on;
hEmpty = plot(nan, nan, 's', 'MarkerFaceColor', customColormap(1,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 12);
hData  = plot(nan, nan, 's', 'MarkerFaceColor', customColormap(2,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 12);
hDMRS  = plot(nan, nan, 's', 'MarkerFaceColor', customColormap(3,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 12);
legend([hEmpty, hData, hDMRS], {'Unallocated RE', 'PDSCH Data RE', 'DM-RS Pilot RE'}, ...
    'Location', 'northeastoutside', 'FontSize', 11);
hold off;
% 
% grid on;
set(gca, 'XTick', 0:carrier.SymbolsPerSlot-1, 'YTick', 0:12:carrier.NSizeGrid*12, 'FontSize', 10);

%% =========================================================================
%% 5. Side-by-Side 4-Way Comparison Plot (Type A vs B, DM-RS Type 1 vs 2)
%% =========================================================================
figure('Name', 'PDSCH Mapping & DM-RS Types Comparison', 'Color', 'w', 'Position', [100, 100, 1200, 750]);

configs = {
    'A', 1, [0, carrier.SymbolsPerSlot], 'Type A + DM-RS Config 1';
    'A', 2, [0, carrier.SymbolsPerSlot], 'Type A + DM-RS Config 2';
    'B', 1, [0, 13],                     'Type B + DM-RS Config 1';
    'B', 2, [0, 13],                     'Type B + DM-RS Config 2'
};

for k = 1:4
    subplot(2, 2, k);
    
    p = nrPDSCHConfig;
    p.PRBSet                  = 0:carrier.NSizeGrid-1;
    p.MappingType             = configs{k, 1};
    p.DMRS.DMRSConfigurationType = configs{k, 2};
    p.SymbolAllocation        = configs{k, 3};
    p.DMRS.DMRSTypeAPosition  = 2;
    p.DMRS.DMRSAdditionalPosition = 1;
    p.DMRS.DMRSLength         = 1;
    
    pIndices = nrPDSCHIndices(carrier, p);
    dIndices = nrPDSCHDMRSIndices(carrier, p);
    
    vGrid = zeros(carrier.NSizeGrid * 12, carrier.SymbolsPerSlot);
    vGrid(pIndices) = 1;
    vGrid(dIndices) = 2;
    
    imagesc(0:carrier.SymbolsPerSlot-1, 0:(carrier.NSizeGrid*12 - 1), vGrid);
    colormap(customColormap);
    caxis([0 2]);
    axis xy;
    
    title(configs{k, 4}, 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    set(gca, 'XTick', 0:2:carrier.SymbolsPerSlot-1, 'YTick', 0:24:carrier.NSizeGrid*12);
    
    hold on;
    for prb = 1:carrier.NSizeGrid-1
        yline(prb*12 - 0.5, 'k:', 'Alpha', 0.3);
    end
    hold off;
end