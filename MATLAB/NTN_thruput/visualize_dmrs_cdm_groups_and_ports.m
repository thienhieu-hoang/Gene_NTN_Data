%% =========================================================================
%  5G NR PDSCH DM-RS CDM Groups & Antenna Ports Visualizer
%  Demonstrates:
%    1. DM-RS Configuration Type 2 (3 CDM groups: mod6={0,1}, {2,3}, {4,5})
%    2. DM-RS Configuration Type 1 (2 CDM groups: even/odd subcarriers)
%    3. Antenna Port to CDM Group mapping (Ports 0-5 / 1000-1005)
%    4. Effect of NumCDMGroupsWithoutData (1, 2, 3) on Data RE puncture/reservation
%  * No UE channel simulation needed — uses MATLAB 5G Toolbox indexing APIs directly.
% =========================================================================
clear; clc; close all;

%% 1. Basic Carrier Setup
carrier = nrCarrierConfig;
carrier.NSizeGrid         = 4;         % 4 PRBs (48 subcarriers) for clear visualization
carrier.SubcarrierSpacing = 30;        % 30 kHz
carrier.CyclicPrefix      = 'Normal';  % 14 OFDM symbols per slot
carrier.NCellID           = 1;
carrier.NSlot             = 0;

%% 2. Detailed Single PRB Breakdown (Config Type 2: 3 CDM Groups)
% In Config Type 2, 1 PRB (12 subcarriers) has 3 CDM groups:
%   - CDM Group 0 (Ports 0, 1): subcarriers 0, 1, 6, 7  (k mod 6 in {0,1})
%   - CDM Group 1 (Ports 2, 3): subcarriers 2, 3, 8, 9  (k mod 6 in {2,3})
%   - CDM Group 2 (Ports 4, 5): subcarriers 4, 5, 10, 11 (k mod 6 in {4,5})

figure('Name', 'DM-RS Config Type 2 - 3 CDM Groups in 1 PRB', 'Color', 'w', 'Position', [100, 100, 1050, 600]);

% Create a single PRB grid for 1 slot (12 subcarriers x 14 symbols)
prbGrid = zeros(12, carrier.SymbolsPerSlot);

% Assume Mapping Type A, 1st DM-RS at symbol 2, additional DM-RS at symbol 11
dmrsSymbols_time = [2, 11]; 

% Fill PDSCH data (value = 1)
prbGrid(:, :) = 1; 

% Assign distinct values for each CDM Group on DM-RS symbols:
% CDM Group 0 -> Value 2 (subcarriers 0, 1, 6, 7)
prbGrid([0, 1, 6, 7] + 1, dmrsSymbols_time + 1) = 2;

% CDM Group 1 -> Value 3 (subcarriers 2, 3, 8, 9)
prbGrid([2, 3, 8, 9] + 1, dmrsSymbols_time + 1) = 3;

% CDM Group 2 -> Value 4 (subcarriers 4, 5, 10, 11)
prbGrid([4, 5, 10, 11] + 1, dmrsSymbols_time + 1) = 4;

% Custom colormap for visualization
cdmColormap = [
    0.95 0.95 0.95;  % 0: Empty / Unused
    0.80 0.88 0.97;  % 1: PDSCH Data REs (Soft Blue)
    0.85 0.20 0.20;  % 2: CDM Group 0 (Red)    - Ports 1000, 1001 (mod6 = 0,1)
    0.20 0.70 0.30;  % 3: CDM Group 1 (Green)  - Ports 1002, 1003 (mod6 = 2,3)
    0.95 0.55 0.10   % 4: CDM Group 2 (Orange) - Ports 1004, 1005 (mod6 = 4,5)
];

subplot(1, 2, 1);
imagesc(0:carrier.SymbolsPerSlot-1, 0:11, prbGrid);
colormap(gca, cdmColormap);
caxis([0 4]);
axis xy;
xlabel('OFDM Symbol Index (Time \rightarrow)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Subcarrier Index in 1 PRB (0 to 11)', 'FontSize', 11, 'FontWeight', 'bold');
title({'1 PRB Zoom: DM-RS Config Type 2', '(3 CDM Groups across Subcarriers)'}, 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', 0:carrier.SymbolsPerSlot-1, 'YTick', 0:11);
grid on;

% Add subcarrier labels on the right
hold on;
for sc = 0:11
    mod6 = mod(sc, 6);
    if ismember(mod6, [0, 1])
        grpStr = 'CDM Grp 0';
    elseif ismember(mod6, [2, 3])
        grpStr = 'CDM Grp 1';
    else
        grpStr = 'CDM Grp 2';
    end
    text(carrier.SymbolsPerSlot - 0.4, sc, sprintf('k=%d (%s)', sc, grpStr), ...
        'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'right', 'Color', [0.2 0.2 0.2]);
end
hold off;

%% 3. Effect of NumCDMGroupsWithoutData (1, 2, vs 3)
% When NumCDMGroupsWithoutData = 3, data is NOT transmitted on any of the 3 CDM groups
% during DM-RS symbols (all 12 subcarriers in the DM-RS symbol are reserved for pilots).
% When NumCDMGroupsWithoutData = 1, data CAN be transmitted on CDM Groups 1 and 2.

subplot(1, 2, 2);
pdsch = nrPDSCHConfig;
pdsch.PRBSet = 0; % 1 PRB
pdsch.MappingType = 'A';
pdsch.SymbolAllocation = [0, 14];
pdsch.DMRS.DMRSConfigurationType = 2;
pdsch.DMRS.DMRSTypeAPosition = 2;
pdsch.DMRS.DMRSAdditionalPosition = 1;
pdsch.DMRS.DMRSPortSet = 0;             % Transmitting only Port 0 (CDM Group 0)
pdsch.DMRS.NumCDMGroupsWithoutData = 3;  % Reserves CDM Group 0, 1, and 2 from PDSCH data

[pdschInd, ~] = nrPDSCHIndices(carrier, pdsch);
dmrsInd       = nrPDSCHDMRSIndices(carrier, pdsch);

gridRes = zeros(12, 14);
gridRes(pdschInd) = 1; % Data
gridRes(dmrsInd)  = 2; % Port 0 Pilot

% Subcarriers in DM-RS symbols that have neither data nor Port 0 DM-RS are reserved
imagesc(0:13, 0:11, gridRes);
colormap(gca, [
    0.90 0.90 0.90;  % 0: Reserved REs (Gray) for other CDM groups (MU-MIMO)
    0.80 0.88 0.97;  % 1: PDSCH Data REs (Soft Blue)
    0.85 0.20 0.20   % 2: Active UE Port 0 DM-RS (Red)
]);
caxis([0 2]);
axis xy;
xlabel('OFDM Symbol Index (Time \rightarrow)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Subcarrier Index in 1 PRB (0 to 11)', 'FontSize', 11, 'FontWeight', 'bold');
title({'Port 0 Grid with NumCDMGroupsWithoutData = 3', '(REs reserved for CDM Grp 1 & 2 shown in Gray)'}, ...
    'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', 0:13, 'YTick', 0:11);
grid on;

%% =========================================================================
%% 4. Full Multi-Port Visualization (Ports 0 to 5 on Carrier Grid)
%% =========================================================================
figure('Name', 'DM-RS Config Type 2: Antenna Ports 0 to 5 Allocation', 'Color', 'w', 'Position', [150, 150, 1200, 700]);

portList = [0, 1, 2, 3, 4, 5];
portCDMGroup = [0, 0, 1, 1, 2, 2]; % Mapping of ports to CDM groups

for idx = 1:length(portList)
    port = portList(idx);
    grp  = portCDMGroup(idx);
    
    p = nrPDSCHConfig;
    p.PRBSet = 0:carrier.NSizeGrid-1;
    p.MappingType = 'A';
    p.SymbolAllocation = [0, 14];
    p.DMRS.DMRSConfigurationType = 2;
    p.DMRS.DMRSTypeAPosition = 2;
    p.DMRS.DMRSAdditionalPosition = 1;
    p.DMRS.DMRSPortSet = port;           % Specific antenna port
    p.DMRS.NumCDMGroupsWithoutData = 3;  % Reserve all 3 groups
    
    pInd = nrPDSCHIndices(carrier, p);
    dInd = nrPDSCHDMRSIndices(carrier, p);
    
    portGrid = zeros(carrier.NSizeGrid * 12, carrier.SymbolsPerSlot);
    portGrid(pInd) = 1; % Data
    portGrid(dInd) = 2; % DM-RS for this port
    
    subplot(2, 3, idx);
    imagesc(0:carrier.SymbolsPerSlot-1, 0:(carrier.NSizeGrid*12 - 1), portGrid);
    colormap(gca, [
        0.88 0.88 0.88; % 0: Reserved / Muted for other ports
        0.75 0.85 0.95; % 1: Data
        0.85 0.15 0.15  % 2: DM-RS Pilot on this port
    ]);
    caxis([0 2]);
    axis xy;
    
    title(sprintf('Port %d (Antenna 100%d) \\rightarrow CDM Group %d', port, port, grp), ...
        'FontSize', 11, 'FontWeight', 'bold');
    xlabel('OFDM Symbol');
    ylabel('Subcarrier Index');
    set(gca, 'XTick', 0:2:13, 'YTick', 0:12:(carrier.NSizeGrid*12));
    grid on;
    
    % Draw PRB boundaries
    hold on;
    for prb = 1:carrier.NSizeGrid-1
        yline(prb*12 - 0.5, 'k--', 'Alpha', 0.3);
    end
    hold off;
end

%% =========================================================================
%% 5. Direct Comparison: Config Type 1 (2 CDM Grps) vs Config Type 2 (3 CDM Grps)
%% =========================================================================
figure('Name', 'Config Type 1 vs Config Type 2 Comparison', 'Color', 'w', 'Position', [200, 200, 1100, 550]);

% --- Subplot 1: Config Type 1 ---
subplot(1, 2, 1);
p1 = nrPDSCHConfig;
p1.PRBSet = 0:1; % 2 PRBs
p1.MappingType = 'A';
p1.DMRS.DMRSConfigurationType = 1; % Type 1: 6 subcarriers/PRB
p1.DMRS.DMRSTypeAPosition = 2;
p1.DMRS.DMRSAdditionalPosition = 1;
p1.DMRS.NumCDMGroupsWithoutData = 2;

p1_grp0_pdsch = p1; p1_grp0_pdsch.DMRS.DMRSPortSet = 0;
p1_grp1_pdsch = p1; p1_grp1_pdsch.DMRS.DMRSPortSet = 2;

p1_grp0 = nrPDSCHDMRSIndices(carrier, p1_grp0_pdsch);
p1_grp1 = nrPDSCHDMRSIndices(carrier, p1_grp1_pdsch);
p1_data = nrPDSCHIndices(carrier, p1);

gridType1 = zeros(24, 14);
gridType1(p1_data) = 1;
gridType1(p1_grp0) = 2; % Group 0 (Even subcarriers: 0, 2, 4, 6, 8, 10)
gridType1(p1_grp1) = 3; % Group 1 (Odd subcarriers: 1, 3, 5, 7, 9, 11)

imagesc(0:13, 0:23, gridType1);
colormap(gca, [0.95 0.95 0.95; 0.8 0.88 0.97; 0.85 0.2 0.2; 0.2 0.7 0.3]);
caxis([0 3]);
axis xy;
title({'DM-RS Config Type 1 (Max 2 CDM Groups)', 'Grp 0: Even Subcarriers | Grp 1: Odd Subcarriers'}, 'FontSize', 11, 'FontWeight', 'bold');
xlabel('OFDM Symbol'); ylabel('Subcarrier Index');
set(gca, 'XTick', 0:13, 'YTick', 0:12:24);
grid on;
yline(11.5, 'k--', 'PRB Boundary', 'LineWidth', 1.2);

% --- Subplot 2: Config Type 2 ---
subplot(1, 2, 2);
p2 = nrPDSCHConfig;
p2.PRBSet = 0:1; % 2 PRBs
p2.MappingType = 'A';
p2.DMRS.DMRSConfigurationType = 2; % Type 2: 4 subcarriers/PRB
p2.DMRS.DMRSTypeAPosition = 2;
p2.DMRS.DMRSAdditionalPosition = 1;
p2.DMRS.NumCDMGroupsWithoutData = 3;

p2_grp0_pdsch = p2; p2_grp0_pdsch.DMRS.DMRSPortSet = 0; % Port 0
p2_grp1_pdsch = p2; p2_grp1_pdsch.DMRS.DMRSPortSet = 2; % Port 2
p2_grp2_pdsch = p2; p2_grp2_pdsch.DMRS.DMRSPortSet = 4; % Port 4

p2_grp0 = nrPDSCHDMRSIndices(carrier, p2_grp0_pdsch);
p2_grp1 = nrPDSCHDMRSIndices(carrier, p2_grp1_pdsch);
p2_grp2 = nrPDSCHDMRSIndices(carrier, p2_grp2_pdsch);
p2_data = nrPDSCHIndices(carrier, p2);

gridType2 = zeros(24, 14);
gridType2(p2_data) = 1;
gridType2(p2_grp0) = 2; % Group 0: {0,1}, {6,7}
gridType2(p2_grp1) = 3; % Group 1: {2,3}, {8,9}
gridType2(p2_grp2) = 4; % Group 2: {4,5}, {10,11}

imagesc(0:13, 0:23, gridType2);
colormap(gca, cdmColormap);
caxis([0 4]);
axis xy;
title({'DM-RS Config Type 2 (Max 3 CDM Groups)', 'Grp 0: mod6={0,1} | Grp 1: mod6={2,3} | Grp 2: mod6={4,5}'}, 'FontSize', 11, 'FontWeight', 'bold');
xlabel('OFDM Symbol'); ylabel('Subcarrier Index');
set(gca, 'XTick', 0:13, 'YTick', 0:12:24);
grid on;
yline(11.5, 'k--', 'PRB Boundary', 'LineWidth', 1.2);

fprintf('=== Visualization script completed successfully! ===\n');
