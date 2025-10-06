% function to generate txWaveform
function varargout = wave_gen(NSizeGrid, SubcarrierSpacing, refSig, flag_plot, N_t)
    carrier = nrCarrierConfig;
    if nargin<2         % just call wavegen()
        carrier.NSizeGrid = 51;            % Bandwidth in number of resource blocks (51 RBs at 30 kHz SCS for 20 MHz BW)
        carrier.SubcarrierSpacing = 30;    % 15, 30, 60, 120, 240 (kHz)
        refSig = "DM-RS"; % Default value
    else
        carrier.NSizeGrid = NSizeGrid;            % Bandwidth in number of resource blocks (51 RBs at 30 kHz SCS for 20 MHz BW)
        carrier.SubcarrierSpacing = SubcarrierSpacing;    % 15, 30, 60, 120, 240 (kHz)
        if nargin==2    % call wave_gen(NSizeGrid, SubcarrierSpacing,
            refSig = "DM-RS"; % Default value
        elseif nargin==3
            % refSig = refSig;
            flag_plot = 0;
        % elseif nargin==4
        %     % refSig = refSig;
        %     flag_plot = flag_plot;
        end
    end

    if refSig == "DM-RS"
        % N_t = 1; % SISO code
        carrier.CyclicPrefix = "Normal";   % "Normal" or "Extended" (Extended CP is relevant for 60 kHz SCS only)
        carrier.NCellID = 2;

        % PDSCH and DM-RS configuration
        pdsch = nrPDSCHConfig;
        pdsch.PRBSet = 0:carrier.NSizeGrid-1; % PDSCH PRB allocation
        pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot];           % PDSCH symbol allocation in each slot
        pdsch.MappingType = "A";     % PDSCH mapping type ("A"(slot-wise),"B"(non slot-wise))
        pdsch.NID = carrier.NCellID;
        pdsch.RNTI = 1;
        pdsch.VRBToPRBInterleaving = 0; % Disable interleaved resource mapping
        pdsch.NumLayers = 1;            % Number of PDSCH transmission layers
        pdsch.Modulation = "16QAM";                       % "QPSK", "16QAM", "64QAM", "256QAM"
        
        % DM-RS configuration
        pdsch.DMRS.DMRSPortSet = 0:pdsch.NumLayers-1; % DM-RS ports to use for the layers
        pdsch.DMRS.DMRSTypeAPosition = 2;      % Mapping type A only. First DM-RS symbol position (2,3)
        pdsch.DMRS.DMRSLength = 1;             % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
        pdsch.DMRS.DMRSAdditionalPosition = 1; % Additional DM-RS symbol positions (max range 0...3)
        pdsch.DMRS.DMRSConfigurationType = 2;  % DM-RS configuration type (1,2)
        pdsch.DMRS.NumCDMGroupsWithoutData = 1;% Number of CDM groups without data
        pdsch.DMRS.NIDNSCID = 1;               % Scrambling identity (0...65535)
        pdsch.DMRS.NSCID = 0;                  % Scrambling initialization (0,1)
    
        % Generate DM-RS indices and symbols
        dmrsSymbols = nrPDSCHDMRS(carrier,pdsch);
        dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);
        
        % Create resource grid
        pdschGrid = nrResourceGrid(carrier);
        
        % Map PDSCH DM-RS symbols to the grid
        pdschGrid(dmrsIndices) = dmrsSymbols;
        if flag_plot %plot Carrier Grid
            % disp(['DM-RS power scaling: ' num2str(powerDMRS) ' dB']);
            plotGrid(size(pdschGrid),dmrsIndices,dmrsSymbols, refSig);
        end
        
        % OFDM-modulate associated resource elements
        txWaveform = nrOFDMModulate(carrier,pdschGrid);

        varargout{1} = txWaveform;
        varargout{2} = carrier;
        varargout{3} = dmrsSymbols;
        varargout{4} = dmrsIndices;
        varargout{5} = pdsch;

    elseif refSig == "CSI-RS"
        if nargin <5
            N_t = 8; % deafault
        end
        carrier.NSlot = 1;
        carrier.NFrame = 0;
        %
        carrier.CyclicPrefix = "Normal";   % "Normal" or "Extended" (Extended CP is relevant for 60 kHz SCS only)
        carrier.NCellID = 2;
        %
        
        if N_t == 1
            csirs = nrCSIRSConfig;
            csirs.CSIRSType = {'nzp','zp'};
            csirs.CSIRSPeriod = {[5 1],[5 1]};
            csirs.Density = {'three','three'};
            csirs.RowNumber = [1 1];
            csirs.SymbolLocations = {1,6};
            csirs.SubcarrierLocations = {1,2};
            csirs.NumRB = NSizeGrid;
        elseif N_t == 4
            csirs = nrCSIRSConfig;
            csirs.CSIRSType = {'nzp','zp'};
            csirs.CSIRSPeriod = {[5 1],[5 1]};
            csirs.Density = {'one','one'};
            csirs.RowNumber = [3 5];
            csirs.SymbolLocations = {1,6};
            csirs.SubcarrierLocations = {6,4};
            csirs.NumRB = NSizeGrid;
        elseif N_t == 8
            csirs = nrCSIRSConfig;
            csirs.CSIRSType = {'nzp','zp'};
            csirs.CSIRSPeriod = {[5 1],[5 1]};
            csirs.Density = {'one','one'};
            csirs.RowNumber = [6 5]; %[3 5];
            csirs.SymbolLocations = {4,6}; % {1,6};
            csirs.SubcarrierLocations = {[1 3 5 7],4}; %{6,4};
            csirs.NumRB = NSizeGrid;
        end

        powerCSIRS = 0; % CSI-RS power scaling

        sym = nrCSIRS(carrier,csirs);
        csirsSym = sym*db2mag(powerCSIRS);
        csirsInd = nrCSIRSIndices(carrier,csirs);

        ports = max(csirs.NumCSIRSPorts);   % Number of antenna ports
        txGrid = nrResourceGrid(carrier,ports);

        if flag_plot %plot Carrier Grid
            disp(['CSI-RS power scaling: ' num2str(powerCSIRS) ' dB']);
            plotGrid(size(txGrid),csirsInd,csirsSym);
        end
        txGrid(csirsInd) = csirsSym;
        [txWaveform,ofdmInfo] = nrOFDMModulate(carrier,txGrid);

        varargout{1} = txWaveform;
        % varargout{2} = ofdmInfo;
        varargout{2} = carrier;
        varargout{3} = csirsSym;
        varargout{4} = csirsInd;
    end
end