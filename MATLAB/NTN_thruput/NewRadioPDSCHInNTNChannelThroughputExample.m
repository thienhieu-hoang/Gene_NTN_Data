simParameters = struct;                   % Create simParameters structure to
                                          % contain all key simulation parameters
simParameters.NFrames = 2;                % Number of 10 ms frames
simParameters.TxPower = 60:65;            % Transmit power (dBm)
simParameters.RxNoiseFigure = 6;          % Noise figure (dB)
simParameters.RxAntennaTemperature = 290; % Antenna temperature (K)

displaySimulationInformation = true;

simParameters.DopplerPreCompensator = true;
simParameters.PreCompensationDopplerShift = [];                   % In Hz
simParameters.RxDopplerCompensator = false;
simParameters.RxDopplerCompensationMethod = "independent time-freq";
% The example uses below fields to estimate Doppler shift, when
% RxDopplerCompensator is set to true and RxDopplerCompensationMethod is
% set to joint time-freq.
% Set the search range of Doppler shift in Hz [MIN,MAX]
simParameters.FrequencyRange = [-50e3 50e3];
% Set the search range resolution of Doppler shift in Hz
simParameters.FrequencyResolution = 1e3;

simParameters.InitialTimingSynchronization = "joint time-freq";
% The example uses below fields to perform initial synchronization, when
% InitialTimingSynchronization is set to joint time-freq.
% Set the initial search range of Doppler shift in Hz [MIN,MAX]
simParameters.InitialFrequencyRange = [-50e3 50e3];
% Set the initial search range resolution of Doppler shift in Hz
simParameters.InitialFrequencyResolution = 1e3;

% Set waveform type and PDSCH numerology (SCS and CP type)
simParameters.Carrier = nrCarrierConfig;
simParameters.Carrier.SubcarrierSpacing = 30;
simParameters.Carrier.CyclicPrefix = "Normal";
% Bandwidth in number of RBs (11 RBs at 30 kHz SCS for 5 MHz bandwidth)
simParameters.Carrier.NSizeGrid = 11;
% Physical layer cell identity
simParameters.Carrier.NCellID = 1;

% PDSCH/DL-SCH parameters
% This PDSCH definition is the basis for all PDSCH transmissions in the
% throughput simulation
simParameters.PDSCH = nrPDSCHConfig;
% This structure is to hold additional simulation parameters for the DL-SCH
% and PDSCH
simParameters.PDSCHExtension = struct();

% Define PDSCH time-frequency resource allocation per slot to be full grid
% (single full grid BWP)
% PDSCH PRB allocation
simParameters.PDSCH.PRBSet = 0:simParameters.Carrier.NSizeGrid-1;
% Starting symbol and number of symbols of each PDSCH allocation
simParameters.PDSCH.SymbolAllocation = [0,simParameters.Carrier.SymbolsPerSlot];
simParameters.PDSCH.MappingType = "A";

% Scrambling identifiers
simParameters.PDSCH.NID = simParameters.Carrier.NCellID;
simParameters.PDSCH.RNTI = 1;

% PDSCH resource block mapping (TS 38.211 Section 7.3.1.6)
simParameters.PDSCH.VRBToPRBInterleaving = 0;
simParameters.PDSCH.VRBBundleSize = 4;

% Define the number of transmission layers to be used
simParameters.PDSCH.NumLayers = 1;

% Define codeword modulation and target coding rate
% The number of codewords is directly dependent on the number of layers so
% ensure that layers are set first before getting the codeword number
if simParameters.PDSCH.NumCodewords > 1
    % Multicodeword transmission (when number of layers being > 4)
    simParameters.PDSCH.Modulation = ["16QAM","16QAM"];
    % Code rate used to calculate transport block sizes
    simParameters.PDSCHExtension.TargetCodeRate = [490 490]/1024;
else
    simParameters.PDSCH.Modulation = "16QAM";
    % Code rate used to calculate transport block size
    simParameters.PDSCHExtension.TargetCodeRate = 490/1024;
end

% DM-RS and antenna port configuration (TS 38.211 Section 7.4.1.1)
simParameters.PDSCH.DMRS.DMRSPortSet = []; % Use empty to auto-configure the DM-RS ports
simParameters.PDSCH.DMRS.DMRSTypeAPosition = 2;
simParameters.PDSCH.DMRS.DMRSLength = 1;
simParameters.PDSCH.DMRS.DMRSAdditionalPosition = 2;
simParameters.PDSCH.DMRS.DMRSConfigurationType = 2;
simParameters.PDSCH.DMRS.NumCDMGroupsWithoutData = 1;
simParameters.PDSCH.DMRS.NIDNSCID = 1;
simParameters.PDSCH.DMRS.NSCID = 0;

% PT-RS configuration (TS 38.211 Section 7.4.1.2)
simParameters.PDSCH.EnablePTRS = 0;
simParameters.PDSCH.PTRS.TimeDensity = 1;
simParameters.PDSCH.PTRS.FrequencyDensity = 2;
simParameters.PDSCH.PTRS.REOffset = "00";
% PT-RS antenna port, subset of DM-RS port set. Empty corresponds to lowest
% DM-RS port number
simParameters.PDSCH.PTRS.PTRSPortSet = [];

% Reserved PRB patterns, if required (for CORESETs, forward compatibility etc)
simParameters.PDSCH.ReservedPRB{1}.SymbolSet = [];   % Reserved PDSCH symbols
simParameters.PDSCH.ReservedPRB{1}.PRBSet = [];      % Reserved PDSCH PRBs
simParameters.PDSCH.ReservedPRB{1}.Period = [];      % Periodicity of reserved resources

% Additional simulation and DL-SCH related parameters
% PDSCH PRB bundling (TS 38.214 Section 5.1.2.3)
simParameters.PDSCHExtension.PRGBundleSize = [];     % 2, 4, or [] to signify "wideband"
% Rate matching or transport block size (TBS) parameters
% Set PDSCH rate matching overhead for TBS (Xoh) to 6 when PT-RS is enabled, otherwise 0
simParameters.PDSCHExtension.XOverhead = 6*simParameters.PDSCH.EnablePTRS;
% HARQ parameters
% Number of parallel HARQ processes to use
simParameters.PDSCHExtension.NHARQProcesses = 1;
% Enable retransmissions for each process, using redundancy version (RV) sequence [0,2,3,1]
simParameters.PDSCHExtension.EnableHARQ = false;
% LDPC decoder parameters
% Available algorithms: Belief propagation, Layered belief propagation,
%                       Normalized min-sum, Offset min-sum
simParameters.PDSCHExtension.LDPCDecodingAlgorithm = "Normalized min-sum";
simParameters.PDSCHExtension.MaximumLDPCIterationCount = 6;

% Define the overall transmission antenna geometry at end-points
% For NTN narrowband channel, only single-input-single-output (SISO)
% transmission is allowed
% Number of PDSCH transmission antennas (1,2,4,8,16,32,64,128,256,512,1024) >= NumLayers
simParameters.NumTransmitAntennas = 1;
if simParameters.PDSCH.NumCodewords > 1 % Multi-codeword transmission
    % Number of UE receive antennas (even number >= NumLayers)
    simParameters.NumReceiveAntennas = 8;
else
    % Number of UE receive antennas (1 or even number >= NumLayers)
    simParameters.NumReceiveAntennas = 1;
end
% Define data type for resource grids and waveforms
simParameters.DataType = "double";

waveformInfo = nrOFDMInfo(simParameters.Carrier);

% Define the general NTN propagation channel parameters
% Set the NTN channel type to Narrowband for an NTN narrowband channel and
% set the NTN channel type to TDL for an NTN TDL channel.
simParameters.NTNChannelType = "TDL";

% Include or exclude free space path loss
simParameters.IncludeFreeSpacePathLoss = true;

% Delay model configuration
% This example models only one-way propagation delay and provides immediate
% feedback without any delay
simParameters.DelayModel = "None";    % "None", "Static", or "Time-varying"

% Set the parameters common to both NTN narrowband and NTN TDL channels
simParameters.CarrierFrequency = 2e9;                  % Carrier frequency (in Hz)
simParameters.ElevationAngle = 50;                     % Elevation angle (in degrees)
simParameters.MobileSpeed = 3*1000/3600;               % Speed of mobile terminal (in m/s)
simParameters.MobileAltitude = 0;                      % Mobile altitude (in m)
simParameters.SatelliteAltitude = 600000;              % Satellite altitude (in m)
simParameters.SampleRate = waveformInfo.SampleRate;
simParameters.RandomStream = "mt19937ar with seed";
simParameters.Seed = 73;
simParameters.OutputDataType = simParameters.DataType;


% Set the following fields for NTN TDL channel
if simParameters.NTNChannelType == "TDL"
    simParameters.DelayProfile = "NTN-TDL-A";
    simParameters.DelaySpread = 30e-9;
end

% Change the current folder to the folder of this file.
if(~isdeployed)
  cd(fileparts(matlab.desktop.editor.getActiveFilename));
end

% Cross-check the PDSCH layering against the channel geometry
HelperNRNTNThroughput.validateNumLayers(simParameters);

% Calculate the Doppler shift due to satellite movement
c = physconst("lightspeed");
satelliteDopplerShift = dopplerShiftCircularOrbit( ...
    simParameters.ElevationAngle,simParameters.SatelliteAltitude, ...
    simParameters.MobileAltitude,simParameters.CarrierFrequency);

% Define NTN TDL channel based on specified fields in simParameters
% structure
if simParameters.NTNChannelType == "TDL"
    channel = nrTDLChannel;
    channel.DelayProfile = simParameters.DelayProfile;
    channel.DelaySpread = simParameters.DelaySpread;
    channel.SatelliteDopplerShift = satelliteDopplerShift;
    channel.MaximumDopplerShift = ...
        simParameters.MobileSpeed*simParameters.CarrierFrequency/c;
    channel.NumTransmitAntennas = simParameters.NumTransmitAntennas;
    channel.NumReceiveAntennas = simParameters.NumReceiveAntennas;
end

% Assign the parameters common to both TDL and narrowband channels
channel.SampleRate = simParameters.SampleRate;
channel.RandomStream = simParameters.RandomStream;
channel.Seed = simParameters.Seed;

% Get the maximum number of delayed samples due to a channel multipath
% component. The maximum number of delayed samples is calculated from the
% channel path with the maximum delay and the implementation delay of the
% channel filter. This number of delay samples is required later to buffer
% and process the received signal with the expected length.
chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + ...
    chInfo.ChannelFilterDelay;

% Compute the noise amplitude per receive antenna
kBoltz = physconst("boltzmann");
NF = 10^(simParameters.RxNoiseFigure/10);
T0 = 290;                                               % Noise temperature at the input (K)
Teq = simParameters.RxAntennaTemperature + T0*(NF-1);   % K
N0_ampl = sqrt(kBoltz*waveformInfo.SampleRate*Teq/2.0);

% Number of transmit power points
numTxPowerPoints = length(simParameters.TxPower);
% Array to store the maximum throughput for all transmit power points
maxThroughput = zeros(numTxPowerPoints,1);
% Array to store the simulation throughput for all transmit power points
simThroughput = zeros(numTxPowerPoints,1);
% Array to store the signal-to-noise ratio (SNR) for all transmit power points
snrVec = zeros(numTxPowerPoints,1);

% Common Doppler shift for use in the simulations
if isempty(simParameters.PreCompensationDopplerShift)
    commonDopplerShift = satelliteDopplerShift;
else
    commonDopplerShift = simParameters.PreCompensationDopplerShift;
end

% Set up RV sequence for all HARQ processes
if simParameters.PDSCHExtension.EnableHARQ
    % In the final report of RAN WG1 meeting #91 (R1-1719301), it was
    % observed in R1-1717405 that if performance is the priority, [0 2 3 1]
    % should be used. If self-decodability is the priority, it should be
    % taken into account that the upper limit of the code rate at which
    % each RV is self-decodable is in the following order: 0>3>2>1
    rvSeq = [0 2 3 1];
else
    % In case of HARQ disabled, RV is set to 0
    rvSeq = 0;
end

% Create DL-SCH encoder System object to perform transport channel encoding
encodeDLSCH = nrDLSCH;
encodeDLSCH.MultipleHARQProcesses = true;
encodeDLSCH.TargetCodeRate = simParameters.PDSCHExtension.TargetCodeRate;

% Create DL-SCH decoder System object to perform transport channel decoding
decodeDLSCH = nrDLSCHDecoder;
decodeDLSCH.MultipleHARQProcesses = true;
decodeDLSCH.TargetCodeRate = simParameters.PDSCHExtension.TargetCodeRate;
decodeDLSCH.LDPCDecodingAlgorithm = simParameters.PDSCHExtension.LDPCDecodingAlgorithm;
decodeDLSCH.MaximumLDPCIterationCount = ...
    simParameters.PDSCHExtension.MaximumLDPCIterationCount;

% Initialize objects to model delay
tmpInfo = HelperNRNTNThroughput.initializeDelayObjects(simParameters,waveformInfo);
staticDelay = tmpInfo.StaticDelay;
variableIntegerDelay = tmpInfo.VariableIntegerDelay;
variableFractionalDelay = tmpInfo.VariableFractionalDelay;
maxVarPropDelay = tmpInfo.MaxVariablePropDelay;
numVariableIntegSamples = tmpInfo.NumVariableIntegerDelaySamples;
numVariableFracDelaySamples = tmpInfo.NumVariableFractionalDelaySamples;
delayInSeconds = tmpInfo.DelayInSeconds;
pathLoss = tmpInfo.PathLoss;
SU = tmpInfo.SlantDistance;

% Get the starting time of each slot
[slotTimes,symLen] = HelperNRNTNThroughput.getSlotTimes( ...
    simParameters.Carrier.SymbolsPerSlot,waveformInfo.SymbolLengths, ...
    waveformInfo.SampleRate,simParameters.NFrames,simParameters.DataType);

% Check the number of HARQ processes and initial propagation delay
initialSlotDelay = find(slotTimes>=delayInSeconds(1),1)-1;
if simParameters.PDSCHExtension.EnableHARQ
    if simParameters.PDSCHExtension.NHARQProcesses < initialSlotDelay
        error("In case of HARQ, this example supports transmission of continuous data only. " + ... 
            "Set the number of HARQ processes (" + (simParameters.PDSCHExtension.NHARQProcesses) +...
            ") to a value greater than or equal to the maximum propagation delay in slots (" + ...
            initialSlotDelay +").")
    end
end

% Initial frequency shift search space
if simParameters.InitialTimingSynchronization == "joint time-freq"
    inifVals = simParameters.InitialFrequencyRange(1):simParameters.InitialFrequencyResolution:simParameters.InitialFrequencyRange(2);
else
    inifVals = 0;
end

% Frequency shift search space
if simParameters.RxDopplerCompensator == 1 ...
        && simParameters.RxDopplerCompensationMethod == "joint time-freq"
    fVals = simParameters.FrequencyRange(1):simParameters.FrequencyResolution:simParameters.FrequencyRange(2);
else
    % In case of no receiver Doppler compensation, treat the frequency
    % value is 0 Hz.
    fVals = 0;
end

% Initialize the power amplifier function handle or System object depending
% on the input configuration
[hpa,hpaDelay,paInputScaleFactor] = ...
    HelperNRNTNThroughput.initializePA(paModel,hasMemory,paCharacteristics,coefficients);

% Repeat hpa to have independent processing for each antenna
hpa = repmat({hpa},1,simParameters.NumTransmitAntennas);

% Update the power amplifier input scaling factor, based on scaleFactor
if ~isempty(scaleFactor)
    paInputScaleFactor = scaleFactor;
end

% Set a threshold value to detect the valid OFDM symbol boundary. For a
% SISO case, a threshold of 0.48 can be used to have probability of
% incorrect boundary detection around 0.01. Use 0 to avoid thresholding
% logic.
dtxThresold = 0.48;

% Use an offset to account for the common delay. The example, by default,
% does not introduce any common delay and only passes through the channel.
sampleDelayOffset = 0; % Number of samples

% Set usePreviousShift variable to true, to use the shift value estimated
% in first slot directly for the consecutive slots. When set to false, the
% shift is calculated for each slot, considering the range of shift values
% to be whole cyclic prefix length. This is used in the estimation of
% integer Doppler shift.
usePreviousShift = false;

% Set useDiffCorr variable to true, to use the shift estimated from
% differential correlation directly in the integer Doppler shift
% estimation. When set to false, the range of shift values also include the
% shift estimated from differential correlation.
useDiffCorr = true;

% Set the amplitude scaling factor to use in energy detection. For the
% default case, a factor of 1.03 is used to avoid missed detections at 60
% dBm transmit power. A value of 0 assumes each sample is an actual signal.
amplThreshold = 1.03;

% Use the minimum number of samples for a slot in the whole frame as
% window length
mrms = dsp.MovingRMS;
slotsPerSubFrameFlag = simParameters.Carrier.SlotsPerSubframe > 1;
mrms.WindowLength = symLen((1+(slotsPerSubFrameFlag))*simParameters.Carrier.SymbolsPerSlot) ...
    -slotsPerSubFrameFlag*symLen(simParameters.Carrier.SymbolsPerSlot);

% Processing loop
for txPowIdx = 1:numTxPowerPoints     % Comment out for parallel computing
% parfor txPowIdx = 1:numTxPowerPoints % Uncomment for parallel computing
    % To reduce the total simulation time, you can execute this loop in
    % parallel by using Parallel Computing Toolbox features. Comment
    % out the for-loop statement and uncomment the parfor-loop statement.
    % If Parallel Computing Toolbox is not installed, parfor-loop defaults
    % to a for-loop statement. Because the parfor-loop iterations are
    % executed in parallel in a nondeterministic order, the simulation
    % information displayed for each transmit power point can be intertwined.
    % To switch off the simulation information display, set the
    % displaySimulationInformation variable (defined earlier in this
    % example) to false.

    % Reset the random number generator so that each transmit power point
    % experiences the same noise realization
    rng(0,"twister");

    % Make copies of the simulation-level parameter structures so that they
    % are not Parallel Computing Toolbox broadcast variables when using parfor
    simLocal = simParameters;
    waveinfoLocal = waveformInfo;

    % Make copies of channel-level parameters to simplify subsequent
    % parameter referencing
    carrier = simLocal.Carrier;
    rxCarrier = carrier;
    pdsch = simLocal.PDSCH;
    pdschextra = simLocal.PDSCHExtension;
    % Copy of the decoder handle to help Parallel Computing Toolbox
    % classification
    decodeDLSCHLocal = decodeDLSCH;
    decodeDLSCHLocal.reset();       % Reset decoder at the start of each transmit power point

    % Make copies of intermediate variables to have warning-free execution
    % with Parallel Computing Toolbox
    thres = dtxThresold;
    sampleOffset = sampleDelayOffset;
    usePrevShift = usePreviousShift;
    useDiffCorrFlag = useDiffCorr;
    N0 = N0_ampl;
    pl_dB = pathLoss;
    varIntegSamples = numVariableIntegSamples;
    varFracSamples = numVariableFracDelaySamples;
    fValsVec = fVals;
    inifValsVec = inifVals;
    threshFactor = amplThreshold;
    initialDelay = initialSlotDelay;
    preDopplerShift = commonDopplerShift;

    % Initialize temporary variables
    offset = 0;
    shiftOut = 0;
    txHarqProc = 0;
    rxHarqProc = 0;
    prevWave = [];
    pathFilters = [];
    rxBuff = [];
    syncCheck = true;

    % Reset the channel so that each transmit power point experiences the
    % same channel realization
    reset(channel);

    % Reset the power amplifier
    for numHPA = 1:numel(hpa)
        if ~isa(hpa{numHPA},"function_handle")
            reset(hpa{numHPA})
        end
    end

    % Reset the delay objects
    reset(staticDelay)
    if isa(variableIntegerDelay,"dsp.VariableIntegerDelay")
        reset(variableIntegerDelay)
    end
    if isa(variableFractionalDelay,"dsp.VariableFractionalDelay")
        reset(variableFractionalDelay)
    end

    % Reset the moving RMS object
    reset(mrms)

    % Transmit power value in dBm
    txPowerdBm = simLocal.TxPower(txPowIdx);

    % Specify the order in which we cycle through the HARQ process
    % identifiers
    harqSequence = 0:pdschextra.NHARQProcesses-1;

    % Initialize the state of all HARQ processes
    % Create a parallel array of all HARQ processes
    harqEntity = cell(pdschextra.NHARQProcesses,1);
    for harqId = 1:pdschextra.NHARQProcesses
        harqEntity{harqId} = HARQEntity(harqSequence(harqId),rvSeq,pdsch.NumCodewords);
    end

    % Total number of slots in the simulation period
    NSlots = simLocal.NFrames*carrier.SlotsPerFrame;

    % Obtain a precoding matrix (wtx) to use in the transmission of the
    % first transport block
    [estChannelGrid,sampleTimes] = HelperNRNTNThroughput.getInitialChannelEstimate(...
        carrier,simLocal.NumTransmitAntennas,channel,simLocal.DataType);
    newWtx = HelperNRNTNThroughput.getPrecodingMatrix( ...
        carrier,pdsch,estChannelGrid,pdschextra.PRGBundleSize);

    % Loop over the entire waveform length
    for nslot = 0:NSlots-1

        % Update carrier slot number to account for new slot transmission
        carrier.NSlot = nslot;

        % Calculate the transport block sizes for the transmission in the slot
        trBlkSizes = nrTBS(pdsch,pdschextra.TargetCodeRate,pdschextra.XOverhead);

        % Set transport block depending on the HARQ process
        for cwIdx = 1:pdsch.NumCodewords
            % Create a new DL-SCH transport block for new data in the
            % current process
            if harqEntity{txHarqProc+1}.NewData(cwIdx)
                trBlk = randi([0 1],trBlkSizes(cwIdx),1,'int8');
                setTransportBlock(encodeDLSCH,trBlk,cwIdx-1,harqEntity{txHarqProc+1}.HARQProcessID);
                % Flush decoder soft buffer explicitly for any new data
                % because of previous RV sequence time out
                if harqEntity{txHarqProc+1}.SequenceTimeout(cwIdx)
                    resetSoftBuffer(decodeDLSCHLocal,cwIdx-1,harqEntity{txHarqProc+1}.HARQProcessID);
                end
            end
        end

        % Get precoding matrix (wtx) calculated in previous slot
        wtx = newWtx;

        % Create a structure with transport block encoder
        dlsch = struct;
        dlsch.Encoder = encodeDLSCH;
        dlsch.RedundancyVersion = harqEntity{txHarqProc+1}.RedundancyVersion;
        dlsch.HARQProcessID = harqEntity{txHarqProc+1}.HARQProcessID;

        % Generate time-domain waveform
        txWaveform0 = HelperNRNTNThroughput.generatePDSCHWaveform( ...
            carrier,pdsch,dlsch,wtx,simLocal.DataType);

        % Normalize the waveform with maximum waveform amplitude
        txWaveform = txWaveform0./max(abs(txWaveform0));
