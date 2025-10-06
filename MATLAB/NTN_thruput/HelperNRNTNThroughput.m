classdef HelperNRNTNThroughput
    %HelperNRNTNThroughput Class defining the supporting functions used in
    %the NR NTN PDSCH Throughput example
    %
    %   Note: This is an undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2021-2025 The MathWorks, Inc.

    methods (Static)
        function validateNumLayers(simParameters)
            % Validate the number of layers, relative to the antenna geometry

            numlayers = simParameters.PDSCH.NumLayers;
            ntxants = simParameters.NumTransmitAntennas;
            nrxants = simParameters.NumReceiveAntennas;

            if contains(simParameters.NTNChannelType,'Narrowband','IgnoreCase',true)
                if (ntxants ~= 1) || (nrxants ~= 1)
                    error(['For NTN narrowband channel, ' ...
                        'the number of transmit and receive antennas must be 1.']);
                end
            end

            antennaDescription = sprintf(...
                'min(NumTransmitAntennas,NumReceiveAntennas) = min(%d,%d) = %d', ...
                ntxants,nrxants,min(ntxants,nrxants));
            if numlayers > min(ntxants,nrxants)
                error('The number of layers (%d) must satisfy NumLayers <= %s', ...
                    numlayers,antennaDescription);
            end

            % Display a warning if the maximum possible rank of the channel equals
            % the number of layers
            if (numlayers > 2) && (numlayers == min(ntxants,nrxants))
                warning(['The maximum possible rank of the channel, given by %s, is equal to' ...
                    ' NumLayers (%d). This may result in a decoding failure under some channel' ...
                    ' conditions. Try decreasing the number of layers or increasing the channel' ...
                    ' rank (use more transmit or receive antennas).'],antennaDescription, ...
                    numlayers); %#ok<SPWRN>
            end

        end

        function [estChannelGrid,sampleTimes] = getInitialChannelEstimate(...
                carrier,nTxAnts,channel,dataType)
            % Obtain channel estimate before first transmission. Use this function to
            % obtain a precoding matrix for the first slot.

            ofdmInfo = nrOFDMInfo(carrier);

            chInfo = info(channel);
            maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) ...
                + chInfo.ChannelFilterDelay;

            % Temporary waveform (only needed for the sizes)
            tmpWaveform = zeros(...
                (ofdmInfo.SampleRate/1000/carrier.SlotsPerSubframe)+maxChDelay,nTxAnts,dataType);

            % Filter through channel and get the path gains and path filters
            [~,pathGains,sampleTimes] = channel(tmpWaveform);
            if isa(channel,'nrTDLChannel')
                pathFilters = getPathFilters(channel);
            else
                pathFilters = chInfo.ChannelFilterCoefficients.';
            end

            % Perfect timing synchronization
            offset = nrPerfectTimingEstimate(pathGains,pathFilters);

            % Perfect channel estimate
            estChannelGrid = nrPerfectChannelEstimate(...
                carrier,pathGains,pathFilters,offset,double(sampleTimes));

        end

        function wtx = getPrecodingMatrix(carrier,pdsch,hestGrid,prgbundlesize)
            % Calculate precoding matrices for all precoding resource block groups
            % (PRGs) in the carrier that overlap with the PDSCH allocation

            % Maximum common resource block (CRB) addressed by carrier grid
            maxCRB = carrier.NStartGrid + carrier.NSizeGrid - 1;

            % PRG size
            if nargin==4 && ~isempty(prgbundlesize)
                Pd_BWP = prgbundlesize;
            else
                Pd_BWP = maxCRB + 1;
            end

            % PRG numbers (1-based) for each RB in the carrier grid
            NPRG = ceil((maxCRB + 1) / Pd_BWP);
            prgset = repmat((1:NPRG),Pd_BWP,1);
            prgset = prgset(carrier.NStartGrid + (1:carrier.NSizeGrid).');

            [~,~,R,P] = size(hestGrid);
            wtx = zeros([pdsch.NumLayers P NPRG],'like',hestGrid);
            for i = 1:NPRG

                % Subcarrier indices within current PRG and within the PDSCH
                % allocation
                thisPRG = find(prgset==i) - 1;
                thisPRG = intersect(thisPRG,pdsch.PRBSet(:) + carrier.NStartGrid,'rows');
                prgSc = (1:12)' + 12*thisPRG';
                prgSc = prgSc(:);

                if (~isempty(prgSc))

                    % Average channel estimate in PRG
                    estAllocGrid = hestGrid(prgSc,:,:,:);
                    Hest = permute(mean(reshape(estAllocGrid,[],R,P)),[2 3 1]);

                    % SVD decomposition
                    [~,~,V] = svd(Hest);
                    wtx(:,:,i) = V(:,1:pdsch.NumLayers).';

                end

            end

            wtx = wtx / sqrt(pdsch.NumLayers); % Normalize by NumLayers

        end

        function estChannelGrid = precodeChannelEstimate(carrier,estChannelGrid,W)
            % Apply precoding matrix W to the last dimension of the channel estimate

            [K,L,R,P] = size(estChannelGrid);
            estChannelGrid = reshape(estChannelGrid,[K*L R P]);
            estChannelGrid = nrPDSCHPrecode( ...
                carrier,estChannelGrid,reshape(1:numel(estChannelGrid),[K*L R P]),W);
            estChannelGrid = reshape(estChannelGrid,K,L,R,[]);

        end

        function [loc,wMovSum,pho,bestAnt] = detectOFDMSymbolBoundary(rxWave,nFFT,cpLen,thres)
            % Detect OFDM symbol boundary by calculating correlation of cyclic prefix

            % Capture the dimensions of received waveform
            [NSamples,R] = size(rxWave);

            % Append zeros of length nFFT across each receive antenna to the
            % received waveform
            waveformZeroPadded = [rxWave;zeros(nFFT,R,'like',rxWave)];

            % Get the portion of waveform till the last nFFT samples
            wavePortion1 = waveformZeroPadded(1:end-nFFT,:);

            % Get the portion of waveform delayed by nFFT
            wavePortion2 = waveformZeroPadded(1+nFFT:end,:);

            % Get the energy of each sample in both the waveform portions
            eWavePortion1 = abs(wavePortion1).^2;
            eWavePortion2 = abs(wavePortion2).^2;

            % Initialize the temporary variables
            wMovSum = zeros([NSamples R]);
            wEnergyPortion1 = zeros([NSamples+cpLen-1 R]);
            wEnergyPortion2 = wEnergyPortion1;

            % Perform correlation for each sample with the sample delayed by nFFT
            waveCorr = wavePortion1.*conj(wavePortion2);
            % Calculate the moving sum value for each cpLen samples, across each
            % receive antenna
            oneVec = ones(cpLen,1);
            for i = 1:R
                wConv = conv(waveCorr(:,i),oneVec);
                wMovSum(:,i) = wConv(cpLen:end);
                wEnergyPortion1(:,i) = conv(eWavePortion1(:,i),oneVec);
                wEnergyPortion2(:,i) = conv(eWavePortion2(:,i),oneVec);
            end

            % Get the normalized correlation value for the moving sum matrix
            pho = abs(wMovSum)./ ...
                (eps+sqrt(wEnergyPortion1(cpLen:end,:).*wEnergyPortion2(cpLen:end,:)));

            % Detect the peak locations in each receive antenna based on the
            % threshold. These peak locations correspond to the starting location
            % of each OFDM symbol in the received waveform.
            loc = cell(R,1);
            m = zeros(R,1);
            phoFactor = ceil(NSamples/nFFT);
            phoExt = [pho; -1*ones(phoFactor*nFFT - NSamples,R)];
            for col = 1:R
                p1 = reshape(phoExt(:,i),[],phoFactor);
                [pks,locTemp] = max(p1);
                locTemp = locTemp + (0:phoFactor-1).*nFFT;
                indicesToConsider = pks>=thres;
                loc{col} = locTemp(indicesToConsider);
                m(col) = max(pks);
            end
            bestAnt = find(m == max(m));

        end

        function [out,detFlag] = estimateFractionalDopplerShift(rxWave,scs, ...
                nFFT,cpLen,thres,flag)
            % Estimate the fractional Doppler shift using cyclic prefix

            if flag
                % Detect the OFDM boundary locations
                [loc,wMovSum,~,bestAnt] =  ...
                    HelperNRNTNThroughput.detectOFDMSymbolBoundary(rxWave, ...
                    nFFT,cpLen,thres);

                % Get the average correlation value at the peak locations for the
                % first receive antenna having maximum correlation value
                wSamples = nan(1,1);
                if ~isempty(loc{bestAnt(1)})
                    wSamples(1) = mean(wMovSum(loc{bestAnt(1)},bestAnt(1)));
                end

                % Compute the fractional Doppler shift
                if ~all(isnan(wSamples))
                    out = -(mean(angle(wSamples),'omitnan')*scs*1e3)/(2*pi);
                    % Flag to indicate that there is at least one OFDM symbol
                    % detected
                    detFlag = 1;
                else
                    out = 0;
                    detFlag = 0;
                end
            else
                out = 0;
                detFlag = 0;
            end

        end

        function [out,shiftOut] = estimateIntegerDopplerShift(carrier,rx,refInd, ...
                refSym,sampleOffset,usePrevShift,useDiffCorr,shiftIn,maxOffset,flag)
            % Estimate the integer Doppler shift using demodulation reference signal

            arguments
                carrier
                rx
                refInd
                refSym
                sampleOffset = 0
                usePrevShift = false
                useDiffCorr = true
                shiftIn = 0
                maxOffset = 0
                flag = false
            end

            if flag

                % Get OFDM information
                ofdmInfo = nrOFDMInfo(carrier);
                cpLen = ofdmInfo.CyclicPrefixLengths(1); % Highest cyclic prefix length
                K = carrier.NSizeGrid*12;                % Number of subcarriers
                L = carrier.SymbolsPerSlot;              % Number of OFDM symbols in slot
                P = ceil(max(double(refInd(:))/(K*L)));  % Number of layers

                % Find the timing offset using differential correlation
                offset = HelperNRNTNThroughput.diffcorr( ...
                    carrier,rx,refInd,refSym);
                if offset > maxOffset
                    offset = 0;
                end

                % Range of shift values to be used in integer frequency offset
                % estimation
                if useDiffCorr
                    % Use offset directly in the shift values
                    shiftValues = offset+1;
                else
                    shiftValues = sampleOffset + shiftIn;
                    if ~(usePrevShift && (shiftIn > 0))
                        % Update range of shift values such that whole cyclic prefix
                        % length is covered
                        shiftValues = sampleOffset + (1:(cpLen+offset));
                    end
                end

                % Initialize temporary variables
                shiftLen = length(shiftValues);
                maxValue = complex(zeros(shiftLen,1));
                binIndex = zeros(shiftLen,1);
                [rxLen,R] = size(rx);
                xWave = zeros([rxLen P],'like',rx);

                % Generate reference waveform
                refGrid = nrResourceGrid(carrier,P);
                refGrid(refInd) = refSym;
                refWave = nrOFDMModulate(carrier,refGrid,'Windowing',0);
                refWave = [refWave; zeros((rxLen-size(refWave,1)),P,'like',refWave)];

                % Find the fast Fourier transform (FFT) bin corresponding to
                % maximum correlation value for each shift value
                for shiftIdx = 1:shiftLen
                    % Use the waveform from the shift value and append zeros
                    tmp = rx(shiftValues(shiftIdx):end,:);
                    rx = [tmp; zeros(rxLen-size(tmp,1),R)];

                    % Compute the correlation of received waveform with reference
                    % waveform across different layers and receive antennas
                    for rIdx = 1:R
                        for p = 1:P
                            xWave(:,rIdx,p) = ...
                                rx(:,rIdx).*conj(refWave(1:length(rx(:,rIdx)),p));
                        end
                    end

                    % Aggregate the correlated waveform across multiple ports and
                    % compute energy of the resultant for each receive antenna
                    x1 = sum(xWave,3);
                    x1P = sum(abs(x1).^2);

                    % Find the index of first receive antenna which has maximum
                    % correlation energy
                    idx = find(x1P == max(x1P),1);

                    % Combine the received waveform which have maximum correlation
                    % energy
                    xWaveCombined = sum(x1(:,idx(1)),2);

                    % Compute FFT of the resultant waveform
                    xWaveCombinedTemp = buffer(xWaveCombined,ofdmInfo.Nfft);
                    xFFT = sum(fftshift(fft(xWaveCombinedTemp)),2);

                    % Store the value and location of peak
                    [maxValue(shiftIdx),binIndex(shiftIdx)] = max(xFFT);
                end

                % FFT bin values
                fftBinValues = (-ofdmInfo.Nfft/2:(ofdmInfo.Nfft/2-1))*(ofdmInfo.SampleRate/ofdmInfo.Nfft);

                % Find the shift index that corresponds to the maximum of peak
                % value of all the shifted waveforms. Use the FFT bin index
                % corresponding to this maximum shift index. The FFT bin value
                % corresponding to this bin index is the integer frequency offset.
                [~,maxId] = max(maxValue);
                loc = binIndex(maxId);
                out = fftBinValues(loc);
                shiftOut = shiftValues(maxId);
            else
                out = 0;
                shiftOut = 1+sampleOffset;
            end

        end

        function out = compensateDopplerShift(inWave,fs,fdSat,flag)
            % Perform Doppler shift correction

            t = (0:size(inWave,1)-1)'/fs;
            if flag
                out = inWave.*exp(1j*2*pi*(-fdSat)*t);
            else
                out = inWave;
            end

        end

        function [offset,mag] = diffcorr(carrier,rx,refInd,refSym)
            % Perform differential correlation for the received signal

            % Get the number of subcarriers, OFDM symbols, and layers
            K = carrier.NSizeGrid*12;                % Number of subcarriers
            L = carrier.SymbolsPerSlot;              % Number of OFDM symbols in slot
            P = ceil(max(double(refInd(:))/(K*L)));  % Number of layers

            % Generate the reference signal
            refGrid = nrResourceGrid(carrier,P);
            refGrid(refInd) = refSym;
            refWave = nrOFDMModulate(carrier,refGrid,'Windowing',0);

            % Get the differential of the received signal and reference signal
            waveform = conj(rx(1:end-1,:)).*rx(2:end,:);
            ref = conj(refWave(1:end-1,:)).*refWave(2:end,:);
            [T,R] = size(waveform);

            % To normalize the xcorr behavior, pad the input waveform to make it
            % longer than the reference signal
            refLen = size(ref,1);
            waveformPad = [waveform; zeros([refLen-T R],'like',waveform)];

            % Store correlation magnitude for each time sample, receive antenna and
            % port
            mag = zeros([max(T,refLen) R P],'like',waveformPad);
            for r = 1:R
                for p = 1:P
                    % Correlate the given antenna of the received signal with the
                    % given port of the reference signal
                    refcorr = xcorr(waveformPad(:,r),ref(:,p));
                    mag(:,r,p) = abs(refcorr(T:end));
                end
            end

            % Sum the magnitudes of the ports
            mag = sum(mag,3);

            % Find timing peak in the sum of the magnitudes of the receive antennas
            [~,peakindex] = max(sum(mag,2));
            offset = peakindex - 1;

        end

        function [tO,fO] = jointTimeFreq(carrier,rx,varargin)
            % Perform joint time-frequency synchronization
            % jointTimeFreq(carrier,rx,refInd,refSym,fSearchSpace)
            % jointTimeFreq(carrier,rx,refGrid,fSearchSpace)
            fSearchSpace = varargin{end};
            numFreqVals = length(fSearchSpace);
            peakVal = zeros(numFreqVals,1);
            peakIdx = peakVal;
            ofdmInfo = nrOFDMInfo(carrier);
            fs = ofdmInfo.SampleRate;
            for fIdx = 1:numFreqVals
                rxCorrected = ...
                    HelperNRNTNThroughput.compensateDopplerShift( ...
                    rx,fs,fSearchSpace(fIdx),true);
                [~,corr] = nrTimingEstimate(carrier,rxCorrected,varargin{1:end-1});
                corr = sum(abs(corr),2);
                [peakVal(fIdx),peakIdx(fIdx)] = max(corr);
            end
            [~,id] = max(peakVal);
            % Estimate frequency shift and timing offset
            fO = fSearchSpace(id);
            tO = peakIdx(id)-1;
        end

        % Functions to model power amplifier nonlinearity
        function out = paMemorylessGaAs2Dot1GHz(in)
            % 2.1 GHz GaAs
            absIn = abs(in).^(2*(1:7));
            out = (-0.618347-0.785905i) * in + (2.0831-1.69506i) * in .* absIn(:,1) + ...
                (-14.7229+16.8335i) * in .* absIn(:,2) + (61.6423-76.9171i) * in .* absIn(:,3) + ...
                (-145.139+184.765i) * in .* absIn(:,4) + (190.61-239.371i)* in .* absIn(:,5) + ...
                (-130.184+158.957i) * in .* absIn(:,6) + (36.0047-42.5192i) * in .* absIn(:,7);

        end

        function out = paMemorylessGaN2Dot1GHz(in)
            % 2.1 GHz GaN
            absIn = abs(in).^(2*(1:4));
            out = (0.999952-0.00981788i) * in + (-0.0618171+0.118845i) * in .* absIn(:,1) + ...
                (-1.69917-0.464933i) * in .* absIn(:,2) + (3.27962+0.829737i) * in .* absIn(:,3) + ...
                (-1.80821-0.454331i) * in .* absIn(:,4);

        end

        function out = paMemorylessCMOS28GHz(in)
            % 28 GHz CMOS
            absIn = abs(in).^(2*(1:7));
            out = (0.491576+0.870835i) * in + (-1.26213+0.242689i) * in .* absIn(:,1) + ...
                (7.11693+5.14105i) * in .* absIn(:,2) + (-30.7048-53.4924i) * in .* absIn(:,3) + ...
                (73.8814+169.146i) * in .* absIn(:,4) + (-96.7955-253.635i)* in .* absIn(:,5) + ...
                (65.0665+185.434i) * in .* absIn(:,6) + (-17.5838-53.1786i) * in .* absIn(:,7);

        end

        function out = paMemorylessGaN28GHz(in)
            % 28 GHz GaN
            absIn = abs(in).^(2*(1:5));
            out = (-0.334697-0.942326i) * in + (0.89015-0.72633i) * in .* absIn(:,1) + ...
                (-2.58056+4.81215i) * in .* absIn(:,2) + (4.81548-9.54837i) * in .* absIn(:,3) + ...
                (-4.41452+8.63164i) * in .* absIn(:,4) + (1.54271-2.94034i)* in .* absIn(:,5);

        end

        function paChar = getDefaultLookup
            % The operating specification for the LDMOS-based Doherty
            % amplifier are:
            % * A frequency of 2110 MHz
            % * A peak power of 300 W
            % * A small signal gain of 61 dB
            % Each row in HAV08_Table specifies Pin (dBm), gain (dB), phase
            % shift (degrees) as derived from figure 4 of Hammi, Oualid, et
            % al. "Power amplifiers' model assessment and memory effects
            % intensity quantification using memoryless post-compensation
            % technique." IEEE Transactions on Microwave Theory and
            % Techniques 56.12 (2008): 3170-3179.

            HAV08_Table =...
                [-35,60.53,0.01;
                -34,60.53,0.01;
                -33,60.53,0.08;
                -32,60.54,0.08;
                -31,60.55,0.1;
                -30,60.56,0.08;
                -29,60.57,0.14;
                -28,60.59,0.19;
                -27,60.6,0.23;
                -26,60.64,0.21;
                -25,60.69,0.28;
                -24,60.76,0.21;
                -23,60.85,0.12;
                -22,60.97,0.08;
                -21,61.12,-0.13;
                -20,61.31,-0.44;
                -19,61.52,-0.94;
                -18,61.76,-1.59;
                -17,62.01,-2.73;
                -16,62.25,-4.31;
                -15,62.47,-6.85;
                -14,62.56,-9.82;
                -13,62.47,-12.29;
                -12,62.31,-13.82;
                -11,62.2,-15.03;
                -10,62.15,-16.27;
                -9,62,-18.05;
                -8,61.53,-20.21;
                -7,60.93,-23.38;
                -6,60.2,-26.64;
                -5,59.38,-28.75];
            % Convert the second column of the HAV08_Table from gain to
            % Pout for use by the memoryless nonlinearity System object.
            paChar = HAV08_Table;
            paChar(:,2) = paChar(:,1) + paChar(:,2);
        end

        function out = getDefaultCoefficients
            % The 2.44 GHz memory polynomial model defined in TR 38.803
            % Appendix A. Memory-polynomial depth is 5 and
            % memory-polynomial degree is 5. Rows in the output corresponds
            % to memory depth.
            out = [20.0875+0.4240i -6.3792-0.5507i 0.5809+0.0644i 1.6619+0.1040i -0.3561-0.1033i; ...
                -59.8327-34.7815i -2.4805+0.9344i 4.2741+0.7696i -2.0014-2.3785i -1.2566+1.0495i; ...
                3.2738e2+8.4121e2i 4.4019e2-3.0714e1i -3.5935e2-9.9152e0i 1.6961e2+7.3829e1i -4.1661-21.1090i; ...
                -1.6352e3-5.5757e3i -2.5782e3+3.3332e2i 1.9915e3-1.4479e2i -9.0167e2-5.4617e2i -93.1907+14.2774i; ...
                2.3022e3+1.2348e4i 4.6476e3-1.4477e3i -2.9998e3+1.6071e3i 9.1856e2+9.8066e2i 8.2544e2+6.1424e2i].';
        end

        function [txWaveform,info] = generatePDSCHWaveform(carrier,pdsch,dlsch,wtx,dt,numSlots)
            arguments
                carrier
                pdsch
                dlsch = struct
                wtx = eye(pdsch.NumLayers)
                dt = "double"
                numSlots = 1
            end

            % Initialize variables
            nTx = size(wtx,2);
            tmpGrid = nrResourceGrid(carrier,nTx,OutputDataType=dt);
            pdschGrid = repmat(tmpGrid,[1 numSlots 1]);
            refGrid = pdschGrid;
            nSlotSymb = carrier.SymbolsPerSlot;
            initialNSlot = carrier.NSlot;
            numCW = pdsch.NumCodewords;

            % Process loop for each slot
            for slotIdx = 0:numSlots-1
                [slotGrid,refSlotGrid] = deal(tmpGrid);
                carrier.NSlot = initialNSlot + slotIdx;

                % Perform PDSCH modulation
                [pdschIndices,pdschIndicesInfo] = nrPDSCHIndices(carrier,pdsch);
                if isempty(fieldnames(dlsch))
                    % Update the codeword with valid bits
                    cw = cell(1,numCW);
                    for i = 1:numCW
                        cw{i} = randi([0 1],pdschIndicesInfo.G(i),1);
                    end
                else
                    % Encode with the inputs provided in the dlsch
                    % structure
                    cw = dlsch.Encoder(pdsch.Modulation,pdsch.NumLayers, ...
                        pdschIndicesInfo.G,dlsch.RedundancyVersion, ...
                        dlsch.HARQProcessID);
                end
                if ~isempty(cw)
                    pdschSymbols = nrPDSCH(carrier,pdsch,cw,OutputDataType=dt);
                    % Perform implementation-specific PDSCH MIMO precoding
                    % and mapping
                    [pdschAntSymbols,pdschAntIndices] = nrPDSCHPrecode( ...
                        carrier,pdschSymbols,pdschIndices,wtx);
                    slotGrid(pdschAntIndices) = pdschAntSymbols;
                end

                % Perform implementation-specific PDSCH DM-RS MIMO
                % precoding and mapping
                dmrsSymbols = nrPDSCHDMRS(carrier,pdsch,OutputDataType=dt);
                dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);
                [dmrsAntSymbols,dmrsAntIndices] = nrPDSCHPrecode( ...
                    carrier,dmrsSymbols,dmrsIndices,wtx);
                slotGrid(dmrsAntIndices) = dmrsAntSymbols;
                refSlotGrid(dmrsAntIndices) = dmrsAntSymbols;

                % Perform implementation-specific PDSCH PT-RS MIMO
                % precoding and mapping
                ptrsSymbols = nrPDSCHPTRS(carrier,pdsch,OutputDataType=dt);
                ptrsIndices = nrPDSCHPTRSIndices(carrier,pdsch);
                [ptrsAntSymbols,ptrsAntIndices] = nrPDSCHPrecode( ...
                    carrier,ptrsSymbols,ptrsIndices,wtx);
                slotGrid(ptrsAntIndices) = ptrsAntSymbols;
                refSlotGrid(ptrsAntIndices) = ptrsAntSymbols;

                % Map to the grid containing multiple slots
                symIdx = (nSlotSymb*slotIdx)+1:(nSlotSymb*(slotIdx+1));
                pdschGrid(:,symIdx,:) = slotGrid;
                refGrid(:,symIdx,:) = refSlotGrid;
            end

            % Perform OFDM modulation
            carrier.NSlot = initialNSlot;
            [txWaveform,info] = nrOFDMModulate(carrier,pdschGrid);

            % Append the resource grids to output structure
            info.ResourceGrid = pdschGrid;
            info.ReferenceGrid = refGrid;
        end

        function [slotTimes,symLen] = getSlotTimes(nSymbSlot,sl,fs,nf,dt)
            % Get the slot time
            symLen = cumsum(sl); % Lengths of each OFDM symbol in a slot
            nSubFrames = 10;
            samples = [0 symLen(nSymbSlot:nSymbSlot:end-1)]'./fs;
            subframeTimes = (0:(nf*nSubFrames)-1)*1e-3;
            slotTimes = cast(reshape(samples+subframeTimes,[],1),dt);
        end

        function info = initializeDelayObjects(simParameters,waveformInfo)
            % Initialize the delay objects by calculating the slant
            % distance given the inputs, simParameters and waveformInfo.

            % Compute path loss based on the elevation angle and satellite
            % altitude
            c = physconst("lightspeed");
            SU = slantRangeCircularOrbit(simParameters.ElevationAngle, ...
                simParameters.SatelliteAltitude,simParameters.MobileAltitude);
            lambda = c/simParameters.CarrierFrequency;
            pathLoss = fspl(SU,lambda)*double(simParameters.IncludeFreeSpacePathLoss); % in dB

            % Get the slot time based on subcarrier spacing
            slotTimes = HelperNRNTNThroughput.getSlotTimes( ...
                simParameters.Carrier.SymbolsPerSlot,waveformInfo.SymbolLengths, ...
                waveformInfo.SampleRate,simParameters.NFrames,simParameters.DataType);

            % Maximum variable propagation delay
            maxVarPropDelay = 0;

            % Compute static delay in seconds and samples
            delayInSeconds = cast(SU./c,simParameters.DataType);
            delayInSamples = delayInSeconds.*waveformInfo.SampleRate;
            integDelaySamples = floor(delayInSamples);
            fracDelaySamples = (delayInSamples - integDelaySamples);
            numVariableFracDelaySamples = repmat(fracDelaySamples,1, ...
                (simParameters.Carrier.SlotsPerFrame*simParameters.NFrames) + 1);
            numVariableIntegSamples = 0;
            pathLoss = repmat(pathLoss,1,numel(slotTimes));
            % Initialize configuration objects for delay modeling
            if simParameters.DelayModel == "None"
                staticDelay = dsp.Delay(Length=0);
                variableIntegerDelay = 0;
                variableFractionalDelay = 0;
                delayInSeconds = 0;
            elseif simParameters.DelayModel == "Static"
                staticDelay = dsp.Delay(Length=integDelaySamples);
                variableIntegerDelay = 0;
                variableFractionalDelay = dsp.VariableFractionalDelay(...
                    InterpolationMethod="Farrow", ...
                    FarrowSmallDelayAction="Use off-centered kernel", ...
                    MaximumDelay=1);
            else
                % Model time-varying delay assuming satellite moves in a
                % circular orbit. The example applies delay for each slot
                % and assumes there is no significant change in delay for
                % each sample.

                % Calculate delay across the simulation time in steps of
                % slot time
                SU = slantRangeCircularOrbit(simParameters.ElevationAngle, ...
                    simParameters.SatelliteAltitude,simParameters.MobileAltitude,slotTimes);
                pathLoss = fspl(SU,lambda).*double(simParameters.IncludeFreeSpacePathLoss);
                delayInSeconds = SU./c;
                delayInSamples = delayInSeconds.*waveformInfo.SampleRate;

                % Compute dynamic range of delay and configure the delay
                % objects accordingly
                integDelaySamples = floor(delayInSamples);
                numStaticDelaySamples = min(integDelaySamples);
                remVariableDelaySamples = delayInSamples - numStaticDelaySamples;
                staticDelay = dsp.Delay(Length=numStaticDelaySamples);
                numVariableIntegSamples = floor(remVariableDelaySamples);
                numVariableIntegSamples(numVariableIntegSamples < 0) = 0;
                maxVarPropDelay = max(numVariableIntegSamples)+2;
                variableIntegerDelay = dsp.VariableIntegerDelay(...
                    MaximumDelay=maxVarPropDelay);
                numVariableFracDelaySamples = remVariableDelaySamples - numVariableIntegSamples;
                variableFractionalDelay = dsp.VariableFractionalDelay( ...
                    InterpolationMethod="Farrow", ...
                    FarrowSmallDelayAction="Use off-centered kernel", ...
                    MaximumDelay=1);
            end

            % Get output structure
            info = struct;
            info.StaticDelay = staticDelay;
            info.VariableIntegerDelay = variableIntegerDelay;
            info.VariableFractionalDelay = variableFractionalDelay;
            info.MaxVariablePropDelay = maxVarPropDelay;
            info.NumVariableIntegerDelaySamples = numVariableIntegSamples;
            info.NumVariableFractionalDelaySamples = numVariableFracDelaySamples;
            info.DelayInSeconds = delayInSeconds;
            info.PathLoss = pathLoss;
            info.SlantDistance = SU;
        end

        function [hpa,hpaDelay,paInputScaleFactor] = initializePA( ...
                paModel,hasMemory,paCharacteristics,coefficients)
            % Initialize the power amplifier function handle or System
            % object depending on the input configuration
            paInputScaleFactor = 0;                                 % in dB
            hpaDelay = 0;
            if hasMemory == 1
                hpa = rf.PAmemory;                                  % Requires RF Toolbox
                if isempty(coefficients)
                    hpa.CoefficientMatrix = ...
                        HelperNRNTNThroughput.getDefaultCoefficients;
                    paInputScaleFactor = -35;
                else
                    hpa.CoefficientMatrix = coefficients;
                end
                hpa.UnitDelay = 1;
                hpaDelay = size(hpa.CoefficientMatrix,1)-1;
            else
                if paModel == "Custom"
                    if isempty(paCharacteristics)
                        hpa = comm.MemorylessNonlinearity(Method="Lookup table", ...
                            Table=HelperNRNTNThroughput.getDefaultLookup);
                        paInputScaleFactor = -35;
                    else
                        hpa = comm.MemorylessNonlinearity(Method="Lookup table", ...
                            Table=paCharacteristics);
                    end
                elseif paModel == "2.1GHz GaAs"
                    hpa = @(in) HelperNRNTNThroughput.paMemorylessGaAs2Dot1GHz(in);
                elseif paModel == "2.1GHz GaN"
                    hpa = @(in) HelperNRNTNThroughput.paMemorylessGaN2Dot1GHz(in);
                elseif paModel == "28GHz CMOS"
                    hpa = @(in) HelperNRNTNThroughput.paMemorylessCMOS28GHz(in);
                else % "28GHz GaN"
                    hpa = @(in) HelperNRNTNThroughput.paMemorylessGaN28GHz(in);
                end
            end
        end
    end

end
