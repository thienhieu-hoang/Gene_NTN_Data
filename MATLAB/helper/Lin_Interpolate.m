function varargout = Lin_Interpolate(Y_noise, pilot_Indices, pilot_Symbols)
% Perform LS and then Linear Interpolation

% Perform linear interpolation of the grid and input the result to the
% neural network This helper function extracts the DM-RS symbols from
% dmrsIndices locations in the received grid rxGrid and performs linear
% interpolation on the extracted pilots.
    

    % work with CSI-RS also
    nzp_ind = find(pilot_Symbols~=0);  % indices of NZP CSI-RS index
    pilot_Indices= pilot_Indices(nzp_ind);
    pilot_Symbols = pilot_Symbols(nzp_ind);

    % Obtain pilot symbol estimates
    dmrsRx = Y_noise(pilot_Indices);
    dmrsEsts = dmrsRx ./ (pilot_Symbols);

    % Create empty grids to fill after linear interpolation
    [H_equalized, rxDMRSGrid, H_linear] = deal(zeros(size(Y_noise)));
    rxDMRSGrid(pilot_Indices)  = pilot_Symbols;
    H_equalized(pilot_Indices) = dmrsEsts;
    
    % Find the row and column coordinates for a given DMRS configuration
    [rows,cols] = find(rxDMRSGrid ~= 0);
    dmrsSubs = [rows,cols,ones(size(cols))];

    dmrsSymbol = unique(dmrsSubs(:,2)); % symbol containing csi-rs / dmrs
    if isscalar(dmrsSymbol)
        % Only 1 symbol containing pilots -> just simply duplicate
        col_ind = dmrsSubs(1,2);
        query_idx = 1:size(H_linear,1);
        H_linear(:,col_ind) = interp1(dmrsSubs(:,1), dmrsEsts, query_idx', 'linear', 'extrap');
        for idx = 1: size(H_linear,2)
            if idx ~= col_ind
                H_linear(:,idx) = H_linear(:, col_ind);
            end
        end
    elseif numel(dmrsSymbol)>1
        % Perform linear interpolation
        dmrsEsts = double(dmrsEsts);
        [l_hest,k_hest] = meshgrid(1:size(H_linear,2),1:size(H_linear,1));
        f = scatteredInterpolant(dmrsSubs(:,2),dmrsSubs(:,1),dmrsEsts);
        H_linear = f(l_hest,k_hest);
    end
    
    if nargout >= 2
        varargout{1} = H_equalized;
        varargout{2} = H_linear;
    end
end