function [mask, w_y, w_x] = fda_create_smooth_window(grid_h, grid_w, win_h_px, win_w_px, window_type, taper_h, taper_w)
% FDA_CREATE_SMOOTH_WINDOW Generates a 2D smooth mixing mask for Fourier Domain Adaptation (FDA).
%
% Eliminates Gibbs ringing / boundary artifacts caused by sharp rectangular boxcar windows
% by providing a smooth continuous roll-off from 1.0 (Target domain) to 0.0 (Source domain).
%
% Inputs:
%   grid_h      - Total grid height / subcarriers (e.g. 132)
%   grid_w      - Total grid width / OFDM symbols (e.g. 14)
%   win_h_px    - Nominal window height along Delay (e.g. 13)
%   win_w_px    - Nominal window width along Doppler (e.g. 3 or 5)
%   window_type - Window type string:
%                   'tukey_outward' (Default) - Keeps core win_h x win_w flat at 1.0,
%                                              smooth cosine taper outside into adjacent pixels.
%                   'tukey_inward'            - Smooth cosine taper starts inside win_h x win_w,
%                                              reaches 0.0 at the boundary.
%                   'supergaussian'           - Flat-top elliptical super-Gaussian.
%                   'hard'                    - Legacy binary rectangular boxcar (0 or 1).
%   taper_h     - Number of transition elements along Delay/height (Default: 2 -> [0.75, 0.25])
%   taper_w     - Number of transition elements along Doppler/width (Default: 1 -> [0.5])
%
% Outputs:
%   mask - 2D matrix (grid_h x grid_w) with blending weights in [0, 1]
%   w_y  - 1D delay profile (grid_h x 1)
%   w_x  - 1D Doppler profile (1 x grid_w)

    if nargin < 5 || isempty(window_type)
        window_type = 'tukey_outward';
    end
    if nargin < 6 || isempty(taper_h)
        taper_h = 2; % Default: 2 transition pixels along delay (132 rows)
    end
    if nargin < 7 || isempty(taper_w)
        taper_w = 1; % Default: 1 transition pixel along Doppler (14 cols)
    end

    cy = floor(grid_h / 2) + 1; % 67 for 132
    cx = floor(grid_w / 2) + 1; % 8 for 14

    r_h = floor(win_h_px / 2);
    r_w = floor(win_w_px / 2);

    window_type = lower(string(window_type));

    switch window_type
        case {"tukey_outward", "tukey", "raised_cosine"}
            % -------------------------------------------------------------
            % Outward Tukey (Raised-Cosine):
            % Core (win_h x win_w) stays strictly 1.0 (100% Target).
            % Tapering rolls off into taper_h / taper_w elements outside.
            % Formula: w = 0.5 * (1 + cos(k * pi / (taper + 1)))
            % For taper_w = 1: k=1 -> 0.5, k=2 -> 0.0
            % For taper_h = 2: k=1 -> 0.75, k=2 -> 0.25, k=3 -> 0.0
            % -------------------------------------------------------------
            w_y = zeros(grid_h, 1);
            for y = 1:grid_h
                d = abs(y - cy);
                if d <= r_h
                    w_y(y) = 1.0;
                elseif d <= (r_h + taper_h)
                    k = d - r_h;
                    w_y(y) = 0.5 * (1 + cos(pi * k / (taper_h + 1)));
                end
            end

            w_x = zeros(1, grid_w);
            for x = 1:grid_w
                d = abs(x - cx);
                if d <= r_w
                    w_x(x) = 1.0;
                elseif d <= (r_w + taper_w)
                    k = d - r_w;
                    w_x(x) = 0.5 * (1 + cos(pi * k / (taper_w + 1)));
                end
            end

            mask = w_y * w_x;

        case "tukey_inward"
            % -------------------------------------------------------------
            % Inward Tukey:
            % Window reaches 0.0 at the win_h x win_w boundary.
            % Transition occurs inside the boundary.
            % -------------------------------------------------------------
            t_h = min(taper_h, max(1, r_h - 1));
            t_w = min(taper_w, max(1, r_w - 1));

            w_y = zeros(grid_h, 1);
            for y = 1:grid_h
                d = abs(y - cy);
                if d <= (r_h - t_h)
                    w_y(y) = 1.0;
                elseif d <= r_h
                    k = d - (r_h - t_h);
                    w_y(y) = 0.5 * (1 + cos(pi * k / (t_h + 1)));
                end
            end

            w_x = zeros(1, grid_w);
            for x = 1:grid_w
                d = abs(x - cx);
                if d <= (r_w - t_w)
                    w_x(x) = 1.0;
                elseif d <= r_w
                    k = d - (r_w - t_w);
                    w_x(x) = 0.5 * (1 + cos(pi * k / (t_w + 1)));
                end
            end

            mask = w_y * w_x;

        case {"supergaussian", "gaussian"}
            % -------------------------------------------------------------
            % Super-Gaussian (Flat-Top Elliptical Gaussian, p = 2):
            % exp(- (norm_dist)^p)
            % Flat inner plateau, infinitely differentiable C^inf roll-off.
            % -------------------------------------------------------------
            [X, Y] = meshgrid(1:grid_w, 1:grid_h);
            norm_dist = ((Y - cy) ./ max(1, r_h)).^2 + ((X - cx) ./ max(1, r_w)).^2;
            p_order = 2;
            mask = exp(- (norm_dist .^ p_order));
            mask(mask < 0.01) = 0; % Cleanly truncate negligible tails

            w_y = mask(:, cx);
            w_x = mask(cy, :);

        case "hard"
            % -------------------------------------------------------------
            % Legacy Binary Rectangular Boxcar:
            % Strictly 1 inside, 0 outside.
            % -------------------------------------------------------------
            mask = zeros(grid_h, grid_w);
            y_range = max(1, cy - r_h) : min(grid_h, cy + r_h);
            x_range = max(1, cx - r_w) : min(grid_w, cx + r_w);
            mask(y_range, x_range) = 1;

            w_y = zeros(grid_h, 1);
            w_y(y_range) = 1.0;
            w_x = zeros(1, grid_w);
            w_x(x_range) = 1.0;

        otherwise
            error('Unsupported window_type: "%s". Use "tukey_outward", "tukey_inward", "supergaussian", or "hard".', window_type);
    end
end
