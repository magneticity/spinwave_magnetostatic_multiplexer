function [maps, fpos] = mag_tfft_select(M, comp, dt, varargin)

%MAG_TFFT_SELECT  Extract amplitude/power maps at specified frequencies
%   [maps, fpos] = MAG_TFFT_SELECT(M, comp, dt, 'Fsel', [f1 f2 ...], ...)
%
% Name-Value:
%   'DimOrder' : 'xyz' (default) | 'xzy' | 'yxz' | 'yzx' | 'zxy' | 'zyx'
%   'Detrend'  : true (default) | false
%   'Window'   : 'hann' (default) | 'hamming' | 'none'
%   'Stat'     : 'amp' (default) | 'power' | 'complex'
%
    p = inputParser;
    addParameter(p, 'DimOrder', 'xyz');
    addParameter(p, 'Detrend', true);
    addParameter(p, 'Window', 'hann');
    addParameter(p, 'Fsel', [], @(v)isnumeric(v) && isvector(v) && ~isempty(v));
    addParameter(p, 'Stat', 'amp');
    parse(p, varargin{:});
    o = p.Results;

    sz = size(M);  Nx_ = sz(2); Ny_ = sz(3); Nz_ = sz(4); Nt = sz(5);
    Fs = 1/dt;

    % --- component index (robust, case-insensitive) ---
    if ischar(comp) || isstring(comp)
        c = lower(char(comp));
        if     strcmp(c,'x'), ci = 1;
        elseif strcmp(c,'y'), ci = 2;
        elseif strcmp(c,'z'), ci = 3;
        else
            error('comp must be ''x'',''y'',''z'' or 1/2/3.');
        end
    elseif isnumeric(comp) && isscalar(comp) && any(comp == [1 2 3])
        ci = comp;
    else
        error('comp must be ''x'',''y'',''z'' or 1/2/3.');
    end

    % extract and reorder to [Nx,Ny,Nz,Nt]
    Xraw = reshape(M(ci,:,:,:,:), [Nx_, Ny_, Nz_, Nt]);
    switch lower(o.DimOrder)
        case 'xyz', perm = [1 2 3 4];
        case 'xzy', perm = [1 3 2 4];
        case 'yxz', perm = [2 1 3 4];
        case 'yzx', perm = [2 3 1 4];
        case 'zxy', perm = [3 1 2 4];
        case 'zyx', perm = [3 2 1 4];
        otherwise, error('Bad DimOrder');
    end
    X = permute(Xraw, perm);                 % [Nx,Ny,Nz,Nt]
    [Nx,Ny,Nz,~] = size(X);
    Nvox = Nx*Ny*Nz;

    % reshape to [Nt, Nvox] for fast FFT along dim 1
    X = reshape(permute(X,[4 1 2 3]), Nt, Nvox);

    % detrend & window
    if o.Detrend
        X = X - mean(X,1,'omitnan');
    end
    switch lower(o.Window)
        case 'hann',    w = hann(Nt,'periodic');
        case 'hamming', w = hamming(Nt,'periodic');
        otherwise,      w = ones(Nt,1);
    end
    X = X .* w;   % implicit expansion

    % FFT (no shift). Frequency grid: k=0..Nt-1 -> f = k*Fs/Nt (wraps at Fs/2).
    F = fft(X, [], 1);

    % map requested positive frequencies to bins (nearest)
    fbin = (0:Nt-1)*(Fs/Nt);
    k = round(o.Fsel/(Fs/Nt));        % nearest bin index (0-based)
    k = max(0, min(Nt-1, k));         % clamp
    fpos = k * (Fs/Nt);               % actual bin freqs

    % gather rows and reshape back to 3-D maps
    maps = cell(numel(k),1);
    for i = 1:numel(k)
        row = F(k(i)+1, :);           % +1 for MATLAB 1-based indexing
        vol = reshape(row, [Nx,Ny,Nz]);
        switch lower(o.Stat)
            case 'complex', maps{i} = vol;
            case 'amp',     maps{i} = abs(vol);
            case 'power',   maps{i} = abs(vol).^2;
            otherwise, error('Stat must be complex|amp|power');
        end
    end
end

