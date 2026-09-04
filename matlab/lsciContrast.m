function K = lsciContrast(image, windowSize)
%LSCICONTRAST Local spatial speckle contrast K = std / mean.
%   K = LSCICONTRAST(IMAGE, WINDOWSIZE) computes the local spatial
%   speckle contrast of a raw laser-speckle image using a sliding
%   square window of side WINDOWSIZE (default 7).
%
%   Reference: D. Briers et al., "Laser speckle contrast imaging:
%   theoretical and practical limitations," J. Biomed. Opt. 18(6),
%   066018 (2013).
%
%   MATLAB port of src/lsci_processing.py:spatial_speckle_contrast.

    if nargin < 2
        windowSize = 7;
    end

    image = double(image);
    kernel = ones(windowSize) / (windowSize^2);

    localMean = conv2(image, kernel, 'same');
    localMeanSq = conv2(image.^2, kernel, 'same');
    localVar = max(localMeanSq - localMean.^2, 0);

    K = sqrt(localVar) ./ max(localMean, 1e-9);
end
