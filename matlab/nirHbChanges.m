function [dHbO2, dHbR] = nirHbChanges(dOD, epsHbO2, epsHbR, pathLengthCm, dpf)
%NIRHBCHANGES Two-wavelength modified Beer-Lambert inversion.
%   [DHBO2, DHBR] = NIRHBCHANGES(DOD, EPSHBO2, EPSHBR, PATHLENGTHCM, DPF)
%   solves the modified Beer-Lambert law for relative changes in
%   oxy- and deoxyhemoglobin concentration [mM] given optical-density
%   changes DOD at two (or more) wavelengths.
%
%   DOD          : n x nWavelengths matrix of Delta-OD values
%   EPSHBO2, EPSHBR : 1 x nWavelengths molar extinction coefficients [mM^-1 cm^-1]
%                   (see src/nir_oxygenation.py module docstring re:
%                   default coefficients being illustrative, not
%                   instrument-calibrated values)
%   PATHLENGTHCM : source-detector separation / tissue thickness [cm]
%   DPF          : differential pathlength factor
%
%   MATLAB port of src/nir_oxygenation.py:solve_hb_changes.

    if nargin < 4
        pathLengthCm = 1.0;
    end
    if nargin < 5
        dpf = 1.0;
    end

    E = [epsHbO2(:), epsHbR(:)] * pathLengthCm * dpf;  % nWavelengths x 2
    sol = E \ dOD';                                     % 2 x n
    dHbO2 = sol(1, :)';
    dHbR = sol(2, :)';
end
