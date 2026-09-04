%RUN_DEMO Small end-to-end MATLAB demo mirroring scripts/run_demo.py.
%   Generates a synthetic speckle image and recovers its contrast map,
%   then generates synthetic two-wavelength NIR data and recovers
%   hemoglobin concentration changes, reporting recovery error for both.

rng(0);

% --- LSCI ---
speckleImage = random('Exponential', 1, 64, 64);
K = lsciContrast(speckleImage, 7);
fprintf('Speckle contrast K: mean=%.3f, range=[%.3f, %.3f]\n', ...
    mean(K(:)), min(K(:)), max(K(:)));

% --- NIR / MBLL ---
epsHbO2 = [0.35, 1.05];
epsHbR  = [0.80, 0.70];

trueHbO2 = -0.02;
trueHbR  = 0.03;
E = [epsHbO2', epsHbR'];
dOD = (E * [trueHbO2; trueHbR])';

[hbo2Hat, hbrHat] = nirHbChanges(dOD, epsHbO2, epsHbR);

fprintf('True   dHbO2=%.4f mM, dHbR=%.4f mM\n', trueHbO2, trueHbR);
fprintf('Recov. dHbO2=%.4f mM, dHbR=%.4f mM\n', hbo2Hat, hbrHat);
