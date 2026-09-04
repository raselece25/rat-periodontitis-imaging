"""
lsci_processing.py

Laser Speckle Contrast Imaging (LSCI) processing: converts raw speckle
images into a relative blood-flow index (BFI) map.

Pipeline:
  1. Spatial speckle contrast:  K(x, y) = std(I) / mean(I) over a local window.
  2. Single-exposure speckle-to-correlation-time inversion (Bandyopadhyay
     et al. model):
         K^2(T) = (beta / 2) * [ (tau_c / T) - (tau_c / T)^2 * (1 - exp(-2T / tau_c)) / ... ]
     A commonly used simplified closed form (Briers 2001-2013 reviews) is:
         K^2(T) = beta * [ (tau_c / 2T)(1 - exp(-2T / tau_c)) ]
     which we invert numerically for tau_c given a measured K and known
     camera exposure time T.
  3. Blood flow index BFI = 1 / tau_c (arbitrary units; only relative
     changes are meaningful without independent flow calibration).

References:
    D. Briers et al., "Laser speckle contrast imaging: theoretical and
    practical limitations," J. Biomed. Opt. 18(6), 066018 (2013).
    D. A. Boas, A. K. Dunn, "Laser speckle contrast imaging in biomedical
    optics," J. Biomed. Opt. 15(1), 011109 (2010).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import uniform_filter


def spatial_speckle_contrast(image: np.ndarray, window: int = 7) -> np.ndarray:
    """Compute local spatial speckle contrast K = std / mean over a window.

    Parameters
    ----------
    image : np.ndarray, 2-D
        Raw speckle image (single exposure).
    window : int
        Side length (pixels) of the square sliding window (odd number
        recommended).

    Returns
    -------
    k : np.ndarray
        Speckle contrast map, same shape as ``image``.
    """
    image = np.asarray(image, dtype=float)
    mean = uniform_filter(image, size=window)
    mean_sq = uniform_filter(image**2, size=window)
    var = np.clip(mean_sq - mean**2, 0.0, None)
    std = np.sqrt(var)
    return std / np.clip(mean, 1e-9, None)


def _contrast_model(tau_c: float, exposure_time: float, beta: float) -> float:
    """K^2(T) predicted by the single-exposure speckle model."""
    x = 2.0 * exposure_time / tau_c
    return beta * (tau_c / (2.0 * exposure_time)) * (1.0 - np.exp(-x))


def contrast_to_correlation_time(
    k: np.ndarray,
    exposure_time: float,
    beta: float = 1.0,
    tau_c_bounds: tuple[float, float] = (1e-6, 1.0),
) -> np.ndarray:
    """Invert measured speckle contrast K to correlation time tau_c per pixel.

    Solves K_measured^2 = model(tau_c) via bisection at every pixel. Pixels
    where no root is bracketed within ``tau_c_bounds`` are returned as NaN.

    Parameters
    ----------
    k : np.ndarray
        Measured speckle contrast (any shape).
    exposure_time : float [s]
        Camera exposure time used to capture the speckle image(s).
    beta : float
        Instrument/speckle-averaging factor (<=1), depends on optics and
        pixel-to-speckle size ratio; commonly calibrated against a
        static (zero-flow) scatterer.
    tau_c_bounds : (float, float)
        Search bounds for the correlation time [s].

    Returns
    -------
    tau_c : np.ndarray
        Estimated correlation time per pixel, same shape as ``k``.
    """
    k_flat = np.asarray(k, dtype=float).ravel()
    tau_c_flat = np.full(k_flat.shape, np.nan)

    lo, hi = tau_c_bounds
    for i, k_val in enumerate(k_flat):
        target_sq = min(max(k_val, 0.0), beta) ** 2  # K^2 cannot exceed beta

        def f(tau_c: float, target_sq: float = target_sq) -> float:
            return _contrast_model(tau_c, exposure_time, beta) - target_sq

        try:
            tau_c_flat[i] = brentq(f, lo, hi)
        except ValueError:
            # No sign change in [lo, hi]: K is at (or numerically past, due
            # to finite-window sampling noise) the model's saturation
            # value. Saturate tau_c at whichever bound the data is
            # closest to, rather than returning NaN.
            tau_c_flat[i] = hi if f(hi) < 0 else lo

    return tau_c_flat.reshape(np.asarray(k).shape)


def blood_flow_index(tau_c: np.ndarray) -> np.ndarray:
    """Relative blood flow index, BFI = 1 / tau_c (arbitrary units)."""
    return 1.0 / np.clip(tau_c, 1e-12, None)
