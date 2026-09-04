"""
nir_oxygenation.py

Two-wavelength continuous-wave NIR spectroscopy: converts relative light
intensity changes at two wavelengths into relative changes in
oxy-hemoglobin (HbO2) and deoxy-hemoglobin (HbR) concentration via the
modified Beer-Lambert law (MBLL), and derives a relative tissue
oxygenation index from them.

    Delta_OD(lambda) = -log10( I(lambda, t) / I(lambda, t0) )
                      = ( eps_HbO2(lambda) * Delta_HbO2
                        + eps_Hb(lambda)   * Delta_HbR ) * L * DPF(lambda)

With two wavelengths this is a 2x2 linear system that can be solved
directly for [Delta_HbO2, Delta_HbR] (in concentration * pathlength
units unless L * DPF is supplied, in which case true concentration
changes are returned).

IMPORTANT -- extinction coefficients: the default coefficients below are
approximate, illustrative values in the range reported in the NIRS
literature (deoxyhemoglobin absorbs more strongly than oxyhemoglobin
below the ~800 nm isosbestic point, and less strongly above it). They
are NOT a substitute for your instrument's calibrated coefficients. For
rigorous work, use a tabulated compilation such as:
    S. Prahl, "Optical Absorption of Hemoglobin," https://omlc.org/spectra/hemoglobin/
and your system's actual source wavelengths and differential pathlength
factor (DPF).

Reference: A. Duncan et al., "Optical pathlength measurements on adult
head, calf and forearm and the head of the newborn infant using phase
resolved optical spectroscopy," Phys. Med. Biol. 40(2), 295-304 (1995).
"""

from __future__ import annotations

import numpy as np

# Approximate, illustrative molar extinction coefficients [mM^-1 cm^-1].
# See module docstring -- substitute calibrated, wavelength-exact values
# for any quantitative use.
DEFAULT_WAVELENGTHS_NM = (740.0, 850.0)
DEFAULT_EPS_HBO2 = np.array([0.35, 1.05])  # mM^-1 cm^-1, at the two wavelengths above
DEFAULT_EPS_HBR = np.array([0.80, 0.70])   # mM^-1 cm^-1


def delta_od(intensity: np.ndarray, baseline_intensity: np.ndarray) -> np.ndarray:
    """Change in optical density relative to a baseline measurement."""
    intensity = np.asarray(intensity, dtype=float)
    baseline_intensity = np.asarray(baseline_intensity, dtype=float)
    ratio = np.clip(intensity, 1e-12, None) / np.clip(baseline_intensity, 1e-12, None)
    return -np.log10(ratio)


def solve_hb_changes(
    delta_od_values: np.ndarray,
    eps_hbo2: np.ndarray = DEFAULT_EPS_HBO2,
    eps_hbr: np.ndarray = DEFAULT_EPS_HBR,
    path_length_cm: float = 1.0,
    dpf: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the modified Beer-Lambert law for two (or more) wavelengths.

    Parameters
    ----------
    delta_od_values : array_like, shape (..., n_wavelengths)
        Change in optical density at each wavelength (last axis).
    eps_hbo2, eps_hbr : array_like, shape (n_wavelengths,)
        Molar extinction coefficients [mM^-1 cm^-1] at the same
        wavelengths, in the same order.
    path_length_cm : float
        Physical source-detector separation or tissue thickness [cm].
    dpf : float
        Differential pathlength factor accounting for the increased
        optical path length due to scattering (actual path = L * DPF).

    Returns
    -------
    delta_hbo2, delta_hbr : np.ndarray
        Estimated concentration changes [mM], broadcasting over any
        leading dimensions of ``delta_od_values``.
    """
    delta_od_values = np.asarray(delta_od_values, dtype=float)
    n_wl = delta_od_values.shape[-1]
    if n_wl < 2:
        raise ValueError("Need at least two wavelengths to solve for HbO2 and HbR")

    # Build the (n_wl, 2) extinction-coefficient design matrix E, solve
    # Delta_OD = E @ [Delta_HbO2, Delta_HbR] * L * DPF via least squares
    # (exact solution when n_wl == 2, least-squares fit if n_wl > 2).
    e_matrix = np.stack([eps_hbo2[:n_wl], eps_hbr[:n_wl]], axis=1)  # (n_wl, 2)
    scale = path_length_cm * dpf

    flat_shape = delta_od_values.shape[:-1]
    delta_od_flat = delta_od_values.reshape(-1, n_wl)

    solution, *_ = np.linalg.lstsq(e_matrix * scale, delta_od_flat.T, rcond=None)
    delta_hbo2 = solution[0].reshape(flat_shape)
    delta_hbr = solution[1].reshape(flat_shape)
    return delta_hbo2, delta_hbr


def relative_oxygenation_index(delta_hbo2: np.ndarray, delta_hbr: np.ndarray) -> np.ndarray:
    """Relative oxygenation index: Delta_HbO2 / (Delta_HbO2 + Delta_HbT).

    This tracks the *direction and relative magnitude* of oxygenation
    change from an unspecified baseline; it is not an absolute StO2
    percentage unless baseline absolute concentrations are also known.
    """
    delta_hbt = delta_hbo2 + delta_hbr
    return delta_hbo2 / np.where(np.abs(delta_hbt) < 1e-9, np.nan, delta_hbt)
