"""
blood_absorption.py

Converts hemoglobin concentrations to a bulk tissue absorption
coefficient mu_a, so that changes in blood volume / oxygenation (as
tracked by nir_oxygenation.py) can be fed into the Monte Carlo /
diffusion light-transport models as a physically motivated mu_a.

    mu_a(lambda) [cm^-1] = ln(10) * ( eps_HbO2(lambda) * [HbO2]
                                     + eps_Hb(lambda)   * [HbR] )   + mu_a_baseline

where concentrations are in mM and eps is in mM^-1 cm^-1 (base-10
convention, matching nir_oxygenation.DEFAULT_EPS_*). The ln(10) factor
converts the base-10 (optical-density) extinction convention to the
natural-log (Beer-Lambert exponential attenuation) convention used by
the transport models in this repository.
"""

from __future__ import annotations

import numpy as np

from .nir_oxygenation import DEFAULT_EPS_HBO2, DEFAULT_EPS_HBR

LN10 = np.log(10.0)


def mua_from_hemoglobin(
    hbo2_mM: float,
    hbr_mM: float,
    wavelength_index: int = 0,
    mua_baseline_per_cm: float = 0.02,
    eps_hbo2: np.ndarray = DEFAULT_EPS_HBO2,
    eps_hbr: np.ndarray = DEFAULT_EPS_HBR,
) -> float:
    """Bulk tissue mu_a [cm^-1] from hemoglobin concentrations at one wavelength.

    Parameters
    ----------
    hbo2_mM, hbr_mM : float
        Oxy- and deoxyhemoglobin concentrations [mM] (whole-tissue,
        already accounting for blood volume fraction).
    wavelength_index : int
        Index into ``eps_hbo2`` / ``eps_hbr`` selecting which of the
        default (or supplied) wavelengths to use.
    mua_baseline_per_cm : float
        Residual absorption from water, lipid, etc. not attributable to
        hemoglobin, added as a constant offset.
    """
    return (
        LN10
        * (eps_hbo2[wavelength_index] * hbo2_mM + eps_hbr[wavelength_index] * hbr_mM)
        + mua_baseline_per_cm
    )
