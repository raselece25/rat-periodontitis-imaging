"""
synthetic_cohort.py

Generates a synthetic longitudinal multimodal-imaging dataset for an
LPS-induced periodontitis model vs. sham/control animals.

IMPORTANT: every number produced by this module is randomly generated
for demonstration purposes. It does NOT contain, reproduce, or derive
from any unpublished data collected under IACUC protocol
#IACUC2025-00060 at the BOIL Lab, Stony Brook University. The intent is
only to give the LSCI/NIR/ML processing code in this repository
realistic-looking, reproducible synthetic data to run end-to-end on.
The qualitative direction of the synthetic trends (LPS animals show
increased local blood flow and reduced tissue oxygenation as
inflammation develops, consistent with the general periodontal/soft-
tissue inflammation literature) is illustrative, not a reported result.
"""

from __future__ import annotations

import numpy as np


def generate_cohort(
    n_per_group: int = 21,
    timepoints_days: tuple[float, ...] = (0, 1, 3, 7, 14),
    seed: int | None = 0,
) -> dict:
    """Create a synthetic two-group (LPS vs control) longitudinal dataset.

    Returns a dict with keys:
        'group'      : array of 'LPS' / 'control', shape (2*n_per_group,)
        'timepoints' : the timepoints_days tuple
        'bfi'        : relative blood-flow index, shape (2*n_per_group, n_t)
        'oxy_index'  : relative oxygenation index, shape (2*n_per_group, n_t)
    """
    rng = np.random.default_rng(seed)
    n_t = len(timepoints_days)
    t = np.asarray(timepoints_days, dtype=float)

    n_total = 2 * n_per_group
    group = np.array(["LPS"] * n_per_group + ["control"] * n_per_group)

    # Baseline (day 0) values are drawn identically for both groups.
    bfi_baseline = rng.normal(1.0, 0.1, size=n_total)
    oxy_baseline = rng.normal(0.65, 0.05, size=n_total)

    # LPS animals: blood flow rises and peaks around day 3-7, then partially
    # resolves; oxygenation index falls over the same window. Control
    # animals fluctuate around baseline with measurement noise only.
    bfi = np.zeros((n_total, n_t))
    oxy = np.zeros((n_total, n_t))

    for i in range(n_total):
        is_lps = group[i] == "LPS"
        peak_day = rng.normal(5.0, 1.0)
        amplitude_bfi = rng.normal(0.9, 0.15) if is_lps else 0.0
        amplitude_oxy = rng.normal(0.18, 0.05) if is_lps else 0.0

        # Log-normal-ish temporal envelope peaking near peak_day, decaying by day 14.
        envelope = np.exp(-0.5 * ((t - peak_day) / 3.5) ** 2)
        envelope[t == 0] = 0.0  # no injury effect yet at baseline

        bfi[i] = bfi_baseline[i] * (1.0 + amplitude_bfi * envelope) + rng.normal(0, 0.04, n_t)
        oxy[i] = oxy_baseline[i] * (1.0 - amplitude_oxy * envelope) + rng.normal(0, 0.02, n_t)

    return {
        "group": group,
        "timepoints": t,
        "bfi": bfi,
        "oxy_index": oxy,
    }
