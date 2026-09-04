import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blood_absorption import mua_from_hemoglobin
from src.lsci_processing import blood_flow_index, contrast_to_correlation_time, spatial_speckle_contrast
from src.ml_pipeline import cross_validated_accuracy, extract_peak_features
from src.monte_carlo import simulate_diffuse_reflectance
from src.nir_oxygenation import relative_oxygenation_index, solve_hb_changes
from src.synthetic_cohort import generate_cohort


def test_spatial_speckle_contrast_higher_for_static_scene():
    rng = np.random.default_rng(0)
    static_speckle = rng.exponential(1.0, size=(40, 40))  # fully developed, uncorrelated
    flowing = np.full((40, 40), 1.0) + rng.normal(0, 0.01, size=(40, 40))  # near-uniform (fast flow)

    k_static = spatial_speckle_contrast(static_speckle, window=7).mean()
    k_flowing = spatial_speckle_contrast(flowing, window=7).mean()

    assert k_static > k_flowing


def test_contrast_to_correlation_time_monotonic_in_k():
    k_values = np.array([0.2, 0.5, 0.8])
    tau_c = contrast_to_correlation_time(k_values, exposure_time=5e-3)
    assert tau_c[0] < tau_c[1] < tau_c[2]


def test_blood_flow_index_is_inverse_of_tau_c():
    tau_c = np.array([0.001, 0.01, 0.1])
    bfi = blood_flow_index(tau_c)
    assert np.allclose(bfi, 1.0 / tau_c)


def test_solve_hb_changes_recovers_known_values():
    from src.nir_oxygenation import DEFAULT_EPS_HBO2, DEFAULT_EPS_HBR

    true_hbo2, true_hbr = -0.01, 0.02
    e_matrix = np.stack([DEFAULT_EPS_HBO2, DEFAULT_EPS_HBR], axis=1)
    d_od = e_matrix @ np.array([true_hbo2, true_hbr])

    hbo2_hat, hbr_hat = solve_hb_changes(d_od[None, :])

    assert np.isclose(hbo2_hat[0], true_hbo2, atol=1e-6)
    assert np.isclose(hbr_hat[0], true_hbr, atol=1e-6)


def test_relative_oxygenation_index_direction():
    # Rising HbO2 with falling HbR should give an oxygenation index > 0.5
    idx = relative_oxygenation_index(np.array([0.02]), np.array([-0.01]))
    assert idx[0] > 0.5


def test_mua_increases_with_more_hemoglobin():
    mua_low = mua_from_hemoglobin(hbo2_mM=0.0, hbr_mM=0.0)
    mua_high = mua_from_hemoglobin(hbo2_mM=0.01, hbr_mM=0.03)
    assert mua_high > mua_low


def test_reflectance_drops_when_absorption_rises_from_hemoglobin():
    mua_low = mua_from_hemoglobin(hbo2_mM=0.0, hbr_mM=0.0)
    mua_high = mua_from_hemoglobin(hbo2_mM=0.0, hbr_mM=0.04)
    rd_low = simulate_diffuse_reflectance(mua_low, musp=10.0, n_photons=8000, seed=0)
    rd_high = simulate_diffuse_reflectance(mua_high, musp=10.0, n_photons=8000, seed=0)
    assert rd_high < rd_low


def test_synthetic_cohort_shape_and_group_labels():
    cohort = generate_cohort(n_per_group=5, seed=0)
    assert cohort["bfi"].shape == (10, 5)
    assert set(cohort["group"]) == {"LPS", "control"}


def test_classifier_beats_chance_on_synthetic_cohort():
    cohort = generate_cohort(n_per_group=21, seed=3)
    x, y = extract_peak_features(cohort)
    result = cross_validated_accuracy(x, y, seed=3)
    assert result["accuracy"] > 0.5
