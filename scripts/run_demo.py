#!/usr/bin/env python3
"""
run_demo.py

End-to-end demonstration of the repository's multimodal NIR + LSCI
analysis pipeline on synthetic data:

  1. Generate a synthetic speckle image with two flow regions and
     recover a blood-flow-index (BFI) map from it (LSCI).
  2. Generate synthetic two-wavelength NIR intensity time courses and
     recover relative HbO2 / HbR changes and an oxygenation index (MBLL).
  3. Use the Monte Carlo simulator to show how the resulting change in
     bulk absorption (from the recovered hemoglobin changes) is expected
     to modulate NIR diffuse reflectance -- a forward-model sanity check
     linking the two modalities.
  4. Generate a synthetic 42-animal (21 LPS / 21 control) longitudinal
     cohort, extract peak-response features, and cross-validate a simple
     classifier distinguishing LPS from control animals.
  5. Save a summary figure to figures/demo_summary.png.

Run with:  python scripts/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.blood_absorption import mua_from_hemoglobin
from src.lsci_processing import blood_flow_index, contrast_to_correlation_time, spatial_speckle_contrast
from src.ml_pipeline import cross_validated_accuracy, extract_peak_features
from src.monte_carlo import simulate_diffuse_reflectance
from src.nir_oxygenation import delta_od, relative_oxygenation_index, solve_hb_changes
from src.synthetic_cohort import generate_cohort


def make_synthetic_speckle_image(size: int = 64, seed: int = 0) -> np.ndarray:
    """Two-region synthetic speckle image: a blurred (high-flow, low
    contrast) circular region on a sharp (low-flow, high-contrast)
    background, purely for demonstrating the LSCI pipeline mechanics.
    """
    rng = np.random.default_rng(seed)
    raw = rng.exponential(1.0, size=(size, size))  # speckle-like statistics

    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(raw, sigma=2.5)

    y, x = np.mgrid[0:size, 0:size]
    mask = (x - size / 2) ** 2 + (y - size / 2) ** 2 < (size / 4) ** 2

    image = np.where(mask, blurred, raw)
    return image / image.mean()


def main() -> None:
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("[1/5] LSCI: synthetic speckle image -> blood-flow-index map...")
    speckle_image = make_synthetic_speckle_image()
    k_map = spatial_speckle_contrast(speckle_image, window=7)
    exposure_time = 5e-3  # 5 ms, typical LSCI camera exposure
    tau_c_map = contrast_to_correlation_time(k_map, exposure_time=exposure_time)
    bfi_map = blood_flow_index(tau_c_map)
    print(f"      K range: [{np.nanmin(k_map):.3f}, {np.nanmax(k_map):.3f}]  "
          f"BFI range: [{np.nanmin(bfi_map):.1f}, {np.nanmax(bfi_map):.1f}] (a.u.)")

    print("[2/5] NIR: synthetic two-wavelength intensities -> Hb changes...")
    rng = np.random.default_rng(1)
    n_t = 40
    true_hbo2 = -0.02 * np.linspace(0, 1, n_t) + rng.normal(0, 0.002, n_t)  # falling
    true_hbr = 0.03 * np.linspace(0, 1, n_t) + rng.normal(0, 0.002, n_t)    # rising

    from src.nir_oxygenation import DEFAULT_EPS_HBO2, DEFAULT_EPS_HBR

    e_matrix = np.stack([DEFAULT_EPS_HBO2, DEFAULT_EPS_HBR], axis=1)  # (2, 2)
    simulated_dOD = (e_matrix @ np.stack([true_hbo2, true_hbr], axis=0)).T  # (n_t, 2)
    simulated_dOD += rng.normal(0, 2e-4, simulated_dOD.shape)

    hbo2_hat, hbr_hat = solve_hb_changes(simulated_dOD)
    oxy_index = relative_oxygenation_index(hbo2_hat, hbr_hat)
    print(f"      recovered Delta_HbO2 RMSE: {np.sqrt(np.mean((hbo2_hat - true_hbo2) ** 2)):.5f} mM")
    print(f"      recovered Delta_HbR  RMSE: {np.sqrt(np.mean((hbr_hat - true_hbr) ** 2)):.5f} mM")

    print("[3/5] Monte Carlo: reflectance sensitivity to hemoglobin-driven mu_a...")
    musp_tissue = 10.0  # cm^-1, typical soft-tissue NIR reduced scattering
    mua_early = mua_from_hemoglobin(hbo2_mM=0.0, hbr_mM=0.0, wavelength_index=0)
    mua_late = mua_from_hemoglobin(hbo2_mM=hbo2_hat[-1], hbr_mM=hbr_hat[-1], wavelength_index=0)
    rd_early = simulate_diffuse_reflectance(mua_early, musp_tissue, n_photons=15_000, seed=2)
    rd_late = simulate_diffuse_reflectance(mua_late, musp_tissue, n_photons=15_000, seed=2)
    print(f"      mu_a: {mua_early:.4f} -> {mua_late:.4f} cm^-1   "
          f"R_d (Monte Carlo): {rd_early:.4f} -> {rd_late:.4f}")

    print("[4/5] Synthetic 42-animal cohort: LPS vs. control classification...")
    cohort = generate_cohort(n_per_group=21, seed=3)
    x, y = extract_peak_features(cohort)
    cv_result = cross_validated_accuracy(x, y, seed=3)
    print(f"      cross-validated classification accuracy: {cv_result['accuracy']:.2f} "
          f"(n={len(y)} animals, chance level = 0.50)")

    print("[5/5] Saving summary figure...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    im0 = axes[0, 0].imshow(speckle_image, cmap="gray")
    axes[0, 0].set_title("Synthetic raw speckle image")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(k_map, cmap="magma")
    axes[0, 1].set_title("Speckle contrast K")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    vmin, vmax = np.nanpercentile(bfi_map, [2, 98])
    im2 = axes[0, 2].imshow(bfi_map, cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Recovered blood-flow index (a.u.)\n(color range clipped to 2nd-98th pct.)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    t_axis = np.arange(n_t)
    axes[1, 0].plot(t_axis, true_hbo2, "b-", label="true dHbO2")
    axes[1, 0].plot(t_axis, hbo2_hat, "b--", label="recovered dHbO2")
    axes[1, 0].plot(t_axis, true_hbr, "r-", label="true dHbR")
    axes[1, 0].plot(t_axis, hbr_hat, "r--", label="recovered dHbR")
    axes[1, 0].set_title("MBLL hemoglobin recovery")
    axes[1, 0].set_xlabel("sample")
    axes[1, 0].set_ylabel("Delta concentration [mM]")
    axes[1, 0].legend(fontsize=7)

    for grp, color in (("LPS", "tab:red"), ("control", "tab:blue")):
        mask = cohort["group"] == grp
        mean_bfi = cohort["bfi"][mask].mean(axis=0)
        axes[1, 1].plot(cohort["timepoints"], mean_bfi, "-o", color=color, label=grp)
    axes[1, 1].set_title("Cohort mean BFI vs. time (synthetic)")
    axes[1, 1].set_xlabel("day")
    axes[1, 1].set_ylabel("BFI (a.u.)")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].scatter(x[y == 1, 0], x[y == 1, 1], c="tab:red", label="LPS")
    axes[1, 2].scatter(x[y == 0, 0], x[y == 0, 1], c="tab:blue", label="control")
    axes[1, 2].set_title(f"Peak features (CV accuracy={cv_result['accuracy']:.2f})")
    axes[1, 2].set_xlabel("peak BFI")
    axes[1, 2].set_ylabel("trough oxygenation index")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Multimodal NIR + LSCI periodontitis-model demo pipeline (synthetic data)", fontsize=12)
    fig.tight_layout()
    out_path = figures_dir / "demo_summary.png"
    fig.savefig(out_path, dpi=150)
    print(f"      saved to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
