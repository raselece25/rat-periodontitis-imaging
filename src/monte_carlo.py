"""
monte_carlo.py

A compact, vectorized Monte Carlo photon-transport simulator for a
semi-infinite, homogeneous turbid medium. In this repository it is used
to illustrate how bulk absorption changes (e.g. increased blood volume
fraction / decreased oxygenation during inflammation) translate into
NIR diffuse reflectance changes -- the same underlying light-transport
physics that the two-wavelength modified Beer-Lambert processing in
``nir_oxygenation.py`` relies on, viewed from the forward-model side.

This follows the standard photon-packet weighting scheme described in:

    L. Wang, S. L. Jacques, L. Zheng, "MCML -- Monte Carlo modeling of
    light transport in multi-layered tissues," Comput. Methods Programs
    Biomed. 47(2), 131-146 (1995).

It is a teaching-scale implementation (single layer, total diffuse
reflectance only, no spatial-frequency modulation) -- not the full
multi-layer / laparoscopic Monte Carlo pipeline used in the underlying
laboratory work. It exists here to demonstrate and sanity-check the
Monte Carlo <-> diffusion-approximation relationship at zero spatial
frequency (mu_eff(fx=0) = sqrt(3 * mua * mu_tr)).
"""

from __future__ import annotations

import numpy as np


def simulate_diffuse_reflectance(
    mua: float,
    musp: float,
    g: float = 0.0,
    n_medium: float = 1.4,
    n_ambient: float = 1.0,
    n_photons: int = 20_000,
    weight_threshold: float = 1e-4,
    roulette_m: int = 10,
    max_steps: int = 8000,
    seed: int | None = 0,
) -> float:
    """Estimate total diffuse reflectance via vectorized Monte Carlo.

    By default this simulates with isotropic scattering (g = 0) using the
    *reduced* scattering coefficient directly as the scattering
    coefficient (mus = musp). This "similarity relation" substitution
    (mus, g) -> (musp, 0) is standard practice for diffuse-regime
    validation runs: the diffusion approximation this repo compares
    against likewise only depends on musp, not on mus and g separately,
    so simulating isotropic scattering with mus = musp gives an
    equivalent diffuse reflectance while requiring far fewer scattering
    events per photon than a highly forward-peaked (e.g. g = 0.9) tissue
    simulation would, keeping this teaching-scale demo fast. Pass a
    nonzero g explicitly if you want a literal (mus, g) simulation
    instead (musp = mus * (1 - g) is then recovered internally).

    Parameters
    ----------
    mua : float [cm^-1]
        Absorption coefficient.
    musp : float [cm^-1]
        Reduced scattering coefficient.
    g : float
        Scattering anisotropy (Henyey-Greenstein). Default 0 (isotropic),
        see note above on the similarity relation.
    n_medium, n_ambient : float
        Refractive indices of tissue and the surrounding medium (air).
    n_photons : int
        Number of photon packets to launch.
    weight_threshold, roulette_m : float, int
        Russian-roulette termination parameters.
    max_steps : int
        Hard cap on scattering events per photon (safety limit).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    rd : float
        Estimated total diffuse reflectance (fraction of launched weight
        that exits back through the top surface), i.e. R_d at fx = 0.
    """
    rng = np.random.default_rng(seed)
    mus = musp / (1.0 - g)  # recover the (unreduced) scattering coefficient
    mu_t = mua + mus

    # Specular (Fresnel) reflectance at normal incidence removes a small
    # fraction of launched weight before it enters the medium.
    r_sp = ((n_medium - n_ambient) / (n_medium + n_ambient)) ** 2

    n = n_photons
    weight = np.full(n, 1.0 - r_sp, dtype=float)
    pos = np.zeros((n, 3), dtype=float)  # x, y, z (z=0 at surface, +z into tissue)
    direction = np.zeros((n, 3), dtype=float)
    direction[:, 2] = 1.0  # launched straight down

    alive = np.ones(n, dtype=bool)
    reflected_weight = 0.0

    for _ in range(max_steps):
        if not np.any(alive):
            break
        idx = np.where(alive)[0]
        m = idx.size

        # Step size from the Beer-Lambert step-length distribution.
        rnd = rng.random(m)
        rnd = np.clip(rnd, 1e-12, 1.0)
        step = -np.log(rnd) / mu_t

        pos[idx] += direction[idx] * step[:, None]

        # Absorption: deposit (mua / mu_t) of the packet's weight.
        absorbed_frac = mua / mu_t
        weight[idx] *= (1.0 - absorbed_frac)

        # Photons that have crossed back above the surface (z < 0) escape;
        # apply Fresnel transmittance and tally the transmitted weight as
        # diffuse reflectance, then kill the packet.
        escaping = idx[pos[idx, 2] < 0.0]
        if escaping.size:
            cos_i = np.clip(-direction[escaping, 2], 1e-6, 1.0)
            r_fresnel = _fresnel_reflectance(cos_i, n_medium, n_ambient)
            transmitted = weight[escaping] * (1.0 - r_fresnel)
            reflected_weight += float(np.sum(transmitted))

            # The remaining (r_fresnel) fraction of the weight undergoes
            # internal reflection at the boundary rather than being lost:
            # mirror the photon back across the surface and flip its
            # z-direction so it continues propagating in the medium.
            weight[escaping] *= r_fresnel
            pos[escaping, 2] = -pos[escaping, 2]
            direction[escaping, 2] = -direction[escaping, 2]

            negligible = escaping[weight[escaping] < weight_threshold * 1e-2]
            alive[negligible] = False

        idx = np.where(alive)[0]
        if idx.size == 0:
            break

        # Henyey-Greenstein scattering: sample a new direction.
        m = idx.size
        rnd2 = rng.random(m)
        if abs(g) > 1e-3:
            cos_theta = (1.0 / (2.0 * g)) * (
                1.0 + g**2 - ((1.0 - g**2) / (1.0 - g + 2.0 * g * rnd2)) ** 2
            )
        else:
            cos_theta = 2.0 * rnd2 - 1.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * rng.random(m)

        direction[idx] = _rotate_direction(direction[idx], cos_theta, sin_theta, phi)

        # Russian roulette for low-weight packets to keep the simulation
        # unbiased while terminating "dead" photons early.
        low = idx[weight[idx] < weight_threshold]
        if low.size:
            survive = rng.random(low.size) < (1.0 / roulette_m)
            weight[low[survive]] *= roulette_m
            alive[low[~survive]] = False

    return reflected_weight / n_photons


def _fresnel_reflectance(cos_i: np.ndarray, n1: float, n2: float) -> np.ndarray:
    """Unpolarized Fresnel reflectance at an interface, with total internal
    reflection handled explicitly."""
    sin_i = np.sqrt(np.clip(1.0 - cos_i**2, 0.0, 1.0))
    sin_t = n1 / n2 * sin_i
    tir = sin_t >= 1.0
    sin_t_c = np.clip(sin_t, 0.0, 1.0 - 1e-12)
    cos_t = np.sqrt(1.0 - sin_t_c**2)

    rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
    r = 0.5 * (rs + rp)
    r = np.where(tir, 1.0, r)
    return r


def _rotate_direction(
    d: np.ndarray, cos_theta: np.ndarray, sin_theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Rotate each direction vector in d by polar angle theta and azimuth phi."""
    ux, uy, uz = d[:, 0], d[:, 1], d[:, 2]
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    denom = np.sqrt(np.clip(1.0 - uz**2, 1e-12, None))
    near_pole = np.abs(uz) > 0.99999

    new_x = np.where(
        near_pole,
        sin_theta * cos_phi,
        sin_theta * (ux * uz * cos_phi - uy * sin_phi) / denom + ux * cos_theta,
    )
    new_y = np.where(
        near_pole,
        sin_theta * sin_phi,
        sin_theta * (uy * uz * cos_phi + ux * sin_phi) / denom + uy * cos_theta,
    )
    new_z = np.where(
        near_pole,
        np.sign(uz) * cos_theta,
        -sin_theta * cos_phi * denom + uz * cos_theta,
    )

    out = np.stack([new_x, new_y, new_z], axis=1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norm, 1e-12, None)
