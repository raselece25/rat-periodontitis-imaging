"""
monte_carlo_two_layer.py

Two-layer extension of the single-layer `simulate_diffuse_reflectance`
Monte Carlo photon-transport model in `monte_carlo.py`, following the same
photon-packet weighting scheme described in:

    L. Wang, S. L. Jacques, L. Zheng, "MCML -- Monte Carlo modeling of
    light transport in multi-layered tissues," Comput. Methods Programs
    Biomed. 47(2), 131-146 (1995).

Layer 1 occupies 0 <= z < thickness1 (e.g. epidermis); layer 2 occupies
z >= thickness1 and is semi-infinite (e.g. dermis). Still teaching-scale:
total diffuse reflectance only, no spatial-frequency modulation, no
lateral photon-position tallying.

Boundary bookkeeping (the actual two-layer addition):

A photon's step is drawn once as a dimensionless number of mean free
paths (`rem`, ~Exp(1)). By the memoryless property of the exponential
distribution, this budget is valid regardless of which layer's mu_t
converts it to physical distance, so it carries unchanged across a
layer boundary. Each outer-loop iteration therefore does ONE of two
things per photon: (a) the remaining budget is smaller than the
distance-to-boundary, so the photon reaches its next scattering site
inside the current layer -- absorption is deposited and a new HG
scattering direction is drawn (exactly as in the single-layer model,
just parameterized per-layer), and a fresh `rem` is drawn for the next
hop; or (b) the boundary is reached first, so the photon is moved
exactly to the boundary, `rem` is decremented by the distance already
spent, and Fresnel reflection/Snell refraction decides whether it
reflects back into the same layer or transmits into the other one
(or, at the top surface, escapes as diffuse reflectance). No new `rem`
is drawn in this branch -- the same hop keeps resolving on subsequent
iterations, so `max_steps` bounds "scatters + crossings" combined
rather than only scatters.

ponytail: thin layer1 relative to a mean free path means more boundary
crossings are needed to resolve each hop, which eats into the same
max_steps budget used for actual scattering events -- if convergence
looks off for a very thin/high-scattering layer 1, raise max_steps
first before suspecting the physics.
"""

from __future__ import annotations

import numpy as np

from monte_carlo import _fresnel_reflectance, _rotate_direction


def simulate_diffuse_reflectance_two_layer(
    mua1: float,
    musp1: float,
    mua2: float,
    musp2: float,
    thickness1: float,
    g1: float = 0.8,
    g2: float = 0.8,
    n1: float = 1.4,
    n2: float = 1.4,
    n_ambient: float = 1.0,
    n_photons: int = 20_000,
    weight_threshold: float = 1e-4,
    roulette_m: int = 10,
    max_steps: int = 8000,
    seed: int | None = 0,
) -> float:
    """Estimate total diffuse reflectance for a two-layer turbid medium.

    Parameters
    ----------
    mua1, musp1, g1, n1 : layer-1 (top) optical properties [mm^-1], anisotropy,
        refractive index.
    mua2, musp2, g2, n2 : layer-2 (semi-infinite) optical properties, same units.
    thickness1 : float [mm]
        Layer-1 thickness. Layer 2 extends from thickness1 to infinity.
    n_ambient, n_photons, weight_threshold, roulette_m, max_steps, seed :
        Same meaning as in `simulate_diffuse_reflectance`.

    Returns
    -------
    rd : float
        Estimated total diffuse reflectance at fx = 0. Passing identical
        (mua2, musp2, g2, n2) == (mua1, musp1, g1, n1) should reproduce
        `simulate_diffuse_reflectance(mua1, musp1, g1, n1, ...)` within
        Monte Carlo noise -- that's the sanity check in `demo()` below.
    """
    rng = np.random.default_rng(seed)
    mus1, mus2 = musp1 / (1.0 - g1), musp2 / (1.0 - g2)
    mu_t = np.array([mua1 + mus1, mua2 + mus2])
    mua_arr = np.array([mua1, mua2])
    g_arr = np.array([g1, g2])
    n_arr = np.array([n1, n2])

    r_sp = ((n1 - n_ambient) / (n1 + n_ambient)) ** 2

    n = n_photons
    weight = np.full(n, 1.0 - r_sp, dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    direction = np.zeros((n, 3), dtype=float)
    direction[:, 2] = 1.0
    layer = np.zeros(n, dtype=np.int8)  # 0 = layer 1, 1 = layer 2

    alive = np.ones(n, dtype=bool)
    rem = -np.log(np.clip(rng.random(n), 1e-12, 1.0))  # dimensionless budget per photon
    reflected_weight = 0.0

    for _ in range(max_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break

        cur_layer = layer[idx]
        uz = direction[idx, 2]
        z = pos[idx, 2]
        mt = mu_t[cur_layer]

        target = np.full(idx.size, np.nan)
        target[(cur_layer == 0) & (uz > 0)] = thickness1  # layer1 -> internal, going down
        target[(cur_layer == 0) & (uz < 0)] = 0.0          # layer1 -> top surface, going up
        target[(cur_layer == 1) & (uz < 0)] = thickness1   # layer2 -> internal, going up
        # (cur_layer == 1) & (uz >= 0): semi-infinite below, no boundary -> stays NaN.

        has_target = ~np.isnan(target)
        s_to_target = np.full(idx.size, np.inf)
        s_to_target[has_target] = np.clip(
            (target[has_target] - z[has_target]) * mt[has_target] / uz[has_target],
            0.0, None,
        )

        scatters_first = rem[idx] <= s_to_target
        scat_idx = idx[scatters_first]
        cross_idx = idx[~scatters_first]

        # --- Reaches next scattering site inside the current layer ---
        if scat_idx.size:
            dist = rem[scat_idx] / mu_t[layer[scat_idx]]
            pos[scat_idx] += direction[scat_idx] * dist[:, None]
            frac = mua_arr[layer[scat_idx]] / mu_t[layer[scat_idx]]
            weight[scat_idx] *= 1.0 - frac

            m = scat_idx.size
            rnd2 = rng.random(m)
            gsc = g_arr[layer[scat_idx]]
            cos_theta = np.where(
                np.abs(gsc) > 1e-3,
                (1.0 / (2.0 * gsc + 1e-30))
                * (1.0 + gsc**2 - ((1.0 - gsc**2) / (1.0 - gsc + 2.0 * gsc * rnd2)) ** 2),
                2.0 * rnd2 - 1.0,
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            phi = 2.0 * np.pi * rng.random(m)
            direction[scat_idx] = _rotate_direction(direction[scat_idx], cos_theta, sin_theta, phi)

            low = scat_idx[weight[scat_idx] < weight_threshold]
            if low.size:
                survive = rng.random(low.size) < (1.0 / roulette_m)
                weight[low[survive]] *= roulette_m
                alive[low[~survive]] = False

            still = scat_idx[alive[scat_idx]]
            if still.size:
                rem[still] = -np.log(np.clip(rng.random(still.size), 1e-12, 1.0))

        # --- Hits a layer/surface boundary before its next scattering site ---
        if cross_idx.size:
            s_used = s_to_target[~scatters_first]
            dist = s_used / mu_t[layer[cross_idx]]
            pos[cross_idx] += direction[cross_idx] * dist[:, None]
            rem[cross_idx] -= s_used

            at_top = (layer[cross_idx] == 0) & (direction[cross_idx, 2] < 0)
            top_idx = cross_idx[at_top]
            int_idx = cross_idx[~at_top]

            if top_idx.size:
                cos_i = np.clip(-direction[top_idx, 2], 1e-6, 1.0)
                r_f = _fresnel_reflectance(cos_i, n1, n_ambient)
                transmitted = weight[top_idx] * (1.0 - r_f)
                reflected_weight += float(np.sum(transmitted))
                weight[top_idx] *= r_f
                pos[top_idx, 2] = -pos[top_idx, 2]
                direction[top_idx, 2] *= -1.0
                dead = top_idx[weight[top_idx] < weight_threshold * 1e-2]
                alive[dead] = False

            if int_idx.size:
                from_layer = layer[int_idx]
                to_layer = 1 - from_layer
                n_from, n_to = n_arr[from_layer], n_arr[to_layer]
                cos_i = np.clip(np.abs(direction[int_idx, 2]), 1e-6, 1.0)
                r_f = _fresnel_reflectance(cos_i, n_from, n_to)
                transmit = rng.random(int_idx.size) >= r_f

                reflect_idx = int_idx[~transmit]
                direction[reflect_idx, 2] *= -1.0

                transmit_idx = int_idx[transmit]
                if transmit_idx.size:
                    direction[transmit_idx] = _refract_direction(
                        direction[transmit_idx], n_from[transmit], n_to[transmit]
                    )
                    layer[transmit_idx] = to_layer[transmit]

    return reflected_weight / n_photons


def _refract_direction(d: np.ndarray, n_from: np.ndarray, n_to: np.ndarray) -> np.ndarray:
    """Snell's-law refraction at a horizontal (z = const) layer boundary,
    preserving the photon's up/down travel sense. Only called on photons
    already selected as transmitting (not totally internally reflected)."""
    ux, uy, uz = d[:, 0], d[:, 1], d[:, 2]
    cos_i = np.clip(np.abs(uz), 1e-6, 1.0)
    sin_i = np.sqrt(np.clip(1.0 - cos_i**2, 0.0, 1.0))
    sin_t = np.clip(n_from / n_to * sin_i, 0.0, 1.0 - 1e-12)
    cos_t = np.sqrt(1.0 - sin_t**2)

    scale = np.where(sin_i > 1e-6, sin_t / np.clip(sin_i, 1e-6, None), 1.0)
    out = np.stack([ux * scale, uy * scale, np.sign(uz) * cos_t], axis=1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norm, 1e-12, None)


def demo() -> None:
    """Sanity check: a two-layer medium with identical layer-1/layer-2
    optical properties should reproduce the single-layer model's Rd
    within Monte Carlo noise."""
    from monte_carlo import simulate_diffuse_reflectance

    mua, musp, g, n = 0.01, 1.0, 0.8, 1.4
    n_photons = 60_000

    rd_single = simulate_diffuse_reflectance(
        mua, musp, g=g, n_medium=n, n_photons=n_photons, seed=1
    )
    rd_two = simulate_diffuse_reflectance_two_layer(
        mua, musp, mua, musp, thickness1=0.5, g1=g, g2=g, n1=n, n2=n,
        n_photons=n_photons, seed=1,
    )
    rel_diff = abs(rd_two - rd_single) / rd_single
    print(f"single-layer Rd = {rd_single:.4f}, two-layer (identical) Rd = {rd_two:.4f}, "
          f"relative diff = {rel_diff:.2%}")
    assert rel_diff < 0.15, "two-layer model should match single-layer when layers are identical"

    # A more absorbing/scattering layer 2 underneath a thin layer 1 should
    # measurably change Rd relative to a homogeneous layer-1-only medium.
    rd_layered = simulate_diffuse_reflectance_two_layer(
        mua, musp, mua2=0.05, musp2=2.0, thickness1=0.1, g1=g, g2=g, n1=n, n2=n,
        n_photons=n_photons, seed=2,
    )
    print(f"thin layer1 over a more absorbing layer2: Rd = {rd_layered:.4f}")
    assert 0.0 <= rd_layered <= 1.0


if __name__ == "__main__":
    demo()
