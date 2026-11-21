"""Vectorized Spatio-Temporal Galactic civilization Monte Carlo simulation."""

from dataclasses import dataclass
import time

import numpy as np

from .astrobiology import AstrobiologyPriors
from .galaxy_model import (T_MW_GYR, R_SUN_LY, Z_SUN_LY, R_LOCAL_LY,
                           build_galactic_rate_grid)

C_LY_PER_YR = 1.0


@dataclass
class SimulationResult:
    prob_ever_10kly: float
    prob_now_10kly: float
    prob_causal_10kly: float
    K_local: np.ndarray
    K_mw: np.ndarray
    N_mw_now: np.ndarray
    nearest_active_ly: np.ndarray
    coords_sample: np.ndarray
    local_spawn_summary: dict
    mw_now_summary: dict
    nearest_summary: dict
    runtime_seconds: float


def _summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": np.nan, "median": np.nan, "p16": np.nan, "p84": np.nan}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "median": np.nan, "p16": np.nan, "p84": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p16": float(np.percentile(finite, 16)),
        "p84": float(np.percentile(finite, 84)),
    }


def run_monte_carlo(n_runs=10_000, priors=None, seed=42, grid=None,
                    max_local_coordinates=10):
    start = time.perf_counter()
    priors = priors if isinstance(priors, AstrobiologyPriors) else AstrobiologyPriors()
    rng = np.random.default_rng(seed)
    sampled = priors.sample_priors(n_runs, rng)
    grid = grid if grid is not None else build_galactic_rate_grid()

    R = grid["R"]
    theta = grid["theta"]
    z = grid["z"]
    tb = grid["t_birth"]
    rate = grid["stellar_birth_rate"]
    volume = grid["volume"]
    local_mask = grid["local_mask"]

    # Each grid point represents an event count; normalize sample-weighted priors below.
    base_density = rate * volume
    dt = np.gradient(tb)
    dt *= 1.0
    mw_by_epoch = base_density.sum(axis=(0, 1, 2))
    local_by_epoch = (base_density * local_mask).sum(axis=(0, 1, 2))
    evolution_mask = (tb[None, :] <=
                      T_MW_GYR - sampled["t_evo_gyr"][:, None])
    astro_weight = (sampled["f_star"] * sampled["eta_e"]
                    * sampled["f_l"] * sampled["f_filter"])
    lam_mw = (np.where(evolution_mask,
                       mw_by_epoch[None, :] * dt[None, :], 0.0).sum(axis=1)
              * astro_weight)
    lam_local = (np.where(evolution_mask,
                          local_by_epoch[None, :] * dt[None, :], 0.0).sum(axis=1)
                 * astro_weight)

    lam_mw = np.nan_to_num(lam_mw, nan=0.0, posinf=0.0, neginf=0.0)
    lam_local = np.nan_to_num(lam_local, nan=0.0, posinf=0.0, neginf=0.0)
    K_mw = rng.poisson(lam_mw)
    K_local = rng.poisson(lam_local)

    ever = K_local > 0
    prob_ever = 100.0 * ever.mean()

    now_mask = sampled["t_evo_gyr"][:, None] if False else None
    local_coords = []
    emergence_times = []
    local_distances = []
    local_ends = []
    now_active = np.zeros(n_runs, dtype=bool)
    causal_active = np.zeros(n_runs, dtype=bool)
    nearest_active = np.full(n_runs, np.inf)

    RR, TT, ZZ = np.meshgrid(R, theta, z, indexing="ij")
    XX = RR * np.cos(TT)
    YY = RR * np.sin(TT)
    dx = XX - R_SUN_LY
    dy = YY
    dz = ZZ - Z_SUN_LY
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    local_weight_cube = dist <= R_LOCAL_LY
    habit_weight = base_density.sum(axis=-1) * local_weight_cube

    for i, count in enumerate(K_local):
        if count == 0:
            continue
        sample_count = min(int(count), max_local_coordinates)
        flat_prob = np.asarray(habit_weight).ravel()
        flat_prob = np.maximum(flat_prob, 0.0)
        if flat_prob.sum() <= 0.0:
            continue
        flat_prob /= flat_prob.sum()
        positions = rng.choice(flat_prob.size, size=sample_count, p=flat_prob)
        rr_idx, tt_idx, zz_idx = np.unravel_index(positions, habit_weight.shape)
        x = R[rr_idx] * np.cos(theta[tt_idx])
        y = R[rr_idx] * np.sin(theta[tt_idx])
        zc = z[zz_idx]
        distance = np.sqrt((x - R_SUN_LY) ** 2 + y * y + (zc - Z_SUN_LY) ** 2)
        t_birth = rng.choice(tb, size=sample_count)
        valid = t_birth + sampled["t_evo_gyr"][i] <= T_MW_GYR
        # Rejection sampling is applied to birth epochs when the evolutionary limit cuts them off.
        attempts = 0
        while not valid.all() and attempts < 20:
            t_birth[~valid] = rng.choice(tb, size=int((~valid).sum()))
            valid = t_birth + sampled["t_evo_gyr"][i] <= T_MW_GYR
            attempts += 1
        t_birth = t_birth[valid]
        if t_birth.size == 0:
            continue
        t_emerge = t_birth + sampled["t_evo_gyr"][i]
        distance = distance[valid]
        t_end_yr = t_emerge * 1.0e9 + sampled["L_yr"][i]
        t_end = t_end_yr / 1.0e9
        active = (t_emerge <= T_MW_GYR) & (T_MW_GYR <= t_end)
        now_active[i] = np.any(active)
        if np.any(active):
            nearest_active[i] = distance[active].min()
        distance_gyr = distance / (C_LY_PER_YR * 1.0e9)
        causal = ((t_emerge + distance_gyr <= T_MW_GYR)
                  & (T_MW_GYR <= t_end + distance_gyr))
        causal_active[i] = np.any(causal)
        take = sample_count
        local_coords.extend(np.column_stack((x[:take], y[:take], zc[:take])))
        emergence_times.extend(t_emerge[:take].tolist())
        local_distances.extend(distance[:take].tolist())
        local_ends.extend(t_end[:take].tolist())

    prob_now = 100.0 * now_active.mean()
    prob_causal = 100.0 * causal_active.mean()

    N_mw_now = np.zeros(n_runs, dtype=int)
    # Approximate whole-Galaxy present population by a Poisson thinning of spawned count
    # using a common now-active fraction computed from each run's evolutionary clock.
    for i, count in enumerate(K_mw):
        fraction = max(0.0, min(1.0,
                                1.0 - sampled["t_evo_gyr"][i] / T_MW_GYR))
        N_mw_now[i] = rng.binomial(count, fraction)

    coords_sample = np.asarray(local_coords, dtype=float)
    if coords_sample.size == 0:
        coords_sample = np.empty((0, 3), dtype=float)

    return SimulationResult(
        prob_ever_10kly=float(prob_ever),
        prob_now_10kly=float(prob_now),
        prob_causal_10kly=float(prob_causal),
        K_local=K_local,
        K_mw=K_mw,
        N_mw_now=N_mw_now,
        nearest_active_ly=nearest_active,
        coords_sample=coords_sample,
        local_spawn_summary=_summary(K_local),
        mw_now_summary=_summary(N_mw_now),
        nearest_summary=_summary(nearest_active),
        runtime_seconds=time.perf_counter() - start,
    )
