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
    lambda_local_now: np.ndarray
    lambda_mw_now: np.ndarray
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


def _nearest_summary(distances, active_counts):
    distances = np.asarray(distances, dtype=float)
    active_counts = np.asarray(active_counts, dtype=int)
    if distances.size == 0 or active_counts.size != distances.size:
        return {"mean": np.nan, "median": np.nan, "p16": np.nan, "p84": np.nan}

    active_distances = distances[(active_counts > 0) & np.isfinite(distances)]
    if active_distances.size == 0:
        return {"mean": np.nan, "median": np.nan, "p16": np.nan, "p84": np.nan}

    def percentile(q):
        rank = max(0, int(np.ceil(q * active_distances.size)) - 1)
        return float(np.sort(active_distances)[rank])

    return {
        "mean": float(np.mean(active_distances)),
        "median": float(np.median(active_distances)),
        "p16": percentile(0.16),
        "p84": percentile(0.84),
    }


def _prepare_local_coordinate_sampler(grid, rng):
    R = np.asarray(grid["R"], dtype=float)
    theta = np.asarray(grid["theta"], dtype=float)
    z = np.asarray(grid["z"], dtype=float)
    dR = np.gradient(R)
    r_edges = np.r_[R[0] - 0.5 * dR[0], R[:-1] + 0.5 * dR[:-1],
                    R[-1] + 0.5 * dR[-1]]
    dtheta = 2.0 * np.pi / theta.size
    theta_lower = theta - 0.5 * dtheta
    if z.size > 1:
        dz = z[1] - z[0]
    else:
        dz = 10_000.0
    z_edges = np.r_[z[0] - 0.5 * dz, z[:-1] + 0.5 * dz,
                    z[-1] + 0.5 * dz]

    weights = (grid["stellar_birth_rate"] * grid["volume"]).sum(axis=-1)
    weights *= grid["local_mask"][..., 0]
    spatial_shape = weights.shape
    weights = np.asarray(weights, dtype=float).ravel()
    candidate_cells = np.flatnonzero(weights > 0.0)
    if candidate_cells.size == 0 or weights[candidate_cells].sum() <= 0.0:
        raise ValueError("The Galactic grid has no weighted cells in the local horizon")
    probabilities = weights[candidate_cells] / weights[candidate_cells].sum()

    def draw(n_events):
        n_events = int(n_events)
        if n_events < 0:
            raise ValueError("n_events must be non-negative")
        if n_events == 0:
            return np.empty((0, 3), dtype=float)

        coordinates = np.empty((n_events, 3), dtype=float)
        pending = np.arange(n_events)
        fallback_cells = np.zeros(n_events, dtype=np.intp)
        for _ in range(50):
            if pending.size == 0:
                break
            selected = rng.choice(
                candidate_cells, size=pending.size, p=probabilities
            )
            fallback_cells[pending] = selected
            ri, ti, zi = np.unravel_index(selected, spatial_shape)
            r_inner = np.maximum(r_edges[ri], 0.0)
            r_outer = np.maximum(r_edges[ri + 1], r_inner)
            radius = np.sqrt(rng.uniform(r_inner ** 2, r_outer ** 2))
            angle = rng.uniform(theta_lower[ti], theta_lower[ti] + dtheta)
            height = rng.uniform(z_edges[zi], z_edges[zi + 1])
            candidate_coords = np.column_stack((
                radius * np.cos(angle),
                radius * np.sin(angle),
                height,
            ))
            distance_squared = (
                (candidate_coords[:, 0] - R_SUN_LY) ** 2
                + candidate_coords[:, 1] ** 2
                + (candidate_coords[:, 2] - Z_SUN_LY) ** 2
            )
            accepted = distance_squared <= R_LOCAL_LY ** 2
            coordinates[pending[accepted]] = candidate_coords[accepted]
            pending = pending[~accepted]

        if pending.size:
            ri, ti, zi = np.unravel_index(
                fallback_cells[pending], spatial_shape
            )
            radius = R[ri]
            angle = theta[ti]
            coordinates[pending] = np.column_stack((
                radius * np.cos(angle),
                radius * np.sin(angle),
                z[zi],
            ))
        return coordinates

    return draw


def sample_local_coordinates(grid, n_events, seed=42):
    """Draw seeded, rate-weighted, sub-voxel coordinates inside the local sphere."""
    rng = np.random.default_rng(seed)
    return _prepare_local_coordinate_sampler(grid, rng)(n_events)


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
    mw_by_epoch = base_density.sum(axis=(0, 1, 2))
    local_by_epoch = (base_density * local_mask).sum(axis=(0, 1, 2))
    cell_left = tb - 0.5 * dt
    cell_right = tb + 0.5 * dt
    latest_birth = T_MW_GYR - sampled["t_evo_gyr"]
    ever_overlap = np.maximum(
        0.0,
        np.minimum(cell_right[None, :], latest_birth[:, None])
        - np.maximum(cell_left[None, :], 0.0),
    )
    active_earliest_birth = np.maximum(
        0.0, latest_birth - sampled["L_yr"] / 1.0e9
    )
    active_overlap = np.maximum(
        0.0,
        np.minimum(cell_right[None, :], latest_birth[:, None])
        - np.maximum(cell_left[None, :], active_earliest_birth[:, None]),
    )
    astro_weight = (sampled["f_star"] * sampled["eta_e"]
                    * sampled["f_l"] * sampled["f_filter"])
    lam_mw = (ever_overlap @ mw_by_epoch) * astro_weight
    lam_local = (ever_overlap @ local_by_epoch) * astro_weight
    lam_mw = np.maximum(
        np.nan_to_num(lam_mw, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    lam_local = np.minimum(
        np.maximum(np.nan_to_num(lam_local, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
        lam_mw,
    )
    K_local = rng.poisson(lam_local)
    K_outside_ever = rng.poisson(np.maximum(lam_mw - lam_local, 0.0))
    K_mw = K_local + K_outside_ever

    ever = K_local > 0
    prob_ever = 100.0 * ever.mean()

    local_coords = []
    causal_active = np.zeros(n_runs, dtype=bool)

    RR, TT, ZZ = np.meshgrid(R, theta, z, indexing="ij")
    XX = RR * np.cos(TT)
    YY = RR * np.sin(TT)
    dx = XX - R_SUN_LY
    dy = YY
    dz = ZZ - Z_SUN_LY
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    sample_local = _prepare_local_coordinate_sampler(grid, rng)

    local_bin_width = 100.0
    outer_bin_width = 250.0
    local_edges = np.arange(0.0, R_LOCAL_LY, local_bin_width)
    outer_count = int(np.ceil((float(np.max(dist)) - R_LOCAL_LY)
                              / outer_bin_width)) + 1
    outer_edges = R_LOCAL_LY + outer_bin_width * np.arange(1, outer_count + 1)
    radial_edges = np.concatenate((local_edges, [R_LOCAL_LY], outer_edges))
    radial_bin = np.searchsorted(radial_edges[1:], dist.ravel(), side="left")
    n_radial_bins = radial_edges.size - 1
    n_local_bins = int(round(R_LOCAL_LY / local_bin_width))
    flat_base_density = base_density.reshape((-1, tb.size))
    radial_epoch_weights = np.zeros((tb.size, n_radial_bins), dtype=float)
    for kt in range(tb.size):
        radial_epoch_weights[kt] = np.bincount(
            radial_bin,
            weights=flat_base_density[:, kt],
            minlength=n_radial_bins,
        )
    radial_lambda_now = (active_overlap @ radial_epoch_weights) * astro_weight[:, None]
    radial_lambda_now = np.maximum(
        np.nan_to_num(radial_lambda_now, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    lambda_local_now = radial_lambda_now[:, :n_local_bins].sum(axis=1)
    lambda_mw_now = radial_lambda_now.sum(axis=1)
    lambda_outside_now = np.maximum(lambda_mw_now - lambda_local_now, 0.0)
    lambda_outside_ever = np.maximum(lam_mw - lam_local, 0.0)

    p_local_active = np.divide(
        lambda_local_now,
        lam_local,
        out=np.zeros_like(lambda_local_now),
        where=lam_local > 0.0,
    )
    p_outside_active = np.divide(
        lambda_outside_now,
        lambda_outside_ever,
        out=np.zeros_like(lambda_outside_now),
        where=lambda_outside_ever > 0.0,
    )
    local_active_counts = rng.binomial(
        K_local, np.clip(p_local_active, 0.0, 1.0)
    )
    outside_active_counts = rng.binomial(
        K_outside_ever, np.clip(p_outside_active, 0.0, 1.0)
    )
    N_mw_now = local_active_counts + outside_active_counts
    nearest_active = np.full(n_runs, np.inf)

    def sample_nearest_distance(radial_weights, count, bin_offset):
        total_weight = radial_weights.sum()
        if count <= 0 or total_weight <= 0.0:
            return np.inf
        minimum_uniform = -np.expm1(
            np.log1p(-rng.random()) / int(count)
        )
        cdf = np.cumsum(radial_weights) / total_weight
        local_index = int(np.searchsorted(cdf, minimum_uniform, side="right"))
        local_index = min(local_index, radial_weights.size - 1)
        bin_index = bin_offset + local_index
        lower = radial_edges[bin_index]
        upper = radial_edges[bin_index + 1]
        distance = lower + (upper - lower) * rng.random()
        if bin_index >= n_local_bins:
            distance = max(distance, np.nextafter(R_LOCAL_LY, np.inf))
        return float(distance)

    for i in range(n_runs):
        if local_active_counts[i] > 0:
            nearest_active[i] = sample_nearest_distance(
                radial_lambda_now[i, :n_local_bins],
                local_active_counts[i],
                0,
            )
        elif outside_active_counts[i] > 0:
            nearest_active[i] = sample_nearest_distance(
                radial_lambda_now[i, n_local_bins:],
                outside_active_counts[i],
                n_local_bins,
            )
    now_active = nearest_active <= R_LOCAL_LY

    for i, count in enumerate(K_local):
        if count == 0:
            continue
        sample_count = min(int(count), max_local_coordinates)
        coordinates = sample_local(sample_count)
        distance = np.sqrt(
            (coordinates[:, 0] - R_SUN_LY) ** 2
            + coordinates[:, 1] ** 2
            + (coordinates[:, 2] - Z_SUN_LY) ** 2
        )
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
        distance_gyr = distance / (C_LY_PER_YR * 1.0e9)
        causal = ((t_emerge + distance_gyr <= T_MW_GYR)
                  & (T_MW_GYR <= t_end + distance_gyr))
        causal_active[i] = np.any(causal)
        local_coords.extend(coordinates)

    prob_now = 100.0 * now_active.mean()
    prob_causal = 100.0 * causal_active.mean()

    coords_sample = np.asarray(local_coords, dtype=float)
    if coords_sample.size == 0:
        coords_sample = np.empty((0, 3), dtype=float)

    return SimulationResult(
        prob_ever_10kly=float(prob_ever),
        prob_now_10kly=float(prob_now),
        prob_causal_10kly=float(prob_causal),
        lambda_local_now=lambda_local_now,
        lambda_mw_now=lambda_mw_now,
        K_local=K_local,
        K_mw=K_mw,
        N_mw_now=N_mw_now,
        nearest_active_ly=nearest_active,
        coords_sample=coords_sample,
        local_spawn_summary=_summary(K_local),
        mw_now_summary=_summary(N_mw_now),
        nearest_summary=_nearest_summary(nearest_active, N_mw_now),
        runtime_seconds=time.perf_counter() - start,
    )
