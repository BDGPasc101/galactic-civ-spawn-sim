import time

import numpy as np

from src.astrobiology import AstrobiologyPriors, PRESETS
from src.galaxy_model import R_LOCAL_LY, R_SUN_LY, Z_SUN_LY, build_galactic_rate_grid
from src.monte_carlo import run_monte_carlo, sample_local_coordinates


def test_causality_limit():
    priors = AstrobiologyPriors(t_evo_mean_gyr=14.0, t_evo_sigma_gyr=0.01)
    result = run_monte_carlo(100, priors, seed=5)
    assert result.prob_ever_10kly == 0.0
    assert result.prob_now_10kly == 0.0


def test_zero_great_filter():
    priors = AstrobiologyPriors(f_i=0.0, f_c=0.5)
    result = run_monte_carlo(500, priors, seed=3)
    assert result.prob_ever_10kly == 0.0
    assert result.prob_now_10kly == 0.0
    assert result.prob_causal_10kly == 0.0
    assert result.mw_now_summary["median"] == 0
    assert np.isnan(result.nearest_summary["median"])


def test_immortal_limit_converges():
    priors = AstrobiologyPriors(L_median_yr=1e11, L_sigma=0.01)
    result = run_monte_carlo(1000, priors, seed=7)
    assert result.prob_now_10kly == result.prob_ever_10kly
    assert result.nearest_summary["median"] <= 10_000.0


def test_light_cone_is_shifted_window():
    result = run_monte_carlo(500, AstrobiologyPriors(), seed=11)
    assert 0.0 <= result.prob_causal_10kly <= 100.0


def test_active_now_intensities_have_consistent_galactic_ratio():
    result = run_monte_carlo(1000, AstrobiologyPriors(), seed=19)
    assert np.all(result.lambda_mw_now >= result.lambda_local_now)
    ratio = result.lambda_mw_now.sum() / result.lambda_local_now.sum()
    assert 5.0 <= ratio <= 100.0


def test_nearest_active_neighbor_is_unconditional():
    priors = AstrobiologyPriors(f_i=0.01, f_c=0.01)
    result = run_monte_carlo(2000, priors, seed=29)
    assert result.prob_now_10kly < 50.0
    if result.mw_now_summary["median"] == 0:
        assert np.isnan(result.nearest_summary["median"])
    else:
        assert result.nearest_summary["median"] > 10_000.0


def test_local_coordinates_are_continuous_and_inside_horizon():
    grid = build_galactic_rate_grid()
    coordinates = sample_local_coordinates(grid, 500, seed=73)
    distance = np.sqrt(
        (coordinates[:, 0] - R_SUN_LY) ** 2
        + coordinates[:, 1] ** 2
        + (coordinates[:, 2] - Z_SUN_LY) ** 2
    )
    radii = np.hypot(coordinates[:, 0], coordinates[:, 1])
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    assert coordinates.shape == (500, 3)
    assert np.all(distance <= R_LOCAL_LY)
    assert np.unique(radii).size == radii.size
    assert np.unique(angles).size == angles.size
    assert not np.any(np.isclose(
        radii[:, None], grid["R"][None, :], rtol=0.0, atol=1e-8
    ))
    angular_offset = np.abs(np.angle(np.exp(
        1j * (angles[:, None] - grid["theta"][None, :])
    )))
    assert not np.any(angular_offset < 1e-12)


def test_monte_carlo_coordinate_samples_are_jittered_and_local():
    result = run_monte_carlo(
        100, AstrobiologyPriors(), seed=74, max_local_coordinates=2
    )
    coords = result.coords_sample
    distance = np.sqrt(
        (coords[:, 0] - R_SUN_LY) ** 2
        + coords[:, 1] ** 2
        + (coords[:, 2] - Z_SUN_LY) ** 2
    )
    assert coords.shape[0] > 0
    assert np.all(distance <= R_LOCAL_LY)


def test_performance():
    started = time.perf_counter()
    result = run_monte_carlo(10_000, AstrobiologyPriors(), seed=42)
    elapsed = time.perf_counter() - started
    assert result.runtime_seconds <= elapsed + 0.1
    assert elapsed < 3.0


def test_probability_relationships_for_all_presets():
    for name, priors in PRESETS.items():
        result = run_monte_carlo(1000, priors, seed=101)
        assert 0.0 <= result.prob_now_10kly <= result.prob_ever_10kly <= 100.0
        assert 0.0 <= result.prob_causal_10kly <= result.prob_ever_10kly <= 100.0
        assert np.all(result.lambda_mw_now >= result.lambda_local_now)
        assert np.array_equal(
            result.N_mw_now,
            result.N_mw_now,
        )
        assert result.mw_now_summary["median"] == float(np.median(result.N_mw_now))


def test_causal_and_now_probabilities_agree_for_immortal_lifetimes():
    result = run_monte_carlo(
        2000,
        AstrobiologyPriors(L_median_yr=1e11, L_sigma=0.01),
        seed=102,
    )
    assert abs(result.prob_causal_10kly - result.prob_now_10kly) <= 5.0


def test_uniform_subvoxel_weight_obeys_r_cubed_cdf():
    R = np.linspace(17_750.0, 35_750.0, 121)
    theta = np.linspace(-np.pi, np.pi, 96, endpoint=False)
    z = np.linspace(-9_900.0, 9_900.0, 49)
    RR, TT, ZZ = np.meshgrid(R, theta, z, indexing="ij")
    dR = np.gradient(R)
    r_edges = np.r_[R[0] - 0.5 * dR[0], R[:-1] + 0.5 * dR[:-1],
                    R[-1] + 0.5 * dR[-1]]
    dtheta = 2.0 * np.pi / theta.size
    dz = z[1] - z[0]
    z_edges = np.r_[z[0] - 0.5 * dz, z[:-1] + 0.5 * dz,
                    z[-1] + 0.5 * dz]
    z_shell = z_edges[1:] - z_edges[:-1]
    r_shell = (r_edges[1:] ** 2 - r_edges[:-1] ** 2) / 2.0
    volume = dtheta * r_shell[:, None, None] * z_shell[None, None, :]
    volume = np.broadcast_to(
        volume[..., None], (R.size, theta.size, z.size, 1)
    ).copy()
    XX = RR * np.cos(TT)
    YY = RR * np.sin(TT)
    local_mask = (
        (XX - R_SUN_LY) ** 2 + YY ** 2 + (ZZ - Z_SUN_LY) ** 2
        <= R_LOCAL_LY ** 2
    )
    local_mask = local_mask[..., None]

    grid = {
        "R": R,
        "theta": theta,
        "z": z,
        "t_birth": np.array([0.0]),
        "volume": volume,
        "stellar_birth_rate": np.ones(
            (R.size, theta.size, z.size, 1), dtype=float
        ),
        "local_mask": local_mask,
    }
    coordinates = sample_local_coordinates(grid, 40_000, seed=77)
    radii = np.sqrt(
        (coordinates[:, 0] - R_SUN_LY) ** 2
        + coordinates[:, 1] ** 2
        + (coordinates[:, 2] - Z_SUN_LY) ** 2
    )

    assert np.all(radii <= R_LOCAL_LY)
    bins = np.linspace(0.0, 10_000.0, 21)
    empirical = np.array([np.mean(radii <= edge) for edge in bins])
    expected = np.clip((bins / 10_000.0) ** 3, 0.0, 1.0)
    assert np.max(np.abs(empirical - expected)) < 0.04


def test_single_run_and_evolution_limit_are_stable():
    single = run_monte_carlo(
        1,
        AstrobiologyPriors(t_evo_mean_gyr=13.59, t_evo_sigma_gyr=0.01),
        seed=301,
    )
    assert single.N_mw_now.shape == (1,)
    assert np.all(np.isfinite(single.N_mw_now))
    assert np.isfinite(single.mw_now_summary["median"])

    limit = run_monte_carlo(
        300,
        AstrobiologyPriors(t_evo_mean_gyr=13.59, t_evo_sigma_gyr=0.01,
                           L_median_yr=1.0),
        seed=302,
    )
    assert limit.prob_now_10kly <= limit.prob_ever_10kly
