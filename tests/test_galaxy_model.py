import numpy as np
import pytest

from src.galaxy_model import (
    R_SUN_LY, Z_SUN_LY, R_D_LY, R_LOCAL_LY,
    baseline_stellar_density, spiral_arm_modulation,
    metallicity_readiness, sterilization_survival,
    build_galactic_rate_grid,
)


def test_solar_coordinates_and_exponential_falloff():
    assert R_SUN_LY == 26_750.0
    assert Z_SUN_LY == 65.0
    inner = baseline_stellar_density([2_000.0], [0.0])[0]
    outer = baseline_stellar_density([20_000.0], [0.0])[0]
    assert outer < inner
    assert outer == pytest.approx(inner * np.exp(-(20_000.0 - 2_000.0) / R_D_LY), rel=1e-6)


def test_spiral_modulates_four_arm_pattern():
    R = np.full(361, R_SUN_LY)
    theta = np.linspace(-np.pi, np.pi, 361)
    arm = spiral_arm_modulation(R, theta, A_arm=1.0)
    assert arm.max() > arm.min() * 1.5
    assert np.isfinite(arm).all()


def test_metallicity_and_supernova_limits():
    early = metallicity_readiness(10_000.0, 1.0)
    mature = metallicity_readiness(10_000.0, 8.0)
    outer = metallicity_readiness(40_000.0, 8.0)
    assert 0.0 <= early < mature
    assert outer < mature
    center = sterilization_survival(2_000.0, 0.0, 5.0)
    solar = sterilization_survival(R_SUN_LY, 0.0, 5.0)
    assert center < solar
    assert 0.0 <= center <= 1.0


def test_local_sphere_mask_volume():
    grid = build_galactic_rate_grid(n_r=41, n_theta=72, n_z=21, n_t=12)
    mask = grid["local_mask"][..., 0]
    volume = grid["volume"][..., 0][mask].sum()
    expected = 4.0 / 3.0 * np.pi * R_LOCAL_LY ** 3
    assert volume > expected * 0.65
    assert volume < expected * 1.40


@pytest.mark.parametrize(
    ("n_r", "n_theta", "n_z", "n_t"),
    [(31, 48, 17, 28), (41, 72, 21, 36), (61, 96, 33, 56)],
)
def test_integrated_birth_weight_is_grid_convergent(n_r, n_theta, n_z, n_t):
    grid = build_galactic_rate_grid(
        n_r=n_r, n_theta=n_theta, n_z=n_z, n_t=n_t
    )
    base = grid["stellar_birth_rate"] * grid["volume"]
    dt = np.gradient(grid["t_birth"])
    mw_weight = np.sum(base * dt[None, None, None, :])
    local_weight = np.sum(
        base * grid["local_mask"] * dt[None, None, None, :]
    )
    assert mw_weight == pytest.approx(1.0e11, rel=2e-2)
    assert np.isfinite(local_weight)
    assert 0.0 < local_weight < mw_weight


def test_metallicity_exact_limits_and_spiral_no_arm_limit():
    assert metallicity_readiness(10_000.0, 1.0) == 0.0
    assert metallicity_readiness(100_000.0, 8.0) == pytest.approx(0.0, abs=1e-8)

    R = np.array([R_SUN_LY])
    theta = np.array([0.0])
    arm = spiral_arm_modulation(R, theta, A_arm=0.0)
    assert np.isfinite(arm).all()
    assert np.all(arm >= 1.0)
