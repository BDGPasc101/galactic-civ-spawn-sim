import numpy as np
import pytest

from src.astrobiology import AstrobiologyPriors, PRESETS


def test_prior_sampling_is_bounded_and_deterministic():
    priors = AstrobiologyPriors()
    a = priors.sample_priors(1000, np.random.default_rng(123))
    b = priors.sample_priors(1000, np.random.default_rng(123))
    for key in a:
        assert np.array_equal(a[key], b[key])
        assert np.isfinite(a[key]).all()
    assert np.all((a["f_l"] > 0.0) & (a["f_l"] <= 1.0))
    assert np.all((a["f_filter"] >= 0.0) & (a["f_filter"] <= 1.0))
    assert np.all((a["eta_e"] >= 0.0) & (a["eta_e"] <= 2.0))


def test_preset_ordering():
    rare = PRESETS["Rare Earth (Strict Great Filters)"]
    baseline = PRESETS["Baseline (Modern Consensus)"]
    optimistic = PRESETS["Copernican Optimist"]
    assert rare.f_filter < baseline.f_filter < optimistic.f_filter
    assert rare.f_l_mean < baseline.f_l_mean < optimistic.f_l_mean


def test_evolution_clock_validation_requires_positive_mean():
    with pytest.raises(ValueError, match="t_evo_mean_gyr"):
        AstrobiologyPriors(t_evo_mean_gyr=0.0).validate()


def test_zero_m_dwarf_weight_zeroes_effective_stellar_fraction():
    priors = AstrobiologyPriors(w_M=0.0)
    sampled = priors.sample_priors(100, np.random.default_rng(1))
    assert np.all(sampled["f_star"] == 0.5 * priors.f_star)


def test_single_run_sampling_is_finite():
    priors = AstrobiologyPriors(t_evo_mean_gyr=13.59, L_median_yr=1.0)
    sampled = priors.sample_priors(1, np.random.default_rng(123))
    assert all(np.isfinite(values).all() for values in sampled.values())
