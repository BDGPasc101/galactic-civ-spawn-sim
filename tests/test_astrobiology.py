import numpy as np

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
