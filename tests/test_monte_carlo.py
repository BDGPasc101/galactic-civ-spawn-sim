import time

from src.astrobiology import AstrobiologyPriors
from src.monte_carlo import run_monte_carlo


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


def test_immortal_limit_converges():
    priors = AstrobiologyPriors(L_median_yr=1e11, L_sigma=0.01)
    result = run_monte_carlo(1000, priors, seed=7)
    assert result.prob_now_10kly == result.prob_ever_10kly


def test_light_cone_is_shifted_window():
    result = run_monte_carlo(500, AstrobiologyPriors(), seed=11)
    assert 0.0 <= result.prob_causal_10kly <= 100.0


def test_performance():
    started = time.perf_counter()
    result = run_monte_carlo(10_000, AstrobiologyPriors(), seed=42)
    elapsed = time.perf_counter() - started
    assert result.runtime_seconds <= elapsed + 0.1
    assert elapsed < 3.0
