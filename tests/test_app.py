import subprocess
import sys

import numpy as np

from app import (
    build_galaxy_map_figure,
    build_posterior_figure,
    build_sensitivity_figure,
    build_timeline_figure,
    make_priors,
    preset_parameters,
    sample_representative_locations,
    sample_representative_events,
)
from src.astrobiology import PRESETS


def test_preset_parameter_mapping_and_filter_override():
    for name in PRESETS:
        mapped = preset_parameters(name)
        assert mapped["eta_e_mean"] == PRESETS[name].eta_e_mean
        assert mapped["w_M"] == PRESETS[name].w_M
        assert mapped["t_evo_mean_gyr"] == PRESETS[name].t_evo_mean_gyr
        assert mapped["f_filter"] == PRESETS[name].f_filter
        assert mapped["L_median_yr"] == PRESETS[name].L_median_yr

    priors = make_priors(
        "Custom", eta_e_mean=0.8, w_M=0.25, t_evo_mean_gyr=3.5,
        f_filter=0.04, L_median_yr=50_000,
    )
    assert priors.eta_e_mean == 0.8
    assert priors.w_M == 0.25
    assert priors.t_evo_mean_gyr == 3.5
    assert np.isclose(priors.f_filter, 0.04)
    assert priors.L_median_yr == 50_000


def test_representative_events_are_seeded_and_use_active_priors():
    priors = make_priors(
        "Custom", eta_e_mean=0.7, w_M=0.4, t_evo_mean_gyr=3.3,
        f_filter=0.02, L_median_yr=120_000,
    )
    first = sample_representative_events(30, priors, 17)
    second = sample_representative_events(30, priors, 17)
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert np.all(first["emergence_gyr"] >= first["t_evo_gyr"])
    assert np.all(first["L_yr"] > 0)
    assert np.all(first["distance_ly"] <= 10_000)


def test_representative_map_locations_are_seeded_and_local():
    first = sample_representative_locations(40, seed=23, A_arm=0.35)
    second = sample_representative_locations(40, seed=23, A_arm=0.35)
    assert np.array_equal(first, second)
    distance = np.sqrt(
        (first[:, 0] - 26_750.0) ** 2
        + first[:, 1] ** 2
        + (first[:, 2] - 65.0) ** 2
    )
    radii = np.hypot(first[:, 0], first[:, 1])
    angles = np.arctan2(first[:, 1], first[:, 0])
    assert np.all(distance <= 10_000)
    assert np.unique(radii).size == radii.size
    assert np.unique(angles).size == angles.size


def test_plotly_figure_builders():
    priors = PRESETS["Baseline (Modern Consensus)"]
    coords = np.array([[26_750.0, 0.0, 65.0], [25_000.0, 500.0, 0.0]])
    figures = [
        build_galaxy_map_figure(coords, priors, seed=8),
        build_posterior_figure([0, 1, 1, 2], [0, 2, 3, 5]),
        build_timeline_figure(priors, seed=8, n_events=5),
        build_sensitivity_figure(
            ["Longevity", "Great filter"], [-1.0, -0.5], [2.0, 1.5]
        ),
    ]
    assert all(figure.data for figure in figures)


def test_app_imports_in_headless_python():
    completed = subprocess.run(
        [sys.executable, "-c", "import app"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_representative_events_use_exact_light_cone_formula():
    priors = make_priors(
        "Custom", eta_e_mean=0.8, w_M=0.0, t_evo_mean_gyr=13.59,
        f_filter=0.04, L_median_yr=1.0,
    )
    events = sample_representative_events(
        100, priors, 91, distances_ly=np.linspace(0.0, 10_000.0, 100)
    )
    today = 13.6
    expected_causal = (
        (events["emergence_gyr"] + events["distance_ly"] / 1.0e9 <= today)
        & (today <= events["end_gyr"] + events["distance_ly"] / 1.0e9)
    )
    expected_active = (
        (events["emergence_gyr"] <= today)
        & (today <= events["end_gyr"])
    )
    assert np.array_equal(events["causal"], expected_causal)
    assert np.array_equal(events["active"], expected_active)


def test_app_figures_are_stable_at_extreme_values():
    priors = make_priors(
        "Custom", eta_e_mean=0.0, w_M=0.0, t_evo_mean_gyr=13.59,
        f_filter=0.0, L_median_yr=1.0,
    )
    coords = np.array([[26_750.0, 0.0, 65.0]])
    figures = [
        build_galaxy_map_figure(coords, priors, seed=8, A_arm=0.0),
        build_posterior_figure(np.zeros(1, dtype=int), np.zeros(1, dtype=int)),
        build_timeline_figure(priors, seed=8, n_events=1),
        build_sensitivity_figure(["Longevity"], [0.0], [0.0]),
    ]
    assert all(figure.data for figure in figures)
