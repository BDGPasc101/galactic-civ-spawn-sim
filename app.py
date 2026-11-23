"""Interactive Streamlit observatory for the galactic civilization simulator."""

from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.astrobiology import PRESETS, AstrobiologyPriors, get_preset
from src.galaxy_model import (
    R_LOCAL_LY,
    R_MAX_LY,
    R_SUN_LY,
    T_MW_GYR,
    Z_SUN_LY,
    baseline_stellar_density,
    build_galactic_rate_grid,
    metallicity_readiness,
    spiral_arm_modulation,
    sterilization_survival,
)
from src.monte_carlo import run_monte_carlo, sample_local_coordinates


PRESET_NAMES = (
    "Baseline (Modern Consensus)",
    "Rare Earth (Strict Great Filters)",
    "Copernican Optimist",
    "Custom",
)
TODAY_GYR = T_MW_GYR
HORIZON_LY = R_LOCAL_LY
def preset_parameters(name):
    """Return UI values for one named prior preset."""
    if name == "Custom":
        name = "Baseline (Modern Consensus)"
    priors = get_preset(name)
    return {
        "eta_e_mean": priors.eta_e_mean,
        "w_M": priors.w_M,
        "t_evo_mean_gyr": priors.t_evo_mean_gyr,
        "f_filter": priors.f_filter,
        "L_median_yr": priors.L_median_yr,
        "A_arm": 0.35,
    }


def make_priors(name, eta_e_mean, w_M, t_evo_mean_gyr, f_filter,
                L_median_yr):
    """Build the Phase 1 prior object from the active scalar controls."""
    base = get_preset(name) if name != "Custom" else PRESETS[
        "Baseline (Modern Consensus)"
    ]
    filter_factor = float(np.sqrt(np.clip(f_filter, 0.0, 1.0)))
    return replace(
        base,
        eta_e_mean=float(eta_e_mean),
        w_M=float(w_M),
        t_evo_mean_gyr=float(t_evo_mean_gyr),
        f_i=filter_factor,
        f_c=filter_factor,
        L_median_yr=float(L_median_yr),
    )


@st.cache_data(show_spinner=False)
def cached_galactic_grid(A_arm):
    """Cache the spatial-temporal grid by its scalar arm-contrast input."""
    return build_galactic_rate_grid(A_arm=float(A_arm))


@st.cache_data(show_spinner=False)
def sample_representative_locations(n_events, seed, A_arm):
    """Draw local map positions from the active model's habitability-weighted grid."""
    grid = cached_galactic_grid(float(A_arm))
    return sample_local_coordinates(grid, int(n_events), seed=int(seed))


@st.cache_data(show_spinner=False)
def cached_simulation(n_runs, seed, eta_e_mean, w_M, t_evo_mean_gyr,
                      f_filter, L_median_yr, A_arm):
    """Run and cache Monte Carlo using only primitive cache-key arguments."""
    priors = make_priors(
        "Custom", eta_e_mean, w_M, t_evo_mean_gyr, f_filter, L_median_yr
    )
    return run_monte_carlo(
        n_runs=int(n_runs),
        priors=priors,
        seed=int(seed),
        grid=cached_galactic_grid(float(A_arm)),
    )


def sample_representative_events(n_events, priors, seed, distances_ly=None):
    """Draw illustrative event times and longevity from active priors."""
    rng = np.random.default_rng(seed)
    sampled = priors.sample_priors(int(n_events), rng)
    latest_birth = np.maximum(0.0, TODAY_GYR - sampled["t_evo_gyr"])
    birth = rng.uniform(0.0, latest_birth)
    emergence = birth + sampled["t_evo_gyr"]
    end = emergence + sampled["L_yr"] / 1.0e9
    if distances_ly is None:
        distances_ly = HORIZON_LY * rng.random(int(n_events)) ** (1.0 / 3.0)
    distances_ly = np.asarray(distances_ly, dtype=float)
    causal = (
        (emergence + distances_ly / 1.0e9 <= TODAY_GYR)
        & (TODAY_GYR <= end + distances_ly / 1.0e9)
    )
    active = (emergence <= TODAY_GYR) & (TODAY_GYR <= end)
    return {
        "birth_gyr": birth,
        "t_evo_gyr": sampled["t_evo_gyr"],
        "emergence_gyr": emergence,
        "end_gyr": end,
        "L_yr": sampled["L_yr"],
        "distance_ly": distances_ly,
        "causal": causal,
        "active": active,
    }


def build_galaxy_map_figure(
    coords_sample, priors, seed=42,
    view_mode="Full Milky Way (100,000 ly)", A_arm=0.35,
):
    """Build the habitability-weighted face-on map and representative overlays."""
    resolution = 230
    extent = R_MAX_LY
    axis = np.linspace(-extent, extent, resolution)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    backdrop = (
        baseline_stellar_density(radius, 0.0)
        * spiral_arm_modulation(radius, theta, A_arm=A_arm)
        * metallicity_readiness(radius, 9.0)
        * sterilization_survival(radius, theta, 9.0)
    )
    backdrop = np.where(radius <= R_MAX_LY, backdrop, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=axis,
        y=axis,
        z=backdrop,
        colorscale="Inferno",
        zmin=0.0,
        zmax=float(np.nanpercentile(backdrop, 99.5)),
        colorbar={
            "title": {"text": "Habitability<br>weight",
                      "font": {"color": "#D8D8E5"}},
            "tickfont": {"color": "#D8D8E5"},
        },
        hovertemplate="Galactic x: %{x:.0f} ly<br>y: %{y:.0f} ly"
                      "<br>Relative habitability: %{z:.3g}<extra></extra>",
        showscale=True,
    ))

    coords = np.asarray(coords_sample, dtype=float)
    if coords.size:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords_sample must have shape (n, 3)")
        if len(coords) > 500:
            selection = np.random.default_rng(seed).choice(
                len(coords), size=500, replace=False
            )
            coords = coords[selection]
        distances = np.sqrt(
            (coords[:, 0] - R_SUN_LY) ** 2
            + coords[:, 1] ** 2
            + (coords[:, 2] - Z_SUN_LY) ** 2
        )
        events = sample_representative_events(
            len(coords), priors, seed, distances_ly=distances
        )
        status_colors = {
            "Causal signal now": "#43F5E3",
            "Active now": "#8DFFB4",
            "Past extinct": "#D88B8B",
        }
        statuses = []
        for causal, active in zip(events["causal"], events["active"]):
            statuses.append(
                "Causal signal now" if causal else
                "Active now" if active else "Past extinct"
            )
        emergence_age = TODAY_GYR - events["emergence_gyr"]
        for status, color in status_colors.items():
            mask = np.asarray(statuses) == status
            if not np.any(mask):
                continue
            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=status,
                marker={
                    "size": 7,
                    "color": color,
                    "line": {"color": "#F8F8FF", "width": 0.5},
                    "opacity": 0.9,
                },
                customdata=np.column_stack((
                    distances[mask],
                    emergence_age[mask],
                    events["L_yr"][mask],
                )),
                hovertemplate=(
                    f"{status}<br>Distance from Earth: %{{customdata[0]:,.0f}} ly"
                    "<br>Emergence age: %{customdata[1]:.2f} Gyr ago"
                    "<br>Sampled longevity: %{customdata[2]:,.0f} yr"
                    "<extra></extra>"
                ),
            ))

    fig.add_trace(go.Scatter(
        x=[R_SUN_LY],
        y=[0],
        mode="markers",
        name="Sun / Earth",
        marker={
            "size": 13,
            "color": "#FFF4B0",
            "symbol": "star",
            "line": {"color": "#FFFFFF", "width": 1.5},
        },
        hovertemplate="Sun / Earth<br>Galactic radius: 26,750 ly<extra></extra>",
    ))
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    fig.add_trace(go.Scatter(
        x=R_SUN_LY + HORIZON_LY * np.cos(angle),
        y=HORIZON_LY * np.sin(angle),
        mode="lines",
        name="10,000 ly Solar Neighborhood Horizon",
        line={"color": "#36DDF5", "width": 1.7, "dash": "dash"},
        hoverinfo="skip",
    ))

    if view_mode == "Zoom: 10,000 ly Solar Neighborhood":
        fig.update_xaxes(range=[R_SUN_LY - HORIZON_LY, R_SUN_LY + HORIZON_LY])
        fig.update_yaxes(range=[-HORIZON_LY, HORIZON_LY])
    else:
        fig.update_xaxes(range=[-R_MAX_LY, R_MAX_LY])
        fig.update_yaxes(range=[-R_MAX_LY, R_MAX_LY])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#05050A",
        plot_bgcolor="#05050A",
        height=650,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        title="Four-arm logarithmic disk · habitability-weighted density",
        xaxis_title="Galactic x (ly)",
        yaxis_title="Galactic y (ly)",
        legend={"bgcolor": "rgba(5,5,10,0.8)", "font": {"color": "#E8E8F0"}},
        annotations=[
            {
                "x": 0.5, "y": 0.56, "xref": "x", "yref": "y",
                "text": "Radiation-sterilized core", "showarrow": False,
                "font": {"color": "#FFB28B", "size": 10},
            },
            {
                "x": -38_000, "y": 0, "xref": "x", "yref": "y",
                "text": "Metal-poor outer rim", "showarrow": False,
                "font": {"color": "#D3C9C0", "size": 10},
            },
        ],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, gridcolor="#20202A")
    fig.update_xaxes(gridcolor="#20202A")
    return fig


def add_interval_lines(fig, values, row, col, color):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return
    p16, median, p84 = np.percentile(values, [16, 50, 84])
    for value, label, dash in (
        (p16, f"16th: {p16:.2f}", "dot"),
        (median, f"Median: {median:.2f}", "solid"),
        (p84, f"84th: {p84:.2f}", "dot"),
    ):
        fig.add_vline(
            x=float(value),
            row=row,
            col=col,
            line={"color": color, "dash": dash, "width": 1.5},
            annotation_text=label,
            annotation_font_color=color,
        )


def build_posterior_figure(local_counts, active_counts):
    """Build side-by-side local spawn and present-day Galactic posteriors."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Local spawns within 10,000 ly",
            "Active Milky Way civilizations now",
        ),
    )
    local_counts = np.asarray(local_counts)
    active_counts = np.asarray(active_counts)
    fig.add_trace(go.Histogram(
        x=local_counts, nbinsx=35, marker_color="#F69A50",
        name="Local spawns", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=active_counts, nbinsx=35, marker_color="#42DDEB",
        name="Active Milky Way", showlegend=False,
    ), row=1, col=2)
    add_interval_lines(fig, local_counts, 1, 1, "#FFE0A6")
    add_interval_lines(fig, active_counts, 1, 2, "#A4F5FA")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#05050A",
        plot_bgcolor="#090912",
        height=420,
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        bargap=0.08,
    )
    fig.update_xaxes(title_text="Civilizations per realization")
    fig.update_yaxes(title_text="Monte Carlo realizations")
    return fig


def build_timeline_figure(priors, seed=42, n_events=35):
    """Build a reproducible representative timeline sampled from active priors."""
    events = sample_representative_events(n_events, priors, seed)
    fig = go.Figure()
    labels = []
    for causal, active in zip(events["causal"], events["active"]):
        labels.append(
            "Causal signal now" if causal else
            "Active now" if active else "Past extinct"
        )
    color_map = {
        "Causal signal now": "#43F5E3",
        "Active now": "#8DFFB4",
        "Past extinct": "#D88B8B",
    }
    for i, label in enumerate(labels):
        fig.add_trace(go.Bar(
            x=[max(events["end_gyr"][i] - events["emergence_gyr"][i], 1e-5)],
            y=[f"Representative twin {i + 1:02d}"],
            base=[events["emergence_gyr"][i]],
            orientation="h",
            marker_color=color_map[label],
            opacity=0.84,
            name=label,
            legendgroup=label,
            showlegend=label not in labels[:i],
            customdata=[[
                events["emergence_gyr"][i],
                events["end_gyr"][i],
                events["distance_ly"][i],
                events["L_yr"][i],
            ]],
            hovertemplate=(
                f"{label}<br>Emergence: %{{customdata[0]:.2f}} Gyr"
                "<br>Extinction: %{customdata[1]:.3f} Gyr"
                "<br>Distance: %{customdata[2]:,.0f} ly"
                "<br>Sampled longevity: %{customdata[3]:,.0f} yr"
                "<extra></extra>"
            ),
        ))
    fig.add_vline(
        x=TODAY_GYR,
        line={"color": "#FFFFFF", "width": 2, "dash": "dash"},
        annotation_text="Today · 13.6 Gyr",
        annotation_position="top",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#05050A",
        plot_bgcolor="#090912",
        title="Ships Passing in the Night · representative prior draws",
        xaxis_title="Time since Milky Way formation (Gyr)",
        yaxis_title="",
        xaxis={"range": [0, TODAY_GYR], "gridcolor": "#252530"},
        height=620,
        margin={"l": 20, "r": 20, "t": 65, "b": 35},
        barmode="overlay",
    )
    return fig


@st.cache_data(show_spinner=False)
def compute_sensitivity(seed, eta_e_mean, w_M, t_evo_mean_gyr,
                        f_filter, L_median_yr, A_arm):
    """Run a compact, deterministic one-factor-at-a-time sensitivity sweep."""
    active = {
        "L": L_median_yr,
        "Great filter": f_filter,
        "Evolution clock": t_evo_mean_gyr,
        "Rocky planet rate": eta_e_mean,
        "M-dwarf weight": w_M,
    }
    bounds = {
        "L": (100.0, 10_000_000.0),
        "Great filter": (1.0e-8, 1.0),
        "Evolution clock": (1.0, 6.0),
        "Rocky planet rate": (0.1, 1.0),
        "M-dwarf weight": (0.0, 1.0),
    }
    base_values = {
        "eta_e_mean": eta_e_mean,
        "w_M": w_M,
        "t_evo_mean_gyr": t_evo_mean_gyr,
        "f_filter": f_filter,
        "L_median_yr": L_median_yr,
    }
    grid = cached_galactic_grid(float(A_arm))
    low_delta = []
    high_delta = []
    base_priors = make_priors(
        "Custom", eta_e_mean, w_M, t_evo_mean_gyr, f_filter, L_median_yr
    )
    baseline = run_monte_carlo(
        n_runs=300, priors=base_priors, seed=int(seed), grid=grid
    ).prob_now_10kly
    control_to_key = {
        "L": "L_median_yr",
        "Great filter": "f_filter",
        "Evolution clock": "t_evo_mean_gyr",
        "Rocky planet rate": "eta_e_mean",
        "M-dwarf weight": "w_M",
    }
    for index, name in enumerate(active):
        low_values = dict(base_values)
        high_values = dict(base_values)
        parameter_key = control_to_key[name]
        low_values[parameter_key], high_values[parameter_key] = bounds[name]
        low_result = run_monte_carlo(
            n_runs=300,
            priors=make_priors("Custom", **{
                "eta_e_mean": low_values["eta_e_mean"],
                "w_M": low_values["w_M"],
                "t_evo_mean_gyr": low_values["t_evo_mean_gyr"],
                "f_filter": low_values["f_filter"],
                "L_median_yr": low_values["L_median_yr"],
            }),
            seed=int(seed) + 2 * index + 1,
            grid=grid,
        ).prob_now_10kly
        high_result = run_monte_carlo(
            n_runs=300,
            priors=make_priors("Custom", **{
                "eta_e_mean": high_values["eta_e_mean"],
                "w_M": high_values["w_M"],
                "t_evo_mean_gyr": high_values["t_evo_mean_gyr"],
                "f_filter": high_values["f_filter"],
                "L_median_yr": high_values["L_median_yr"],
            }),
            seed=int(seed) + 2 * index + 2,
            grid=grid,
        ).prob_now_10kly
        low_delta.append(low_result - baseline)
        high_delta.append(high_result - baseline)
    return list(active), np.asarray(low_delta), np.asarray(high_delta), baseline


def build_sensitivity_figure(names, low_delta, high_delta):
    """Build a tornado chart of low/high shifts in local active-twin probability."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=low_delta,
        orientation="h",
        name="Low literature bound",
        marker_color="#E68B62",
        hovertemplate="%{y}<br>Shift: %{x:+.2f} percentage points<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=names,
        x=high_delta,
        orientation="h",
        name="High literature bound",
        marker_color="#55D8D1",
        hovertemplate="%{y}<br>Shift: %{x:+.2f} percentage points<extra></extra>",
    ))
    fig.add_vline(x=0, line={"color": "#FFFFFF", "width": 1})
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#05050A",
        plot_bgcolor="#090912",
        barmode="overlay",
        title="Shift in P(Twin Alive Now ≤ 10,000 ly) from active baseline",
        xaxis_title="Probability shift (percentage points)",
        yaxis_title="Parameter",
        height=400,
        margin={"l": 20, "r": 20, "t": 70, "b": 35},
    )
    return fig


def _switch_to_custom():
    st.session_state["scenario_preset"] = "Custom"


def _format_summary(summary, suffix=""):
    median = summary["median"]
    p16 = summary["p16"]
    p84 = summary["p84"]
    if not np.isfinite(median):
        return "—"
    return f"{median:,.2f}{suffix} <small>({p16:,.2f}–{p84:,.2f})</small>"


def _render_kpis(result):
    mw = result.mw_now_summary
    nearest = result.nearest_summary
    if np.isfinite(nearest["median"]):
        nearest_text = f"{nearest['median']:,.0f} ly"
    elif mw["median"] == 0:
        nearest_text = "None active in MW"
    else:
        nearest_text = "No nearest distance estimated"
    cards = [
        ("P(Twin Ever) ≤ 10,000 ly", result.prob_ever_10kly),
        ("P(Twin Alive Now) ≤ 10,000 ly", result.prob_now_10kly),
        ("P(Causal Light-Cone) ≤ 10,000 ly", result.prob_causal_10kly),
    ]
    cols = st.columns(3)
    for col, (title, value) in zip(cols, cards):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{title}</div>'
            f'<div class="kpi-value">{value:.2f}%</div></div>',
            unsafe_allow_html=True,
        )
    cols = st.columns(2)
    cols[0].markdown(
        '<div class="kpi-card"><div class="kpi-label">'
        'Milky Way active twins now · median (16th–84th percentile)</div>'
        f'<div class="kpi-value">{_format_summary(mw)}</div></div>',
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        '<div class="kpi-card"><div class="kpi-label">'
        'Median nearest active neighbor</div>'
        f'<div class="kpi-value">{nearest_text}</div></div>',
        unsafe_allow_html=True,
    )


def _render_verdict(result):
    if result.prob_now_10kly >= 50.0:
        verdict = (
            "A nearby civilization active at the same time is favored under "
            "these assumptions; a local SETI search has substantial target "
            "probability."
        )
    elif result.prob_ever_10kly >= 50.0:
        verdict = (
            "A local technological species is plausible across Galactic "
            "history, but short lifetimes make contemporaneous overlap less "
            "likely. Search persistent technosignatures as well as beacons."
        )
    else:
        verdict = (
            "The active model assigns low probability to a local technological "
            "twin. This is conditional on the selected priors, not evidence "
            "that the Solar Neighborhood is uninhabited."
        )
    st.markdown(
        f'<div class="verdict"><strong>Applied Theorem Verdict</strong><br>{verdict}'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_sidebar():
    st.sidebar.markdown("## SETI Observatory Controls")
    preset = st.sidebar.selectbox(
        "Scenario preset",
        PRESET_NAMES,
        key="scenario_preset",
    )
    if "last_preset" not in st.session_state:
        st.session_state["last_preset"] = preset
        for key, value in preset_parameters(preset).items():
            state_key = "f_filter_log10" if key == "f_filter" else key
            state_value = np.log10(max(value, 1.0e-8)) if key == "f_filter" else value
            st.session_state.setdefault(state_key, state_value)
    elif preset != st.session_state["last_preset"]:
        if preset != "Custom":
            for key, value in preset_parameters(preset).items():
                state_key = "f_filter_log10" if key == "f_filter" else key
                st.session_state[state_key] = (
                    np.log10(max(value, 1.0e-8))
                    if key == "f_filter" else value
                )
        st.session_state["last_preset"] = preset

    resolution = st.sidebar.radio(
        "Monte Carlo resolution",
        (1_000, 10_000),
        index=1,
        format_func=lambda value: (
            "1,000 (Fast Preview)" if value == 1_000
            else "10,000 (Full Monte Carlo Run)"
        ),
    )
    seed = st.sidebar.number_input(
        "Integer random seed", min_value=0, max_value=2_147_483_647,
        value=42, step=1,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Tunable astrobiological and Galactic parameters")
    eta = st.sidebar.slider(
        "Rocky habitable-zone planet rate (η⊕)",
        0.0, 2.0, key="eta_e_mean", step=0.01,
        on_change=_switch_to_custom,
    )
    weight = st.sidebar.slider(
        "M-dwarf habitability weight (wM)",
        0.0, 1.0, key="w_M", step=0.01,
        on_change=_switch_to_custom,
    )
    t_evo = st.sidebar.slider(
        "Mean evolutionary gestation (Gyr)",
        0.0, 10.0, key="t_evo_mean_gyr", step=0.1,
        on_change=_switch_to_custom,
    )
    f_filter = st.sidebar.slider(
        "Great Filter survival probability · log₁₀(fᵢ · f꜀)",
        min_value=-8.0, max_value=0.0,
        key="f_filter_log10", step=0.01,
        on_change=_switch_to_custom,
    )
    f_filter = 10.0 ** f_filter
    longevity = st.sidebar.slider(
        "Median technological longevity (years)",
        min_value=100.0, max_value=10_000_000.0,
        key="L_median_yr", format="%.0f", step=None,
        on_change=_switch_to_custom,
    )
    arm = st.sidebar.slider(
        "Spiral arm density contrast (Aarm)",
        0.0, 1.0, key="A_arm", step=0.01,
        on_change=_switch_to_custom,
    )
    return preset, int(resolution), int(seed), eta, weight, t_evo, f_filter, longevity, arm


def main():
    st.set_page_config(
        page_title="SETI Observatory · Galactic Civilization Simulator",
        page_icon="✦",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(ellipse at 50% 0%, #111126 0%, #05050A 58%); color: #E8E8F0; }
        [data-testid="stSidebar"] { background: #090912; border-right: 1px solid #252536; }
        .kpi-card { background: linear-gradient(135deg,#141424,#0B0B14); border: 1px solid #35354A; border-radius: 12px; padding: 18px 20px; min-height: 100px; margin-bottom: 12px; box-shadow: 0 8px 26px #0007; }
        .kpi-label { color: #A7A7BC; font-size: .88rem; letter-spacing: .03em; }
        .kpi-value { color: #F7F7FF; font-size: 1.55rem; font-weight: 700; margin-top: 7px; }
        .kpi-value small { color: #A7D9DC; font-size: .72rem; font-weight: 500; }
        .verdict { background: linear-gradient(100deg,#12383D,#182136); border-left: 4px solid #43E4DA; border-radius: 8px; padding: 16px 20px; margin: 8px 0 22px; color: #E8F8F7; }
        h1, h2, h3 { letter-spacing: .02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("✦ SETI Observatory")
    st.caption(
        "A spatio-temporal view of technological life in the Milky Way · "
        "probabilities are conditional on the selected priors."
    )
    controls = _render_sidebar()
    preset, n_runs, seed, eta, weight, t_evo, f_filter, longevity, arm = controls
    priors = make_priors(
        preset, eta, weight, t_evo, f_filter, longevity
    )
    with st.spinner(f"Simulating {n_runs:,} Galactic realizations…"):
        result = cached_simulation(
            n_runs, seed, eta, weight, t_evo, f_filter, longevity, arm
        )

    _render_kpis(result)
    st.caption(
        "A 'twin' denotes the model's assumed technological civilization "
        "(approximately Kardashev 0.73+)."
    )
    _render_verdict(result)
    tab_map, tab_posteriors, tab_theorem = st.tabs([
        "Milky Way Map & Local Horizon",
        "Posteriors & Ships Passing in the Night",
        "Great Filter Sensitivity & Theorem",
    ])
    with tab_map:
        map_coords = result.coords_sample
        if map_coords.size == 0:
            map_coords = sample_representative_locations(120, seed, arm)
        map_mode = st.radio(
            "Map view",
            ("Full Milky Way (100,000 ly)", "Zoom: 10,000 ly Solar Neighborhood"),
            horizontal=True,
        )
        st.plotly_chart(
            build_galaxy_map_figure(
                map_coords, priors, seed, map_mode, A_arm=arm
            ),
            use_container_width=True,
        )
        st.caption(
            "Representative realization drawn from active priors. Map positions "
            "(when the Monte Carlo sample is empty), status, and event metadata "
            "are illustrative—not per-civilization timestamps returned by the "
            "Phase 1 Monte Carlo interface."
        )
    with tab_posteriors:
        st.plotly_chart(
            build_posterior_figure(result.K_local, result.N_mw_now),
            use_container_width=True,
        )
        st.plotly_chart(
            build_timeline_figure(priors, seed),
            use_container_width=True,
        )
        st.caption(
            "Representative realization drawn from active priors. Timeline "
            "emergence and extinction markers are seeded illustrative samples, "
            "not tracked events from the Monte Carlo output."
        )
    with tab_theorem:
        with st.spinner("Running compact sensitivity sweep…"):
            sensitivity = compute_sensitivity(
                seed, eta, weight, t_evo, f_filter, longevity, arm
            )
        names, low_delta, high_delta, baseline = sensitivity
        st.metric(
            "Fast-sweep baseline · P(Twin Alive Now ≤ 10,000 ly)",
            f"{baseline:.2f}%",
        )
        st.plotly_chart(
            build_sensitivity_figure(names, low_delta, high_delta),
            use_container_width=True,
        )
        st.caption(
            "One-factor-at-a-time estimates use 300 seeded realizations per "
            "bound; they are a directional sensitivity diagnostic."
        )
        with st.expander("Applied theorem · 4D non-homogeneous Poisson process"):
            st.markdown(
                r"""
                For a realization with sampled priors, the expected number of
                technological civilizations is

                \[
                \Lambda_i =
                \int_{0}^{T_{\rm MW}}\!\!dt_{\rm birth}
                \int_{V}\!\!dR\,d\theta\,dz\;R\,
                \dot{\rho}_{\star}(R,\theta,z,t_{\rm birth})\,
                f_{\star,i}\,\eta_{\oplus,i}\,f_{l,i}\,f_{{\rm filter},i}\,
                \mathbf{1}[t_{\rm birth}+t_{{\rm evo},i}\le T_{\rm MW}],
                \qquad K_i\sim{\rm Poisson}(\Lambda_i).
                \]

                The local probability is evaluated inside the 10,000 ly
                Solar Neighborhood. A civilization is active now if
                \(t_{\rm emerge}\le T_{\rm MW}\le t_{\rm emerge}+L/10^9\).
                A causal signal intersects Earth now if
                \(t_{\rm emerge}+d/(10^9\,{\rm ly/Gyr})\le T_{\rm MW}
                \le t_{\rm end}+d/(10^9\,{\rm ly/Gyr})\).
                """
            )
        with st.expander("Key assumptions"):
            st.markdown(
                "- Galactocentric structure is represented by a 4-arm logarithmic disk with a smooth metallicity and sterilization model.\n"
                "- Each realization samples the abiogenesis, evolutionary, filter, and longevity priors independently as implemented in Phase 1.\n"
                "- The present-day whole-Galaxy count follows the Phase 1 Poisson-thinning approximation; local overlap and causal-signal probabilities use its local sampled population.\n"
                "- The sensitivity chart varies one parameter at a time over broad literature-motivated bounds; it is not a joint posterior."
            )
        with st.expander("Literature grounding"):
            st.markdown(
                "- **Gaia DR3** — Milky Way stellar structure and solar-neighborhood context.\n"
                "- **Lineweaver, Fenner & Gibson (2004)** — Galactic Habitable Zone and its chemical-evolution history.\n"
                "- **Bryson et al. (2021)** — occurrence rates of rocky habitable-zone planets from Kepler data.\n"
                "- **Kipping (2020)** — Bayesian constraints on abiogenesis and life's emergence.\n"
                "- **Carter (1983)** — hard-step framing for evolutionary timescales."
            )


if __name__ == "__main__":
    main()
