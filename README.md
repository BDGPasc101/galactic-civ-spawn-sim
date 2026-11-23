# Galactic Civilization Spawn Simulator

This repository models technological civilizations as a non-homogeneous Poisson
process over the Galactic disk.

## Galactic structure

The thin disk uses

\[
\Sigma(R,z)\propto
\exp(-R/R_d)\operatorname{sech}^{2}(z/(2h_z)),
\]

with \(R_d=8500\) ly and \(h_z=980\) ly. The model includes a 4-arm logarithmic
spiral with \(12.5^\circ\) pitch angle and an Orion-Cygnus spur. Constants use
\(R_{\odot}=26750\) ly, \(\theta_\odot=0\), and \(z_\odot=65\) ly. The Milky Way
age is \(T_{\rm MW}=13.6\) Gyr.

The Galactic Habitable Zone combines smooth metallicity readiness,
\(f_{\rm metal}(R,t_{\rm birth})\), with sterilization survival

\[
S_{\rm SN}=\exp[-\Gamma_{\rm SN}(R,\theta,t_{\rm birth})\tau_{\rm bio}].
\]

Volumes use the cylindrical Jacobian \(R\,dR\,d\theta\,dz\). The local mask
includes voxels satisfying
\(|\mathbf r-\mathbf r_\odot|\le10000\) ly.

## 4D Spatio-Temporal Drake equation

For Monte Carlo run \(i\),

\[
\Lambda_i =
\int_V dt_{\rm birth}\,dV\,
\dot R_\star(R,\theta,z,t_{\rm birth})\,
f_{\rm star,i}\,\eta_{\oplus,i}\,f_{l,i}\,f_{\rm filter,i}\,
\mathbf 1[t_{\rm birth}+t_{\rm evo,i}\le T_{\rm MW}],
\]

and
\(K_i\sim{\rm Poisson}(\Lambda_i)\). Separate integrations produce local and
whole-Milky-Way counts.

Unit conversion is explicit: 1 Gyr \(=10^9\) yr, and a light travel distance
\(d\) ly corresponds to \(d/c=d/10^9\) Gyr. A civilization is active now when

\[
t_{\rm emerge}\le T_{\rm MW}\le t_{\rm emerge}+L_{\rm yr}/10^9.
\]

The expected active-now population integrates the same birth process over the
overlap window
\[
\max(0,T_{\rm MW}-t_{\rm evo}-L_{\rm yr}/10^9)
\le t_{\rm birth}\le T_{\rm MW}-t_{\rm evo}.
\]
The local and whole-Galaxy active intensities use this identical time window;
the local intensity applies the 10,000 ly mask. The reported nearest-neighbor
median is unconditional across realizations, with realizations containing no
active Galaxy-wide civilization represented internally by an infinite nearest
distance and summarized as unavailable when the median active count is zero.

Its radio light-cone is present when

\[
t_{\rm emerge}+d/c\le T_{\rm MW}\le t_{\rm end}+d/c.
\]

This shifted window is not generally a superset of simultaneous Galactic
existence; it is a past-light-cone window of width \(L_{\rm yr}/10^9\) Gyr.

## Priors and literature

- Gaia DR3 and Bland-Hawthorn & Gerhard (2016): Galactic structure and solar
  neighborhood.
- Lineweaver, Fenner & Gibson (2004): Galactic Habitable Zone.
- Kepler DR25 and Bryson et al. (2021): \(\eta_\oplus\) distribution.
- Kipping (2020): abiogenesis Bayesian inference.
- Carter's hard-step model: evolutionary gestation-time priors.
- A Kroupa/Chabrier-inspired \(0.5\)-\(1.3M_\odot\) FGK window is represented
  by \(f_{\rm star}\), with an adjustable M-dwarf weight.

## Usage

```bash
pip install -r requirements.txt
streamlit run app.py
pytest
python -c "from src.monte_carlo import run_monte_carlo; r=run_monte_carlo(); print(r.prob_ever_10kly, r.prob_now_10kly)"
```

The Streamlit SETI Observatory provides scenario controls, cached Monte Carlo
summaries, a habitability-weighted Milky Way map, posterior distributions, and a
parameter sensitivity view. Map status metadata and the cosmic event timeline
are deterministic representative samples drawn from the active astrobiology
priors (including the selected random seed, evolutionary clock, and longevity);
they are illustrative because Phase 1 does not expose per-civilization event
timestamps. The reported probability and count metrics are from the Phase 1
simulation result.

Run the full test suite with `pytest` after launching or changing the
observatory.
