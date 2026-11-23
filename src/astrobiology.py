"""Astrobiology priors and named scientific presets."""

from dataclasses import dataclass
import numpy as np


@dataclass
class AstrobiologyPriors:
    f_star: float = 0.135
    w_M: float = 0.5
    eta_e_distro: str = "lognormal"
    eta_e_mean: float = 0.37
    eta_e_sigma: float = 0.55
    f_l_mean: float = 0.10
    f_l_sigma: float = 0.15
    t_evo_mean_gyr: float = 4.2
    t_evo_sigma_gyr: float = 1.0
    f_i: float = 0.05
    f_c: float = 0.05
    L_median_yr: float = 1.0e6
    L_sigma: float = 1.5

    @property
    def f_filter(self):
        return float(np.clip(self.f_i * self.f_c, 0.0, 1.0))

    def validate(self):
        for name, value in (
            ("f_star", self.f_star), ("w_M", self.w_M),
            ("eta_e_mean", self.eta_e_mean), ("f_i", self.f_i),
            ("f_c", self.f_c), ("f_l_mean", self.f_l_mean),
        ):
            if not np.isfinite(value) or value < 0.0 or value > 2.0:
                raise ValueError(f"{name} must be in [0, 2]")
        if not 0.0 <= self.f_star <= 1.0 or not 0.0 <= self.w_M <= 1.0:
            raise ValueError("f_star and w_M must be in [0, 1]")
        if not 0.0 <= self.f_l_mean <= 1.0:
            raise ValueError("f_l_mean must be in [0, 1]")
        if self.t_evo_mean_gyr <= 0.0 or self.t_evo_sigma_gyr < 0.0:
            raise ValueError(
                "t_evo_mean_gyr must be positive and "
                "t_evo_sigma_gyr must be non-negative"
            )
        if self.L_median_yr <= 0.0 or self.L_sigma < 0.0:
            raise ValueError("Longevity parameters must be positive")

    def sample_priors(self, n_runs, rng):
        self.validate()
        rng = np.random.default_rng(rng) if not hasattr(rng, "normal") else rng
        n = int(n_runs)
        eta_e = np.clip(rng.lognormal(np.log(self.eta_e_mean + 1e-12),
                                      self.eta_e_sigma, n), 0.0, 2.0)
        f_l = np.clip(rng.normal(self.f_l_mean, self.f_l_sigma, n), 1e-12, 1.0)
        t_evo = np.clip(rng.lognormal(np.log(self.t_evo_mean_gyr),
                                      self.t_evo_sigma_gyr / self.t_evo_mean_gyr, n),
                        0.0, 100.0)
        L = rng.lognormal(np.log(self.L_median_yr), self.L_sigma, n)
        f_star_eff = np.clip(self.f_star * (1.0 - 0.5 * (1.0 - self.w_M)),
                             0.0, 1.0)
        f_filter = np.full(n, self.f_filter)
        return {
            "f_star": np.full(n, f_star_eff),
            "eta_e": eta_e,
            "f_l": f_l,
            "t_evo_gyr": t_evo,
            "f_filter": f_filter,
            "L_yr": L,
        }


PRESETS = {
    "Baseline (Modern Consensus)": AstrobiologyPriors(),
    "Rare Earth (Strict Great Filters)": AstrobiologyPriors(
        f_l_mean=0.01, f_l_sigma=0.02, f_i=0.001, f_c=0.01,
        t_evo_mean_gyr=5.2, t_evo_sigma_gyr=1.2, L_median_yr=2.0e5,
    ),
    "Copernican Optimist": AstrobiologyPriors(
        f_l_mean=0.5, f_l_sigma=0.2, f_i=0.5, f_c=0.5,
        t_evo_mean_gyr=3.2, t_evo_sigma_gyr=0.7, L_median_yr=1.0e7,
    ),
}


def get_preset(name):
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return PRESETS[name]
