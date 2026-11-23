"""Milky Way structure and Galactic Habitable Zone models."""

import numpy as np

R_MAX_LY = 50_000.0
R_D_LY = 8_500.0
H_Z_LY = 980.0
T_MW_GYR = 13.6
R_SUN_LY = 26_750.0
THETA_SUN_RAD = 0.0
Z_SUN_LY = 65.0
R_LOCAL_LY = 10_000.0
PITCH_RAD = np.deg2rad(12.5)
LY_PER_PC = 3.26156


def sech2(x):
    y = np.cosh(np.clip(x, -350.0, 350.0))
    return 1.0 / (y * y)


def baseline_stellar_density(R, z):
    R = np.asarray(R, dtype=float)
    z = np.asarray(z, dtype=float)
    disk = np.exp(-R / R_D_LY)
    vertical = sech2(z / (2.0 * H_Z_LY))
    return disk * vertical


def spiral_arm_modulation(R, theta, A_arm=0.35, n_arms=4):
    R = np.asarray(R, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phase = n_arms * (theta - np.log(np.maximum(R, 1.0) / R_D_LY) / np.tan(PITCH_RAD))
    arm_fraction = 0.5 * (1.0 + np.cos(phase))
    arm = 1.0 - A_arm + 2.0 * A_arm * arm_fraction
    spur = 0.20 * np.exp(-((R - R_SUN_LY) / 1_800.0) ** 2
                         - ((theta - THETA_SUN_RAD) / 0.18) ** 2)
    return np.clip(arm * (1.0 + spur), 0.0, 3.0)


def metallicity_readiness(R, t_birth):
    R = np.asarray(R, dtype=float)
    t_birth = np.asarray(t_birth, dtype=float)
    inner_enrichment = 1.0 - np.exp(-np.maximum(t_birth - 1.0, 0.0) / 3.0)
    center_bonus = np.exp(-R / 18_000.0)
    radial_drop = 1.0 / (1.0 + np.exp((R - 35_000.0) / 3_000.0))
    f = inner_enrichment * (0.55 + 0.45 * center_bonus) * radial_drop
    return np.clip(f, 0.0, 1.0)


def sterilization_survival(R, theta, t_birth, tau_bio=0.5):
    R = np.asarray(R, dtype=float)
    theta = np.asarray(theta, dtype=float)
    t_birth = np.asarray(t_birth, dtype=float)
    bulge = 1.0 + 4.0 * np.exp(-(R / 7_500.0) ** 2)
    arms = spiral_arm_modulation(R, theta, A_arm=0.55)
    high_sfr = 1.0 + 1.8 * np.exp(-((t_birth - 3.0) / 2.2) ** 2)
    solar_gamma = 1.0e-3
    gamma = solar_gamma * bulge * arms * high_sfr
    return np.clip(np.exp(-gamma * tau_bio), 0.0, 1.0)


def build_galactic_rate_grid(n_r=31, n_theta=48, n_z=17, n_t=28,
                             A_arm=0.35, birth_rate_per_solar_mass=1.0,
                             local_cluster_variance=0.08):
    R = np.linspace(1_000.0, R_MAX_LY, n_r)
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    z = np.linspace(-5_000.0, 5_000.0, n_z)
    t_birth = np.linspace(0.0, T_MW_GYR, n_t)

    RR, TT, ZZ = np.meshgrid(R, theta, z, indexing="ij")
    modulation = spiral_arm_modulation(RR, TT, A_arm=A_arm)
    density = baseline_stellar_density(RR, ZZ) * modulation

    dR = np.gradient(R)
    dtheta = 2.0 * np.pi / n_theta
    dz = z[1] - z[0] if n_z > 1 else 10_000.0
    volume = (np.broadcast_to(RR[..., None], RR.shape + (n_t,))
              * dtheta * np.asarray(dR)[:, None, None, None] * dz)
    # The direct Jacobian entry above is vectorized below with exact shell volumes.
    volume = np.empty((n_r, n_theta, n_z, n_t), dtype=float)
    r_edges = np.r_[R[0] - 0.5 * dR[0], R[:-1] + 0.5 * dR[:-1],
                    R[-1] + 0.5 * dR[-1]]
    z_edges = np.r_[z[0] - 0.5 * dz, z[:-1] + 0.5 * dz, z[-1] + 0.5 * dz]
    r_shell = (r_edges[1:] ** 2 - r_edges[:-1] ** 2) / 2.0
    z_shell = z_edges[1:] - z_edges[:-1]
    volume[:] = dtheta * r_shell[:, None, None, None] * z_shell[None, None, :, None]
    volume = np.broadcast_to(volume, (n_r, n_theta, n_z, n_t)).copy()

    rng = np.random.default_rng(314159)
    cluster = np.exp(rng.normal(0.0, local_cluster_variance,
                                size=(n_r, n_theta, n_z)))
    density *= cluster
    rate = np.empty((n_r, n_theta, n_z, n_t), dtype=float)
    for kt, tb in enumerate(t_birth):
        metal = metallicity_readiness(RR[..., 0], tb)
        sn = sterilization_survival(RR[..., 0], TT[..., 0], tb)
        rate[..., kt] = birth_rate_per_solar_mass * density * metal[..., None] * sn[..., None]

    # Normalize the integrated prior grid to 1e11 born stars over the Galaxy's
    # 13.6 Gyr history. This keeps downstream Poisson counts computationally
    # tractable while preserving relative space-time structure.
    dt = np.gradient(t_birth)
    integrated_stars = np.sum(rate * volume * dt[None, None, None, :])
    if integrated_stars > 0.0:
        rate *= 1.0e11 * birth_rate_per_solar_mass / integrated_stars

    x = RR * np.cos(TT)
    y = RR * np.sin(TT)
    x_sun = R_SUN_LY * np.cos(THETA_SUN_RAD)
    y_sun = R_SUN_LY * np.sin(THETA_SUN_RAD)
    local_mask = np.sqrt((x - x_sun) ** 2 + (y - y_sun) ** 2
                         + (ZZ - Z_SUN_LY) ** 2) <= R_LOCAL_LY
    local_mask = np.repeat(local_mask[..., None], n_t, axis=-1)

    return {
        "R": R,
        "theta": theta,
        "z": z,
        "t_birth": t_birth,
        "volume": volume,
        "stellar_birth_density": density,
        "stellar_birth_rate": rate,
        "local_mask": local_mask,
    }
