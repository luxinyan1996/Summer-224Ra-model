"""
Calculate water-phase 224Ra activity when sea-ice area Ai is minimal.

Units
-----
M0, Mw, Mi, Nw, Ni, Pw_atoms_m2, Pi : atoms m^-2
time                                  : day
decay constants                       : day^-1
reported Pw                           : dpm (100 L)^-1

The conversion from atoms m^-2 to dpm (100 L)^-1 assumes that the
water-column inventory is uniformly distributed over WATER_DEPTH_M.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Change this path only when a different output location is required.
DEFAULT_OUTPUT_PATH = Path("results/Ai_min_Pw_Pi_results.csv")


# =========================================================
# Model parameters (day^-1)
# =========================================================

LAMBDA_M = 0.00033
LAMBDA_N = 0.00099
LAMBDA_P = 0.189
LAMBDA_REMOVE = 0.0017

FREEZING_TRANSFER_FRACTION = 0.87
WATER_DEPTH_M = 2.0
MIN_AREA = 1.0e-6

YEARS = 14
DT_DAYS = 0.25


# =========================================================
# Annual 228Ra parent inventory M0 (atoms m^-2)
# Values between specified years are linearly interpolated.
# =========================================================

M0_TIME_POINTS = {
    0: 3.20e8,
    5: 3.20e8,
    6: 5.19e8,
    13: 5.18e8,
}


def get_m0_at_time(t_days):
    """Return M0 in atoms m^-2 at time t (days)."""

    t_years = np.asarray(t_days, dtype=float) / 365.0
    years = np.asarray(list(M0_TIME_POINTS.keys()), dtype=float)
    values = np.asarray(list(M0_TIME_POINTS.values()), dtype=float)
    return np.interp(t_years, years, values)


# =========================================================
# Seasonal ice and water areas
# =========================================================

def get_geometry(t_days):
    """Return Ai, Aw, their derivatives, and seasonal phase."""

    k = 2.0 * np.pi / 365.0

    ai_raw = 5.0 + 5.0 * np.cos(k * (t_days - 90.0))
    aw_raw = 10.0 - ai_raw

    dai_dt = -5.0 * k * np.sin(k * (t_days - 90.0))
    daw_dt = -dai_dt

    melting = dai_dt < 0.0
    freezing = not melting

    ai = max(float(ai_raw), MIN_AREA)
    aw = max(float(aw_raw), MIN_AREA)

    return ai, aw, dai_dt, daw_dt, melting, freezing


# =========================================================
# ODE system
# State vector: [Nw, Ni, Pw, Pi, Mi], all in atoms m^-2
# =========================================================

def system(t, y):
    nw, ni, pw, pi, mi = y

    # Mw is prescribed by M0 and is not depleted in the water phase.
    mw = float(get_m0_at_time(t))

    ai, aw, dai_dt, daw_dt, melting, freezing = get_geometry(t)

    s1 = 1.0 if melting else 0.0
    s2 = 1.0 if freezing else 0.0

    # Area-weighted N equations.
    d_aw_nw_dt = (
        LAMBDA_M * mw * aw
        - (LAMBDA_N + LAMBDA_REMOVE) * nw * aw
        + s1 * (-dai_dt) * ni
        - s2 * FREEZING_TRANSFER_FRACTION * dai_dt * nw
    )

    d_ai_ni_dt = (
        LAMBDA_M * mi * ai
        - LAMBDA_N * ni * ai
        - s1 * (-dai_dt) * ni
        + s2 * FREEZING_TRANSFER_FRACTION * dai_dt * nw
    )

    # Area-weighted 224Ra equations.
    d_aw_pw_dt = (
        LAMBDA_N * nw * aw
        - LAMBDA_P * pw * aw
        + s1 * (-dai_dt) * pi
        - s2 * FREEZING_TRANSFER_FRACTION * dai_dt * pw
    )

    d_ai_pi_dt = (
        LAMBDA_N * ni * ai
        - LAMBDA_P * pi * ai
        - s1 * (-dai_dt) * pi
        + s2 * FREEZING_TRANSFER_FRACTION * dai_dt * pw
    )

    # Convert area-weighted derivatives back to state derivatives.
    d_nw_dt = (d_aw_nw_dt - nw * daw_dt) / aw
    d_ni_dt = (d_ai_ni_dt - ni * dai_dt) / ai
    d_pw_dt = (d_aw_pw_dt - pw * daw_dt) / aw
    d_pi_dt = (d_ai_pi_dt - pi * dai_dt) / ai

    # Mi decays throughout the year. During freezing, new ice receives
    # 87% of the prescribed water-phase parent inventory Mw.
    d_ai_mi_dt = (
        -LAMBDA_M * mi * ai
        - s1 * (-dai_dt) * mi
        + s2 * FREEZING_TRANSFER_FRACTION * dai_dt * mw
    )
    d_mi_dt = (d_ai_mi_dt - mi * dai_dt) / ai

    return np.asarray(
        [d_nw_dt, d_ni_dt, d_pw_dt, d_pi_dt, d_mi_dt],
        dtype=float,
    )


# =========================================================
# Fixed-step RK4 solver
# =========================================================

def rk4_solver(func, t_eval, y0):
    y = np.zeros((len(y0), len(t_eval)), dtype=float)
    y[:, 0] = np.asarray(y0, dtype=float)
    y[4, 0] = float(get_m0_at_time(t_eval[0]))

    for i in range(len(t_eval) - 1):
        t_i = t_eval[i]
        t_next = t_eval[i + 1]
        dt = t_next - t_i
        y_i = y[:, i].copy()

        _, _, _, _, _, freezing_i = get_geometry(t_i)
        _, _, _, _, _, freezing_next = get_geometry(t_next)

        k1 = func(t_i, y_i)
        k2 = func(t_i + dt / 2.0, y_i + dt * k1 / 2.0)
        k3 = func(t_i + dt / 2.0, y_i + dt * k2 / 2.0)
        k4 = func(t_next, y_i + dt * k3)

        y_next = y_i + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        y_next = np.maximum(y_next, 0.0)

        # At the start of each freezing season (Ai minimum), reset Mi to
        # that year's prescribed M0, as required by the model assumption.
        if freezing_next and not freezing_i:
            y_next[4] = float(get_m0_at_time(t_next))

        y[:, i + 1] = y_next

    return y


# =========================================================
# Unit conversion
# =========================================================

def ra224_atoms_m2_to_dpm_100l(atoms_m2, water_depth_m=WATER_DEPTH_M):
    """Convert 224Ra inventory from atoms m^-2 to dpm (100 L)^-1.

    atoms m^-2 / depth (m) -> atoms m^-3
    atoms m^-3 * 0.1 m^3  -> atoms per 100 L
    atoms * lambda/day / 1440 -> decays per minute (dpm)
    """

    atoms_m2 = np.asarray(atoms_m2, dtype=float)
    return (
        atoms_m2
        / water_depth_m
        * 0.1
        * LAMBDA_P
        / 1440.0
    )


# =========================================================
# Run model and report Pw and Pi at annual Ai minima
# =========================================================

def main(output_path=DEFAULT_OUTPUT_PATH):
    # DT_DAYS = 0.25 makes every annual Ai minimum (day 272.5) an exact
    # integration point, so no temporal interpolation is required.
    t_eval = np.arange(
        0.0,
        365.0 * YEARS + DT_DAYS / 2.0,
        DT_DAYS,
    )

    y_result = rk4_solver(
        system,
        t_eval,
        [0.0, 0.0, 0.0, 0.0, float(get_m0_at_time(0.0))],
    )

    pw_atoms_m2 = y_result[2]
    pi_atoms_m2 = y_result[3]

    records = []
    for model_year in range(YEARS):
        ai_min_day = model_year * 365.0 + 272.5
        index = int(round(ai_min_day / DT_DAYS))

        records.append({
            "model_year": model_year,
            "time_day": ai_min_day,
            "M0_atoms_m2": float(get_m0_at_time(ai_min_day)),
            "Pw_atoms_m2_at_Ai_min": pw_atoms_m2[index],
            "Pw_dpm_100L_at_Ai_min": float(
                ra224_atoms_m2_to_dpm_100l(pw_atoms_m2[index])
            ),
            "Pi_atoms_m2_at_Ai_min": pi_atoms_m2[index],
            "Pi_dpm_100L_at_Ai_min": float(
                ra224_atoms_m2_to_dpm_100l(pi_atoms_m2[index])
            ),
        })

    results = pd.DataFrame(records)

    print(
        results.to_string(
            index=False,
            formatters={
                "M0_atoms_m2": "{:.6e}".format,
                "Pw_atoms_m2_at_Ai_min": "{:.6e}".format,
                "Pw_dpm_100L_at_Ai_min": "{:.6f}".format,
                "Pi_atoms_m2_at_Ai_min": "{:.6e}".format,
                "Pi_dpm_100L_at_Ai_min": "{:.6f}".format,
            },
        )
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


if __name__ == "__main__":
    # No command-line argument parsing is used, so this works in both a
    # normal Python process and a Jupyter notebook cell.
    main()
