"""
Predict water-phase and seasonal-ice-phase 224Ra under SSP sea-ice change.

This version models only the seasonal ice that completely melts each summer.
The seasonal ice area is read from the Excel column ``Amelt`` and follows:

1. previous freeze_onset -> current tmax: freeze from 0 to Amelt;
2. current tmax -> current tmin: melt from Amelt to 0;
3. current tmin -> current freeze_onset: no seasonal ice.

The ice inventories are exactly zero after complete summer melt. New ice is
formed from the current water phase during the next freezing season. Water-
phase Nw and Pw remain continuous between years.

Two additional annual diagnostics are included:

* the actual maximum Pi during the melt interval tmax <= t < tmin, together
  with its day and its time before tmin;
* the area that no longer freezes is propagated across years. The reference
  area is the largest Amelt observed so far in that scenario, so the model Pw
  is area-weighted with 1.80 dpm/100 L over the cumulative non-freezing area.

Units
-----
* m0_value and all M/N/P concentrations: atoms m^-2
* Pw and Pi output: dpm per 100 L, using a 2 m water depth
* Amelt and MODEL_DOMAIN_AREA must use the same area unit
* summer_224Ra_input_stock_model has units
  (224Ra atoms m^-2) * (input sea-ice area unit). It is an area-weighted
  model stock and is not multiplied by a physical km^2-to-m^2 conversion.

The program is run from the command line. It writes one CSV containing only
the two requested annual diagnostics:

* ``Pi``: maximum seasonal-ice 224Ra activity during the melt interval;
* ``Pw``: area-weighted water-phase 224Ra activity at ``tmin``.

Example
-------
python nature_submit_Pi_Pw.py data/Extended_Data_3.xlsx results/Pi_Pw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Default paths
# ============================================================================

DEFAULT_INPUT_XLSX = Path("data/Extended_Data_3.xlsx")
DEFAULT_OUTPUT_CSV = Path("results/Pi_Pw.csv")


# ============================================================================
# Model parameters
# ============================================================================

# Decay/removal constants, day^-1
LAMBDA_M = 0.00033
LAMBDA_N = 0.00099
LAMBDA_P = 0.189
LAMBDA_REMOVE = 0.0017

FREEZING_TRANSFER_FRACTION = 0.87
WATER_DEPTH_M = 2.0
DAYS_PER_YEAR = 365.0
DT_DAYS = 0.25
AREA_EPSILON = 1.0e-12
NON_FREEZING_AREA_PW_DPM_100L = 1.80

# The 1989 run starts from the periodic steady state of the 1989 forcing,
# rather than from zero N/P inventories. The fixed-year preconditioning is
# repeated until the five stock variables change by less than this tolerance.
INITIALIZE_1989_AT_PERIODIC_BALANCE = True
BALANCE_MAX_CYCLES = 200
BALANCE_RTOL = 1.0e-9
BALANCE_ATOL = 1.0e-6

# Fixed modeled surface domain. This is constant in every year and scenario.
# Annual Amax and Amin_table are not used by the differential model; only
# Amelt controls the seasonal ice that freezes and melts. The value below is
# the maximum historical domain area in the supplied table, in the same area
# unit as Amelt. It must be larger than every Amelt value.
MODEL_DOMAIN_AREA = 16.249

REQUIRED_COLUMNS = [
    "scenario",
    "year",
    "Amelt",
    "m0_value",
    "tmax",
    "tmin",
    "freeze_onset",
]


# ============================================================================
# Input and unit conversion
# ============================================================================

def read_forcing(path: Path) -> pd.DataFrame:
    """Read and validate the annual forcing table."""

    forcing = pd.read_excel(path, sheet_name="Sheet1")

    # Accept both the compact internal names used by the original workbook
    # and the unit-bearing names used in the later exported workbook.
    column_aliases = {
        "Year": "year",
        "A_melt(106 km-2)": "Amelt",
        "A_melt (106 km-2)": "Amelt",
        "A_melt  (106 km-2)": "Amelt",
        "228Ra (atoms m-2)": "m0_value",
        "tmax (DOY)": "tmax",
        "tmin (DOY)": "tmin",
        "freeze_onset (DOY)": "freeze_onset",
    }
    forcing = forcing.rename(columns=column_aliases)

    missing = [column for column in REQUIRED_COLUMNS if column not in forcing]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    forcing = forcing[REQUIRED_COLUMNS].copy()

    numeric_columns = [
        "year", "Amelt", "m0_value", "tmax", "tmin",
        "freeze_onset",
    ]
    for column in numeric_columns:
        forcing[column] = pd.to_numeric(forcing[column], errors="raise")

    forcing["scenario"] = forcing["scenario"].astype(str).str.strip()
    forcing["year"] = forcing["year"].astype(int)
    forcing = forcing.sort_values(["scenario", "year"]).reset_index(drop=True)

    if forcing[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("The forcing table contains missing values.")

    if forcing.duplicated(["scenario", "year"]).any():
        raise ValueError("Duplicate scenario-year rows were found.")

    for scenario, group in forcing.groupby("scenario", sort=False):
        years = group["year"].to_numpy()
        if len(years) > 1 and not np.all(np.diff(years) == 1):
            raise ValueError(f"Scenario {scenario} does not contain consecutive years.")

    bad_timing = forcing[
        (forcing["tmax"] < 0)
        | (forcing["tmax"] > forcing["tmin"])
        | (forcing["tmin"] > forcing["freeze_onset"])
        | (forcing["freeze_onset"] > DAYS_PER_YEAR)
    ]
    if not bad_timing.empty:
        raise ValueError(
            "Each row must satisfy 0 <= tmax <= tmin <= freeze_onset <= 365."
        )

    if (forcing["Amelt"] < 0).any():
        raise ValueError("Amelt cannot be negative.")

    if forcing["Amelt"].max() >= MODEL_DOMAIN_AREA:
        raise ValueError(
            "MODEL_DOMAIN_AREA must be larger than every Amelt value. "
            f"Current maximum Amelt = {forcing['Amelt'].max():g}."
        )

    return forcing


def atoms_m2_to_dpm_100l(atoms_m2):
    """Convert a 224Ra column inventory from atoms m^-2 to dpm/100 L."""

    atoms_m2 = np.asarray(atoms_m2, dtype=float)
    return atoms_m2 / WATER_DEPTH_M * 0.1 * LAMBDA_P / 1440.0


def area_weight_pw_dpm_100l(
    model_pw_dpm_100l, current_amelt, reference_amelt
):
    """Apply cumulative non-freezing-area weighting to Pw.

    ``reference_amelt`` is the largest Amelt observed so far in the same
    scenario. The difference between that reference and the current Amelt is
    the cumulative area that has stopped freezing. It remains at the fixed
    water-phase activity 1.80 dpm/100 L and is therefore carried into later
    years. If a later Amelt increases, the cumulative difference shrinks.

    The weighted value is

        (current_Amelt * model_Pw
        + cumulative_non_freezing_area * 1.80) / reference_Amelt.
    """

    model_pw_dpm_100l = np.asarray(model_pw_dpm_100l, dtype=float)

    if reference_amelt is None or reference_amelt <= AREA_EPSILON:
        corrected = model_pw_dpm_100l.copy()
        reduced_area = 0.0
        applied = False
    else:
        reduced_area = max(float(reference_amelt - current_amelt), 0.0)
        corrected = (
            float(current_amelt) * model_pw_dpm_100l
            + reduced_area * NON_FREEZING_AREA_PW_DPM_100L
        ) / float(reference_amelt)
        applied = reduced_area > AREA_EPSILON

    if corrected.ndim == 0:
        corrected = float(corrected)

    return corrected, reduced_area, applied


# ============================================================================
# Smooth seasonal ice geometry
# ============================================================================

def half_cosine_decrease(start_value, end_value, elapsed, duration):
    """Smoothly decrease from start_value to end_value."""

    if duration <= 0:
        return float(end_value)
    u = np.clip(elapsed / duration, 0.0, 1.0)
    return float(
        end_value
        + 0.5 * (start_value - end_value) * (1.0 + np.cos(np.pi * u))
    )


def half_cosine_increase(start_value, end_value, elapsed, duration):
    """Smoothly increase from start_value to end_value."""

    if duration <= 0:
        return float(end_value)
    u = np.clip(elapsed / duration, 0.0, 1.0)
    return float(
        start_value
        + 0.5 * (end_value - start_value) * (1.0 - np.cos(np.pi * u))
    )


class SeasonalIceScenarioModel:
    """Model one scenario using only the completely melting ice area."""

    def __init__(self, scenario_data: pd.DataFrame):
        self.data = scenario_data.sort_values("year").reset_index(drop=True)
        self.start_year = int(self.data["year"].iloc[0])
        self.end_year = int(self.data["year"].iloc[-1])
        self.rows = {
            int(row.year): row for row in self.data.itertuples(index=False)
        }
        self.periodic_start = False
        self.initial_balance_cycles = 0
        self.initial_balance_error = np.nan

    def row_for_year(self, year: int):
        """Return a row, extending the first/last row beyond the table."""

        bounded_year = min(max(year, self.start_year), self.end_year)
        return self.rows[bounded_year]

    def year_and_day(self, t_days: float):
        """Convert elapsed model days to calendar year and day of year."""

        offset = int(np.floor((t_days + 1.0e-10) / DAYS_PER_YEAR))
        year = self.start_year + offset
        day = t_days - offset * DAYS_PER_YEAR
        if day < 0:
            day = 0.0
        return year, day

    def m0(self, t_days: float) -> float:
        """Return the calendar year's constant water-phase 228Ra M0."""

        year, _ = self.year_and_day(t_days)
        return float(self.row_for_year(year).m0_value)

    def seasonal_ice_area(self, t_days: float) -> float:
        """Return seasonal ice area As. Only Amelt is used."""

        year, day = self.year_and_day(t_days)
        row = self.row_for_year(year)

        if day < float(row.tmax):
            if year <= self.start_year and self.periodic_start:
                # For the balanced 1989 start, the preceding freeze season is
                # also the 1989 forcing repeated periodically.
                previous = self.row_for_year(self.start_year)
                elapsed = day + DAYS_PER_YEAR - float(previous.freeze_onset)
                duration = (
                    DAYS_PER_YEAR
                    - float(previous.freeze_onset)
                    + float(row.tmax)
                )
                area = half_cosine_increase(
                    0.0, float(row.Amelt), elapsed, duration
                )
            elif year <= self.start_year:
                # This branch is retained only for an explicitly unbalanced
                # run. Normal runs use the periodic 1989 initialization.
                area = float(row.Amelt)
            else:
                previous = self.row_for_year(year - 1)
                elapsed = day + DAYS_PER_YEAR - float(previous.freeze_onset)
                duration = (
                    DAYS_PER_YEAR
                    - float(previous.freeze_onset)
                    + float(row.tmax)
                )
                area = half_cosine_increase(
                    0.0, float(row.Amelt), elapsed, duration
                )

        elif day <= float(row.tmin):
            area = half_cosine_decrease(
                float(row.Amelt),
                0.0,
                day - float(row.tmax),
                float(row.tmin) - float(row.tmax),
            )

        elif day < float(row.freeze_onset):
            area = 0.0

        else:
            next_row = self.row_for_year(year + 1)
            elapsed = day - float(row.freeze_onset)
            duration = (
                DAYS_PER_YEAR
                - float(row.freeze_onset)
                + float(next_row.tmax)
            )
            area = half_cosine_increase(
                0.0, float(next_row.Amelt), elapsed, duration
            )

        if abs(area) < AREA_EPSILON:
            return 0.0
        return max(float(area), 0.0)

    @staticmethod
    def concentrations(states, seasonal_area):
        """Return Nw, Pw, Mi, Ni and Pi from stock state variables."""

        qnw, qpw, qm, qn, qp = states
        water_area = MODEL_DOMAIN_AREA - seasonal_area

        nw = qnw / water_area
        pw = qpw / water_area

        if seasonal_area > AREA_EPSILON:
            mi = qm / seasonal_area
            ni = qn / seasonal_area
            pi = qp / seasonal_area
        else:
            mi = 0.0
            ni = 0.0
            pi = 0.0

        return np.asarray([nw, pw, mi, ni, pi], dtype=float)

    def reaction_derivatives(self, t_days, states, seasonal_area):
        """Radioactive production, decay and water removal for stocks."""

        qnw, qpw, qm, qn, qp = states
        water_area = MODEL_DOMAIN_AREA - seasonal_area
        mw = self.m0(t_days)

        # Mw is externally maintained at M0, so it does not decrease in the
        # water phase. Its decay production of Nw continues at all times.
        d_qnw = (
            LAMBDA_M * mw * water_area
            - (LAMBDA_N + LAMBDA_REMOVE) * qnw
        )
        d_qpw = LAMBDA_N * qnw - LAMBDA_P * qpw

        # In seasonal ice, Mi always decays to Ni. The chain then produces Pi.
        d_qm = -LAMBDA_M * qm
        d_qn = LAMBDA_M * qm - LAMBDA_N * qn
        d_qp = LAMBDA_N * qn - LAMBDA_P * qp

        return np.asarray([d_qnw, d_qpw, d_qm, d_qn, d_qp], dtype=float)

    def reaction_rk4(self, t_days, states, dt, seasonal_area):
        """Advance only radioactive reactions over one substep."""

        k1 = self.reaction_derivatives(t_days, states, seasonal_area)
        k2 = self.reaction_derivatives(
            t_days + dt / 2.0, states + dt * k1 / 2.0, seasonal_area
        )
        k3 = self.reaction_derivatives(
            t_days + dt / 2.0, states + dt * k2 / 2.0, seasonal_area
        )
        k4 = self.reaction_derivatives(
            t_days + dt, states + dt * k3, seasonal_area
        )
        result = states + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return np.maximum(result, 0.0)

    def transfer_for_area_change(self, t_mid, states, area_before, area_after):
        """Transfer M/N/P stocks during freezing or melting.

        Returns
        -------
        updated_states : ndarray
        melted_qp : float
            224Ra stock transferred from seasonal ice to water in this step.
        """

        qnw, qpw, qm, qn, qp = states.copy()
        melted_qp = 0.0
        delta_area = area_after - area_before

        if delta_area > AREA_EPSILON:
            # Freezing: newly formed ice receives M0 and current Nw/Pw.
            water_area_before = MODEL_DOMAIN_AREA - area_before
            nw = qnw / water_area_before
            pw = qpw / water_area_before
            new_ice_area = delta_area

            # Every newly frozen seasonal-ice increment starts at the current
            # year's M0, exactly as specified. The 0.87 transfer fraction is
            # applied only to water-phase N and P entrained during freezing.
            transfer_qm = new_ice_area * self.m0(t_mid)
            transfer_qn = FREEZING_TRANSFER_FRACTION * new_ice_area * nw
            transfer_qp = FREEZING_TRANSFER_FRACTION * new_ice_area * pw

            # M0 in water is prescribed and therefore is not depleted.
            qm += transfer_qm

            # N and P are conserved between the water and seasonal ice stocks.
            transfer_qn = min(transfer_qn, qnw)
            transfer_qp = min(transfer_qp, qpw)
            qnw -= transfer_qn
            qpw -= transfer_qp
            qn += transfer_qn
            qp += transfer_qp

        elif delta_area < -AREA_EPSILON and area_before > AREA_EPSILON:
            # Melting: remove the same fraction of every ice stock as the
            # fraction of seasonal ice area that disappeared during the step.
            melted_fraction = min((-delta_area) / area_before, 1.0)

            transfer_qm = qm * melted_fraction
            transfer_qn = qn * melted_fraction
            transfer_qp = qp * melted_fraction

            qm -= transfer_qm
            qn -= transfer_qn
            qp -= transfer_qp

            # Melted M joins the maintained water M0 reservoir; only N and P
            # need explicit prognostic water stocks.
            qnw += transfer_qn
            qpw += transfer_qp
            melted_qp = transfer_qp

        if area_after <= AREA_EPSILON:
            # The modeled seasonal ice has completely melted. Any round-off
            # residual is transferred before the ice stocks are cleared.
            qnw += qn
            qpw += qp
            melted_qp += qp
            qm = 0.0
            qn = 0.0
            qp = 0.0

        updated = np.asarray([qnw, qpw, qm, qn, qp], dtype=float)
        return np.maximum(updated, 0.0), float(max(melted_qp, 0.0))

    def advance_step(self, t0, state, dt):
        """Advance one Strang-split step and return state plus melt transfer."""

        t1 = t0 + dt
        t_mid = 0.5 * (t0 + t1)
        area0 = self.seasonal_ice_area(t0)
        area1 = self.seasonal_ice_area(t1)

        state = self.reaction_rk4(t0, state, dt / 2.0, area0)
        state, melted_qp = self.transfer_for_area_change(
            t_mid, state, area0, area1
        )
        state = self.reaction_rk4(t_mid, state, dt / 2.0, area1)
        return np.maximum(state, 0.0), melted_qp, area0, area1

    def periodic_1989_initial_state(self):
        """Find the periodic year-start state under fixed 1989 forcing.

        A separate one-row model is used so that both the preceding and
        following seasonal cycles use the 1989 forcing. Its year-start area
        is the area reached during the repeated 1989 freeze season, not the
        full Amelt area.
        """

        first_row = self.data.iloc[[0]].copy()
        periodic_model = SeasonalIceScenarioModel(first_row)
        periodic_model.periodic_start = True

        n_steps = int(round(DAYS_PER_YEAR / DT_DAYS))
        initial_area = periodic_model.seasonal_ice_area(0.0)
        state = np.zeros(5, dtype=float)
        state[2] = initial_area * periodic_model.m0(0.0)

        error = np.inf
        for cycle in range(1, BALANCE_MAX_CYCLES + 1):
            previous = state.copy()
            t = 0.0
            for _ in range(n_steps):
                state, _, _, _ = periodic_model.advance_step(
                    t, state, DT_DAYS
                )
                t += DT_DAYS

            scale = np.maximum(np.abs(previous), 1.0)
            error = float(np.max(np.abs(state - previous) / scale))
            if error <= BALANCE_RTOL:
                self.initial_balance_cycles = cycle
                self.initial_balance_error = error
                return state

        self.initial_balance_cycles = BALANCE_MAX_CYCLES
        self.initial_balance_error = error
        raise RuntimeError(
            "1989 periodic initialization did not converge within "
            f"{BALANCE_MAX_CYCLES} cycles; final relative change={error:.3e}."
        )

    def solve(self):
        """Run the model and return time, stock states and annual melt input."""

        number_of_years = self.end_year - self.start_year + 1
        end_time = number_of_years * DAYS_PER_YEAR
        time = np.arange(0.0, end_time + DT_DAYS / 2.0, DT_DAYS)
        states = np.zeros((5, len(time)), dtype=float)

        if INITIALIZE_1989_AT_PERIODIC_BALANCE:
            self.periodic_start = True
            states[:, 0] = self.periodic_1989_initial_state()
        else:
            initial_area = self.seasonal_ice_area(0.0)
            states[2, 0] = initial_area * self.m0(0.0)

        annual_melt_qp = {
            year: 0.0 for year in range(self.start_year, self.end_year + 1)
        }
        annual_melt_area = {
            year: 0.0 for year in range(self.start_year, self.end_year + 1)
        }

        for index in range(len(time) - 1):
            t0 = time[index]
            dt = time[index + 1] - t0
            t_mid = t0 + dt / 2.0

            state, melted_qp, area0, area1 = self.advance_step(
                t0, states[:, index].copy(), dt
            )
            states[:, index + 1] = state

            if area1 < area0 - AREA_EPSILON:
                year, _ = self.year_and_day(t_mid)
                year = min(max(year, self.start_year), self.end_year)
                annual_melt_qp[year] += melted_qp
                annual_melt_area[year] += area0 - area1

        return time, states, annual_melt_qp, annual_melt_area


# ============================================================================
# Output tables
# ============================================================================

def calculate_predictions(forcing: pd.DataFrame):
    """Calculate daily Pw/Pi and annual summer-melt results."""

    daily_tables = []
    annual_tables = []

    for scenario, scenario_data in forcing.groupby("scenario", sort=True):
        scenario_data = scenario_data.sort_values("year").reset_index(drop=True)
        model = SeasonalIceScenarioModel(scenario_data)
        time, states, melt_qp, melted_area = model.solve()
        amelt_by_year = {
            int(row.year): float(row.Amelt)
            for row in scenario_data.itertuples(index=False)
        }
        # Cumulative transmission: the reference area is the largest seasonal
        # melt area reached so far. A reduction therefore remains represented
        # in later years until Amelt recovers to that historical maximum.
        reference_amelt_by_year = {}
        running_reference_amelt = None
        for row in scenario_data.itertuples(index=False):
            current_amelt = float(row.Amelt)
            if running_reference_amelt is None:
                running_reference_amelt = current_amelt
            else:
                running_reference_amelt = max(
                    running_reference_amelt, current_amelt
                )
            reference_amelt_by_year[int(row.year)] = running_reference_amelt

        final_time = (model.end_year - model.start_year + 1) * DAYS_PER_YEAR
        integer_day_mask = (time < final_time) & np.isclose(time % 1.0, 0.0)
        daily_indices = np.flatnonzero(integer_day_mask)

        daily_records = []
        for index in daily_indices:
            t = float(time[index])
            year, day = model.year_and_day(t)
            seasonal_area = model.seasonal_ice_area(t)
            nw, pw, mi, ni, pi = model.concentrations(
                states[:, index], seasonal_area
            )
            model_pw_dpm = float(atoms_m2_to_dpm_100l(pw))
            reference_amelt = reference_amelt_by_year[year]
            cumulative_non_freezing_area = max(
                reference_amelt - amelt_by_year[year], 0.0
            )
            weighted_pw_dpm, reduced_area, correction_applied = (
                area_weight_pw_dpm_100l(
                    model_pw_dpm, amelt_by_year[year], reference_amelt
                )
            )
            daily_records.append({
                "scenario": scenario,
                "year": year,
                "day_of_year": int(round(day)),
                "seasonal_ice_area": seasonal_area,
                "Pw_model_dpm_100L": model_pw_dpm,
                "Pw_area_weighted_dpm_100L": weighted_pw_dpm,
                # Pw_dpm_100L is retained as the main output and now contains
                # the requested area-weighted value.
                "Pw_dpm_100L": weighted_pw_dpm,
                "Pi_dpm_100L": float(atoms_m2_to_dpm_100l(pi)),
                "reference_Amelt_cumulative": reference_amelt,
                "cumulative_non_freezing_area": cumulative_non_freezing_area,
                "Pw_area_weighting_applied": correction_applied,
                "Nw_atoms_m2": nw,
                "Ni_atoms_m2": ni,
                "Mi_atoms_m2": mi,
            })
        daily_tables.append(pd.DataFrame(daily_records))

        annual_records = []
        for row in scenario_data.itertuples(index=False):
            year = int(row.year)
            reference_amelt = reference_amelt_by_year[year]
            cumulative_non_freezing_area = max(
                reference_amelt - float(row.Amelt), 0.0
            )
            year_start = (year - model.start_year) * DAYS_PER_YEAR
            t_at_tmax = year_start + float(row.tmax)
            t_at_tmin = year_start + float(row.tmin)

            index_tmax = int(round(t_at_tmax / DT_DAYS))
            index_tmin = int(round(t_at_tmin / DT_DAYS))

            area_tmax = model.seasonal_ice_area(t_at_tmax)
            area_tmin = model.seasonal_ice_area(t_at_tmin)

            _, pw_tmax, _, _, pi_tmax = model.concentrations(
                states[:, index_tmax], area_tmax
            )
            _, pw_tmin, _, _, pi_tmin = model.concentrations(
                states[:, index_tmin], area_tmin
            )

            pw_tmax_model_dpm = float(atoms_m2_to_dpm_100l(pw_tmax))
            pw_tmin_model_dpm = float(atoms_m2_to_dpm_100l(pw_tmin))
            (
                pw_tmax_weighted_dpm,
                reduced_area,
                correction_applied,
            ) = area_weight_pw_dpm_100l(
                pw_tmax_model_dpm, float(row.Amelt), reference_amelt
            )
            pw_tmin_weighted_dpm, _, _ = area_weight_pw_dpm_100l(
                pw_tmin_model_dpm, float(row.Amelt), reference_amelt
            )

            # Search every DT_DAYS model step during the melt interval. The
            # endpoint tmin is excluded because seasonal ice area and Pi are
            # exactly zero there. This reports the true computed maximum,
            # rather than assuming it must occur at the final time step.
            melt_start_index = index_tmax
            melt_end_index_exclusive = index_tmin
            melt_indices = np.arange(
                melt_start_index, melt_end_index_exclusive, dtype=int
            )

            if len(melt_indices) == 0:
                pi_max_dpm = 0.0
                pi_max_index = index_tmax
            else:
                pi_melt_dpm = np.empty(len(melt_indices), dtype=float)
                for local_index, state_index in enumerate(melt_indices):
                    state_time = float(time[state_index])
                    state_area = model.seasonal_ice_area(state_time)
                    _, _, _, _, pi_value = model.concentrations(
                        states[:, state_index], state_area
                    )
                    pi_melt_dpm[local_index] = float(
                        atoms_m2_to_dpm_100l(pi_value)
                    )
                max_local_index = int(np.argmax(pi_melt_dpm))
                pi_max_index = int(melt_indices[max_local_index])
                pi_max_dpm = float(pi_melt_dpm[max_local_index])

            pi_max_time = float(time[pi_max_index])
            pi_max_day = pi_max_time - year_start
            pi_max_area = model.seasonal_ice_area(pi_max_time)
            pi_max_hours_before_tmin = (t_at_tmin - pi_max_time) * 24.0

            # Divide the area-weighted melt stock by the fixed modeled domain
            # to express its concentration-equivalent contribution.
            melt_equivalent_atoms_m2 = melt_qp[year] / MODEL_DOMAIN_AREA

            annual_records.append({
                "scenario": scenario,
                "year": year,
                "Amelt_input": float(row.Amelt),
                "reference_Amelt_cumulative": reference_amelt,
                "cumulative_non_freezing_area": cumulative_non_freezing_area,
                "Pw_area_weighting_applied": correction_applied,
                "non_freezing_area_Pw_dpm_100L": (
                    NON_FREEZING_AREA_PW_DPM_100L
                ),
                "melted_area_check": float(melted_area[year]),
                "m0_atoms_m2": float(row.m0_value),
                "initial_1989_periodic_balance_cycles": (
                    model.initial_balance_cycles
                ),
                "initial_1989_periodic_balance_relative_error": (
                    model.initial_balance_error
                ),
                "Pw_at_tmax_model_dpm_100L": pw_tmax_model_dpm,
                "Pw_at_tmax_area_weighted_dpm_100L": (
                    pw_tmax_weighted_dpm
                ),
                "Pw_at_tmax_dpm_100L": pw_tmax_weighted_dpm,
                "Pi_at_tmax_dpm_100L": float(
                    atoms_m2_to_dpm_100l(pi_tmax)
                ),
                "Pi_max_during_melt_dpm_100L": pi_max_dpm,
                "Pi_max_day_of_year": pi_max_day,
                "Pi_max_hours_before_tmin": pi_max_hours_before_tmin,
                "seasonal_ice_area_at_Pi_max": pi_max_area,
                "Pi_max_is_last_step_before_tmin": bool(
                    pi_max_index == index_tmin - 1
                ),
                "Pw_at_tmin_model_dpm_100L": pw_tmin_model_dpm,
                "Pw_at_tmin_area_weighted_dpm_100L": (
                    pw_tmin_weighted_dpm
                ),
                "Pw_at_tmin_dpm_100L": pw_tmin_weighted_dpm,
                # At tmin all modeled seasonal ice has melted, so Pi is zero.
                "Pi_at_tmin_dpm_100L": float(
                    atoms_m2_to_dpm_100l(pi_tmin)
                ),
                "summer_224Ra_input_stock_model": float(melt_qp[year]),
                "summer_224Ra_input_equiv_dpm_100L": float(
                    atoms_m2_to_dpm_100l(melt_equivalent_atoms_m2)
                ),
            })

        annual_tables.append(pd.DataFrame(annual_records))

    daily = pd.concat(daily_tables, ignore_index=True)
    annual = pd.concat(annual_tables, ignore_index=True)
    return daily, annual


def run_model(
    input_xlsx: Path = DEFAULT_INPUT_XLSX,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
):
    """Run the model and save only the requested ``Pi`` and ``Pw`` columns.

    The source annual table uses the original diagnostic names. The submitted
    result deliberately keeps only Excel column P (Pi maximum) and column V
    (area-weighted Pw at tmin), renamed to ``Pi`` and ``Pw`` respectively.
    Rows retain the deterministic order produced by scenario and year sorting.
    """

    forcing = read_forcing(Path(input_xlsx))
    _, annual = calculate_predictions(forcing)

    required = [
        "Pi_max_during_melt_dpm_100L",       # original annual-table column P
        "Pw_at_tmin_area_weighted_dpm_100L", # original annual-table column V
    ]
    missing = [column for column in required if column not in annual.columns]
    if missing:
        raise RuntimeError(f"Required diagnostic columns are missing: {missing}")

    result = annual[required].rename(
        columns={
            "Pi_max_during_melt_dpm_100L": "Pi",
            "Pw_at_tmin_area_weighted_dpm_100L": "Pw",
        }
    )

    numeric = result[["Pi", "Pw"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise RuntimeError("The model produced invalid Pi or Pw values.")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Model completed.")
    print(f"Output: {output_csv.resolve()}")
    print(f"Rows: {len(result)}")
    print(result.head().to_string(index=False))
    return result


def main():
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict annual 224Ra and write only Pi and Pw to a CSV file."
        )
    )
    parser.add_argument(
        "input_xlsx",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_XLSX,
        help="Excel forcing table (default: data/Extended_Data_3.xlsx).",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV (default: results/Pi_Pw.csv).",
    )
    # Jupyter/IPython adds an internal ``--f=...kernel.json`` argument when
    # a script is executed from a notebook. Keep the two positional model
    # arguments and ignore only those unrelated notebook arguments.
    args, _ = parser.parse_known_args()
    run_model(args.input_xlsx, args.output_csv)


if __name__ == "__main__":
    main()
