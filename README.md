# Seasonal sea-ice `224Ra` model

This repository contains the two Python scripts used to calculate water-phase and seasonal-ice-phase 224Ra diagnostics under the supplied sea-ice forcing.

## Repository layout

```text
Summer-224Ra-model/
|-- README.md
|-- requirements.txt
|-- data/
|   `-- Extended_Data_3.xlsx
|-- src/
|   |-- Future_projected_224Ra.py
|   `-- Simulated-summer-224Ra.py
`-- results/
    |-- Pi_Pw.csv
    `-- Ai_min_Pw_Pi_results.csv
```

## Requirements

- Python 3.9 or newer
- NumPy >=1.23 and <3
- pandas >=1.5 and <3
- openpyxl >=3.0 and <4

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Input data

`Future_projected_224Ra.py` reads:

```text
data/Extended_Data_3.xlsx
```

The workbook must contain a worksheet named `Sheet1` and one row per scenario-year. The required fields are:

```text
scenario
year
Amelt
m0_value
tmax
tmin
freeze_onset
```

The script also accepts the unit-bearing column names documented in its source code, such as `A_melt(106 km-2)`, `228Ra (atoms m-2)`, and `tmax (DOY)`.

## Running the models

Run these commands from the repository root:

```bash
python src/Future_projected_224Ra.py data/Extended_Data_3.xlsx results/Pi_Pw.csv
```

This produces:

```text
results/Pi_Pw.csv
```

The output contains the annual `Pi` and `Pw` diagnostics:

- `Pi` denotes the maximum 224Ra activity in the seasonal-ice phase during the summer melt interval, representing the ice-phase activity immediately before or during melt.
- `Pw` denotes the area-weighted 224Ra activity in the summer water phase at `tmin`, when the modeled seasonal ice has melted.

Both activities are reported in dpm (100 L)^-1.

## Background (^{228}Ra) forcing and model-year mapping

In `Simulated-summer-224Ra.py`, `M0` denotes the prescribed background (^{228}Ra) inventory of seawater. It is specified in atoms m^-2 and linearly interpolated between the time points defined in `M0_TIME_POINTS`.

The first five simulated years (`model_year` 0-4) are spin-up years used to allow the radionuclide system to approach a stable seasonal cycle. They are not assigned to the reported calendar-year observations. The subsequent model-year mapping is:

| `model_year` | Interpretation |
|---:|---|
| 5 | 2007 |
| 9 | 2011 |
| 13 | 2015 |

Thus, the outputs for model years 5, 9, and 13 correspond to the 2007, 2011, and 2015 cases, respectively.

The idealized seasonal-ice calculation is run with:

```bash
python src/Simulated-summer-224Ra.py
```

This produces:

```text
results/Ai_min_Pw_Pi_results.csv
```

This second script uses the parameter values defined in the source file and does not require an external input table.

## Units

- Inventories and concentrations in the model: atoms m^-2
- Reported activities: dpm (100 L)^-1
- Water-column depth used for conversion: 2 m
- Time: days
- Decay and removal constants: day^-1

See the module docstrings and parameter comments for the model equations, initial conditions, seasonal-area functions, and numerical integration settings.

## Reproducibility

The scripts use deterministic fixed-step fourth-order Runge-Kutta integration. Results are written as UTF-8 CSV files. To reproduce the supplied outputs, install the dependencies listed above, place the forcing workbook at the path above, and run both commands from the repository root.

## Code availability

This repository is intended to accompany the manuscript submission and contains the source code, forcing workbook, and generated diagnostic tables needed to reproduce the reported calculations.
