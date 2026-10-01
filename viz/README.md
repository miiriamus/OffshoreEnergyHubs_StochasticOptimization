# `viz/` — Plots for the stochastic model

All the project's figures live here. Two groups:

- **Model results** (`run_plots.py`): validation figures for the EH models.
  - *Single-model*: internals of one model (SOC, charge/discharge, heat pump, H₂ storage…).
  - *Comparison*: one variable across several models (wind used, curtailment, gas…).
  - Each plot shows the probability-weighted **expected value** and, optionally, the individual scenarios behind it.
- **Scenario analysis** (`scenario_analysis/`): figures about the scenarios themselves (clustering, effect of the number of scenarios `k`).

## Quick start

### Model results

1. Edit the settings at the top of `run_plots.py` (study day, models, figures).
2. Run `run_plots.py` from Spyder, or from the project root:

```bash
python -m viz.run_plots
```

In Spyder, restart the kernel after editing any file in `viz/`, otherwise old versions of the modules stay in memory.

Figures are saved to `outputs/<STUDY_DAY>/` as `.png` and `.pdf`.

**Input:** `outputs/<MODEL>_stc_results_<STUDY_DAY>.xlsx`, with sheets `results` (columns `scenario`, `t`, one column per variable) and `first_stage` (columns `t` + first-stage variables).

### Scenario analysis

From the project root:

```bash
python -m viz.scenario_analysis.clustering   # representative days from k-medoids
python -m viz.scenario_analysis.k_sweep      # RMSE, EVPI/VSS and RP vs. number of scenarios
```

Figures are saved to `outputs/scenario_analysis/` as `.png` and `.pdf`.

## Files

| File | What it does |
|---|---|
| `run_plots.py` | Entry point for model-result figures. The only file you edit day to day |
| `specs.py` | Catalogue of available figures (which columns, labels, units) |
| `config.py` | Colours, plot style, folders |
| `data.py` | Loads results and scenario probabilities |
| `engine.py` | Drawing helpers (scenario fan, expected line/bar, legend, save) |
| `single_model.py` | Draws single-model figures from a spec (traces or discrete states) |
| `comparison.py` | Draws comparison figures from a spec |
| `scenario_analysis/clustering.py` | Representative days vs. all historical days (reads `data/processed/historical_data.csv`) |
| `scenario_analysis/k_sweep.py` | Effect of `k` on RMSE, EVPI/VSS and RP (values typed in the script) |

A **spec** (specification) is a description of a figure — columns, colours, labels, filename — not the drawing itself. The drawing functions read it.

**How it fits together:** `run_plots.py` loads the data, then `single_model.py` / `comparison.py` read the figure definitions from `specs.py` and draw them using `engine.py`. The `scenario_analysis` scripts are standalone, but reuse the style, colours and saving from `config.py` and `engine.py`.

## Customize

| I want to… | Edit |
|---|---|
| Change day, models or figures for a run | Constants in `run_plots.py` |
| Add a single-model figure | `specs.py` + `run_plots.py` |
| Add a discrete-state figure (Off/Standby/On…) | `specs.py` + `run_plots.py` |
| Compare a new variable across models | `specs.py` + `run_plots.py` |
| Add a new model (e.g. EH6) | `config.py` + `run_plots.py` |
| Change colours or fonts | `config.py` |
| Change figure size | `figsize` in `single_model.py` / `comparison.py` |
| Change export format or resolution | `save_fig` in `engine.py` |
| Update the `k` sweep numbers | `df` in `scenario_analysis/k_sweep.py` |
| Change the number of clusters | `N_SCENARIOS` in `scenario_analysis/clustering.py` |

### Add a single-model figure

In `specs.py`, add an entry to `SINGLE_MODEL_FIGURES`:

```python
"HSS": FigureSpec(
    key="HSS",
    traces=[
        Trace("SOC_H2", COLORS["soc_grey"], "Expected H2 SOC", kind="bar", show_scenarios=False),
        Trace("HY_in", COLORS["blue"], "H2 charge"),
        Trace("HY_out", COLORS["red"], "H2 discharge"),
    ],
    ylabel="H2 power (MW) / H2 energy (MWh)",
    filename_template="{model}_HSS_{day}",
),
```

Then add `"HSS"` to `SINGLE_MODEL_FIGURES` in `run_plots.py`.

Tips: `column` must match a column name in `results`. For state variables (SOC), use `kind="bar", show_scenarios=False`.

### Add a discrete-state figure (Off / Standby / On…)

For elements with mutually exclusive binary states, add a `StateSpec` to `SINGLE_MODEL_FIGURES` in `specs.py` (see `ELZ_STATES`):

```python
"ELZ_STATES": StateSpec(
    key="ELZ_STATES",
    state_columns=["I_off", "I_stb", "I_el"],   # binary indicator of state 0, 1, 2
    state_labels=["Off", "Standby", "On"],
    ylabel="Electrolyzer state",
    filename_template="{model}_ELZ_states_{day}",
),
```

It draws one step line per scenario; lines are slightly offset vertically so they do not overlap (the real level is the one at the axis tick). Add the key to `SINGLE_MODEL_FIGURES` in `run_plots.py` as usual. The dictionary `SINGLE_MODEL_FIGURES` in `specs.py` mixes `FigureSpec` and `StateSpec` entries; `run_single_model_plots` (in `single_model.py`) chooses the right drawing function for each.

### Compare a new variable

In `specs.py`, add an entry to `COMPARISON_ELEMENTS`:

```python
"E_DA": ComparisonSpec("E_DA", "E_DA", "Electricity from DA Market (MW)", stage="first"),
```

Then add `"E_DA"` to `COMPARISON_FIGURES` in `run_plots.py`. Use `stage="first"` for first-stage variables. The column must exist in every model being compared.

### Add a new model

1. `config.py`: add a colour, e.g. `MODEL_COLORS["EH6"] = COLORS["blue"]`.
2. Make sure `outputs/EH6_stc_results_<STUDY_DAY>.xlsx` exists.
3. `run_plots.py`: add `"EH6"` to `MODEL_NAMES`.

## Assumptions

- Hours are 1–24 (fixed x-axis in `engine.finalize_axes`).
- Every scenario has the same hours in the same order.
- Scenario names in `results` match those returned by `load_probabilities()`.
- In a `StateSpec`, the state indicators are mutually exclusive and sum to 1 (e.g. `I_off + I_stb + I_el = 1`).
