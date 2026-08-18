# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:04:27 2026

@author: Miriam_Ucendo
@filename: EH3_stc_run.py

EH model 2, stochastic optimization model run
"""

import time
import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.input_data import load_input_data
from src.models import EH3_stc_model

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

# Archivos de entrada
historical_file = ROOT / "data" / "processed" / "historical_data.csv"
scenarios_file = ROOT / "data" / "processed" / "scenarios_12.csv"
params_file = ROOT / "data" / "processed" / "params.xlsx"

# Day to optimise
study_day = "2026-05-20"
study_day = pd.to_datetime(study_day).date()

# Guardar solución óptima
OUTPUT_PATH = ROOT / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)
output_file = OUTPUT_PATH / f"EH3_stc_results_{study_day}.xlsx"

# =====================================================================
# LOAD DATA
# =====================================================================

SCENARIOS, SCENARIOS_DATES, time_periods, PARAMS, PROB, DATA = load_input_data(
    params_file, historical_file, scenarios_file)

# =====================================================================
# BUILD MODEL
# =====================================================================

print(f"Study day: {study_day}")

model = EH3_stc_model(
    time_periods=time_periods,
    SCENARIOS=SCENARIOS,
    PROB=PROB,
    p=PARAMS,
    DATA=DATA,
    study_day=study_day,
    scenario_days=SCENARIOS_DATES
)

# =====================================================================
# SOLVER
# =====================================================================

solver = pyo.SolverFactory("highs")

start = time.perf_counter()
result = solver.solve(model, tee=True)
elapsed = time.perf_counter() - start

print(f"Status      : {result.solver.termination_condition}")
print(f"Solve time  : {elapsed:.2f} s")

if result.solver.termination_condition != pyo.TerminationCondition.optimal:
    raise RuntimeError("No optimal solution found.")

print(f"Expected cost: {pyo.value(model.cost):.2f}")

# =====================================================================
# SAVE RESULTS
# =====================================================================

rows = []
dual_rows = []

for s in SCENARIOS:

    scenario_day = SCENARIOS_DATES[s]

    for t in time_periods:

        rows.append({

            "scenario": s,
            "scenario_day": scenario_day,
            "t": t,

            # Inputs
            "DE": pyo.value(model.De[t]),
            "DH": pyo.value(model.Dh[t]),
            "DC": pyo.value(model.Dc[t]),
            "Wind": pyo.value(model.Wind[t, s]),

            # Electricity
            "Wind_used": pyo.value(model.Wind_used[t, s]),
            "Curt": pyo.value(model.Wind[t, s]) - pyo.value(model.Wind_used[t, s]),
            "E_DA": pyo.value(model.E_DA[t]),
            "E_IDA": pyo.value(model.E_IDA[t, s]),
            "E_2": pyo.value(model.E_2[t, s]),
            "E_3": pyo.value(model.E_3[t, s]),

            # ESS
            "E_c": pyo.value(model.E_c[t, s]),
            "E_d": pyo.value(model.E_d[t, s]),
            "SOC": pyo.value(model.SOC[t, s]),
            "I_ch": pyo.value(model.I_ch[t, s]),

            # Gas
            "G": pyo.value(model.G[t]),
            "G1": pyo.value(model.G1[t, s]),
            "G2": pyo.value(model.G2[t, s]),

            # Heat
            "H1": pyo.value(model.H1[t, s]),
            "H2": pyo.value(model.H2[t, s]),

            # Electric Heat Pump (EHP)
            "H_EHP": pyo.value(model.H_EHP[t, s]),
            "C_EHP": pyo.value(model.C_EHP[t, s]),
            "I_h": pyo.value(model.I_h[t, s]),

        })


# =====================================================================
# FIRST-STAGE DECISIONS
# =====================================================================

first_stage = []

for t in time_periods:

    first_stage.append({

        "t": t,

        "E_DA": pyo.value(model.E_DA[t]),
        "G": pyo.value(model.G[t]),

        "lambda_DA": pyo.value(model.lam_DA[t]),
        "lambda_g": pyo.value(model.lam_g[t]),

    })

# =====================================================================
# EXPORT
# =====================================================================

with pd.ExcelWriter(output_file) as writer:

    pd.DataFrame(rows).to_excel(
        writer,
        sheet_name="results",
        index=False
    )

    pd.DataFrame(first_stage).to_excel(
        writer,
        sheet_name="first_stage",
        index=False
    )


print(f"\nResults saved to {output_file}")
