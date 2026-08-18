# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:53:39 2026

@author: Miriam_Ucendo
@filename: metrics_EH2_run.py

Compute the stochastic metrics EVPI, VSS, EEV, RP, WS
"""

import os
import time
import numpy as np
import pyomo.environ as pyo
import pandas as pd
from pathlib import Path
from datetime import date
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 1. IMPORTAMOS los componentes desde nuestros otros dos archivos independientes
from src.input_data import load_input_data
from src.models import EH2_model, EH2_stc_model

# =====================================================================
# CONFIGURATION
# =====================================================================

historical_file = ROOT / "data" / "processed" / "historical_data.csv"
scenarios_file  = ROOT / "data" / "processed" / "scenarios_12.csv"
params_file     = ROOT / "data" / "processed" / "params.xlsx"

# Day to optimise
study_day = "2026-05-20"
study_day = pd.to_datetime(study_day).date()

# Solver set
solver = pyo.SolverFactory("highs")

# =====================================================================
# LOAD DATA
# =====================================================================

SCENARIOS, SCENARIOS_DATES, time_periods, params, PROB, DATA = load_input_data(
    params_file, historical_file, scenarios_file)

# Create a synthetic day (very irreal) to save the mean values on it
mean_day = date(2100,1,1) 

for t in time_periods:

    DATA["Precio_IDA"][(mean_day,t)] = sum(
        PROB[s] * DATA["Precio_IDA"][(SCENARIOS_DATES[s],t)]
        for s in SCENARIOS
    )

    DATA["Wind"][(mean_day,t)] = sum(
        PROB[s] * DATA["Wind"][(SCENARIOS_DATES[s],t)]
        for s in SCENARIOS
    )
    
    DATA["DE"][(mean_day,t)] = sum(
        PROB[s] * DATA["DE"][(SCENARIOS_DATES[s],t)]
        for s in SCENARIOS
    )
    
# =====================================================================
# WS
# =====================================================================

print("--- Calculating WS (Wait-and-See) ---")

WS = 0

for s in SCENARIOS:

    model_ws = EH2_model(
        time_periods,
        params,
        DATA,
        study_day,
        SCENARIOS_DATES[s],
    )

    result = solver.solve(model_ws)
    
    print(result.solver.status)
    print(result.solver.termination_condition)

    WS += PROB[s] * pyo.value(model_ws.cost)
    
# =====================================================================
# RP
# =====================================================================

print("--- Calculating RP (Recourse Problem) ---")

model_rp = EH2_stc_model(time_periods, 
                         SCENARIOS, 
                         PROB, 
                         params, 
                         DATA, 
                         study_day, 
                         SCENARIOS_DATES,
)

# Solver
start_time = time.perf_counter()    
result = solver.solve(model_rp, tee=False)
solve_duration = time.perf_counter() - start_time  

# Metric
RP = pyo.value(model_rp.cost)

# =====================================================================
# EV
# =====================================================================
        
print("--- Calculating EV (Expected Value Problem) ---")

model_ev = EH2_model(
    time_periods,
    params,
    DATA,
    study_day,
    mean_day,      # synthetic expected scenario
)

# Solver
start_time = time.perf_counter()    
result = solver.solve(model_ev, tee=False)
solve_duration = time.perf_counter() - start_time  

# Extract first-stage decisions from the EV solution
E_DA_ev = {t: pyo.value(model_ev.E_DA[t]) for t in time_periods}
G_ev = {t: pyo.value(model_ev.G[t]) for t in time_periods}

# =====================================================================
# EEV
# =====================================================================
    
print("--- Calculating EEV (Expected Value of the EV Solution) ---")

model_eev = EH2_stc_model(
    time_periods,
    SCENARIOS,
    PROB,
    params,
    DATA,
    study_day,
    SCENARIOS_DATES,
)

# Use solution of EV
for t in time_periods:
    model_eev.E_DA[t].fix(E_DA_ev[t])
    model_eev.G[t].fix(G_ev[t])
    
# Solver
start_time = time.perf_counter()    
result = solver.solve(model_eev, tee=False)
solve_duration = time.perf_counter() - start_time 

if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    EEV = pyo.value(model_eev.cost)
else:
    EEV = float('inf') 
    print("WARNING: EEV is infeasible. First-stage decisions from the EV model cannot satisfy all scenario constraints.")
 
# =====================================================================
# DISPLAY RESULTS
# =====================================================================

EVPI = RP - WS
VSS = EEV - RP

print("\n" + "="*40)
print(" STOCHASTIC PROGRAMMING METRICS")
print("="*40)
print(f"WS  (Wait-and-See)  : {WS:.2f}")
print(f"RP  (Recourse Prob) : {RP:.2f}")
print(f"EEV (EVS / Exp. EV) : {EEV:.2f}")
print("-" * 40)
print(f"EVPI (Expected Value of Perfect Info) : {EVPI:.2f}  [= RP - WS]")
print(f"VSS  (Value of Stochastic Solution)   : {VSS:.2f}  [= EEV - RP]")
print("="*40)
