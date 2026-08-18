# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:22:37 2026

@author: Miriam_Ucendo
@filename: EH1_run.py

EH model 1 : CHP, Transformer, Furnace, Absorption Chiller

"""

import os
import time
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import
from src.input_data import load_input_data
from src.models import EH1_model

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

# Archivos de entrada
historical_file = ROOT / "data" / "processed" / "historical_data.csv"
scenarios_file = ROOT / "data" / "processed" / "scenarios_12.csv"
params_file = ROOT / "data" / "processed" / "params.xlsx"

study_day = "2026-05-20"
study_day = pd.to_datetime(study_day).date()

# Guardar resultados
optimal_sol = []
dual_records = []

# Archivo Excel donde guardaremos los resultados finales
OUTPUT_PATH = ROOT / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)
output_file = OUTPUT_PATH / f"EH1_results_{study_day}.xlsx"

# Limpieza previa del archivo de salida si ya existe
if os.path.exists(output_file):
    try:
        os.remove(output_file)
    except PermissionError:
        print(f"⚠️ ERROR: No se pudo eliminar '{output_file}'. Ciérralo en Excel primero.")

def safe_value(expr):
    """Función auxiliar para extraer valores de Pyomo de forma segura."""
    try:
        val = pyo.value(expr)
        return val if val is not None else 0.0
    except:
        return 0.0

# =====================================================================
# CARGA DE DATOS
# =====================================================================

SCENARIOS, SCENARIOS_DATES, time_periods, PARAMS, PROBAB, DATA = load_input_data(
    params_file,
    historical_file,
    scenarios_file
)


# =====================================================================
# EJECUCIÓN
# =====================================================================

with pd.ExcelWriter(output_file, mode='w') as writer:
    
    # Inicialización temporal para proteger el motor de openpyxl
    pd.DataFrame().to_excel(writer, sheet_name="Temp_Init")
    
    for s in SCENARIOS:

        scenario_day = SCENARIOS_DATES[s]
    
        print(f"\n--- Escenario {s}: {scenario_day} ---")
    
        model = EH1_model(time_periods, PARAMS, DATA, study_day, scenario_day)
        
        # Solver        
        solver = pyo.SolverFactory("highs")
        start_time = time.perf_counter()    
        result = solver.solve(model, tee=False)
        solve_duration = time.perf_counter() - start_time   

        # Results                
        print(f"Status       : {result.solver.termination_condition}")
        print(f"Solve Time   : {solve_duration:.2f} seconds")
        try:
            cost_val = pyo.value(model.cost)
            print(f"Coste Mínimo      : {cost_val:.2f}\n")
            optimal_sol.append(cost_val)
        except ValueError:
            print("Coste Mínimo      : No se pudo calcular (Modelo Infactible)\n")
            optimal_sol.append(0.0)
            continue

        # Save data        
        res = []
        for t in time_periods:
            res.append({
                "t":            t,
                # Inputs
                "DE":           pyo.value(model.De[t]),
                "DH":           pyo.value(model.Dh[t]),
                "DC":           pyo.value(model.Dc[t]),
                "Wind":         pyo.value(model.Wind[t]),
                # Variables
                "Wind_used":    pyo.value(model.Wind_used[t]),
                "Curt":         pyo.value(model.Wind[t]) - pyo.value(model.Wind_used[t]),
                "E_DA":         pyo.value(model.E_DA[t]),
                "E_IDA":        pyo.value(model.E_IDA[t]),
                "E":            pyo.value(model.E[t]),
                "G":            pyo.value(model.G[t]),
                "G1":           pyo.value(model.G1[t]),
                "G2":           pyo.value(model.G2[t]),
                "H1":           pyo.value(model.H1[t]),
                "H2":           pyo.value(model.H2[t]),
            })

        # Create sheet in the results excel
        sheet_name = f"S{s}_{scenario_day.strftime('%m%d')}"
        
        pd.DataFrame(res).to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Pestaña '{s}' guardada correctamente.")
        
    # Limpiar la pestaña vacía inicial de inicialización
    try:
        if len(writer.sheets) > 1 and "Temp_Init" in writer.sheets:
            writer.book.remove(writer.book["Temp_Init"])
    except:
        pass

print("\n¡Bucle de optimización finalizado!")
print(f"Promedio de Costes Mínimos: {np.mean(optimal_sol):.2f}")
                                                           
