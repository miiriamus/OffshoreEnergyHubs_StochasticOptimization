# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:03:07 2026

@author: Miriam_Ucendo
@filename: generator_scenarios

"Given a time-series for all the random variables, generate 
a desired number of scenarios"
"""

from pathlib import Path
import numpy as np

import pandas as pd
import tsam.timeseriesaggregation as tsam

# =====================================================
# CONFIGURACIÓN
# =====================================================

N_SCENARIOS = 12

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "processed" / "historical_data.csv"

OUTPUT = ROOT / "processed" / f"scenarios_{N_SCENARIOS}.csv"


print("=== Generador de escenarios ===\n")

# =====================================================
# LECTURA
# =====================================================

print("Leyendo datos históricos...")

hist = pd.read_csv(INPUT)

# Tsam necesita un índice datetime
hist["Fecha"] = pd.to_datetime(hist["Fecha"])

# Índice horario continuo
hist["datetime"] = (
    hist["Fecha"] +
    pd.to_timedelta(hist["Hora"] - 1, unit="h")
)

hist = hist.set_index("datetime")

conteo = hist.groupby("Fecha").size()

print(conteo[conteo != 24])

print(hist)

# =====================================================
# VARIABLES ALEATORIAS
# =====================================================

raw = hist[
    [
        "Precio_IDA",
        "Produccion_Eolica",
        "DE",
        "DH",
        "DC"
    ]
].copy()

# =====================================================
# TSAM
# =====================================================

print("Generando escenarios...\n")

aggregation = tsam.TimeSeriesAggregation(

    raw,

    noTypicalPeriods=N_SCENARIOS,

    hoursPerPeriod=24,

    clusterMethod="k_medoids",

    solver="highs",

    weightDict={
        "Precio_IDA": 1.0,
        "Produccion_Eolica": 1.0,
        "DE": 1.0,
        "DH": 1.0,
        "DC": 1.0,
    },

    extremePeriodMethod="new_cluster_center"

)

aggregation.createTypicalPeriods()

print(type(aggregation.clusterPeriodNoOccur))
print(aggregation.clusterPeriodNoOccur)

print(type(aggregation.typicalPeriods))
print(aggregation.typicalPeriods.head())

# =====================================================
# ESCENARIOS REPRESENTATIVOS
# =====================================================

cluster_centers = aggregation.clusterCenterIndices
occurrences = aggregation.clusterPeriodNoOccur

# Lista de días del histórico
days = (
    hist["Fecha"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

# Fechas correspondientes a los medoides
fechas = [
    days.iloc[i].date()
    for i in cluster_centers
]

# Probabilidades
total = sum(occurrences.values())

probabilidades = [
    occurrences[i] / total
    for i in sorted(occurrences.keys())
]

# DataFrame de salida
scenarios = pd.DataFrame({
    "Escenario": range(1, len(cluster_centers) + 1),
    "Cluster_center": cluster_centers,
    "Dia_representativo": fechas,
    "Probabilidad": probabilidades
})

# =====================================================
# EXPORTAR
# =====================================================

scenarios.to_csv(

    OUTPUT,

    index=False,

    encoding="utf-8"

)

print("Escenarios generados correctamente.\n")

print(scenarios)

print("\nArchivo guardado en:")

print(OUTPUT)


# =====================================================
# VALIDITY
# =====================================================

validation = aggregation.accuracyIndicators()

print(validation)
