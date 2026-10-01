# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:49:36 2026

@author: Miriam_Ucendo
@filename: clustering_plot
"""
import numpy as np

import matplotlib.pyplot as plt


import pandas as pd
import tsam.timeseriesaggregation as tsam

from viz.config import ROOT, RESULTS_PATH, apply_style
from viz.engine import save_fig

# =====================================================
# CONFIGURACIÓN
# =====================================================

N_SCENARIOS = 12

DATA = ROOT / "data"

INPUT = DATA / "processed" / "historical_data.csv"

OUTPUT = DATA / "processed" / f"scenarios_{N_SCENARIOS}.csv"

FIG_FOLDER = RESULTS_PATH / "scenario_analysis"
FIG_FOLDER.mkdir(parents=True, exist_ok=True)


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
        "Produccion_Eolica"
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
    },

    extremePeriodMethod="new_cluster_center"

)

aggregation.createTypicalPeriods()

print(type(aggregation.clusterPeriodNoOccur))
print(aggregation.clusterPeriodNoOccur)

for attr in dir(aggregation):
    if "cluster" in attr.lower():
        print(attr)
        
print(type(aggregation.clusterPeriodIdx))
print(aggregation.clusterPeriodIdx[:10])



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
# FIGURE SETTINGS
# =====================================================

variables = [
    ("Produccion_Eolica", "Wind generation (MWh)"),
    ("Precio_IDA", "Intraday electricity price (€/MWh)"),
]

colors = plt.cm.tab10(np.linspace(0, 1, N_SCENARIOS))

typical = aggregation.typicalPeriods

cluster_assignment = aggregation.clusterOrder
cluster_occurrences = aggregation.clusterPeriodNoOccur

n_days = len(cluster_assignment)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10),
    sharex=True
)

# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================

apply_style()

# =====================================================
# LOOP VARIABLES
# =====================================================

for row, (var, ylabel) in enumerate(variables):

    ax_left = axes[row,0]
    ax_right = axes[row,1]

    # ---------------------------------------------
    # Representative days
    # ---------------------------------------------

    for c in range(N_SCENARIOS):

        profile = typical.loc[c][var].values

        prob = 100 * cluster_occurrences[c] / n_days

        ax_left.plot(
            range(24),
            profile,
            color=colors[c],
            lw=2.5,
            label=f"C{c+1} ({prob:.1f}%)"
        )

    ax_left.set_ylabel(ylabel)

    ax_left.grid(alpha=.3)

    # ---------------------------------------------
    # Historical days coloured by cluster
    # ---------------------------------------------

    for day in range(n_days):

        cluster = cluster_assignment[day]

        start = day * 24
        end = (day + 1) * 24

        profile = raw.iloc[start:end][var].values

        ax_right.plot(
            range(24),
            profile,
            color=colors[cluster],
            alpha=0.48,
            lw=0.8
        )

    # Plot representative day on top
    for c in range(N_SCENARIOS):

        profile = typical.loc[c][var].values

        ax_right.plot(
            range(24),
            profile,
            color=colors[c],
            lw=2.8
        )

    ax_right.grid(alpha=.3)

# =====================================================
# AXES
# =====================================================

for ax in axes.flatten():
    ax.set_xlim(0,23)

axes[1,0].set_xlabel("Hour")
axes[1,1].set_xlabel("Hour")

# =====================================================
# LEGEND
# =====================================================

handles, labels = axes[0,0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.10),
    ncol=min(6, N_SCENARIOS),
    frameon=False
)

save_fig(fig, FIG_FOLDER, f"clustering_k{N_SCENARIOS}")
