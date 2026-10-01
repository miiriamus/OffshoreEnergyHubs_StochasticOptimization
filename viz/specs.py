# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:22:37 2026

@author: Miriam_Ucendo
@filename: specs.py

Catálogo declarativo de "qué se puede plotear". Añadir un elemento nuevo
al pipeline de validación = añadir una entrada aquí. No hace falta
escribir una función nueva salvo que el layout visual sea genuinamente
distinto (composición de varias trazas en una figura) — ver
single_model.py / comparison.py.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field
from typing import List
from viz.config import COLORS


@dataclass
class Trace:
    column: str
    color: str
    label: str
    kind: str = "line"           # "line" | "bar" | "stack" (barras apiladas, en orden de trazas)
    lw: float = 3
    show_scenarios: bool = True   # dibujar el fan de escenarios individuales detrás


@dataclass
class FigureSpec:
    """Figura de un único modelo, compuesta por una o varias trazas en los mismos ejes."""
    key: str
    traces: List[Trace]
    ylabel: str
    filename_template: str  # p.ej. "{model}_ESS_{day}"
    
    
@dataclass
class StateSpec:
    """
    Estado discreto de un elemento: una línea escalonada por escenario.
    `state_columns` son los indicadores binarios de cada estado, en orden
    (nivel 0, 1, 2...); `state_labels` los nombres que aparecen en el eje y.
    """
    key: str
    state_columns: List[str]
    state_labels: List[str]
    ylabel: str
    filename_template: str


@dataclass
class ComparisonSpec:
    """Figura cruzada entre modelos: misma columna, una línea por modelo."""
    key: str
    column: str
    ylabel: str
    stage: str = "second"   # "first" | "second"


# -------------------------------------------------------------------
# CATÁLOGO — figuras internas de un modelo
# (elementos que NO tiene sentido comparar entre modelos: storage, HP...)
# -------------------------------------------------------------------
SINGLE_MODEL_FIGURES = {
    "ESS": FigureSpec(
        key="ESS",
        traces=[
            Trace("SOC", COLORS["soc_grey"], "Expected SOC", kind="bar", show_scenarios=False),
            Trace("E_c", COLORS["blue"], "Expected charge"),
            Trace("E_d", COLORS["red"], "Expected discharge"),
        ],
        ylabel="Power (MW) / Energy (MWh)",
        filename_template="{model}_ESS_{day}",
    ),
    "EHP": FigureSpec(
        key="EHP",
        traces=[
            Trace("E_3", COLORS["blue"], "E_3"),
            Trace("H_EHP", COLORS["orange"], "H_EHP"),
            Trace("C_EHP", COLORS["green"], "C_EHP"),
        ],
        ylabel="Power (MW)",
        filename_template="{model}_EHP_{day}",
    ),
    # --- Para tu OEH: añadir aquí electrolizador, desalación, H2 storage...
    # "ELECTROLYZER": FigureSpec(
    #     key="ELECTROLYZER",
    #     traces=[
    #         Trace("E_elz", COLORS["blue"], "Electrolyzer power"),
    #         Trace("H2_out", COLORS["green"], "H2 production"),
    #     ],
    #     ylabel="Power (MW) / H2 flow (kg/h)",
    #     filename_template="{model}_ELECTROLYZER_{day}",
    # ),
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


    # Estados del electrolizador: 0 = Off, 1 = Standby, 2 = On (electrolisis).
    # I_off + I_stb + I_el = 1 en cada hora y escenario.
    "ELZ_STATES": StateSpec(
        key="ELZ_STATES",
        state_columns=["I_off", "I_stb", "I_el"],
        state_labels=["Off", "Standby", "On"],
        ylabel="Electrolyzer state",
        filename_template="{model}_ELZ_states_{day}",
    ),
}

# -------------------------------------------------------------------
# CATÁLOGO — comparación entre modelos
# Solo lo que realmente es comparable (curtailment, gas, wind_used...).
# Storage se queda fuera a propósito: comparar SOC entre modelos con
# baterías de tamaño distinto no aporta información.
# -------------------------------------------------------------------
COMPARISON_ELEMENTS = {
    "Wind_used": ComparisonSpec("Wind_used", "Wind_used", "Wind Power (MW)"),
    "Curt":      ComparisonSpec("Curt", "Curt", "Curtailed Wind (MW)"),
    "G1":        ComparisonSpec("G1", "G1", "CHP Gas (MW)"),
    "G2":        ComparisonSpec("G2", "G2", "Furnace Gas (MW)"),
    "G":         ComparisonSpec("G", "G", "Gas Purchased (MW)", stage="first"),
    # "E_DA":    ComparisonSpec("E_DA", "E_DA", "Electricity from DA Market (MW)", stage="first"),
}
