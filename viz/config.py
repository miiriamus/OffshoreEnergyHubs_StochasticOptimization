# -*- coding: utf-8 -*-
"""
config.py

Fuente única de verdad para: paleta de colores, estilo matplotlib y rutas
de salida. Nada aquí sabe qué es un "EH3" ni qué es una "SOC" — solo
configuración transversal.
"""

from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "outputs"

# -----------------------------------------------------------------------
# Paleta de colores (una sola vez, en un solo sitio)
# -----------------------------------------------------------------------
COLORS = {
    "blue": "#1F4E79",
    "orange": "#D55E00",
    "green": "#2A9D8F",
    "red": "#C44E52",
    "soc_grey": "#A6A6A6",
}

# Color fijo por modelo -> todas las figuras (single-model y comparación)
# usan el mismo color para EH1..EH4, no hay que recordarlo cada vez.
MODEL_COLORS = {
    "EH1": COLORS["blue"],
    "EH2": COLORS["orange"],
    "EH3": COLORS["green"],
    "EH4": COLORS["red"],
}

MPL_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "Arial",
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 18,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
}


def apply_style() -> None:
    plt.rcParams.update(MPL_STYLE)


def get_output_folder(study_day: str) -> Path:
    folder = RESULTS_PATH / study_day
    folder.mkdir(parents=True, exist_ok=True)
    return folder
