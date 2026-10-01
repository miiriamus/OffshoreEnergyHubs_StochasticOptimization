# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:22:37 2026

@author: Miriam_Ucendo
@filename: engine.py

Primitivas de bajo nivel, agnósticas al modelo. Nada aquí sabe qué es
"EH3" ni qué es "ESS" — solo sabe dibujar un fan de escenarios, una
línea/barra de valor esperado, una leyenda, y guardar una figura.
Esto es lo que mantiene single_model.py y comparison.py cortos: cualquier
figura nueva se compone reutilizando estas piezas.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def compute_expected(df: pd.DataFrame, column: str, scenarios: List, prob_dict: dict) -> pd.DataFrame:
    """Valor esperado (ponderado por probabilidad) de `column` sobre escenarios."""
    expected = None
    for s in scenarios:
        d = df[df["scenario"] == s]
        if expected is None:
            expected = d[["t", column]].copy()
            expected[column] *= prob_dict[s]
        else:
            expected[column] += prob_dict[s] * d[column].values
    return expected


def draw_scenario_fan(ax, df, column, scenarios, color, alpha=0.3, lw=1.0, zorder=2):
    """Dibuja cada escenario individual como línea tenue (contexto detrás de la media)."""
    for s in scenarios:
        d = df[df["scenario"] == s]
        ax.plot(d["t"], d[column], color=color, alpha=alpha, lw=lw, zorder=zorder)


def draw_expected_line(ax, expected_df, column, color, lw=3, zorder=3):
    ax.plot(expected_df["t"], expected_df[column], color=color, lw=lw, zorder=zorder)


def draw_expected_bar(ax, expected_df, column, color, width=0.75, zorder=1, bottom=None):
    ax.bar(
        expected_df["t"], expected_df[column],
        width=width, color=color, edgecolor="grey", zorder=zorder, bottom=bottom,
    )


def finalize_axes(ax, ylabel, xlabel="Hour"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 25))
    ax.set_xlim(0.5, 24.5)
    ax.grid(axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_legend(ax, labels: List[str], colors: List[str], ncol=None):
    """
    La leyenda se construye SIEMPRE a partir de los mismos (label, color)
    que se usaron para dibujar -> imposible que diverjan (a diferencia del
    script original, donde 'Expected discharge' se dibujaba en Rojo pero
    la leyenda la pintaba en Naranja).
    """
    handles = [Line2D([0], [0], color=c, lw=3) for c in colors]
    ax.legend(
        handles, labels,
        frameon=False,
        ncol=ncol or len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
    )


def save_fig(fig, output_folder: Path, filename: str):
    fig.tight_layout()
    fig.savefig(output_folder / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_folder / f"{filename}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
