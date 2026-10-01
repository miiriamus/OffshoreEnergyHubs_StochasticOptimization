# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:22:37 2026

@author: Miriam_Ucendo
@filename: single_model.py

Figuras que describen el comportamiento interno de UN modelo (SOC,
carga/descarga de batería, flujos de la bomba de calor...). Genérico
sobre `FigureSpec`: no hay una función por elemento, hay una función que
interpreta el spec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.engine import (
    compute_expected, draw_scenario_fan, draw_expected_line,
    draw_expected_bar, finalize_axes, add_legend, save_fig,
)
from viz.specs import SINGLE_MODEL_FIGURES


def plot_single_model_figure(model, spec, scenarios, prob_dict, output_folder, study_day):
    fig, ax = plt.subplots(figsize=(13, 6))
    df = model.results

    stack_bottom = None

    for trace in spec.traces:
        expected = compute_expected(df, trace.column, scenarios, prob_dict)

        if trace.show_scenarios:
            draw_scenario_fan(ax, df, trace.column, scenarios, trace.color)

        if trace.kind == "bar":
            draw_expected_bar(ax, expected, trace.column, trace.color)
        elif trace.kind == "stack":
            draw_expected_bar(ax, expected, trace.column, trace.color, bottom=stack_bottom)
            values = expected[trace.column].values
            stack_bottom = values if stack_bottom is None else stack_bottom + values
        else:
            draw_expected_line(ax, expected, trace.column, trace.color, lw=trace.lw)

    finalize_axes(ax, spec.ylabel)
    add_legend(
        ax,
        labels=[t.label for t in spec.traces],
        colors=[t.color for t in spec.traces],
    )

    filename = spec.filename_template.format(model=model.name, day=study_day)
    save_fig(fig, output_folder, filename)
    
    
def plot_state_figure(model, spec, scenarios, prob_dict, output_folder, study_day):
    """
    Estado discreto de un elemento (p.ej. electrolizador: 0=Off, 1=Standby, 2=On).
    Una línea escalonada por escenario. Las líneas se desplazan ligeramente en
    vertical para que no se tapen entre sí; el nivel real es el de la marca del eje.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    df = model.results
 
    # tab20 alterna tonos oscuros/claros del mismo color: se reordena para separar escenarios contiguos
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    colors = [plt.cm.tab20(order[i % 20]) for i in range(len(scenarios))]
    offsets = np.linspace(-0.22, 0.22, len(scenarios))
 
    for i, s in enumerate(scenarios):
        d = df[df["scenario"] == s].sort_values("t")
        state = sum(level * d[col].values for level, col in enumerate(spec.state_columns))
        ax.step(
            d["t"], state + offsets[i],
            where="mid", color=colors[i], lw=1.8, label=f"S{s}", zorder=2,
        )
 
    finalize_axes(ax, spec.ylabel)
    ax.set_yticks(range(len(spec.state_labels)))
    ax.set_yticklabels(spec.state_labels)
    ax.set_ylim(-0.5, len(spec.state_labels) - 0.5)
    ax.legend(
        frameon=False, ncol=6, fontsize=14,
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
    )
 
    filename = spec.filename_template.format(model=model.name, day=study_day)
    save_fig(fig, output_folder, filename)


def run_single_model_plots(model, figure_keys, scenarios, prob_dict, output_folder, study_day):
    for key in figure_keys:
        spec = SINGLE_MODEL_FIGURES[key]
        if hasattr(spec, "state_columns"):   # StateSpec (sin isinstance: robusto a recargas de módulos en Spyder)
            plot_state_figure(model, spec, scenarios, prob_dict, output_folder, study_day)
        else:
            plot_single_model_figure(model, spec, scenarios, prob_dict, output_folder, study_day)
