# -*- coding: utf-8 -*-
"""
single_model.py

Figuras que describen el comportamiento interno de UN modelo (SOC,
carga/descarga de batería, flujos de la bomba de calor...). Genérico
sobre `FigureSpec`: no hay una función por elemento, hay una función que
interpreta el spec.
"""

import matplotlib.pyplot as plt

from .engine import (
    compute_expected, draw_scenario_fan, draw_expected_line,
    draw_expected_bar, finalize_axes, add_legend, save_fig,
)
from .specs import SINGLE_MODEL_FIGURES


def plot_single_model_figure(model, spec, scenarios, prob_dict, output_folder, study_day):
    fig, ax = plt.subplots(figsize=(13, 6))
    df = model.results

    for trace in spec.traces:
        expected = compute_expected(df, trace.column, scenarios, prob_dict)

        if trace.show_scenarios:
            draw_scenario_fan(ax, df, trace.column, scenarios, trace.color)

        if trace.kind == "bar":
            draw_expected_bar(ax, expected, trace.column, trace.color)
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


def run_single_model_plots(model, figure_keys, scenarios, prob_dict, output_folder, study_day):
    for key in figure_keys:
        spec = SINGLE_MODEL_FIGURES[key]
        plot_single_model_figure(model, spec, scenarios, prob_dict, output_folder, study_day)
