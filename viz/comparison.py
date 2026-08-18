# -*- coding: utf-8 -*-
"""
comparison.py

Figuras que comparan UNA variable entre varios modelos EH. Solo las
variables registradas en specs.COMPARISON_ELEMENTS aparecen aquí — el
storage y otros elementos "internos" se quedan fuera a propósito
(ver comentario en specs.py).
"""

import matplotlib.pyplot as plt

from .engine import (
    compute_expected, draw_scenario_fan, draw_expected_line,
    finalize_axes, add_legend, save_fig,
)
from .specs import COMPARISON_ELEMENTS


def plot_comparison_figure(models, spec, scenarios, prob_dict, output_folder, study_day):
    fig, ax = plt.subplots(figsize=(13, 6))

    for model in models:
        color = model.color

        if spec.stage == "first":
            ax.plot(
                model.first_stage["t"], model.first_stage[spec.column],
                color=color, lw=3,
            )
        else:
            df = model.results
            draw_scenario_fan(ax, df, spec.column, scenarios, color, alpha=0.35, lw=1.2)
            expected = compute_expected(df, spec.column, scenarios, prob_dict)
            draw_expected_line(ax, expected, spec.column, color, lw=3.5)

    finalize_axes(ax, spec.ylabel)
    add_legend(
        ax,
        labels=[m.name for m in models],
        colors=[m.color for m in models],
    )

    filename = f"{spec.key}_{study_day}"
    save_fig(fig, output_folder, filename)


def run_comparison_plots(models, element_keys, scenarios, prob_dict, output_folder, study_day):
    model_list = list(models.values())
    for key in element_keys:
        spec = COMPARISON_ELEMENTS[key]
        plot_comparison_figure(model_list, spec, scenarios, prob_dict, output_folder, study_day)
