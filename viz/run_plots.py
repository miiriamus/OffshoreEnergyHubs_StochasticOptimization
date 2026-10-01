# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:22:37 2026

@author: Miriam_Ucendo
@filename: run_plots.py

Único archivo que se toca en el día a día: elige el día de estudio, qué
modelos cargar, y qué figuras/elementos generar. Todo lo demás vive en
viz/ y no debería necesitar cambios cuando solo quieres validar un
elemento nuevo o comparar una variable nueva.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from viz.config import apply_style, get_output_folder
from viz.data import load_models, load_probabilities
from viz.single_model import run_single_model_plots
from viz.comparison import run_comparison_plots

STUDY_DAY = "2026-05-20"
MODEL_NAMES = ["EH1", "EH2", "EH3", "EH5"]

# Elementos internos a validar de UN modelo concreto (storage, HP...)
SINGLE_MODEL_TO_PLOT = "EH5"
SINGLE_MODEL_FIGURES = ["ESS", "HSS", "ELZ_STATES", "EHP"]   # "EHP" también disponible

# Elementos que sí tiene sentido comparar entre modelos
COMPARISON_FIGURES = ["Wind_used", "Curt", "G1", "G2", "G"]


def main():
    apply_style()
    output_folder = get_output_folder(STUDY_DAY)

    models = load_models(MODEL_NAMES, STUDY_DAY)
    prob_dict = load_probabilities()
    scenarios = sorted(models[MODEL_NAMES[0]].results["scenario"].unique())

    run_single_model_plots(
        models[SINGLE_MODEL_TO_PLOT], SINGLE_MODEL_FIGURES,
        scenarios, prob_dict, output_folder, STUDY_DAY,
    )

    run_comparison_plots(
        models, COMPARISON_FIGURES,
        scenarios, prob_dict, output_folder, STUDY_DAY,
    )

    print("All figures saved successfully!")


if __name__ == "__main__":
    main()
