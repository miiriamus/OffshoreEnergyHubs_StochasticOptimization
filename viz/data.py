# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:22:37 2026

@author: Miriam_Ucendo
@filename: data.py

Carga de resultados (first_stage + results por escenario) y de
probabilidades de escenario. Todo lo demás en el paquete trabaja con
`ModelData`, nunca con rutas o dicts crudos.
"""

from dataclasses import dataclass
from typing import Dict, List
import sys
import pandas as pd

from .config import ROOT, RESULTS_PATH, MODEL_COLORS

sys.path.insert(0, str(ROOT))
from src.input_data import load_input_data  # noqa: E402


@dataclass
class ModelData:
    name: str
    results: pd.DataFrame        # segunda etapa, indexado por escenario
    first_stage: pd.DataFrame    # primera etapa, determinista
    color: str


def load_probabilities() -> Dict[str, float]:
    historical_file = ROOT / "data" / "processed" / "historical_data.csv"
    scenarios_file = ROOT / "data" / "processed" / "scenarios_12.csv"
    params_file = ROOT / "data" / "processed" / "params.xlsx"

    scenarios_data, _, _, _, prob, _ = load_input_data(
        params_file, historical_file, scenarios_file,
    )
    return {s: prob[s] for s in scenarios_data}


def load_model(model_name: str, study_day: str) -> ModelData:
    """Carga los resultados de un único modelo EH (p.ej. 'EH3')."""
    path = RESULTS_PATH / f"{model_name}_stc_results_{study_day}.xlsx"
    results = pd.read_excel(path, sheet_name="results")
    first_stage = pd.read_excel(path, sheet_name="first_stage")
    return ModelData(
        name=model_name,
        results=results,
        first_stage=first_stage,
        color=MODEL_COLORS[model_name],
    )


def load_models(model_names: List[str], study_day: str) -> Dict[str, ModelData]:
    return {m: load_model(m, study_day) for m in model_names}
