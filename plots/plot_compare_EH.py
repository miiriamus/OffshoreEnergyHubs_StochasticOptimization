# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:27:08 2026

@author: Miriam_Ucendo

Comparison plots between stochastic models.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.input_data import load_input_data

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_PATH = ROOT / "outputs"
study_day = RESULTS_PATH / "2026-05-20"

output_folder = study_day

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"📁 Folder '{output_folder}' created successfully.")

# =============================================================================
# LOAD RESULTS
# =============================================================================

EH1 = pd.read_excel(RESULTS_PATH / f"EH1_stc_results_{study_day}.xlsx", sheet_name="results")
EH2 = pd.read_excel(RESULTS_PATH / f"EH2_stc_results_{study_day}.xlsx", sheet_name="results")
EH3 = pd.read_excel(RESULTS_PATH / f"EH3_stc_results_{study_day}.xlsx", sheet_name="results")
EH4 = pd.read_excel(RESULTS_PATH / f"EH4_stc_results_{study_day}.xlsx", sheet_name="results")

EH1_first = pd.read_excel(RESULTS_PATH / f"EH1_stc_results_{study_day}.xlsx", sheet_name="first_stage")
EH2_first = pd.read_excel(RESULTS_PATH / f"EH2_stc_results_{study_day}.xlsx", sheet_name="first_stage")
EH3_first = pd.read_excel(RESULTS_PATH / f"EH3_stc_results_{study_day}.xlsx", sheet_name="first_stage")
EH4_first = pd.read_excel(RESULTS_PATH / f"EH4_stc_results_{study_day}.xlsx", sheet_name="first_stage")

SCENARIOS = sorted(EH1["scenario"].unique())

# =============================================================================
# LOAD SCENARIO PROBABILITIES
# =============================================================================

historical_file = ROOT / "data" / "processed" / "historical_data.csv"
scenarios_file = ROOT / "data" / "processed" / "scenarios_12.csv"
params_file = ROOT / "data" / "processed" / "params.xlsx"

SCENARIOS_DATA, SCENARIOS_DATES, time_periods, PARAMS, PROB, DATA = load_input_data(
    params_file,
    historical_file,
    scenarios_file,
)

PROB_DICT = {s: PROB[s] for s in SCENARIOS_DATA}

# =============================================================================
# COLOURS
# =============================================================================

Azul     = "#1F4E79"
Naranja  = "#D55E00"
Verde    = "#2A9D8F"
Rojo      = "#C44E52"   # Brick red


MODELS = [
    ("EH1", EH1, EH1_first, Azul),
    ("EH2", EH2, EH2_first, Naranja),
    ("EH3", EH3, EH3_first, Verde),
    ("EH4", EH4, EH4_first, Rojo),
]

# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({

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
    "axes.linewidth": 0.8

})

# =============================================================================
# FIRST-STAGE VARIABLES
# =============================================================================

def plot_first_stage(column, ylabel, filename):

    plt.figure(figsize=(13, 6))

    for _, _, first_stage, color in MODELS:

        plt.plot(
            first_stage["t"],
            first_stage[column],
            color=color,
            lw=3,
        )

    plt.xlabel("Hour")
    plt.ylabel(ylabel)

    plt.xticks(range(1, 25))
    plt.xlim(1, 24)

    plt.grid(axis="y", alpha=0.35)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=Azul, lw=3),
        Line2D([0], [0], color=Naranja, lw=3),
        Line2D([0], [0], color=Verde, lw=3),
    ]

    plt.legend(
        legend_handles,
        ["EH1", "EH2", "EH3", "EH4"],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(study_day, f"{filename}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(study_day, f"{filename}.pdf"),
        bbox_inches="tight",
    )

    plt.show()
    
# =============================================================================
# SECOND-STAGE VARIABLES
# =============================================================================

def plot_second_stage(column, ylabel, filename, MODELS, weighted=True):

    plt.figure(figsize=(13, 6))

    for _, dataframe, _, color in MODELS:

        
        expected = None

        for s in SCENARIOS:
        
            d = dataframe[dataframe["scenario"] == s]
        
            if expected is None:
                expected = d[["t", column]].copy()
                expected[column] *= PROB_DICT[s]
            else:
                expected[column] += PROB_DICT[s] * d[column].values
        
            # Plot every scenario
            plt.plot(
                d["t"],
                d[column],
                color=color,
                lw=1.2,
                alpha=0.35,
            )
            
        # Plot expected value
        plt.plot(
            expected["t"],
            expected[column],
            color=color,
            lw=3.5,
        )


    plt.xlabel("Hour")
    plt.ylabel(ylabel)

    plt.xticks(range(1, 25))
    plt.xlim(1, 24)

    plt.grid(axis="y", alpha=0.35)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=Azul, lw=3),
        Line2D([0], [0], color=Naranja, lw=3),
        Line2D([0], [0], color=Verde, lw=3),
    ]

    plt.legend(
        legend_handles,
        ["EH1", "EH2", "EH3"],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(study_day, f"{filename}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(study_day, f"{filename}.pdf"),
        bbox_inches="tight",
    )

    plt.show()
    
# =============================================================================
# FIRST-STAGE VARIABLES
# =============================================================================

#plot_first_stage("E_DA", "Electricity from DA Market (MW)", f"E_DA_{study_day}")
plot_first_stage("G", "Gas Purchased (MW)", f"G_{study_day}")

# =============================================================================
# SECOND-STAGE VARIABLES
# =============================================================================

# Electricity
plot_second_stage("Wind_used", "Wind Power (MW)", f"Wind_used_{study_day}", MODELS)
plot_second_stage("Curt", "Curtailed Wind (MW)", f"Curt_{study_day}", MODELS)
#plot_second_stage("E", "Electricity Flow (MW)", f"E_{study_day}")
#plot_second_stage("E_IDA", "Intraday Electricity (MW)", f"E_IDA_{study_day}")
#plot_second_stage("E_TODAY", "Today's Market Electricity (MW)", f"E_TODAY_{study_day}")

# Gas
plot_second_stage("G1", "CHP Gas (MW)", f"G1_{study_day}", MODELS)
plot_second_stage("G2", "Furnace Gas (MW)", f"G2_{study_day}", MODELS)

# Heat
#plot_second_stage("H1", "Heat to Demand (MW)", f"H1_{study_day}")
#plot_second_stage("H2", "Heat to Chiller (MW)", f"H2_{study_day}")

# ESS
#plot_second_stage("SOC", "State of Charge (MWh)", f"SOC_{study_day}")
#plot_second_stage("E_c", "Battery Charge (MW)", f"E_c_{study_day}")
#plot_second_stage("E_d", "Battery Discharge (MW)", f"E_d_{study_day}")

# Heat Pump
#plot_second_stage("H_EHP", "Heat Pump Heat (MW)", f"H_EHP_{study_day}", MODELS)
#plot_second_stage("C_EHP", "Heat Pump Cooling (MW)", f"C_EHP_{study_day}")
#plot_second_stage("E_3", "Heat Pump Electricity (MW)", f"E_3_{study_day}", MODELS)

