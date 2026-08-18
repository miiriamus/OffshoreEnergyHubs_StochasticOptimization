# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 05:54:52 2026

@author: Miriam_Ucendo
@filename: plot_ESS

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
study_day = "2026-05-20"

output_folder = RESULTS_PATH / study_day

# Creamos la carpeta automáticamente si no existe
if not os.path.exists(study_day):
    os.makedirs(study_day)
    print(f"📁 Carpeta '{study_day}' creada con éxito.")

# =============================================================================
# LOAD RESULTS
# =============================================================================

d2 = pd.read_excel(RESULTS_PATH /
    f"EH2_stc_results_{study_day}.xlsx",
    sheet_name="results",
)

d3 = pd.read_excel(RESULTS_PATH /
    f"EH3_stc_results_{study_day}.xlsx",
    sheet_name="results",
)

d4 = pd.read_excel(RESULTS_PATH /
    f"EH4_stc_results_{study_day}.xlsx",
    sheet_name="results",
)

SCENARIOS = sorted(d2["scenario"].unique())

# =============================================================================
# LOAD SCENARIO PROBABILITIES
# =============================================================================

historical_file = ROOT / "data" / "processed" / "historical_data.csv"
scenarios_file  = ROOT / "data" / "processed" / "scenarios_12.csv"
params_file     = ROOT / "data" / "processed" / "params.xlsx"

SCENARIOS_DATA, SCENARIOS_DATES, time_periods, PARAMS, PROB, DATA = load_input_data(
    params_file,
    historical_file,
    scenarios_file,
)

PROB_DICT = {s: PROB[s] for s in SCENARIOS_DATA}

# =============================================================================
# COLOURS
# =============================================================================

Blue = "#1F4E79"
Orange = "#D55E00"
Green = "#2A9D8F"
Rojo      = "#C44E52"   # Brick red
SOC_color = "#A6A6A6"

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
# DEF PLOT ESS
# =============================================================================

def plot_ESS(database):

    fig, ax = plt.subplots(figsize=(13,6))

    # ----------------------------------------------------------
    # Expected values
    # ----------------------------------------------------------

    soc_expected = None
    charge_expected = None
    discharge_expected = None

    for s in SCENARIOS:

        d = database[database["scenario"] == s]

        if soc_expected is None:

            soc_expected = d[["t","SOC"]].copy()
            soc_expected["SOC"] *= PROB_DICT[s]

            charge_expected = d[["t","E_c"]].copy()
            charge_expected["E_c"] *= PROB_DICT[s]

            discharge_expected = d[["t","E_d"]].copy()
            discharge_expected["E_d"] *= PROB_DICT[s]

        else:

            soc_expected["SOC"] += PROB_DICT[s] * d["SOC"].values
            charge_expected["E_c"] += PROB_DICT[s] * d["E_c"].values
            discharge_expected["E_d"] += PROB_DICT[s] * d["E_d"].values

    # ----------------------------------------------------------
    # SOC bars
    # ----------------------------------------------------------

    ax.bar(
        soc_expected["t"],
        soc_expected["SOC"],
        width=0.75,
        color=SOC_color,
        edgecolor="grey",
        label="Expected SOC",
        zorder=1,
    )

    # ----------------------------------------------------------
    # Every scenario
    # ----------------------------------------------------------

    for s in SCENARIOS:

        d = database[database["scenario"] == s]

        ax.plot(
            d["t"],
            d["E_c"],
            color=Blue,
            alpha=0.30,
            lw=1,
            zorder=2,
        )

        ax.plot(
            d["t"],
            d["E_d"],
            color=Rojo,
            alpha=0.30,
            lw=1,
            zorder=2,
        )

    # ----------------------------------------------------------
    # Expected charge/discharge
    # ----------------------------------------------------------

    ax.plot(
        charge_expected["t"],
        charge_expected["E_c"],
        color=Blue,
        lw=3,
        label="Expected charge",
        zorder=3,
    )

    ax.plot(
        discharge_expected["t"],
        discharge_expected["E_d"],
        color=Rojo,
        lw=3,
        label="Expected discharge",
        zorder=3,
    )

    # ----------------------------------------------------------
    # Plot every charging/discharging scenario
    # ----------------------------------------------------------

    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW) / Energy (MWh)")
    
    ax.set_xticks(range(1, 25))
    ax.set_xlim(0.5, 24.5)  

    ax.grid(axis="y", alpha=0.35)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    legend_handles = [
    
        plt.Rectangle((0,0),1,1,
                      facecolor="lightgrey",
                      edgecolor="grey"),
    
        Line2D([0],[0], color=Blue, lw=3),
    
        Line2D([0],[0], color=Orange, lw=3),
    
    ]
    
    ax.legend(
        legend_handles,
        [
            "Expected SOC",
            "Expected charge",
            "Expected discharge",
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5,1.10),
    )
    
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_folder, f"EH3_ESS_{study_day}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    plt.savefig(
        os.path.join(output_folder, f"EH3_ESS_{study_day}.pdf"),
        bbox_inches="tight",
    )

    plt.show()
    
    
# =============================================================================
# DEF PLOT EHP
# =============================================================================
    
def plot_EHP(database):

    fig, ax = plt.subplots(figsize=(13,6))

    # ----------------------------------------------------------
    # Expected values
    # ----------------------------------------------------------

    E3_expected = None
    H_expected = None
    C_expected = None

    for s in SCENARIOS:

        d = database[database["scenario"] == s]

        if E3_expected is None:

            E3_expected = d[["t","E_3"]].copy()
            E3_expected["E_3"] *= PROB_DICT[s]

            H_expected = d[["t","H_EHP"]].copy()
            H_expected["H_EHP"] *= PROB_DICT[s]

            C_expected = d[["t","C_EHP"]].copy()
            C_expected["C_EHP"] *= PROB_DICT[s]

        else:

            E3_expected["E_3"] += PROB_DICT[s] * d["E_3"].values
            H_expected["H_EHP"] += PROB_DICT[s] * d["H_EHP"].values
            C_expected["C_EHP"] += PROB_DICT[s] * d["C_EHP"].values
            
        # ----------------------------------------------------------
        # Plot every scenario
        # ----------------------------------------------------------
    
    for s in SCENARIOS:

        d = database[database["scenario"] == s]

        ax.plot(
            d["t"],
            d["E_3"],
            color=Blue,
            alpha=0.20,
            lw=1,
        )

        ax.plot(
            d["t"],
            d["H_EHP"],
            color=Orange,
            alpha=0.20,
            lw=1,
        )

        ax.plot(
            d["t"],
            d["C_EHP"],
            color=Green,
            alpha=0.20,
            lw=1,
        )
    # ----------------------------------------------------------
    # Expected values
    # ----------------------------------------------------------

    ax.plot(
        E3_expected["t"],
        E3_expected["E_3"],
        color=Blue,
        lw=3,
        label="E_3",
    )

    ax.plot(
        H_expected["t"],
        H_expected["H_EHP"],
        color=Orange,
        lw=3,
        label="H_EHP",
    )

    ax.plot(
        C_expected["t"],
        C_expected["C_EHP"],
        color=Green,
        lw=3,
        label="C_EHP",
    )
    
    # ---------------------------------------------------------
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    
    ax.set_xticks(range(1, 25))
    ax.set_xlim(0.5, 24.5)
    
    ax.grid(axis="y", alpha=0.35)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    legend_handles = [
    
        Line2D([0], [0], color=Blue, lw=3),
    
        Line2D([0], [0], color=Orange, lw=3),
    
        Line2D([0], [0], color=Green, lw=3),
    
    ]
    
    ax.legend(
        legend_handles,
        [
            "E_3",
            "H_EHP",
            "C_EHP",
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
    )
    
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(output_folder, f"EH3_EHP_{study_day}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    plt.savefig(
        os.path.join(output_folder, f"EH3_EHP_{study_day}.pdf"),
        bbox_inches="tight",
    )
    
    plt.show()
# =============================================================================
# PLOT EH2
# =============================================================================

#plot_ESS(d2)

# =============================================================================
# PLOT EH3
# =============================================================================

plot_ESS(d3)
#plot_EHP(d3)

for var in ["Wind_used", "E_IDA", "E_c", "E_d", "E_3", "H_EHP"]:
    print(var)
    print(d3.groupby("scenario")[var].sum())
