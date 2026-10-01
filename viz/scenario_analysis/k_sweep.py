# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:52:00 2026

@author: Miriam_Ucendo
@filename: plot_k_sweep
"""

import pandas as pd
import matplotlib.pyplot as plt

from viz.config import COLORS, RESULTS_PATH, apply_style
from viz.engine import save_fig

# ─── COLORES ────────────────────────────────────────────────────────
Azul     = COLORS["blue"]
Naranja  = COLORS["orange"]

FIG_FOLDER = RESULTS_PATH / "scenario_analysis"
FIG_FOLDER.mkdir(parents=True, exist_ok=True)

# =====================================================
# DATA
# =====================================================

df = pd.DataFrame({

    "k": [4, 6, 8, 10, 12, 24, 48],

    "RMSE_price": [0.113213, 0.098926, 0.088617, 0.085907, 0.084847, 0.066921, 0.054420],
    "RMSE_wind":  [0.113535, 0.108683, 0.105574, 0.097834, 0.087115, 0.076571, 0.052973],

    "EVPI": [11262.98, 13361.41, 16139.89, 15334.19, 15488.80, 15584.24, 15430.55],
    "VSS":  [4370.06, 3802.90, 3949.62, 4007.45, 3318.05, 4249.43, 4710.27],

    "RP": [112742.96, 115825.29, 120892.55, 117902.72,
           117984.62, 118997.16, 119384.59]

})

# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================

apply_style()


# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(1, 3, figsize=(15,5))

# -----------------------------------------------------
# RMSE
# -----------------------------------------------------

ax[0].plot(
    df["k"],
    df["RMSE_price"],
    marker="o",
    linewidth=2,
    label="Price",
    color=Azul
)

ax[0].plot(
    df["k"],
    df["RMSE_wind"],
    marker="s",
    linewidth=2,
    label="Wind",
    color=Naranja
)


ax[0].set_xlabel("Number of representative scenarios")
ax[0].set_ylabel("RMSE")


ax[0].legend()

# -----------------------------------------------------
# EVPI + VSS
# -----------------------------------------------------

ax[1].plot(
    df["k"],
    df["EVPI"],
    marker="o",
    linewidth=2,
    label="EVPI",
    color=Azul
)

ax[1].plot(
    df["k"],
    df["VSS"],
    marker="s",
    linewidth=2,
    label="VSS",
    color=Naranja
)


ax[1].set_xlabel("Number of representative scenarios")
ax[1].set_ylabel("€")


ax[1].legend()

# -----------------------------------------------------
# RP
# -----------------------------------------------------

ax[2].plot(
    df["k"],
    df["RP"],
    marker="o",
    linewidth=2,
    color=Azul
)


ax[2].set_xlabel("Number of representative scenarios")
ax[2].set_ylabel("Objective value (€)")


save_fig(fig, FIG_FOLDER, "k_sweep")
