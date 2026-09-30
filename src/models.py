# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 09:43:51 2026

@author: Miriam_Ucendo
@file_name: models.py (The .mod equivalent)
Defines the structure of the Energy Hub optimization model.
"""

import pyomo.environ as pyo

eps = 1

## actually used models ##
###############################################################################
######  ENERGY HUB 1    ##############################
###############################################################################

def EH1_model(time_periods, p, DATA, study_day, scenario_day):
    
    # =====================================================================
    # MODEL
    # =====================================================================

    model = pyo.ConcreteModel(name="Full_Energy_Hub")

    # =====================================================================
    # SETS
    # =====================================================================

    model.T = pyo.Set(initialize=time_periods)

    # =====================================================================
    # PARAMETERS
    # =====================================================================

    # ---------- Scalar parameters ----------

    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"])
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])

    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])

    # ---------- Time-dependent parameters ----------

    model.De = pyo.Param(model.T, initialize={t: DATA["DE"][(study_day, t)] for t in time_periods})
    model.Dh = pyo.Param(model.T, initialize={t: DATA["DH"][(study_day, t)] for t in time_periods})
    model.Dc = pyo.Param(model.T, initialize={t: DATA["DC"][(study_day, t)] for t in time_periods})
    
    model.lam_DA = pyo.Param(model.T, initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods})
    model.lam_g = pyo.Param(model.T, initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods})
    
    model.lam_IDA = pyo.Param(model.T, initialize={t: DATA["Precio_IDA"][(scenario_day, t)] for t in time_periods})
    model.Wind = pyo.Param(model.T, initialize={t: DATA["Wind"][(scenario_day, t)] for t in time_periods})
    
    # =====================================================================
    # VARIABLES
    # =====================================================================

    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_IDA = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.Wind_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.E = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.H1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # =====================================================================
    # OBJECTIVE
    # =====================================================================

    # min cost = sum_t [ lambda_DA * E_DA + lambda_TODAY * E_TODAY + lambda_g * G ]
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t]
            + m.lam_IDA[t] * m.E_IDA[t]
            + m.lam_g[t] * m.G[t]
            + eps * (m.Wind[t] - m.Wind_used[t])
            for t in m.T
        )

    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # =====================================================================
    # CONSTRAINTS
    # =====================================================================

    # Electricity balance

    def eq_b(m, t):
        return (
            m.Wind_used[t]
            + m.E_DA[t]
            + m.E_IDA[t]
            ==
            m.E[t]
        )

    model.eq_b = pyo.Constraint(model.T, rule=eq_b)

    # Wind utilization
    def eq_curt(m, t):
        return m.Wind_used[t] <= m.Wind[t]
    model.eq_curt = pyo.Constraint(model.T, rule=eq_curt)

#### Remark: The variable Curt_t defined in the model is not present here 
####         because is redundant

    # Electricity demand
    def eq_c(m, t):
        return (
            m.eta_ee * m.E[t]
            + m.eta_ge * m.G1[t]
            ==
            m.De[t]
        )
    model.eq_c = pyo.Constraint(model.T, rule=eq_c)

    # Gas balance
    def eq_d(m, t):
        return (
            m.G[t] 
            == 
            m.G1[t] + m.G2[t]
            )
    model.eq_d = pyo.Constraint(model.T, rule=eq_d)

    # Furnace balance
    def eq_e(m, t):
        return (
            m.eta_ghf * m.G2[t]
            ==
            m.H1[t] + m.H2[t]
        )
    model.eq_e = pyo.Constraint(model.T, rule=eq_e)

    # Heat demand
    def eq_f(m, t):
        return (
            m.eta_gh * m.G1[t]
            + m.H1[t]
            ==
            m.Dh[t]
        )
    model.eq_f = pyo.Constraint(model.T, rule=eq_f)

    # Cooling demand
    def eq_g(m, t):
        return (
            m.eta_hc * m.H2[t]
            ==
            m.Dc[t]
        )
    model.eq_g = pyo.Constraint(model.T, rule=eq_g)

    # CHP capacity
    def limit_G1(m, t):
        return m.G1[t] <= m.Chpmax
    model.limit_G1 = pyo.Constraint(model.T, rule=limit_G1)

    # Furnace capacity
    def limit_G2(m, t):
        return m.G2[t] <= m.Fmax
    model.limit_G2 = pyo.Constraint(model.T, rule=limit_G2)

    # Chiller capacity
    def limit_H2(m, t):
        return m.H2[t] <= m.CBmax
    model.limit_H2 = pyo.Constraint(model.T, rule=limit_H2)

    return model
    
def EH1_stc_model(time_periods, SCENARIOS, PROB, p, DATA, study_day, scenario_days):

    # =====================================================================
    # MODEL
    # =====================================================================

    model = pyo.ConcreteModel(name="Stochastic_Energy_Hub")

    # =====================================================================
    # SETS
    # =====================================================================

    model.T = pyo.Set(initialize=time_periods)
    model.S = pyo.Set(initialize=SCENARIOS)

    # =====================================================================
    # PARAMETERS
    # =====================================================================

    # ---------- Scalar parameters ----------

    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"])
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])

    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])

    # ---------- Deterministic parameters (study day) ----------

    model.De = pyo.Param(
        model.T,
        initialize={t: DATA["DE"][(study_day, t)] for t in time_periods}
    )

    model.Dh = pyo.Param(
        model.T,
        initialize={t: DATA["DH"][(study_day, t)] for t in time_periods}
    )

    model.Dc = pyo.Param(
        model.T,
        initialize={t: DATA["DC"][(study_day, t)] for t in time_periods}
    )

    model.lam_DA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods}
    )

    model.lam_g = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods}
    )

    # ---------- Stochastic parameters (scenario days) ----------

    model.lam_IDA = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Precio_IDA"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )

    model.Wind = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Wind"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )

    # =====================================================================
    # VARIABLES
    # =====================================================================

    # ---------- First stage ----------

    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # ---------- Second stage ----------

    model.E_IDA = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    model.Wind_used = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    model.E = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    model.G1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    model.H1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    # =====================================================================
    # OBJECTIVE
    # =====================================================================

    def obj_rule(m):
        return (
            sum(
                m.lam_DA[t] * m.E_DA[t]
                + m.lam_g[t] * m.G[t]
                for t in m.T
            )
            +
            sum(
                PROB[s] * sum(
                    m.lam_IDA[t, s] * m.E_IDA[t, s]
                    + eps * (m.Wind[t, s] - m.Wind_used[t, s])
                    for t in m.T
                )
                for s in m.S
            )
        )

    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # =====================================================================
    # CONSTRAINTS
    # =====================================================================

    # Electricity balance

    def eq_b(m, t, s):
        return (
            m.Wind_used[t, s]
            + m.E_DA[t]
            + m.E_IDA[t, s]
            ==
            m.E[t, s]
        )

    model.eq_b = pyo.Constraint(model.T, model.S, rule=eq_b)

    # Wind utilization

    def eq_curt(m, t, s):
        return m.Wind_used[t, s] <= m.Wind[t, s]

    model.eq_curt = pyo.Constraint(model.T, model.S, rule=eq_curt)

    # Electricity demand

    def eq_c(m, t, s):
        return (
            m.eta_ee * m.E[t, s]
            + m.eta_ge * m.G1[t, s]
            ==
            m.De[t]
        )

    model.eq_c = pyo.Constraint(model.T, model.S, rule=eq_c)

    # Gas balance

    def eq_d(m, t, s):
        return (
            m.G[t]
            ==
            m.G1[t, s] + m.G2[t, s]
        )

    model.eq_d = pyo.Constraint(model.T, model.S, rule=eq_d)

    # Furnace balance

    def eq_e(m, t, s):
        return (
            m.eta_ghf * m.G2[t, s]
            ==
            m.H1[t, s] + m.H2[t, s]
        )

    model.eq_e = pyo.Constraint(model.T, model.S, rule=eq_e)

    # Heat demand

    def eq_f(m, t, s):
        return (
            m.eta_gh * m.G1[t, s]
            + m.H1[t, s]
            ==
            m.Dh[t]
        )

    model.eq_f = pyo.Constraint(model.T, model.S, rule=eq_f)

    # Cooling demand

    def eq_g(m, t, s):
        return (
            m.eta_hc * m.H2[t, s]
            ==
            m.Dc[t]
        )

    model.eq_g = pyo.Constraint(model.T, model.S, rule=eq_g)

    # CHP capacity

    def limit_G1(m, t, s):
        return m.G1[t, s] <= m.Chpmax

    model.limit_G1 = pyo.Constraint(model.T, model.S, rule=limit_G1)

    # Furnace capacity

    def limit_G2(m, t, s):
        return m.G2[t, s] <= m.Fmax

    model.limit_G2 = pyo.Constraint(model.T, model.S, rule=limit_G2)

    # Chiller capacity

    def limit_H2(m, t, s):
        return m.H2[t, s] <= m.CBmax

    model.limit_H2 = pyo.Constraint(model.T, model.S, rule=limit_H2)

    return model

###############################################################################
######  ENERGY HUB 2    ##############################
###############################################################################

def EH2_model(time_periods, p, DATA, study_day, scenario_day):
    
    # =====================================================================
    # MODEL
    # =====================================================================

    model = pyo.ConcreteModel(name="Energy_Hub2")

    # =====================================================================
    # SETS
    # =====================================================================

    model.T = pyo.Set(initialize=time_periods)

    # =====================================================================
    # PARAMETERS
    # =====================================================================
    # ---------- Conversion efficiencies ----------

    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"])
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])

    # ---------- Battery ----------

    model.eta_c = pyo.Param(initialize=p["eta_c"])
    model.eta_d = pyo.Param(initialize=p["eta_d"])

    model.E_min_c = pyo.Param(initialize=p["E_min_c"])
    model.E_max_c = pyo.Param(initialize=p["E_max_c"])

    model.E_min_d = pyo.Param(initialize=p["E_min_d"])
    model.E_max_d = pyo.Param(initialize=p["E_max_d"])

    model.SOC_min = pyo.Param(initialize=p["SOC_min"])
    model.SOC_max = pyo.Param(initialize=p["SOC_max"])
    model.SOC_ini = pyo.Param(initialize=p["SOC_ini"])

    # ---------- Capacities ----------

    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])

    # ---------- Time-dependent parameters ----------

    model.De = pyo.Param(model.T,
                         initialize={t: DATA["DE"][(study_day, t)] for t in time_periods})

    model.Dh = pyo.Param(model.T,
                         initialize={t: DATA["DH"][(study_day, t)] for t in time_periods})

    model.Dc = pyo.Param(model.T,
                         initialize={t: DATA["DC"][(study_day, t)] for t in time_periods})

    model.lam_DA = pyo.Param(model.T,
                             initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods})

    model.lam_g = pyo.Param(model.T,
                            initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods})

    model.lam_IDA = pyo.Param(model.T,
                              initialize={t: DATA["Precio_IDA"][(scenario_day, t)] for t in time_periods})

    model.Wind = pyo.Param(model.T,
                           initialize={t: DATA["Wind"][(scenario_day, t)] for t in time_periods})
    
    # =====================================================================
    # VARIABLES
    # =====================================================================

    # Electricity

    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_IDA = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.Wind_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.E = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # Battery

    model.E_c = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_d = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.SOC = pyo.Var(
        model.T,
        bounds=(model.SOC_min, model.SOC_max)
    )

    model.I_ch = pyo.Var(model.T, domain=pyo.Binary)
    model.I_dch = pyo.Var(model.T, domain=pyo.Binary)

    # Gas

    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # Heat

    model.H1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # =====================================================================
    # OBJECTIVE
    # =====================================================================

    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t]
            + m.lam_IDA[t] * m.E_IDA[t]
            + m.lam_g[t] * m.G[t]
            + eps * (m.Wind[t] - m.Wind_used[t])
            for t in m.T
        )

    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # =====================================================================
    # CONSTRAINTS
    # =====================================================================

    # Electric input balance
    def eq_b(m, t):
        return (
            m.Wind_used[t]
            + m.E_DA[t]
            + m.E_IDA[t]
            ==
            m.E_c[t]
            + m.E[t]
        )
    model.eq_b = pyo.Constraint(model.T, rule=eq_b)

    # Wind curtailment
    def eq_curt(m, t):
        return m.Wind_used[t] <= m.Wind[t]
    model.eq_curt = pyo.Constraint(model.T, rule=eq_curt)

    # Electricity demand balance
    def eq_c(m, t):
        return (
            m.eta_ee * m.E[t]
            + m.E_d[t]
            + m.eta_ge * m.G1[t]
            ==
            m.De[t]
        )
    model.eq_c = pyo.Constraint(model.T, rule=eq_c)

    # State of Charge (SOC) tracking
    def eq_d(m, t):
        if t == m.T.first():
            return (
                m.SOC[t]
                ==
                m.SOC_ini
                + m.eta_c * m.E_c[t]
                - m.E_d[t] / m.eta_d
            )
        else:
            return (
                m.SOC[t]
                ==
                m.SOC[t-1]
                + m.eta_c * m.E_c[t]
                - m.E_d[t] / m.eta_d
            )
    model.eq_d = pyo.Constraint(model.T, rule=eq_d)

    # Battery charging lower limit
    def eq_e_lower(m, t):
        return m.E_min_c * m.I_ch[t] <= m.E_c[t]
    model.eq_e_lower = pyo.Constraint(model.T, rule=eq_e_lower)

    # Battery charging upper limit
    def eq_e_upper(m, t):
        return m.E_c[t] <= m.E_max_c * m.I_ch[t]
    model.eq_e_upper = pyo.Constraint(model.T, rule=eq_e_upper)

    # Battery discharging lower limit
    def eq_f_lower(m, t):
        return m.E_min_d * m.I_dch[t] <= m.E_d[t]
    model.eq_f_lower = pyo.Constraint(model.T, rule=eq_f_lower)

    # Battery discharging upper limit
    def eq_f_upper(m, t):
        return m.E_d[t] <= m.E_max_d * m.I_dch[t]
    model.eq_f_upper = pyo.Constraint(model.T, rule=eq_f_upper)

    # Battery simultaneous charge/discharge exclusion
    def eq_g(m, t):
        return m.I_ch[t] + m.I_dch[t] <= 1
    model.eq_g = pyo.Constraint(model.T, rule=eq_g)

    # Gas input split
    def eq_h(m, t):
        return m.G[t] == m.G1[t] + m.G2[t]
    model.eq_h = pyo.Constraint(model.T, rule=eq_h)

    # Furnace heat generation split
    def eq_i(m, t):
        return (
            m.eta_ghf * m.G2[t]
            ==
            m.H1[t] + m.H2[t]
        )
    model.eq_i = pyo.Constraint(model.T, rule=eq_i)

    # Heat demand balance
    def eq_j(m, t):
        return (
            m.eta_gh * m.G1[t]
            + m.H1[t]
            ==
            m.Dh[t]
        )
    model.eq_j = pyo.Constraint(model.T, rule=eq_j)

    # Cooling demand balance
    def eq_k(m, t):
        return (
            m.eta_hc * m.H2[t]
            ==
            m.Dc[t]
        )
    model.eq_k = pyo.Constraint(model.T, rule=eq_k)

    # CHP gas capacity limit
    def limit_G1(m, t):
        return m.G1[t] <= m.Chpmax
    model.limit_G1 = pyo.Constraint(model.T, rule=limit_G1)

    # Furnace gas capacity limit
    def limit_G2(m, t):
        return m.G2[t] <= m.Fmax
    model.limit_G2 = pyo.Constraint(model.T, rule=limit_G2)

    # Chiller heat input capacity limit
    def limit_H2(m, t):
        return m.H2[t] <= m.CBmax
    model.limit_H2 = pyo.Constraint(model.T, rule=limit_H2)
    
    return model


def EH2_stc_model(time_periods, SCENARIOS, PROB, p, DATA, study_day, scenario_days):

    # =====================================================================
    # MODEL
    # =====================================================================

    model = pyo.ConcreteModel(name="Stochastic_Energy_Hub_with_ESS")

    # =====================================================================
    # SETS
    # =====================================================================

    model.T = pyo.Set(initialize=time_periods)
    model.S = pyo.Set(initialize=SCENARIOS)

    # =====================================================================
    # PARAMETERS
    # =====================================================================

    # ---------- Conversion efficiencies ----------

    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"])
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])

    # ---------- Battery ----------

    model.eta_c = pyo.Param(initialize=p["eta_c"])
    model.eta_d = pyo.Param(initialize=p["eta_d"])

    model.E_min_c = pyo.Param(initialize=p["E_min_c"])
    model.E_max_c = pyo.Param(initialize=p["E_max_c"])

    model.E_min_d = pyo.Param(initialize=p["E_min_d"])
    model.E_max_d = pyo.Param(initialize=p["E_max_d"])

    model.SOC_min = pyo.Param(initialize=p["SOC_min"])
    model.SOC_max = pyo.Param(initialize=p["SOC_max"])
    model.SOC_ini = pyo.Param(initialize=p["SOC_ini"])

    # ---------- Capacities ----------

    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])

    # ---------- Deterministic parameters (study day) ----------

    model.De = pyo.Param(
        model.T,
        initialize={t: DATA["DE"][(study_day, t)] for t in time_periods}
    )

    model.Dh = pyo.Param(
        model.T,
        initialize={t: DATA["DH"][(study_day, t)] for t in time_periods}
    )

    model.Dc = pyo.Param(
        model.T,
        initialize={t: DATA["DC"][(study_day, t)] for t in time_periods}
    )

    model.lam_DA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods}
    )

    model.lam_g = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods}
    )

    # ---------- Stochastic parameters (scenario days) ----------

    model.lam_IDA = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Precio_IDA"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )

    model.Wind = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Wind"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )
    
    # =====================================================================
    # VARIABLES
    # =====================================================================

    # ---------- First stage ----------

    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # ---------- Second stage ----------

    model.E_IDA = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    model.Wind_used = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    model.E_c = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_d = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    model.SOC = pyo.Var(
        model.T,
        model.S,
        bounds=(model.SOC_min, model.SOC_max)
    )

    model.I_ch = pyo.Var(model.T, model.S, domain=pyo.Binary)

    model.G1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    model.H1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    # =====================================================================
    # OBJECTIVE
    # =====================================================================
    
    def obj_rule(m):
        return (
            sum(
                m.lam_DA[t] * m.E_DA[t]
                + m.lam_g[t] * m.G[t]
                for t in m.T
            )
            +
            sum(
                PROB[s] * sum(
                    m.lam_IDA[t, s] * m.E_IDA[t, s]
                    + eps * (m.Wind[t, s] - m.Wind_used[t, s])
                    for t in m.T
                )
                for s in m.S
            )
        )
    
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # =====================================================================
    # CONSTRAINTS
    # =====================================================================
    
    # Electric input balance
    def eq_b(m, t, s):
        return (
            m.Wind_used[t, s]
            + m.E_DA[t]
            + m.E_IDA[t, s]
            ==
            m.E_c[t, s]
            + m.E[t, s]
        )
    model.eq_b = pyo.Constraint(model.T, model.S, rule=eq_b)
    
    # Wind curtailment
    def eq_curt(m, t, s):
        return m.Wind_used[t, s] <= m.Wind[t, s]
    model.eq_curt = pyo.Constraint(model.T, model.S, rule=eq_curt)
    
    # Electricity demand balance
    def eq_c(m, t, s):
        return (
            m.eta_ee * m.E[t, s]
            + m.E_d[t, s]
            + m.eta_ge * m.G1[t, s]
            ==
            m.De[t]
        )
    model.eq_c = pyo.Constraint(model.T, model.S, rule=eq_c)
    
    # State of Charge (SOC) tracking
    def eq_d(m, t, s):
        if t == m.T.first():
            return (
                m.SOC[t, s]
                ==
                m.SOC_ini
                + m.eta_c * m.E_c[t, s]
                - m.E_d[t, s] / m.eta_d
            )
        else:
            return (
                m.SOC[t, s]
                ==
                m.SOC[t-1, s]
                + m.eta_c * m.E_c[t, s]
                - m.E_d[t, s] / m.eta_d
            )
    model.eq_d = pyo.Constraint(model.T, model.S, rule=eq_d)
    
    # Battery charging lower limit
    def eq_e_lower(m, t, s):
        return m.E_min_c * m.I_ch[t, s] <= m.E_c[t, s]
    model.eq_e_lower = pyo.Constraint(model.T, model.S, rule=eq_e_lower)
    
    # Battery charging upper limit
    def eq_e_upper(m, t, s):
        return m.E_c[t, s] <= m.E_max_c * m.I_ch[t, s]
    model.eq_e_upper = pyo.Constraint(model.T, model.S, rule=eq_e_upper)
    
    # Battery discharging lower limit
    def eq_f_lower(m, t, s):
        return m.E_min_d * (1 - m.I_ch[t, s]) <= m.E_d[t, s]
    model.eq_f_lower = pyo.Constraint(model.T, model.S, rule=eq_f_lower)
    
    # Battery discharging upper limit
    def eq_f_upper(m, t, s):
        return m.E_d[t, s] <= m.E_max_d * (1 - m.I_ch[t, s])
    model.eq_f_upper = pyo.Constraint(model.T, model.S, rule=eq_f_upper)
    
    # Gas input split
    def eq_h(m, t, s):
        return (
            m.G[t]
            ==
            m.G1[t, s] + m.G2[t, s]
        )
    model.eq_h = pyo.Constraint(model.T, model.S, rule=eq_h)
    
    # Furnace heat generation split
    def eq_i(m, t, s):
        return (
            m.eta_ghf * m.G2[t, s]
            ==
            m.H1[t, s] + m.H2[t, s]
        )
    model.eq_i = pyo.Constraint(model.T, model.S, rule=eq_i)
    
    # Heat demand balance
    def eq_j(m, t, s):
        return (
            m.eta_gh * m.G1[t, s]
            + m.H1[t, s]
            ==
            m.Dh[t]
        )
    model.eq_j = pyo.Constraint(model.T, model.S, rule=eq_j)
    
    # Cooling demand balance
    def eq_k(m, t, s):
        return (
            m.eta_hc * m.H2[t, s]
            ==
            m.Dc[t]
        )
    model.eq_k = pyo.Constraint(model.T, model.S, rule=eq_k)
    
    # CHP gas capacity limit
    def limit_G1(m, t, s):
        return m.G1[t, s] <= m.Chpmax
    model.limit_G1 = pyo.Constraint(model.T, model.S, rule=limit_G1)
    
    # Furnace gas capacity limit
    def limit_G2(m, t, s):
        return m.G2[t, s] <= m.Fmax
    model.limit_G2 = pyo.Constraint(model.T, model.S, rule=limit_G2)
    
    # Chiller heat input capacity limit
    def limit_H2(m, t, s):
        return m.H2[t, s] <= m.CBmax
    model.limit_H2 = pyo.Constraint(model.T, model.S, rule=limit_H2)
    
    return model
    
###############################################################################
######  ENERGY HUB 3    ##############################
###############################################################################

def EH3_model(time_periods, p, DATA, study_day, scenario_day):
    
    # =====================================================================
    # MODEL
    # =====================================================================
    
    model = pyo.ConcreteModel(name="Full_Energy_Hub_with_ESS_EHP")
    
    # =====================================================================
    # SETS
    # =====================================================================
    
    model.T = pyo.Set(initialize=time_periods)
    
    # =====================================================================
    # PARAMETERS
    # =====================================================================
    
    # ---------- Scalar parameters ----------
    
    # Conversion efficiencies
    model.eta_c = pyo.Param(initialize=p["eta_c"])
    model.eta_d = pyo.Param(initialize=p["eta_d"])
    
    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"])
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])
    
    # ESS parameters
    model.E_min_c = pyo.Param(initialize=p["E_min_c"])
    model.E_max_c = pyo.Param(initialize=p["E_max_c"])
    model.E_min_d = pyo.Param(initialize=p["E_min_d"])
    model.E_max_d = pyo.Param(initialize=p["E_max_d"])
    
    model.SOC_min = pyo.Param(initialize=p["SOC_min"])
    model.SOC_max = pyo.Param(initialize=p["SOC_max"])
    model.SOC_ini = pyo.Param(initialize=p["SOC_ini"])
    
    # Device capacities
    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])
    
    # Electric Heat Pump
    model.COP = pyo.Param(initialize=p["COP"])
    
    model.C_EHP_min = pyo.Param(initialize=p["C_EHP_min"])
    model.C_EHP_max = pyo.Param(initialize=p["C_EHP_max"])
    
    model.H_EHP_min = pyo.Param(initialize=p["H_EHP_min"])
    model.H_EHP_max = pyo.Param(initialize=p["H_EHP_max"])
    
    # ---------- Time-dependent parameters ----------
    
    model.De = pyo.Param(
        model.T,
        initialize={t: DATA["DE"][(study_day, t)] for t in time_periods}
    )
    
    model.Dh = pyo.Param(
        model.T,
        initialize={t: DATA["DH"][(study_day, t)] for t in time_periods}
    )
    
    model.Dc = pyo.Param(
        model.T,
        initialize={t: DATA["DC"][(study_day, t)] for t in time_periods}
    )
    
    model.lam_DA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods}
    )
    
    model.lam_g = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods}
    )
    
    model.lam_IDA = pyo.Param( 
        model.T,
        initialize={t: DATA["Precio_IDA"][(scenario_day, t)] for t in time_periods}
    )
    
    model.Wind = pyo.Param(
        model.T,
        initialize={t: DATA["Wind"][(scenario_day, t)] for t in time_periods}
    )
    
    # =====================================================================
    # VARIABLES
    # =====================================================================
    
    # ---------- Electricity ----------
    
    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_IDA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.Wind_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.E = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- ESS ----------
    
    model.E_c = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_d = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.SOC = pyo.Var(
        model.T,
        domain=pyo.Reals,
        bounds=(model.SOC_min, model.SOC_max)
    )
    
    model.I_ch = pyo.Var(model.T, domain=pyo.Binary)
    model.I_dch = pyo.Var(model.T, domain=pyo.Binary)
    
    # ---------- Gas ----------
    
    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.G1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- Furnace ----------
    
    model.H1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- Electric Heat Pump ----------
    
    model.E_3 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.H_EHP = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.C_EHP = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.I_h = pyo.Var(model.T, domain=pyo.Binary)
    model.I_c = pyo.Var(model.T, domain=pyo.Binary)
    
    # =====================================================================
    # OBJECTIVE
    # =====================================================================
    
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t]
            + m.lam_IDA[t] * m.E_IDA[t]
            + m.lam_g[t] * m.G[t]
            + eps * (m.Wind[t] - m.Wind_used[t])
            for t in m.T
        )
    
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # =====================================================================
    # CONSTRAINTS
    # =====================================================================
    
    # Electric input balance
    def eq_b(m, t):
        return (
            m.Wind_used[t]
            + m.E_DA[t]
            + m.E_IDA[t]
            ==
            m.E_c[t]
            + m.E[t]
        )
    model.eq_b = pyo.Constraint(model.T, rule=eq_b)
    
    # Wind utilization
    def eq_curt(m, t):
        return m.Wind_used[t] <= m.Wind[t]
    model.eq_curt = pyo.Constraint(model.T, rule=eq_curt)
    
    # Electricity demand balance
    def eq_c(m, t):
        return (
            m.eta_ee * m.E[t]
            + m.E_d[t]
            + m.eta_ge * m.G1[t]
            ==
            m.E_3[t] + m.De[t]
        )
    model.eq_c = pyo.Constraint(model.T, rule=eq_c)
    
    # State of Charge (SOC) tracking
    def eq_d(m, t):
        if t == m.T.first():
            return (
                m.SOC[t]
                ==
                m.SOC_ini
                + m.eta_c * m.E_c[t]
                - m.E_d[t] / m.eta_d
            )
        else:
            return (
                m.SOC[t]
                ==
                m.SOC[t-1]
                + m.eta_c * m.E_c[t]
                - m.E_d[t] / m.eta_d
            )
    model.eq_d = pyo.Constraint(model.T, rule=eq_d)
    
    # Battery charging lower limit
    def eq_e_lower(m, t):
        return m.E_min_c * m.I_ch[t] <= m.E_c[t]
    model.eq_e_lower = pyo.Constraint(model.T, rule=eq_e_lower)
    
    # Battery charging upper limit
    def eq_e_upper(m, t):
        return m.E_c[t] <= m.E_max_c * m.I_ch[t]
    model.eq_e_upper = pyo.Constraint(model.T, rule=eq_e_upper)
    
    # Battery discharging lower limit
    def eq_f_lower(m, t):
        return m.E_min_d * m.I_dch[t] <= m.E_d[t]
    model.eq_f_lower = pyo.Constraint(model.T, rule=eq_f_lower)
    
    # Battery discharging upper limit
    def eq_f_upper(m, t):
        return m.E_d[t] <= m.E_max_d * m.I_dch[t]
    model.eq_f_upper = pyo.Constraint(model.T, rule=eq_f_upper)
    
    # Battery simultaneous charge/discharge exclusion
    def eq_g(m, t):
        return m.I_ch[t] + m.I_dch[t] <= 1
    model.eq_g = pyo.Constraint(model.T, rule=eq_g)
    
    # Gas input split
    def eq_h(m, t):
        return (
            m.G[t]
            ==
            m.G1[t] + m.G2[t]
        )
    model.eq_h = pyo.Constraint(model.T, rule=eq_h)
    
    # Furnace heat generation split
    def eq_i(m, t):
        return (
            m.eta_ghf * m.G2[t]
            ==
            m.H1[t] + m.H2[t]
        )
    model.eq_i = pyo.Constraint(model.T, rule=eq_i)
    
    # Heat demand balance
    def eq_j(m, t):
        return (
            m.eta_gh * m.G1[t]
            + m.H1[t]
            + m.H_EHP[t]
            ==
            m.Dh[t]
        )
    model.eq_j = pyo.Constraint(model.T, rule=eq_j)
    
    # Cooling demand balance
    def eq_k(m, t):
        return (
            m.eta_hc * m.H2[t]
            + m.C_EHP[t]
            ==
            m.Dc[t]
        )
    model.eq_k = pyo.Constraint(model.T, rule=eq_k)
    
    # Electric Heat Pump balance
    def eq_l(m, t):
        return (
            m.COP * m.E_3[t]
            ==
            m.H_EHP[t] + m.C_EHP[t]
        )
    model.eq_l = pyo.Constraint(model.T, rule=eq_l)
    
    # Heat pump heating lower limit
    def eq_m_lower(m, t):
        return m.H_EHP_min * m.I_h[t] <= m.H_EHP[t]
    model.eq_m_lower = pyo.Constraint(model.T, rule=eq_m_lower)
    
    # Heat pump heating upper limit
    def eq_m_upper(m, t):
        return m.H_EHP[t] <= m.H_EHP_max * m.I_h[t]
    model.eq_m_upper = pyo.Constraint(model.T, rule=eq_m_upper)
    
    # Heat pump cooling lower limit
    def eq_n_lower(m, t):
        return m.C_EHP_min * m.I_c[t] <= m.C_EHP[t]
    model.eq_n_lower = pyo.Constraint(model.T, rule=eq_n_lower)
    
    # Heat pump cooling upper limit
    def eq_n_upper(m, t):
        return m.C_EHP[t] <= m.C_EHP_max * m.I_c[t]
    model.eq_n_upper = pyo.Constraint(model.T, rule=eq_n_upper)
    
    # Heat pump cannot heat and cool simultaneously
    def eq_o(m, t):
        return m.I_h[t] + m.I_c[t] <= 1
    model.eq_o = pyo.Constraint(model.T, rule=eq_o)
    
    # CHP gas capacity limit
    def limit_G1(m, t):
        return m.G1[t] <= m.Chpmax
    model.limit_G1 = pyo.Constraint(model.T, rule=limit_G1)
    
    # Furnace gas capacity limit
    def limit_G2(m, t):
        return m.G2[t] <= m.Fmax
    model.limit_G2 = pyo.Constraint(model.T, rule=limit_G2)
    
    # Chiller heat input capacity limit
    def limit_H2(m, t):
        return m.H2[t] <= m.CBmax
    model.limit_H2 = pyo.Constraint(model.T, rule=limit_H2)
    
    return model


def EH3_stc_model(time_periods, SCENARIOS, PROB, p, DATA, study_day, scenario_days):

    # =====================================================================
    # MODEL
    # =====================================================================

    model = pyo.ConcreteModel(name="Stochastic_Energy_Hub_EHP")

    # =====================================================================
    # SETS
    # =====================================================================

    model.T = pyo.Set(initialize=time_periods)
    model.S = pyo.Set(initialize=SCENARIOS)

    # =====================================================================
    # PARAMETERS
    # =====================================================================

    # ---------- Scalar parameters ----------

    # Transformer / CHP / Furnace / Chiller efficiencies
    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"])
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])

    # ESS
    model.eta_c = pyo.Param(initialize=p["eta_c"])
    model.eta_d = pyo.Param(initialize=p["eta_d"])

    model.E_min_c = pyo.Param(initialize=p["E_min_c"])
    model.E_max_c = pyo.Param(initialize=p["E_max_c"])
    model.E_min_d = pyo.Param(initialize=p["E_min_d"])
    model.E_max_d = pyo.Param(initialize=p["E_max_d"])

    model.SOC_min = pyo.Param(initialize=p["SOC_min"])
    model.SOC_max = pyo.Param(initialize=p["SOC_max"])
    model.SOC_ini = pyo.Param(initialize=p["SOC_ini"])

    # Device capacities
    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])

    # Electric Heat Pump (EHP)
    model.COP = pyo.Param(initialize=p["COP"])

    model.H_EHP_min = pyo.Param(initialize=p["H_EHP_min"])
    model.H_EHP_max = pyo.Param(initialize=p["H_EHP_max"])

    model.C_EHP_min = pyo.Param(initialize=p["C_EHP_min"])
    model.C_EHP_max = pyo.Param(initialize=p["C_EHP_max"])

    # ---------- Deterministic parameters (study day) ----------

    model.De = pyo.Param(
        model.T,
        initialize={t: DATA["DE"][(study_day, t)] for t in time_periods}
    )

    model.Dh = pyo.Param(
        model.T,
        initialize={t: DATA["DH"][(study_day, t)] for t in time_periods}
    )

    model.Dc = pyo.Param(
        model.T,
        initialize={t: DATA["DC"][(study_day, t)] for t in time_periods}
    )

    model.lam_DA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods}
    )

    model.lam_g = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods}
    )

    # ---------- Stochastic parameters (scenario days) ----------

    model.lam_IDA = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Precio_IDA"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )

    model.Wind = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Wind"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )
    
    # =====================================================================
    # VARIABLES
    # =====================================================================
    
    # ---------- First stage ----------
    
    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- Second stage ----------
    
    # Electricity
    
    model.E_IDA = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.Wind_used = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.E_2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    # ESS
    
    model.E_c = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_d = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.SOC = pyo.Var(
        model.T,
        model.S,
        domain=pyo.Reals,
        bounds=(model.SOC_min, model.SOC_max)
    )
    
    model.I_ch = pyo.Var(model.T, model.S, domain=pyo.Binary)
    
    # Gas
    
    model.G1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    # Furnace & Chiller
    
    model.H1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    # Electric Heat Pump (EHP)
    
    model.E_3 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.H_EHP = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.C_EHP = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.I_h = pyo.Var(model.T, model.S, domain=pyo.Binary)
        
    # =====================================================================
    # OBJECTIVE
    # =====================================================================
    
    def obj_rule(m):
        return (
            sum(
                m.lam_DA[t] * m.E_DA[t]
                + m.lam_g[t] * m.G[t]
                for t in m.T
            )
            +
            sum(
                PROB[s] * sum(
                    m.lam_IDA[t, s] * m.E_IDA[t, s]
                    + eps * (m.Wind[t, s] - m.Wind_used[t, s])
                    for t in m.T
                )
                for s in m.S
            )
        )
    
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # =====================================================================
    # CONSTRAINTS
    # =====================================================================
    
    # Electricity balance
    
    def eq_b(m, t, s):
        return (
            m.Wind_used[t, s]
            + m.E_DA[t]
            + m.E_IDA[t, s]
            ==
            m.E_c[t, s]
            + m.E_2[t, s]
        )
    
    model.eq_b = pyo.Constraint(model.T, model.S, rule=eq_b)
    
    # Wind utilization
    
    def eq_curt(m, t, s):
        return m.Wind_used[t, s] <= m.Wind[t, s]
    
    model.eq_curt = pyo.Constraint(model.T, model.S, rule=eq_curt)
    
    # Electricity demand
    
    def eq_c(m, t, s):
        return (
            m.eta_ee * m.E_2[t, s]
            + m.E_d[t, s]
            + m.eta_ge * m.G1[t, s]
            ==
            m.E_3[t, s]
            + m.De[t]
        )
    
    model.eq_c = pyo.Constraint(model.T, model.S, rule=eq_c)
    
    # State of Charge (SOC)
    
    def eq_d(m, t, s):
        if t == m.T.first():
            return (
                m.SOC[t, s]
                ==
                m.SOC_ini
                + m.eta_c * m.E_c[t, s]
                - m.E_d[t, s] / m.eta_d
            )
        else:
            return (
                m.SOC[t, s]
                ==
                m.SOC[t-1, s]
                + m.eta_c * m.E_c[t, s]
                - m.E_d[t, s] / m.eta_d
            )
    
    model.eq_d = pyo.Constraint(model.T, model.S, rule=eq_d)
    
    # Battery charging lower limit
    def eq_f_lower(m, t, s):
        return m.E_min_c * m.I_ch[t, s] <= m.E_c[t, s]
    
    model.eq_f_lower = pyo.Constraint(model.T, model.S, rule=eq_f_lower)
    
    # Battery charging upper limit
    def eq_f_upper(m, t, s):
        return m.E_c[t, s] <= m.E_max_c * m.I_ch[t, s]
    
    model.eq_f_upper = pyo.Constraint(model.T, model.S, rule=eq_f_upper)
    
    # Battery discharging lower limit
    def eq_g_lower(m, t, s):
        return m.E_min_d * ( 1 - m.I_ch[t, s] ) <= m.E_d[t, s]
    
    model.eq_g_lower = pyo.Constraint(model.T, model.S, rule=eq_g_lower)
    
    # Battery discharging upper limit
    
    def eq_g_upper(m, t, s):
        return m.E_d[t, s] <= m.E_max_d * ( 1 - m.I_ch[t, s] )
    
    model.eq_g_upper = pyo.Constraint(model.T, model.S, rule=eq_g_upper)
    

    # Gas balance
    
    def eq_j(m, t, s):
        return (
            m.G[t]
            ==
            m.G1[t, s] + m.G2[t, s]
        )
    
    model.eq_j = pyo.Constraint(model.T, model.S, rule=eq_j)
    
    # Heat demand
    
    def eq_k(m, t, s):
        return (
            m.eta_gh * m.G1[t, s]
            + m.H1[t, s]
            + m.H_EHP[t, s]
            ==
            m.Dh[t]
        )
    
    model.eq_k = pyo.Constraint(model.T, model.S, rule=eq_k)
    
    # Furnace balance
    
    def eq_l(m, t, s):
        return (
            m.eta_ghf * m.G2[t, s]
            ==
            m.H1[t, s] + m.H2[t, s]
        )
    
    model.eq_l = pyo.Constraint(model.T, model.S, rule=eq_l)
    
    # Cooling demand
    
    def eq_m(m, t, s):
        return (
            m.eta_hc * m.H2[t, s]
            + m.C_EHP[t, s]
            ==
            m.Dc[t]
        )
    
    model.eq_m = pyo.Constraint(model.T, model.S, rule=eq_m)
    
    # Electric Heat Pump balance
    
    def eq_n(m, t, s):
        return (
            m.COP * m.E_3[t, s]
            ==
            m.H_EHP[t, s] + m.C_EHP[t, s]
        )
    
    model.eq_n = pyo.Constraint(model.T, model.S, rule=eq_n)
    
    # EHP heating lower limit
    
    def eq_o_lower(m, t, s):
        return m.H_EHP_min * m.I_h[t, s] <= m.H_EHP[t, s]
    
    model.eq_o_lower = pyo.Constraint(model.T, model.S, rule=eq_o_lower)
    
    # EHP heating upper limit
    
    def eq_o_upper(m, t, s):
        return m.H_EHP[t, s] <= m.H_EHP_max * m.I_h[t, s]
    
    model.eq_o_upper = pyo.Constraint(model.T, model.S, rule=eq_o_upper)
    
    # EHP cooling lower limit
    
    def eq_p_lower(m, t, s):
        return m.C_EHP_min * (1-m.I_h[t, s]) <= m.C_EHP[t, s]
    
    model.eq_p_lower = pyo.Constraint(model.T, model.S, rule=eq_p_lower)
    
    # EHP cooling upper limit
    
    def eq_p_upper(m, t, s):
        return m.C_EHP[t, s] <= m.C_EHP_max * (1-m.I_h[t, s])
    
    model.eq_p_upper = pyo.Constraint(model.T, model.S, rule=eq_p_upper)
     
    # CHP capacity
    
    def limit_G1(m, t, s):
        return m.G1[t, s] <= m.Chpmax
    
    model.limit_G1 = pyo.Constraint(model.T, model.S, rule=limit_G1)
    
    # Furnace capacity
    
    def limit_G2(m, t, s):
        return m.G2[t, s] <= m.Fmax
    
    model.limit_G2 = pyo.Constraint(model.T, model.S, rule=limit_G2)
    
    # Chiller capacity
    
    def limit_H2(m, t, s):
        return m.H2[t, s] <= m.CBmax
    
    model.limit_H2 = pyo.Constraint(model.T, model.S, rule=limit_H2)
    
    return model

###############################################################################
######  ENERGY HUB 4    ##############################
###############################################################################

def EH4_model(time_periods, p, DATA, study_day, scenario_day):
    
    # =====================================================================
    # MODEL
    # =====================================================================
    
    model = pyo.ConcreteModel(name="Energy_Hub_with_ESS_EHP_H2")
    
    # =====================================================================
    # SETS
    # =====================================================================
    
    model.T = pyo.Set(initialize=time_periods)
    
    # =====================================================================
    # PARAMETERS
    # =====================================================================
    
    # ---------- Scalar parameters ----------
    
    # Conversion efficiencies
    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"])
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])
    
    # ESS parameters
    model.eta_c = pyo.Param(initialize=p["eta_c"])
    model.eta_d = pyo.Param(initialize=p["eta_d"])
    
    model.E_min_c = pyo.Param(initialize=p["E_min_c"])
    model.E_max_c = pyo.Param(initialize=p["E_max_c"])
    model.E_min_d = pyo.Param(initialize=p["E_min_d"])
    model.E_max_d = pyo.Param(initialize=p["E_max_d"])
    
    model.SOC_min = pyo.Param(initialize=p["SOC_min"])
    model.SOC_max = pyo.Param(initialize=p["SOC_max"])
    model.SOC_ini = pyo.Param(initialize=p["SOC_ini"])
    
    # H2 Tank & Conversion parameters
    model.eta_EL = pyo.Param(initialize=p["eta_EL"])
    model.eta_F = pyo.Param(initialize=p["eta_F"])
    
    model.el_max = pyo.Param(initialize=p["el_max"])
    model.fc_max = pyo.Param(initialize=p["fc_max"])
    
    model.SOC_H2_min = pyo.Param(initialize=p["SOC_H2_min"])
    model.SOC_H2_max = pyo.Param(initialize=p["SOC_H2_max"])
    model.SOC_H2_ini = pyo.Param(initialize=p["SOC_H2_ini"])
    
    # Device capacities
    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])
    
    # Electric Heat Pump
    model.COP = pyo.Param(initialize=p["COP"])
    
    model.C_EHP_min = pyo.Param(initialize=p["C_EHP_min"])
    model.C_EHP_max = pyo.Param(initialize=p["C_EHP_max"])
    
    model.H_EHP_min = pyo.Param(initialize=p["H_EHP_min"])
    model.H_EHP_max = pyo.Param(initialize=p["H_EHP_max"])
    
    # ---------- Time-dependent parameters ----------
    
    model.De = pyo.Param(
        model.T,
        initialize={t: DATA["DE"][(study_day, t)] for t in time_periods}
    )
    
    model.Dh = pyo.Param(
        model.T,
        initialize={t: DATA["DH"][(study_day, t)] for t in time_periods}
    )
    
    model.Dc = pyo.Param(
        model.T,
        initialize={t: DATA["DC"][(study_day, t)] for t in time_periods}
    )
    
    model.lam_DA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods}
    )
    
    model.lam_g = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods}
    )
    
    model.lam_IDA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_IDA"][(scenario_day, t)] for t in time_periods}
    )
    
    model.Wind = pyo.Param(
        model.T,
        initialize={t: DATA["Wind"][(scenario_day, t)] for t in time_periods}
    )
    
    # =====================================================================
    # VARIABLES
    # =====================================================================
    
    # ---------- Electricity ----------
    
    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_IDA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.Wind_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.E = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- ESS ----------
    
    model.E_c = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_d = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.SOC = pyo.Var(
        model.T,
        domain=pyo.Reals,
        bounds=(model.SOC_min, model.SOC_max)
    )
    
    model.I_ch = pyo.Var(model.T, domain=pyo.Binary)
    model.I_dch = pyo.Var(model.T, domain=pyo.Binary)
    
    # ---------- H2 Storage System ----------
    
    model.E_H2y_plus = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_H2y_minus = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.SOC_H2 = pyo.Var(
        model.T,
        domain=pyo.Reals,
        bounds=(model.SOC_H2_min, model.SOC_H2_max)
    )
    
    model.I_H2y_plus = pyo.Var(model.T, domain=pyo.Binary)
    model.I_H2y_minus = pyo.Var(model.T, domain=pyo.Binary)
    
    # ---------- Gas ----------
    
    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.G1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- Furnace ----------
    
    model.H1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- Electric Heat Pump ----------
    
    model.E_3 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.H_EHP = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.C_EHP = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.I_h = pyo.Var(model.T, domain=pyo.Binary)
    model.I_c = pyo.Var(model.T, domain=pyo.Binary)
    
    # =====================================================================
    # OBJECTIVE
    # =====================================================================
    
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t]
            + m.lam_IDA[t] * m.E_IDA[t]
            + m.lam_g[t] * m.G[t]
            + eps * (m.Wind[t] - m.Wind_used[t])
            for t in m.T
        )
    
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # =====================================================================
    # CONSTRAINTS
    # =====================================================================
    
    # Electric input balance
    def eq_b(m, t):
        return (
            m.Wind_used[t]
            + m.E_DA[t]
            + m.E_IDA[t]
            ==
            m.E_c[t]
            + m.E[t]
            + m.E_H2y_plus[t]
        )
    model.eq_b = pyo.Constraint(model.T, rule=eq_b)
    
    # Wind utilization
    def eq_curt(m, t):
        return m.Wind_used[t] <= m.Wind[t]
    model.eq_curt = pyo.Constraint(model.T, rule=eq_curt)
    
    # Electricity demand balance
    def eq_c(m, t):
        return (
            m.eta_ee * m.E[t]
            + m.E_d[t]
            + m.eta_ge * m.G1[t]
            + m.E_H2y_minus[t]
            ==
            m.E_3[t] + m.De[t]
            )
    model.eq_c = pyo.Constraint(model.T, rule=eq_c)
    
    # State of Charge (SOC) tracking - Battery
    def eq_d(m, t):
        if t == m.T.first():
            return (
                m.SOC[t]
                ==
                m.SOC_ini
                + m.eta_c * m.E_c[t]
                - m.E_d[t] / m.eta_d
            )
        else:
            return (
                m.SOC[t]
                ==
                m.SOC[t-1]
                + m.eta_c * m.E_c[t]
                - m.E_d[t] / m.eta_d
            )
    model.eq_d = pyo.Constraint(model.T, rule=eq_d)
    
    # Battery charging lower limit
    def eq_e_lower(m, t):
        return m.E_min_c * m.I_ch[t] <= m.E_c[t]
    model.eq_e_lower = pyo.Constraint(model.T, rule=eq_e_lower)
    
    # Battery charging upper limit
    def eq_e_upper(m, t):
        return m.E_c[t] <= m.E_max_c * m.I_ch[t]
    model.eq_e_upper = pyo.Constraint(model.T, rule=eq_e_upper)
    
    # Battery discharging lower limit
    def eq_f_lower(m, t):
        return m.E_min_d * m.I_dch[t] <= m.E_d[t]
    model.eq_f_lower = pyo.Constraint(model.T, rule=eq_f_lower)
    
    # Battery discharging upper limit
    def eq_f_upper(m, t):
        return m.E_d[t] <= m.E_max_d * m.I_dch[t]
    model.eq_f_upper = pyo.Constraint(model.T, rule=eq_f_upper)
    
    # Battery simultaneous charge/discharge exclusion
    def eq_g(m, t):
        return m.I_ch[t] + m.I_dch[t] <= 1
    model.eq_g = pyo.Constraint(model.T, rule=eq_g)
    
    # ---------- NEW CONSTRAINTS: H2 TANK & CONVERSION ----------
    
    # H2 Tank Balance Equation (eq:H2_tank_a)
    def eq_H2_tank_a(m, t):
        if t == m.T.first():
            return (m.SOC_H2[t] 
                ==  m.SOC_H2_ini
                + m.E_H2y_plus[t] * m.eta_EL
                - m.E_H2y_minus[t] / m.eta_F
            )
        else:
            return (
                m.SOC_H2[t]
                == m.SOC_H2[t-1]
                + m.E_H2y_plus[t] * m.eta_EL
                - m.E_H2y_minus[t] / m.eta_F
            )
    model.eq_H2_tank_a = pyo.Constraint(model.T, rule=eq_H2_tank_a)
    
    # Electrolyzer Capacity Limit (eq:H2_tank_b_upper)
    def eq_H2_tank_b(m, t):
        return m.E_H2y_plus[t] <= m.el_max * m.I_H2y_plus[t]
    model.eq_H2_tank_b = pyo.Constraint(model.T, rule=eq_H2_tank_b)
    
    # Fuel Cell Capacity Limit and Exclusion (eq:H2_tank_c_upper)
    def eq_H2_tank_c(m, t):
        return m.E_H2y_minus[t] <= m.fc_max * m.I_H2y_minus[t]
    model.eq_H2_tank_c = pyo.Constraint(model.T, rule=eq_H2_tank_c)
    
    # P2H or H2P, just one at a time (or none of them)
    def eq_H2_tank_d(m,t):
        return m.I_H2y_plus[t] + m.I_H2y_minus[t] <= 1
    model.eq_H2_tank_d = pyo.Constraint(model.T, rule=eq_H2_tank_d)
    
    # ------------------------------------------------------------
    
    # Gas input split
    def eq_h(m, t):
        return (
            m.G[t]
            ==
            m.G1[t] + m.G2[t]
        )
    model.eq_h = pyo.Constraint(model.T, rule=eq_h)
    
    # Furnace heat generation split
    def eq_i(m, t):
        return (
            m.eta_ghf * m.G2[t]
            ==
            m.H1[t] + m.H2[t]
        )
    model.eq_i = pyo.Constraint(model.T, rule=eq_i)
    
    # Heat demand balance
    def eq_j(m, t):
        return (
            m.eta_gh * m.G1[t]
            + m.H1[t]
            + m.H_EHP[t]
            ==
            m.Dh[t]
        )
    model.eq_j = pyo.Constraint(model.T, rule=eq_j)
    
    # Cooling demand balance
    def eq_k(m, t):
        return (
            m.eta_hc * m.H2[t]
            + m.C_EHP[t]
            ==
            m.Dc[t]
        )
    model.eq_k = pyo.Constraint(model.T, rule=eq_k)
    
    # Electric Heat Pump balance
    def eq_l(m, t):
        return (
            m.COP * m.E_3[t]
            ==
            m.H_EHP[t] + m.C_EHP[t]
        )
    model.eq_l = pyo.Constraint(model.T, rule=eq_l)
    
    # Heat pump heating lower limit
    def eq_m_lower(m, t):
        return m.H_EHP_min * m.I_h[t] <= m.H_EHP[t]
    model.eq_m_lower = pyo.Constraint(model.T, rule=eq_m_lower)
    
    # Heat pump heating upper limit
    def eq_m_upper(m, t):
        return m.H_EHP[t] <= m.H_EHP_max * m.I_h[t]
    model.eq_m_upper = pyo.Constraint(model.T, rule=eq_m_upper)
    
    # Heat pump cooling lower limit
    def eq_n_lower(m, t):
        return m.C_EHP_min * m.I_c[t] <= m.C_EHP[t]
    model.eq_n_lower = pyo.Constraint(model.T, rule=eq_n_lower)
    
    # Heat pump cooling upper limit
    def eq_n_upper(m, t):
        return m.C_EHP[t] <= m.C_EHP_max * m.I_c[t]
    model.eq_n_upper = pyo.Constraint(model.T, rule=eq_n_upper)
    
    # Heat pump cannot heat and cool simultaneously
    def eq_o(m, t):
        return m.I_h[t] + m.I_c[t] <= 1
    model.eq_o = pyo.Constraint(model.T, rule=eq_o)
    
    # CHP gas capacity limit
    def limit_G1(m, t):
        return m.G1[t] <= m.Chpmax
    model.limit_G1 = pyo.Constraint(model.T, rule=limit_G1)
    
    # Furnace gas capacity limit
    def limit_G2(m, t):
        return m.G2[t] <= m.Fmax
    model.limit_G2 = pyo.Constraint(model.T, rule=limit_G2)
    
    # Chiller heat input capacity limit
    def limit_H2(m, t):
        return m.H2[t] <= m.CBmax
    model.limit_H2 = pyo.Constraint(model.T, rule=limit_H2)
    
    return model

def EH4_stc_model(time_periods, SCENARIOS, PROB, p, DATA, study_day, scenario_days):

    # =====================================================================
    # MODEL
    # =====================================================================

    model = pyo.ConcreteModel(name="Stochastic_Energy_Hub_ESS_EHP_H2")

    # =====================================================================
    # SETS
    # =====================================================================

    model.T = pyo.Set(initialize=time_periods)
    model.S = pyo.Set(initialize=SCENARIOS)

    # =====================================================================
    # PARAMETERS
    # =====================================================================

    # ---------- Scalar parameters ----------

    # Transformer / CHP / Furnace / Chiller efficiencies
    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"])
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])

    # ESS
    model.eta_c = pyo.Param(initialize=p["eta_c"])
    model.eta_d = pyo.Param(initialize=p["eta_d"])

    model.E_min_c = pyo.Param(initialize=p["E_min_c"])
    model.E_max_c = pyo.Param(initialize=p["E_max_c"])
    model.E_min_d = pyo.Param(initialize=p["E_min_d"])
    model.E_max_d = pyo.Param(initialize=p["E_max_d"])

    model.SOC_min = pyo.Param(initialize=p["SOC_min"])
    model.SOC_max = pyo.Param(initialize=p["SOC_max"])
    model.SOC_ini = pyo.Param(initialize=p["SOC_ini"])

    # Hydrogen Storage System (H2)
    model.eta_EL = pyo.Param(initialize=p["eta_EL"])
    model.eta_F = pyo.Param(initialize=p["eta_F"])
    
    model.el_max = pyo.Param(initialize=p["el_max"])
    model.fc_max = pyo.Param(initialize=p["fc_max"])
    
    model.SOC_H2_min = pyo.Param(initialize=p["SOC_H2_min"])
    model.SOC_H2_max = pyo.Param(initialize=p["SOC_H2_max"])
    model.SOC_H2_ini = pyo.Param(initialize=p["SOC_H2_ini"])

    # Device capacities
    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])

    # Electric Heat Pump (EHP)
    model.COP = pyo.Param(initialize=p["COP"])

    model.H_EHP_min = pyo.Param(initialize=p["H_EHP_min"])
    model.H_EHP_max = pyo.Param(initialize=p["H_EHP_max"])

    model.C_EHP_min = pyo.Param(initialize=p["C_EHP_min"])
    model.C_EHP_max = pyo.Param(initialize=p["C_EHP_max"])

    # ---------- Deterministic parameters (study day) ----------

    model.De = pyo.Param(
        model.T,
        initialize={t: DATA["DE"][(study_day, t)] for t in time_periods}
    )

    model.Dh = pyo.Param(
        model.T,
        initialize={t: DATA["DH"][(study_day, t)] for t in time_periods}
    )

    model.Dc = pyo.Param(
        model.T,
        initialize={t: DATA["DC"][(study_day, t)] for t in time_periods}
    )

    model.lam_DA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods}
    )

    model.lam_g = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods}
    )

    # ---------- Stochastic parameters (scenario days) ----------

    model.lam_IDA = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Precio_IDA"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )

    model.Wind = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Wind"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )
    
    # =====================================================================
    # VARIABLES
    # =====================================================================
    
    # ---------- First stage ----------
    
    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- Second stage ----------
    
    # Electricity
    
    model.E_IDA = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.Wind_used = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.E_2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    # ESS (Battery)
    
    model.E_c = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_d = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.SOC = pyo.Var(
        model.T,
        model.S,
        domain=pyo.Reals,
        bounds=(model.SOC_min, model.SOC_max)
    )
    
    model.I_ch = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.I_dch = pyo.Var(model.T, model.S, domain=pyo.Binary)

    # H2 Storage System
    model.E_H2y_plus = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_H2y_minus = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.SOC_H2 = pyo.Var(
        model.T,
        model.S,
        domain=pyo.Reals,
        bounds=(model.SOC_H2_min, model.SOC_H2_max)
    )
    
    model.I_H2y_plus = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.I_H2y_minus = pyo.Var(model.T, model.S, domain=pyo.Binary)
    
    # Gas
    
    model.G1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    # Furnace & Chiller
    
    model.H1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    # Electric Heat Pump (EHP)
    
    model.E_3 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.H_EHP = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.C_EHP = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.I_h = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.I_c = pyo.Var(model.T, model.S, domain=pyo.Binary)
        
    # =====================================================================
    # OBJECTIVE
    # =====================================================================
    
    def obj_rule(m):
        return (
            sum(
                m.lam_DA[t] * m.E_DA[t]
                + m.lam_g[t] * m.G[t]
                for t in m.T
            )
            +
            sum(
                PROB[s] * sum(
                    m.lam_IDA[t, s] * m.E_IDA[t, s]
                    + eps * (m.Wind[t, s] - m.Wind_used[t, s])
                    for t in m.T
                )
                for s in m.S
            )
        )
    
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # =====================================================================
    # CONSTRAINTS
    # =====================================================================
    
    # Electricity balance
    
    def eq_b(m, t, s):
        return (
            m.Wind_used[t, s]
            + m.E_DA[t]
            + m.E_IDA[t, s]
            ==
            m.E_c[t, s]
            + m.E_2[t, s]
        )
    
    model.eq_b = pyo.Constraint(model.T, model.S, rule=eq_b)
    
    # Wind utilization
    
    def eq_curt(m, t, s):
        return m.Wind_used[t, s] <= m.Wind[t, s]
    
    model.eq_curt = pyo.Constraint(model.T, model.S, rule=eq_curt)
    
    # Electricity demand
    
    def eq_c(m, t, s):
        return (
            m.eta_ee * m.E_2[t, s]
            + m.E_d[t, s]
            + m.eta_ge * m.G1[t, s]
            + m.E_H2y_minus[t, s]
            ==
            m.E_3[t, s]
            + m.De[t]
            + m.E_H2y_plus[t, s]
        )
    
    model.eq_c = pyo.Constraint(model.T, model.S, rule=eq_c)
    
    # State of Charge (SOC) - Battery
    
    def eq_d(m, t, s):
        if t == m.T.first():
            return (
                m.SOC[t, s]
                ==
                m.SOC_ini
                + m.eta_c * m.E_c[t, s]
                - m.E_d[t, s] / m.eta_d
            )
        else:
            return (
                m.SOC[t, s]
                ==
                m.SOC[t-1, s]
                + m.eta_c * m.E_c[t, s]
                - m.E_d[t, s] / m.eta_d
            )
    
    model.eq_d = pyo.Constraint(model.T, model.S, rule=eq_d)

    # ---------- Hydrogen Storage Constraints ----------

    # State of Charge (SOC) - H2 Tank
    def eq_H2_tank_a(m, t, s):
        if t == m.T.first():
            return (
                m.SOC_H2[t, s]
                ==
                m.SOC_H2_ini
                + m.E_H2y_plus[t, s] * m.eta_EL
                - m.E_H2y_minus[t, s] / m.eta_F
            )
        else:
            return (
                m.SOC_H2[t, s]
                ==
                m.SOC_H2[t-1, s]
                + m.E_H2y_plus[t, s] * m.eta_EL
                - m.E_H2y_minus[t, s] / m.eta_F
            )

    model.eq_H2_tank_a = pyo.Constraint(model.T, model.S, rule=eq_H2_tank_a)

    # Electrolyzer upper limit
    def eq_H2_tank_b_upper(m, t, s):
        return m.E_H2y_plus[t, s] <= m.el_max * m.I_H2y_plus[t, s]
    model.eq_H2_tank_b_upper = pyo.Constraint(model.T, model.S, rule=eq_H2_tank_b_upper)

    # Fuel Cell upper limit
    def eq_H2_tank_c_upper(m, t, s):
        return m.E_H2y_minus[t, s] <= m.fc_max * m.I_H2y_minus[t, s]
    model.eq_H2_tank_c_upper = pyo.Constraint(model.T, model.S, rule=eq_H2_tank_c_upper)
    
    # P2H or H2P, just 1 at a time (or none of them)
    def eq_H2_tank_d(m, t, s):
        return m.I_H2y_plus[t, s] + m.I_H2y_minus[t, s] <= 1
    model.eq_H2_tank_d = pyo.Constraint(model.T, model.S, rule=eq_H2_tank_d)

    # Battery charging lower limit
    def eq_f_lower(m, t, s):
        return m.E_min_c * m.I_ch[t, s] <= m.E_c[t, s]
    
    model.eq_f_lower = pyo.Constraint(model.T, model.S, rule=eq_f_lower)
    
    # Battery charging upper limit
    def eq_f_upper(m, t, s):
        return m.E_c[t, s] <= m.E_max_c * m.I_ch[t, s]
    model.eq_f_upper = pyo.Constraint(model.T, model.S, rule=eq_f_upper)
    
    # Battery discharging lower limit
    def eq_g_lower(m, t, s):
        return m.E_min_d * m.I_dch[t, s] <= m.E_d[t, s]
    model.eq_g_lower = pyo.Constraint(model.T, model.S, rule=eq_g_lower)
    
    # Battery discharging upper limit
    def eq_g_upper(m, t, s):
        return m.E_d[t, s] <= m.E_max_d * m.I_dch[t, s]
    model.eq_g_upper = pyo.Constraint(model.T, model.S, rule=eq_g_upper)
    
    def eq_h(m, t, s):
        return m.I_ch[t, s] + m.I_dch[t, s] <= 1
    model.eq_h = pyo.Constraint(model.T, model.S, rule=eq_h)
    
    # Gas balance
    def eq_j(m, t, s):
        return (
            m.G[t]
            ==
            m.G1[t, s] + m.G2[t, s]
        )
    model.eq_j = pyo.Constraint(model.T, model.S, rule=eq_j)
    
    # Heat demand
    def eq_k(m, t, s):
        return (
            m.eta_gh * m.G1[t, s]
            + m.H1[t, s]
            + m.H_EHP[t, s]
            ==
            m.Dh[t]
        )
    
    model.eq_k = pyo.Constraint(model.T, model.S, rule=eq_k)
    
    # Furnace balance
    
    def eq_l(m, t, s):
        return (
            m.eta_ghf * m.G2[t, s]
            ==
            m.H1[t, s] + m.H2[t, s]
        )
    
    model.eq_l = pyo.Constraint(model.T, model.S, rule=eq_l)
    
    # Cooling demand
    
    def eq_m(m, t, s):
        return (
            m.eta_hc * m.H2[t, s]
            + m.C_EHP[t, s]
            ==
            m.Dc[t]
        )
    
    model.eq_m = pyo.Constraint(model.T, model.S, rule=eq_m)
    
    # Electric Heat Pump balance
    
    def eq_n(m, t, s):
        return (
            m.COP * m.E_3[t, s]
            ==
            m.H_EHP[t, s] + m.C_EHP[t, s]
        )
    
    model.eq_n = pyo.Constraint(model.T, model.S, rule=eq_n)
    
    # EHP heating lower limit
    
    def eq_o_lower(m, t, s):
        return m.H_EHP_min * m.I_h[t, s] <= m.H_EHP[t, s]
    
    model.eq_o_lower = pyo.Constraint(model.T, model.S, rule=eq_o_lower)
    
    # EHP heating upper limit
    
    def eq_o_upper(m, t, s):
        return m.H_EHP[t, s] <= m.H_EHP_max * m.I_h[t, s]
    
    model.eq_o_upper = pyo.Constraint(model.T, model.S, rule=eq_o_upper)
    
    # EHP cooling lower limit
    
    def eq_p_lower(m, t, s):
        return m.C_EHP_min * m.I_c[t, s] <= m.C_EHP[t, s]
    model.eq_p_lower = pyo.Constraint(model.T, model.S, rule=eq_p_lower)
    
    # EHP cooling upper limit
    
    def eq_p_upper(m, t, s):
        return m.C_EHP[t, s] <= m.C_EHP_max * m.I_c[t,s]
    model.eq_p_upper = pyo.Constraint(model.T, model.S, rule=eq_p_upper)
    
    # EHP cooling or heating just one at a time (or none of them)
    def eq_q(m, t, s):
        return m.I_h[t,s] + m.I_c[t,s] <= 1
    model.eq_q = pyo.Constraint(model.T, model.S, rule=eq_q)
    
    # CHP capacity
    
    def limit_G1(m, t, s):
        return m.G1[t, s] <= m.Chpmax
    
    model.limit_G1 = pyo.Constraint(model.T, model.S, rule=limit_G1)
    
    # Furnace capacity
    
    def limit_G2(m, t, s):
        return m.G2[t, s] <= m.Fmax
    
    model.limit_G2 = pyo.Constraint(model.T, model.S, rule=limit_G2)
    
    # Chiller capacity
    
    def limit_H2(m, t, s):
        return m.H2[t, s] <= m.CBmax
    
    model.limit_H2 = pyo.Constraint(model.T, model.S, rule=limit_H2)
        
    return model

###############################################################################
######  ENERGY HUB 5    ##############################
###############################################################################

def EH5_model(time_periods, p, DATA, study_day, scenario_day):
    
    # =====================================================================
    # MODEL
    # =====================================================================
    
    model = pyo.ConcreteModel(name="Energy_Hub_EH5_with_H2")
    
    # =====================================================================
    # SETS
    # =====================================================================
    
    model.T = pyo.Set(initialize=time_periods)
    
    # =====================================================================
    # PARAMETERS
    # =====================================================================
    
    # ---------- Conversion efficiencies ----------
    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])   # eta_gh^CHP
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"]) # eta_gh^F
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])
    
    # ---------- ESS parameters ----------
    model.eta_c = pyo.Param(initialize=p["eta_c"])
    model.eta_d = pyo.Param(initialize=p["eta_d"])
    model.E_min_c = pyo.Param(initialize=p["E_min_c"])
    model.E_max_c = pyo.Param(initialize=p["E_max_c"])
    model.E_min_d = pyo.Param(initialize=p["E_min_d"])
    model.E_max_d = pyo.Param(initialize=p["E_max_d"])
    model.SOC_min = pyo.Param(initialize=p["SOC_min"])
    model.SOC_max = pyo.Param(initialize=p["SOC_max"])
    model.SOC_ini = pyo.Param(initialize=p["SOC_ini"])
    
    # ---------- H2 & SOEC Electrolyzer parameters ----------
    model.eta_EL = pyo.Param(initialize=p["eta_EL"])
    model.el_min = pyo.Param(initialize=p["el_min"])
    model.el_max = pyo.Param(initialize=p["el_max"])
    model.p_stb = pyo.Param(initialize=p["p_stb"])
    model.ramp_up_el = pyo.Param(initialize=p["ramp_up_el"])
    model.ramp_down_el = pyo.Param(initialize=p["ramp_down_el"])
    model.h_warmup = pyo.Param(initialize=p["h_warmup"])
    # Initial condition: Let's assume that the electrolyzer starts in Standby
    model.I_stb_ini = pyo.Param(initialize=1)
    model.I_el_ini = pyo.Param(initialize=0)
    model.I_off_ini = pyo.Param(initialize=0)
    
    # ---------- Auxiliary Systems parameters ----------
    model.alpha_DES = pyo.Param(initialize=p["alpha_DES"])
    model.eta_com = pyo.Param(initialize=p["eta_com"])
    model.rho_com = pyo.Param(initialize=p["rho_com"])
    
    # ---------- Fuel Cell parameters ----------
    model.eta_F = pyo.Param(initialize=p["eta_F"])
    model.fc_min = pyo.Param(initialize=p["fc_min"])
    model.fc_max = pyo.Param(initialize=p["fc_max"])
    model.ramp_up_F = pyo.Param(initialize=p["ramp_up_F"])
    model.ramp_down_F = pyo.Param(initialize=p["ramp_down_F"])
    
    # ---------- H2 Storage Tank parameters ----------
    model.eta_H2 = pyo.Param(initialize=p["eta_H2"])
    model.SOC_H2_min = pyo.Param(initialize=p["SOC_H2_min"])
    model.SOC_H2_max = pyo.Param(initialize=p["SOC_H2_max"])
    model.SOC_H2_ini = pyo.Param(initialize=p["SOC_H2_ini"])
    
    # ---------- Device capacities ----------
    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])
    
    # ---------- Electric Heat Pump ----------
    model.COP = pyo.Param(initialize=p["COP"])
    model.C_EHP_min = pyo.Param(initialize=p["C_EHP_min"])
    model.C_EHP_max = pyo.Param(initialize=p["C_EHP_max"])
    model.H_EHP_min = pyo.Param(initialize=p["H_EHP_min"])
    model.H_EHP_max = pyo.Param(initialize=p["H_EHP_max"])
    
    # ---------- Time-dependent parameters ----------
    model.De = pyo.Param(model.T, initialize={t: DATA["DE"][(study_day, t)] for t in time_periods})
    model.Dh = pyo.Param(model.T, initialize={t: DATA["DH"][(study_day, t)] for t in time_periods})
    model.Dc = pyo.Param(model.T, initialize={t: DATA["DC"][(study_day, t)] for t in time_periods})
    
    model.lam_DA = pyo.Param(model.T, initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods})
    model.lam_g = pyo.Param(model.T, initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods})
    model.lam_IDA = pyo.Param(model.T, initialize={t: DATA["Precio_IDA"][(scenario_day, t)] for t in time_periods})
    model.Wind = pyo.Param(model.T, initialize={t: DATA["Wind"][(scenario_day, t)] for t in time_periods})
    
    # =====================================================================
    # VARIABLES
    # =====================================================================
    
    # ---------- Electricity Base Variables ----------
    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_IDA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.Wind_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.Curt = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_2 = pyo.Var(model.T, domain=pyo.NonNegativeReals) # Renamed from E
    
    # ---------- ESS (Battery) ----------
    model.E_c = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_d = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.T, domain=pyo.Reals, bounds=(model.SOC_min, model.SOC_max))
    model.I_ch = pyo.Var(model.T, domain=pyo.Binary)
    model.I_dch = pyo.Var(model.T, domain=pyo.Binary)
    
    # ---------- H2 Storage & Conversion Systems ----------
    model.E_el = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_F = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_DES = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.E_com = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.HY_in = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.HY_out = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.SOC_H2 = pyo.Var(model.T, domain=pyo.Reals, bounds=(model.SOC_H2_min, model.SOC_H2_max))
    
    # Tri-state Logic & Fuel Cell Binaries
    model.I_off = pyo.Var(model.T, domain=pyo.Binary)
    model.I_stb = pyo.Var(model.T, domain=pyo.Binary)
    model.I_el = pyo.Var(model.T, domain=pyo.Binary)
    model.I_f = pyo.Var(model.T, domain=pyo.Binary)
    
    # ---------- Gas & Heat / Cooling Coupling ----------
    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.H1 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.E_3 = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.H_EHP = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.C_EHP = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.I_h = pyo.Var(model.T, domain=pyo.Binary)
    model.I_c = pyo.Var(model.T, domain=pyo.Binary)
    
    # =====================================================================
    # OBJECTIVE
    # =====================================================================
    
    def obj_rule(m):
        # Mantiene la misma estructura solicitada
        return sum(m.lam_DA[t] * m.E_DA[t] + m.lam_IDA[t] * m.E_IDA[t] + m.lam_g[t] * m.G[t]
            + eps * (m.Wind[t] - m.Wind_used[t]) 
            for t in m.T
        )
    
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # =====================================================================
    # CONSTRAINTS
    # =====================================================================
    # ---------------------------------------------------------------------
    # 1. BASE EH5 BALANCES & COUPLED ELECTRICAL INPUT BUS
    # ---------------------------------------------------------------------
    # Electrical Input Balance (eq:EH5_a)
    def eq_EH5_a(m, t):
        return m.Wind_used[t] + m.E_DA[t] + m.E_IDA[t] == m.E_DES[t] + m.E_com[t] + m.E_el[t] + m.E_c[t] + m.E_2[t]
    model.eq_EH5_a = pyo.Constraint(model.T, rule=eq_EH5_a)
    
    # Wind Utilization Limit (eq:EH5_b)
    def eq_EH5_b(m, t):
        return m.Wind_used[t] <= m.Wind[t]
    model.eq_EH5_b = pyo.Constraint(model.T, rule=eq_EH5_b)
    
    # Wind Curtailment (eq:EH5_c)
    def eq_EH5_c(m, t):
        return m.Curt[t] == m.Wind[t] - m.Wind_used[t]
    model.eq_EH5_c = pyo.Constraint(model.T, rule=eq_EH5_c)
    
    # Electricity Demand Balance (eq:EH5_d)
    def eq_EH5_d(m, t):
        return m.E_F[t] + m.eta_ee * m.E_2[t] + m.E_d[t] + m.eta_ge * m.G1[t] == m.De[t] + m.E_3[t]
    model.eq_EH5_d = pyo.Constraint(model.T, rule=eq_EH5_d)

    # ---------------------------------------------------------------------
    # 2. ELECTRIC BATTERY STORAGE SYSTEM (ESS)
    # ---------------------------------------------------------------------
    # State of Charge (SOC) tracking - Battery (eq:EH5_e)
    def eq_EH5_e(m, t):
        if t == m.T.first():
            return m.SOC[t] == m.SOC_ini + m.eta_c * m.E_c[t] - m.E_d[t] / m.eta_d
        else:
            return m.SOC[t] == m.SOC[t-1] + m.eta_c * m.E_c[t] - m.E_d[t] / m.eta_d
    model.eq_EH5_e = pyo.Constraint(model.T, rule=eq_EH5_e)
    
    # Battery charging limits (eq:EH5_f)
    def eq_EH5_f_lower(m, t):
        return m.E_min_c * m.I_ch[t] <= m.E_c[t]
    model.eq_EH5_f_lower = pyo.Constraint(model.T, rule=eq_EH5_f_lower)
    
    def eq_EH5_f_upper(m, t):
        return m.E_c[t] <= m.E_max_c * m.I_ch[t]
    model.eq_EH5_f_upper = pyo.Constraint(model.T, rule=eq_EH5_f_upper)
    
    # Battery discharging limits (eq:EH5_g)
    def eq_EH5_g_lower(m, t):
        return m.E_min_d * m.I_dch[t] <= m.E_d[t]
    model.eq_EH5_g_lower = pyo.Constraint(model.T, rule=eq_EH5_g_lower)
    
    def eq_EH5_g_upper(m, t):
        return m.E_d[t] <= m.E_max_d * m.I_dch[t]
    model.eq_EH5_g_upper = pyo.Constraint(model.T, rule=eq_EH5_g_upper)
    
    # Note: eq:EH5_h bounds handled directly in Variable declaration for SOC
    
    # Battery simultaneous charge/discharge exclusion (eq:EH5_i)
    def eq_EH5_i(m, t):
        return m.I_dch[t] + m.I_ch[t] <= 1
    model.eq_EH5_i = pyo.Constraint(model.T, rule=eq_EH5_i)

    # ---------------------------------------------------------------------
    # 3. GAS, HEATING, AND COOLING COUPLING (EH5)
    # ---------------------------------------------------------------------
    # Gas input split (eq:EH5_j)
    def eq_EH5_j(m, t):
        return m.G[t] == m.G1[t] + m.G2[t]
    model.eq_EH5_j = pyo.Constraint(model.T, rule=eq_EH5_j)
    
    # Heat demand balance (eq:EH5_k)
    def eq_EH5_k(m, t):
        return m.eta_gh * m.G1[t] + m.H1[t] + m.H_EHP[t] == m.Dh[t]
    model.eq_EH5_k = pyo.Constraint(model.T, rule=eq_EH5_k)
    
    # Furnace heat generation split (eq:EH5_l)
    def eq_EH5_l(m, t):
        return m.eta_ghf * m.G2[t] == m.H1[t] + m.H2[t]
    model.eq_EH5_l = pyo.Constraint(model.T, rule=eq_EH5_l)
    
    # Cooling demand balance (eq:EH5_m)
    def eq_EH5_m(m, t):
        return m.eta_hc * m.H2[t] + m.C_EHP[t] == m.Dc[t]
    model.eq_EH5_m = pyo.Constraint(model.T, rule=eq_EH5_m)
    
    # Electric Heat Pump balance (eq:EH5_n)
    def eq_EH5_n(m, t):
        return m.C_EHP[t] + m.H_EHP[t] == m.E_3[t] * m.COP
    model.eq_EH5_n = pyo.Constraint(model.T, rule=eq_EH5_n)
    
    # Heat pump heating limits (eq:EH5_o)
    def eq_EH5_o_lower(m, t):
        return m.H_EHP_min * m.I_h[t] <= m.H_EHP[t]
    model.eq_EH5_o_lower = pyo.Constraint(model.T, rule=eq_EH5_o_lower)
    
    def eq_EH5_o_upper(m, t):
        return m.H_EHP[t] <= m.H_EHP_max * m.I_h[t]
    model.eq_EH5_o_upper = pyo.Constraint(model.T, rule=eq_EH5_o_upper)
    
    # Heat pump cooling limits (eq:EH5_p)
    def eq_EH5_p_lower(m, t):
        return m.C_EHP_min * m.I_c[t] <= m.C_EHP[t]
    model.eq_EH5_p_lower = pyo.Constraint(model.T, rule=eq_EH5_p_lower)
    
    def eq_EH5_p_upper(m, t):
        return m.C_EHP[t] <= m.C_EHP_max * m.I_c[t]
    model.eq_EH5_p_upper = pyo.Constraint(model.T, rule=eq_EH5_p_upper)
    
    # Heat pump mutually exclusive state (eq:EH5_q)
    def eq_EH5_q(m, t):
        return m.I_h[t] + m.I_c[t] <= 1
    model.eq_EH5_q = pyo.Constraint(model.T, rule=eq_EH5_q)

    # Note: Additional implicit capacity limits from older setup
    def limit_G1(m, t): return m.G1[t] <= m.Chpmax
    model.limit_G1 = pyo.Constraint(model.T, rule=limit_G1)
    
    def limit_G2(m, t): return m.G2[t] <= m.Fmax
    model.limit_G2 = pyo.Constraint(model.T, rule=limit_G2)
    
    def limit_H2(m, t): return m.H2[t] <= m.CBmax
    model.limit_H2 = pyo.Constraint(model.T, rule=limit_H2)

    # ---------------------------------------------------------------------
    # 4. HIGH-TEMPERATURE SOEC ELECTROLYZER (TRI-STATE LOGIC)
    # --------------------------------------------------------------------- 
    # Mutually Exclusive Operating States (eq:SOEC_states)
    def eq_SOEC_states(m, t):
        return m.I_off[t] + m.I_stb[t] + m.I_el[t] == 1
    model.eq_SOEC_states = pyo.Constraint(model.T, rule=eq_SOEC_states)
    
    # Startup State Transition Logic (eq:SOEC_startup_logic)
    def eq_SOEC_startup_logic(m, t):
        if t == m.T.first(): return pyo.Constraint.Skip
        return m.I_el[t] <= m.I_stb[t-1] + m.I_el[t-1]
    model.eq_SOEC_startup_logic = pyo.Constraint(model.T, rule=eq_SOEC_startup_logic)
    
    # Shutdown State Transition Logic (eq:SOEC_shutdown_logic)
    def eq_SOEC_shutdown_logic(m, t):
        if t == m.T.first(): return pyo.Constraint.Skip
        return m.I_off[t] <= m.I_off[t-1] + m.I_el[t-1]
    model.eq_SOEC_shutdown_logic = pyo.Constraint(model.T, rule=eq_SOEC_shutdown_logic)
    
    # Warm-up limit (eq:eq_SOEC_no_direct_start)
    def eq_SOEC_no_direct_start(m, t):
        h = int(pyo.value(m.h_warmup))
        # \forall t \in T \setminus \{1,...,h\}
        if t <= h:
            return pyo.Constraint.Skip
        # Summation over the previous h hours
        off_sum = sum(m.I_off[t - tau] for tau in range(1, h + 1))
        return m.I_el[t] <= (1 / h) * off_sum + m.I_stb[t-1] + m.I_el[t-1]
    model.eq_SOEC_no_direct_start = pyo.Constraint(model.T, rule=eq_SOEC_no_direct_start)
    
    # Power Consumption Bounds & Standby Power (eq:SOEC_power)
    def eq_SOEC_power_lower(m, t):
        return m.el_min * m.I_el[t] + m.p_stb * m.I_stb[t] <= m.E_el[t]
    model.eq_SOEC_power_lower = pyo.Constraint(model.T, rule=eq_SOEC_power_lower)
    
    def eq_SOEC_power_upper(m, t):
        return m.E_el[t] <= m.el_max * m.I_el[t] + m.p_stb * m.I_stb[t]
    model.eq_SOEC_power_upper = pyo.Constraint(model.T, rule=eq_SOEC_power_upper)
    
    # Hydrogen Production based on active power (eq:SOEC_production)
    def eq_SOEC_production(m, t):
        return m.HY_in[t] == m.eta_EL * (m.E_el[t] - m.p_stb * m.I_stb[t])
    model.eq_SOEC_production = pyo.Constraint(model.T, rule=eq_SOEC_production)
    
    # Ramping Limits (eq:SOEC_ramp)
    def eq_SOEC_ramp_lower(m, t):
        if t == m.T.first(): return pyo.Constraint.Skip
        return -m.ramp_down_el <= m.E_el[t] - m.E_el[t-1]
    model.eq_SOEC_ramp_lower = pyo.Constraint(model.T, rule=eq_SOEC_ramp_lower)
    
    def eq_SOEC_ramp_upper(m, t):
        if t == m.T.first(): return pyo.Constraint.Skip
        return m.E_el[t] - m.E_el[t-1] <= m.ramp_up_el
    model.eq_SOEC_ramp_upper = pyo.Constraint(model.T, rule=eq_SOEC_ramp_upper)

    # ---------------------------------------------------------------------
    # 5. AUXILIARY SYSTEMS: WATER DESALINIZATION & COMPRESSOR
    # ---------------------------------------------------------------------
    # Power for water desalination (eq:DES_power)
    def eq_DES_power(m, t):
        return m.E_DES[t] == m.alpha_DES * (m.E_el[t] - m.p_stb * m.I_stb[t])
    model.eq_DES_power = pyo.Constraint(model.T, rule=eq_DES_power)
    
    # Power for Hydrogen compression (eq:COM_power)
    def eq_COM_power(m, t):
        return m.E_com[t] == m.eta_com * m.rho_com * m.HY_in[t]
    model.eq_COM_power = pyo.Constraint(model.T, rule=eq_COM_power)

    # ---------------------------------------------------------------------
    # 6. FUEL CELL RE-ELECTRIFICATION SYSTEM
    # ---------------------------------------------------------------------
    # Hydrogen to Electricity Conversion (eq:FC_conversion)
    def eq_FC_conversion(m, t):
        return m.E_F[t] == m.eta_F * m.HY_out[t]
    model.eq_FC_conversion = pyo.Constraint(model.T, rule=eq_FC_conversion)
    
    # Fuel Cell Capacity bounds (eq:FC_bounds)
    def eq_FC_bounds_lower(m, t):
        return m.fc_min * m.I_f[t] <= m.E_F[t]
    model.eq_FC_bounds_lower = pyo.Constraint(model.T, rule=eq_FC_bounds_lower)
    
    def eq_FC_bounds_upper(m, t):
        return m.E_F[t] <= m.fc_max * m.I_f[t]
    model.eq_FC_bounds_upper = pyo.Constraint(model.T, rule=eq_FC_bounds_upper)
    
    # Fuel Cell Ramping limits (eq:FC_ramps)
    def eq_FC_ramp_lower(m, t):
        if t == m.T.first(): return pyo.Constraint.Skip
        return -m.ramp_down_F <= m.E_F[t] - m.E_F[t-1]
    model.eq_FC_ramp_lower = pyo.Constraint(model.T, rule=eq_FC_ramp_lower)
    
    def eq_FC_ramp_upper(m, t):
        if t == m.T.first(): return pyo.Constraint.Skip
        return m.E_F[t] - m.E_F[t-1] <= m.ramp_up_F
    model.eq_FC_ramp_upper = pyo.Constraint(model.T, rule=eq_FC_ramp_upper)

    # ---------------------------------------------------------------------
    # 7. HYDROGEN STORAGE TANK (HSS)
    # ---------------------------------------------------------------------
    # HSS State of Charge Tracking (eq:HSS_SOC)
    def eq_HSS_SOC(m, t):
        if t == m.T.first():
            return m.SOC_H2[t] == m.SOC_H2_ini + m.eta_H2 * m.HY_in[t] - (m.HY_out[t] / m.eta_H2)
        else:
            return m.SOC_H2[t] == m.SOC_H2[t-1] + m.eta_H2 * m.HY_in[t] - (m.HY_out[t] / m.eta_H2)
    model.eq_HSS_SOC = pyo.Constraint(model.T, rule=eq_HSS_SOC)
    
    # Note: eq:HSS_bounds handles min/max SOC for the tank in the Variable declarations
    
    # Limits on Tank injection and withdrawal flows (eq:HSS_flows)
    def eq_HSS_flows_in(m, t):
        # Uses soc_max^H2 (SOC_H2_max) as the Big-M parameter for logical constraint
        return m.HY_in[t] <= m.SOC_H2_max * m.I_el[t]
    model.eq_HSS_flows_in = pyo.Constraint(model.T, rule=eq_HSS_flows_in)
    
    def eq_HSS_flows_out(m, t):
        return m.HY_out[t] <= m.SOC_H2_max * m.I_f[t]
    model.eq_HSS_flows_out = pyo.Constraint(model.T, rule=eq_HSS_flows_out)
    
    # Decoupling logical: SOEC and Fuel Cell cannot run simultaneously (eq:H2_decoupling)
    def eq_H2_decoupling(m, t):
        return m.I_el[t] + m.I_f[t] <= 1
    model.eq_H2_decoupling = pyo.Constraint(model.T, rule=eq_H2_decoupling)

    return model

def EH5_stc_model(time_periods, SCENARIOS, PROB, p, DATA, study_day, scenario_days, eps=1e-3):
    
    # =====================================================================
    # MODEL
    # =====================================================================
    
    model = pyo.ConcreteModel(name="Stochastic_Energy_Hub_EH5")
    
    # =====================================================================
    # SETS
    # =====================================================================
    
    model.T = pyo.Set(initialize=time_periods)
    model.S = pyo.Set(initialize=SCENARIOS)
    
    # =====================================================================
    # PARAMETERS
    # =====================================================================
    
    # ---------- Conversion efficiencies ----------
    model.eta_ee = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh = pyo.Param(initialize=p["eta_gh"])   
    model.eta_ghf = pyo.Param(initialize=p["eta_ghf"]) 
    model.eta_hc = pyo.Param(initialize=p["eta_hc"])
    
    # ---------- ESS parameters ----------
    model.eta_c = pyo.Param(initialize=p["eta_c"])
    model.eta_d = pyo.Param(initialize=p["eta_d"])
    model.E_min_c = pyo.Param(initialize=p["E_min_c"])
    model.E_max_c = pyo.Param(initialize=p["E_max_c"])
    model.E_min_d = pyo.Param(initialize=p["E_min_d"])
    model.E_max_d = pyo.Param(initialize=p["E_max_d"])
    model.SOC_min = pyo.Param(initialize=p["SOC_min"])
    model.SOC_max = pyo.Param(initialize=p["SOC_max"])
    model.SOC_ini = pyo.Param(initialize=p["SOC_ini"])
    
    # ---------- H2 & SOEC Electrolyzer parameters ----------
    model.eta_EL = pyo.Param(initialize=p["eta_EL"])
    model.el_min = pyo.Param(initialize=p["el_min"])
    model.el_max = pyo.Param(initialize=p["el_max"])
    model.p_stb = pyo.Param(initialize=p["p_stb"])
    model.ramp_up_el = pyo.Param(initialize=p["ramp_up_el"])
    model.ramp_down_el = pyo.Param(initialize=p["ramp_down_el"])
    model.h_warmup = pyo.Param(initialize=p["h_warmup"])
    
    # ---------- Auxiliary Systems parameters ----------
    model.alpha_DES = pyo.Param(initialize=p["alpha_DES"])
    model.eta_com = pyo.Param(initialize=p["eta_com"])
    model.rho_com = pyo.Param(initialize=p["rho_com"])
    
    # ---------- Fuel Cell parameters ----------
    model.eta_F = pyo.Param(initialize=p["eta_F"])
    model.fc_min = pyo.Param(initialize=p["fc_min"])
    model.fc_max = pyo.Param(initialize=p["fc_max"])
    model.ramp_up_F = pyo.Param(initialize=p["ramp_up_F"])
    model.ramp_down_F = pyo.Param(initialize=p["ramp_down_F"])
    
    # ---------- H2 Storage Tank parameters ----------
    model.eta_H2 = pyo.Param(initialize=p["eta_H2"])
    model.SOC_H2_min = pyo.Param(initialize=p["SOC_H2_min"])
    model.SOC_H2_max = pyo.Param(initialize=p["SOC_H2_max"])
    model.SOC_H2_ini = pyo.Param(initialize=p["SOC_H2_ini"])
    
    # ---------- Device capacities ----------
    model.Chpmax = pyo.Param(initialize=p["Chpmax"])
    model.Fmax = pyo.Param(initialize=p["Fmax"])
    model.CBmax = pyo.Param(initialize=p["CBmax"])
    
    # ---------- Electric Heat Pump ----------
    model.COP = pyo.Param(initialize=p["COP"])
    model.C_EHP_min = pyo.Param(initialize=p["C_EHP_min"])
    model.C_EHP_max = pyo.Param(initialize=p["C_EHP_max"])
    model.H_EHP_min = pyo.Param(initialize=p["H_EHP_min"])
    model.H_EHP_max = pyo.Param(initialize=p["H_EHP_max"])
    
    # ---------- Deterministic parameters (study day) ----------
    model.De = pyo.Param(
        model.T,
        initialize={t: DATA["DE"][(study_day, t)] for t in time_periods}
    )

    model.Dh = pyo.Param(
        model.T,
        initialize={t: DATA["DH"][(study_day, t)] for t in time_periods}
    )

    model.Dc = pyo.Param(
        model.T,
        initialize={t: DATA["DC"][(study_day, t)] for t in time_periods}
    )

    model.lam_DA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods}
    )

    model.lam_g = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods}
    )

    # ---------- Stochastic parameters (scenario days) ----------
    model.lam_IDA = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Precio_IDA"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )

    model.Wind = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["Wind"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )
    
    # =====================================================================
    # VARIABLES
    # =====================================================================
    
    # ---------- First stage ----------
    model.E_DA = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # ---------- Second stage ----------
    
    # Electricity Base
    model.E_IDA = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.Wind_used = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.Curt = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals) 
    
    # ESS (Battery)
    model.E_c = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_d = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.T, model.S, domain=pyo.Reals, bounds=(model.SOC_min, model.SOC_max))
    model.I_ch = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.I_dch = pyo.Var(model.T, model.S, domain=pyo.Binary)
    
    # H2 Storage & Conversion Systems
    model.E_el = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_F = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_DES = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_com = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.HY_in = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.HY_out = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.SOC_H2 = pyo.Var(model.T, model.S, domain=pyo.Reals, bounds=(model.SOC_H2_min, model.SOC_H2_max))
    
    # Tri-state Logic & Fuel Cell Binaries
    model.I_off = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.I_stb = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.I_el = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.I_f = pyo.Var(model.T, model.S, domain=pyo.Binary)
    
    # Gas & Heat / Cooling Coupling
    model.G1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.G2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.H1 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H2 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    model.E_3 = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H_EHP = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.C_EHP = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.I_h = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.I_c = pyo.Var(model.T, model.S, domain=pyo.Binary)
    
    # =====================================================================
    # OBJECTIVE
    # =====================================================================
    
    def obj_rule(m):
        return (
            sum(
                m.lam_DA[t] * m.E_DA[t]
                + m.lam_g[t] * m.G[t]
                for t in m.T
            )
            +
            sum(
                PROB[s] * sum(
                    m.lam_IDA[t, s] * m.E_IDA[t, s]
                    + eps * (m.Wind[t, s] - m.Wind_used[t, s])
                    for t in m.T
                )
                for s in m.S
            )
        )
    
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # =====================================================================
    # CONSTRAINTS
    # =====================================================================
    
    # ---------------------------------------------------------------------
    # 1. BASE EH5 BALANCES & COUPLED ELECTRICAL INPUT BUS
    # ---------------------------------------------------------------------
    def eq_EH5_a(m, t, s):
        return m.Wind_used[t, s] + m.E_DA[t] + m.E_IDA[t, s] == m.E_DES[t, s] + m.E_com[t, s] + m.E_el[t, s] + m.E_c[t, s] + m.E_2[t, s]
    model.eq_EH5_a = pyo.Constraint(model.T, model.S, rule=eq_EH5_a)
    
    def eq_EH5_b(m, t, s): return m.Wind_used[t, s] <= m.Wind[t, s]
    model.eq_EH5_b = pyo.Constraint(model.T, model.S, rule=eq_EH5_b)
    
    def eq_EH5_c(m, t, s): return m.Curt[t, s] == m.Wind[t, s] - m.Wind_used[t, s]
    model.eq_EH5_c = pyo.Constraint(model.T, model.S, rule=eq_EH5_c)
    
    def eq_EH5_d(m, t, s):
        return m.E_F[t, s] + m.eta_ee * m.E_2[t, s] + m.E_d[t, s] + m.eta_ge * m.G1[t, s] == m.De[t] + m.E_3[t, s]
    model.eq_EH5_d = pyo.Constraint(model.T, model.S, rule=eq_EH5_d)

    # ---------------------------------------------------------------------
    # 2. ELECTRIC BATTERY STORAGE SYSTEM (ESS)
    # ---------------------------------------------------------------------
    def eq_EH5_e(m, t, s):
        if t == m.T.first():
            return m.SOC[t, s] == m.SOC_ini + m.eta_c * m.E_c[t, s] - m.E_d[t, s] / m.eta_d
        return m.SOC[t, s] == m.SOC[t-1, s] + m.eta_c * m.E_c[t, s] - m.E_d[t, s] / m.eta_d
    model.eq_EH5_e = pyo.Constraint(model.T, model.S, rule=eq_EH5_e)
    
    def eq_EH5_f_lower(m, t, s): return m.E_min_c * m.I_ch[t, s] <= m.E_c[t, s]
    model.eq_EH5_f_lower = pyo.Constraint(model.T, model.S, rule=eq_EH5_f_lower)
    
    def eq_EH5_f_upper(m, t, s): return m.E_c[t, s] <= m.E_max_c * m.I_ch[t, s]
    model.eq_EH5_f_upper = pyo.Constraint(model.T, model.S, rule=eq_EH5_f_upper)
    
    def eq_EH5_g_lower(m, t, s): return m.E_min_d * m.I_dch[t, s] <= m.E_d[t, s]
    model.eq_EH5_g_lower = pyo.Constraint(model.T, model.S, rule=eq_EH5_g_lower)
    
    def eq_EH5_g_upper(m, t, s): return m.E_d[t, s] <= m.E_max_d * m.I_dch[t, s]
    model.eq_EH5_g_upper = pyo.Constraint(model.T, model.S, rule=eq_EH5_g_upper)
    
    def eq_EH5_i(m, t, s): return m.I_dch[t, s] + m.I_ch[t, s] <= 1
    model.eq_EH5_i = pyo.Constraint(model.T, model.S, rule=eq_EH5_i)

    # ---------------------------------------------------------------------
    # 3. GAS, HEATING, AND COOLING COUPLING (EH5)
    # ---------------------------------------------------------------------
    def eq_EH5_j(m, t, s): return m.G[t] == m.G1[t, s] + m.G2[t, s]
    model.eq_EH5_j = pyo.Constraint(model.T, model.S, rule=eq_EH5_j)
    
    def eq_EH5_k(m, t, s): return m.eta_gh * m.G1[t, s] + m.H1[t, s] + m.H_EHP[t, s] == m.Dh[t]
    model.eq_EH5_k = pyo.Constraint(model.T, model.S, rule=eq_EH5_k)
    
    def eq_EH5_l(m, t, s): return m.eta_ghf * m.G2[t, s] == m.H1[t, s] + m.H2[t, s]
    model.eq_EH5_l = pyo.Constraint(model.T, model.S, rule=eq_EH5_l)
    
    def eq_EH5_m(m, t, s): return m.eta_hc * m.H2[t, s] + m.C_EHP[t, s] == m.Dc[t]
    model.eq_EH5_m = pyo.Constraint(model.T, model.S, rule=eq_EH5_m)
    
    def eq_EH5_n(m, t, s): return m.C_EHP[t, s] + m.H_EHP[t, s] == m.E_3[t, s] * m.COP
    model.eq_EH5_n = pyo.Constraint(model.T, model.S, rule=eq_EH5_n)
    
    def eq_EH5_o_lower(m, t, s): return m.H_EHP_min * m.I_h[t, s] <= m.H_EHP[t, s]
    model.eq_EH5_o_lower = pyo.Constraint(model.T, model.S, rule=eq_EH5_o_lower)
    
    def eq_EH5_o_upper(m, t, s): return m.H_EHP[t, s] <= m.H_EHP_max * m.I_h[t, s]
    model.eq_EH5_o_upper = pyo.Constraint(model.T, model.S, rule=eq_EH5_o_upper)
    
    def eq_EH5_p_lower(m, t, s): return m.C_EHP_min * m.I_c[t, s] <= m.C_EHP[t, s]
    model.eq_EH5_p_lower = pyo.Constraint(model.T, model.S, rule=eq_EH5_p_lower)
    
    def eq_EH5_p_upper(m, t, s): return m.C_EHP[t, s] <= m.C_EHP_max * m.I_c[t, s]
    model.eq_EH5_p_upper = pyo.Constraint(model.T, model.S, rule=eq_EH5_p_upper)
    
    def eq_EH5_q(m, t, s): return m.I_h[t, s] + m.I_c[t, s] <= 1
    model.eq_EH5_q = pyo.Constraint(model.T, model.S, rule=eq_EH5_q)

    def limit_G1(m, t, s): return m.G1[t, s] <= m.Chpmax
    model.limit_G1 = pyo.Constraint(model.T, model.S, rule=limit_G1)
    
    def limit_G2(m, t, s): return m.G2[t, s] <= m.Fmax
    model.limit_G2 = pyo.Constraint(model.T, model.S, rule=limit_G2)
    
    def limit_H2(m, t, s): return m.H2[t, s] <= m.CBmax
    model.limit_H2 = pyo.Constraint(model.T, model.S, rule=limit_H2)

    # ---------------------------------------------------------------------
    # 4. HIGH-TEMPERATURE SOEC ELECTROLYZER
    # --------------------------------------------------------------------- 
    def eq_SOEC_states(m, t, s): return m.I_off[t, s] + m.I_stb[t, s] + m.I_el[t, s] == 1
    model.eq_SOEC_states = pyo.Constraint(model.T, model.S, rule=eq_SOEC_states)
    
    def eq_SOEC_startup_logic(m, t, s):
        if t == m.T.first(): return pyo.Constraint.Skip
        return m.I_el[t, s] <= m.I_stb[t-1, s] + m.I_el[t-1, s]
    model.eq_SOEC_startup_logic = pyo.Constraint(model.T, model.S, rule=eq_SOEC_startup_logic)
    
    def eq_SOEC_shutdown_logic(m, t, s):
        if t == m.T.first(): return pyo.Constraint.Skip
        return m.I_off[t, s] <= m.I_off[t-1, s] + m.I_el[t-1, s]
    model.eq_SOEC_shutdown_logic = pyo.Constraint(model.T, model.S, rule=eq_SOEC_shutdown_logic)
    
    def eq_SOEC_no_direct_start(m, t, s):
        h = int(pyo.value(m.h_warmup))
        if t <= h: return pyo.Constraint.Skip
        off_sum = sum(m.I_off[t - tau, s] for tau in range(1, h + 1))
        return m.I_el[t, s] <= (1 / h) * off_sum + m.I_stb[t-1, s] + m.I_el[t-1, s]
    model.eq_SOEC_no_direct_start = pyo.Constraint(model.T, model.S, rule=eq_SOEC_no_direct_start)
    
    def eq_SOEC_power_lower(m, t, s): return m.el_min * m.I_el[t, s] + m.p_stb * m.I_stb[t, s] <= m.E_el[t, s]
    model.eq_SOEC_power_lower = pyo.Constraint(model.T, model.S, rule=eq_SOEC_power_lower)
    
    def eq_SOEC_power_upper(m, t, s): return m.E_el[t, s] <= m.el_max * m.I_el[t, s] + m.p_stb * m.I_stb[t, s]
    model.eq_SOEC_power_upper = pyo.Constraint(model.T, model.S, rule=eq_SOEC_power_upper)
    
    def eq_SOEC_production(m, t, s): return m.HY_in[t, s] == m.eta_EL * (m.E_el[t, s] - m.p_stb * m.I_stb[t, s])
    model.eq_SOEC_production = pyo.Constraint(model.T, model.S, rule=eq_SOEC_production)
    
    def eq_SOEC_ramp_lower(m, t, s):
        if t == m.T.first(): return pyo.Constraint.Skip
        return -m.ramp_down_el <= m.E_el[t, s] - m.E_el[t-1, s]
    model.eq_SOEC_ramp_lower = pyo.Constraint(model.T, model.S, rule=eq_SOEC_ramp_lower)
    
    def eq_SOEC_ramp_upper(m, t, s):
        if t == m.T.first(): return pyo.Constraint.Skip
        return m.E_el[t, s] - m.E_el[t-1, s] <= m.ramp_up_el
    model.eq_SOEC_ramp_upper = pyo.Constraint(model.T, model.S, rule=eq_SOEC_ramp_upper)

    # ---------------------------------------------------------------------
    # 5. AUXILIARY SYSTEMS
    # ---------------------------------------------------------------------
    def eq_DES_power(m, t, s): return m.E_DES[t, s] == m.alpha_DES * (m.E_el[t, s] - m.p_stb * m.I_stb[t, s])
    model.eq_DES_power = pyo.Constraint(model.T, model.S, rule=eq_DES_power)
    
    def eq_COM_power(m, t, s): return m.E_com[t, s] == m.eta_com * m.rho_com * m.HY_in[t, s]
    model.eq_COM_power = pyo.Constraint(model.T, model.S, rule=eq_COM_power)

    # ---------------------------------------------------------------------
    # 6. FUEL CELL
    # ---------------------------------------------------------------------
    def eq_FC_conversion(m, t, s): return m.E_F[t, s] == m.eta_F * m.HY_out[t, s]
    model.eq_FC_conversion = pyo.Constraint(model.T, model.S, rule=eq_FC_conversion)
    
    def eq_FC_bounds_lower(m, t, s): return m.fc_min * m.I_f[t, s] <= m.E_F[t, s]
    model.eq_FC_bounds_lower = pyo.Constraint(model.T, model.S, rule=eq_FC_bounds_lower)
    
    def eq_FC_bounds_upper(m, t, s): return m.E_F[t, s] <= m.fc_max * m.I_f[t, s]
    model.eq_FC_bounds_upper = pyo.Constraint(model.T, model.S, rule=eq_FC_bounds_upper)
    
    def eq_FC_ramp_lower(m, t, s):
        if t == m.T.first(): return pyo.Constraint.Skip
        return -m.ramp_down_F <= m.E_F[t, s] - m.E_F[t-1, s]
    model.eq_FC_ramp_lower = pyo.Constraint(model.T, model.S, rule=eq_FC_ramp_lower)
    
    def eq_FC_ramp_upper(m, t, s):
        if t == m.T.first(): return pyo.Constraint.Skip
        return m.E_F[t, s] - m.E_F[t-1, s] <= m.ramp_up_F
    model.eq_FC_ramp_upper = pyo.Constraint(model.T, model.S, rule=eq_FC_ramp_upper)

    # ---------------------------------------------------------------------
    # 7. HYDROGEN STORAGE TANK (HSS)
    # ---------------------------------------------------------------------
    def eq_HSS_SOC(m, t, s):
        if t == m.T.first():
            return m.SOC_H2[t, s] == m.SOC_H2_ini + m.eta_H2 * m.HY_in[t, s] - (m.HY_out[t, s] / m.eta_H2)
        return m.SOC_H2[t, s] == m.SOC_H2[t-1, s] + m.eta_H2 * m.HY_in[t, s] - (m.HY_out[t, s] / m.eta_H2)
    model.eq_HSS_SOC = pyo.Constraint(model.T, model.S, rule=eq_HSS_SOC)
    
    def eq_HSS_flows_in(m, t, s): return m.HY_in[t, s] <= m.SOC_H2_max * m.I_el[t, s]
    model.eq_HSS_flows_in = pyo.Constraint(model.T, model.S, rule=eq_HSS_flows_in)
    
    def eq_HSS_flows_out(m, t, s): return m.HY_out[t, s] <= m.SOC_H2_max * m.I_f[t, s]
    model.eq_HSS_flows_out = pyo.Constraint(model.T, model.S, rule=eq_HSS_flows_out)
    
    def eq_H2_decoupling(m, t, s): return m.I_el[t, s] + m.I_f[t, s] <= 1
    model.eq_H2_decoupling = pyo.Constraint(model.T, model.S, rule=eq_H2_decoupling)

    return model