# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 09:43:51 2026

@author: Miriam_Ucendo
@file_name: models.py (The .mod equivalent)
Defines the structure of the Energy Hub optimization model.
"""

import pyomo.environ as pyo

eps = 1

## preliminar models ##

###############################################################################
######  WITHOUT ESS   #########################################################
###############################################################################

def EH_model(time_periods, p, data, lambda_today_s, E_RES_s):
    # ─── MODEL ────────────────────────────────────────────────────────
    model = pyo.ConcreteModel(name="Full_Energy_Hub")
    # DUAL
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # ─── SETS ────────────────────────────────────────────────────────────────────
    model.T = pyo.Set(initialize=time_periods)
    
    # ─── PARAMETERS ───────────────────────────────────────────────────────
    # Scalar
    model.eta_ee   = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge   = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh   = pyo.Param(initialize=p["eta_gh"])    # CHP heat efficiency  (eta_gh^CHP)
    model.eta_ghf  = pyo.Param(initialize=p["eta_ghf"])   # Furnace efficiency   (eta_gh^F)
    model.eta_hc   = pyo.Param(initialize=p["eta_hc"])
    model.Chpmax   = pyo.Param(initialize=p["Chpmax"])
    model.Fmax     = pyo.Param(initialize=p["Fmax"])
    model.CBmax    = pyo.Param(initialize=p["CBmax"])
    
    # Indexed Parameters
    model.Dh        = pyo.Param(model.T, initialize={t: data["Dh"][t] for t in time_periods})
    model.De        = pyo.Param(model.T, initialize={t: data["De"][t] for t in time_periods})
    model.Dc        = pyo.Param(model.T, initialize={t: data["Dc"][t] for t in time_periods})
    model.lam_DA    = pyo.Param(model.T, initialize={t: data["lambda_DA"][t] for t in time_periods})
    model.lam_g     = pyo.Param(model.T, initialize={t: data["lambda_g"][t] for t in time_periods}) 
    
    model.E_RES     = pyo.Param(model.T, initialize={t: E_RES_s[t] for t in time_periods})
    model.lam_TODAY = pyo.Param(model.T, initialize={t: lambda_today_s[t] for t in time_periods})
    
    # ─── VARIABLES ───────────────────────────────────────────────────────────────
    model.E_DA       = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Day-ahead electricity purchased
    model.E_TODAY    = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Intraday electricity purchased
    model.E_RES_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Renewable electricity used
    model.E          = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Total electricity into hub
    model.G          = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Total gas purchased
    model.G1         = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Gas to CHP
    model.G2         = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Gas to furnace
    model.H1         = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Heat from furnace to demand
    model.H2         = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Heat from furnace to chiller
    
    # ─── OBJECTIVE FUNCTION ──────────────────────────────────────────────────────
    # min cost = sum_t [ lambda_DA * E_DA + lambda_TODAY * E_TODAY + lambda_g * G ]
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t] +
            m.lam_TODAY[t] * m.E_TODAY[t] +
            m.lam_g[t] * m.G[t] +
            eps * (m.E_RES[t] - m.E_RES_used[t])
            for t in m.T
        )
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # ─── CONSTRAINTS ─────────────────────────────────────────────────────────────
    
    # (b) Electricity supply balance: RES_used + DA + TODAY = E
    def eq_b(m, t):
        return m.E_RES_used[t] + m.E_DA[t] + m.E_TODAY[t] == m.E[t]
    model.eq_b = pyo.Constraint(model.T, rule=eq_b)
    
    # (b.2) Curtailment
    def eq_curt(m,t):
        return m.E_RES_used[t] <= m.E_RES[t]
    model.eq_curt = pyo.Constraint(model.T, rule=eq_curt)
    
    # (c) Electricity demand satisfaction
    def eq_c(m, t):
        return m.eta_ee * m.E[t] + m.eta_ge * m.G1[t] == m.De[t]
    model.eq_c = pyo.Constraint(model.T, rule=eq_c)
    
    # (d) Gas balance: G = G1 + G2
    def eq_d(m, t):
        return m.G[t] == m.G1[t] + m.G2[t]
    model.eq_d = pyo.Constraint(model.T, rule=eq_d)
    
    # (e) Furnace heat output split: eta_ghf * G2 = H1 + H2
    def eq_e(m, t):
        return m.eta_ghf * m.G2[t] == m.H1[t] + m.H2[t]
    model.eq_e = pyo.Constraint(model.T, rule=eq_e)
    
    # (f) Heat demand satisfaction: eta_gh^CHP * G1 + H1 = Dh
    def eq_f(m, t):
        return m.eta_gh * m.G1[t] + m.H1[t] == m.Dh[t]
    model.eq_f = pyo.Constraint(model.T, rule=eq_f)
    
    # (g) Cooling demand satisfaction: eta_hc * H2 = Dc
    def eq_g(m, t):
        return m.eta_hc * m.H2[t] == m.Dc[t]
    model.eq_g = pyo.Constraint(model.T, rule=eq_g)
    
    # (h) CHP gas capacity limit
    model.limit_G1 = pyo.Constraint(model.T, rule=lambda m, t: m.G1[t] <= m.Chpmax)
    
    # (i) Furnace gas capacity limit
    model.limit_G2 = pyo.Constraint(model.T, rule=lambda m, t: m.G2[t] <= m.Fmax)
    
    # (j) Chiller heat input capacity limit
    model.limit_H2 = pyo.Constraint(model.T, rule=lambda m, t: m.H2[t] <= m.CBmax)
    
    return model

def EH_stc_model(time_periods, SCENARIOS, PROB, p, data, lambda_today, E_RES): 
    # ─── MODEL ────────────────────────────────────────────────────────
    model = pyo.ConcreteModel(name="Stc_Energy_Hub")
    # DUAL
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # ─── SETS ────────────────────────────────────────────────────────────────────
    model.T = pyo.Set(initialize=time_periods)
    model.S = pyo.Set(initialize=SCENARIOS)

    # ─── PARAMETERS ───────────────────────────────────────────────────────
    # Scalar
    model.eta_ee   = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge   = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh   = pyo.Param(initialize=p["eta_gh"])    # CHP heat efficiency  (eta_gh^CHP)
    model.eta_ghf  = pyo.Param(initialize=p["eta_ghf"])   # Furnace efficiency   (eta_gh^F)
    model.eta_hc   = pyo.Param(initialize=p["eta_hc"])
    model.Chpmax   = pyo.Param(initialize=p["Chpmax"])
    model.Fmax     = pyo.Param(initialize=p["Fmax"])
    model.CBmax    = pyo.Param(initialize=p["CBmax"])

    # Indexed Parameters
    model.Dh          = pyo.Param(model.T, initialize={t: data["Dh"][t] for t in time_periods})
    model.De          = pyo.Param(model.T, initialize={t: data["De"][t] for t in time_periods})
    model.Dc          = pyo.Param(model.T, initialize={t: data["Dc"][t] for t in time_periods})
    model.E_RES       = pyo.Param(model.T, model.S, initialize=E_RES)

    model.lam_DA      = pyo.Param(model.T, initialize={t: data["lambda_DA"][t] for t in time_periods})
    model.lam_TODAY   = pyo.Param(model.T, model.S, initialize=lambda_today)
    model.lam_g       = pyo.Param(model.T, initialize={t: data["lambda_g"][t] for t in time_periods})

    # ─── VARIABLES ───────────────────────────────────────────────────────────────
    # FIRST-STAGE
    model.E_DA    = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G       = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # SECOND-STAGE
    model.E_TODAY     = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_RES_used  = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E           = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.G1          = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.G2          = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H1          = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H2          = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

    # ─── OBJECTIVE FUNCTION ──────────────────────────────────────────────────────
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t] + m.lam_g[t] * m.G[t] + 
            sum(PROB[s] * m.lam_TODAY[t,s] * m.E_TODAY[t,s] + 
                eps * (m.E_RES[t,s] - m.E_RES_used[t,s]) for s in m.S)
            for t in m.T
        )
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ─── CONSTRAINTS ─────────────────────────────────────────────────────────────

    # (b) Electricity supply balance: RES + DA + TODAY = E
    def eq_b(m, t, s):
        return m.E_RES_used[t,s] + m.E_DA[t] + m.E_TODAY[t,s] == m.E[t,s]
    model.eq_b = pyo.Constraint(model.T, model.S, rule=eq_b)

    # (b.2) Curtailment
    def eq_curt(m, t, s):
        return m.E_RES_used[t,s] <= m.E_RES[t,s]
    model.eq_curt = pyo.Constraint(model.T, model.S, rule=eq_curt)
    
    # (c) Electricity demand satisfaction
    def eq_c(m, t, s):
        return m.eta_ee * m.E[t,s] + m.eta_ge * m.G1[t,s] == m.De[t]
    model.eq_c = pyo.Constraint(model.T, model.S, rule=eq_c)

    # (d) Gas balance: G = G1 + G2
    def eq_d(m, t, s):
        return m.G[t] == m.G1[t,s] + m.G2[t,s]
    model.eq_d = pyo.Constraint(model.T, model.S, rule=eq_d)

    # (e) Furnace heat output split: eta_ghf * G2 = H1 + H2
    def eq_e(m, t, s):
        return m.eta_ghf * m.G2[t,s] == m.H1[t,s] + m.H2[t,s]
    model.eq_e = pyo.Constraint(model.T, model.S, rule=eq_e)

    # (f) Heat demand satisfaction: eta_gh^CHP * G1 + H1 = Dh
    def eq_f(m, t, s):
        return m.eta_gh * m.G1[t,s] + m.H1[t,s] == m.Dh[t]
    model.eq_f = pyo.Constraint(model.T, model.S, rule=eq_f)

    # (g) Cooling demand satisfaction: eta_hc * H2 = Dc
    def eq_g(m, t, s):
        return m.eta_hc * m.H2[t,s] == m.Dc[t]
    model.eq_g = pyo.Constraint(model.T, model.S, rule=eq_g)

    # (h) CHP gas capacity limit
    model.limit_G1 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.G1[t,s] <= m.Chpmax)

    # (i) Furnace gas capacity limit
    model.limit_G2 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.G2[t,s] <= m.Fmax)

    # (j) CB heat input limit
    model.limit_H2 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.H2[t,s] <= m.CBmax)
    
    return model

###############################################################################
######  WITH ESS   ############################################################
###############################################################################

def EH2_ESS_model(time_periods, p, data, lambda_today_s, E_RES_s):
    """
    Builds and returns a Pyomo ConcreteModel based on provided scenario data.
    """
    # ─── MODEL ────────────────────────────────────────────────────────
    model = pyo.ConcreteModel(name="Full_Energy_Hub_with_ESS")
    # DUAL
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # ─── SETS ────────────────────────────────────────────────────────────────
    model.T = pyo.Set(initialize=time_periods)
    
    # ─── PARAMETERS ──────────────────────────────────────────────────────────
    model.eta_ee   = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge   = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh   = pyo.Param(initialize=p["eta_gh"])    
    model.eta_ghf  = pyo.Param(initialize=p["eta_ghf"])   
    model.eta_hc   = pyo.Param(initialize=p["eta_hc"])
    
    model.eta_c    = pyo.Param(initialize=p["eta_c"])       
    model.eta_d    = pyo.Param(initialize=p["eta_d"])       
    model.E_min_c  = pyo.Param(initialize=p["E_min_c"])     
    model.E_max_c  = pyo.Param(initialize=p["E_max_c"])     
    model.E_min_d  = pyo.Param(initialize=p["E_min_d"])     
    model.E_max_d  = pyo.Param(initialize=p["E_max_d"])     
    model.SOC_min  = pyo.Param(initialize=p["SOC_min"])     
    model.SOC_max  = pyo.Param(initialize=p["SOC_max"])     
    model.SOC_ini  = pyo.Param(initialize=p["SOC_ini"])     

    model.Chpmax   = pyo.Param(initialize=p["Chpmax"])
    model.Fmax     = pyo.Param(initialize=p["Fmax"])
    model.CBmax    = pyo.Param(initialize=p["CBmax"])
    
    model.Dh        = pyo.Param(model.T, initialize={t: data["Dh"][t] for t in time_periods})
    model.De        = pyo.Param(model.T, initialize={t: data["De"][t] for t in time_periods})
    model.Dc        = pyo.Param(model.T, initialize={t: data["Dc"][t] for t in time_periods})
    model.lam_DA    = pyo.Param(model.T, initialize={t: data["lambda_DA"][t] for t in time_periods})
    model.lam_g     = pyo.Param(model.T, initialize={t: data["lambda_g"][t] for t in time_periods}) 
    
    model.E_RES    = pyo.Param(model.T, initialize={t: E_RES_s[t] for t in time_periods})
    model.lam_TODAY = pyo.Param(model.T, initialize={t: lambda_today_s[t] for t in time_periods})
    
    # ─── VARIABLES ───────────────────────────────────────────────────────────
    # Electricity
    model.E_DA       = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.E_TODAY    = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.E_RES_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.E_2        = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    
    # ESS
    model.E_c     = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.E_d     = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.SOC     = pyo.Var(model.T, domain=pyo.Reals, bounds=(model.SOC_min, model.SOC_max))  
    model.I_ch    = pyo.Var(model.T, domain=pyo.Binary)  
    model.I_dch   = pyo.Var(model.T, domain=pyo.Binary)  
        
    # Gas
    model.G       = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.G1      = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.G2      = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    
    #Furnance
    model.H1      = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.H2      = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    
    
    # ─── OBJECTIVE FUNCTION ──────────────────────────────────────────────────
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t] +
            m.lam_TODAY[t] * m.E_TODAY[t] +
            m.lam_g[t] * m.G[t] +
            eps * (m.E_RES[t] - m.E_RES_used[t])
            for t in m.T
        )
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
   
    # ─── CONSTRAINTS ─────────────────────────────────────────────────────────
    
    # Electric input balance
    def eq_b(m, t):
        return m.E_RES_used[t] + m.E_DA[t] + m.E_TODAY[t] == m.E_c[t] + m.E_2[t]
    model.eq_b = pyo.Constraint(model.T, rule=eq_b)
    
    # (b.2) Curtailment
    def eq_curt(m,t):
        return m.E_RES_used[t] <= m.E_RES[t]
    model.eq_curt = pyo.Constraint(model.T, rule=eq_curt)
    
    # Electricity demand balance
    def eq_d(m, t):
        return m.eta_ee * m.E_2[t] + m.E_d[t] + m.eta_ge * m.G1[t] == m.De[t]
    model.eq_d = pyo.Constraint(model.T, rule=eq_d)
    
    # State of Charge (SOC) tracking
    def eq_e(m, t):
        if t == m.T.first():
            return m.SOC[t] == m.SOC_ini + m.E_c[t] * m.eta_c - m.E_d[t] / m.eta_d
        else:
            return m.SOC[t] == m.SOC[t-1] + m.E_c[t] * m.eta_c - m.E_d[t] / m.eta_d
    model.eq_e = pyo.Constraint(model.T, rule=eq_e)
    
    # Battery charging lower limit
    def eq_f_lower(m, t):
        return m.E_min_c * m.I_ch[t] <= m.E_c[t]
    model.eq_f_lower = pyo.Constraint(model.T, rule=eq_f_lower)
    
    # Battery charging upper limit
    def eq_f_upper(m, t):
        return m.E_c[t] <= m.E_max_c * m.I_ch[t]
    model.eq_f_upper = pyo.Constraint(model.T, rule=eq_f_upper)
    
    # Battery discharging lower limit
    def eq_g_lower(m, t):
        return m.E_min_d * m.I_dch[t] <= m.E_d[t]
    model.eq_g_lower = pyo.Constraint(model.T, rule=eq_g_lower)
    
    # Battery discharging upper limit
    def eq_g_upper(m, t):
        return m.E_d[t] <= m.E_max_d * m.I_dch[t]
    model.eq_g_upper = pyo.Constraint(model.T, rule=eq_g_upper)

    # Battery simultaneous charge/discharge exclusion
    def eq_i(m, t):
        return m.I_dch[t] + m.I_ch[t] <= 1
    model.eq_i = pyo.Constraint(model.T, rule=eq_i)
    
    # Gas input split
    def eq_j(m, t):
        return m.G[t] == m.G1[t] + m.G2[t]
    model.eq_j = pyo.Constraint(model.T, rule=eq_j)
    
    # Heat demand balance
    def eq_k(m, t):
        return m.eta_gh * m.G1[t] + m.H1[t] == m.Dh[t]
    model.eq_k = pyo.Constraint(model.T, rule=eq_k)
    
    # Furnace heat generation split
    def eq_l(m, t):
        return m.eta_ghf * m.G2[t] == m.H1[t] + m.H2[t]
    model.eq_l = pyo.Constraint(model.T, rule=eq_l)
    
    # Cooling demand balance
    def eq_m(m, t):
        return m.eta_hc * m.H2[t] == m.Dc[t]
    model.eq_m = pyo.Constraint(model.T, rule=eq_m)
    
    # CHP gas capacity limit
    model.limit_G1 = pyo.Constraint(model.T, rule=lambda m, t: m.G1[t] <= m.Chpmax)
    
    # Furnace gas capacity limit
    model.limit_G2 = pyo.Constraint(model.T, rule=lambda m, t: m.G2[t] <= m.Fmax)
    
    # Chiller heat input capacity limit
    model.limit_H2 = pyo.Constraint(model.T, rule=lambda m, t: m.H2[t] <= m.CBmax)   
    
    return model

# Now, the stochastic model, which needs the scenarios and the probab of each scenario
def EH2_ESS_stc_model(time_periods, SCENARIOS, PROB, p, data, lambda_today, E_RES): 
    # ─── MODEL ────────────────────────────────────────────────────────
    model = pyo.ConcreteModel(name="ESS_Stc_Energy_Hub")
    # DUAL
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # ─── SETS ────────────────────────────────────────────────────────────────────
    model.T = pyo.Set(initialize=time_periods)
    model.S = pyo.Set(initialize=SCENARIOS)

    # ─── PARAMETERS ───────────────────────────────────────────────────────
    # Scalar
    model.eta_ee   = pyo.Param(initialize=p["eta_ee"])
    model.eta_ge   = pyo.Param(initialize=p["eta_ge"])
    model.eta_gh   = pyo.Param(initialize=p["eta_gh"])    # CHP heat efficiency  (eta_gh^CHP)
    model.eta_ghf  = pyo.Param(initialize=p["eta_ghf"])   # Furnace efficiency   (eta_gh^F)
    model.eta_hc   = pyo.Param(initialize=p["eta_hc"])
    model.Chpmax   = pyo.Param(initialize=p["Chpmax"])
    model.Fmax     = pyo.Param(initialize=p["Fmax"])
    model.CBmax    = pyo.Param(initialize=p["CBmax"])

    # Capacity limits
    model.eta_c    = pyo.Param(initialize=p["eta_c"])       
    model.eta_d    = pyo.Param(initialize=p["eta_d"])       
    model.E_min_c  = pyo.Param(initialize=p["E_min_c"])     
    model.E_max_c  = pyo.Param(initialize=p["E_max_c"])     
    model.E_min_d  = pyo.Param(initialize=p["E_min_d"])     
    model.E_max_d  = pyo.Param(initialize=p["E_max_d"])     
    model.SOC_min  = pyo.Param(initialize=p["SOC_min"])     
    model.SOC_max  = pyo.Param(initialize=p["SOC_max"])     
    model.SOC_ini  = pyo.Param(initialize=p["SOC_ini"])

    # Indexed Parameters
    
    model.Dh          = pyo.Param(model.T, initialize={t: data["Dh"][t] for t in time_periods})
    model.De          = pyo.Param(model.T, initialize={t: data["De"][t] for t in time_periods})
    model.Dc          = pyo.Param(model.T, initialize={t: data["Dc"][t] for t in time_periods})
    model.E_RES       = pyo.Param(model.T, model.S, initialize=E_RES)

    model.lam_DA      = pyo.Param(model.T, initialize={t: data["lambda_DA"][t] for t in time_periods})
    model.lam_TODAY   = pyo.Param(model.T, model.S, initialize=lambda_today)
    model.lam_g       = pyo.Param(model.T, initialize={t: data["lambda_g"][t] for t in time_periods})

    # ─── VARIABLES ───────────────────────────────────────────────────────────────
    # FIRST-STAGE
    model.E_DA    = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.G       = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # SECOND-STAGE
    model.E_TODAY    = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.E_RES_used = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.E_c        = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.E_2        = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.E_d        = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.G1         = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.G2         = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H1         = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.H2         = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
            
    model.SOC     = pyo.Var(model.T, model.S, domain=pyo.Reals, bounds=(model.SOC_min, model.SOC_max))  

    model.I_ch    = pyo.Var(model.T, model.S, domain=pyo.Binary)  
    model.I_dch   = pyo.Var(model.T, model.S, domain=pyo.Binary)  

    # ─── OBJECTIVE FUNCTION ──────────────────────────────────────────────────────
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t] + m.lam_g[t] * m.G[t] + 
            sum(PROB[s] * m.lam_TODAY[t,s] * m.E_TODAY[t,s] + 
                eps * (m.E_RES[t,s] - m.E_RES_used[t,s]) for s in m.S)
            for t in m.T
        )
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ─── CONSTRAINTS (STOCHASTIC VERSION) ─────────────────────────────────────────

    # (b) Electric input balance
    def eq_b(m, t, s):
        return m.E_RES_used[t, s] + m.E_DA[t] + m.E_TODAY[t, s] == m.E_c[t, s] + m.E_2[t, s]
    model.eq_b = pyo.Constraint(model.T, model.S, rule=eq_b)
    
    # (b.2) Curtailment
    def eq_curt(m, t, s):
        return m.E_RES_used[t,s] <= m.E_RES[t,s]
    model.eq_curt = pyo.Constraint(model.T, model.S, rule=eq_curt)

    # (d) Electric demand 
    def eq_d(m, t, s):
        return m.eta_ee * m.E_2[t, s] + m.E_d[t, s] + m.eta_ge * m.G1[t, s] == m.De[t]
    model.eq_d = pyo.Constraint(model.T, model.S, rule=eq_d)

    # (e) State of Charge (SOC) tracking per scenario
    def eq_e(m, t, s):
        if t == m.T.first():
            return m.SOC[t, s] == m.SOC_ini + m.E_c[t, s] * m.eta_c - m.E_d[t, s] / m.eta_d
        else:
            return m.SOC[t, s] == m.SOC[t-1, s] + m.E_c[t, s] * m.eta_c - m.E_d[t, s] / m.eta_d
    model.eq_e = pyo.Constraint(model.T, model.S, rule=eq_e)

    # (f1) ESS charging limits: Lower Bound
    def eq_f_lower(m, t, s):
        return m.E_min_c * m.I_ch[t, s] <= m.E_c[t, s]
    model.eq_f_lower = pyo.Constraint(model.T, model.S, rule=eq_f_lower)

    # (f2) ESS charging limits: Upper Bound
    def eq_f_upper(m, t, s):
        return m.E_c[t, s] <= m.E_max_c * m.I_ch[t, s]
    model.eq_f_upper = pyo.Constraint(model.T, model.S, rule=eq_f_upper)

    # (g1) ESS discharging limits: Lower Bound
    def eq_g_lower(m, t, s):
        return m.E_min_d * m.I_dch[t, s] <= m.E_d[t, s]
    model.eq_g_lower = pyo.Constraint(model.T, model.S, rule=eq_g_lower)

    # (g2) ESS discharging limits: Upper Bound
    def eq_g_upper(m, t, s):
        return m.E_d[t, s] <= m.E_max_d * m.I_dch[t, s]
    model.eq_g_upper = pyo.Constraint(model.T, model.S, rule=eq_g_upper)

    # (i) Battery operations exclusivity (No charge & discharge at the same time)
    def eq_i(m, t, s):
        return m.I_dch[t, s] + m.I_ch[t, s] <= 1
    model.eq_i = pyo.Constraint(model.T, model.S, rule=eq_i)

    # (j) Gas input balance
    # G is 1st stage (no s). G1 and G2 depend on the scenario s.
    def eq_j(m, t, s):
        return m.G[t] == m.G1[t, s] + m.G2[t, s]
    model.eq_j = pyo.Constraint(model.T, model.S, rule=eq_j)

    # (k) Heat demand satisfaction
    def eq_k(m, t, s):
        return m.eta_gh * m.G1[t, s] + m.H1[t, s] == m.Dh[t]
    model.eq_k = pyo.Constraint(model.T, model.S, rule=eq_k)

    # (l) Furnace efficiency and heat output split
    def eq_l(m, t, s):
        return m.eta_ghf * m.G2[t, s] == m.H1[t, s] + m.H2[t, s]
    model.eq_l = pyo.Constraint(model.T, model.S, rule=eq_l)

    # (m) Cooling demand satisfaction
    def eq_m(m, t, s):
        return m.eta_hc * m.H2[t, s] == m.Dc[t]
    model.eq_m = pyo.Constraint(model.T, model.S, rule=eq_m)

    # Capacity limits for Energy Hub elements per scenario
    model.limit_G1 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.G1[t, s] <= m.Chpmax)
    model.limit_G2 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.G2[t, s] <= m.Fmax)
    model.limit_H2 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.H2[t, s] <= m.CBmax)
    
    return model

###############################################################################
######  NEW COMPONENTS   #########################################################
###############################################################################

def EH3_EHP_model(time_periods, p, data, lambda_today_s, E_RES_s):
    # ─── MODEL ────────────────────────────────────────────────────────
    model = pyo.ConcreteModel(name="Full_Energy_Hub")
    # DUAL
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # ─── SETS ────────────────────────────────────────────────────────────────────
    model.T = pyo.Set(initialize=time_periods)
    
    # ─── PARAMETERS ───────────────────────────────────────────────────────
    # --- Scalar ---
    # efficiencies
    model.eta_c     = pyo.Param(initialize=p["eta_c"]) #ESS
    model.eta_d     = pyo.Param(initialize=p["eta_d"]) #ESS
    model.eta_ee    = pyo.Param(initialize=p["eta_ee"]) #Tr
    model.eta_ge    = pyo.Param(initialize=p["eta_ge"]) #CHP
    model.eta_gh    = pyo.Param(initialize=p["eta_gh"]) #CHP
    model.eta_hc    = pyo.Param(initialize=p["eta_hc"]) #CB
    model.eta_ghf   = pyo.Param(initialize=p["eta_ghf"]) #F
    
    # ESS
    model.E_min_c  = pyo.Param(initialize=p["E_min_c"])     
    model.E_max_c  = pyo.Param(initialize=p["E_max_c"])     
    model.E_min_d  = pyo.Param(initialize=p["E_min_d"])     
    model.E_max_d  = pyo.Param(initialize=p["E_max_d"])     
    model.SOC_min  = pyo.Param(initialize=p["SOC_min"])     
    model.SOC_max  = pyo.Param(initialize=p["SOC_max"])     
    model.SOC_ini  = pyo.Param(initialize=p["SOC_ini"])     
    
    # Device Capacities
    model.Chpmax   = pyo.Param(initialize=p["Chpmax"])
    model.Fmax     = pyo.Param(initialize=p["Fmax"])
    model.CBmax    = pyo.Param(initialize=p["CBmax"])
    
    # New Heat Pump (EHP) and COP parameters
    model.COP       = pyo.Param(initialize=p["COP"])
    model.C_EHP_min = pyo.Param(initialize=p["C_EHP_min"])
    model.C_EHP_max = pyo.Param(initialize=p["C_EHP_max"])
    model.H_EHP_min = pyo.Param(initialize=p["H_EHP_min"])
    model.H_EHP_max = pyo.Param(initialize=p["H_EHP_max"])
    
    # --- Indexed Parameters ---
    model.Dh        = pyo.Param(model.T, initialize={t: data["Dh"][t] for t in time_periods})
    model.De        = pyo.Param(model.T, initialize={t: data["De"][t] for t in time_periods})
    model.Dc        = pyo.Param(model.T, initialize={t: data["Dc"][t] for t in time_periods})
    model.lam_DA    = pyo.Param(model.T, initialize={t: data["lambda_DA"][t] for t in time_periods})
    model.lam_g     = pyo.Param(model.T, initialize={t: data["lambda_g"][t] for t in time_periods}) 
    
    model.E_RES    = pyo.Param(model.T, initialize={t: E_RES_s[t] for t in time_periods})
    model.lam_TODAY = pyo.Param(model.T, initialize={t: lambda_today_s[t] for t in time_periods})
    
    # ─── VARIABLES ───────────────────────────────────────────────────────────────
    # Electricity
    model.E_DA       = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.E_TODAY    = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.E_RES_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.E_2        = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    
    # ESS
    model.E_c     = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.E_d     = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.SOC     = pyo.Var(model.T, domain=pyo.Reals, bounds=(model.SOC_min, model.SOC_max))  
    # binary ESS
    model.I_ch    = pyo.Var(model.T, domain=pyo.Binary)  
    model.I_dch   = pyo.Var(model.T, domain=pyo.Binary)  
        
    # Gas
    model.G       = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.G1      = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.G2      = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    
    #Furnance
    model.H1      = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.H2      = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    
    # EHP    
    model.E_3     = pyo.Var(model.T, domain=pyo.NonNegativeReals)   #NEW (EHP)
    model.H_EHP   = pyo.Var(model.T, domain=pyo.NonNegativeReals) 
    model.C_EHP   = pyo.Var(model.T, domain=pyo.NonNegativeReals) 
    # binary EHP
    model.I_h     = pyo.Var(model.T, domain=pyo.Binary)  
    model.I_c     = pyo.Var(model.T, domain=pyo.Binary)  


    
    
    # ─── OBJECTIVE FUNCTION ──────────────────────────────────────────────────────
    # min cost = sum_t [ lambda_DA * E_DA + lambda_TODAY * E_TODAY + lambda_g * G ]
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t] +
            m.lam_TODAY[t] * m.E_TODAY[t] +
            m.lam_g[t] * m.G[t] +
            eps * (m.E_RES[t] - m.E_RES_used[t])
            for t in m.T
        )
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # ─── CONSTRAINTS ─────────────────────────────────────────────────────────────
    
    # (b) Electricity supply balance: RES + DA + TODAY = Ec + E2
    def eq_b(m, t):
        return m.E_RES_used[t] + m.E_DA[t] + m.E_TODAY[t] == m.E_2[t] + m.E_c[t]
    model.eq_b = pyo.Constraint(model.T, rule=eq_b)
    
    # (b.2) Curtailment
    def eq_curt(m,t):
        return m.E_RES_used[t] <= m.E_RES[t]
    model.eq_curt = pyo.Constraint(model.T, rule=eq_curt)
    
    # (c) Electricity demand satisfaction
    def eq_c(m, t):
        return m.eta_ee * m.E_2[t] + m.E_d[t] + m.eta_ge * m.G1[t] == m.E_3[t] + m.De[t]
    model.eq_c = pyo.Constraint(model.T, rule=eq_c)
    
    # (d) State of Charge (SOC) tracking
    def eq_d(m, t):
        if t == m.T.first():
            return m.SOC[t] == m.SOC_ini + m.E_c[t] * m.eta_c - m.E_d[t] / m.eta_d
        else:
            return m.SOC[t] == m.SOC[t-1] + m.E_c[t] * m.eta_c - m.E_d[t] / m.eta_d
    model.eq_d = pyo.Constraint(model.T, rule=eq_d)
    
    # Battery charging lower limit
    def eq_f_lower(m, t):
        return m.E_min_c * m.I_ch[t] <= m.E_c[t]
    model.eq_f_lower = pyo.Constraint(model.T, rule=eq_f_lower)
    
    # Battery charging upper limit
    def eq_f_upper(m, t):
        return m.E_c[t] <= m.E_max_c * m.I_ch[t]
    model.eq_f_upper = pyo.Constraint(model.T, rule=eq_f_upper)
    
    # Battery discharging lower limit
    def eq_g_lower(m, t):
        return m.E_min_d * m.I_dch[t] <= m.E_d[t]
    model.eq_g_lower = pyo.Constraint(model.T, rule=eq_g_lower)
    
    # Battery discharging upper limit
    def eq_g_upper(m, t):
        return m.E_d[t] <= m.E_max_d * m.I_dch[t]
    model.eq_g_upper = pyo.Constraint(model.T, rule=eq_g_upper)

    # Battery simultaneous charge/discharge exclusion
    def eq_i(m, t):
        return m.I_dch[t] + m.I_ch[t] <= 1
    model.eq_i = pyo.Constraint(model.T, rule=eq_i)
    
    # Gas input balance
    def eq_j(m, t):
        return m.G[t] == m.G1[t] + m.G2[t]
    model.eq_j = pyo.Constraint(model.T, rule=eq_j)
    
    # Heat demand balance
    def eq_k(m, t):
        return m.eta_gh * m.G1[t] + m.H1[t] + m.H_EHP[t] == m.Dh[t]
    model.eq_k = pyo.Constraint(model.T, rule=eq_k)
    
    # Furnace heat generation split
    def eq_l(m, t):
        return m.eta_ghf * m.G2[t] == m.H1[t] + m.H2[t]
    model.eq_l = pyo.Constraint(model.T, rule=eq_l)
    
    # Cooling demand balance
    def eq_m(m, t):
        return m.eta_hc * m.H2[t] + m.C_EHP[t] == m.Dc[t]
    model.eq_m = pyo.Constraint(model.T, rule=eq_m)
    
    # EHP balance
    def eq_n(m, t):
        return m.E_3[t] * m.COP == m.H_EHP[t] + m.C_EHP[t]
    model.eq_n = pyo.Constraint(model.T, rule=eq_n)
    
    # EHP heat lower limit
    def eq_o_lower(m, t):
        return m.H_EHP_min * m.I_h[t] <= m.H_EHP[t]
    model.eq_o_lower = pyo.Constraint(model.T, rule=eq_o_lower)
    
    # EHP heat upper limit
    def eq_o_upper(m, t):
        return m.H_EHP[t] <= m.H_EHP_max * m.I_h[t]
    model.eq_o_upper = pyo.Constraint(model.T, rule=eq_o_upper)
    
    # EHP cooling lower limit
    def eq_p_lower(m, t):
        return m.C_EHP_min * m.I_c[t] <= m.C_EHP[t]
    model.eq_p_lower = pyo.Constraint(model.T, rule=eq_p_lower)
    
    # EHP cooling upper limit
    def eq_p_upper(m, t):
        return m.C_EHP[t] <= m.C_EHP_max * m.I_c[t]
    model.eq_p_upper = pyo.Constraint(model.T, rule=eq_p_upper)
    
    # EHP just heating or cooling
    def eq_q(m, t):
        return m.I_c[t] + m.I_h[t] <= 1
    model.eq_q = pyo.Constraint(model.T, rule=eq_q)
    
    # CHP gas capacity limit
    model.limit_G1 = pyo.Constraint(model.T, rule=lambda m, t: m.G1[t] <= m.Chpmax)
    
    # Furnace gas capacity limit
    model.limit_G2 = pyo.Constraint(model.T, rule=lambda m, t: m.G2[t] <= m.Fmax)
    
    # Chiller heat input capacity limit
    model.limit_H2 = pyo.Constraint(model.T, rule=lambda m, t: m.H2[t] <= m.CBmax)   
    
    return model

def EH3_EHP_stc_model(time_periods, SCENARIOS, PROB, p, data, lambda_today, E_RES):
    # ─── MODEL ────────────────────────────────────────────────────────
    model = pyo.ConcreteModel(name="Full_Energy_Hub_Stc")
    # DUAL
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # ─── SETS ────────────────────────────────────────────────────────────────────
    model.T = pyo.Set(initialize=time_periods)
    model.S = pyo.Set(initialize=SCENARIOS)
    
    # ─── PARAMETERS ───────────────────────────────────────────────────────
    # --- Scalar ---
    # efficiencies
    model.eta_c     = pyo.Param(initialize=p["eta_c"]) #ESS
    model.eta_d     = pyo.Param(initialize=p["eta_d"]) #ESS
    model.eta_ee    = pyo.Param(initialize=p["eta_ee"]) #Tr
    model.eta_ge    = pyo.Param(initialize=p["eta_ge"]) #CHP
    model.eta_gh    = pyo.Param(initialize=p["eta_gh"]) #CHP
    model.eta_hc    = pyo.Param(initialize=p["eta_hc"]) #CB
    model.eta_ghf   = pyo.Param(initialize=p["eta_ghf"]) #F
    
    # ESS
    model.E_min_c  = pyo.Param(initialize=p["E_min_c"])     
    model.E_max_c  = pyo.Param(initialize=p["E_max_c"])     
    model.E_min_d  = pyo.Param(initialize=p["E_min_d"])     
    model.E_max_d  = pyo.Param(initialize=p["E_max_d"])     
    model.SOC_min  = pyo.Param(initialize=p["SOC_min"])     
    model.SOC_max  = pyo.Param(initialize=p["SOC_max"])     
    model.SOC_ini  = pyo.Param(initialize=p["SOC_ini"])     
    
    # Device Capacities
    model.Chpmax   = pyo.Param(initialize=p["Chpmax"])
    model.Fmax     = pyo.Param(initialize=p["Fmax"])
    model.CBmax    = pyo.Param(initialize=p["CBmax"])
    
    # New Heat Pump (EHP) and COP parameters
    model.COP       = pyo.Param(initialize=p["COP"])
    model.C_EHP_min = pyo.Param(initialize=p["C_EHP_min"])
    model.C_EHP_max = pyo.Param(initialize=p["C_EHP_max"])
    model.H_EHP_min = pyo.Param(initialize=p["H_EHP_min"])
    model.H_EHP_max = pyo.Param(initialize=p["H_EHP_max"])
    
    # --- Indexed Parameters ---
    model.Dh        = pyo.Param(model.T, initialize={t: data["Dh"][t] for t in time_periods})
    model.De        = pyo.Param(model.T, initialize={t: data["De"][t] for t in time_periods})
    model.Dc        = pyo.Param(model.T, initialize={t: data["Dc"][t] for t in time_periods})
    
    model.lam_DA    = pyo.Param(model.T, initialize={t: data["lambda_DA"][t] for t in time_periods})
    model.lam_g     = pyo.Param(model.T, initialize={t: data["lambda_g"][t] for t in time_periods}) 
    
    # Stochastic Parameters (indexed by T and S)
    model.E_RES     = pyo.Param(model.T, model.S, initialize=E_RES)
    model.lam_TODAY = pyo.Param(model.T, model.S, initialize=lambda_today)
    
    # ─── VARIABLES ───────────────────────────────────────────────────────────────
    # FIRST-STAGE (Independent of scenarios)
    model.E_DA    = pyo.Var(model.T, domain=pyo.NonNegativeReals)  
    model.G       = pyo.Var(model.T, domain=pyo.NonNegativeReals)  

    # SECOND-STAGE (Dependent on scenarios)
    # Electricity
    model.E_TODAY    = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.E_RES_used = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.E_2        = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    
    # ESS
    model.E_c     = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.E_d     = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.SOC     = pyo.Var(model.T, model.S, domain=pyo.Reals, bounds=(model.SOC_min, model.SOC_max))  
    
    # binary ESS
    model.I_ch    = pyo.Var(model.T, model.S, domain=pyo.Binary)  
    model.I_dch   = pyo.Var(model.T, model.S, domain=pyo.Binary)  
        
    # Gas split
    model.G1      = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.G2      = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    
    # Furnace & Chiller
    model.H1      = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    model.H2      = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)  
    
    # EHP      
    model.E_3     = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)   #NEW (EHP)
    model.H_EHP   = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals) 
    model.C_EHP   = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals) 
    
    # binary EHP
    model.I_h     = pyo.Var(model.T, model.S, domain=pyo.Binary)  
    model.I_c     = pyo.Var(model.T, model.S, domain=pyo.Binary)  

    
    # ─── OBJECTIVE FUNCTION ──────────────────────────────────────────────────────
    # min cost = sum_t [ lambda_DA * E_DA + lambda_g * G + sum_s (PROB * lambda_TODAY * E_TODAY) ]
    def obj_rule(m):
        return sum(
            m.lam_DA[t] * m.E_DA[t] + m.lam_g[t] * m.G[t] + 
            sum(PROB[s] * m.lam_TODAY[t, s] * m.E_TODAY[t, s] + 
                eps * (m.E_RES[t,s] - m.E_RES_used[t,s]) for s in m.S)
            for t in m.T
        )
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # ─── CONSTRAINTS ─────────────────────────────────────────────────────────────
    
    # (b) Electricity supply balance: RES + DA + TODAY = Ec + E2
    def eq_b(m, t, s):
        return m.E_RES_used[t, s] + m.E_DA[t] + m.E_TODAY[t, s] == m.E_2[t, s] + m.E_c[t, s]
    model.eq_b = pyo.Constraint(model.T, model.S, rule=eq_b)
    
    # (b.2) Curtailment
    def eq_curt(m, t, s):
        return m.E_RES_used[t,s] <= m.E_RES[t,s]
    model.eq_curt = pyo.Constraint(model.T, model.S, rule=eq_curt)
    
    # (c) Electricity demand satisfaction
    def eq_c(m, t, s):
        return m.eta_ee * m.E_2[t, s] + m.E_d[t, s] + m.eta_ge * m.G1[t, s] == m.E_3[t, s] + m.De[t]
    model.eq_c = pyo.Constraint(model.T, model.S, rule=eq_c)
    
    # (d) State of Charge (SOC) tracking
    def eq_d(m, t, s):
        if t == m.T.first():
            return m.SOC[t, s] == m.SOC_ini + m.E_c[t, s] * m.eta_c - m.E_d[t, s] / m.eta_d
        else:
            return m.SOC[t, s] == m.SOC[t-1, s] + m.E_c[t, s] * m.eta_c - m.E_d[t, s] / m.eta_d
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
        return m.E_min_d * m.I_dch[t, s] <= m.E_d[t, s]
    model.eq_g_lower = pyo.Constraint(model.T, model.S, rule=eq_g_lower)
    
    # Battery discharging upper limit
    def eq_g_upper(m, t, s):
        return m.E_d[t, s] <= m.E_max_d * m.I_dch[t, s]
    model.eq_g_upper = pyo.Constraint(model.T, model.S, rule=eq_g_upper)

    # Battery simultaneous charge/discharge exclusion
    def eq_i(m, t, s):
        return m.I_dch[t, s] + m.I_ch[t, s] <= 1
    model.eq_i = pyo.Constraint(model.T, model.S, rule=eq_i)
    
    # Gas input balance (G is 1st stage, G1 and G2 are 2nd stage)
    def eq_j(m, t, s):
        return m.G[t] == m.G1[t, s] + m.G2[t, s]
    model.eq_j = pyo.Constraint(model.T, model.S, rule=eq_j)
    
    # Heat demand balance
    def eq_k(m, t, s):
        return m.eta_gh * m.G1[t, s] + m.H1[t, s] + m.H_EHP[t, s] == m.Dh[t]
    model.eq_k = pyo.Constraint(model.T, model.S, rule=eq_k)
    
    # Furnace heat generation split
    def eq_l(m, t, s):
        return m.eta_ghf * m.G2[t, s] == m.H1[t, s] + m.H2[t, s]
    model.eq_l = pyo.Constraint(model.T, model.S, rule=eq_l)
    
    # Cooling demand balance
    def eq_m(m, t, s):
        return m.eta_hc * m.H2[t, s] + m.C_EHP[t, s] == m.Dc[t]
    model.eq_m = pyo.Constraint(model.T, model.S, rule=eq_m)
    
    # EHP balance
    def eq_n(m, t, s):
        return m.E_3[t, s] * m.COP == m.H_EHP[t, s] + m.C_EHP[t, s]
    model.eq_n = pyo.Constraint(model.T, model.S, rule=eq_n)
    
    # EHP heat lower limit
    def eq_o_lower(m, t, s):
        return m.H_EHP_min * m.I_h[t, s] <= m.H_EHP[t, s]
    model.eq_o_lower = pyo.Constraint(model.T, model.S, rule=eq_o_lower)
    
    # EHP heat upper limit
    def eq_o_upper(m, t, s):
        return m.H_EHP[t, s] <= m.H_EHP_max * m.I_h[t, s]
    model.eq_o_upper = pyo.Constraint(model.T, model.S, rule=eq_o_upper)
    
    # EHP cooling lower limit
    def eq_p_lower(m, t, s):
        return m.C_EHP_min * m.I_c[t, s] <= m.C_EHP[t, s]
    model.eq_p_lower = pyo.Constraint(model.T, model.S, rule=eq_p_lower)
    
    # EHP cooling upper limit
    def eq_p_upper(m, t, s):
        return m.C_EHP[t, s] <= m.C_EHP_max * m.I_c[t, s]
    model.eq_p_upper = pyo.Constraint(model.T, model.S, rule=eq_p_upper)
    
    # EHP just heating or cooling
    def eq_q(m, t, s):
        return m.I_c[t, s] + m.I_h[t, s] <= 1
    model.eq_q = pyo.Constraint(model.T, model.S, rule=eq_q)
    
    # CHP gas capacity limit
    model.limit_G1 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.G1[t, s] <= m.Chpmax)
    
    # Furnace gas capacity limit
    model.limit_G2 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.G2[t, s] <= m.Fmax)
    
    # Chiller heat input capacity limit
    model.limit_H2 = pyo.Constraint(model.T, model.S, rule=lambda m, t, s: m.H2[t, s] <= m.CBmax)   
    
    return model


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
    
    # EHP cooling lower limit
    def eq_p_lower(m, t, s):
        return m.C_EHP_min * m.I_c[t,s] <= m.C_EHP[t,s]
    model.eq_p_lower = pyo.Constraint(model.T, model.S, rule=eq_p_lower)
    
    # EHP cooling upper limit
    def eq_p_upper(m, t, s):
        return m.C_EHP[t,s] <= m.C_EHP_max * m.I_c[t,s]
    model.eq_p_upper = pyo.Constraint(model.T, model.S, rule=eq_p_upper)
    
    # EHP just heating or cooling
    def eq_q(m, t, s):
        return m.I_c[t,s] + m.I_h[t,s] <= 1
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

    model.lam_DA = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_DA"][(study_day, t)] for t in time_periods}
    )

    model.lam_g = pyo.Param(
        model.T,
        initialize={t: DATA["Precio_Gas"][(study_day, t)] for t in time_periods}
    )
    
    model.Dh = pyo.Param(
        model.T,
        initialize={t: DATA["DH"][(study_day, t)] for t in time_periods}
    )

    model.Dc = pyo.Param(
        model.T,
        initialize={t: DATA["DC"][(study_day, t)] for t in time_periods}
    )

    # ---------- Stochastic parameters (scenario days) ----------

    model.De = pyo.Param(
        model.T,
        model.S,
        initialize={
            (t, s): DATA["DE"][(scenario_days[s], t)]
            for s in SCENARIOS
            for t in time_periods
        }
    )

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
            + m.De[t, s]
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