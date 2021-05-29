# Simplified and slightly modularized from:
# https://github.com/LeonardoL87/SARS-CoV-2-Model-with-and-without-temperature-dependence

import pickle
import datetime

import math as m
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from typing import Callable
from functools import partial
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from lmfit import minimize, Parameters, Parameter, report_fit, Model, Minimizer

N_POPS = {'Barcelona': 2.227e6,
          'Catalunya': 7780479,
          'Lombardia': 10060574,
          'Thuringen': 2392000
          }

dt = 1 / 24


def T_inv(T, date_T, ini: int):
    T = pd.concat([T, T])
    T_hat = savgol_filter(T, 51, 3)  # window size 51, polynomial order 3

    pos = date_T[ini].day + 30 * date_T[ini].month - 30
    T_inv = 1 - T / np.mean(T) + 1.5
    T_inv = T_inv[pos:len(T_inv) - 1]

    T_inv_hat = 1 - T_hat / np.mean(T_hat) + 1.5
    T_inv_hat = T_inv_hat[pos:len(T_inv_hat) - 1]

    t = np.arange(len(T_inv))

    T_inv = T_inv_hat

    return [t, T_inv]


def SEIQRDP(t, y, ps, beta_func: Callable, total_pop: int):
    """
    Differential equation system. Can be formulated from the ODE system. Returns a column vector
    with the system at time t
    """
    delta = ps['delta'].value
    alpha = ps['alpha0'].value * m.exp(-ps['alpha1'].value * t)
    betQ = beta_func(ps['betaQ'].value, t)
    bet = beta_func(ps['beta'].value, t)
    gamma = ps['gamma'].value
    Lambda = ps['lambda0'].value * (1. - m.exp(-ps['lambda1'].value * t))
    kappa = ps['kappa0'].value * m.exp(-ps['kappa1'].value * t)
    tau = ps['tau0'].value * (1. - m.exp(-ps['tau1'].value * t))
    rho = 0
    N = total_pop
    S, E, I, Q, R, D, P, V = np.clip(y, a_min=0, a_max=None)
    beta = (1 / N) * (bet * I + betQ * Q)
    e_frac = (1.0 - m.exp(-beta * dt))  # |exposed prob
    i_frac = (1.0 - m.exp(-gamma * dt))  # |infection prob
    r_frac = (1.0 - m.exp(-Lambda * dt))  # |recovered prob
    p_frac = (1.0 - m.exp(-alpha * dt))  # |protected prob
    d_frac = (1.0 - m.exp(-kappa * dt))  # |death prob
    rel_frac = (1.0 - m.exp(-tau * dt))  # |release prob
    rep_frac = (1.0 - m.exp(-delta * dt))  # |detected prob
    vac_frac = (1.0 - m.exp(-rho * dt))  # |vaccinated prob

    exposed = np.random.binomial(S, e_frac)
    protected = np.random.binomial(S, p_frac)
    infection = np.random.binomial(E, i_frac)
    detected = np.random.binomial(I, rep_frac)
    recovery = np.random.binomial(Q, r_frac)
    deaths = np.random.binomial(Q, d_frac)

    released = np.random.binomial(P, rel_frac)
    vaccinated = np.random.binomial(S, vac_frac)

    S = S - exposed - protected + released - vaccinated  # | Susceptible
    E = E + exposed - infection  # | Exposed
    I = I + infection - detected  # | Infected
    Q = Q + detected - recovery - deaths  # | Detected
    R = R + recovery  # | Recovered
    D = D + deaths  # | Deaths
    P = P + protected - released  # | Protected
    V = V + vaccinated  # | Total Cases
    return [S, E, I, Q, R, D, P, V]


def simulate(t, u, ps, beta_func: Callable, total_pop: int):
    """Returns a matrix with the dynamic of each population"""
    S = np.zeros(len(t))
    E = np.zeros(len(t))
    I = np.zeros(len(t))
    Q = np.zeros(len(t))
    R = np.zeros(len(t))
    D = np.zeros(len(t))
    P = np.zeros(len(t))
    Y = np.zeros(len(t))
    for j in range(len(t)):
        u = SEIQRDP(t[j], u, ps, beta_func=beta_func, total_pop=total_pop)
        S[j], E[j], I[j], Q[j], R[j], D[j], P[j], Y[j] = u
    return {'t': t, 'S': S, 'E': E, 'I': I, 'Q': Q, 'R': R, 'D': D, 'P': P, 'V': Y}


def interpolation(y, t, ti):
    """Single interpolation of 1 vector"""
    f = interpolate.interp1d(t, y, kind='nearest')
    f2 = f(ti)
    return f2


def sys_interpolation(Y, t, ti):
    col = Y.columns
    datcol = col[1:len(col)]
    y_interp = {}
    y_interp['t'] = ti
    for i in datcol:
        yi = Y[str(i)].to_numpy()
        f2 = interpolation(yi, t, ti)
        y_interp[str(i)] = f2

    return y_interp


def COVID_SEIRC(y, t, ps, beta_func: Callable, total_pop: int):
    """Definition of the model in deterministic way for fitting purposes"""

    alpha0 = ps['alpha0'].value
    alpha1 = ps['alpha1'].value
    bet = ps['beta'].value
    betQ = ps['betaQ'].value
    gam = ps['gamma'].value
    delt = ps['delta'].value
    lambda0 = ps['lambda0'].value
    lambda1 = ps['lambda1'].value
    kappa0 = ps['kappa0'].value
    kappa1 = ps['kappa1'].value
    tau0 = ps['tau0'].value
    tau1 = ps['tau1'].value

    rho = 0
    S, E, I, Q, R, D, C, V = y

    alpha = lambda x: alpha0 * m.exp(-alpha1 * x)
    beta = partial(beta_func, b=bet)
    betaQ = partial(beta_func, b=betQ)

    gamma = gam
    delta = delt
    Lambda = lambda x: lambda0 * (1. - m.exp(-lambda1 * x))
    kappa = lambda x: kappa0 * m.exp(-kappa1 * x)

    tau = lambda x: tau0 * (1. - m.exp(-tau1 * x))

    BETA = (beta(t) * I + betaQ(t) * Q) * 1 / total_pop
    #    ___________equations___________________________________
    dS = tau(t) * C - alpha(t) * S - BETA * S - rho * S
    dE = -gamma * E + BETA * S
    dI = gamma * E - delta * I
    dQ = delta * I - Lambda(t) * Q - kappa(t) * Q
    dR = Lambda(t) * Q
    dD = kappa(t) * Q
    dC = alpha(t) * S - tau(t) * C
    dV = rho * S
    return dS, dE, dI, dQ, dR, dD, dC, dV


def run_model(beta_type: 'str', location: str, params_path, n_iter: int = 100, seed: int = 23,
              keep_top: int = None):

    keep_top = n_iter if keep_top is None else keep_top
    np.random.seed(seed)
    with open(params_path, 'rb') as fh:
        ini, time, active, confirmed, recovered, deaths, _, T, date_T, params_set = pickle.load(fh)

    tb, beta = T_inv(T, date_T, ini=ini)
    total_pop = N_POPS[location]
    f = interp1d(tb, beta, kind='cubic')

    beta_terms = {'constant': lambda b, x: b,
                  'seasonal': lambda b, x: b * (1 + np.sin(2 * np.pi * x * 1 / 360)),
                  'temperature': lambda b, x: b * f(x)
                  }

    beta_func = beta_terms[beta_type]
    E0 = active[ini]
    I0 = active[ini]
    Q0 = active[ini]
    R0 = recovered[ini]
    D0 = deaths[ini]

    P0 = 0
    V0 = 0
    S0 = total_pop - E0 - I0 - Q0 - R0 - D0 - P0 - V0

    outputs = list()
    squared_errors = []
    for i in tqdm(np.arange(n_iter), desc='Iterations', leave=False):
        # ===============SETTING THE ORIGINAL SET OF PARAMETERS===========================
        dt = 1 / 24
        y0 = [S0, E0, I0, Q0, R0, D0, P0, V0]

        tf = len(time)
        tl = int(tf / dt)
        t = np.linspace(0, tf - 1, tl)
        paropt = params_set[i]
        params_set.append(paropt)

        sir_out = pd.DataFrame(simulate(t, y0, paropt, beta_func, total_pop=total_pop))

        ti = np.linspace(t[0], t[len(t) - 1], int(t[len(t) - 1] - t[0]) + 1)
        sir_out = pd.DataFrame(sys_interpolation(sir_out, t, ti))
        squared_error = (sir_out[['Q', 'R', 'D']].sum(axis=1) - confirmed).pow(2).sum()
        squared_errors.append(squared_error)
        outputs.append(sir_out)
    squared_errors = np.array(squared_errors)
    best_fits = np.argsort(squared_errors)[:keep_top]
    results = np.array(outputs)[best_fits]
    vector = results.mean(axis=0)[:, :-1]
    std = results.std(axis=0)[:, :-1]
    t, s, e, i, q, r, d, p = std.transpose()
    t, S, E, I, Q, R, D, P = vector.transpose()
    TC = Q + R + D
    tc = q + r + d

    results_df = (pd.concat([pd.DataFrame(vector),
                             pd.Series(TC),
                             pd.DataFrame(std),
                             pd.Series(tc)],
                            axis=1))

    results_df.columns = ['t', 'S', 'E', 'I', 'Q', 'R', 'D', 'P', 'Total Cases', 'std t', 'std S',
       'std E', 'std I', 'std Q', 'std R', 'std D', 'std P', 'std Total Cases']

    results_df = (results_df
                  .drop(columns=['std t', 't'])
                  .assign(N=total_pop)
                  .assign(date=time.reset_index(drop=True))
                  )

    return results_df
