"""
Usage:
    1. cd src
    2. python3 features/equity_risk_factor.py 
"""

import os, sys
from pathlib import Path
import warnings

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import numpy as np
import pandas as pd
import math
from src.utils import *

from src.features.risk_free_rate import g3_parameters


def vega(n,i,l):
    # Define parameters
    gamma = np.array([[1, 0.135, -0.787], [0.135, 1, -0.539], [-0.787, -0.539, 1]])
    param_risk_g3 = g3_parameters('Q') # kappa:0, mu:1, sigma:2
    kappa_i = param_risk_g3[i,0]
    kappa_l = param_risk_g3[l,0]
    sigma_i = param_risk_g3[i,2]
    sigma_l = param_risk_g3[l,2]
    gamma_il = gamma[i,l]
    # Compute elements
    element_p = ((sigma_i*sigma_l)/(kappa_i*kappa_l))*gamma_il
    element_1 = (1-((1-kappa_i)**n))/kappa_i
    element_2 = (1-((1-kappa_l)**n))/kappa_l
    element_3 = (1-((1-kappa_i)**n)*((1-kappa_l)**n))/ (1-((1-kappa_i))*((1-kappa_l)))
    # Compute final value
    v = element_p*(n - element_1 - element_2 + element_3)
    return v

def m_nti(n, x_t_i, i):
    # Define parameters
    param_risk_g3 = g3_parameters('Q') # kappa:0, mu:1, sigma:2
    kappa_i = param_risk_g3[i,0]
    mu_i = param_risk_g3[i,1]
    # Compute final value
    m = (x_t_i-mu_i)*((1-((1-kappa_i)**n))/kappa_i) + mu_i*n
    return m

def p_factor(t, T, x_t):

    #Compute first element
    element_1 = 0
    for i in range(3):
        element_1 += m_nti(T-t, x_t[:,i], i) 
    element_1 = (1/12)*element_1

    #Compute second element
    element_2 = 0 
    for i in range(3):
        for l in range(3):
            element_2 += vega(T-t,i,l)
    element_2 = (((1/12)**2)/2)*element_2

    # Compute final value
    p = np.exp(element_1 + element_2)

    return p


def fut_value(t, T, j, s, x_t):

    #element_1 
    s_j_t = s[t,:,j]
    p_t_n = p_factor(t, T, x_t[t,:,:])
    fut_t_T = s_j_t*p_t_n

    return fut_t_T

def gain_loss(n, returns_equity, x_t, S_0):

    #compute values of equities
    s = np.zeros([returns_equity.shape[0]+1,returns_equity.shape[1],returns_equity.shape[2]])
    for j in range(2):
        # Fund value
        returns_j = returns_equity[:,:,j]
        # Create a new row full of 1s
        new_row = np.ones((1, returns_j.shape[1]))
        # Insert the new row at the desired position (e.g., at the beginning, index 0)
        accumulation_factor = np.insert(np.exp(returns_j), 0, new_row, axis=0)
        s_j = S_0*np.cumprod(accumulation_factor,axis=0)
        s[:,:,j] = s_j

    #compute gains and losses
    deltas = np.zeros([returns_equity.shape[0],returns_equity.shape[1],returns_equity.shape[2]])
    p_values = np.zeros([returns_equity.shape[0],returns_equity.shape[1],returns_equity.shape[2]])
    for j in range(2):
        for t in range(returns_equity.shape[0]):
            deltas[t,:,j] = fut_value(t+1, t+n, j, s, x_t) - fut_value(t, t+n, j, s, x_t)
            p_values[t,:,j] = p_factor(t, t+n, x_t[t,:,:])

    return deltas, s, p_values

def gain_loss_present(n, returns_equity, x_t, S_0, r_t):
    deltas, s, p_values = gain_loss(n, returns_equity, x_t, S_0)
    discount_factor = (np.exp(-1*np.cumsum(r_t*(1/12),axis=0))[:-1,:])
    deltas[:,:,0] = discount_factor*deltas[:,:,0]
    deltas[:,:,1] = discount_factor*deltas[:,:,1]
    return deltas, s, p_values
