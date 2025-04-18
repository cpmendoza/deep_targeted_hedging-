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

def policy_account_value(returns_mixedfund, mixed_fund, A_0, F_0, epsilon):

    #Periodic fee
    omega = 0.0286 if mixed_fund==True else 0.0206
    omega = 1-(1-omega)**(1/12)

    # Fund value
    # Create a new row full of 1s
    new_row = np.ones((1, returns_mixedfund.shape[1]))

    # Insert the new row at the desired position (e.g., at the beginning, index 0)
    accumulation_factor = np.insert(np.exp(returns_mixedfund), 0, new_row, axis=0)
    fund_value = F_0*np.cumprod(accumulation_factor,axis=0)
    A = np.zeros([returns_mixedfund.shape[0]+1,returns_mixedfund.shape[1]])
    A[0,:] = A_0 if epsilon==None else (A_0+epsilon)

    account_value_factor = (fund_value[1:,:]/fund_value[:-1,:])*(1-omega)

    for t in range(returns_mixedfund.shape[0]):
        A[t+1,:] = A[t,:]*account_value_factor[t,:]

    return A, fund_value


def guaranteed_amount(G_0, A_t, ratchet_provision, rate):

    G_t = np.zeros([A_t.shape[0],A_t.shape[1]])
    G_t[0,:] = G_0

    if ratchet_provision == True:
        zeta = np.zeros([A_t.shape[1]])
        for t in range(A_t.shape[0]-1):
            if (t+1)%12 == 0:
                zeta = np.zeros([A_t.shape[1]])
            indicator = ((A_t[t+1,:]/G_t[t,:])>=(1+rate))*((A_t.shape[0]-1-120)>t)*(zeta<1)
            zeta += indicator
            G_t[t+1,:] = A_t[t+1,:]*indicator + G_t[t,:]*(1-indicator)
    else:
        G_t[:,:] = G_0

    return G_t

def lapse_rate(A_t,G_t):
    
    #Parameters
    m_t = A_t/G_t
    gamma_1 = 0.02
    gamma_2 = 0.10
    delta_1 = 0.4434
    delta_2 = 1.7420 
    delta = 1/12
    component_1 = gamma_1 + (gamma_2-gamma_1)*(m_t-delta_1)/(delta_2-delta_1)
    component_2 = np.minimum(gamma_2,component_1)
    annualized_lapse_rate = np.maximum(gamma_1,component_2)
    lapse_rate_t = 1 - (1-annualized_lapse_rate)**delta
    return lapse_rate_t

def survival_probability(x,t,risk_innovations,u_t,number_simulations):

    #Parameters
    beta_1 = 0.1856 #anual
    beta_2 = -4.050*(10**(-3))
    beta_3 = 2.769*(10**(-5))
    file_path = os.path.join(main_folder,'data/raw/estimated_series.csv')
    a_b = pd.read_csv(file_path).to_numpy()
    c = -1.8325
    sigma_v = np.sqrt(2.1957)
    delta = 1/12
    y_x = math.floor((x+t)/12)
    standard_deviation = beta_1 + beta_2*y_x + beta_3*(y_x**2)
    epsilon = np.random.normal(loc=0, scale=standard_deviation, size=number_simulations)*risk_innovations
    u_t_1 = c + u_t + np.random.normal(loc=0, scale=sigma_v, size=number_simulations)*risk_innovations

    #survival probability
    component_1 = np.exp(a_b[y_x,0] + a_b[y_x,1]*u_t_1 + epsilon)
    component_2 = np.minimum(1,component_1)
    survival_probability_t = (1-component_2)**delta

    return u_t_1, survival_probability_t

def cash_flow(x, A_t, G_t, risk_innovations, u_t, mixed_fund):

    # Parameters
    omega = 0.0286 if mixed_fund==True else 0.0206
    omega = 1-(1-omega)**(1/12)

    lapse_rate_t = lapse_rate(A_t,G_t)
    a_t = np.zeros([lapse_rate_t.shape[0],lapse_rate_t.shape[1]])
    lambda_l = -np.log(1-lapse_rate_t[:-1,:])
    lambda_m = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])
    number_simulations = lapse_rate_t.shape[1]
    a_t[0,:] = 1

    u_t_1 = np.zeros([lapse_rate_t.shape[0],lapse_rate_t.shape[1]])
    u_t_1[0,:] = u_t

    for t in range(lapse_rate_t.shape[0]-1):
        u_t, survival_probability_t = survival_probability(x,t,risk_innovations,u_t,number_simulations)
        u_t_1[t+1,:] = u_t
        a_t[t+1,:] = a_t[t,:]*survival_probability_t*(1-lapse_rate_t[t,:])
        lambda_m[t,:] = -np.log(survival_probability_t)

    fee = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])
    for t in range(lapse_rate_t.shape[0]-1):
        fee[t,:] = a_t[t,:]*(omega/(1-omega))*A_t[t+1,:]

    lapse_penalty = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])

    for t in range(lapse_rate_t.shape[0]-1):
        function_p = max(0,0.07-0.01*math.floor((t)/12))
        q_l = (lambda_l[t,:]/(lambda_m[t,:]+lambda_l[t,:]))*(1-np.exp(-1*(lambda_m[t,:]+lambda_l[t,:])))
        lapse_penalty[t,:] = a_t[t,:]*q_l*A_t[t+1,:]*function_p

    benefit = np.maximum(0,G_t[-1,:]-A_t[-1,:])*a_t[-1,:]

    CF = fee + lapse_penalty
    CF[-1,:] += -1*benefit

    return CF, u_t_1, a_t

def present_value(r_t, CF_t):
    delta = 1/12
    pi_0 = ((np.exp(-1*np.cumsum(r_t*delta,axis=0))[:-1,:])*CF_t).sum(axis=0)
    avg_pi = pi_0.mean()
    return pi_0, avg_pi, ((np.exp(-1*np.cumsum(r_t*delta,axis=0))[:-1,:])*CF_t)
