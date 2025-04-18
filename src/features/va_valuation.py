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
import math
from src.features.risk_free_rate import r_t_paths
from src.features.risk_free_rate import g3_parameters
from src.features.equity_risk_factor import equity_process
from src.features.mortality_risk_factor import policy_account_value
from src.features.mortality_risk_factor import guaranteed_amount
from src.features.mortality_risk_factor import cash_flow, present_value
from src.features.mortality_risk_factor import survival_probability
from src.features.future_contracts import fut_value, p_factor
from src.features.future_contracts import gain_loss_present

def va_value(num_simulations, dynamics, time_steps, x_init, dynamic_risk_free_rate, risk_innovations_risk_free_rate,
            floor_risk_free_rate, lower_bound, risk_innovations_equity, mixed_fund, h_values, h_values_f,
             x, risk_innovations_mortality, A_0, F_0, epsilon, G_0, ratchet_provision, rate, u_t, n, S_0):

    # Risk-free rate factor
    r_t, x_t, g_t = r_t_paths(num_simulations, x_init, dynamic_risk_free_rate, time_steps, dynamics, risk_innovations_risk_free_rate, floor_risk_free_rate, lower_bound)

    # Equity factors
    returns_equity, ht_equity, returns_mixedfund, ht_mixedfund, _, _ = equity_process(num_simulations, time_steps, dynamics, risk_innovations_equity, mixed_fund, h_values, h_values_f, r_t, x_t, g_t)

    # Mortality risk
    A_t, F_t = policy_account_value(returns_mixedfund, mixed_fund, A_0, F_0, epsilon)
    G_t = guaranteed_amount(G_0, A_t, ratchet_provision, rate)
    CF_t, u_t, a_t = cash_flow(x, A_t, G_t, risk_innovations_mortality, u_t, mixed_fund)
    pi_0, avg_pi, CF_t = present_value(r_t, CF_t)

    # Future contracts for hedging
    deltas_discounted, s, p_values = gain_loss_present(n, returns_equity, x_t, S_0, r_t)

    # State space
    list_deltas = [x_t[:-1,-1000:,:], ht_equity[:-1,-1000:,:], np.expand_dims(ht_mixedfund[:-1,-1000:],axis=-1), np.expand_dims(A_t[:-1,-1000:],axis=-1), np.expand_dims(F_t[:-1,-1000:],axis=-1), np.expand_dims(G_t[:-1,-1000:],axis=-1), np.expand_dims(u_t[:-1,-1000:],axis=-1), s[:-1,-1000:,:], p_values[:,-1000:,:], np.expand_dims(a_t[:-1,-1000:],axis=-1)]
    list_X = [x_t[:-1,:,:],returns_equity,np.expand_dims(returns_mixedfund, axis=-1),np.expand_dims(u_t[:-1,:], axis=-1),np.expand_dims(A_t[:-1,:], axis=-1),np.expand_dims(G_t[:-1,:], axis=-1)]
    X_t = np.concatenate(list_X, axis=2)
    deltas_input = np.concatenate(list_deltas, axis=2)

    return X_t, deltas_discounted, deltas_input, pi_0, avg_pi, CF_t[:,-1000:]

def va_value_variation(num_simulations, dynamics, time_steps, x_init, dynamic_risk_free_rate, risk_innovations_risk_free_rate,
            floor_risk_free_rate, lower_bound, risk_innovations_equity, mixed_fund, h_values, h_values_f,
             x, risk_innovations_mortality, A_0, F_0, epsilon, G_0, ratchet_provision, rate, u_t):

    # Risk-free rate factor
    r_t, x_t, g_t = r_t_paths(num_simulations, x_init, dynamic_risk_free_rate, time_steps, dynamics, risk_innovations_risk_free_rate, floor_risk_free_rate, lower_bound)

    # Equity factors
    _, _, returns_mixedfund, _, _, _ = equity_process(num_simulations, time_steps, dynamics, risk_innovations_equity, mixed_fund, h_values, h_values_f, r_t, x_t, g_t)

    # Mortality risk
    A_t, _ = policy_account_value(returns_mixedfund, mixed_fund, A_0, F_0, epsilon)
    G_t = guaranteed_amount(G_0, A_t, ratchet_provision, rate)
    CF_t, _ = cash_flow(x, A_t, G_t, risk_innovations_mortality, u_t, mixed_fund)
    _, avg_pi = present_value(r_t, CF_t)

    epsilon = 0.01

    A_t, _ = policy_account_value(returns_mixedfund, mixed_fund, A_0, F_0, epsilon)
    G_t = guaranteed_amount(G_0, A_t, ratchet_provision, rate)
    CF_t, _ = cash_flow(x, A_t, G_t, risk_innovations_mortality, u_t, mixed_fund)
    _, avg_pi_epsilon = present_value(r_t, CF_t)
    Delta = (avg_pi_epsilon-avg_pi)/epsilon

    return Delta

def delta_d(num_simulations, dynamics, time_steps, x_init, risk_innovations_risk_free_rate, dynamic_risk_free_rate, floor_risk_free_rate, lower_bound,
            mixed_fund, h_values, h_values_f, risk_innovations_equity, x, A_0, F_0, G_0, ratchet_provision, rate, u_t, risk_innovations_mortality, a_t_d):

    param_risk_g3 = g3_parameters(dynamics)
    x_t_g3 = np.zeros([time_steps+1,num_simulations,3])

    # Define the mean vector and covariance matrix
    correlation_matrix = np.array([[1, 0.135, -0.787], [0.135, 1, -0.539], [-0.787, -0.539, 1]])
    # Compute the Cholesky decomposition (L is the lower triangular matrix)
    L = np.linalg.cholesky(correlation_matrix)
    # Get the upper triangular matrix L^T
    L_T = L.T
    gaussian_simul = np.random.multivariate_normal([0,0,0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], time_steps*num_simulations)
    gaussian_simul = gaussian_simul.reshape([time_steps,num_simulations,3])
    gaussian_factor = gaussian_simul.copy()
    gaussian_factor[:,:,1] =  L_T[0,1]*gaussian_simul[:,:,0] + L_T[1,1]*gaussian_simul[:,:,1]
    gaussian_factor[:,:,2] =  L_T[0,2]*gaussian_simul[:,:,0] + L_T[1,2]*gaussian_simul[:,:,1] + L_T[2,2]*gaussian_simul[:,:,1]

    if dynamic_risk_free_rate == True:
        x_t_g3[0,:,:] = np.tile(x_init, (num_simulations, 1))
        for t in range(time_steps):
            # kappa:0, mu:1, sigma:2
            x_t_g3[t+1,:,:] = x_t_g3[t,:,:] + param_risk_g3[:,0]*(param_risk_g3[:,1]-x_t_g3[t,:,:]) + risk_innovations_risk_free_rate*gaussian_factor[t,:,:]*param_risk_g3[:,2]
    else:
        x_t_g3 = x_t_g3
    r_t = np.maximum(x_t_g3.sum(axis=2),lower_bound)

    del gaussian_simul, L

    # Equity

    #Parameters of the G3 model
    # kappa:0, mu:1, sigma:2, lambda: 3
    param_risk_g3 = g3_parameters('Q')
    gamma = np.array([[1, 0.135, -0.787], [0.135, 1, -0.539], [-0.787, -0.539, 1]])  # Covariance matrix

    # Parameters of two equity funds:
    # lambda:0, omega:1, alpha:2, gamma:3, beta:4
    parameters_equity = np.array([[0.08477, -1.0132, -0.01083, 0.29438, 0.84031], [0.12810, -1.5390, -0.16422, 0.28580, 0.75939]])
    # define the mean vector and covariance matrix
    corr = 0.76384  # Covariance matrix
    # array to store the returns
    returns = np.zeros([time_steps,num_simulations,2])
    h_t = np.zeros([time_steps+1,num_simulations,2])
    h_t[0,:,:] = np.tile(h_values, (num_simulations, 1))
    # generate samples
    gaussian_sim = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], time_steps*num_simulations)*risk_innovations_equity
    gaussian_sim = gaussian_sim.reshape([time_steps,num_simulations,2])
    gaussian_factor_e = gaussian_sim.copy()
    gaussian_factor_e[:,:,1] = corr*gaussian_sim[:,:,0] + np.sqrt((1-corr**2))*gaussian_sim[:,:,1]
    delta = 1/12

    del gaussian_sim

    # Parameters of the mixed fund
    # parameters of the mized fund
    thetas_mv = np.array([[0.00258, -6.80690, -1.50116, -3.86662],[0.00020, 0.39095, -0.32789, -0.99664]])
    thetas_sv = np.array([[0.06202, 0.04657],[0.49773, 0.05913]])
    # omega:0, alpha:1, gamma:2, beta:3
    params_rv = np.array([[-0.29261, -0.25294, 0.19228, 0.97454],[-0.51325, -0.06924, -0.20790, 0.94287]])
    # array to store the returns
    returns_f = np.zeros([time_steps,num_simulations])
    h_t_f = np.zeros([time_steps+1,num_simulations])
    # generate samples
    gaussian_factor_f = np.random.normal(loc=0.0, scale=1.0, size=time_steps*num_simulations)*risk_innovations_equity
    gaussian_factor_f = gaussian_factor_f.reshape([time_steps,num_simulations])

    if mixed_fund==True:
        thetas_m = thetas_mv[1,:]
        thetas_s = thetas_sv[1,:]
        # omega:0, alpha:1, gamma:2, beta:3
        params_r = params_rv[1,:]
        h_t_f[0,:] = h_values_f[1]
    else:
        thetas_m = thetas_mv[0,:]
        thetas_s = thetas_sv[0,:]
        # omega:0, alpha:1, gamma:2, beta:3
        params_r = params_rv[0,:]
        h_t_f[0,:] = h_values_f[0]

    if dynamics == 'P':
        #Simulate processes under the physical probability measure
        for t in range(time_steps):
            #Simulation of the two equity funds
            returns[t,:,:] = np.tile(r_t[t,:]*delta, (2, 1)).T + parameters_equity[:,0]*np.sqrt(h_t[t,:,:]) - (1/2)*h_t[t,:,:] + np.sqrt(h_t[t,:,:])*gaussian_factor_e[t,:,:]
            element_1 = parameters_equity[:,1]+parameters_equity[:,2]*gaussian_factor_e[t,:,:]
            element_2 = parameters_equity[:,3]*(np.abs(gaussian_factor_e[t,:,:])-2/np.sqrt(2*np.pi))
            element_3 = parameters_equity[:,4]*np.log(h_t[t,:,:])
            h_t[t+1,:,:] = np.exp(element_1 + element_2 + element_3)
            #simulation of mixed fund
            element_1 = r_t[t,:]*delta + thetas_m[0]
            element_2 = ((x_t_g3[t+1,:,:]-(1-param_risk_g3[:,0])*x_t_g3[t,:,:])*thetas_m[1:]).sum(axis=1)
            element_3 = (returns[t,:,:]*thetas_s).sum(axis=1)
            element_4 = np.sqrt(h_t_f[t,:])*gaussian_factor_f[t,:]
            returns_f[t,:] = element_1+element_2+element_3+element_4
            element_1 = params_r[0] + params_r[1]*gaussian_factor_f[t,:]
            element_2 = params_r[2]*(np.abs(gaussian_factor_f[t,:])-2/np.sqrt(2*np.pi))
            element_3 = params_r[3]*np.log(h_t_f[t,:])
            h_t_f[t+1,:] = np.maximum(0.0000020833, np.exp(element_1+element_2+element_3))

    else:

    #Simulate processes under the physical probability measure
        for t in range(time_steps):
            #Simulation of the two equity funds
            returns[t,:] = np.tile(r_t[t,:]*delta, (2, 1)).T - (1/2)*h_t[t,:,:] + np.sqrt(h_t[t,:,:])*gaussian_factor_e[t,:,:]
            element_1 = parameters_equity[:,1]+parameters_equity[:,2]*(gaussian_factor_e[t,:,:]-parameters_equity[:,0])
            element_2 = parameters_equity[:,3]*(np.abs(gaussian_factor_e[t,:,:]-parameters_equity[:,0])-2/np.sqrt(2*np.pi))
            element_3 = parameters_equity[:,4]*np.log(h_t[t,:,:])
            h_t[t+1,:,:] = np.exp(element_1 + element_2 + element_3)
            
            #Simulation of mixed fund
            #computation of sigma_F
            sigma_F = 0
            for i in range(3):
                for i_prime in range(3):
                    sigma_F += thetas_m[i+1]*thetas_m[i_prime+1]*param_risk_g3[i,2]*param_risk_g3[i_prime,2]*gamma[i,i_prime]
            sigma_F += (thetas_s*thetas_s*h_t[t,:,:]).sum(axis=1)
            sigma_F += 2*thetas_s[0]*thetas_s[1]*0.76384*np.sqrt(h_t[t,:,0]*h_t[t,:,1]) + h_t_f[t,:]
            sigma_F = np.sqrt(sigma_F)

            #computation of epsilon
            epsilon_f = (param_risk_g3[:,2]*thetas_m[1:]*gaussian_factor[t,:,:]).sum(axis=1) + (thetas_s*np.sqrt(h_t[t,:,:])*gaussian_factor_e[t,:,:]).sum(axis=1) + np.sqrt(h_t_f[t,:])*gaussian_factor_f[t,:]
            epsilon_f = epsilon_f/sigma_F
            #computation of phi
            phi = thetas_m[0] + (thetas_m[1:]*param_risk_g3[:,0]*param_risk_g3[:,1]).sum() + (thetas_s*(np.tile(r_t[t,:]*delta, (2, 1)).T-(1/2)*h_t[t,:,:])).sum(axis=1)
            #computation of lambda_f
            lambda_f = (1/np.sqrt(h_t_f[t,:]))*(phi+(1/2)*(sigma_F**2))
            #computation of the return
            returns_f[t,:] = r_t[t,:]*delta - (1/2)*(sigma_F**2) + (sigma_F)*epsilon_f
            #h of the fund
            element_1 = params_r[0] + params_r[1]*(gaussian_factor_f[t,:]-lambda_f)
            element_2 = params_r[2]*(np.abs(gaussian_factor_f[t,:]-lambda_f)-2/np.sqrt(2*np.pi))
            element_3 = params_r[3]*np.log(h_t_f[t,:])
            h_t_f[t+1,:] = np.maximum(0.0000020833, np.exp(element_1+element_2+element_3))

    # policy_account_value
            
    #Periodic fee
    omega = 0.0286 if mixed_fund==True else 0.0206
    omega = 1-(1-omega)**(1/12)

    # Fund value
    # Create a new row full of 1s
    new_row = np.ones((1, returns_f.shape[1]))

    # Insert the new row at the desired position (e.g., at the beginning, index 0)
    accumulation_factor = np.insert(np.exp(returns_f), 0, new_row, axis=0)

    fund_value = F_0*np.cumprod(accumulation_factor,axis=0)
    

    values = list()

    for epsilon in [None,0.01]:

        A = np.zeros([returns_f.shape[0]+1,returns_f.shape[1]])
        A[0,:] = A_0 if epsilon==None else (A_0+epsilon)

        account_value_factor = (fund_value[1:,:]/fund_value[:-1,:])*(1-omega)

        for t in range(returns_f.shape[0]):
            A[t+1,:] = A[t,:]*account_value_factor[t,:]

        # guaranteed_amount
        G_t = np.zeros([A.shape[0],A.shape[1]])
        G_t[0,:] = G_0

        if ratchet_provision == True:
            zeta = np.zeros([A.shape[1]])
            for t in range(A.shape[0]-1):
                if (t+1)%12 == 0:
                    zeta = np.zeros([A.shape[1]])
                indicator = ((A[t+1,:]/G_t[t,:])>=(1+rate))*((A.shape[0]-1-120)>t)*(zeta<1)
                zeta += indicator
                G_t[t+1,:] = A[t+1,:]*indicator + G_t[t,:]*(1-indicator)
        else:
            G_t[:,:] = G_0

        # lapse_rate
        m_t = A/G_t
        gamma_1 = 0.02
        gamma_2 = 0.10
        delta_1 = 0.4434
        delta_2 = 1.7420 
        delta = 1/12
        component_1 = gamma_1 + (gamma_2-gamma_1)*(m_t-delta_1)/(delta_2-delta_1)
        component_2 = np.minimum(gamma_2,component_1)
        annualized_lapse_rate = np.maximum(gamma_1,component_2)
        lapse_rate_t = 1 - (1-annualized_lapse_rate)**delta

        # survival_probability

        a_t = np.zeros([lapse_rate_t.shape[0],lapse_rate_t.shape[1]])
        lambda_l = -np.log(1-lapse_rate_t[:-1,:])
        lambda_m = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])
        number_simulations = lapse_rate_t.shape[1]
        a_t[0,:] = a_t_d

        u_t_1 = np.zeros([lapse_rate_t.shape[0],lapse_rate_t.shape[1]])
        u_t_1[0,:] = u_t
        u_t_2 = u_t
        for t in range(lapse_rate_t.shape[0]-1):
            u_t_2, survival_probability_t = survival_probability(x,t,risk_innovations_mortality,u_t_2,number_simulations)
            u_t_1[t+1,:] = u_t_2
            a_t[t+1,:] = a_t[t,:]*survival_probability_t*(1-lapse_rate_t[t,:])
            lambda_m[t,:] = -np.log(survival_probability_t)

        fee = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])
        for t in range(lapse_rate_t.shape[0]-1):
            fee[t,:] = a_t[t,:]*(omega/(1-omega))*A[t+1,:]

        lapse_penalty = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])

        for t in range(lapse_rate_t.shape[0]-1):
            function_p = max(0,0.07-0.01*math.floor((t)/12))
            q_l = (lambda_l[t,:]/(lambda_m[t,:]+lambda_l[t,:]))*(1-np.exp(-1*(lambda_m[t,:]+lambda_l[t,:])))
            lapse_penalty[t,:] = a_t[t,:]*q_l*A[t+1,:]*function_p

        benefit = np.maximum(0,G_t[-1,:]-A[-1,:])*a_t[-1,:]

        CF = fee + lapse_penalty
        CF[-1,:] += -1*benefit

        delta = 1/12
        pi_0 = ((np.exp(-1*np.cumsum(r_t*delta,axis=0))[:-1,:])*CF).sum(axis=0)
        values.append(pi_0.mean())

    delta = (values[1]-values[0])/0.01

    return delta, values[0]


def cash_flow_simulation(num_simulations, dynamics, time_steps, x_init, risk_innovations_risk_free_rate, dynamic_risk_free_rate, floor_risk_free_rate, lower_bound,
            mixed_fund, h_values, h_values_f, risk_innovations_equity, x, A_0, F_0, G_0, ratchet_provision, rate, u_t, risk_innovations_mortality, day, limit_day, s_t, n):
    
    input = np.zeros([num_simulations,11])

    param_risk_g3 = g3_parameters(dynamics)
    x_t_g3 = np.zeros([time_steps+1,num_simulations,3])

    # Define the mean vector and covariance matrix
    correlation_matrix = np.array([[1, 0.135, -0.787], [0.135, 1, -0.539], [-0.787, -0.539, 1]])
    # Compute the Cholesky decomposition (L is the lower triangular matrix)
    L = np.linalg.cholesky(correlation_matrix)
    # Get the upper triangular matrix L^T
    L_T = L.T
    gaussian_simul = np.random.multivariate_normal([0,0,0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], time_steps*num_simulations)
    gaussian_simul = gaussian_simul.reshape([time_steps,num_simulations,3])
    gaussian_factor = gaussian_simul.copy()
    gaussian_factor[:,:,1] =  L_T[0,1]*gaussian_simul[:,:,0] + L_T[1,1]*gaussian_simul[:,:,1]
    gaussian_factor[:,:,2] =  L_T[0,2]*gaussian_simul[:,:,0] + L_T[1,2]*gaussian_simul[:,:,1] + L_T[2,2]*gaussian_simul[:,:,1]

    if dynamic_risk_free_rate == True:
        x_t_g3[0,:,:] = np.tile(x_init, (num_simulations, 1))
        for t in range(time_steps):
            # kappa:0, mu:1, sigma:2
            x_t_g3[t+1,:,:] = x_t_g3[t,:,:] + param_risk_g3[:,0]*(param_risk_g3[:,1]-x_t_g3[t,:,:]) + risk_innovations_risk_free_rate*gaussian_factor[t,:,:]*param_risk_g3[:,2]
    else:
        x_t_g3 = x_t_g3
    r_t = np.maximum(x_t_g3.sum(axis=2),lower_bound)

    del gaussian_simul, L

    input[:,:3] = x_t_g3[1,:,:]

    # Equity

    #Parameters of the G3 model
    # kappa:0, mu:1, sigma:2, lambda: 3
    param_risk_g3 = g3_parameters('Q')
    gamma = np.array([[1, 0.135, -0.787], [0.135, 1, -0.539], [-0.787, -0.539, 1]])  # Covariance matrix

    # Parameters of two equity funds:
    # lambda:0, omega:1, alpha:2, gamma:3, beta:4
    parameters_equity = np.array([[0.08477, -1.0132, -0.01083, 0.29438, 0.84031], [0.12810, -1.5390, -0.16422, 0.28580, 0.75939]])
    # define the mean vector and covariance matrix
    corr = 0.76384  # Covariance matrix
    # array to store the returns
    returns = np.zeros([time_steps,num_simulations,2])
    h_t = np.zeros([time_steps+1,num_simulations,2])
    h_t[0,:,:] = np.tile(h_values, (num_simulations, 1))
    # generate samples
    gaussian_sim = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], time_steps*num_simulations)*risk_innovations_equity
    gaussian_sim = gaussian_sim.reshape([time_steps,num_simulations,2])
    gaussian_factor_e = gaussian_sim.copy()
    gaussian_factor_e[:,:,1] = corr*gaussian_sim[:,:,0] + np.sqrt((1-corr**2))*gaussian_sim[:,:,1]
    delta = 1/12

    del gaussian_sim

    # Parameters of the mixed fund
    # parameters of the mized fund
    thetas_mv = np.array([[0.00258, -6.80690, -1.50116, -3.86662],[0.00020, 0.39095, -0.32789, -0.99664]])
    thetas_sv = np.array([[0.06202, 0.04657],[0.49773, 0.05913]])
    # omega:0, alpha:1, gamma:2, beta:3
    params_rv = np.array([[-0.29261, -0.25294, 0.19228, 0.97454],[-0.51325, -0.06924, -0.20790, 0.94287]])
    # array to store the returns
    returns_f = np.zeros([time_steps,num_simulations])
    h_t_f = np.zeros([time_steps+1,num_simulations])
    # generate samples
    gaussian_factor_f = np.random.normal(loc=0.0, scale=1.0, size=time_steps*num_simulations)*risk_innovations_equity
    gaussian_factor_f = gaussian_factor_f.reshape([time_steps,num_simulations])

    if mixed_fund==True:
        thetas_m = thetas_mv[1,:]
        thetas_s = thetas_sv[1,:]
        # omega:0, alpha:1, gamma:2, beta:3
        params_r = params_rv[1,:]
        h_t_f[0,:] = h_values_f[1]
    else:
        thetas_m = thetas_mv[0,:]
        thetas_s = thetas_sv[0,:]
        # omega:0, alpha:1, gamma:2, beta:3
        params_r = params_rv[0,:]
        h_t_f[0,:] = h_values_f[0]

    if dynamics == 'P':
        #Simulate processes under the physical probability measure
        for t in range(time_steps):
            #Simulation of the two equity funds
            returns[t,:,:] = np.tile(r_t[t,:]*delta, (2, 1)).T + parameters_equity[:,0]*np.sqrt(h_t[t,:,:]) - (1/2)*h_t[t,:,:] + np.sqrt(h_t[t,:,:])*gaussian_factor_e[t,:,:]
            element_1 = parameters_equity[:,1]+parameters_equity[:,2]*gaussian_factor_e[t,:,:]
            element_2 = parameters_equity[:,3]*(np.abs(gaussian_factor_e[t,:,:])-2/np.sqrt(2*np.pi))
            element_3 = parameters_equity[:,4]*np.log(h_t[t,:,:])
            h_t[t+1,:,:] = np.exp(element_1 + element_2 + element_3)
            #simulation of mixed fund
            element_1 = r_t[t,:]*delta + thetas_m[0]
            element_2 = ((x_t_g3[t+1,:,:]-(1-param_risk_g3[:,0])*x_t_g3[t,:,:])*thetas_m[1:]).sum(axis=1)
            element_3 = (returns[t,:,:]*thetas_s).sum(axis=1)
            element_4 = np.sqrt(h_t_f[t,:])*gaussian_factor_f[t,:]
            returns_f[t,:] = element_1+element_2+element_3+element_4
            element_1 = params_r[0] + params_r[1]*gaussian_factor_f[t,:]
            element_2 = params_r[2]*(np.abs(gaussian_factor_f[t,:])-2/np.sqrt(2*np.pi))
            element_3 = params_r[3]*np.log(h_t_f[t,:])
            h_t_f[t+1,:] = np.maximum(0.0000020833, np.exp(element_1+element_2+element_3))

        input[:,3:5] = h_t[1,:,:]
        input[:,5] = h_t_f[1,:]

        #compute values of equities
        s = np.zeros([returns.shape[0]+1,returns.shape[1],returns.shape[2]])
        for j in range(2):
            # Fund value
            returns_j = returns[:,:,j]
            # Create a new row full of 1s
            new_row = np.ones((1, returns_j.shape[1]))
            # Insert the new row at the desired position (e.g., at the beginning, index 0)
            accumulation_factor = np.insert(np.exp(returns_j), 0, new_row, axis=0)
            s_j = s_t[j]*np.cumprod(accumulation_factor,axis=0)
            s[:,:,j] = s_j
            
        #compute gains and losses
        deltas = np.zeros([returns.shape[0],returns.shape[1],returns.shape[2]])
        p_values = np.zeros([returns.shape[0],returns.shape[1],returns.shape[2]])
        for j in range(2):
            for t in range(returns.shape[0]):
                deltas[t,:,j] = fut_value(t+1, t+n, j, s, x_t_g3) - fut_value(t, t+n, j, s, x_t_g3)
                p_values[t,:,j] = p_factor(t, t+n, x_t_g3[t,:,:])

        #discount_factor = (np.exp(-1*r_t[1,:]*(1/12)))
        deltas[:,:,0] = deltas[:,:,0]
        deltas[:,:,1] = deltas[:,:,1]

    else:

    #Simulate processes under the physical probability measure
        for t in range(time_steps):
            #Simulation of the two equity funds
            returns[t,:] = np.tile(r_t[t,:]*delta, (2, 1)).T - (1/2)*h_t[t,:,:] + np.sqrt(h_t[t,:,:])*gaussian_factor_e[t,:,:]
            element_1 = parameters_equity[:,1]+parameters_equity[:,2]*(gaussian_factor_e[t,:,:]-parameters_equity[:,0])
            element_2 = parameters_equity[:,3]*(np.abs(gaussian_factor_e[t,:,:]-parameters_equity[:,0])-2/np.sqrt(2*np.pi))
            element_3 = parameters_equity[:,4]*np.log(h_t[t,:,:])
            h_t[t+1,:,:] = np.exp(element_1 + element_2 + element_3)
            
            #Simulation of mixed fund
            #computation of sigma_F
            sigma_F = 0
            for i in range(3):
                for i_prime in range(3):
                    sigma_F += thetas_m[i+1]*thetas_m[i_prime+1]*param_risk_g3[i,2]*param_risk_g3[i_prime,2]*gamma[i,i_prime]
            sigma_F += (thetas_s*thetas_s*h_t[t,:,:]).sum(axis=1)
            sigma_F += 2*thetas_s[0]*thetas_s[1]*0.76384*np.sqrt(h_t[t,:,0]*h_t[t,:,1]) + h_t_f[t,:]
            sigma_F = np.sqrt(sigma_F)

            #computation of epsilon
            epsilon_f = (param_risk_g3[:,2]*thetas_m[1:]*gaussian_factor[t,:,:]).sum(axis=1) + (thetas_s*np.sqrt(h_t[t,:,:])*gaussian_factor_e[t,:,:]).sum(axis=1) + np.sqrt(h_t_f[t,:])*gaussian_factor_f[t,:]
            epsilon_f = epsilon_f/sigma_F
            #computation of phi
            phi = thetas_m[0] + (thetas_m[1:]*param_risk_g3[:,0]*param_risk_g3[:,1]).sum() + (thetas_s*(np.tile(r_t[t,:]*delta, (2, 1)).T-(1/2)*h_t[t,:,:])).sum(axis=1)
            #computation of lambda_f
            lambda_f = (1/np.sqrt(h_t_f[t,:]))*(phi+(1/2)*(sigma_F**2))
            #computation of the return
            returns_f[t,:] = r_t[t,:]*delta - (1/2)*(sigma_F**2) + (sigma_F)*epsilon_f
            #h of the fund
            element_1 = params_r[0] + params_r[1]*(gaussian_factor_f[t,:]-lambda_f)
            element_2 = params_r[2]*(np.abs(gaussian_factor_f[t,:]-lambda_f)-2/np.sqrt(2*np.pi))
            element_3 = params_r[3]*np.log(h_t_f[t,:])
            h_t_f[t+1,:] = np.maximum(0.0000020833, np.exp(element_1+element_2+element_3))

    # policy_account_value
            
    #Periodic fee
    omega = 0.0286 if mixed_fund==True else 0.0206
    omega = 1-(1-omega)**(1/12)

    # Fund value
    # Create a new row full of 1s
    new_row = np.ones((1, returns_f.shape[1]))

    # Insert the new row at the desired position (e.g., at the beginning, index 0)
    accumulation_factor = np.insert(np.exp(returns_f), 0, new_row, axis=0)

    fund_value = F_0*np.cumprod(accumulation_factor,axis=0)


    for epsilon in [None]:

        A = np.zeros([returns_f.shape[0]+1,returns_f.shape[1]])
        A[0,:] = A_0 if epsilon==None else (A_0+epsilon)

        account_value_factor = (fund_value[1:,:]/fund_value[:-1,:])*(1-omega)

        for t in range(returns_f.shape[0]):
            A[t+1,:] = A[t,:]*account_value_factor[t,:]

        # guaranteed_amount
        G_t = np.zeros([A.shape[0],A.shape[1]])
        G_t[0,:] = G_0

        if ratchet_provision == True:
            zeta = np.zeros([A.shape[1]])
            for t in range(A.shape[0]-1):
                if (t+1)%12 == 0:
                    zeta = np.zeros([A.shape[1]])
                indicator = ((A[t+1,:]/G_t[t,:])>=(1+rate))*((A.shape[0]-1-120)>t)*(zeta<1)
                zeta += indicator
                G_t[t+1,:] = A[t+1,:]*indicator + G_t[t,:]*(1-indicator)
        else:
            G_t[:,:] = G_0

        input[:,6] = fund_value[1,:]
        input[:,7] = A[1,:]
        input[:,8] = G_t[1,:]

        # lapse_rate
        m_t = A/G_t
        gamma_1 = 0.02
        gamma_2 = 0.10
        delta_1 = 0.4434
        delta_2 = 1.7420 
        delta = 1/12
        component_1 = gamma_1 + (gamma_2-gamma_1)*(m_t-delta_1)/(delta_2-delta_1)
        component_2 = np.minimum(gamma_2,component_1)
        annualized_lapse_rate = np.maximum(gamma_1,component_2)
        lapse_rate_t = 1 - (1-annualized_lapse_rate)**delta

        # survival_probability

        a_t = np.zeros([lapse_rate_t.shape[0],lapse_rate_t.shape[1]])
        lambda_l = -np.log(1-lapse_rate_t[:-1,:])
        lambda_m = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])
        number_simulations = lapse_rate_t.shape[1]
        a_t[0,:] = 1

        u_t_1 = np.zeros([lapse_rate_t.shape[0],lapse_rate_t.shape[1]])
        u_t_1[0,:] = u_t
        u_t_2 = u_t
        for t in range(lapse_rate_t.shape[0]-1):
            u_t_2, survival_probability_t = survival_probability(x,t,risk_innovations_mortality,u_t_2,number_simulations)
            u_t_1[t+1,:] = u_t_2
            a_t[t+1,:] = a_t[t,:]*survival_probability_t*(1-lapse_rate_t[t,:])
            lambda_m[t,:] = -np.log(survival_probability_t)

        input[:,9] = u_t_1[1,:]
        input[:,10] = day

        fee = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])
        for t in range(lapse_rate_t.shape[0]-1):
            fee[t,:] = a_t[t,:]*(omega/(1-omega))*A[t+1,:]

        lapse_penalty = np.zeros([lapse_rate_t.shape[0]-1,lapse_rate_t.shape[1]])

        for t in range(lapse_rate_t.shape[0]-1):
            function_p = max(0,0.07-0.01*math.floor((t)/12))
            q_l = (lambda_l[t,:]/(lambda_m[t,:]+lambda_l[t,:]))*(1-np.exp(-1*(lambda_m[t,:]+lambda_l[t,:])))
            lapse_penalty[t,:] = a_t[t,:]*q_l*A[t+1,:]*function_p

        benefit = np.maximum(0,G_t[-1,:]-A[-1,:])*a_t[-1,:]

        CF = fee + lapse_penalty

        if day == limit_day:
            CF[-1,:] += -1*benefit

        CF = CF

    return CF, deltas, input


###### comparison

def va_value_comparison(num_simulations, dynamics, time_steps, x_init, dynamic_risk_free_rate, risk_innovations_risk_free_rate,
            floor_risk_free_rate, lower_bound, risk_innovations_equity, mixed_fund, h_values, h_values_f,
             x, risk_innovations_mortality, A_0, F_0, epsilon, G_0, ratchet_provision, rate, u_t, n, S_0):

    # Risk-free rate factor
    r_t, x_t, g_t = r_t_paths(num_simulations, x_init, dynamic_risk_free_rate, time_steps, dynamics, risk_innovations_risk_free_rate, floor_risk_free_rate, lower_bound)

    # Equity factors
    returns_equity, ht_equity, returns_mixedfund, ht_mixedfund, _, _ = equity_process(num_simulations, time_steps, dynamics, risk_innovations_equity, mixed_fund, h_values, h_values_f, r_t, x_t, g_t)

    # Mortality risk
    A_t, F_t = policy_account_value(returns_mixedfund, mixed_fund, A_0, F_0, epsilon)
    G_t = guaranteed_amount(G_0, A_t, ratchet_provision, rate)
    CF_t, u_t, a_t = cash_flow(x, A_t, G_t, risk_innovations_mortality, u_t, mixed_fund)
    pi_0, avg_pi, CF_t_1 = present_value(r_t, CF_t)

    # Future contracts for hedging
    deltas_discounted, s, p_values = gain_loss_present(n, returns_equity, x_t, S_0, r_t)

    # State space
    list_comparison = [np.expand_dims(A_t[:,-1000:],axis=-1), np.expand_dims(F_t[:,-1000:],axis=-1)]

    return list_comparison, a_t