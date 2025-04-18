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
from src.utils import *
from src.features.risk_free_rate import g3_parameters

def equity_process_path(h_values, h_values_f, time_steps, dynamics, risk_innovations, mixed_fund, r_t_p, x_t_p, g_t_p):

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
    returns = np.zeros([time_steps,2])
    h_t = np.zeros([time_steps+1,2])
    h_t[0,:] = h_values
    # generate samples
    gaussian_sim = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], time_steps)*risk_innovations
    gaussian_factor = gaussian_sim.copy()
    gaussian_factor[:,1] = corr*gaussian_sim[:,0] + np.sqrt((1-corr**2))*gaussian_sim[:,1]
    delta = 1/12

    # Parameters of the mixed fund
    # parameters of the mized fund
    thetas_mv = np.array([[0.00258, -6.80690, -1.50116, -3.86662],[0.00020, 0.39095, -0.32789, -0.99664]])
    thetas_sv = np.array([[0.06202, 0.04657],[0.49773, 0.05913]])
    # omega:0, alpha:1, gamma:2, beta:3
    params_rv = np.array([[-0.29261, -0.25294, 0.19228, 0.97454],[-0.51325, -0.06924, -0.20790, 0.94287]])
    # array to store the returns
    returns_f = np.zeros([time_steps])
    h_t_f = np.zeros([time_steps+1])
    # generate samples
    gaussian_factor_f = np.random.normal(loc=0.0, scale=1.0, size=time_steps)*risk_innovations

    if mixed_fund==True:
        thetas_m = thetas_mv[1,:]
        thetas_s = thetas_sv[1,:]
        # omega:0, alpha:1, gamma:2, beta:3
        params_r = params_rv[1,:]
        h_t_f[0] = h_values_f[1]
    else:
        thetas_m = thetas_mv[0,:]
        thetas_s = thetas_sv[0,:]
        # omega:0, alpha:1, gamma:2, beta:3
        params_r = params_rv[0,:]
        h_t_f[0] = h_values_f[0]  

    if dynamics == 'P':
        #Simulate processes under the physical probability measure
        for t in range(time_steps):
            #Simulation of the two equity funds
            returns[t,:] = r_t_p[t]*delta + parameters_equity[:,0]*np.sqrt(h_t[t,:]) - (1/2)*h_t[t,:] + np.sqrt(h_t[t,:])*gaussian_factor[t,:]
            element_1 = parameters_equity[:,1]+parameters_equity[:,2]*gaussian_factor[t,:]
            element_2 = parameters_equity[:,3]*(np.abs(gaussian_factor[t,:])-2/np.sqrt(2*np.pi))
            element_3 = parameters_equity[:,4]*np.log(h_t[t,:])
            h_t[t+1,:] = np.exp(element_1 + element_2 + element_3)
            #simulation of mixed fund
            element_1 = r_t_p[t]*delta + thetas_m[0]
            element_2 = ((x_t_p[t+1,:]-(1-param_risk_g3[:,0])*x_t_p[t,:])*thetas_m[1:]).sum()
            element_3 = (returns[t,:]*thetas_s).sum()
            element_4 = np.sqrt(h_t_f[t])*gaussian_factor_f[t]
            returns_f[t] = element_1+element_2+element_3+element_4
            element_1 = params_r[0] + params_r[1]*gaussian_factor_f[t]
            element_2 = params_r[2]*(np.abs(gaussian_factor_f[t])-2/np.sqrt(2*np.pi))
            element_3 = params_r[3]*np.log(h_t_f[t])
            h_t_f[t+1] = max(0.0000020833, np.exp(element_1+element_2+element_3))
    else:
        #Simulate processes under the physical probability measure
        for t in range(time_steps):
            #Simulation of the two equity funds
            returns[t,:] = r_t_p[t]*delta - (1/2)*h_t[t,:] + np.sqrt(h_t[t,:])*gaussian_factor[t,:]
            element_1 = parameters_equity[:,1]+parameters_equity[:,2]*(gaussian_factor[t,:]-parameters_equity[:,0])
            element_2 = parameters_equity[:,3]*(np.abs(gaussian_factor[t,:]-parameters_equity[:,0])-2/np.sqrt(2*np.pi))
            element_3 = parameters_equity[:,4]*np.log(h_t[t,:])
            h_t[t+1,:] = np.exp(element_1 + element_2 + element_3)
            
            #Simulation of mixed fund
            #computation of sigma_F
            sigma_F = 0
            for i in range(3):
                for i_prime in range(3):
                    sigma_F += thetas_m[i+1]*thetas_m[i_prime+1]*param_risk_g3[i,2]*param_risk_g3[i_prime,2]*gamma[i,i_prime]
            sigma_F += (thetas_s*thetas_s*h_t[t,:]).sum()
            sigma_F += 2*thetas_s[0]*thetas_s[1]*0.76384*np.sqrt(h_t[t,0]*h_t[t,1]) + h_t_f[t]
            sigma_F = np.sqrt(sigma_F)
            #computation of epsilon
            epsilon_f = (param_risk_g3[:,2]*thetas_m[1:]*g_t_p[t,:]).sum() + (thetas_s*np.sqrt(h_t[t,:])*gaussian_factor[t,:]).sum() + np.sqrt(h_t_f[t])*gaussian_factor_f[t]
            epsilon_f = epsilon_f/sigma_F
            #computation of phi
            phi = thetas_m[0] + (thetas_m[1:]*param_risk_g3[:,0]*param_risk_g3[:,1]).sum() + (thetas_s*(r_t_p[t]*delta-(1/2)*h_t[t,:])).sum()
            #computation of lambda_f
            lambda_f = (1/np.sqrt(h_t_f[t]))*(phi+(1/2)*(sigma_F**2))
            #computation of the return
            returns_f[t] = r_t_p[t]*delta - (1/2)*(sigma_F**2) + (sigma_F)*epsilon_f
            #h of the fund
            element_1 = params_r[0] + params_r[1]*(gaussian_factor_f[t]-lambda_f)
            element_2 = params_r[2]*(np.abs(gaussian_factor_f[t]-lambda_f)-2/np.sqrt(2*np.pi))
            element_3 = params_r[3]*np.log(h_t_f[t])
            h_t_f[t+1] = max(0.0000020833, np.exp(element_1+element_2+element_3))

    return returns, h_t, returns_f, h_t_f, gaussian_factor, gaussian_factor_f

def equity_process(num_simulations, time_steps, dynamics, risk_innovations, mixed_fund, h_values, h_values_f, r_t, x_t, g_t):

    # arrays to store simulations
    returns_equity = np.zeros([time_steps,num_simulations,2])
    ht_equity = np.zeros([time_steps+1,num_simulations,2])
    returns_mixedfund = np.zeros([time_steps,num_simulations])
    ht_mixedfund = np.zeros([time_steps+1,num_simulations])
    gaussian_factor = np.zeros([time_steps,num_simulations,2])
    gaussian_factor_f = np.zeros([time_steps,num_simulations])

    for i in range(num_simulations):
        r_t_p = r_t[:,i]
        x_t_p = x_t[:,i,:]
        g_t_p = g_t[:,i,:]
        returns_equity[:,i,:], ht_equity[:,i,:], returns_mixedfund[:,i], ht_mixedfund[:,i], gaussian_factor[:,i,:], gaussian_factor_f[:,i]  = equity_process_path(h_values, h_values_f, time_steps, dynamics, risk_innovations, mixed_fund, r_t_p, x_t_p, g_t_p)

    return returns_equity, ht_equity, returns_mixedfund, ht_mixedfund, gaussian_factor, gaussian_factor_f