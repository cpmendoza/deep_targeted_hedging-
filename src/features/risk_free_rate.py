"""
Usage:
    1. cd src
    2. python3 features/risk_free_rate.py 
"""

import os, sys
from pathlib import Path
import warnings

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import numpy as np
from src.utils import *

def g3_parameters(dynamics):

    """Function that provides parameters of the G3 model

        Parameters
        ----------
        dynamics      : string - 'P' for P dynamics and 'Q' for risk-neutral dynamics

        Returns
        -------
        param_risk_g3 : numpy array - parameters matrix

    """

    # kappa:0, mu:1, sigma:2, lambda: 3
    param_risk_g3 = np.array([[0.00594, 0.01176, 0.00524, 0.918], [0.04228, -0.00043, 0.00482, -5.473], [0.02049, 0.04627, 0.00788, 1.134]])
    param_risk = param_risk_g3.copy()
    if dynamics == 'Q':
        # 0 - kappa
        param_risk[:,0] = param_risk_g3[:,0]-param_risk_g3[:,2]*param_risk_g3[:,3]
        # 1 - mu
        param_risk[:,1] = (param_risk_g3[:,0]*param_risk_g3[:,1])/(param_risk_g3[:,0]-param_risk_g3[:,2]*param_risk_g3[:,3])
        # 2 - sigma
        param_risk[:,2] = param_risk_g3[:,2]

    return param_risk[:,:-1]

def risk_free_rate(x_init, dynamic_risk_free_rate, time_steps, dynamics, risk_innovations, floor_risk_free_rate, lower_bound):

    """Function to simulate one path of the G3 model

        Parameters
        ----------
        x_init                 : list    - List with the three initial values of the path
        dynamic_risk_free_rate : boolean - True for dynamic risk-free rate
        time_steps             : integer - number of time steps for the path simulation
        dynamics               : string  - 'P' for P dynamics and 'Q' for risk-neutral dynamics
        risk_innovations       : boolean - True to include stochastic innovations
        floor_risk_free_rate   : boolean - True to floor the risk-free rate values
        lower_bound            : double  - lower bound to floor risk-free rate

        Returns
        -------
        x_t : numpy array - risk-free rate simulated path

    """

    param_risk_g3 = g3_parameters(dynamics)
    x_t_g3 = np.zeros([time_steps+1,3])

    # Define the mean vector and covariance matrix
    correlation_matrix = np.array([[1, 0.135, -0.787], [0.135, 1, -0.539], [-0.787, -0.539, 1]])
    # Compute the Cholesky decomposition (L is the lower triangular matrix)
    L = np.linalg.cholesky(correlation_matrix)
    # Get the upper triangular matrix L^T
    L_T = L.T
    gaussian_simul = np.random.multivariate_normal([0,0,0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], time_steps)
    gaussian_factor = gaussian_simul.copy()
    gaussian_factor[:,1] =  L_T[0,1]*gaussian_simul[:,0] + L_T[1,1]*gaussian_simul[:,1]
    gaussian_factor[:,2] =  L_T[0,2]*gaussian_simul[:,0] + L_T[1,2]*gaussian_simul[:,1] + L_T[2,2]*gaussian_simul[:,1]

    if dynamic_risk_free_rate == True:

        x_t_g3[0,:] = x_init

        for t in range(time_steps):
            # kappa:0, mu:1, sigma:2
            x_t_g3[t+1,:] = x_t_g3[t,:] + param_risk_g3[:,0]*(param_risk_g3[:,1]-x_t_g3[t,:]) + risk_innovations*gaussian_factor[t,:]*param_risk_g3[:,2]

        x_t = x_t_g3.sum(axis=1)

        if floor_risk_free_rate == True:
            x_t = np.maximum(x_t,lower_bound)

    else:
        x_t = x_t_g3.sum(axis=1)
    
    return x_t, x_t_g3, gaussian_factor


def r_t_paths(num_simulations, x_init, dynamic_risk_free_rate, time_steps, dynamics, risk_innovations, floor_risk_free_rate, lower_bound):

    """Function to simulate multiple paths of the G3 model

        Parameters
        ----------
        num_simulations        : interger - Number of simulated paths
        x_init                 : list     - List with the three initial values of the path
        dynamic_risk_free_rate : boolean  - True for dynamic risk-free rate
        time_steps             : integer  - number of time steps for the path simulation
        dynamics               : string   - 'P' for P dynamics and 'Q' for risk-neutral dynamics
        risk_innovations       : boolean  - True to include stochastic innovations
        floor_risk_free_rate   : boolean  - True to floor the risk-free rate values
        lower_bound            : double   - lower bound to floor risk-free rate

        Returns
        -------
        x_t : numpy array - risk-free rate simulated path

    """

    r_t = np.zeros([time_steps+1,num_simulations])
    x_t = np.zeros([time_steps+1,num_simulations,3])
    g_t = np.zeros([time_steps,num_simulations,3])
    for i in range(num_simulations):
        r_t[:,i], x_t[:,i,:], g_t[:,i,:] = risk_free_rate(x_init, dynamic_risk_free_rate, time_steps, dynamics, risk_innovations, floor_risk_free_rate, lower_bound)
    return r_t, x_t, g_t

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    