"""
Usage:
    1. cd src
    2. python3 features/cvar_estimation.py 
"""

import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from src.features.va_valuation import cash_flow_simulation
import tensorflow.compat.v1 as tf
from src.models.pricing_agent import DeepAgent
from src.utils import *

class cvar_estimation_(object):
    
    """Class to simulate the market environment
    
    Parameters
    ----------
    config_file : simulation settings for the cvar_estimation 
    
    """
    
    def __init__(self, config_file_simulation, config_file_agent):

        # Parameters
        self.config_file = config_file_simulation['simulation']
        self.mixed_fund = self.config_file['mixed_fund']
        self.ratchet_provision = self.config_file['ratchet_provision']
        self.time_steps = self.config_file['time_steps']

        # Acummulated discount factor
        self.lower_bound = self.config_file['lower_bound']

        self.rate = self.config_file['rate']
        self.x_init = self.config_file['x']
        self.dynamic_risk_free_rate = self.config_file['dynamic_risk_free_rate']
        self.floor_risk_free_rate = self.config_file['floor_risk_free_rate']

        # Restore neaural network
        self.config_file_pricing_agent = config_file_agent['agent']
        self.batch_size    = 1000
        self.nbs_units     = self.config_file_pricing_agent['nbs_units']
        self.lr            = self.config_file_pricing_agent['lr']
        self.dropout_par   = self.config_file_pricing_agent['dropout_par']
        self.epochs        = self.config_file_pricing_agent['epochs']
        self.display_plot  = self.config_file_pricing_agent['display_plot']
        self.nbs_input     = 11


    def estimation(self):
        
        """Function to estimate CVaR

        Parameters
        ----------
        parameters : config_file

        Returns
        -------

        """
        
        #Parameters for Shapley decomposition

        name_1 = "rbc" if self.mixed_fund==False else 'assupmtion'
        name_2 = str(self.time_steps)
        name_3 = "ratchet" if self.ratchet_provision==True else ""
        name = f"{name_1}_{name_2}_{name_3}.npy"

        gl_va = np.load(os.path.join(main_folder,"data/processed",f"gv_{name}"))[-1000:,:]
        hedging_instruments = np.load(os.path.join(main_folder,"data/processed",f"hi_{name}"))[:,-1000:,:,:]
        deltas_input = np.load(os.path.join(main_folder,"data/processed",f"de_{name}"))
        position = np.load(os.path.join(main_folder,"data/processed",f"po_{name}"))
        price = np.load(os.path.join(main_folder,"data/processed",f"pa_{name}"))

        # Acummulated discount factor
        r_t = np.maximum(deltas_input[:,:,:3,0].sum(axis=2), self.lower_bound)
        discount_factor = (np.exp(-1*np.cumsum(r_t*(1/12), axis=0)))
        discount_factor = np.vstack((np.ones((1, discount_factor.shape[1])), discount_factor))

        #General parameters
        risk_innovations_risk_free_rate, risk_innovations_equity, risk_innovations_mortality = [True, True, True]
        x = self.x_init*12

        # CVaR for the whole position (no shap decomposition)
        shap = 0

        # Restore neaural network
        os.chdir(os.path.join(main_folder, f"models"))
        train_input   = None
        train_output  = None

        name_1 = f"Pricing_agent_FFNN_dropout_{str(int(self.dropout_par*100))}_pricing_va_{name_1}_{name_2}_{name_3}"

        # Compile the neural network
        rl_network = DeepAgent(self.batch_size, self.nbs_input, self.nbs_units, self.lr, self.dropout_par, name_1)

        # Start cvar estimation
        cvar = np.zeros([240,1000,2])
        index = 0

        print("-- CVaR estimation starts --")
        print("-- Progress of CVaR estimation: ", end='', flush=True)

        with tf.Session() as sess:
            rl_network.restore(sess, f"{name_1}.ckpt")

            for fac in [1,-1]:

                # Gain & loss per time step including hedging
                cash_flows_hedging_portfolio_delta = ((hedging_instruments*fac*np.load(os.path.join(main_folder,"data/processed",f"po_{name}"))).sum(axis=2))[:,:,0]
                cash_flows_va = np.load(os.path.join(main_folder,"data/processed",f"cf_{name}"))[:,:,0]
                va_price = np.vstack((np.load(os.path.join(main_folder,"data/processed",f"pa_{name}"))[:,:,0], np.zeros((1, discount_factor.shape[1]))))
                va_price = va_price * discount_factor
                cash_flows = va_price[1:,:] - va_price[:-1,:] + cash_flows_va + cash_flows_hedging_portfolio_delta
                gl = cash_flows.cumsum(axis=0)
                gl = np.vstack((np.zeros((1, gl.shape[1])),gl))

                for sim in range(1000):
                    for time in range(240):
                        print(f"\r Factor: {fac} - Simulation:{sim} - Time:{time}", end='', flush=True)
                        # General parameters
                        # Parameters of the function for risk free rate
                        x_init_d = deltas_input[time,sim,:3,shap]

                        # Parameters of the function for equity
                        h_values_d = deltas_input[time,sim,3:5,shap]
                        h_values_f_d = np.array([deltas_input[time,sim,5,shap],deltas_input[time,sim,5,shap]])
                        # Parameters of the function for mortality risk
                        x_d = x + time 
                        A_0_d = deltas_input[time,sim,6,shap]
                        F_0_d = deltas_input[time,sim,7,shap]
                        G_0_d = deltas_input[time,sim,8,shap]
                        u_t_d = deltas_input[time,sim,9,shap]
                        s_t   = deltas_input[time,sim,10:12,shap]
                        n = 3 # Number of periods for future contract

                        day = time
                        limit_day = self.time_steps-1

                        #The simulation of the followinf provides cash flow of va, gain of the hedging instruments (all of then computed with present value at time t - no discounted values at time zero)
                        #We simulate time t+1 with time t data, and then bring time t+1 back to time t
                        CF, increments_hedging, input = cash_flow_simulation(1000, 'P', 1, x_init_d, risk_innovations_risk_free_rate, self.dynamic_risk_free_rate, self.floor_risk_free_rate, self.lower_bound, 
                                                                            self.mixed_fund, h_values_d, h_values_f_d, risk_innovations_equity, x_d, A_0_d, F_0_d, G_0_d, self.ratchet_provision, self.rate, u_t_d, risk_innovations_mortality,
                                                                            day, limit_day, s_t, n)

                        test_input    = input
                        test_output   = np.zeros([1000,1])

                        #Compute estimated va price with neural network
                        predicted = rl_network.predict(train_input, train_output, test_input, test_output, sess)
                        predicted = predicted.reshape(-1)

                        #Compute final cash flow at time t
                        hedging_gains = (increments_hedging[0,:,:]*fac*position[time,sim,:,0]).sum(axis=1)
                        preliminary_cash_flow = CF[0,:] + hedging_gains + predicted 

                        #Compute present value of the cash_flow at time zero
                        cash_flow_present_value = discount_factor[time+1,sim]*preliminary_cash_flow - discount_factor[time,sim]*price[time,sim,0]
                        
                        current_gl = cash_flow_present_value #+ gl[time,sim]
                        hedging_err = -1*current_gl
                        cvar[time,sim,index] = np.mean(np.sort(hedging_err)[int(0.95*hedging_err.shape[0]):])
                
                index += 1

        print("\n--Storing CVaR estimation--")
        np.save(os.path.join(main_folder,"data/processed",f"cvar_estimation_{name}"),cvar)
        print("\n-- CVaR estimation stored --") 

        return
        
    
if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file_agent = load_config(os.path.join(main_folder,'cfgs','config_pricing_agent.yml'))
    config_file_simulation = load_config(os.path.join(main_folder,'cfgs','config_market_simulation.yml'))
    estimation_ = cvar_estimation_(config_file_simulation,config_file_agent)
    _ = estimation_.estimation()