"""
Usage:
    1. cd src
    2. python3 features/market_simulator.py 
"""

import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from src.features.va_valuation import va_value, delta_d
from src.utils import *

class market_simulator_class(object):
    
    """Class to simulate the market environment
    
    Parameters
    ----------
    config_file : simulation settings for the JIVR model and the underlying asset and 
    
    """
    
    def __init__(self,config_file):

        # Parameters
        self.config_file = config_file['simulation']

        # General parameters
        self.num_simulations = self.config_file['num_simulations']
        self.dynamics = self.config_file['dynamics']
        self.time_steps = self.config_file['time_steps']

        # Parameters of the function for risk free rate
        self.x_init = self.config_file['x_init']
        self.dynamic_risk_free_rate = self.config_file['dynamic_risk_free_rate']
        self.floor_risk_free_rate = self.config_file['floor_risk_free_rate']
        self.lower_bound = self.config_file['lower_bound']

        # Parameters of the function for equity
        self.mixed_fund = self.config_file['mixed_fund']
        self.h_values = np.array([eval(x) for x in self.config_file['h_values']]) # initial values two equity
        self.h_values_f = np.array([eval(x) for x in self.config_file['h_values_f']]) # initial values mixed fund

        # Parameters of the function for mortality risk
        self.x = self.config_file['x']*12 #age in months
        self.A_0 = self.config_file['A_0']
        self.F_0 = self.config_file['F_0']
        self.epsilon = None
        self.G_0 = self.config_file['G_0']
        self.ratchet_provision = self.config_file['ratchet_provision']
        self.rate = self.config_file['rate']
        self.u_t = self.config_file['u_t']

        # Parameters for future contracts
        self.n = self.config_file['n']
        self.S_0 = self.config_file['S_0']

        self.seed = self.config_file['seed'] #Seed to ensure replicability 

    def simulation(self):
        
        """Function to simulate the whole market environment

        Parameters
        ----------
        parameters : config_file

        Returns
        -------

        """
        
        #Parameters for Shapley decomposition
        combinations = [[True, True, True], [True, True, False], [True, False, True], [True, False, False], [False, True, True], [False, True, False], [False, False, True], [False, False, False]]
        gl_tot_list = list()
        X_list = list()
        deltas_list = list()
        deltas_input_list = list()
        cashflows = list()

        np.random.seed(self.seed)

        print("-- Market simulation starts --")
        print("-- Progress of market simulation: ", end='', flush=True)
        #Computation of Shapley decomposition
        i_index = 0
        for risk_innovations_risk_free_rate, risk_innovations_equity, risk_innovations_mortality in combinations:
            #Cash flows under the physical measure 
            X_t, deltas_discounted, deltas_input, pi_0, _, CF_t = va_value(self.num_simulations, self.dynamics, self.time_steps, self.x_init, self.dynamic_risk_free_rate, risk_innovations_risk_free_rate,
                        self.floor_risk_free_rate, self.lower_bound, risk_innovations_equity, self.mixed_fund, self.h_values, self.h_values_f,
                        self.x, risk_innovations_mortality, self.A_0, self.F_0, self.epsilon, self.G_0, self.ratchet_provision, self.rate, self.u_t, self.n, self.S_0)

            #VA pracice under the risk-neutral measure
            *_, avg_pi, _ = va_value(1000, 'Q', self.time_steps, self.x_init, self.dynamic_risk_free_rate, risk_innovations_risk_free_rate,
                        self.floor_risk_free_rate, self.lower_bound, risk_innovations_equity, self.mixed_fund, self.h_values, self.h_values_f,
                        self.x, risk_innovations_mortality, self.A_0, self.F_0, self.epsilon, self.G_0, self.ratchet_provision, self.rate, self.u_t, self.n, self.S_0)

            gl_tot_list.append(pi_0 - avg_pi)
            X_list.append(X_t)
            deltas_list.append(deltas_discounted)
            deltas_input_list.append(deltas_input)
            cashflows.append(CF_t)
            print("\r--Progress of market simulation: {:.2%}".format((i_index+1)/8), end='', flush=True)
            i_index+=1
        
        print("\n--Storing market features--")
        del X_t, deltas_discounted, pi_0, deltas_input, CF_t

        # Shapley decomposition 
        time = gl_tot_list[7]
        rate = 0.333333333*(gl_tot_list[3]-gl_tot_list[7])+0.166666667*((gl_tot_list[1]-gl_tot_list[5])+(gl_tot_list[2]-gl_tot_list[6]))+0.333333333*(gl_tot_list[0]-gl_tot_list[4])
        equity = 0.333333333*(gl_tot_list[5]-gl_tot_list[7])+0.166666667*((gl_tot_list[1]-gl_tot_list[3])+(gl_tot_list[4]-gl_tot_list[6]))+0.333333333*(gl_tot_list[0]-gl_tot_list[2])
        mortality = 0.333333333*(gl_tot_list[6]-gl_tot_list[7])+0.166666667*((gl_tot_list[2]-gl_tot_list[3])+(gl_tot_list[4]-gl_tot_list[5]))+0.333333333*(gl_tot_list[0]-gl_tot_list[1])
        total = time + rate + equity + mortality

        del time, rate , mortality, total 

        # Name of the file
        name_1 = "rbc" if self.mixed_fund==False else 'assupmtion'
        name_2 = str(self.time_steps)
        name_3 = "ratchet" if self.ratchet_provision==True else ""
        name = f"{name_1}_{name_2}_{name_3}.npy"

        # Stock information together

        hedging_instruments = np.stack(deltas_list, axis=-1)
        np.save(os.path.join(main_folder,"data/processed",f"hi_{name}"),hedging_instruments)
        del deltas_list, hedging_instruments
        print("\n-- Hedging instruments array stored --")

        gl_va               = np.stack(gl_tot_list, axis=-1)
        np.save(os.path.join(main_folder,"data/processed",f"gv_{name}"),gl_va)
        del gl_va, gl_tot_list
        print("\n-- Gains and losses array stored --")

        deltas_input        = np.stack(deltas_input_list, axis=-1)
        np.save(os.path.join(main_folder,"data/processed",f"de_{name}"),deltas_input)
        del deltas_input_list, deltas_input
        print("\n-- Deltas input array stored --")

        cash_flow           = np.stack(cashflows, axis=-1)
        np.save(os.path.join(main_folder,"data/processed",f"cf_{name}"),cash_flow)
        del cashflows, cash_flow 
        print("\n-- Cash flow array stored --") 

        # Store simulated environment
        np.save(os.path.join(main_folder,"data/processed",f"et_{name}"),equity)

        input               = np.stack(X_list, axis=-1)
        np.save(os.path.join(main_folder,"data/processed",f"xt_{name}"),input)
        del X_list, input
        print("\n-- Input array stored --")
        
        print("\n--Simulation of simulation features completed--")

        return 
        
    def delta_features(self):
    
        """Function that computes delta features assuming that the market environment is already simulated 

        Parameters
        ----------
        parameters : config_file

        Returns
        -------

        """
        # Name of the file
        name_1 = "rbc" if self.mixed_fund==False else 'assupmtion'
        name_2 = str(self.time_steps)
        name_3 = "ratchet" if self.ratchet_provision==True else ""
        name = f"{name_1}_{name_2}_{name_3}.npy"

        # Define paths
        gv_path = os.path.join(main_folder, "data/processed", f"gv_{name}")
        hi_path = os.path.join(main_folder, "data/processed", f"hi_{name}")
        de_path = os.path.join(main_folder, "data/processed", f"de_{name}")

        # Check if all files exist
        if not all(os.path.exists(path) for path in [gv_path, hi_path, de_path]):
            raise FileNotFoundError("One or more required files are missing. Please simulate the environment first.")

        # Load data
        deltas_input = np.load(de_path)
        np.random.seed(self.seed)

        # Computation of delta features
        #General parameters
        combinations = [[True, True, True], [True, True, False], [True, False, True], [True, False, False], [False, True, True], [False, True, False], [False, False, True], [False, False, False]]

        print("-- Delta hedging computation starts --")
        print("-- Progress of features computation: ")

        deltas_a = np.zeros([self.time_steps,1000,8])
        price =  np.zeros([self.time_steps,1000,8])
        for shap in range(8):
            risk_innovations_risk_free_rate, risk_innovations_equity, risk_innovations_mortality = combinations[shap]
            for sim in range(1000): 
                for time in range(self.time_steps):
        
                    # General parameters
                    time_steps_d = self.time_steps - time
                    # Parameters of the function for risk free rate
                    x_init_d = deltas_input[time,sim,:3,shap]
                    # Parameters of the function for equity
                    h_values_d = deltas_input[time,sim,3:5,shap]
                    h_values_f_d = np.array([deltas_input[time,sim,5,shap],deltas_input[time,sim,5,shap]])
                    # Parameters of the function for mortality risk
                    x_d = self.x + time 
                    A_0_d = deltas_input[time,sim,6,shap]
                    F_0_d = deltas_input[time,sim,7,shap]
                    G_0_d = deltas_input[time,sim,8,shap]
                    u_t_d = deltas_input[time,sim,9,shap]
                    a_t_d = deltas_input[time,sim,-1,shap]

                    deltas_a[time,sim,shap], price[time,sim,shap] = delta_d(1000, 'Q', time_steps_d, x_init_d, risk_innovations_risk_free_rate, self.dynamic_risk_free_rate, self.floor_risk_free_rate, self.lower_bound, 
                                                            self.mixed_fund, h_values_d, h_values_f_d, risk_innovations_equity, x_d, A_0_d, F_0_d, G_0_d, self.ratchet_provision, self.rate, u_t_d, risk_innovations_mortality, a_t_d)

                    print(f"\r Shap:{shap+1}-Simulation:{sim+1}-Time:{time}", end='', flush=True)

        print("\n--Storing delta hedging features--")

        np.save(os.path.join(main_folder,"data/processed",f"da_{name}"),deltas_a)
        np.save(os.path.join(main_folder,"data/processed",f"pa_{name}"),price)

        print("--Delta hedging features computation completed--")
        
        return 
    
    def delta_hedging(self):
    
        """Function that computes delta hedging strategy 

        Parameters
        ----------
        parameters : config_file

        Returns
        -------

        """
        # Name of the file
        name_1 = "rbc" if self.mixed_fund==False else 'assupmtion'
        name_2 = str(self.time_steps)
        name_3 = "ratchet" if self.ratchet_provision==True else ""
        name = f"{name_1}_{name_2}_{name_3}.npy"

        # Define paths
        deltas_a_path = os.path.join(main_folder,"data/processed",f"da_{name}")
        price_path = os.path.join(main_folder,"data/processed",f"pa_{name}")
        de_path = os.path.join(main_folder, "data/processed", f"de_{name}")

        # Check if all files exist
        if not all(os.path.exists(path) for path in [deltas_a_path, price_path, de_path]):
            raise FileNotFoundError("One or more required files are missing. Please simulate the environment and compute delta features first.")

        # Load data
        deltas_a = np.load(deltas_a_path)
        price = np.load(price_path)
        deltas_input = np.load(de_path)

        #General parameters 
        thetas_sv = np.array([[0.06202, 0.04657],[0.49773, 0.05913]])
        if self.mixed_fund==True:
            thetas_s = thetas_sv[1,:]
        else:
            thetas_s = thetas_sv[0,:]

        # Compute positions
        print("-- Positions computation starts --")
        positions = np.zeros([self.time_steps,1000,2,8])
        for index in range(2):
            positions[:,:,index,:] = -1*(deltas_a[:,:,:]*(deltas_input[:,:,6,:]*thetas_s[index]))/((deltas_input[:,:,10:12,:][:,:,index,:])*(deltas_input[:,:,12:14,:][:,:,index,:]))

        # Store position
        np.save(os.path.join(main_folder,"data/processed",f"po_{name}"),positions)

        print("\n--Positions computation completed and stored--")

        return
        
    
if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_market_simulation.yml'))
    market_model = market_simulator_class(config_file)
    market_model.simulation()
    market_model.delta_features()
    market_model.delta_hedging()