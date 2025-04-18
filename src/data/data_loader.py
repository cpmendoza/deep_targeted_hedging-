"""
Usage:
    1. cd src
    2. python data/data_loader.py
"""

import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os, sys
import numpy as np
import math

from pathlib import Path

def load_standard_datasets(name):
    
    """Function that loads the sets to create the training set

        Parameters
        ----------
        maturity     : time to maturity of the options

        Returns
        -------
        S : Matrix of underlying asset prices
        B : Coefficients of the IV surface 
        H : Volatility of the underlying asset

      """
    
    # 1) Load datasets to train and test deep hedging algorithm (Simulated paths, Betas IV, volatilities)
    # 1.1) matrix of shappley decomposition
    contribution = np.load(os.path.join(f"et_{name}"))
    # 1.2) matrix of gains and losses of VA
    gl_va = np.load(os.path.join(f"gv_{name}"))
    # 1.3) Present values of hedging instruments gains
    HO    = np.load(os.path.join(f"hi_{name}"))
    # 1.4) state sapce variables
    input = np.load(os.path.join(f"xt_{name}"))
    
    return contribution, gl_va, HO, input

def training_variables(mixed_fund, time_steps, ratchet_provision, hedging):

    """Function that loads the sets to create the training set

        Parameters
        ----------
        temporality                  : Temporality of the simulation
        r                            : Risk-free rate                  
        q                            : Dividend yield rate
        delta                        : Step-size (daily rebalancing)
        tc                           : Transaction cost level
        backtest                     : Simulation based on real or simulated data
        lower_bound                  : Lower bound to clip IV surface values
        issmile                      : To determine the type of greeks 
        hedged_options_maturities    : Option maturities for instruments to hedge
        hedged_option_types          : Option types {Call: False, Put: True}
        hedged_moneyness             : Option moneyness for instruments to hedge {"ATM","OTM","ITM"}
        hedged_positions             : Number of shares of each option to hedge
        hedging_intruments_maturity  : Option maturities for hedging instruments
        hedging_option_types         : Option types {Call: False, Put: True}
        hedging_option_moneyness     : Option moneyness for hedging instruments {"ATM","OTM","ITM"}
        dynamics                     : Simulation type {Single intrument: 'static', New instrument every day: 'dynamic'}     

        Returns
        -------
        id                   : Acronym for the hedging problem
        train_input          : Training set (normalized stock price and features)
        test_input           : Test set (normalized stock price and features)
        HO_train             : Prices of hedging instruments in the training set
        HO_test              : Prices of hedging instruments in the validation set
        cash_flows_train     : Cash flows of hedged portfolio for training set
        cash_flows_test      : Cash flows of hedged portfolio for validation set
        risk_free_factor     : Risk-free rate update factor exp(h*r)
        dividendyield_factor : Dividend yield update factor exp(h*d)
        underlying_train     : Underlying asset prices for training set
        underlying_test      : Underlying asset prices for validation set

      """

    owd = os.getcwd()
    try:
      #first change dir to build_dir path
      
      os.chdir(os.path.join(main_folder, f"data/processed/"))
      name_1 = "rbc" if mixed_fund==False else 'assupmtion'
      name_2 = str(time_steps)
      name_3 = "ratchet" if ratchet_provision==True else ""
      id = f"{name_1}_{name_2}_{name_3}.npy"
      contribution, gl_va, HO, input = load_standard_datasets(id)
      
      #Define training arrays
      n_timesteps = time_steps
      """
      if hedging == "contribution":
        train_input = np.expand_dims(input[:,0:100000,:,0], axis=-1)
        test_input  = np.expand_dims(input[:,100000:,:,0], axis=-1)
        HO_train    = np.expand_dims(HO[:,0:100000,:,0], axis=-1)
        HO_test     = np.expand_dims(HO[:,100000:,:,0], axis=-1)
        gl_va_train = np.expand_dims(gl_va[0:100000,0], axis=-1)
        gl_va_test  = np.expand_dims(gl_va[100000:,0], axis=-1)
        contribution_train = contribution[0:100000]
        contribution_test  = contribution[100000:]
      else:
      """
      train_input = input[:,0:-1000,:,:]
      test_input  = input[:,-1000:,:,:]
      HO_train    = HO[:,0:-1000,:,:]
      HO_test     = HO[:,-1000:,:,:]
      gl_va_train = gl_va[0:-1000,:]
      gl_va_test  = gl_va[-1000:,:]
      contribution_train = contribution[0:-1000]
      contribution_test  = contribution[-1000:]      
      
    finally:
      #change dir back to original working directory (owd)
      os.chdir(owd)

    id = f"{name_1}_{name_2}_{name_3}"
      
    return id, n_timesteps, train_input, test_input, HO_train, HO_test, gl_va_train, gl_va_test, contribution_train, contribution_test 
  



