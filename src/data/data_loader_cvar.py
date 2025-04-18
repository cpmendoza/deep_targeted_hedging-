"""
Usage:
    1. cd src
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
        name     : root name for files

        Returns
        -------
        input  : input for pricing agent (state vectors)
        output : estimated va price based on Monte Carlo method

      """
    
    # 1) Load datasets to train and test pricing agent Pi
    input = np.load(os.path.join(main_folder,"data/processed",f"xt_{name}.npy"))[:,100000:,:,0]
    position = np.load(os.path.join(main_folder,"data/processed",f"po_{name}.npy"))[:,:,:,0]
    hedging_instruments = np.load(os.path.join(main_folder,"data/processed",f"hi_{name}.npy"))[:,-1000:,:,:]
    cvar = np.load(os.path.join(main_folder,"data/processed",f"cvar_{name}.npy"))
    
    return input, position, hedging_instruments, cvar

def training_variables(mixed_fund, time_steps, ratchet_provision):

    """Function that loads the sets to create the training set

        Parameters
        ----------
        mixed_fund        :
        time_steps        :
        ratchet_provision :

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
      id = f"{name_1}_{name_2}_{name_3}"
      input, position, hedging_instruments, cvar = load_standard_datasets(id)
      
      #Define training arrays
      input_cvar_nework = np.zeros([240,1000,12,2])
      input_cvar_nework[:,:,:9,0] = input
      input_cvar_nework[:,:,:9,1] = input

      cash_flows_hedging_portfolio_delta = ((hedging_instruments*np.load(os.path.join(main_folder,"data/processed",f"po_{id}.npy"))).sum(axis=2))[:,:,0]
      cash_flows = cash_flows_hedging_portfolio_delta
      gl = cash_flows.cumsum(axis=0)
      input_cvar_nework[1:,:,9,0] = gl[1:,:]

      cash_flows_hedging_portfolio_delta = ((hedging_instruments*(-1)*np.load(os.path.join(main_folder,"data/processed",f"po_{id}.npy"))).sum(axis=2))[:,:,0]
      cash_flows = cash_flows_hedging_portfolio_delta
      gl = cash_flows.cumsum(axis=0)
      input_cvar_nework[1:,:,9,1] = gl[1:,:]

      input_cvar_nework[:,:,10:12,0] = position[:,:,:]
      input_cvar_nework[:,:,10:12,1] = -1*position[:,:,:]


      # Reshape the input array
      input = np.transpose(input_cvar_nework, (0, 1, 3, 2))  # Shape: (240, 1000, 2, 12)
      input = input.reshape(-1, 12)            # Shape: (240*1000*2, 12)

      # Reshape the output array
      output = cvar.reshape(-1, 1)   

      # Generate a random permutation of indices
      indices = np.random.permutation(input.shape[0])

      # Shuffle both arrays using the same indices
      input = input[indices]
      output = output[indices]

      # Delete negative values
      negative_indices = np.where(output > 0)[0]
      input = input[negative_indices]
      output = output[negative_indices]

      input = input[:402000,:]
      output = output[:402000]

      train_input   = input[:360000]
      train_output  = output[:360000]
      test_input    = input[360000:]
      test_output   = output[360000:]
      
    finally:
      #change dir back to original working directory (owd)
      os.chdir(owd)

    id = f"{name_1}_{name_2}_{name_3}"
      
    return id, train_input, test_input, train_output, test_output
  



