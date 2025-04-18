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
    # 1.1) matrix with pricing input
    deltas_input = np.load(os.path.join(main_folder,"data/processed",f"de_{name}"))
    # 1.2) matrix of VA price
    price = np.load(os.path.join(main_folder,"data/processed",f"pa_{name}"))
    
    return deltas_input, price

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
      id = f"{name_1}_{name_2}_{name_3}.npy"
      deltas_input, price = load_standard_datasets(id)
      
      #Define training arrays
      input = deltas_input[:,:,:10,0]
      # Create the new element (240, 1000) where the first row is all 0s, second all 1s, and so on... this one represents the month
      new_element = np.tile(np.arange(240).reshape(240, 1), (1, 1000))
      # Expand new_element to match the third dimension and concatenate along that axis
      input = np.concatenate((input, new_element[..., np.newaxis]), axis=2)
      input = input.reshape(240 * 1000, 11)
      output = price[:,:,0].reshape(240 * 1000, 1)
      # Generate a random permutation of indices
      indices = np.random.permutation(input.shape[0])
      # Shuffle both arrays using the same indices
      input = input[indices]
      output = output[indices]    

      train_input   = input[:200000]
      train_output  = output[:200000]
      test_input    = input[200000:]
      test_output   = output[200000:]


      
    finally:
      #change dir back to original working directory (owd)
      os.chdir(owd)

    id = f"{name_1}_{name_2}_{name_3}"
      
    return id, train_input, test_input, train_output, test_output
  



