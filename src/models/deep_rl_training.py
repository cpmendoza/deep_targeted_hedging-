"""
Usage:
    1. cd src
    2. python3 models/deep_rl_training.py 
"""

import os, sys
from pathlib import Path
import warnings

# Set environment variable
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:urllib3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Suppress specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

from src.utils import *
from src.data.data_loader import *
from src.models.deep_agent import train_network
from src.models.deep_agent import network_inference

def rl_agent(config_file_simulation,config_file_agent):
    
    """Function that trains the RL agent based on the configuration of the config files
    
    Parameters
    ----------
    config_file_simulation : simulation settings for the JIVR model and the underlying asset 
    config_file_agent : hyperparameters of the RL agent

    Output
    ----------
    deltas: hedging strategies
    
    """
    # 0) Default parameters 
    # Parameters of market simulation
    mixed_fund = config_file_simulation['mixed_fund']
    time_steps = config_file_simulation['time_steps']
    ratchet_provision = config_file_simulation['ratchet_provision']
    hedging = config_file_agent['hedging']

    # 1) Loading data in the right shape for RL-agent input
    id, n_timesteps, train_input, test_input, HO_train, HO_test, gl_va_train, gl_va_test, contribution_train, contribution_test = training_variables(mixed_fund, time_steps, ratchet_provision, hedging)

    # 2) First layer of RL agent hyperparameters
    network          = config_file_agent['network']        # Neural network architecture {"LSTM","RNNFNN","FFNN"}
    hedging          = hedging
    nbs_point_traj   = n_timesteps                         # time steps 
    batch_size       = config_file_agent['batch_size']     # batch size {296,1000}
    nbs_input        = train_input.shape[2]                # number of features
    nbs_units        = config_file_agent['nbs_units']      # neurons per layer/cell
    nbs_assets       = HO_train.shape[-2]                  # number of hedging intruments
    lr               = config_file_agent['lr']             # learning rate of the Adam optimizer
    dropout_par      = config_file_agent['dropout_par']    # dropout regularization parameter 
    riskaversion     = config_file_agent['riskaversion']   # CVaR confidence level (0,1)
    epochs           = config_file_agent['epochs']         # Number of epochs, training iterations
    penalization     = config_file_agent['penalization']   # Inclusion of CVaR penalization {0,1}

    # Second layer of parameters
    id                   = id                   # Acronym for the hedging problem
    train_input          = train_input          # Training set (normalized stock price and features)
    test_input           = test_input           # Test set (normalized stock price and features)
    HO_train             = HO_train             # Prices of hedging instruments in the training set
    HO_test              = HO_test              # Prices of hedging instruments in the validation set
    gl_va_train          = gl_va_train          # Cash flows of hedged portfolio for training set
    gl_va_test           = gl_va_test           # Cash flows of hedged portfolio for validation set
    contribution_train   = contribution_train   # Underlying asset prices for training set
    contribution_test    = contribution_test    # Underlying asset prices for validation set

    # Third layer of parameters
    display_plot    = config_file_agent['display_plot']     # Display plot of training and validation loss 

    # 3) Train RL agent
    loss_train_epoch, cvar = train_network(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging, 
                                 train_input, gl_va_train, gl_va_test, HO_train, HO_test, 
                                 contribution_train, contribution_test, riskaversion, test_input, epochs, display_plot, id, penalization)

    print("--- Deep agent trained and stored in ../models/.. ---")

    # 4) Compute hedging strategy
    portfolio, deltas, name = network_inference(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging, 
                                 train_input, gl_va_train, gl_va_test, HO_train, HO_test, 
                                 contribution_train, contribution_test, riskaversion, test_input, epochs, display_plot, id, penalization)
    
    print("--- Hedging startegy stored in ../results/Training/.. ---")

    print("--- Risk allocation metrics ---")
    _ = strategy_evaluation(portfolio)

    return 
    
if __name__ == "__main__":

    # Set environment variable
    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_agent.yml'))
    config_file_agent = config_file["agent"]
    config_file = load_config(os.path.join(main_folder,'cfgs','config_market_simulation.yml'))
    config_file_simulation = config_file["simulation"]
    _ = rl_agent(config_file_simulation,config_file_agent)
    