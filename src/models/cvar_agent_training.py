"""
Usage:
    1. cd src
    2. python3 models/cvar_agent_training.py 
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

from src.models.cvar_agent import DeepAgent
import tensorflow.compat.v1 as tf
import pickle

from src.utils import *
from src.data.data_loader_cvar import *
from src.models.cvar_agent import train_network, network_inference

def rl_agent(config_file_simulation,config_file_agent):
    
    """Function that trains the RL agent based on the configuration of the config files
    
    Parameters
    ----------
    config_file_simulation    : simulation settings for the JIVR model and the underlying asset 
    config_file_pricing_agent : hyperparameters of the RL agent

    Output
    ----------
    deltas: hedging strategies
    
    """
    # 0) Default parameters 
    # Parameters of market simulation
    mixed_fund = config_file_simulation['mixed_fund']
    time_steps = config_file_simulation['time_steps']
    ratchet_provision = config_file_simulation['ratchet_provision']

    # 1) Loading data in the right pricing agent
    id, train_input, test_input, train_output, test_output = training_variables(mixed_fund, time_steps, ratchet_provision)

    # 2) First layer of hyperparameters
    batch_size    = config_file_agent['batch_size']
    nbs_input     = train_input.shape[1]
    nbs_units     = config_file_agent['nbs_units']
    lr            = config_file_agent['lr']
    dropout_par   = config_file_agent['dropout_par']
    epochs        = config_file_agent['epochs']
    display_plot  = config_file_agent['display_plot']

    # 3) Train pricing agent
    loss_train_epoch = train_network(batch_size, nbs_input, nbs_units, lr, dropout_par, train_input, train_output, test_input, test_output, epochs, display_plot, id)

    print("--- CVaR agent trained and stored in ../models/.. ---")

    name_1 = f"FFNN_dropout_{str(int(dropout_par*100))}_cvar_net_{id}"

    os.chdir(os.path.join(main_folder, f"models"))

    # Compile the neural network
    rl_network = DeepAgent(batch_size, nbs_input, nbs_units, lr, dropout_par, name_1)
                            
    print("-------------------------------------------------------------")
    print(id)
    print("-------------------------------------------------------------")

    # Start training
    print('---Inference start---')
    with tf.Session() as sess:
        rl_network.restore(sess, f"{name_1}.ckpt")
        predicted = rl_network.predict(train_input, train_output, test_input, test_output, sess)
        
        # Access the model parameters
        model_variables = tf.trainable_variables()  # or use tf.global_variables() for all variables

        # List to store the parameter values as numpy arrays
        parameter_list = []
        
        # Evaluate and print each variable's value
        for var in model_variables:
            var_value = sess.run(var)
            parameter_list.append(var_value)


    print("--- Storing CVaR agent parameters in ../models/.. ---")

    with open(f'{name_1}.pkl', 'wb') as f:
        pickle.dump(parameter_list, f)

    print("--- CVaR net parameters stored ---")

    return 
    
if __name__ == "__main__":

    # Set environment variable
    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_cvar_agent.yml'))
    config_file_agent = config_file["agent"]
    config_file = load_config(os.path.join(main_folder,'cfgs','config_market_simulation.yml'))
    config_file_simulation = config_file["simulation"]
    _ = rl_agent(config_file_simulation,config_file_agent)
    