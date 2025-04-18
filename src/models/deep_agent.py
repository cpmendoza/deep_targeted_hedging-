"""
Usage:
    1. cd src
    2. python 
"""


import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow.compat.v1 as tf
import random
import datetime as dt
import matplotlib.pyplot as plt
import pickle
from tensorflow.python.framework import ops

# Load and convert parameters once outside the function
def load_and_convert_parameters():
    name = "FFNN_dropout_50_cvar_net_assupmtion_240_"
    name = os.path.join(main_folder, "models", f'{name}.pkl')
    
    # Load the parameters
    with open(name, 'rb') as f:
        parameter_list = pickle.load(f)
    
    # Convert all parameters to TensorFlow tensors
    tensor_parameter_list = [tf.convert_to_tensor(param, dtype=tf.float32) for param in parameter_list]
    
    return tensor_parameter_list

# Adapted function to work with tensors, no need to convert inside the function
def cvar_estimation(test_input, parameter_list):
    layer = None
    for i in range(5):
        A = test_input if i == 0 else layer
        B = parameter_list[2*i]
        b = parameter_list[2*i+1]
        
        # Linear component
        linear_component = tf.matmul(A, B) + b
        
        # ReLU activation for first 4 layers, no activation on the last layer
        layer = tf.nn.relu(linear_component) if i < 4 else linear_component
    
    return layer


class DeepAgent(object):
    """
    Inputs:
    network        : neural network architechture {LSTM,RNN-FNN,FFNN}
    nbs_point_traj : if [S_0,...,S_N] ---> nbs_point_traj = N+1
    batch_size     : size of mini-batch
    nbs_input      : number of features (without considerint V_t)
    nbs_units      : number of neurons per layer
    nbs_assets     : dimension of the output layer (number of hedging instruments)
    lambda_m       : regularization parameter for soft-constraint in lagrange multiplier
    constraint_max : Lower bound of the output layer activation function
    loss_type      : loss function for the optimization procedure {CVaR,SMSE,MSE}
    lr             : learning rate hyperparameter of the Adam optimizer
    dropout_par:   : dropout regularization parameter [0,1]
    isput          : condition to determine the option type for the hedging error {True,False}
    prepro_stock   : {Log, Log-moneyness, Nothing} - what transformation was used for stock prices
    name           : name to store the trained model

    # Disclore     : Class adapted from https://github.com/alexandrecarbonneau/Deep-Equal-Risk-Pricing-of-Financial-Derivatives-with-Multiple-Hedging-Instruments/blob/main/Example%20of%20ERP%20with%20deep%20hedging%20multi%20hedge%20-%20Final.ipynb
    """
    def __init__(self, network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging, penalization, name):
        tf.compat.v1.disable_eager_execution()
        ops.reset_default_graph()

        # 0) Deep hedging parameters parameters
        self.network        = network
        self.nbs_point_traj = nbs_point_traj
        self.batch_size     = batch_size
        self.nbs_units      = nbs_units
        self.nbs_assets     = nbs_assets
        self.lr             = lr
        self.dropout_par    = dropout_par
        self.penalization   = penalization
        self.deltas         = tf.zeros(shape = [nbs_point_traj-1, batch_size, nbs_assets], dtype=tf.float32)  # array to store position in the hedging instruments
        self.input_c        = tf.zeros(shape = [batch_size, 12], dtype=tf.float32)
        self.local_cvar     = tf.zeros(shape = [batch_size, 1], dtype=tf.float32)
        self.cvar_history   = tf.zeros(shape = [nbs_point_traj-1,1], dtype=tf.float32)

        tensor_parameter_list = load_and_convert_parameters()
        
        # 1) Soft-constraint parameters
        self.hedging = hedging

        # 2) Output parameters
        self.portfolio      = tf.zeros(shape = [nbs_point_traj, batch_size])

        # 3) Placeholder for deep hedging elements
        index = 8 #if self.hedging == 'allocation' else 1        
        self.input                = tf.placeholder(tf.float32, [nbs_point_traj, batch_size, nbs_input, index])      # normalized prices and features
        self.gl_va                = tf.placeholder(tf.float32, [batch_size, index])                                 # gains and losses of VA
        self.riskaversion         = tf.placeholder(tf.float32)                                                  # CVaR confidence level (alpha in (0,1))
        self.hedging_instruments  = tf.placeholder(tf.float32, [nbs_point_traj, batch_size, self.nbs_assets, index]) # hedging instruments prices
        self.contribution         = tf.placeholder(tf.float32, [batch_size])                                    # Contribution of shap value

        # 5) Network architechture for the deep hedging algorithm
        if (self.network == "LSTM"):
          # 5.1.1) Four LSTM cells (the dimension of the hidden state and the output is the same)
          layer_1 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units) 
          layer_2 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_3 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_4 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          # 5.1.2) Output layer of dimension nbs_assets (outputs the position in the hedging instruments)
          layer_out = tf.layers.Dense(self.nbs_assets, None)
        elif (self.network == "RNNFNN"):
          # 5.2.1) Two LSTM cells (the dimension of the hidden state and the output is the same)
          #      Two regular layers with RELU activation function and dropout regularization per layer
          layer_1 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units) 
          layer_2 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_3 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_3 = tf.keras.layers.Dropout(self.dropout_par)
          layer_4 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_4 = tf.keras.layers.Dropout(self.dropout_par)
          # 5.2.2) Output layer of dimension nbs_assets (outputs the position in the hedging instruments)
          layer_out = tf.layers.Dense(self.nbs_assets, None)
        elif (self.network == "FFNN"):
          # 5.3.1) Four regular layers with RELU activation function and dropout regularization per layer
          layer_1 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_1 = tf.keras.layers.Dropout(self.dropout_par)
          layer_2 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_2 = tf.keras.layers.Dropout(self.dropout_par)
          layer_3 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_3 = tf.keras.layers.Dropout(self.dropout_par)
          layer_4 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_4 = tf.keras.layers.Dropout(self.dropout_par)
          # 5.3.2) Output layer of dimension one (outputs the position in the underlying)
          layer_out = tf.layers.Dense(self.nbs_assets, None)

        # 6) Compute hedging strategies for all time-steps
        self.gains_s = tf.zeros(shape = [batch_size,index], dtype=tf.float32)
        #Iteration over Shapley scenarios
        for s in range(index): 
         
            # First cash flow and first possible action
            V_t_pre = tf.zeros(shape = [batch_size])                        
            layer = tf.zeros(shape = [batch_size,self.nbs_assets], dtype=tf.float32)
            
            # Based on initial information, we start rebalancing hedging portfolio #
            for t in range(self.nbs_point_traj):
                
                input_t = tf.concat([self.input[t,:,:,s], tf.expand_dims(V_t_pre, axis = 1)], axis=1)
                #input_t = tf.concat([input_t, layer], axis=1)
                time_t = tf.fill([batch_size, self.nbs_assets], tf.cast(t/self.nbs_point_traj, tf.float32))
                input_t = tf.concat([input_t, time_t], axis=1)
                input_t = input_t if self.network == "FFNN" else tf.expand_dims(input_t  , axis = 1)
                
                #RL Agent
                if (self.network == "LSTM"):
                    # forward prop at time 't'
                    layer = layer_1(input_t)
                    layer = layer_2(tf.expand_dims(layer, axis = 1))
                    layer = layer_3(tf.expand_dims(layer, axis = 1))
                    layer = layer_4(tf.expand_dims(layer, axis = 1))
                    layer = layer_out(layer)

                elif (self.network == "RNNFNN"):
                    # forward prop at time 't'
                    layer = layer_1(input_t)
                    layer = layer_2(tf.expand_dims(layer, axis = 1))
                    layer = layer_3(layer)
                    layer = layer_drop_3(layer)
                    layer = layer_4(layer)
                    layer = layer_drop_4(layer)
                    layer = layer_out(layer)

                else:
                    # forward prop at time 't'
                    layer = layer_1(input_t)
                    layer = layer_drop_1(layer)
                    layer = layer_2(layer)
                    layer = layer_drop_2(layer)
                    layer = layer_3(layer)
                    layer = layer_drop_3(layer)
                    layer = layer_4(layer)
                    layer = layer_drop_4(layer)
                    layer = layer_out(layer)
                    layer = tf.where(layer > 0, -layer, layer)

                #Local CVaR computation when all risk factors are considered  
                if s ==0:
                    self.input_c = tf.concat([self.input[t,:,:,s], tf.expand_dims(V_t_pre, axis = 1)], axis=1)
                    self.input_c = tf.concat([self.input_c, layer], axis=1)
                    self.local_cvar = tf.reduce_mean(cvar_estimation(self.input_c, tensor_parameter_list))
                    #Store CVaR at each time step
                    self.cvar_history = tf.expand_dims(self.local_cvar,axis=0) if t==0 else tf.concat([self.cvar_history, tf.expand_dims(self.local_cvar,axis=0)], axis=0)
            
                # Compile trading strategies
                if (t==0):
                    # At t = 0, need to expand the dimension to have [nbs_point_traj, batch_size, nbs_assets]
                    deltas = tf.expand_dims(layer,axis=0)                      
                else:
                    # Store the rest of the hedging positions
                    deltas = tf.concat([deltas, tf.expand_dims(layer, axis = 0)], axis = 0)
                    
                #Compute gains and losses
                self.h_gain   = tf.zeros(shape = [batch_size], dtype=tf.float32)
                for a in range(self.nbs_assets):
                    self.h_gain += layer[:,a]*self.hedging_instruments[t,:,a,s]

                # Compute the portoflio value for the next period
                V_t_pre = V_t_pre + self.h_gain

            gains_s = (self.gl_va[:,s] + V_t_pre) if self.hedging == 'allocation' else  V_t_pre
            if s==0:
                self.gains_s = tf.expand_dims(gains_s, axis = 1)
                self.deltas  = tf.expand_dims(deltas, axis = -1)
            else:
                self.deltas  = tf.concat([self.deltas,tf.expand_dims(deltas, axis = -1)], axis = -1)
                self.gains_s = tf.concat([self.gains_s, tf.expand_dims(gains_s,axis=1)], axis = 1)
        
        #Define arrays to compute Shapley decompostions
        self.time = self.gains_s[:,7]
        self.rate = 0.333333333*(self.gains_s[:,3]-self.gains_s[:,7])+0.166666667*((self.gains_s[:,1]-self.gains_s[:,5])+(self.gains_s[:,2]-self.gains_s[:,6]))+0.333333333*(self.gains_s[:,0]-self.gains_s[:,4])
        self.equity = 0.333333333*(self.gains_s[:,5]-self.gains_s[:,7])+0.166666667*((self.gains_s[:,1]-self.gains_s[:,3])+(self.gains_s[:,4]-self.gains_s[:,6]))+0.333333333*(self.gains_s[:,0]-self.gains_s[:,2])
        self.mortality = 0.333333333*(self.gains_s[:,6]-self.gains_s[:,7])+0.166666667*((self.gains_s[:,2]-self.gains_s[:,3])+(self.gains_s[:,4]-self.gains_s[:,5]))+0.333333333*(self.gains_s[:,0]-self.gains_s[:,1])
        self.total = self.time + self.rate + self.equity + self.mortality

        # 6) Compute objective function
        if self.hedging == 'allocation':
            #Minimize the distance bewteen the portfolio value and the contribution
            self.local_var = tf.compat.v1.reduce_sum(self.cvar_history)
            self.hedging_err = tf.nn.relu(-1*(self.equity))
            self.loss =  tf.reduce_mean(self.hedging_err) + self.local_var
        else: 
            #Minimize the distance bewteen the portfolio value and the contribution
            self.local_var = tf.reduce_mean(tf.nn.relu(self.cvar_history))
            self.hedging_err = tf.nn.relu(-1*(self.contribution + self.equity))
            self.loss =  tf.reduce_mean(self.hedging_err) + (tf.reduce_mean(tf.square(self.rate)) + tf.reduce_mean(tf.square(self.mortality)) + tf.reduce_mean(tf.square(self.time))) + self.penalization*self.local_var 

        # 8) SGD step with the adam optimizer
        optimizer  = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train = optimizer.minimize(self.loss)

        # 9) Save the model
        self.saver      = tf.train.Saver()
        self.model_name = name   # name of the neural network to save

    # Function to compute the CVaR_{alpha} outside the optimization, i.e. at the end of each epoch in this case
    def loss_out_optim(self, hedging_err, time, rate, mortality, hedging):
        if (hedging == "allocation"):
            loss = np.max(hedging_err)
        elif (hedging == "contribution"):
            loss = np.mean(np.square(np.where(hedging_err>0,hedging_err,0))) + np.mean(np.square(time)) + np.mean(np.square(rate)) + np.mean(np.square(mortality))
        return loss

    # ---------------------------------------------------------------------------------------#
    # Function to call the deep hedging algorithm batch-wise
    """
    Input:
    train_input          : Training set (normalized stock price and features)
    underlying_train     : Underlying asset prices for training set
    underlying_test      : Underlying asset prices for validation set
    HO_train             : Prices of hedging instruments in the training set
    HO_test              : Prices of hedging instruments in the validation set
    cash_flows_train     : Cash flows of hedged portfolio for training set
    cash_flows_test      : Cash flows of hedged portfolio for validation set
    risk_free_factor     : Risk-free rate update factor exp(h*r)
    dividendyield_factor : Dividend yield update factor exp(h*d)
    transaction_cost     : Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion         : CVaR confidence level (0,1)
    test_input           : Test set (normalized stock price and features) 
    sess                 : tensorflow session
    epochs               : Number of epochs, training iterations
    """
    def train_deephedging(self, train_input, gl_va_train, gl_va_test, HO_train, HO_test, contribution_train, contribution_test, riskaversion, test_input, sess, epochs):

        sample_size       = train_input.shape[1]               # total number of paths in the train set
        sample_size_valid = test_input.shape[1]
        batch_size        = self.batch_size
        idx               = np.arange(sample_size)       # [0,1,...,sample_size-1]
        idx_valid         = np.arange(sample_size_valid)
        start             = dt.datetime.now()            # Time-to-train
        self.loss_epochs  = 9999999*np.ones((epochs,6))      # Store the loss at the end of each epoch for the train
        valid_loss_best   = 999999999
        epoch             = 0
        risk_metric = "CVaR" if self.hedging == "allocation" else "MSE"

        # Loop for each epoch until the maximum number of epochs
        while (epoch < epochs):
            hedging_err_train = []  # Store hedging errors obtained for one complete epoch
            time_err_train = []
            rate_err_train = []
            mortality_err_train = []
            hedging_err_valid = []
            time_err_test = []
            rate_err_test = []
            mortality_err_test = []
            np.random.shuffle(idx)  # Randomize the dataset (not useful in this case since dataset is simulated iid)

            # loop over each batch size
            for i in range(int(sample_size/batch_size)):

                # Indexes of paths used for the mini-batch
                indices = idx[i*batch_size : (i+1)*batch_size]

                # SGD step
                _, hedging_err, time, rate, mortality, cvar = sess.run([self.train, self.hedging_err, self.time, self.rate, self.mortality, self.local_var],
                                               {self.input                 : train_input[:,indices,:,:],
                                                self.gl_va                 : gl_va_train[indices,:],
                                                self.riskaversion          : riskaversion,
                                                self.hedging_instruments   : HO_train[:,indices,:,:],
                                                self.contribution          : contribution_train[indices]})
                

                hedging_err_train.append(hedging_err)
                time_err_train.append(time)
                rate_err_train.append(rate)
                mortality_err_train.append(mortality)

            # 2) Evaluate performance on the valid set - we don't train
            for i in range(int(sample_size_valid/batch_size)):
                indices_valid = idx_valid[i*batch_size : (i+1)*batch_size]
                hedging_err, time, rate, mortality, cvar = sess.run([self.hedging_err, self.time, self.rate, self.mortality, self.local_var],
                                               {self.input                 : test_input[:,indices_valid,:,:],
                                                self.gl_va                 : gl_va_test[indices_valid,:],
                                                self.riskaversion          : riskaversion,
                                                self.hedging_instruments   : HO_test[:,indices_valid,:,:],
                                                self.contribution          : contribution_test[indices_valid]})

                hedging_err_valid.append(hedging_err)
                time_err_test.append(time)
                rate_err_test.append(rate)
                mortality_err_test.append(mortality)

            # 3) Store the loss on the train and valid sets after each epoch
            self.loss_epochs[epoch,0] = self.loss_out_optim(np.concatenate(hedging_err_train, axis=0),
                                                            np.concatenate(time_err_train, axis=0),
                                                             np.concatenate(rate_err_train, axis=0),
                                                              np.concatenate(mortality_err_train, axis=0), self.hedging)
            self.loss_epochs[epoch,1] = self.loss_out_optim(np.concatenate(hedging_err_valid, axis=0),
                                                            np.concatenate(time_err_test, axis=0),
                                                             np.concatenate(rate_err_test, axis=0),
                                                              np.concatenate(mortality_err_test, axis=0), self.hedging) #, axis=1

            model_saved = 0
            # 4) Test if best epoch so far on valid set; if so, save model parameters.
            if((self.loss_epochs[epoch,1] < valid_loss_best)): #& (self.loss_epochs[epoch,1]>0)
                valid_loss_best = self.loss_epochs[epoch,1]
                self.saver.save(sess, self.model_name + '.ckpt')
                model_saved = 1
                
            # Print the CVaR value at the end of each epoch
            if (epoch+1) % 1 == 0:
                if (model_saved ==1):
                    print('Epoch %d, Time elapsed:'% (epoch+1), dt.datetime.now()-start, " - Model saved")
                else:
                    print('Epoch %d, Time elapsed:'% (epoch+1), dt.datetime.now()-start)
                print('  Train - SMSE: %.6f, Time: %.6f, Rate: %.6f, Mortality: %.6f, Local-CVaR: %.6f' % 
                      (np.mean(np.square(np.where(np.concatenate(hedging_err_train, axis=0)>0,np.concatenate(hedging_err_train, axis=0),0))),np.mean(np.square(time_err_train)),
                       np.mean(np.square(rate_err_train)),np.mean(np.square(mortality_err_train)),cvar))
                print('  Valid - SMSE: %.6f, Time: %.6f, Rate: %.6f, Mortality: %.6f, Local-CVaR: %.6f' % 
                      (np.mean(np.square(np.where(np.concatenate(hedging_err_valid, axis=0)>0,np.concatenate(hedging_err_valid, axis=0),0))),np.mean(np.square(time_err_test)),
                       np.mean(np.square(rate_err_test)),np.mean(np.square(mortality_err_test)),cvar))
            
            epoch+=1  # increment the epoch

        # End of training
        print("---Finished training results---")
        print('Time elapsed:', dt.datetime.now()-start)

        # Return the learning curve
        return self.loss_epochs, cvar


    # Function which will call the deep hedging optimization batchwise
    def training(self, train_input, gl_va_train, gl_va_test, HO_train, HO_test, contribution_train, contribution_test, riskaversion, test_input, sess, epochs):
        sess.run(tf.global_variables_initializer())
        loss_train_epoch, cvar = self.train_deephedging(train_input, gl_va_train, gl_va_test, HO_train, HO_test, contribution_train, contribution_test, riskaversion, test_input, sess, epochs)
        return loss_train_epoch, cvar

    # ---------------------------------------------------------------------- #
    # Function to compute the hedging strategies of a trained neural network
    # - Doesn't train the neural network, only outputs the hedging strategies
    def predict(self, train, gl_va, HO, contribution, riskaversion, sess):
        sample_size = train.shape[1]
        batch_size  = self.batch_size
        idx         = np.arange(sample_size)  # [0,1,...,sample_size-1]
        strategy_pred = [] # hedging strategies
        portfolio_pred = []

        # loop over sample size to do one complete epoch
        for i in range(int(sample_size/batch_size)):

            # mini-batch of paths (even if not training to not get memory issue)
            indices = idx[i*batch_size : (i+1)*batch_size]
            portfolio, deltas = sess.run([self.gains_s, self.deltas],
                                    {self.input                : train[:,indices,:,:],
                                    self.gl_va                 : gl_va[indices,:],
                                    self.riskaversion          : riskaversion,
                                    self.hedging_instruments   : HO[:,indices,:,:],
                                    self.contribution          : contribution[indices]})

            # Append the batch of hedging strategies
            strategy_pred.append(deltas)
            portfolio_pred.append(portfolio)

        return np.concatenate(portfolio_pred,axis=0), np.concatenate(strategy_pred,axis=1)

    def restore(self, sess, checkpoint):
        self.saver.restore(sess, checkpoint)


def train_network(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging,
                   train_input, gl_va_train, gl_va_test, HO_train, HO_test,
                     contribution_train, contribution_test, riskaversion, test_input, epochs, display_plot, id, penalization):
    
    # Function to train deep hedging algorithm
    """
    Input:
    network              : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj       : time steps 
    batch_size           : batch size {296,1000}
    nbs_input            : number of features
    nbs_units            : neurons per layer/cell
    nbs_assets           : number of hedging intruments
    constraint_min       : minimum value of the portfolio for soft constraint
    lambda_m             : labmda for lagrange multiplier (soft constraint)
    autoencoder          : autoencoder component included
    nbs_latent_spc       : dimension of the latent space
    lambda_auto          : regularization parameter to ponderate autoencoder loss function
    loss_type            : loss function {"CVaR","MSE","SMSE"}
    lr                   : learning rate of the Adam optimizer
    dropout_par          : dropout regularization parameter 
    transaction_cost     : Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion         : CVaR confidence level (0,1)
    epochs               : Number of epochs, training iterations
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
    display_plot         : Display plot of training and validation loss 
    penalization         : Penalization weight for the local CVaR

    Output:
    loss_train_epoch : Loss history per epochs

    """
    
    owd = os.getcwd()
    try:
        os.chdir(os.path.join(main_folder, f"models"))

        if hedging == "allocation":
            name = f"Deep_agent_{network}_dropout_{str(int(dropout_par*100))}_allocation_{id}"
        else:
            name = f"Deep_agent_{network}_dropout_{str(int(dropout_par*100))}_contribution_{id}"

        
        # Compile the neural network
        rl_network = DeepAgent(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging, penalization, name)
                             
        print("-------------------------------------------------------------")
        print(name)
        print("-------------------------------------------------------------")

        # Start training
        print("-----------------------Training start------------------------")
        with tf.Session() as sess:
            loss_train_epoch, cvar = rl_network.training(train_input, gl_va_train, gl_va_test, HO_train, HO_test, contribution_train, contribution_test, riskaversion, test_input, sess, epochs)
        print('---Training end---')

        if display_plot == True:
            loss_type = "CVaR" if hedging == "allocation" else "MSE"
            # Plot the learning curve on the train set, i.e. the CVaR on the train set at the end of each epoch
            lin_nb_epoch = np.linspace(1, loss_train_epoch.shape[0], loss_train_epoch.shape[0])
            fig = plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,0],label="Train error")
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,1],label="Test error")
            plt.title(f"Loss function: {loss_type}")
            plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            plt.show()
    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)
    #np.save(os.path.join(main_folder, f"data/results/Training/results_extra/loss_functions/loss_{name}.npy"),loss_train_epoch)

    return loss_train_epoch, cvar

def retrain_network(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging,
                   train_input, gl_va_train, gl_va_test, HO_train, HO_test,
                     contribution_train, contribution_test, riskaversion, test_input, epochs, display_plot, id, penalization):

    # Function to retrain deep hedging algorithm
    """
    Input:
    network              : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj       : time steps 
    batch_size           : batch size {296,1000}
    nbs_input            : number of features
    nbs_units            : neurons per layer/cell
    nbs_assets           : number of hedging intruments
    constraint_min       : minimum value of the portfolio for soft constraint
    lambda_m             : labmda for lagrange multiplier (soft constraint)
    autoencoder          : autoencoder component included
    nbs_latent_spc       : dimension of the latent space
    lambda_auto          : regularization parameter to ponderate autoencoder loss function
    loss_type            : loss function {"CVaR","MSE","SMSE"}
    lr                   : learning rate of the Adam optimizer
    dropout_par          : dropout regularization parameter 
    transaction_cost     : Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion         : CVaR confidence level (0,1)
    epochs               : Number of epochs, training iterations
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
    display_plot         : Display plot of training and validation loss 
    penalization         : Penalization weight for the local CVaR

    Output:
    loss_train_epoch : Loss history per epochs

    """

    owd = os.getcwd()
    try:
        os.chdir(os.path.join(main_folder, f"models"))

        if hedging == "allocation":
            name = f"Deep_agent_{network}_dropout_{str(int(dropout_par*100))}_allocation_{id}"
        else:
            name = f"Deep_agent_{network}_dropout_{str(int(dropout_par*100))}_contribution_{id}"

        
        # Compile the neural network
        rl_network = DeepAgent(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging, penalization, name)
                             
        print("-------------------------------------------------------------")
        print(name)
        print("-------------------------------------------------------------")

        # Start training
        start = dt.datetime.now()
        print('---Training start---')
        with tf.Session() as sess:
            rl_network.restore(sess, f"{name}.ckpt")
            loss_train_epoch = rl_network.train_deephedging(train_input, gl_va_train, gl_va_test, HO_train, HO_test, contribution_train, contribution_test, riskaversion, test_input, sess, epochs)
        print('---Training end---')

        if display_plot == True:
            loss_type = "CVaR" if hedging == "allocation" else "MSE"
            # Plot the learning curve on the train set, i.e. the CVaR on the train set at the end of each epoch
            lin_nb_epoch = np.linspace(1, loss_train_epoch.shape[0], loss_train_epoch.shape[0])
            fig = plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,0],label="Train error")
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,1],label="Test error")
            plt.title(f"Loss function: {loss_type}")
            plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            plt.show()
    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)
    
    return loss_train_epoch


def network_inference(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging,
                   train_input, gl_va_train, gl_va_test, HO_train, HO_test,
                     contribution_train, contribution_test, riskaversion, test_input, epochs, display_plot, id, penalization):
    
    # Function to test deep hedging algorithm
    """
    Input:
    network              : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj       : time steps 
    batch_size           : batch size {296,1000}
    nbs_input            : number of features
    nbs_units            : neurons per layer/cell
    nbs_assets           : number of hedging intruments
    constraint_min       : minimum value of the portfolio for soft constraint
    lambda_m             : labmda for lagrange multiplier (soft constraint)
    autoencoder          : autoencoder component included
    nbs_latent_spc       : dimension of the latent space
    lambda_auto          : regularization parameter to ponderate autoencoder loss function
    loss_type            : loss function {"CVaR","MSE","SMSE"}
    lr                   : learning rate of the Adam optimizer
    dropout_par          : dropout regularization parameter 
    transaction_cost     : Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion         : CVaR confidence level (0,1)
    epochs               : Number of epochs, training iterations
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
    display_plot         : Display plot of training and validation loss
    penalization         : Penalization weight for the local CVaR

    Output:
    deltas           : Position in the hedging instruments

    """
    
    owd = os.getcwd()

    os.chdir(os.path.join(main_folder, f"models"))

    if hedging == "allocation":
        name = f"Deep_agent_{network}_dropout_{str(int(dropout_par*100))}_allocation_{id}"
    else:
        name = f"Deep_agent_{network}_dropout_{str(int(dropout_par*100))}_contribution_{id}"

    
    # Compile the neural network
    rl_network = DeepAgent(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lr, dropout_par, hedging, penalization, name)
                            
    print("-------------------------------------------------------------")
    print(name)
    print("-------------------------------------------------------------")

    # Start training
    start = dt.datetime.now()
    print('---Inference start---')
    with tf.Session() as sess:
        rl_network.restore(sess, f"{name}.ckpt")
        portfolio, deltas = rl_network.predict(test_input, gl_va_test, HO_test, contribution_test, riskaversion, sess)
        os.chdir(owd)
        os.chdir(os.path.join(main_folder, f"data/results"))
        np.save(f"{name}",deltas[:,:,:])

    print('---Inference end---')
    os.chdir(owd)

    return portfolio, deltas, name
