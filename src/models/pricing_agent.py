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
from tensorflow.python.framework import ops

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
    def __init__(self, batch_size, nbs_input, nbs_units, lr, dropout_par, name):
        tf.compat.v1.disable_eager_execution()
        ops.reset_default_graph()

        # 0) Deep hedging parameters parameters
        self.batch_size     = batch_size
        self.nbs_units      = nbs_units
        self.lr             = lr
        self.dropout_par    = dropout_par

        # 3) Placeholder for deep hedging elements    
        self.input                = tf.placeholder(tf.float32, [batch_size, nbs_input])      # normalized prices and features
        self.output               = tf.placeholder(tf.float32, [batch_size,1]) 

        layer_1 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
        layer_drop_1 = tf.keras.layers.Dropout(self.dropout_par)
        layer_2 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
        layer_drop_2 = tf.keras.layers.Dropout(self.dropout_par)
        layer_3 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
        layer_drop_3 = tf.keras.layers.Dropout(self.dropout_par)
        layer_4 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
        layer_drop_4 = tf.keras.layers.Dropout(self.dropout_par)
        # 5.3.2) Output layer of dimension one (outputs the position in the underlying)
        layer_out = tf.layers.Dense(1, None) 

        # Forward propagation
        layer = layer_1(self.input)
        layer = layer_drop_1(layer)
        layer = layer_2(layer)
        layer = layer_drop_2(layer)
        layer = layer_3(layer)
        layer = layer_drop_3(layer)
        layer = layer_4(layer)
        layer = layer_drop_4(layer)
        layer = layer_out(layer)   #[batch_size, 1]

        self.predicted = layer 

        self.hedging_err = tf.square(layer-self.output)
        self.loss = tf.reduce_mean(self.hedging_err)

        # 8) SGD step with the adam optimizer
        optimizer  = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train = optimizer.minimize(self.loss)

        # 9) Save the model
        self.saver      = tf.train.Saver()
        self.model_name = name   # name of the neural network to save

    # Function to compute the CVaR_{alpha} outside the optimization, i.e. at the end of each epoch in this case
    def loss_out_optim(self, hedging_err):
        loss_t = np.mean(np.square(hedging_err))
        return loss_t

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
    def train_deephedging(self, train_input, train_output, test_input, test_output, sess, epochs):

        sample_size       = train_input.shape[0]               # total number of paths in the train set
        sample_size_valid = test_input.shape[0]
        batch_size        = self.batch_size
        idx               = np.arange(sample_size)       # [0,1,...,sample_size-1]
        idx_valid         = np.arange(sample_size_valid)
        start             = dt.datetime.now()            # Time-to-train
        self.loss_epochs  = 9999999*np.ones((epochs,6))      # Store the loss at the end of each epoch for the train
        valid_loss_best   = 999999999
        epoch             = 0

        # Loop for each epoch until the maximum number of epochs
        while (epoch < epochs):
            hedging_err_train = []  # Store hedging errors obtained for one complete epoch
            hedging_err_valid = []
            np.random.shuffle(idx)  # Randomize the dataset (not useful in this case since dataset is simulated iid)


            # loop over each batch size
            for i in range(int(sample_size/batch_size)):

                # Indexes of paths used for the mini-batch
                indices = idx[i*batch_size : (i+1)*batch_size]

                # SGD step
                _, hedging_err = sess.run([self.train, self.hedging_err],
                                               {self.input                 : train_input[indices,:],
                                                self.output                : train_output[indices,:]})


                hedging_err_train.append(hedging_err)

            # 2) Evaluate performance on the valid set - we don't train
            for i in range(int(sample_size_valid/batch_size)):
                indices_valid = idx_valid[i*batch_size : (i+1)*batch_size]
                hedging_err_v = sess.run([self.hedging_err],
                                               {self.input                 : test_input[indices_valid,:],
                                                self.output                : test_output[indices_valid,:]})

                hedging_err_valid.append(hedging_err_v)


            # 3) Store the loss on the train and valid sets after each epoch
            self.loss_epochs[epoch,0] = self.loss_out_optim(np.concatenate(hedging_err_train))
            self.loss_epochs[epoch,1] = self.loss_out_optim(np.concatenate(hedging_err_valid)) #, axis=1

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
                print('  Train - %s: %.5f, Valid: %.5f' % ("MSE",
                                                            self.loss_epochs[epoch,0],self.loss_epochs[epoch,1]))
            
            epoch+=1  # increment the epoch

        # End of training
        print("---Finished training results---")
        print('Time elapsed:', dt.datetime.now()-start)

        # Return the learning curve
        return self.loss_epochs


    # Function which will call the deep hedging optimization batchwise
    def training(self, train_input, train_output, test_input, test_output, sess, epochs):
        sess.run(tf.global_variables_initializer())
        loss_train_epoch = self.train_deephedging(train_input, train_output, test_input, test_output, sess, epochs)
        return loss_train_epoch

    # ---------------------------------------------------------------------- #
    # Function to compute the hedging strategies of a trained neural network
    # - Doesn't train the neural network, only outputs the hedging strategies
    def predict(self, train_input, train_output, test_input, test_output, sess):
        sample_size = test_input.shape[0]
        batch_size  = self.batch_size
        idx         = np.arange(sample_size)  # [0,1,...,sample_size-1]
        strategy_pred = [] # hedging strategies
        portfolio_pred = []

        # loop over sample size to do one complete epoch
        for i in range(int(sample_size/batch_size)):

            # mini-batch of paths (even if not training to not get memory issue)
            indices = idx[i*batch_size : (i+1)*batch_size]
            predicted = sess.run([self.predicted],
                                    {self.input                 : test_input[indices,:],
                                     self.output                : test_output[indices,:]})

            # Append the batch of hedging strategies
            strategy_pred.append(predicted)

        return np.concatenate(strategy_pred,axis=1)

    def restore(self, sess, checkpoint):
        self.saver.restore(sess, checkpoint)


def train_network(batch_size, nbs_input, nbs_units, lr, dropout_par, train_input, train_output, test_input, test_output, epochs, display_plot, name):
    
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

    Output:
    loss_train_epoch : Loss history per epochs

    """
    
    owd = os.getcwd()
    try:
        os.chdir(os.path.join(main_folder, f"models"))

        name = f"Pricing_agent_FFNN_dropout_{str(int(dropout_par*100))}_pricing_va_{name}"

        # Compile the neural network
        rl_network = DeepAgent(batch_size, nbs_input, nbs_units, lr, dropout_par, name)
                             
        print("-------------------------------------------------------------")
        print(name)
        print("-------------------------------------------------------------")

        # Start training
        print("-----------------------Training start------------------------")
        with tf.Session() as sess:
            loss_train_epoch = rl_network.training(train_input, train_output, test_input, test_output, sess, epochs)
        print('---Training end---')

        if display_plot == True:
            # Plot the learning curve on the train set, i.e. the CVaR on the train set at the end of each epoch
            lin_nb_epoch = np.linspace(1, loss_train_epoch.shape[0], loss_train_epoch.shape[0])
            fig = plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,0],label="Train error")
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,1],label="Test error")
            plt.title(f"Loss function: MSE")
            plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            plt.show()
    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)
    #np.save(os.path.join(main_folder, f"data/results/Training/results_extra/loss_functions/loss_{name}.npy"),loss_train_epoch)

    return loss_train_epoch

def retrain_network(batch_size, nbs_input, nbs_units, lr, dropout_par, train_input, train_output, test_input, test_output, epochs, display_plot, name):

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

    Output:
    loss_train_epoch : Loss history per epochs

    """

    owd = os.getcwd()
    try:
        os.chdir(os.path.join(main_folder, f"models"))

        name = f"Pricing_agent_FFNN_dropout_{str(int(dropout_par*100))}_pricing_va_{name}"

        # Compile the neural network
        rl_network = DeepAgent(batch_size, nbs_input, nbs_units, lr, dropout_par, name)
                             
        print("-------------------------------------------------------------")
        print(name)
        print("-------------------------------------------------------------")

        # Start training
        print("-----------------------Training start------------------------")

        # Start training
        start = dt.datetime.now()
        print('---Training start---')
        with tf.Session() as sess:
            rl_network.restore(sess, f"{name}.ckpt")
            loss_train_epoch = rl_network.train_deephedging(train_input, train_output, test_input, test_output, sess, epochs)
        print('---Training end---')

        if display_plot == True:
            # Plot the learning curve on the train set, i.e. the CVaR on the train set at the end of each epoch
            lin_nb_epoch = np.linspace(1, loss_train_epoch.shape[0], loss_train_epoch.shape[0])
            fig = plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,0],label="Train error")
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,1],label="Test error")
            plt.title(f"Loss function: MSE")
            plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            plt.show()
    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)
    
    return loss_train_epoch


def network_inference(batch_size, nbs_input, nbs_units, lr, dropout_par, train_input, train_output, test_input, test_output, epochs, display_plot, name):
    
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

    Output:
    deltas           : Position in the hedging instruments

    """
    
    owd = os.getcwd()

    os.chdir(os.path.join(main_folder, f"models"))

    name = f"Pricing_agent_FFNN_dropout_{str(int(dropout_par*100))}_pricing_va_{name}"

    # Compile the neural network
    rl_network = DeepAgent(batch_size, nbs_input, nbs_units, lr, dropout_par, name)
                            
    print("-------------------------------------------------------------")
    print(name)
    print("-------------------------------------------------------------")

    # Start training
    start = dt.datetime.now()
    print('---Inference start---')
    with tf.Session() as sess:
        rl_network.restore(sess, f"{name}.ckpt")
        predicted = rl_network.predict(train_input, train_output, test_input, test_output, sess)
        os.chdir(owd)
        os.chdir(os.path.join(main_folder, f"data/results"))
        #np.save(f"{name}",deltas[:,:,:])

    print('---Inference end---')
    os.chdir(owd)

    return predicted