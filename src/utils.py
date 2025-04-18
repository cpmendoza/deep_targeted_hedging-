import logging
import yaml
from tqdm import tqdm
import warnings
import time
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

class garch_neutral_dynamics(object):

    """Class to use GARCH neutral dynamics:
        -Paths simulation
        -Option pricing 
        -Delta computation 
    
    Parameters
    ----------
    config_file : simulation settings for the JIVR model and the underlying asset and 
    
    """
    
    def __init__(self,r ,mu, omega, alpha_1, beta_1, gamma_1):

        # Parameters
        self.r = r/252          #Daily risk-free rate
        self.q = 0.01772245/252     #Daily dividend yield
        self.mu = mu            #GARCH model parameter
        self.omega = omega      #GARCH model parameter
        self.alpha_1 = alpha_1  #GARCH model parameter
        self.beta_1 = beta_1    #GARCH model parameter
        self.gamma_1 = gamma_1  #GARCH model parameter

    def gjr_garch_simulation_neutral_measure(self, S, initial_variance, num_paths, num_steps):

        """Function that simulates market dynamics based on one of the two econometric models  

        Parameters
        ----------
        S: initial underlying asset price
        initial_variance : single value - Last fitted conditional variance
        num_paths : single value - Number of paths to be simulated
        num_steps : single value - Number of time-steps for the simulation 

        Returns
        -------
        returns : numpy array - simulated paths of the return based on the econometric model
        conditional_variance : numpy array - simulated paths of the conditional variance based on the econometric model

        """

        epsilon = 0.001
        conditional_variance = np.zeros((num_paths, num_steps+1))
        conditional_variance_epsilon = np.zeros((num_paths, num_steps+1))
        returns = np.zeros((num_paths, num_steps))
        returns_epsilon = np.zeros((num_paths, num_steps))

        #Initialize
        conditional_variance[:, 0] = initial_variance
        conditional_variance_epsilon[:, 0] = ((np.sqrt(initial_variance)+epsilon)**2)

        #New residual - risk free measure
        nabla_t = (self.mu-self.r+self.q+initial_variance/2)/np.sqrt(initial_variance)
        nabla_t_epsilon = (self.mu-self.r+self.q+((np.sqrt(initial_variance)+epsilon)**2)/2)/np.sqrt(((np.sqrt(initial_variance)+epsilon)**2))
        #generate modify shocks considering the risk netural measure
        normal_simulation = np.random.normal(0, 1, num_paths)
        z_t = normal_simulation-nabla_t
        z_t_epsilon = normal_simulation-nabla_t_epsilon
        #residual 
        last_residual = np.sqrt(initial_variance) * z_t
        last_residual_epsilon = np.sqrt(((np.sqrt(initial_variance)+epsilon)**2)) * z_t_epsilon

        
        for t in range(1, num_steps+1):
            # Calculate indicator function
            indicator =  (last_residual < 0)*1
            indicator_epsilon =  (last_residual_epsilon < 0)*1
            # Calculate conditional variance
            conditional_variance[:, t] = self.omega + (self.alpha_1 + self.gamma_1 * indicator) * (last_residual**2) + self.beta_1 * conditional_variance[:, t-1]
            conditional_variance[:, t] = np.minimum(np.maximum(conditional_variance[:, t],0.0001),0.004)
            conditional_variance_epsilon[:, t] = self.omega + (self.alpha_1 + self.gamma_1 * indicator_epsilon) * (last_residual_epsilon**2) + self.beta_1 * conditional_variance_epsilon[:, t-1]
            conditional_variance_epsilon[:, t] = np.minimum(np.maximum(conditional_variance_epsilon[:, t],0.0001),0.004)

            # Generate standardized random shocks
            nabla_t = (self.mu-self.r+self.q+conditional_variance[:, t]/2)/np.sqrt(conditional_variance[:, t])
            nabla_t_epsilon = (self.mu-self.r+self.q+conditional_variance_epsilon[:, t]/2)/np.sqrt(conditional_variance_epsilon[:, t])
            normal_simulation = np.random.normal(0, 1, num_paths)
            z_t = normal_simulation-nabla_t
            z_t_epsilon = normal_simulation-nabla_t_epsilon
            # Noise 
            last_residual = np.sqrt(conditional_variance[:, t]) * z_t
            last_residual_epsilon = np.sqrt(conditional_variance_epsilon[:, t]) * z_t_epsilon
            # Generate returns
            returns[:, t-1] = self.mu + last_residual
            returns_epsilon[:, t-1] = self.mu + last_residual_epsilon

        # Simulated paths 
        S_0 = S
        Stock_paths = S_0*np.cumprod(np.exp(returns),axis=1)
        Stock_paths = np.insert(Stock_paths,0,S_0,axis=1)

        # Simulated paths with epsilon
        epsilon = 0.1
        S_0_epsilon = S + epsilon
        Stock_paths_epsilon = S_0_epsilon*np.cumprod(np.exp(returns),axis=1)
        Stock_paths_epsilon = np.insert(Stock_paths_epsilon,0,S_0_epsilon,axis=1)

        # Simulated paths with volatility alteration
        S_0 = S
        Stock_paths_vol = S_0*np.cumprod(np.exp(returns_epsilon),axis=1)
        Stock_paths_vol = np.insert(Stock_paths_vol,0,S_0,axis=1)

        return Stock_paths, Stock_paths_epsilon, Stock_paths_vol 
    
    def option_price(self, S):

        """Function that compute the option price based on Monte-Carlo simulation 

        Parameters
        ----------
        S: numpy array - Simulated paths of the underlying asset

        Returns
        -------
        option_price : single value
        """

        t = S.shape[1]-1
        option_price = (np.maximum(S[:,-1]-100,0)*np.exp(-self.r*t/252)).mean()
        return option_price
    
    def delta_garch(self, S, initial_variance, strike, num_paths, num_steps):

        """Function that compute delta at a given state

        Parameters
        ----------
        S: numpy array - initial_value
        num_paths : single value - Number of paths to be simulated
        num_steps : single value - Number of time-steps for the simulation 

        Returns
        -------
        delta : single value - Position in the underlying asset
        """
        
        Stock_paths, Stock_paths_epsilon, Stock_paths_vol = self.gjr_garch_simulation_neutral_measure(S, initial_variance, num_paths, num_steps)
        t = Stock_paths.shape[1]-1
        delta = np.exp(-t*self.r)*((Stock_paths[:,-1]/Stock_paths[:,0])*(Stock_paths[:,-1]>strike)).mean()
        delta_epsilon = np.exp(-t*self.r)*((Stock_paths_epsilon[:,-1]/Stock_paths_epsilon[:,0])*(Stock_paths_epsilon[:,-1]>strike)).mean()

        #Option price 
        price = self.option_price(Stock_paths)
        price_vol = self.option_price(Stock_paths_vol)

        return delta, delta_epsilon, price, price_vol
    
    def vectorial_delta(self, Stock_paths, initial_variance, strike, num_paths, num_steps):

        """Function that compute delta at a given state for a vector

        Parameters
        ----------
        Stock_paths: numpy array - vector of underlying asset prices
        initial_variance: numpy array - vector of underlying asset conditional variances
        strike: Strike price
        num_paths : single value - Number of paths to be simulated
        num_steps : single value - Number of time-steps for the simulation 

        Returns
        -------
        delta : numpy array - vector of estimated deltas
        """

        delta_1 = np.nan
        result = list()
        result_epsilon = list()
        price = list()
        price_vol = list()
        for element in tqdm(range(Stock_paths.shape[0])):
            attempt = 0
            while attempt < 3:
                try:
                    delta_1, delta_1_epsilon, price_1, price_1_vol = self.delta_garch(Stock_paths[element],initial_variance[element], strike, num_paths, num_steps)
                    break
                except Exception as e:
                    attempt += 1
            result.append(delta_1)
            result_epsilon.append(delta_1_epsilon)
            price.append(price_1)
            price_vol.append(price_1_vol) 
        
        return result, result_epsilon, price, price_vol
    
    def array_delta(self, Stock_paths, conditional_volatility, strike, num_paths_estimation):

        volatility = (conditional_volatility**2)/252
        deltas = np.zeros((Stock_paths.shape[0],Stock_paths.shape[1]-1))
        deltas_epsilon = np.zeros((Stock_paths.shape[0],Stock_paths.shape[1]-1))
        option_price = np.zeros((Stock_paths.shape[0],Stock_paths.shape[1]-1))
        option_price_vol = np.zeros((Stock_paths.shape[0],Stock_paths.shape[1]-1))
        time_steps = Stock_paths.shape[1]-1
        for time_step in range(time_steps):
            print(f"---- Computation at step {time_step} ----")
            num_steps = time_steps-time_step
            delta_t, delta_t_epsilon, price, price_vol = self.vectorial_delta(Stock_paths[:,time_step], volatility[:,time_step], strike, num_paths_estimation, num_steps)
            deltas[:,time_step] = delta_t
            deltas_epsilon[:,time_step] = delta_t_epsilon
            option_price[:,time_step] = price
            option_price_vol[:,time_step] = price_vol
        
        return deltas, deltas_epsilon, option_price, option_price_vol
    
def strategy_evaluation(portfolio):

    gl_tot_list = portfolio

    time = gl_tot_list[:,7]
    rate = 0.333333333*(gl_tot_list[:,3]-gl_tot_list[:,7])+0.166666667*((gl_tot_list[:,1]-gl_tot_list[:,5])+(gl_tot_list[:,2]-gl_tot_list[:,6]))+0.333333333*(gl_tot_list[:,0]-gl_tot_list[:,4])
    equity = 0.333333333*(gl_tot_list[:,5]-gl_tot_list[:,7])+0.166666667*((gl_tot_list[:,1]-gl_tot_list[:,3])+(gl_tot_list[:,4]-gl_tot_list[:,6]))+0.333333333*(gl_tot_list[:,0]-gl_tot_list[:,2])
    mortality = 0.333333333*(gl_tot_list[:,6]-gl_tot_list[:,7])+0.166666667*((gl_tot_list[:,2]-gl_tot_list[:,3])+(gl_tot_list[:,4]-gl_tot_list[:,5]))+0.333333333*(gl_tot_list[:,0]-gl_tot_list[:,1])
    total = time + rate + equity + mortality

    hedging_err = (-1*(total)) 
    index_cvar = np.argsort(hedging_err)[int(0.95*hedging_err.shape[0]):]

    results = pd.DataFrame([[-hedging_err.mean(),equity.mean(),rate.mean(),mortality.mean(),time.mean()],
                            #[np.cov(hedging_err,hedging_err)[0,1],np.cov(-equity,hedging_err)[0,1],np.cov(-rate,hedging_err)[0,1],np.cov(-mortality,hedging_err)[0,1],np.cov(-time,hedging_err)[0,1]],
                            [np.mean(hedging_err[index_cvar]),np.mean(-equity[index_cvar]),np.mean(-rate[index_cvar]),np.mean(-mortality[index_cvar]),np.mean(-time[index_cvar])],
                            [np.cov(total,total)[0,1], np.cov(total,equity)[0,1], np.cov(total,rate)[0,1], np.cov(total,mortality)[0,1]]])

    results.columns = ["Total","Equity","Rate","Mortality","Time"]
    results.index = ["Expectation","CVaR:95%","Variance"]
    results = results.fillna(0)
    print(results.T)

    return
    
