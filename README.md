## Hedging targeted risks with reinforcement learning: application to life insurance contracts with embedded guarantees

This repository contains the deep hedging environment used in our paper, CO Pérez-Mendoza and F Godin (2025), where we develop a deep reinforcement learning (RL) framework for hedging financial instruments by focusing on specific, user-defined sources of risk—referred to as targeted risks. The approach is grounded in Shapley value decompositions, which attribute profit and loss to individual risk factors based on their impact on projected cash flows. Using this decomposition, we train deep RL agents to hedge only the targeted risks, leaving non-targeted exposures unaffected. To further enhance performance, we incorporate a joint network architecture where the hedging agent leverages local risk signals from a Conditional Value at Risk (CVaR) estimation network. The repository consists of three main components:

- Component 1: Environment generation based on the data-driven simulator borrow from Godin et al. (2023).
- Component 2: Implementation of an artificial neural network to estimate one-day-ahead CVaR for projected cash flows.
- Component 3: Implementation of the RL agent to hedge the risk—referred to as targeted risks.

## Short description

1. The environment simulators, component 1, are contained in the `src/features/` folder. 

    - `market_simulator.py` simulates the market environment using the stochastic dynamics proposed by Godin et al. (2023).

2. The CVaR network pipeline, component 2, is contained in the `src/models/` folder. 

    - `cvar_agent_training.py` trains the neural network to estimate the one-day-ahead CVaR based on the simulated environment.

2. Deep RL model, component 3, is contained in the `src/models/` folder. 

    - `deep_rl_training.py` contains all model functionalities through a python class that trains and assesses the performance of RL agents based on the FFNN architecture outlined in our paper.

Examples showcasing the utilization of the pipeline can be observed in the notebooks directory.
The accompanying Python script follows the notebook's structure to execute the full pipeline.

## How to run

1. **Prerequisities**
    - Python 3.9.6 was used as development environment.
    - The latest versions of pip

2. **Environment setup**

- Clone the project repository:

```nohighlight
git clone https://github.com/cpmendoza/deep_targeted_hedging-.git
cd deep_targeted_hedging-
```

- Create and activate a virtual environment:

```nohighlight
python -m venv venv
source venv/bin/activate
```

- Install the requirements using `pip`

```nohighlight
pip install -r requirements.txt
```

- Alternatively, start with an empty virtual environment and install packages during execution on as-required basis.

3. **Modify parameters**: The default parameters can be modified in the configuration files located in the cfgs folder:

- `config_market_simulator.yml`: General parameters for the simulation.

- `config_pricing_agent.yml`: Hyperparameters of the ANN to estimate the variable annuity price.

- `config_cvar_agent.yml`: Hyperparameters of CVaR ANN.

- `config_agent.yml`: Hyperparameters of the RL optimization problem.

4. **Running the script**: We provide two options to run the deep targeted hedging pipeline:

- Option 1.  The two main components of the pipeline can be executed independently by following the examples provided in `cvar_pipeline.ipynb` and `deep_hedging_pipeline.ipynb`, both located in the `notebooks` folder. Please execute them in this order, as the deep agent requires the CVaR ANN for training. These notebooks already outline the training of the CVaR ANN and the performance metrics of the RL-CVaR agent (without local risk penalization), as shown in Table 3 of our paper F. Godin (2025).

- Option 2. As an alternative to running the notebooks, the pipeline can be executed directly from the terminal by running the corresponding functions in each `.py` file, as indicated within the scripts.

## Directory structure

```nohighlight
├── LICENSE
├── README.md                   <- The top-level README for this project.
├── cfgs                        <- Configuration files for environment simulation and RL model parameters.
│
├── data
│   ├── raw                     <- Historical estimated parameters of mortality structure.
│   ├── interim                 <- Intermediate files used for preprocessing.
│   ├── processed               <- Simulated market environment.
│   └── results                 <- Deep hedging strategies (RL agents output).
│
├── notebooks                   <- Jupyter notebook with pipeline examples.
│
├── models                      <- Folder to store trained RL agents.
│
├── src                         <- Source code for use in this project.
│   │
│   ├── data                         <- Scripts to download and generate data.
│   │   ├── data_loader_va_price.py  <- Script to transform data into the right format for the pricing agent.
│   │   ├── data_loader_cvar.py      <- Script to transform data into the right format for the cvar ANN.
│   │   └── data_loader.py           <- Script to transform data into the right format for the Deep RL model.
│   │
│   ├── features                     <- Scripts to generate market environment.
│   │   ├── risk_free_rate.py        <- Script to generate risk-free rate structure.
│   │   ├── equity_risk_factor.py    <- Script to generate underlying funds.
│   │   ├── future_contracts.py      <- Script to generate hedging instruments.
│   │   ├── mortality_risk_factor.py <- Script to generate mortality simulation.
│   │   ├── va_valuation.py          <- Script to generate va price.
│   │   ├── market_simulator.py      <- Script to generate market environment.
│   │   └── cvar_estimation.py       <- Script to generate cvar estimation.
│   │
│   ├── models                          <- Scripts to train models and then use trained models to make
│   │   │                                  hedging strategies.
│   │   ├── princing_agent.py           <- Script create ANN as class objects.
│   │   ├── princing_agent_training.py  <- Script to fit and make inference.
│   │   ├── cvar_agent.py               <- Script create ANN as class objects.
│   │   ├── cvar_agent_training.py      <- Script to fit and make inference.
│   │   ├── deep_rl_agent.py            <- Script create RL agents as class objects.
│   │   └── deep_rl_training.py         <- Script to fit and make inference.
│   │
│   ├── visualization              <- Scripts to compute performance metrics of the models.
│   │   └── strategy_evalution.py  <- Scripts to compute performance metrics.
│   │
│   └── utils.py                   <- data utility for configuration files.
│ 
└── requirements.txt               <- The file for reproducing the pip-based virtual environment.
```
