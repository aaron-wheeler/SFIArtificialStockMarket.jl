# Sante Fe Institute Artificial Stock Market

## General description
The SFI artificial stock market is a model of an single-asset financial market that assumes heterogeneous agents follow simple decision-making rules to endogenously reproduce empirical market phenomena. 

## Model description

### Data structure

Multiple data structures have been defined in `ABM/data_struct.jl` to organise the different variables and their domains according to the theoretical outline. The values of all properties were obtained directly from the original SFI artificial stock market unless otherwise stated. 

1. A properties mutable struct (`ModelProperties`) defines several model parameters and variables:
  - `num_agents` : number of agents (& shares of risky asset) in the model (default = `25`)
  - `λ` : degree of agents' risk-aversion (default = `0.5`)
  - `num_predictors` : number of conditioned predictors each agent employs at a time (default = `100`)
  - `t`: current time step in simulation (default = `1`)
  - `state_vector` : vector for tracking the state of the market; updated at time `t` and observed by all agents
  - `price` : price vector of risky asset; updated at time `t` (new clearing price set by total demand) and observed by all agents
  - `dividend`: dividend vector of risky asset; updated at time `t` and observed by all agents
  - `trading_volume` : volume vector for total demand of risky asset; updated at time `t`
  - `volatility` : volatility vector calculated using clearing price of risky asset; updated at time `t`
  - `initialization_t` : number of time steps for initialization period of model (default = `499`)
  - `generalization_t` : number of inactive time steps until generalization is invoked (default = `4000`)
  - `k` : frequency of learning for agents across the simulation; defined as a average number of time steps and determined by market regime (Complex regime: `k` = `250`, Rational regime: `k` = `1000`)
  - `pGAcrossover`: crossover for genetic algorithm (probability of recombination); determined by market regime
  - `τ`: time horizon length for accuracy-updating of predictor; determined by market regime
  - `r` : constant interest rate of risk-free bond (default = `0.1`)
  - `ρ` : autoregressive parameter for dividend process (default = `0.95`) 
  - `d̄` : baseline constant for dividend process (default = `10.0`)
  - `σ_ε` : error-variance for dividend process (default = `0.0743`)
  - `σ_pd` : rational expectation price-plus-dividend variance (default = `4.0`)
  - `a_min` : minimum value of forecasting parameter a (default = `0.7`)
  - `a_max` : maximum value of forecasting parameter a (default = `1.2`)
  - `b_min` : minimum value of forecasting parameter b (default = `-10.0`)
  - `b_max` : maximum value of forecasting parameter b (default = `19.002`)
  - `C`: cost levied for fitness measure's specificity (default = `0.005`)
  - `price_min` : minimum price for risky asset (default = `0.01`) # from 2008 textbook
  - `price_max` : maximum price for risky asset (default = `200.0`) # from 2008 textbook
  - `init_cash` : initial cash balance of each agent (default = `20000.0`)
  - `eta` : parameter used to help adjust price to balance supply & demand (default = `0.0005`)
  - `min_excess` : excess demand must be smaller than this to stop price adjustment process (default = `0.01`)
  - `max_rounds` : maximum allowable iterations for the specialist to adjust trading conditions for clearing price (default = `20`)
  - `trade_restriction` : trading restriction per period (default = `10.0`)
  - `short_restriction` : shorting restriction per period (default = `-5.0`)
  - `cash_restriction` : minimum cash allowed at any time period for each agent (default = `-2000.0`)
  - `num_elimination` : number of predictors to be replaced with every GA implementation (default = `20`)
  - `pcond_mut` : probability that a condition bit will be flipped in GA mutation procedure (default = `0.03`)
  - `pparam_mut_long` : probability that a forecasting param is pulled from wide dist in GA mutation (default = `0.2`)
  - `pparam_mut_short` : probability that a forecasting param is pulled from short dist in GA mutation (default  = `0.2`)
  - `percent_mut_short` : percent difference of current value to use for short dist range in GA mutation (default= `0.05`)
  - `mdf_price` : price variable for data collection & analysis (default= `0.0`)
  - `mdf_dividend` : dividend variable for data collection & analysis (default= `0.0`)
  - `track_bits` : boolean parameter that determines whether models tracks and stores `frac_bits...` variables (needed for for data collection & analysis, but comes with significant computational cost)
  - `frac_bits_set` : average number of total bits set across all agents and predictors (default= `0.0`)
  - `frac_bits_fund` : average number of fundamental bits set across all agents and predictors (default= `0.0`)
  - `frac_bits_tech` : average number of technical bits set across all agents and predictors (default= `0.0`) 

2. An agent mutable struct (`Trader`) defines the agent variables:
  - `id`: unique identifier for each agent i
  - `pos`: defines agents' position on a grid space as a Tuple{Int,Int}. Currently unused in this simulation. 
  - `relative_cash`: each agent's relative cash held at time `t`
  - `relative_holdings`: each agent's relative number of shares held at time `t` (default = `1.0`)
  - `relative_wealth`: each agent's relative wealth (value of cash + holdings) at time `t`
  - `predictors` : The set of evolving conditional predicting vectors that are unique to each agent; agents use these to forecast price and dividend
  - `predict_acc`: the accuracy of agent i's jth predictor (most accurate is used); updated each time predictor j is active 
  - `fitness_j`: fitness measure for selecting which predictors undergo recombination in genetic algorithm
  - `chosen_j`: the index of the predictor an agent has selected to forecast price and dividend (default = `100`)
  - `demand_xi`: an agent's demand for holding shares of risky asset; the x number of shares that an agent wishes to hold at the end of the timestep
  - `active_predictors` : each agent's vector for tracking which predictors j are active based on the current market state
  - `forecast`: each agent's vector for determining which predictor j will be used to make forecasts
  - `active_j_records`: each agent's matrix that contains information about each predictor's historical use.


### The model 

The initialization and stepping functions of the model are described in [`ABM/model.jl`](model.jl). The separate module [`SFIArtificialStockMarket.jl`](SFIArtificialStockMarket.jl) contains the isolated functions for initialization and model stepping, in addition to functions for order execution and agent evolution.

After an initialization period `initialization_t`, the model undergoes a warm-up period `pre_SS_t` and then the outputs are recorded for the next `recorded_t` time steps and can be used for analysis.

#### Core model 

The ABM model is defined by the `init_model` function.

This model is composed of the following elements:
 - `Trader` : the agent struct (see [Data Structure, 2](#Data-structure));
 - `space`: agents are located on a periodic 10x10 GridSpace; (*this model element is currently unused in this simulation*)
 - `properties` : the parameters and variables of the model (see [Data Structure, 1](#Data-structure));
 - `scheduler` : agents are activated by their sequential id (`Schedulers.by_id`); 
 - `rng` : stores a seeded random number generator to be used in the model.

 The functions `init_state!` and `init_agents!` are then called to complete the initialization procedure. 

#### Initializing Market State

- `init_state!`: 

  This function defines the creation of the market state at step 0.
  The `price` and `dividend` values are initialized using the parameters from `properties` and used to generate the first market state vector `state_vector`.

#### Initializing Agents

- `init_agents!`: 

  This function defines the creation of agents at step 0. At this stage of the simulation, a key assumption is that each agent holds `1.0` share and devotes the same level of confidence to each predictor.

  Agents create and instantiate all agent variables used in the step function.  

#### Stepping function

- `model_step!`:

  This function defines what happens in the model during each time step, which includes updating model and agent variables and the order execution, genetic algorithm, and generalization procedures. 


The simulations of the model are run by executing the [`ABM/run.jl`](run.jl) file. This file is also where the `num_agents`, `track_bits`, and market regime (`k`, `pGAcrossover`, `τ`) parameters are defined.

According the market regime, the genetic algorithm will be invoked: 
  - Complex regime (fast learning; default): every `k = 250` periods on average; with parameters `pGAcrossover =  0.1, τ = 75`
  - Rational regime (slow learning): every `k = 1000` periods on average; with parameters `pGAcrossover =  0.3, τ = 150`

The aforementioned script [`ABM/run.jl`](run.jl)

- sets the values of model parameters,
- creates the ABM model as defined in the project files,
- runs the model over `model_runs` (default = `260,000`) number of time steps
  - (`pre_SS_t` (default = `250,000`) warm-up steps and `recorded_t` (default = `10,000`) recorded steps)
and does this same process `num_model_ensembles` (default = `1`) number of times under different random seeds and aggregates the results over the repetitions,
- and collects and saves the generated agent (`adata`) and model (`mdata`) data.

Data and plots are saved locally. File names are static and will overwrite existing files with same name (e.g. data from previous runs of the model will be overwritten).