# Sante Fe Institute Artificial Stock Market

## General description
The SFI artificial market is a simple model of an asset market that assumes heterogeneous agents who follow inductive reasoning and endogenously reproduce empirical market phenomena. 

## Model description

### Data structure

Multiple data structures have been defined in `ABM/data_struct.jl` to organise the different variables and their domains according to the theoretical outline.

1. A properties struct (`ModelProperties`) defines several model parameters:
  - `num_agents` : number of agents in the model (default = `25`)
  - `λ` : degree of agents' risk-aversion (default = `0.5`)
  - `num_predictors` : number of conditioned predictors each agent employs at a time (default = `100`)
  - `bit1`: fundamental bit; Price * interest/dividend > 0.25 (`State.bit1`)
  - `bit2`: fundamental bit; Price * interest/dividend > 0.50 (`State.bit2`)
  - `bit3`: fundamental bit; Price * interest/dividend > 0.75 (`State.bit3`)
  - `bit4`: fundamental bit; Price * interest/dividend > 0.875 (`State.bit4`)
  - `bit5`: fundamental bit; Price * interest/dividend > 1.00 (`State.bit5`)
  - `bit6`: fundamental bit; Price * interest/dividend > 1.125 (`State.bit6`)
  - `bit7` : technical bit; Price > 5-period moving average of past prices (MA) (`State.bit7`)
  - `bit8` : technical bit; Price > 10-period MA (`State.bit8`)
  - `bit9` : technical bit; Price > 100-period MA (`State.bit9`)
  - `bit10` : technical bit; Price > 500-period MA (`State.bit10`)
  - `bit11`: experimental control; always on: 1 (`State.bit11`)
  - `bit12`: experimental control; always off: 0 (`State.bit12`)
  - `warm_up_t` : number of time steps for initial warm-up period of model (default = `250,000`)
  - `κ` : frequency of learning for agents across the simulation; average number of time steps determined by market regime
  - `regime` : the market regime can be either 
    - `Complex`, i.e. all agents continually explore prediction space at fast (realistic) rates;
    - `Rational`, i.e. all agents continually explore prediction space at slow rates;
  - `num_shares` : number of shares of risky asset (default = `25`)
  - `r` : constant interest rate of risk-free bond (default = `0.1`)
  - `ρ` : autoregressive parameter for dividend process (default = `0.95`) 
  - `d̄` : baseline constant for dividend process (default = `10.0`)
  - `ε` : gaussian noise term for dividend process (`~N(0,σ_ε)`)
  - `σ_ε` : error-variance for dividend process (default = `0.0743`)
  - `σ_pd` : price-plus-dividend variance in the h.r.e.e. (default = `4.0`)
  - `M`: constant for recombination fitness measure (default = `0.0`) (`Arbitrary`)
  - `C`: cost levied for fitness measure specificity (default = `0.005`)
  - `init_price` : initial price for risky asset (default = `X`) (**To do: Replace with min/max?**)
  - `init_dividend` : initial dividend for risky asset (default = `X`) (**To do: Replace with min/max?**)
  - `init_cash` : initial cash balance of each agent (default = `20000.0`)

2. A State struct (`State`) defines the varying parameters of the applied simulation treatments:
  - `bit1`: Price * interest/dividend > 0.25 , set to `1` signals occurence of state 
  - `bit2`: Price * interest/dividend > 0.50 , set to `1` signals occurence of state 
  - `bit3`: Price * interest/dividend > 0.75 , set to `1` signals occurence of state 
  - `bit4`: Price * interest/dividend > 0.875, set to `1` signals occurence of state 
  - `bit5`: Price * interest/dividend > 1.00 , set to `1` signals occurence of state 
  - `bit6`: Price * interest/dividend > 1.125, set to `1` signals occurence of state 
  - `bit7` : Price > 5-period moving average of past prices (MA), set to `1` signals occurence of state
  - `bit8` : Price > 10-period MA, set to `1` signals occurence of state
  - `bit9` : Price > 100-period MA, set to `1` signals occurence of state
  - `bit10` :  Price > 500-period MA , set to `1` signals occurence of state
  - `bit11`: always on: 1 (`State.bit11`)
  - `bit12`: always off: 0 (`State.bit12`) 
  - `price` : price vector of risky asset; updated at time t (new clearing price set by total demand) and observed by all agents
  - `dividend`: dividend vector of risky asset; updated at time t and observed by all agents
  - `volume` : volume vector using total demand for risky asset; updated at time t
  - `volatility` : volatility vector calculated using clearing price of risky asset; updated at time t
  - `technical_activity` : vector that stores the number of set technical trading bits; updated at time t 

  We distinguish over four thousand different market states in the simulation, a bit is "set" if it is `0` (no signal) or `1` (signal), `missing` otherwise. For example:
  
  - `"All_Fund_Ex"` : Only fundamental bits; `bit1 = 1, bit2 = 1, bit3 = 1, bit4 = 1, bit5 = 1, bit6 = 1, bit7 = 0, bit8 = 0, bit9 = 0, bit10 = 0, bit11 = 1, bit12 = 0`, i.e. a state where the market Price * interest/dividend ratio is larger than 1.125 and the Price is less than the 5-period MA
  - `"All_Tech_Ex"` : Only technical bits; `bit1 = 0, bit2 = 0, bit3 = 0, bit4 = 0, bit5 = 0, bit6 = 0, bit7 = 1, bit8 = 1, bit9 = 1, bit10 = 1, bit11 = 1, bit12 = 0`, i.e. a state where the market Price * interest/dividend ratio is smaller than 0.25 and the Price is more than the 500-period MA
  - `"Pred_Ex"` : An example predictor; `bit1 = 1, bit2 = 1, bit3 = missing, bit4 = 0, bit5 = 0, bit6 = 0, bit7 = 1, bit8 = 0, bit9 = 0, bit10 = 0, bit11 = 1, bit12 = 0`, i.e. this predictor would match a state with a Price * interest/dividend ratio that is larger than 0.50 and smaller than 0.875 and a Price that is more than the 5-period MA and less than the 10-period MA

3. An agent struct (`Trader`) defines the agent variables:
  - `id`: unique identifier for each agent
  - `relative_cash`: each agent's relative cash held (**To do: Also include profit, wealth (would require position var)?**)
  - `pos`: defines agents' position on a grid space as a Tuple{Int,Int} (**To do: For visualization, relate to wealth or holding status?**)
  - `predictors` : The set of evolving conditional predictors each agent uses to forecast price and dividend
  - `predict_acc`: the accuracy of agent i's jth predictor (most accurate is used); updated each time predictor j is active 
  - `fitness_j`: fitness measure for selecting which predictors for recombination in genetic algorithm
  - `expected_pd`: agent i's prediction j of next period's price and dividend; linear combination of current price and dividend
  - `demand_xi`: an agent's demand for holding shares of risky asset
  - `σ_i`: an agent's forcast of the conditional variance of price-plus-dividend
  - `δ` : degree of deviation from learning frequency (term used for GA selection) (**To do: determine if this is needed**)
  - `a`: linear forecasting parameter; uniform about [`0.7, 1.2`] 
  - `b`: linear forecasting parameter; uniform about [`-10.0, 19.002`]
  - `JX`: crossover for genetic algorithm (probability of recombination)
  - `τ`: relevant horizon length for accuracy-updating parameter for predictor 
  - `s`: fitness measure specificity; number of bits that are set in the predictor's condition array 

### The model 

The initialisation and stepping functions of the model are described in [`ABM/model.jl`](model.jl). The separate module [`evolution`](evolution.jl) defines agent behavior and agent learning via isolated functions and the genetic algorithm.

#### Core model 

The ABM model is defined by the `init_model` function.

Whatever the environment for market behavior as described by the `regime` property, the model is composed of the following elements:
 - `Trader` : the agent struct (see [Data Structure, 3](#Data-structure));
 - `State`  : the market struct (see [Data Structure, 2](#Data-structure)); 
 - `space`: agents are located on a periodic 10x10 GridSpace; (**To do: Update**)
 - `properties` : the parameters of the model (see [Data Structure, 1](#Data-structure));
 - `scheduler` : agents are activated at random (`Schedulers.randomly`); 
 - `rng` : stores a seeded random number generator to be used in the model.

The model is ran for an initial warm-up period (determined by `warm_up_t`) and then the outputs are recorded for the next 10,000 time steps and used for analysis. 

#### Initialising Agents

- `init_agents!`: 

  This function defines the creation of agents at step 0. 
  At this stage of the simulations, each agent holds 1 share and it is assumed that each agent devotes the same level of confidence to each predictor. 
  Agents create and instantiate all parameters used in the step function.  
  Agents also form their predictors (`evolution.init_predictors`).

  - `init_state!`: 

  This function defines the creation of the market state at step 0.
  The initial `price` and `dividend` is initialized using the parameters from `properties`. 

- `evolution.init_predictors`: 

  Agents' value types defined by their `predictors` depend on the value types distribution `model.dist` defined when creating the ABM model. This procedure will initialize forecasting parameters following a uniform distribution of values centered around h.r.e.e. ones. 
  - `a`: uniform range [`0.7, 1.2`] 
  - `b`: uniform range [`-10.0, 19.002`]
  The variance of all new predictors is intialized in all cases to (`σ_pd`)

- `evolution.init_learning`: 
  
  Each agent will replace the lowest performing 20% of predictors at random intervals, asynchronously across agents, every `κ` time steps on average based on the market regime. The predictors are replaced by using uniform crossover and mutation in the genetic algorithm (`evolution.GA!`). Each agent type will have a different deviation parameter (`δ`) to ensure learning does not occur for all agents in certain periods. The function defines a dictionary which associates to each `predictors` symbol the predictor accuracy (`predict_acc`), fitness measure (`fitness_j`), and respective `δ`. 

- `evolution.GA!`:

  According to the exploration rate of the market regime (`model.regime`), the genetic algorithm will be invoked: 
  - `Complex` : every `κ = 250` periods on average; `JX =  0.1, τ = 75`
  - `Rational` : every `κ = 1000` periods on average;  `JX =  0.3, τ = 150`

#### Stepping function

What happens in the model during each step is described in the `model_step!` function, which includes all the remaining functions defined in the file. Thanks to this specification, we ensure that each action is sequentially executed. 

The stepping function is split into two sections, the initial warm-up section, followed by the steady-state section. All of the following steps are performed in both sections, but the generated dataframes are only collected during the steady-state section. 

During each step of the simulation, all agents are randomly activated to act according to the following specifications (Note that not all of these variables are observed by the agents and thus, some do not affect agent behaviour at all):

1. Calculate each agents' expected output, `expected_pd` and `demand_xi` (`evolution.update_output!`). 
2. Sum the total demand and equate it to number of shares issued to determine and broadcast the new clearing price and dividend (i.e., simulate market specialist and price formation mechanism). Then perform following conditional action: 
  If simulation time < `warm_up_t`: 
    Calculate the realised_output, `dividend`, `price`, `volume`, `volatility`, and `technical_activity` (`evolution.update_realised_output!`). 
  Else:
    Calculate the realised_output, `dividend`, `price`, `volume`, `volatility`, and `technical_activity` (`evolution.update_realised_output!`). 
    Append each output to their respective vectors. The `dividend` and `price` process vectors are made public to the agents for the next time step.     
3. Update individual agents properties (i.e., assets held, cumulative wealth) (`evolution.update_rewards!`).

After these model calculations, some additional agent actions are done (or not done) at the end of the current model step:  

4. Update the predictor accuracy and fitness measure (`evolution.update_predict_acc!` and `evolution.update_fitness_j!`).
5. Use predictor accuracy, fitness measure, and `δ` to determine if selected for recombination. 
6. Update their error variance `σ_i`.
7. Undergo the genetic algorithm (`evolution.GA!`) or not based on aformentioned factors.

The simulations of the model are run by executing the [`ABM/run.jl`](run.jl) file. 

The aforementioned script

- defines the parameter structure of states,
- sets the values of model parameters,
- creates the ABM model as defined in the project files,
- runs the model 25 times over 260,000 steps (250,000 warm-up and 10,000 recorded) under different random seeds and aggregates the results over the repetitions, 
- and collects and saves the generated dataframes.

Data and plots are saved locally. File names are static and will overwrite existing files with same name (e.g. data from previous runs of the model will be overwritten).