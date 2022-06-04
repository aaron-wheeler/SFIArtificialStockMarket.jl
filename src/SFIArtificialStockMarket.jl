module SFIArtificialStockMarket

using Agents
using Distributions
using Random
using StatsBase
using ForwardDiff
using DataFrames
using StaticArrays
using Statistics
using JuMP
using Ipopt
using Roots

## Update Market State 

"""
Update dividend vector

The dividend vector is mutated via an autoregressive dividend process and made public to all agents.
Gaussian noise term `ε` is independent & identically distributed and has zero mean and variance σ_ε.
"""
function dividend_process!(dividend, d̄, ρ, σ_ε)
    ε = rand(Normal(0.0,σ_ε))
    dt = d̄ + ρ*(last(dividend) - d̄) + ε
    dividend[1] = dividend[2]
    dividend[2] = dt
end

"""
Update the market state vector

The market state vector is mutated and made public to all agents. 

Assign "1" or "0" values depending on the presence of bit signals
- Signal present -> "1"
- Signal absent -> "0"

Each index (or "bit") of the market state vector indicates a specific state feature:
- `bit1`: fundamental bit; Price * interest/dividend > 0.25 
- `bit2`: fundamental bit; Price * interest/dividend > 0.50 
- `bit3`: fundamental bit; Price * interest/dividend > 0.75 
- `bit4`: fundamental bit; Price * interest/dividend > 0.875
- `bit5`: fundamental bit; Price * interest/dividend > 1.00 
- `bit6`: fundamental bit; Price * interest/dividend > 1.125
- `bit7` : technical bit; Price > 5-period moving average of past prices (MA)
- `bit8` : technical bit; Price > 10-period MA
- `bit9` : technical bit; Price > 100-period MA
- `bit10` : technical bit; Price > 500-period MA
- `bit11` : experimental control; always on: 1
- `bit12` : experimental control; always off: 0
"""
function update_market_vector!(state_vector, price, dividend, r)
    # Fundamental bits
    if last(price) * r / last(dividend) > 0.25
        state_vector[1] = 1
    else
        state_vector[1] = 0
    end

    if last(price) * r / last(dividend) > 0.5
        state_vector[2] = 1
    else
        state_vector[2] = 0
    end

    if last(price) * r / last(dividend) > 0.75
        state_vector[3] = 1
    else
        state_vector[3] = 0
    end

    if last(price) * r / last(dividend) > 0.875
        state_vector[4] = 1
    else
        state_vector[4] = 0
    end

    if last(price) * r / last(dividend) > 1.0
        state_vector[5] = 1
    else
        state_vector[5] = 0
    end

    if last(price) * r / last(dividend) > 1.125
        state_vector[6] = 1
    else
        state_vector[6] = 0
    end

    # Technical bits, the `period` in MA formula is set to 1 time step
    if last(price) > mean(@view price[(end-6):end])
        state_vector[7] = 1
    else
        state_vector[7] = 0
    end

    if last(price) > mean(@view price[(end-9):end])
        state_vector[8] = 1
    else
        state_vector[8] = 0
    end

    if last(price) > mean(@view price[(end-99):end])
        state_vector[9] = 1
    else
        state_vector[9] = 0
    end

    if last(price) > mean(@view price[(end-499):end])
        state_vector[10] = 1
    else
        state_vector[10] = 0
    end

    # Default bits, always on/off
    state_vector[11] = 1

    state_vector[12] = 0
end


## Initialization (done for each agent individually)

"""
    `init_predictors() → predictors`

Agent `predictors` vector is coupled to a unique vector of vectors, where the individual predictors are the 
innermost vectors and are formatted consistently for j = {1, 2,... num_predictors}:
    predictor_j[1:3] = [a_j, b_j, σ_j]
    predictor_j[4:15] = 0, 1, or missing by probability distribution
The final predictor is a default predictor that is matched in all market states. 
"""
function init_predictors(num_predictors, σ_pd, a_min, a_max, b_min, b_max)
    predictors = Vector{Any}(undef, num_predictors)
    for i = 1:(num_predictors-1) # minus one so that we can add default predictor
        bit_vec = Vector{Any}(undef, 15) # 15 total elements in bit vector
        # bit heterogeneity for elements [1:3]
        bit_vec[1] = rand(Uniform(a_min, a_max)) # a
        bit_vec[2] = rand(Uniform(b_min, b_max)) # b 
        bit_vec[3] = σ_pd # initial σ_i = σ_pd
        # bit indicators for elements [4:15]
        bits = @view bit_vec[4:15]
        indicator = @SVector [missing, 1, 0]
        p_indicator = @SVector [0.9, 0.05, 0.05] # TODO: Make weights non-hardcoded
        Distributions.sample!(indicator, Weights(p_indicator), bits) 
        predictors[i] = bit_vec
    end
    # default predictor
    default_bit_vec = Vector{Any}(missing, 15)
    # default heterogeneity
    default_bit_vec[1] = sum(predictors[i][1] * (1 / predictors[i][3])
                                   for i = 1:(num_predictors-1)) / sum(1 / predictors[i][3] for i = 1:(num_predictors-1)) # default a
    default_bit_vec[2] = sum(predictors[i][2] * (1 / predictors[i][3])
                                   for i = 1:(num_predictors-1)) / sum(1 / predictors[i][3] for i = 1:(num_predictors-1)) # default b 
    default_bit_vec[3] = σ_pd # initial default σ_i = σ_pd
    predictors[num_predictors] = default_bit_vec
    return predictors
end

"""
    `init_learning() → δ, predict_acc, fitness_j`

Constructs and initializes each agent's `predict_acc`, 'fitness_j`, and `δ` coupled to unique `id`.
- `δ` returned as vector, needed for asynch recombination  
- `predict_acc` returned as vector, associated to each predictor for agent `id`
- `fitness_j` returned as vector, associated to each `predict_acc` for agent `id`
"""
function init_learning(σ_pd, C, num_predictors, predictors)
    predict_acc = fill(σ_pd, num_predictors) # (σ_i), initialized as σ_pd(4.0) in first period
    fitness_j = Vector{Float64}(undef, num_predictors)
    for i = 1:num_predictors
        s = count(!ismissing, (@view predictors[i][4:15])) # `specificity`, number of bits that are "set" (not missing)
        fitness_j[i] = -1 * (predict_acc[i]) - C * s
    end

    return predict_acc, fitness_j
end


## Order Execution Mechanism 

"""
    `match_predictors() → active_predictors, forecast[a, b, σ_i, id]`

- Determine which predictors are active based on market state, return vector of indices for `active_predictors`
- `active_predictors` vector needed for agent predictor updating steps
- Among the active predictors, select the one with the highest prediction accuracy. If multiple:
    - tiebreaker 1 -> highest fitness measure
    - tiebreaker 2 -> default predictor (j=100)
- From this predictor, return vector composed of a, b, σ_i, and agent ID for `forecast`
- `forecast` vector needed for later aggregation and sending through demand function
"""
function match_predictors(id, num_predictors, predictors, state_vector, predict_acc, fitness_j, σ_pd)
    # Initialize vector to store indices of all active predictors
    active_predictors = Int[]

    for j = 1:num_predictors
        # reset predictor_j_match with each iteration
        predictor_j_match = 0
        j_id = j

        # make nested function call to see if predictor[j] bits match the state vector
        bit_j = @view predictors[j][4:15]
        predictor_j_match = match_predict(bit_j, state_vector)

        # if predictor[j] meets match criteria, append to active_predictors 
        if predictor_j_match == 12
            active_predictors = push!(active_predictors, j_id)
        else
            nothing
        end
    end

    matched_collection = zeros(Float64, 3, length(active_predictors))

    for (index, value) in pairs(IndexStyle(active_predictors), active_predictors)
        matched_collection[1, index] = value
        matched_collection[2, index] = predict_acc[value]
        matched_collection[3, index] = fitness_j[value]
    end

    row, col = size(matched_collection)
    if col < 2 # if default predictor is only one active, take fitness-weighted average of all a, b as forecast
        chosen_j = 0
        a = sum(predictors[i][1] * (fitness_j[i])
                for i = 1:(num_predictors)) / sum(fitness_j[i] for i = 1:(num_predictors))
        b = sum(predictors[i][2] * (fitness_j[i])
                for i = 1:(num_predictors)) / sum(fitness_j[i] for i = 1:(num_predictors))
        # variance is just init var (σ_pd = 4.0)
        forecast = [chosen_j, a, b, σ_pd]

        return active_predictors, forecast, chosen_j
    end

    # Set chosen_j to index of predictor used to form agent demand
    chosen_j = 0
    # indices where all minima are found, minima because highest accuracy is one with lowest squarest forecast error
    highest_acc = findall((@view matched_collection[2, :]) .== minimum((@view matched_collection[2, :])))

    if length(highest_acc) == 1
        chosen_j = Int(matched_collection[1, getindex(highest_acc)])
    else
        fit_j = @view matched_collection[:, getindex([highest_acc])]
        fittest = StatsBase.sample(findall(fit_j[3, :] .== maximum(fit_j[3, :])))
        chosen_j = Int(fit_j[1, fittest])
    end

    # forecast vector composed of a, b, σ_i
    forecast = predictors[chosen_j][1:3]

    # add agent ID to forecast vector at position 1 for demand fn
    forecast = vcat(id, forecast)

    return active_predictors, forecast, chosen_j
end
# nested function to match predictor and state vector bits--only used in match_predictors()
function match_predict(bit_j, state_vector)
    predictor_j_match = 0
    for signal = 1:12
        if bit_j[signal] === missing || bit_j[signal] == state_vector[signal] # short-cicuit `or`
            predictor_j_match += 1
        else
            continue
        end
    end

    return predictor_j_match
end

"""
    `get_demand_slope() → demand, slope`

Calculate an individual agent's demand and slope and enforce trading, cash balance, and holdings constraints.

- An agent's individual demand is their target number of holdings based on a risk aversion computation.
- An agent's slope is the derivative of their demand computation w.r.t price
"""
function get_demand_slope(a, b, σ_i, trial_price, dt, r, λ, relative_holdings, trade_restriction, cash_restriction, short_restriction, relative_cash)
    forecast = a * (trial_price + dt) + b

    if forecast >= 0.0
        demand = ((forecast - trial_price * (1 + r)) / (λ * σ_i)) - relative_holdings
        slope = (a - (1 + r)) / (λ * σ_i)
    else
        forecast = 0.0
        demand = ((-trial_price * (1 + r)) / (λ * σ_i)) - relative_holdings
        slope = (1 + r) / (λ * σ_i)
    end

    # set and enforce trading constraints
    if demand > trade_restriction
        demand = trade_restriction
        slope = 0.0
    elseif demand < -trade_restriction
        demand = -trade_restriction
        slope = 0.0
    end

    # set and enforce cash and holding constraints
    # If buying, we check to see if we're within borrowing limits
    if (demand > 0.0)
        if (demand * trial_price > (relative_cash - cash_restriction))
            if (relative_cash - cash_restriction > 0.0)
                demand = (relative_cash - cash_restriction) / trial_price
                slope = -demand / trial_price
            else
                demand = 0.0
                slope = 0.0
            end
        end
        # If selling, we check to make sure we have enough stock to sell
    elseif (demand < 0.0 && demand + relative_holdings < short_restriction)
        demand = short_restriction - relative_holdings
        slope = 0.0
    end

    return demand, slope
end


## Market State Updating

"""
Update historical volatility vector

30-day historical volatility used (standard deviation of the daily gain or loss from each of the past 30 time steps).
"""
function update_volatility(price)
    hist_p30 = price[(end-29):(end)]
    sd_p30 = sqrt((sum((hist_p30 .- mean(hist_p30)) .^ 2)) / 30)
    return sd_p30
end

"""
    `update_frac_bits() → frac_bits_set, frac_bits_fund, frac_bits_tech`

Update three fractions of "set" (i.e., non-missing) bits being tracked in the simulation:
- all 12 bits(including the dummy ones),
- 6 fundamental bits,
- and the 4 technical bits.
These values will then be averaged over all agents and all predictors (done in model.jl). 
"""
function update_frac_bits(predictors)
    frac_bits_set = 0
    frac_bits_fund = 0
    frac_bits_tech = 0
    for i = 1:length(predictors)
        s_all = count(!ismissing, (@view predictors[i][4:15])) # total number of "set" bits
        s_fund = count(!ismissing, (@view predictors[i][4:9])) # total number of "set" fundamental bits
        s_tech = count(!ismissing, (@view predictors[i][10:13])) # total number of "set" technical bits
        frac_bits_set += s_all
        frac_bits_fund += s_fund
        frac_bits_tech += s_tech
    end
    return frac_bits_set, frac_bits_fund, frac_bits_tech
end


## Agent Updating (done for each agent individually)

"""
Update individual agent financial metrics

From order execution output, update `relative_cash` & `relative_holdings`
"""
function update_rewards!(df_trades, agent)
    for i = 1:nrow(df_trades)
        if df_trades[i, :AgentID] == agent.id
            agent.relative_holdings = df_trades[i, :demand_xi]
            agent.relative_cash = df_trades[i, :Current_cash]
        end
    end
end

"""
Update accuracy of each active predictor.

The predictor square error/deviation and `predict_acc` value itself are constrained to maximum values.
The maximum values were obtained from X...
"""
function update_predict_acc!(agent, τ, price, dividend)
    for i = 1:length(agent.predict_acc)
        if i .∈ Ref(agent.active_predictors)
            a_j = agent.predictors[i][1]
            b_j = agent.predictors[i][2]
            deviation = (((price[end] + dividend[end]) - (a_j * (price[end-1] + dividend[end-1]) + b_j))^2)
            # Enforce max value of predictor error/deviation to be 500.0
            # TODO: Make this constraint non-hardcoded
            if deviation > 500.0
                deviation = 500.0
            end
            agent.predict_acc[i] = (1 - (1 / τ)) * agent.predict_acc[i] + (1 / τ) * deviation

            # Enforce max value of predict_acc to be 100.0 (necessary to validate C=0.005)
            # TODO: Make this constraint non-hardcoded
            if agent.predict_acc[i] > 100.0
                agent.predict_acc[i] = 100.0
            end
        end
    end
end

"""
Update matrix that contains information about each predictors historical use.

First column records whether or not each predictor has ever been active before, `0`-> No, `1`-> Yes.
Second column records the last t step that each predictor was either initiated, active, replaced, or generalized.
Both columns needed for generalization procedure and GA offspring forecast variance procedure.
"""
function update_active_j_records!(num_predictors, active_predictors, active_j_records, t)
    for i = 1:num_predictors
        if in.(i, Ref(active_predictors)) == true
            active_j_records[i, 1] = 1
            active_j_records[i, 2] = t
        end
    end
end


## Genetic Algorithm Invocation (done for each agent individually)

"""
    `GA_crossover() → crossed_j`

Recombination via GA_crossover(), occurs with probability `pGAcrossover`.

Crossover procedure:
- Offspring constructed from two unique parents
- Each parent is chosen via tournament selection from `elite_j`
- Tournament selection is randomized and selects larger of two `fitness_j` values
    - If parents with same fitness are randomly selected, one is randomly picked over other (see: argmax())
    - *Replace above process with drawing again? Would this make any difference at all?
- Offspring condition statement:
    - Uniform crossover of both parent's condition statements
    - Each bit constructed one at a time from corresponding parent's bits with equal probability
- Offspring forecasting parameters:
    - The a, b parameters of the offspring is chosen randomly w/ equal prob from following 3 methods:
        1. Randomly adopt one of the parents' a, b (all from one or all from the other) with equal probability
        2. Component-wise crossover; randomly choose a from one parent and b from the other with equal probability
        3. Take weighted avg (1/σ_j) of two parents a, b. Weights normalized to sum to 1. 
- Offspring forecast variance:
    - Inherit average var of parents, unless both of these parent predictors have never been matched before
    (never been an active predictor before). In this case, the offspring will adopt the median forecast error over
    all elite predictors
"""
function GA_crossover(elite_j, df_GA, active_j_records)
    # tournament selection
    tournament_C1 = StatsBase.samplepair(elite_j)
    tournament_C2 = StatsBase.samplepair(elite_j)
    fittest_C1 = argmax([df_GA[tournament_C1[1], :fitness_j], df_GA[tournament_C1[2], :fitness_j]])
    parent_1 = tournament_C1[fittest_C1]
    fittest_C2 = argmax([df_GA[tournament_C2[1], :fitness_j], df_GA[tournament_C2[2], :fitness_j]])
    parent_2 = tournament_C2[fittest_C2]

    # parent condition statements
    parent_1_cond = df_GA[parent_1, :predictors][4:15]
    parent_2_cond = df_GA[parent_2, :predictors][4:15]

    # offspring condition statement
    offspring_cond = Any[]

    # uniform crossover
    function cond_cross(offspring_cond, bit)
        if rand([1, 2]) == 1
            push!(offspring_cond, parent_1_cond[bit])
        else
            push!(offspring_cond, parent_2_cond[bit])
        end
    end

    for bit = 1:length(parent_1_cond)
        cond_cross(offspring_cond, bit)
    end

    # parent forecast parameters
    parent_1_params = df_GA[parent_1, :predictors][1:2]
    parent_2_params = df_GA[parent_2, :predictors][1:2]

    # parent forecast variance
    parent_1_var = df_GA[parent_1, :predictors][3]
    parent_2_var = df_GA[parent_2, :predictors][3]

    # offspring forecast parameters
    offspring_params = Any[]

    # randomly choose one of three crossover methods
    function param_cross(offspring_params, parent_1_params, parent_2_params, parent_1_var, parent_2_var)
        r = rand([1, 2, 3])
        if r == 1
            method_1 = rand([1, 2])
            if method_1 == 1
                push!(offspring_params, parent_1_params[1]) # a
                push!(offspring_params, parent_1_params[2]) # b
            else
                push!(offspring_params, parent_2_params[1]) # a
                push!(offspring_params, parent_2_params[2]) # b
            end
        elseif r == 2
            method_2 = rand([1, 2])
            if method_2 == 1
                push!(offspring_params, parent_1_params[1]) # a
                push!(offspring_params, parent_2_params[2]) # b
            else
                push!(offspring_params, parent_2_params[1]) # a
                push!(offspring_params, parent_1_params[2]) # b
            end
        elseif r == 3
            norm_weight_1 = (1 / parent_1_var) / ((1 / parent_1_var) + (1 / parent_2_var))
            norm_weight_2 = (1 / parent_2_var) / ((1 / parent_1_var) + (1 / parent_2_var))
            a = ((norm_weight_1 * parent_1_params[1]) + (norm_weight_2 * parent_2_params[1])) / (norm_weight_1 + norm_weight_2)
            b = ((norm_weight_1 * parent_1_params[2]) + (norm_weight_2 * parent_2_params[2])) / (norm_weight_1 + norm_weight_2)
            push!(offspring_params, a)
            push!(offspring_params, b)
        end
    end

    param_cross(offspring_params, parent_1_params, parent_2_params, parent_1_var, parent_2_var)

    # offspring forecast variance
    offspring_var = Any[]

    # median var of all elite if never been active before, average of parents otherwise
    if active_j_records[parent_1, 1] == 0 && active_j_records[parent_2, 1] == 0
        elite_var = df_GA[:, :predict_acc]
        filter!(x -> !(isnan(x)), elite_var)
        push!(offspring_var, median(elite_var))
    else
        push!(offspring_var, (parent_1_var + parent_2_var) / 2)
    end

    # returning new crossed over predictor
    crossed_j = vcat(offspring_params, offspring_var, offspring_cond)

    return crossed_j
end

"""
    `GA_mutation() → mutated_j`

Recombination via GA_mutation(), occurs with probability (1 - `pGAcrossover`)

Mutation procedure:
- Offspring constructed from one parent among the elite
- Unique parent is chosen via tournament selection from `elite_j`
- Tournament selection is randomized and selects larger of two `fitness_j` values
    - If parents with same fitness are randomly selected, one is randomly picked over other (see: argmax())
    - *Replace above process with drawing again? Would this make any difference at all?
- Offspring condition statement:
    - The offspring adopts the parents condition bits which are "flipped" at random
    - With probability 0.03, each bit in vector undergoes the following:
        0 -> missing (prob 2/3), 0 -> 1 (prob 1/3)
        1 -> missing (prob 2/3), 1 -> 0 (prob 1/3)
        missing -> 0 (prob 1/3), missing -> 1 (prob 1/3), unchanged (prob 1/3)
- Offspring forecasting parameters:
    - The offspring adopts the parents a, b with random numbers added to them
    - Mutation may do one of three things, independently for each parameter:
        1. With prob 0.2, the parameter is changed to random value in its allowable range
        2. With prob 0.2, the parameter is chosen randomly from a uniform distribution from its current value ± 0.05 * its 
           max-min range. Values outside the allowable range are set to the respective max or min. 
        3. With prob 0.6, the parameter is left unchanged. 
- Offspring forecast variance:
    - The offspring adopts the median forecast error over all elite predictors
"""
function GA_mutation(elite_j, df_GA, pcond_mut, a_min, a_max, b_min, b_max, pparam_mut_long, pparam_mut_short, percent_mut_short)
    # tournament selection
    tournament_M1 = StatsBase.samplepair(elite_j)
    fittest_M1 = argmax([df_GA[tournament_M1[1], :fitness_j], df_GA[tournament_M1[2], :fitness_j]])
    parent_1 = tournament_M1[fittest_M1]

    # parent condition statement
    parent_1_cond = df_GA[parent_1, :predictors][4:15]

    # offspring condition statement
    offspring_cond = Any[]

    # Mutation procedure for offspring condition bits
    function cond_mutat(offspring_cond, bit)
        if ismissing(parent_1_cond[bit]) == true
            r = rand([1, 2, 3])
            if r == 1
                push!(offspring_cond, 0)
            elseif r == 2
                push!(offspring_cond, 1)
            else
                push!(offspring_cond, missing)
            end
        elseif parent_1_cond[bit] == 0
            if rand() ≤ 2 / 3
                push!(offspring_cond, missing)
            else
                push!(offspring_cond, 1)
            end
        else
            if rand() ≤ 2 / 3
                push!(offspring_cond, missing)
            else
                push!(offspring_cond, 0)
            end
        end
    end

    for bit = 1:length(parent_1_cond)
        if rand() ≤ pcond_mut
            cond_mutat(offspring_cond, bit)
        else
            push!(offspring_cond, parent_1_cond[bit])
        end
    end

    # parent forecast parameters
    parent_1_params = df_GA[parent_1, :predictors][1:2]

    # offspring forecast parameters
    offspring_params = Any[]

    # randomly choose one of three mutation methods
    function param_mutat(offspring_params, parent_1_params)
        # for a
        r = rand()
        if r > (pparam_mut_long + pparam_mut_short)
            a = parent_1_params[1]
        elseif r ≤ pparam_mut_long
            a = rand(Uniform(a_min, a_max))
        else
            a = rand(Uniform((parent_1_params[1] -
                              (percent_mut_short * (a_max - a_min))), (parent_1_params[1] + (percent_mut_short * (a_max - a_min)))))
            # setting param to bound values if it exceeds allowable range
            if a < a_min
                a = a_min
            elseif a > a_max
                a = a_max
            end
        end

        # for b
        r = rand()
        if r > (pparam_mut_long + pparam_mut_short)
            b = parent_1_params[2]
        elseif r ≤ pparam_mut_long
            b = rand(Uniform(b_min, b_max))
        else
            b = rand(Uniform((parent_1_params[2] -
                              (percent_mut_short * (b_max - b_min))), (parent_1_params[2] + (percent_mut_short * (b_max - b_min)))))
            # setting param to bound values if it exceeds allowable range
            if b < b_min
                b = b_min
            elseif b > b_max
                b = b_max
            end
        end

        push!(offspring_params, a)
        push!(offspring_params, b)
    end

    param_mutat(offspring_params, parent_1_params)

    # offspring forecast variance
    offspring_var = Any[]

    # median var of all elite
    elite_var = df_GA[:, :predict_acc]
    filter!(x -> !(isnan(x)), elite_var)
    push!(offspring_var, median(elite_var))

    # returning new mutated predictor
    mutated_j = vcat(offspring_params, offspring_var, offspring_cond)

    return mutated_j
end


end # End of module
