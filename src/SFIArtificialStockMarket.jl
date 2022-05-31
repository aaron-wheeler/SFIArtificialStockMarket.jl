# comment workaround before module statement so that VS Code doesn't bug out
module SFIArtificialStockMarket

using Agents
using Distributions
#using Pipe
using Random
using StatsBase
using ForwardDiff
using DataFrames
using MLStyle
using Statistics
using JuMP
using Ipopt
using Roots

## Update Market State 

"""
    `dividend_process() → dividend`

Autoregressive dividend process is appended to vector and made public to all agents
Gaussian noise term `ε` is independent & identically distributed and has zero mean and variance σ_ε
"""
function dividend_process!(dividend, d̄, ρ, σ_ε)
    ε = rand(Normal(0.0,σ_ε))
    dt = d̄ + ρ*(last(dividend) - d̄) + ε
    dividend[1] = dividend[2]
    dividend[2] = dt
end
# function dividend_process(d̄, ρ, dividend, σ_ε)
#     ε = rand(Normal(0.0, σ_ε)) # way to include random seed? Move this to model.jl?
#     dt = d̄ + ρ * (last(dividend) - d̄) + ε
#     dividend = push!(dividend, dt)
#     return dividend
# end

"""
    `update_market_vector() → bit 1-12`

Update global market state bit vector, assign "1" or "0" values depending on the presence of bit signals
Signal present -> "1"
Signal absent -> "0"
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

# function update_market_vector(price, dividend, r)
#     # Fundamental bits
#     if last(price) * r / last(dividend) > 0.25
#         bit1 = 1
#     else
#         bit1 = 0
#     end

#     if last(price) * r / last(dividend) > 0.5
#         bit2 = 1
#     else
#         bit2 = 0
#     end

#     if last(price) * r / last(dividend) > 0.75
#         bit3 = 1
#     else
#         bit3 = 0
#     end

#     if last(price) * r / last(dividend) > 0.875
#         bit4 = 1
#     else
#         bit4 = 0
#     end

#     if last(price) * r / last(dividend) > 1.0
#         bit5 = 1
#     else
#         bit5 = 0
#     end

#     if last(price) * r / last(dividend) > 1.125
#         bit6 = 1
#     else
#         bit6 = 0
#     end

#     # Technical bits, the `period` in MA formula is set to 1 time step
#     if last(price) > mean(price[(end-6):end])
#         bit7 = 1
#     else
#         bit7 = 0
#     end

#     if last(price) > mean(price[(end-9):end])
#         bit8 = 1
#     else
#         bit8 = 0
#     end

#     if last(price) > mean(price[(end-99):end])
#         bit9 = 1
#     else
#         bit9 = 0
#     end

#     if last(price) > mean(price[(end-499):end])
#         bit10 = 1
#     else
#         bit10 = 0
#     end

#     # Default bits, always on/off
#     bit11 = 1

#     bit12 = 0

#     # Construct vector
#     state_vector = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10, bit11, bit12]

#     return state_vector
# end

## Initialization (done for each agent individually)

"""
    `init_predictors() → predictors`

Agent `predictors` vector is coupled to unique `id`.

**Should accuracy and fitness measure also be appended to each predictor?
"""
function init_predictors(num_predictors, σ_pd, a_min, a_max, b_min, b_max) # Add an identifier? 
    predictors = Vector{Any}(undef, 0)
    for i = 1:(num_predictors-1) # minus one so that we can add default predictor
        heterogeneity = Vector{Any}(undef, 3)
        heterogeneity[1] = rand(Uniform(a_min, a_max)) # a
        heterogeneity[2] = rand(Uniform(b_min, b_max)) # b 
        heterogeneity[3] = σ_pd # initial σ_i = σ_pd
        bit_vec = Vector{Any}(undef, 12)
        Distributions.sample!([missing, 1, 0], Weights([0.9, 0.05, 0.05]), bit_vec)
        bit_vec = vcat(heterogeneity, bit_vec)
        predictors = push!(predictors, bit_vec)
    end
    # default predictor
    default_heterogeneity = Vector{Any}(undef, 3)
    default_heterogeneity[1] = sum(predictors[i][1] * (1 / predictors[i][3])
                                   for i = 1:(num_predictors-1)) / sum(1 / predictors[i][3] for i = 1:(num_predictors-1)) # default a
    default_heterogeneity[2] = sum(predictors[i][2] * (1 / predictors[i][3])
                                   for i = 1:(num_predictors-1)) / sum(1 / predictors[i][3] for i = 1:(num_predictors-1)) # default b 
    default_heterogeneity[3] = σ_pd # initial default σ_i = σ_pd
    default_bit_vec = Vector{Any}(missing, 12)
    default_bit_vec = vcat(default_heterogeneity, default_bit_vec)
    predictors = push!(predictors, default_bit_vec)
    return predictors # append indentifiers for each predictor?
end


"""
    `init_learning() → δ, predict_acc, fitness_j`

Constructs and initializes each agent's `predict_acc`, 'fitness_j`, and `δ` coupled to unique `id`.
- `δ` returned as vector, needed for asynch recombination  
- `predict_acc` returned as vector, associated to each predictor for agent `id`
- `fitness_j` returned as vector, associated to each `predict_acc` for agent `id`

# FOR ERROR MESSAGES/CONSISTENCY CHECKS**
# println(sum(δ)) # == T
# println(mean(δ)) # == k
"""
function init_learning(σ_pd, C, num_predictors, predictors)
    predict_acc = fill(σ_pd, num_predictors) # (σ_i), initialized as σ_pd(4.0) in first period, set as σ_pd to avoid loop
    fitness_j = Vector{Float64}(undef, 0)
    for i = 1:num_predictors
        s = count(!ismissing, predictors[i][4:15]) # specificity, number of bits that are "set" (not missing)
        f_j = -1 * (predict_acc[i]) - C * s
        fitness_j = push!(fitness_j, f_j)
    end

    return predict_acc, fitness_j
end

# function init_learning(GA_frequency, δ_dist, σ_pd, C, num_predictors, predictors)  # Add an identifier for agent?
#     δ = Vector{Int}(undef, GA_frequency)
#     Distributions.sample!(δ_dist, δ; replace = false, ordered = false)
#     δ = cumsum(δ)

#     predict_acc = fill(σ_pd, num_predictors) # (σ_i), initialized as σ_pd(4.0) in first period, set as σ_pd to avoid loop
#     fitness_j = Vector{Float64}(undef, 0)
#     for i = 1:num_predictors
#         s = count(!ismissing, predictors[i][4:15]) # specificity, number of bits that are "set" (not missing)
#         f_j = -1 * (predict_acc[i]) - C * s
#         fitness_j = push!(fitness_j, f_j)
#     end

#     return δ, predict_acc, fitness_j #Append identifying number for predicts and fitnesses?
# end


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

**Add `active_predictors` and `forecast` to agent struct? Or init to zero/replace @each time step?
**Need to know index of predictor sent to demand (chosen_j) for any reason? Not returned here, just its a, b, etc.
"""
function match_predictors(id, num_predictors, predictors, state_vector, predict_acc, fitness_j, σ_pd)
    # Initialize matrix to store indices of all active predictors
    active_predictors = Int[]

    # nested function to match predictor and state vector bits (only used here)
    match_predict(bit_j, predictor_j, j_id) =
        for signal = 1:12
            @match bit_j begin
                if bit_j[signal] === missing || bit_j[signal] == state_vector[signal]
                end => push!(predictor_j, j_id)
                _ => continue
            end
        end

    for j = 1:num_predictors
        # reset predictor_j with each iteration
        predictor_j = Vector{Int}(undef, 0)
        j_id = j

        # call nested function to see if predictor[j] bits match the state vector
        match_predict(predictors[j][4:15], predictor_j, j_id)

        # if predictor[j] meets match criteria, append to active_predictors matrix
        if length(predictor_j) == 12
            active_predictors = push!(active_predictors, predictor_j[1])
        else
            nothing
        end

    end

    matched_collection = zeros(Int, 3, 0)

    for (index, value) in pairs(IndexStyle(active_predictors), active_predictors)
        matched_collection = hcat(matched_collection, [value, predict_acc[value], fitness_j[value]])
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
    highest_acc = findall(matched_collection[2, :] .== minimum(matched_collection[2, :]))

    if length(highest_acc) == 1
        chosen_j = Int(matched_collection[1, getindex(highest_acc)])
    else
        # highest_fitness = findall(matched_collection[3, :] .== maximum(matched_collection[3, :]))
        # fittest_acc = Vector{Int}(undef, 0)

        # for i = 1:size(matched_collection, 2)
        #     if in.(i, Ref(highest_acc)) == true && in.(i, Ref(highest_fitness)) == true
        #         fittest_acc = push!(fittest_acc, i)
        #     end
        # end

        # if length(fittest_acc) == 1
        #     chosen_j = Int(matched_collection[1, getindex(fittest_acc)])
        # else
        #     # length(fittest_acc) > 1, pick one randomly?
        #     chosen_j = 100
        # end
        fit_j = matched_collection[:, getindex([highest_acc])]
        fittest = StatsBase.sample(findall(fit_j[3, :] .== maximum(fit_j[3, :])))
        chosen_j = Int(fit_j[1, fittest])
    end

    # forecast vector composed of a, b, σ_i
    forecast = predictors[chosen_j][1:3]

    # add agent ID to forecast vector at position 1 for demand fn
    forecast = vcat(id, forecast)

    return active_predictors, forecast, chosen_j
end


"""
Balance market orders and calculate agent demand. 

At market equilibrium, the specialist obtains the clearing price for the risky asset and rations market orders by the 
explicit trading constraints `trade_restriction` & `short_restriction`
Rationing procedure:
    If N - round_error > 0 (i.e. positive)
        Negative difference --> Add extra share(s) to be bought (+1)
    End

    If N - round_error < 0 (i.e. negative)
        Positive difference --> Add extra share(s) to be sold (-1)
    End

    - For every missing share: 
    - Add extra shares by rank order highest ration_imbalance values
    - Handle ties randomly to avoid bias. Sort(rev order when -1) --> Random shuffle

    Question: Is it better to ration one side (i.e. only sell more) or split rationing (i.e. sell more and buy less)
    Answer (Orig model): Ration one side
    Answer (This model): If ration diff is greater then traders on ex. supply side, then traders on demand side will buy less 
    to avoid messing with any 0.0 rounding diff stemming from constraint cutoffs

ERROR TERMS TO INCLUDE LATER**
- Convergence not reached for newton's method under itermax
- demand_xi not being equivalent to 25 at end of rationing procedure
"""
function get_demand!(num_agents, price, dividend, r, λ, expected_xi, relative_cash, relative_holdings,
    trade_restriction, short_restriction, itermax, price_min, price_max)
    N = num_agents # number of shares is equivalent to number of agents in model 
    dt = last(dividend)
    Identifier = convert(Vector{Int}, expected_xi[1, :])
    a = convert(Vector{Float64}, expected_xi[2, :])
    b = convert(Vector{Float64}, expected_xi[3, :])
    σ_i = convert(Vector{Float64}, expected_xi[4, :])

    # # Solving for clearing price via newton's method
    # f(pt) = sum(((a[i] * (pt + dt) + b[i] - pt * (1 + r)) / (λ * σ_i[i])) for i = 1:num_agents) - N
    # pt = last(price) # initial condition, last observed price 
    # pt_iter = [] # More efficent way to do this, with no vector?
    # for i = 1:itermax
    #     if i == 1
    #         pt = pt - (f(pt) / ForwardDiff.derivative(f, pt))
    #         push!(pt_iter, pt)
    #     else
    #         pt = pt - (f(pt) / ForwardDiff.derivative(f, pt))
    #         push!(pt_iter, pt)
    #         # check convergence criteria, defaulted to 7 digits
    #         if isequal(round(pt_iter[end-1], digits = 7), round(pt_iter[end], digits = 7))
    #             break
    #         end
    #     end
    # end
    # cprice = last(pt_iter)

    # # Solving for clearing price via NLP optimization solver (Ipopt)
    # price_specialist = Model(Ipopt.Optimizer)
    # set_optimizer_attribute(price_specialist, "print_level", 0) # suppress solver output message
    # set_optimizer_attribute(price_specialist, "max_iter", itermax) # max number of iterations
    # set_optimizer_attribute(price_specialist, "tol", 1e-4) # convergence criteria
    # set_time_limit_sec(price_specialist, 60.0) # max allowable time for model solving
    # # Set constraints on price variable and solve trivial scalar obj fn (zero degrees of freedom)
    # @variable(price_specialist, price_min <= pt <= price_max, start = last(price)) # initial condition -> last observed price
    # @constraint(price_specialist, sum(((a[i] * (pt + dt) + b[i] - pt * (1 + r)) / (λ * σ_i[i])) for i = 1:num_agents) - N >= 0.0)
    # @objective(price_specialist, Min, 1.0)
    # # Price specialist obtains clearing price and stores value
    # JuMP.optimize!(price_specialist)
    # cprice = value(pt)

    # Solving for clearing price via derivative-free root-finding algorithm
    f(pt) = sum(((a[i]*(pt + dt) + b[i] - pt*(1 + r)) / (λ*σ_i[i])) for i in 1:num_agents) - N
    pt = last(price) # initial condition, last observed price 
    cprice = find_zero(f, pt)
    # set and enforce constraints on price variable
    if cprice < price_min || cprice > price_max
        cprice = cprice < price_min ? price_min : price_max
    end

    # calculate individual agent demand
    test_demand_N_convergence = Vector{Float64}(undef, 0)
    for i = 1:num_agents
        demand = ((a[i] * (cprice + dt) + b[i] - cprice * (1 + r)) / (λ * σ_i[i]))
        push!(test_demand_N_convergence, demand)
    end

    # Rounding witchcraft
    for i = 1:length(test_demand_N_convergence)
        test_demand_N_convergence[i] = round(test_demand_N_convergence[i], digits = 1)
    end

    # Way to do this without making this vector every time?
    xi_excess = Vector{Float64}(undef, 0)
    for i = 1:N
        if test_demand_N_convergence[i, 1] > trade_restriction
            push!(xi_excess, (test_demand_N_convergence[i, 1] - (test_demand_N_convergence[i, 1] - trade_restriction)))
        elseif test_demand_N_convergence[i, 1] < short_restriction
            push!(xi_excess, (test_demand_N_convergence[i, 1] - (test_demand_N_convergence[i, 1] - short_restriction)))
        else
            push!(xi_excess, test_demand_N_convergence[i, 1])
        end
    end
    test_demand_N_convergence = xi_excess

    # To determine rounding difference and adjust share rationing
    test_demand_round_diff = Vector{Float64}(undef, 0)
    append!(test_demand_round_diff, test_demand_N_convergence)

    for i = 1:length(test_demand_N_convergence)
        test_demand_round_diff[i] = trunc(Int64, test_demand_round_diff[i])
    end
    round_error = sum(test_demand_round_diff)
    round_error = convert(Int, round_error)

    demand_ration = hcat(test_demand_N_convergence, test_demand_round_diff)

    # round difference
    ration_imbalance = N - round_error
    demand_ration_imbalance = vec(diff(demand_ration, dims = 2))
    demand_ration_imbalance .= round.(demand_ration_imbalance, digits = 1)
    demand_ration = hcat(demand_ration, demand_ration_imbalance)

    df = DataFrame(AgentID = Identifier, Current_holding = relative_holdings, Current_cash = relative_cash,
        init_xi = demand_ration[:, 1], round_xi = demand_ration[:, 2], xi_diff = demand_ration[:, 3])

    if ration_imbalance > 0
        # shuffle and sorts agents by xi_diff, largest negative value to largest positive value
        df = df[shuffle(1:nrow(df)), :]
        sort!(df, :xi_diff)
    elseif ration_imbalance < 0
        # shuffle and sorts agents by xi_diff, largest positive value to largest negative value
        df = df[shuffle(1:nrow(df)), :]
        sort!(df, order(:xi_diff), rev = true)
    end

    # number of shares the agents want to posess at time t
    df[!, :demand_xi] = df[:, :round_xi]

    # Rationing supply side, more shares sold (-) or less shares bought (if needed, -)
    i = 1
    overbought_shares = abs(ration_imbalance)
    while overbought_shares > 0
        if df[i, :xi_diff] < 0
            break
        elseif df[i, :demand_xi] <= short_restriction
            i += 1
            continue
        elseif df[i, :xi_diff] > 0
            df[i, :demand_xi] -= 1.0
            i += 1
            overbought_shares -= 1
        else
            df = df[shuffle(1:nrow(df)), :]
            sort!(df, :xi_diff)
            for j = 1:Int((abs(ration_imbalance) - i + 1))
                if overbought_shares == 0
                    break
                elseif j > 25
                    continue
                elseif df[j, :demand_xi] <= short_restriction
                    continue
                else
                    df[j, :demand_xi] -= 1.0
                    overbought_shares -= 1
                end
            end
            df = df[shuffle(1:nrow(df)), :]
            sort!(df, order(:xi_diff), rev = true)
            i = 1
        end
    end

    # Rationing demand side, more shares bought (+) or less shares sold (if needed, +)
    oversold_shares = ration_imbalance
    while oversold_shares > 0
        if df[i, :xi_diff] > 0
            break
        elseif df[i, :demand_xi] >= trade_restriction
            i += 1
            continue
        elseif df[i, :xi_diff] < 0
            df[i, :demand_xi] += 1.0
            i += 1
            oversold_shares -= 1
        else
            df = df[shuffle(1:nrow(df)), :]
            sort!(df, order(:xi_diff), rev = true)
            for j = 1:Int((ration_imbalance - i + 1))
                if oversold_shares == 0
                    break
                elseif j > 25
                    continue
                elseif df[j, :demand_xi] >= trade_restriction
                    continue
                else
                    df[j, :demand_xi] += 1.0
                    oversold_shares -= 1
                end
            end
            df = df[shuffle(1:nrow(df)), :]
            sort!(df, :xi_diff)
            i = 1
        end
    end
    df[!, :demand_xi] = convert.(Int, df[:, :demand_xi])
    ration_imbalance = N - sum(df[:, :demand_xi])
    cprice = round(cprice; digits = 2) # any issues with doing this?
    return df, cprice
end


"""
Collect new agent share amount, cash and calculate shares traded, agent profit at time t 

Enforce cash constraint `X` and return df with adjusted agent metrics
If the budget constraint is violated:
    -The agent(s) trying to enter a restricted cash position will reduce share demand and receive cash back
    -The leftover share(s) will go to the agent(s) with the largest short position and nonnegative cash balance
    -Process is repeated until all agents possess allowable cash balance
    -This process ensures that the balance of global cash, shares, etc. is conserved

ERROR TERMS TO INCLUDE LATER**
- Conservation tests for total cash, profit, demand, and shares traded.
- Check again for passing of trade constraints in case there is adjustment
- Additional final check to ensure all nonnegative cash values
"""
function get_trades!(df, cprice, cash_restriction)
    # The :shares_traded column needed for trading volume vector
    df[!, :shares_traded] = [(df[i, :demand_xi] - df[i, :Current_holding]) for i = 1:nrow(df)]
    df[!, :shares_traded] = convert.(Float64, df[:, :shares_traded])
    # The :profit column is specific to time t, different from net profit 
    df[!, :profit_t] = [(df[i, :shares_traded] * cprice * -1) for i = 1:nrow(df)]
    df[:, :Current_cash] = [(df[i, :Current_cash] + df[i, :profit_t]) for i = 1:nrow(df)]

    # Enforcing agent cash constraint
    for i = 1:nrow(df)
        if df[i, :Current_cash] < cash_restriction
            while df[i, :Current_cash] < cash_restriction
                # subtract share (sell one)
                df[i, :Current_cash] += cprice
                df[i, :demand_xi] -= 1
                df[i, :shares_traded] -= 1.0
                df[i, :profit_t] += cprice
                # add share (buy one) elsewhere
                for j = 1:nrow(df)
                    if getindex(df[j, :demand_xi]) == minimum(df[:, :demand_xi]) && getindex(df[j, :Current_cash]) > 0.0
                        df[j, :Current_cash] -= cprice
                        df[j, :demand_xi] += 1
                        df[j, :shares_traded] += 1.0
                        df[j, :profit_t] -= cprice
                        break
                    end
                end
            end
        end
    end

    # Return adjusted agent metrics
    select!(df, :AgentID, :Current_cash, :demand_xi, :shares_traded, :profit_t)
    sort!(df)
    return df
end


"""
Will update this later

"""
function get_demand_slope!(a, b, σ_i, trial_price, dt, r, λ, relative_holdings, trade_restriction, cash_restriction, short_restriction, relative_cash)
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
Update trading volume vector

Trading volume measured as the number of shares that have changed hands during time t

- How Can Trading Volume Exceed Shares Outstanding?
https://www.investopedia.com/ask/answers/041015/why-trading-volume-important-investors.asp
"""
function update_trading_volume!(num_agents, df_trades)
    volume_t = 0
    for i = 1:num_agents
        if df_trades[i, :shares_traded] > 0
            volume_t += getindex(df_trades[i, :shares_traded])
        end
    end
    return volume_t
end

"""
Update historical volatility vector

30-day historical volatility used (standard deviation of the daily gain or loss from each of the past 30 time steps).
"""
function update_volatility!(price)
    hist_p30 = price[(end-29):(end)]
    sd_p30 = sqrt((sum((hist_p30 .- mean(hist_p30)) .^ 2)) / 30)
    return sd_p30
end


"""
Update value tracking the fraction of "set" (i.e., non-missing) bits present in model

Tracking 3 fractions here, number of "set" bits over: all 12 bits(including the dummy ones), 6 fundamental bits, and the 4 technical bits.
These values will then be averaged over all agents and all predictors (done in model.jl). 
"""
function update_frac_bits!(predictors, frac_bits_set, frac_bits_fund, frac_bits_tech)
    for i = 1:length(predictors)
        s_all = count(!ismissing, predictors[i][4:15]) # total number of "set" bits
        s_fund = count(!ismissing, predictors[i][4:9]) # total number of "set" fundamental bits
        s_tech = count(!ismissing, predictors[i][10:13]) # total number of "set" technical bits
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
function update_rewards!(df_trades, agent) # input seperate agent elements as arguments instead of entire agent struct?
    for i = 1:nrow(df_trades)
        if df_trades[i, :AgentID] == agent.id
            agent.relative_holdings = df_trades[i, :demand_xi]
            agent.relative_cash = df_trades[i, :Current_cash]
        end
    end
end


"""
Update accuracy of each active predictor. 
"""
# function update_predict_acc!(predict_acc, active_predictors, predictors, τ, price, dividend)
#     for i = 1:length(predict_acc)
#         if i .∈ Ref(active_predictors)
#             a_j = predictors[i][1]
#             b_j = predictors[i][2]
#             predict_acc[i] = (1 - (1 / τ)) * predict_acc[i] +
#                              (1 / τ) * (((price[end] + dividend[end]) - (a_j * (price[end-1] + dividend[end-1]) + b_j))^2)
#             # Enforce max value of predict_acc to be 500.0 (necessary to validate C=0.005)
#             if predict_acc[i] > 500.0
#                 predict_acc[i] = 500.0
#             end
#         end
#     end
# end

function update_predict_acc!(agent, τ, price, dividend)
    for i = 1:length(agent.predict_acc)
        if i .∈ Ref(agent.active_predictors)
            a_j = agent.predictors[i][1]
            b_j = agent.predictors[i][2]
            # agent.predict_acc[i] = (1 - (1 / τ)) * agent.predict_acc[i] +
            #                        (1 / τ) * (((price[end] + dividend[end]) - (a_j * (price[end-1] + dividend[end-1]) + b_j))^2)
            deviation = (((price[end] + dividend[end]) - (a_j * (price[end-1] + dividend[end-1]) + b_j))^2)
            if deviation > 500.0
                deviation = 500.0
            end
            agent.predict_acc[i] = (1 - (1 / τ)) * agent.predict_acc[i] + (1 / τ) * deviation
            # Enforce max value of predict_acc to be 100.0 (necessary to validate C=0.005)
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
Recombination via GA_crossover(), occurs with probability `pGAcrossover`

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
    offspring_var = Any[] # does this need to be vector? just set equal to value in if statement?

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
    offspring_var = Any[] # does this need to be vector? just set equal to value in if statement?

    # median var of all elite
    elite_var = df_GA[:, :predict_acc]
    filter!(x -> !(isnan(x)), elite_var)
    push!(offspring_var, median(elite_var))

    # returning new mutated predictor
    mutated_j = vcat(offspring_params, offspring_var, offspring_cond)

    return mutated_j
end


end # End of module
