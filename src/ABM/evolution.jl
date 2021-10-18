# comment before module statement so that VS Code doesn't bug out 
module evolution

using Agents
using Distributions
#using Pipe
using Random
using StatsBase
using ForwardDiff
using DataFrames
using MLStyle

## Update Market State 

"""
    `dividend_process() → dividend`

Autoregressive dividend process is appended to vector and made public to all agents
Gaussian noise term `ε` is independent & identically distributed and has zero mean and variance σ_ε
"""
function dividend_process(d̄, ρ, dividend, σ_ε)
    ε = rand(Normal(0.0,σ_ε)) # way to include random seed? Move this to model.jl?
    dt = d̄ + ρ*(last(dividend) - d̄) + ε
    dividend = push!(dividend, dt)
    return dividend
end

"""
    `update_market_vector() → bit 1-12`

Update global market state bit vector, assign "1" or "0" values depending on the presence of bit signals
Signal present -> "1"
Signal absent -> "0"
"""
function update_market_vector(price, dividend, r)
    # Fundamental bits
    if last(price) * r/last(dividend) > 0.25
        bit1 = 1
    else
        bit1 = 0
    end
    
    if last(price) * r/last(dividend) > 0.5
        bit2 = 1
    else
        bit2 = 0
    end
    
    if last(price) * r/last(dividend) > 0.75
        bit3 = 1
    else
        bit3 = 0
    end
    
    if last(price) * r/last(dividend) > 0.875
        bit4 = 1
    else
        bit4 = 0
    end
    
    if last(price) * r/last(dividend) > 1.0
        bit5 = 1
    else
        bit5 = 0
    end
    
    if last(price) * r/last(dividend) > 1.125
        bit6 = 1
    else
        bit6 = 0
    end
    
    # Technical bits, the `period` in MA formula is set to 1 time step
    if last(price) > mean(price[(end-6):end])
        bit7 = 1
    else
        bit7 = 0
    end   
    
    if last(price) > mean(price[(end-9):end])
        bit8 = 1
    else
        bit8 = 0
    end   
    
    if last(price) > mean(price[(end-99):end])
        bit9 = 1
    else
        bit9 = 0
    end   
    
    if last(price) > mean(price[(end-499):end])
        bit10 = 1
    else
        bit10 = 0
    end   
    
    return bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10
end

## Initialization (done for each agent individually)

"""
    `init_predictors() → predictors`

Agent `predictors` vector is coupled to unique `id`.

**Should accuracy and fitness measure also be appended to each predictor?
"""
function init_predictors(num_predictors, σ_pd) # Add an identifier? 
    predictors = Vector{Any}(undef, 0) 
    for i in 1:(num_predictors-1) # minus one so that we can add default predictor
        heterogeneity = Vector{Any}(undef, 3)
        heterogeneity[1] = rand(Uniform(0.7,1.2)) # a
        heterogeneity[2] = rand(Uniform(-10.0, 19.002)) # b 
        heterogeneity[3] = σ_pd # initial σ_i = σ_pd
        bit_vec = Vector{Any}(undef, 12)
        Distributions.sample!([missing, 1, 0], Weights([0.9, 0.05, 0.05]), bit_vec)
        bit_vec = vcat(heterogeneity, bit_vec)
        predictors = push!(predictors, bit_vec)
    end
    # default predictor
    default_heterogeneity = Vector{Any}(undef, 3)
    default_heterogeneity[1] = sum(predictors[i][1]*(1/predictors[i][3]) 
        for i in 1:(num_predictors-1)) / sum(1/predictors[i][3] for i in 1:(num_predictors-1)) # default a
    default_heterogeneity[2] = sum(predictors[i][2]*(1/predictors[i][3]) 
        for i in 1:(num_predictors-1)) / sum(1/predictors[i][3] for i in 1:(num_predictors-1)) # default b 
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
function init_learning(GA_frequency, δ_dist, σ_pd, C, num_predictors, predictors)  # Add an identifier for agent?
    δ = Vector{Float64}(undef, GA_frequency)
    Distributions.sample!(δ_dist, δ; replace=false, ordered=false)
    
    predict_acc = fill(σ_pd, 100) # (σ_i), initialized as σ_pd(4.0) in first period, set as σ_pd to avoid loop
    fitness_j = Vector{Float64}(undef, 0)
    for i in 1:num_predictors
        s = count(!ismissing, predictors[i][4:15]) # specificity, number of bits that are "set" (not missing)
        f_j = -1*(predict_acc[i]) - C*s
        fitness_j = push!(fitness_j, f_j)
    end
    
    return δ, predict_acc, fitness_j #Append identifying number for predicts and fitnesses?
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

**Add `active_predictors` and `forecast` to agent struct? Or init to zero/replace @each time step?
**Need to know index of predictor sent to demand (chosen_j) for any reason? Not returned here, just its a, b, etc.
"""
function match_predictors(id, num_predictors, predictors, state_vector, predict_acc, fitness_j)
    # Initialize matrix to store indices of all active predictors
    active_predictors = Int[] 

    # nested function to match predictor and state vector bits (only used here)
    match_predict(bit_j, predictor_j, j_id) =
        for signal in 1:12
            @match bit_j begin
            if bit_j[signal] === missing || bit_j[signal] == state_vector[signal] end => push!(predictor_j, j_id) 
                    _        => continue
            end
        end

    for j in 1:num_predictors
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

    # Set chosen_j to index of predictor used to form agent demand
    chosen_j = 0
    highest_acc = findall(matched_collection[2,:] .== maximum(matched_collection[2,:])) # indices where all maxima are found

    if length(highest_acc) == 1
        chosen_j = Int(matched_collection[1, getindex(highest_acc)])
    else
        highest_fitness = findall(matched_collection[3,:] .== maximum(matched_collection[3,:]))
        fittest_acc = Vector{Int}(undef, 0)

        for i = 1:size(matched_collection, 2)
            if in.(i, Ref(highest_acc)) == true && in.(i, Ref(highest_fitness)) == true
                fittest_acc = push!(fittest_acc, i)
            end
        end

        if length(fittest_acc) == 1
            chosen_j = Int(matched_collection[1, getindex(fittest_acc)])
        else
            chosen_j = 100
        end
    end

    # forecast vector composed of a, b, σ_i, and agent ID
    forecast = predictors[chosen_j][1:3]

    # add agent ID to forecast vector at position 1 for demand fn
    forecast = vcat(id, forecast)
    
    return active_predictors, forecast
end


"""
Balance market orders and calculate agent demand. 

At market equilibrium, the specialist obtains the clearing price for the risky asset and rations market orders by the 
explicit trading constraints `X.......`, ......
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
function get_demand!(num_agents, N, price, dividend, r, λ, expected_xi, relative_cash, relative_holdings, 
        trade_restriction, short_restriction, itermax)
    dt = last(dividend)
    Identifier = convert(Vector{Int}, expected_xi[1, :])
    a = convert(Vector{Float64}, expected_xi[2, :])
    b = convert(Vector{Float64}, expected_xi[3, :]) 
    σ_i = convert(Vector{Float64}, expected_xi[4, :])
    f(pt) = sum(((a[i]*(pt + dt) + b[i] - pt*(1 + r)) / (λ*σ_i[i])) for i in 1:num_agents) - N
    pt = last(price) # initial condition, last observed price 
    pt_iter = [] # More efficent way to do this, with no vector?

    # Solving for clearing price via newton's method
    for i in 1:itermax
        global pt = pt - (f(pt) / ForwardDiff.derivative(f, pt))
        push!(pt_iter, pt) # this makes price add 500 elements to the vec each time... bad
    end
    cprice = last(pt_iter)

    test_demand_N_convergence = Vector{Float64}(undef, 0)
    for i in 1:num_agents
        demand = ((a[i]*(cprice + dt) + b[i] - cprice*(1 + r)) / (λ*σ_i[i]))
        push!(test_demand_N_convergence, demand)
    end
    sum(test_demand_N_convergence)

    # Rounding witchcraft
    for i in 1:length(test_demand_N_convergence)
        test_demand_N_convergence[i] = round(test_demand_N_convergence[i], digits = 1)
    end

    # Way to do this without making this vector every time?
    xi_excess = Vector{Float64}(undef, 0)
    for i = 1:N
        if test_demand_N_convergence[i,1] > trade_restriction
            push!(xi_excess, (test_demand_N_convergence[i,1] - (test_demand_N_convergence[i,1] - trade_restriction)))
        elseif test_demand_N_convergence[i,1] < short_restriction
            push!(xi_excess, (test_demand_N_convergence[i,1] - (test_demand_N_convergence[i,1] - short_restriction)))
        else
            push!(xi_excess, test_demand_N_convergence[i,1])
        end
    end
    test_demand_N_convergence = xi_excess

    # To determine rounding difference and adjust share rationing
    test_demand_round_diff = Vector{Float64}(undef, 0)
    append!(test_demand_round_diff, test_demand_N_convergence)

    for i in 1:length(test_demand_N_convergence)
        test_demand_round_diff[i] = trunc(Int64, test_demand_round_diff[i])
    end
    round_error = sum(test_demand_round_diff)
    round_error = convert(Int, round_error)
    test_demand_round_diff

    demand_ration = hcat(test_demand_N_convergence, test_demand_round_diff)

    # round difference
    ration_imbalance = N - round_error 
    demand_ration_imbalance = vec(diff(demand_ration, dims=2))
    demand_ration_imbalance .= round.(demand_ration_imbalance, digits = 1)
    demand_ration = hcat(demand_ration, demand_ration_imbalance)

    df = DataFrame(AgentID = Identifier, Current_holding = relative_holdings, Current_cash = relative_cash, 
        init_xi = demand_ration[:, 1], round_xi = demand_ration[:, 2], xi_diff = demand_ration[:, 3])

    if ration_imbalance > 0
        # shuffle and sorts agents by xi_diff, largest negative value to highest positive value
        df = df[shuffle(1:nrow(df)), :]
        sort!(df, :xi_diff)
    elseif ration_imbalance < 0
        # shuffle and sorts agents by xi_diff, largest positive value to highest negative value
        df = df[shuffle(1:nrow(df)), :]
        sort!(df, order(:xi_diff), rev=true)
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
                elseif df[j, :demand_xi] <= short_restriction
                    continue
                else
                    df[j, :demand_xi] -= 1.0
                    overbought_shares -= 1
                end
            end
            df = df[shuffle(1:nrow(df)), :]
            sort!(df, order(:xi_diff), rev=true)
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
            sort!(df, order(:xi_diff), rev=true)
            for j = 1:Int((ration_imbalance - i + 1))
                if oversold_shares == 0
                    break
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
    df[!,:demand_xi] = convert.(Int, df[:,:demand_xi])
    ration_imbalance = N - sum(df[:, :demand_xi])
    cprice = round(cprice; digits = 2) # any issues with doing this?
    df[!, :clearing_price] = [cprice for i in 1:nrow(df)]
    return df
end


"""
Collect new agent share amount, cash and calculate shares traded, agent profit at time t 

Enforce cash constraint `X` and return df with adjusted agent metrics
If the budget constraint is violated:
    -Buy as many as allowed, and the remaining leftover share(s) goes to agent(s) with largest short position
    -All agent metrics are then balanced to ensure global cash, shares, etc. are conserved

ERROR TERMS TO INCLUDE LATER**
- Conservation tests for total cash, profit, demand, and shares traded.
- Check again for passing of trade constraints in case there is adjustment
- Additional final check to ensure all nonnegative cash values
"""
function get_trades!(df, cash_restriction)
    # The :shares_traded column needed for trading volume vector
    df[!, :shares_traded] = [(df[i, :demand_xi] - df[i, :Current_holding]) for i in 1:nrow(df)]
    df[!,:shares_traded] = convert.(Float64, df[:,:shares_traded])
    cprice = df[1, :clearing_price]
    # The :profit column is specific to time t, different from net profit 
    df[!,:profit_t] = [(df[i, :shares_traded] * cprice * -1) for i in 1:nrow(df)]
    df[:,:Current_cash] = [(df[i,:Current_cash] + df[i, :profit_t]) for i in 1:nrow(df)]

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
                    if getindex(df[j, :demand_xi]) == minimum(df[:, :demand_xi])
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


# ## Agent Updating (done for each agent individually)

# """
# Update individual agent financial metrics

# From order execution output, update `relative_cash` & `relative_holdings`
# """
# function update_rewards!()
    
# end


# """
# Update accuracy of each active predictor. 
# """
 
# function update_predict_acc!!(predictors, price, dividend)

# end


# """
# Update accuracy of each active predictor. 
# """

# function update_fitness_j!()
    
# end


# **SAVING GA STUFF FOR AFTER INTEGRATION TESTING
# """
# Check recombination status. 
    
# Will use predictor accuracy, fitness measure, and `δ` to determine if selected for recombination via GA.

# **Return boolean here, followed by if statment holding GA function? 
# ** if statement t - last_t_δ == δ
# """
# function check_recombination()
    
# end




# ### -------IRRELEVANT PREV ABM STUFF------- ###

# """
# Set initial values for δ depending on agent value type.
# """
# function init_delta(status)
#     δ_dict = Dict(
#         :ST => 2/3,
#         :O => 1,
#         :SE => 2/3,
#         :C => 1/3
#     )
#     return δ_dict[status]
# end


# """
#     `find_peers!(agent, model) → ids`

# Construct agent's peer group depending on the given `model.env`.

# - `"Global"` → Collect all agent ids in the model in a list, including self.
# - `"Neighbours"` → Collect local neighbours of `agent` in a list, excluding self. Number of 
# neighbours depends on chosen GridSpace (i.e. `periodic` and `metric` keywords).
# - `"Random"` → Collect a random list of agent ids with length `model.num_peers`, excluding 
# self.
# """
# function find_peers!(agent, model)
#     if model.env == "Global"
#         agent.peers = collect(allids(model))
#     elseif model.env == "Neighbours"
#         agent.peers = collect(nearby_ids(agent, model))
#     elseif model.env == "Random"
#         agent.peers = @pipe collect(allids(model)) |>
#             filter!(id -> id != agent.id, _) |>
#             shuffle!(model.rng, _) |>
#             rand(model.rng, _, model.num_peers)
#     end
# end

# ## Model logic

# """
#     `update_norm_coop!(agent, model)`

# Update cooperation norm perceived by agent.
# Sums up cooperation times of agent's peers (see `find_peers()`) and calculates their mean.
# Effective adjustment of agent's cooperation norm is modified by `model.h`.
# """
# function update_norm_coop!(agent, model)
#     agent.norm_coop = 
#         (1-model.h) * agent.norm_coop + 
#         model.h * mean(model[id].time_cooperation for id in agent.peers)
# end

# """
#     `update_norm_shirk!(agent, model)`

# Update shirking norm perceived by agent.
# Sums up shirking times of agent's peers (see `find_peers()`) and calculates their mean.
# Effective adjustment of agent's shirking norm is modified by `model.h`.
# """
# function update_norm_shirk!(agent, model)
#     agent.norm_shirk = 
#         (1-model.h) * agent.norm_shirk + 
#         model.h * mean(model[id].time_shirking for id in agent.peers)
# end

# """
# Update agent's need for autonomy (ϕ).
# """
# function update_phi!(agent, Σ)
#     if Σ == 0.5 # no monitoring
#         agent.ϕ = 0.0
#     else # monitoring 
#         if agent.status == :O
#             agent.ϕ = (Σ - 0.5) * agent.norm_shirk * agent.δ
#         elseif agent.status == :C
#             agent.ϕ = (0.5 - Σ) * agent.norm_shirk * agent.δ
#         else
#             agent.ϕ = 0.0
#         end
#     end
# end

# """
# Update agent's willingness to cooperate (γ).
# """
# function update_gamma!(agent)
#     if agent.status == :ST
#         agent.γ = 0.5 * agent.norm_coop * agent.δ
#     elseif agent.status == :SE
#         agent.γ = -0.5 * agent.norm_coop * agent.δ
#     else
#         agent.γ = 0.0
#     end
# end

# """
# Update agent's responsiveness to rewards (ρ).
# """
# function update_rho!(agent, λ)
#     if agent.status == :SE
#         if λ == 1.0 # cooperative
#             agent.ρ = 0.1 * agent.norm_coop * agent.δ
#         else # competitive
#             agent.ρ = -0.5 * agent.norm_coop * agent.δ
#         end
#     elseif agent.status == :ST
#         if λ == 1.0
#             agent.ρ = 0.5 * agent.norm_coop * agent.δ
#         else 
#             agent.ρ = -0.1 * agent.norm_coop * agent.δ
#         end
#     else
#         agent.ρ = 0.0
#     end
# end

# """
# Update time that an agent spends on shirking.

# Create triangular distribution for agent and make a random draw from that distribution.
# Calculate shirking time and bound it between the natural limits of 0 and (residual) τ.
# """
# function spend_time_shirking!(agent, τ, rng)
#     a = agent.norm_shirk - agent.norm_shirk * agent.δ
#     b = agent.norm_shirk + agent.norm_shirk * agent.δ
#     c = agent.norm_shirk + agent.ϕ
#     dist = TriangularDist(a, b, c)
#     agent.time_shirking = @pipe rand(rng, dist) |>
#         max(_, 0) |> # 0 or higher
#         min(_, τ) # τ or lower
# end

# """
# Update time that an agent spends on cooperation.

# Create triangular distribution for agent and make a random draw from that distribution.
# Calculate cooperation time and bound it between the natural limits of 0 and (residual) τ.
# """
# function spend_time_cooperation!(agent, τ, rng)
#     a = agent.norm_coop - agent.norm_coop * agent.δ
#     b = agent.norm_coop + agent.norm_coop * agent.δ
#     c = agent.norm_coop + agent.γ + agent.ρ
#     dist = TriangularDist(a, b, c)
#     agent.time_cooperation = @pipe rand(rng, dist) |>
#         max(_, 0) |> # 0 or higher
#         min(_, τ) # τ or lower
# end

# """
# Update time that an agent spends on individual tasks.

# Calculate time spent on individual tasks dependent on maximum working time,
# cooperation time and shirking time.
# """
# function spend_time_individual!(agent, τ)
#     agent.time_individual = τ - agent.time_cooperation - agent.time_shirking
#     isapprox(agent.time_individual, 0.0, atol=10^-15) && (agent.time_individual = 0.0)
# end

# """
# Update agent's deviations from cooperation and shirking norm.
# """
# function update_deviations!(agent)
#     agent.deviation_norm_coop = agent.time_cooperation - agent.norm_coop
#     agent.deviation_norm_shirk = agent.time_shirking - agent.norm_shirk
# end

# """
# Update agent's output based on task interdependency κ, agent's time spent on individual
# tasks and the mean cooperation time of all employees.
# """
# function update_output!(agent, κ, mean_coop)
#     agent.output = agent.time_individual ^ (1 - κ) * mean_coop ^ κ
# end


# """
# Update agents' realised output as a percentage ratio of their own output and the 
# optimal group output (OGO).
# """
# function update_realised_output!(agent, OGO)
#     agent.realised_output = (agent.output / OGO) * 100 
# end

# """
# Update agents' realised output as a percentage ratio of their own output and the 
# highest individual output of all agents.
# """

# function update_realised_output_max!(agent, max_output)
#     agent.realised_output_max = (agent.output / max_output) * 100 
# end

# """
# Update agent's reward according to scenario parameters.

# If no PFP schemes are implemented (μ = 0.0), then only the base wage (ω) is paid.
# Otherwise PFP schemes result in two possible reward calculations on top of the base wage (ω):
# - Competitive settings pay for personal output (see `update_output!`).
# - Cooperative settings pay for mean output across all agents.
# """
# function update_rewards!(agent, μ, λ, base_wage, mean_output)
#     agent.reward = base_wage + μ * ((1 - λ) * agent.output + λ * mean_output)
# end

# """ 
# Update Gini Coefficient of reward inequality within the population of employees. 

# Since the data consists only of positive values, the equation implemented 
# as a simplification of the usual notation based on the relative mean difference. 
# See https://www.statsdirect.com/help/nonparametric_methods/gini_coefficient.htm. 
# """
# function update_gini_index!(model)
#     sorted_reward = (agent.reward for agent in allagents(model)) |>
#         collect |>
#         sort
#     share = sorted_reward ./ sum(sorted_reward)
#     n = length(share)
#     index = 1:1:n |> collect
#     coeff = 2 .* index .- n .- 1
#     model.gini_index = sum(coeff .* share) / (n .* sum(share))
# end   

end
 
