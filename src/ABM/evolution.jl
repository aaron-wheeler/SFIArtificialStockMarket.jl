module evolution

using Agents
using Distributions
#using Pipe
using Random
using StatsBase
using ForwardDiff
using DataFrames

## Update Market State 

"""
    `dividend_process() → dividend`

Autoregressive dividend process is appended to vector and made public to all agents
Gaussian noise term `ε` is independent & identically distributed and has zero mean and variance σ_ε
"""
function dividend_process(d̄, ρ, dividend, σ_ε)
    ε = rand(Normal(0.0,σ_ε))
    dt = d̄ + ρ*(last(dividend) - d̄) + ε
    dividend = push!(dividend, dt)
    return dividend
end

"""
    `update_market_vector() → bit 1-12`

Update global market state bit vector, assign "1" or "0" values depending on the presence of bit signals
"""
function update_market_vector(bit1, bit2, bit3...)
    # if 
        
    # end

    # return bit1, bit2, bit3... # is it possible to do this?
end

## Initialization

"""
    `init_predictors() → predictors`

Agent `predictors` vector is coupled to unique `id`.
"""

function init_predictors(num_predictors) # Add an identifier? 
    predictors = Vector{Any}(undef, 0) 
    for i in 1:(num_predictors-1) # minus one so that we can add default predictor
        heterogeneity = Vector{Any}(undef, 3)
        heterogeneity[1] = rand(Uniform(0.7,1.2)) # a
        heterogeneity[2] = rand(Uniform(-10.0, 19.002)) # b 
        heterogeneity[3] = σ_pd # initial σ_i = σ_pd
        bit_vec = Vector{Any}(undef, 12)
        sample!([missing, 1, 0], Weights([0.9, 0.05, 0.05]), bit_vec)
        bit_vec = vcat(heterogeneity, bit_vec)
        predictors = push!(predictors, bit_vec)
    end
    return predictors
end

# Add step for "Default" predictor.....

"""
    `init_learning() → `

Constructs and initializes each agent's `predict_acc`, 'fitness_j`, and `δ` coupled to unique `id`.
"""

function init_learning(N,δ_dist)   # Add an identifier? Seperate into 2 vectors (otherwise remove a,b)?
    #predict_acc = Vector{Any}(undef, 0) # Put this step somewhere else? How to initialize this? return?
    #fitness_j = Vector{Any}(undef, 0) # Put this step somewhere else?
    δ = Vector{Any}(undef, N)
    sample!(δ_dist, δ; replace=false, ordered=false)
end

## Order Execution Mechanism 

"""

- Determine which predictors are active based on market state 
- Among the active predictors, select the one with the highest fitness measure
- From this predictor, return a matrix `forecast` composed of a, b, σ_i, and agent ID 
"""

function match_predictor()
    
    return forecast
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
function get_demand!(num_agents, N, price, dividend, r, λ, forecast, relative_cash, relative_holdings, 
        trade_restriction, short_restriction, itermax)
    dt = last(dividend)
    Identifier = convert(Vector{Int}, forecast[:, 1])
    a = forecast[:, 2]
    b = forecast[:, 3] 
    σ_i = forecast[:, 4]
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


## Agent Updating

"""
Calculate expected price and dividend forecasts. 

This function is employed after clearing p is obtained. Needed for determining indv agent demand and updating acc
"""
 
function update_exp!(predictors, price, dividend) # only done for predictors that are active, this must be implemented beforehand? Initial?
    expected_pd = Vector{Float64}(undef, 0) 
    for i in 1:(length(predictors)+1) # add one to include the default pred.... (add default to predictors and remove +1?)
        linear_pd_forecast = (predictors[i][1])*(price + dividend) + (predictors[i][2])
        push!(expected_pd, linear_pd_forecast)
    end
    # select expected_pd belonging to the active predictor with the highest accuracy
    # return single_expected_pd
end






### -------IRRELEVANT PREV ABM STUFF------- ###

"""
Set initial values for δ depending on agent value type.
"""
function init_delta(status)
    δ_dict = Dict(
        :ST => 2/3,
        :O => 1,
        :SE => 2/3,
        :C => 1/3
    )
    return δ_dict[status]
end


"""
    `find_peers!(agent, model) → ids`

Construct agent's peer group depending on the given `model.env`.

- `"Global"` → Collect all agent ids in the model in a list, including self.
- `"Neighbours"` → Collect local neighbours of `agent` in a list, excluding self. Number of 
neighbours depends on chosen GridSpace (i.e. `periodic` and `metric` keywords).
- `"Random"` → Collect a random list of agent ids with length `model.num_peers`, excluding 
self.
"""
function find_peers!(agent, model)
    if model.env == "Global"
        agent.peers = collect(allids(model))
    elseif model.env == "Neighbours"
        agent.peers = collect(nearby_ids(agent, model))
    elseif model.env == "Random"
        agent.peers = @pipe collect(allids(model)) |>
            filter!(id -> id != agent.id, _) |>
            shuffle!(model.rng, _) |>
            rand(model.rng, _, model.num_peers)
    end
end

## Model logic

"""
    `update_norm_coop!(agent, model)`

Update cooperation norm perceived by agent.
Sums up cooperation times of agent's peers (see `find_peers()`) and calculates their mean.
Effective adjustment of agent's cooperation norm is modified by `model.h`.
"""
function update_norm_coop!(agent, model)
    agent.norm_coop = 
        (1-model.h) * agent.norm_coop + 
        model.h * mean(model[id].time_cooperation for id in agent.peers)
end

"""
    `update_norm_shirk!(agent, model)`

Update shirking norm perceived by agent.
Sums up shirking times of agent's peers (see `find_peers()`) and calculates their mean.
Effective adjustment of agent's shirking norm is modified by `model.h`.
"""
function update_norm_shirk!(agent, model)
    agent.norm_shirk = 
        (1-model.h) * agent.norm_shirk + 
        model.h * mean(model[id].time_shirking for id in agent.peers)
end

"""
Update agent's need for autonomy (ϕ).
"""
function update_phi!(agent, Σ)
    if Σ == 0.5 # no monitoring
        agent.ϕ = 0.0
    else # monitoring 
        if agent.status == :O
            agent.ϕ = (Σ - 0.5) * agent.norm_shirk * agent.δ
        elseif agent.status == :C
            agent.ϕ = (0.5 - Σ) * agent.norm_shirk * agent.δ
        else
            agent.ϕ = 0.0
        end
    end
end

"""
Update agent's willingness to cooperate (γ).
"""
function update_gamma!(agent)
    if agent.status == :ST
        agent.γ = 0.5 * agent.norm_coop * agent.δ
    elseif agent.status == :SE
        agent.γ = -0.5 * agent.norm_coop * agent.δ
    else
        agent.γ = 0.0
    end
end

"""
Update agent's responsiveness to rewards (ρ).
"""
function update_rho!(agent, λ)
    if agent.status == :SE
        if λ == 1.0 # cooperative
            agent.ρ = 0.1 * agent.norm_coop * agent.δ
        else # competitive
            agent.ρ = -0.5 * agent.norm_coop * agent.δ
        end
    elseif agent.status == :ST
        if λ == 1.0
            agent.ρ = 0.5 * agent.norm_coop * agent.δ
        else 
            agent.ρ = -0.1 * agent.norm_coop * agent.δ
        end
    else
        agent.ρ = 0.0
    end
end

"""
Update time that an agent spends on shirking.

Create triangular distribution for agent and make a random draw from that distribution.
Calculate shirking time and bound it between the natural limits of 0 and (residual) τ.
"""
function spend_time_shirking!(agent, τ, rng)
    a = agent.norm_shirk - agent.norm_shirk * agent.δ
    b = agent.norm_shirk + agent.norm_shirk * agent.δ
    c = agent.norm_shirk + agent.ϕ
    dist = TriangularDist(a, b, c)
    agent.time_shirking = @pipe rand(rng, dist) |>
        max(_, 0) |> # 0 or higher
        min(_, τ) # τ or lower
end

"""
Update time that an agent spends on cooperation.

Create triangular distribution for agent and make a random draw from that distribution.
Calculate cooperation time and bound it between the natural limits of 0 and (residual) τ.
"""
function spend_time_cooperation!(agent, τ, rng)
    a = agent.norm_coop - agent.norm_coop * agent.δ
    b = agent.norm_coop + agent.norm_coop * agent.δ
    c = agent.norm_coop + agent.γ + agent.ρ
    dist = TriangularDist(a, b, c)
    agent.time_cooperation = @pipe rand(rng, dist) |>
        max(_, 0) |> # 0 or higher
        min(_, τ) # τ or lower
end

"""
Update time that an agent spends on individual tasks.

Calculate time spent on individual tasks dependent on maximum working time,
cooperation time and shirking time.
"""
function spend_time_individual!(agent, τ)
    agent.time_individual = τ - agent.time_cooperation - agent.time_shirking
    isapprox(agent.time_individual, 0.0, atol=10^-15) && (agent.time_individual = 0.0)
end

"""
Update agent's deviations from cooperation and shirking norm.
"""
function update_deviations!(agent)
    agent.deviation_norm_coop = agent.time_cooperation - agent.norm_coop
    agent.deviation_norm_shirk = agent.time_shirking - agent.norm_shirk
end

"""
Update agent's output based on task interdependency κ, agent's time spent on individual
tasks and the mean cooperation time of all employees.
"""
function update_output!(agent, κ, mean_coop)
    agent.output = agent.time_individual ^ (1 - κ) * mean_coop ^ κ
end


"""
Update agents' realised output as a percentage ratio of their own output and the 
optimal group output (OGO).
"""
function update_realised_output!(agent, OGO)
    agent.realised_output = (agent.output / OGO) * 100 
end

"""
Update agents' realised output as a percentage ratio of their own output and the 
highest individual output of all agents.
"""

function update_realised_output_max!(agent, max_output)
    agent.realised_output_max = (agent.output / max_output) * 100 
end

"""
Update agent's reward according to scenario parameters.

If no PFP schemes are implemented (μ = 0.0), then only the base wage (ω) is paid.
Otherwise PFP schemes result in two possible reward calculations on top of the base wage (ω):
- Competitive settings pay for personal output (see `update_output!`).
- Cooperative settings pay for mean output across all agents.
"""
function update_rewards!(agent, μ, λ, base_wage, mean_output)
    agent.reward = base_wage + μ * ((1 - λ) * agent.output + λ * mean_output)
end

""" 
Update Gini Coefficient of reward inequality within the population of employees. 

Since the data consists only of positive values, the equation implemented 
as a simplification of the usual notation based on the relative mean difference. 
See https://www.statsdirect.com/help/nonparametric_methods/gini_coefficient.htm. 
"""
function update_gini_index!(model)
    sorted_reward = (agent.reward for agent in allagents(model)) |>
        collect |>
        sort
    share = sorted_reward ./ sum(sorted_reward)
    n = length(share)
    index = 1:1:n |> collect
    coeff = 2 .* index .- n .- 1
    model.gini_index = sum(coeff .* share) / (n .* sum(share))
end   

end
 
