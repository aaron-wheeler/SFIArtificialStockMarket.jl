include("data_struct.jl") 
include("evolution.jl")

## Initialization

"""
    `init_model(; seed, properties...) → ABM`

Create ABM model with given `seed`, and other `properties`.
"""
function init_model(; seed::UInt32, properties...)
    space = GridSpace((10,10), periodic = true, metric = :euclidean )  # TODO: Investigate this
    model = ABM(
        Trader, 
        space; # This does nothing as of right now, brainstorm use (Power law wealth distribution space?)
        properties = ModelProperties(; properties...), 
        scheduler = Schedulers.randomly, # TODO: Investigate this
        rng = MersenneTwister(seed) # Is this used anywhere in simulation? dividend_process?
    )
    init_state!(model)
    println("Market State Initiated")
    init_agents!(model)
    println("Agents Initiated")
    return model

end

"""
Initialize market state.
"""
function init_state!(model)
    dividend = Vector{Float64}(undef, 0)
    init_dividend = model.d̄
    price = Vector{Float64}(undef, 0)
    init_price = init_dividend / model.r

    # # Initialize all these by making default value in data_struct.jl?
    # model.t = 1
    # model.bit1 = 0 
    # model.bit2 = 0
    # model.bit3 = 0
    # model.bit4 = 0
    # model.bit5 = 0
    # model.bit6 = 0
    # model.bit7 = 0
    # model.bit8 = 0
    # model.bit9 = 0
    # model.bit10 = 0
    # model.bit11 = 1
    # model.bit12 = 0
    
    model.price = push!(price, init_price)
    model.dividend = push!(dividend, init_dividend)

    # Include these?
    # model.trading_volume = Vector{Any}(undef, 0)
    # model.volatility = Vector{Any}(undef, 0)
    # model.technical_activity = Vector{Any}(undef, 0)
    
    # Initialization period, generate historical dividend and prices
    price_div_history_t = 1
    while price_div_history_t <= model.initialization_t
        model.dividend = evolution.dividend_process(model.d̄, model.ρ, model.dividend, model.σ_ε)
        model.price = push!(model.price, (last(model.dividend) / model.r))
        price_div_history_t += 1
    end

    # generate first state bit vector sequence
    bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10 = evolution.update_market_vector(model.price, model.dividend, model.r) # Have to append model.X to each bit?
    bit11 = 1
    bit12 = 0
    model.state_vector = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10, bit11, bit12]

    return model
end

"""
Initialize and add agents.

ERROR TERMS TO INCLUDE LATER**
- For future user-input case, may need to specify that `GA_frequency` (T/k) needs to be an Int
"""
function init_agents!(model)
    T = model.warm_up_t + model.recorded_t # Total sim time
    GA_frequency = Int(T / model.k) # num times GA is invoked across total simulation
    n = Int(GA_frequency / model.k_var) # scaling factor for consistent k range over time
    δ_dist_1 = repeat(Vector{Int}(((model.k - (model.k_var/2)) + 1) : (model.k - 1)), n) # - half of k_var
    δ_dist_2 = repeat([model.k, model.k], n) # middle portion of k_var
    δ_dist_3 = repeat(Vector{Int}((model.k + 1) : ((model.k + (model.k_var/2)) - 1)), n) # + half of k_var
    δ_dist = vcat(δ_dist_1, δ_dist_2, δ_dist_3)
    for id in 1:model.num_agents # Properties included in `Trader` here are ones that don't have default value in data_struct.jl or may be user changed later
        a = Trader(
            id = id, 
            pos = (1,1),
            relative_cash = model.init_cash,
            σ_i = model.σ_pd
        )
        
        a.predictors = evolution.init_predictors(model.num_predictors, model.σ_pd)
        a.δ, a.predict_acc, a.fitness_j = evolution.init_learning(GA_frequency, δ_dist, model.σ_pd, model.C, model.num_predictors, a.predictors)
        # a.active_predictors, a.forecast = evolution.match_predictors(a.id, model.num_predictors, a.predictors, model.state_vector, a.predict_acc, a.fitness_j)
        a.active_predictors = Vector{Int}(undef, 0)
        a.forecast = Vector{Any}(undef, 0)
        a.last_active_j = zeros(Int, model.num_predictors)

        add_agent_single!(a, model) 
    end

    return model
end

## Stepping

"""
Define what happens in the model.
"""
function model_step!(model)
    scheduled_agents = (model[id] for id in model.scheduler(model))

    # Exogenously determine dividend and post for all agents
    model.dividend = evolution.dividend_process(model.d̄, model.ρ, model.dividend, model.σ_ε)

    # Update market state vector
    bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10 = evolution.update_market_vector(model.price, model.dividend, model.r)
    bit11 = 1
    bit12 = 0
    model.state_vector = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10, bit11, bit12]

    # Agent expectation steps
    for agent in scheduled_agents
        agent.active_predictors, agent.forecast = evolution.match_predictors(agent.id, model.num_predictors, agent.predictors, model.state_vector, agent.predict_acc, agent.fitness_j)
        evolution.update_tracking_j!(model.num_predictors, agent.active_predictors, agent.last_active_j, model.t)
    end

    # Collect demands of all individual agents and return aggregate forecast matrix `expected_xi`
    expected_xi = zeros(Float64, 4, 0)
    relative_cash_t = Vector{Float64}(undef, 0)
    relative_holdings_t = Vector{Int}(undef, 0)
    
    for agent in scheduled_agents # will this work in agent id order? What is scheduled_agents order? -> Set as `randomly` rn
        # safer to do this all in one data structure and then sort by agent id in next step to ensure consistency?
        expected_xi = hcat(expected_xi, agent.forecast)
        relative_cash_t = push!(relative_cash_t, agent.relative_cash)
        relative_holdings_t = push!(relative_holdings_t, agent.relative_holdings)

    end

    # check order consistency of expected_xi[1,:] and relative_cash, relative_holdings

    # Price formation mechanism
    df_demand, clearing_price = evolution.get_demand!(model.num_agents, model.N, model.price, model.dividend, model.r, model.λ, expected_xi, relative_cash_t, relative_holdings_t, 
        model.trade_restriction, model.short_restriction, model.itermax)

    # Update price vector
    model.price = push!(model.price, clearing_price)

    # Order execution mechanism here, get_trades()
    df_trades = evolution.get_trades!(df_demand, clearing_price, model.cash_restriction)

    # Update trading volume vector
    evolution.update_trading_volume!(model.num_agents, df_trades, model.trading_volume)

    # Update historical volatility vector
    evolution.update_volatility!(model.price, model.volatility)

    # Calculate and update individual agent financial rewards (cash and holdings)
    for agent in scheduled_agents
        evolution.update_rewards!(df_trades, agent)
    end

    # Update agent forecasting metrics 
    for agent in scheduled_agents
        evolution.update_predict_acc!(agent, model.τ, model.price, model.dividend)
    end


    # **SAVING GA STUFF FOR AFTER INTEGRATION TESTING
    # Check recombination status for individual agent, and if true, then undergo GA 
    # for agent in scheduled_agents

    #     # if rand(model.rng, Bool)
    #     #     InVaNo.spend_time_shirking!(agent, model.τ, model.rng)
    #     #     residual_τ = model.τ - agent.time_shirking
    #     #     InVaNo.spend_time_cooperation!(agent, residual_τ, model.rng)
    #     # else
    #     #     InVaNo.spend_time_cooperation!(agent, model.τ, model.rng)
    #     #     residual_τ = model.τ - agent.time_cooperation
    #     #     InVaNo.spend_time_shirking!(agent, residual_τ, model.rng)
    #     # end
    #     # InVaNo.spend_time_individual!(agent, model.τ)
    #     # InVaNo.update_deviations!(agent)
    # end

    # Increment time step
    model.t += 1

    return model
end
