# include("data_struct.jl") 
# include("SFIArtificialStockMarket.jl")

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
    # instantiate price and dividend
    dividend = Vector{Float64}(undef, 0)
    init_dividend = model.d̄
    price = Vector{Float64}(undef, 0)
    init_price = init_dividend / model.r    
    model.price = push!(price, init_price)
    model.dividend = push!(dividend, init_dividend)
    
    # Initialization period, generate historical dividend and prices
    price_div_history_t = 1
    while price_div_history_t <= model.initialization_t
        model.dividend = SFIArtificialStockMarket.dividend_process(model.d̄, model.ρ, model.dividend, model.σ_ε)
        model.price = push!(model.price, (last(model.dividend) / model.r))
        price_div_history_t += 1
    end

    # generate first state bit vector sequence
    model.state_vector = SFIArtificialStockMarket.update_market_vector(model.price, model.dividend, model.r)

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
            pos = (1, 1),
            relative_cash = model.init_cash,
            relative_wealth = model.init_cash + last(model.price) * 1, # wealth = cash + price*holdings
            σ_i = model.σ_pd,
            # active_j_records = zeros(Int, model.num_predictors, 2)
        )
    
        a.predictors = SFIArtificialStockMarket.init_predictors(model.num_predictors, model.σ_pd, model.a_min, model.a_max, model.b_min, model.b_max)
        a.δ, a.predict_acc, a.fitness_j = SFIArtificialStockMarket.init_learning(GA_frequency, δ_dist, model.σ_pd, model.C, model.num_predictors, a.predictors)
        # a.active_predictors, a.forecast = SFIArtificialStockMarket.match_predictors(a.id, model.num_predictors, a.predictors, model.state_vector, a.predict_acc, a.fitness_j)
        a.active_predictors = Vector{Int}(undef, 0)
        a.forecast = Vector{Any}(undef, 0)
        a.active_j_records = zeros(Int, model.num_predictors, 2)
    
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
    model.dividend = SFIArtificialStockMarket.dividend_process(model.d̄, model.ρ, model.dividend, model.σ_ε)

    # Adjust agent dividend and fixed income asset earnings and pay taxes
    for agent in scheduled_agents
        # Update cash
        agent.relative_cash += model.r * agent.relative_cash + agent.relative_holdings * last(model.dividend) # risk-free asset earnings + risky asset earnings
        agent.relative_cash -= agent.relative_wealth * model.r # pay taxes on previous total wealth
    
        # Update wealth
        agent.relative_wealth = agent.relative_cash + last(model.price) * agent.relative_holdings
    end

    # Update market state vector
    model.state_vector = SFIArtificialStockMarket.update_market_vector(model.price, model.dividend, model.r)

    # Agent expectation steps
    for agent in scheduled_agents
        agent.active_predictors, agent.forecast = SFIArtificialStockMarket.match_predictors(agent.id, model.num_predictors, agent.predictors, model.state_vector, agent.predict_acc, agent.fitness_j)
        SFIArtificialStockMarket.update_active_j_records!(model.num_predictors, agent.active_predictors, agent.active_j_records, model.t)
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
    df_demand, clearing_price = SFIArtificialStockMarket.get_demand!(model.num_agents, model.N, model.price, model.dividend, model.r, model.λ, expected_xi, relative_cash_t, relative_holdings_t,
        model.trade_restriction, model.short_restriction, model.itermax, model.price_min, model.price_max)

    # Update price vector
    model.price = push!(model.price, clearing_price)

    # Order execution mechanism here, get_trades()
    df_trades = SFIArtificialStockMarket.get_trades!(df_demand, clearing_price, model.cash_restriction)

    # Update trading volume vector
    SFIArtificialStockMarket.update_trading_volume!(model.num_agents, df_trades, model.trading_volume)

    # Update historical volatility vector
    SFIArtificialStockMarket.update_volatility!(model.price, model.volatility)

    # Calculate and update individual agent financial rewards (cash and holdings)
    for agent in scheduled_agents
        SFIArtificialStockMarket.update_rewards!(df_trades, agent)
        agent.relative_wealth = agent.relative_cash + clearing_price * agent.relative_holdings
    end

    # Update agent forecasting metrics 
    for agent in scheduled_agents
        SFIArtificialStockMarket.update_predict_acc!(agent.predict_acc, agent.active_predictors, agent.predictors, model.τ, model.price, model.dividend)
    end

    # Each individual agent checks to see if they are to be selected for GA 
    for agent in scheduled_agents
        # Check recombination status, undergo GA if true
        if in.(model.t, Ref(agent.δ)) == true
            # Begin GA
            for i = 1:length(agent.predict_acc)
                # Update variance estimate of each predictor `σ_j` to its current active value `predict_acc`
                agent.predictors[i][3] = agent.predict_acc[i]
                # Update fitness measure of each predictor using new `σ_j`
                s = count(!ismissing, agent.predictors[i][4:15])
                f_j = -1 * (agent.predict_acc[i]) - model.C * s
                agent.fitness_j[i] = f_j
            end

            # Worst performing (least fit) `num_elimination` predictors are collected for elimination
            eliminated_j = sortperm(agent.fitness_j[1:99])[1:model.num_elimination] # excluding default vec

            # Make new 1-100 organized dataframe for these next steps
            df_GA = DataFrame(predict_acc = agent.predict_acc, fitness_j = agent.fitness_j, predictors = agent.predictors)
            #allowmissing!(df_GA)

            # Retain rows not included in eliminated_j for new `elite` vectors
            # Isolate elite predictors, predict_acc, fitness_j and set eliminated rows to NaN (preserves type)
            for index = 1:nrow(df_GA)
                if in.(index, Ref(eliminated_j)) == true
                    df_GA[index, :] = fill(NaN, ncol(df_GA))
                end
            end

            # Construct vector for elite predictors
            elite_j = setdiff(1:100, eliminated_j)

            # Make 20 new predictors using GA procedure as `replacement_j`
            replacement_j = Vector{Any}(undef, 0)

            # Invoke one of the two possible GA procedures
            if rand() ≤ model.pGAcrossover
                for i = 1:model.num_elimination
                    crossed_j = SFIArtificialStockMarket.GA_crossover(elite_j, df_GA, agent.active_j_records)
                    replacement_j = push!(replacement_j, crossed_j)
                end
            else
                for i = 1:model.num_elimination
                    mutated_j = SFIArtificialStockMarket.GA_mutation(elite_j, df_GA, model.pcond_mut, model.a_min, model.a_max, model.b_min, model.b_max, model.pparam_mut_long, model.pparam_mut_short,
                        model.percent_mut_short)
                    replacement_j = push!(replacement_j, mutated_j)
                end
            end
            #println(replacement_j)

            # Merge `replacement_j` into `elite_j` using the eliminated indices from `eliminated_j`
            sort!(eliminated_j)
            j = 1
            for index = 1:nrow(df_GA)
                if in.(index, Ref(eliminated_j)) == true
                    # Input replacement predictors
                    df_GA[index, :predictors] = replacement_j[j]
                    # make new predict_acc equal to current replacement_j variance
                    df_GA[index, :predict_acc] = replacement_j[j][3]
                    # calculate new fitness_j for each new replacement vec, same procedure as done in initialization
                    s = count(!ismissing, df_GA[index, :predictors][4:15])
                    f_j = -1 * (df_GA[index, :predict_acc]) - model.C * s
                    df_GA[index, :fitness_j] = f_j
                    # update active_j_records for new replacement predictors
                    agent.active_j_records[index, 1] = 0
                    agent.active_j_records[index, 2] = model.t
                    j += 1
                end
            end

            # update default predictor forecasting params to be weighted (1/σ_j) average of all other predictor forecasting params (a, b)
            df_GA[100, :predictors][1] = sum(df_GA[i, :predictors][1] * (1 / df_GA[i, :predictors][3])
                                             for i = 1:(model.num_predictors-1)) / sum(1 / df_GA[i, :predictors][3] for i = 1:(model.num_predictors-1)) # default a
            df_GA[100, :predictors][2] = sum(df_GA[i, :predictors][2] * (1 / df_GA[i, :predictors][3])
                                             for i = 1:(model.num_predictors-1)) / sum(1 / df_GA[i, :predictors][3] for i = 1:(model.num_predictors-1)) # default b

            # Complete GA procedure and update respective Agent attributes from df_GA
            agent.predictors = df_GA[:, :predictors]
            agent.predict_acc = df_GA[:, :predict_acc]
            agent.fitness_j = df_GA[:, :fitness_j]
        end
    end

    # Each individual agent checks to see if they are to undergo generalization procedure
    for agent in scheduled_agents
        # Check to see if enough time has passed for generalization to apply
        if model.t > model.generalization_t
            for i = 1:model.num_predictors
                # predictors that haven't been matched for over `generalization_t` time steps are generalized
                if agent.active_j_records[i, 2] < (model.t - model.generalization_t)
                    # Generalization modifies 1/4 of all "set" bits 0/1 by setting them to `missing` and setting fitness_j to median value
                    s = count(!ismissing, agent.predictors[i][4:15]) # total number of "set" bits
                    # Determining 1/4 of integer number of bits takes the "ceiling" of the value, as per original source code
                    num_gen_bits = div(s, 4, RoundUp) # number of bits to generalize
                    for bit = 1:num_gen_bits
                        set_bit = findfirst(!ismissing, agent.predictors[i][4:15])
                        agent.predictors[i][3+set_bit] = missing # change "set" bit position in condition statement to missing
                    end
                    # fitness_j of predictor reset to median value
                    agent.fitness_j[i] = median(agent.fitness_j)
                end
            end
        end
    end

    # Update mdf collection variables (have to do this bc these are vectors) **TODO: Reconsider having these as growing vectors in first place?? Make just big enough to for state_vector?
    model.mdf_price = last(model.price)
    model.mdf_dividend = last(model.dividend)
    model.mdf_trading_volume = last(model.trading_volume)
    model.mdf_volatility = last(model.volatility)

    # Time tracking print messages (for debugging)
    if model.t % 10000 == 0
        println(model.t)
    end

    # Increment time step
    model.t += 1

    return model
end
