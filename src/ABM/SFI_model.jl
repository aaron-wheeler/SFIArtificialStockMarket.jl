include("SFI_utilities.jl")

## Initialization

"""
    `init_model(; seed, properties...) → ABM`

Create ABM model with given `seed`, and other `properties`.
"""
function init_model(; seed::UInt32, properties...)
    space = GridSpace((10,10), periodic = true, metric = :euclidean )
    model = ABM(
        Trader, 
        space; # currenly not used in simulation
        properties = ModelProperties(; properties...), 
        scheduler = Schedulers.by_id,
        rng = MersenneTwister(seed)
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
    init_dividend = model.d̄
    init_price = init_dividend / model.r    
    model.price = push!(model.price, init_price)
    model.dividend[2] = init_dividend
    # Initialization period, generate historical dividend and price
    price_div_history_t = 1
    while price_div_history_t <= model.initialization_t
        dividend_process!(model.dividend, model.d̄, model.ρ, model.σ_ε)
        model.price = push!(model.price, (last(model.dividend) / model.r))
        price_div_history_t += 1
    end
    # generate first market state vector sequence
    update_market_vector!(model.state_vector, model.price, model.dividend, model.r)

    return model
end

"""
Initialize and add agents.
"""
function init_agents!(model)
    # Populate model with agents one-by-one
    for id in 1:model.num_agents
        a = Trader(
            id = id,
            pos = (1, 1),
            relative_cash = model.init_cash,
            relative_wealth = model.init_cash + last(model.price) * 1 # multiply by 1 since each agent holds one share
        )
        a.predictors = init_predictors(model.num_predictors, model.σ_pd, model.a_min, model.a_max, model.b_min, model.b_max)
        a.predict_acc, a.fitness_j = init_learning(model.σ_pd, model.C, model.num_predictors, a.predictors)
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
    dividend_process!(model.dividend, model.d̄, model.ρ, model.σ_ε)
    # Adjust agent dividend and fixed income asset earnings and pay taxes
    for agent in scheduled_agents
        # Update cash
        agent.relative_cash += model.r * agent.relative_cash + agent.relative_holdings * last(model.dividend) # risk-free asset earnings + risky asset earnings
        agent.relative_cash -= agent.relative_wealth * model.r # pay taxes on previous total wealth
        if agent.relative_cash < model.cash_restriction
            agent.relative_cash = model.cash_restriction
        end
        # Update wealth
        agent.relative_wealth = agent.relative_cash + last(model.price) * agent.relative_holdings
    end

    # Update market state vector
    update_market_vector!(model.state_vector, model.price, model.dividend, model.r)

    # Agent expectation steps
    for agent in scheduled_agents
        agent.active_predictors, agent.forecast, agent.chosen_j = match_predictors(agent.id, model.num_predictors, agent.predictors, model.state_vector, agent.predict_acc, agent.fitness_j, model.σ_pd)
        update_active_j_records!(model.num_predictors, agent.active_predictors, agent.active_j_records, model.t)
    end

    # Collect demands of all individual agents and return aggregate forecast matrix `expected_xi`
    expected_xi = zeros(Float64, 4, 0)
    for agent in scheduled_agents
        expected_xi = hcat(expected_xi, agent.forecast)
    end

    # Prepare variables needed for order execution
    dt = last(model.dividend)
    a = convert(Vector{Float64}, expected_xi[2, :])
    b = convert(Vector{Float64}, expected_xi[3, :])
    σ_i = convert(Vector{Float64}, expected_xi[4, :])

    # AUCTIONEER-MEDIATED FRACTIONAL MARKET CLEARING ALGORITHM
    # Initialize
    slope_total = 0.0
    bid_total = 0.0
    ask_total = 0.0
    trial_price = 0.0
    bid_frac = 0.0
    ask_frac = 0.0
    volume = 0.0

    for nround in 1:model.max_rounds
        if nround == 1
            trial_price = last(model.price)
        else
            imbalance = bid_total - ask_total
            if (imbalance <= model.min_excess && imbalance >= -model.min_excess)
                break
            end
            # update price
            if slope_total != 0
                trial_price -= imbalance / slope_total
            else
                trial_price *= 1 + model.eta * imbalance
            end
        end

        # Set and enforce constraints on price variable
        if trial_price < model.price_min || trial_price > model.price_max
            trial_price = trial_price < model.price_min ? model.price_min : model.price_max
        end

        # Get agent's demand and sum bids, asks, and slopes
        slope_total = 0.0
        bid_total = 0.0
        ask_total = 0.0

        for agent in scheduled_agents
            slope = 0.0
            agent.demand_xi, slope = get_demand_slope(a[agent.id], b[agent.id], σ_i[agent.id], trial_price, dt, model.r, model.λ, agent.relative_holdings,
                model.trade_restriction, model.cash_restriction, model.short_restriction, agent.relative_cash)
            slope_total += slope
            if agent.demand_xi > 0.0
                bid_total += agent.demand_xi
            elseif agent.demand_xi < 0.0
                ask_total -= agent.demand_xi
            end
        end

        # Match up bids and asks (condition ? (return if true) : (return if false))
        volume = (bid_total > ask_total ? ask_total : bid_total)
        bid_frac = (bid_total > 0.0 ? volume / bid_total : 0.0)
        ask_frac = (ask_total > 0.0 ? volume / ask_total : 0.0)
    end
    clearing_price = trial_price

    # Complete trades 
    for agent in scheduled_agents
        if agent.demand_xi > 0.0
            agent.relative_holdings += agent.demand_xi * bid_frac
            agent.relative_cash -= agent.demand_xi * bid_frac * clearing_price
        elseif agent.demand_xi < 0.0
            agent.relative_holdings += agent.demand_xi * ask_frac
            agent.relative_cash -= agent.demand_xi * ask_frac * clearing_price
        end
        agent.relative_wealth = agent.relative_cash + clearing_price * agent.relative_holdings
    end

    # Update price vector
    model.price = push!(model.price, clearing_price)
    popfirst!(model.price)

    # Update trading volume
    model.trading_volume = volume

    # Update historical volatility
    model.volatility = update_volatility(model.price)

    # Update agent forecasting metrics
    if model.t > model.τ
        for agent in scheduled_agents
            update_predict_acc!(agent, model.τ, model.price, model.dividend)
        end
    end

    # Each individual agent checks to see if they are to be selected for GA 
    for agent in scheduled_agents
        # Check recombination status, undergo GA if true
        if model.t > model.τ && rand() < (1 / model.k) && model.t < model.warm_up_t
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

            # Retain rows not included in eliminated_j for new `elite` vectors
            # Isolate elite predictors, predict_acc, fitness_j and set eliminated rows to NaN (preserves type)
            for index = 1:nrow(df_GA)
                if in.(index, Ref(eliminated_j)) == true
                    df_GA[index, :] = fill(NaN, ncol(df_GA))
                end
            end

            # Construct vector for elite predictors
            elite_j = setdiff(1:100, eliminated_j)

            # Prepare vector for new replacement predictors
            replacement_j = Vector{Any}(undef, 0)

            # Invoke one of the two possible GA procedures
            if rand() ≤ model.pGAcrossover
                for i = 1:model.num_elimination
                    crossed_j = GA_crossover(elite_j, df_GA, agent.active_j_records)
                    replacement_j = push!(replacement_j, crossed_j)
                end
            else
                for i = 1:model.num_elimination
                    mutated_j = GA_mutation(elite_j, df_GA, model.pcond_mut, model.a_min, model.a_max, model.b_min, model.b_max, model.pparam_mut_long, model.pparam_mut_short,
                        model.percent_mut_short)
                    replacement_j = push!(replacement_j, mutated_j)
                end
            end

            # Merge `replacement_j` into `elite_j` using the eliminated indices from `eliminated_j`
            sort!(eliminated_j)
            j = 1
            for index = 1:nrow(df_GA)
                if in.(index, Ref(eliminated_j)) == true
                    # Input replacement predictors
                    df_GA[index, :predictors] = replacement_j[j]
                    # Make new predict_acc equal to current replacement_j variance
                    df_GA[index, :predict_acc] = replacement_j[j][3]
                    # Calculate new fitness_j for each new replacement vec, same procedure as done in initialization
                    s = count(!ismissing, df_GA[index, :predictors][4:15])
                    f_j = -1 * (df_GA[index, :predict_acc]) - model.C * s
                    df_GA[index, :fitness_j] = f_j
                    # Update active_j_records for new replacement predictors
                    agent.active_j_records[index, 1] = 0
                    agent.active_j_records[index, 2] = model.t
                    j += 1
                end
            end

            # Update default predictor forecasting params to be weighted (1/σ_j) average of all other predictor forecasting params (a, b)
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
                    # `fitness_j` of predictor reset to median value
                    agent.fitness_j[i] = median(agent.fitness_j)
                end
            end
        end
    end

    # Update values tracking set bits
    if model.track_bits == true
        # Initialize "set" bit count
        model.frac_bits_set = 0.0
        model.frac_bits_fund = 0.0
        model.frac_bits_tech = 0.0
        for agent in scheduled_agents
            model.frac_bits_set, model.frac_bits_fund, model.frac_bits_tech = update_frac_bits(agent.predictors)
        end
        # Average over all rules and agents
        model.frac_bits_set = model.frac_bits_set / (model.num_predictors * 12 * model.num_agents) # 12 total bits in predictor
        model.frac_bits_fund = model.frac_bits_fund / (model.num_predictors * 6 * model.num_agents) # 6 fundamental bits in predictor
        model.frac_bits_tech = model.frac_bits_tech / (model.num_predictors * 4 * model.num_agents) # 4 technical bits in predictor
    end
    
    # Update mdf collection variables
    model.mdf_price = clearing_price
    model.mdf_dividend = last(model.dividend)

    # Simulation progress tracking print messages
    if model.t % 10000 == 0 && model.print_progress == true
        println("$(model.t) time steps passed")
    end

    # Increment time step
    model.t += 1

    return model
end
