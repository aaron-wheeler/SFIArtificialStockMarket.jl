include("structs.jl") 
include("InVaNo.jl")

## Initialisation

"""
    `init_model(; seed, env, properties...) → ABM`

Create ABM model with given `seed`, `env`, and other `properties`.
"""
# function init_model(; seed::UInt32, env, properties...)
#     if env in ("Global", "Neighbours", "Random")
#         space = GridSpace((10,10), periodic = true, metric = :euclidean )
#         model = ABM(
#             Employee, 
#             space; 
#             properties = ModelProperties(; env, properties...), 
#             scheduler = Schedulers.randomly,
#             rng = MersenneTwister(seed)
#         )
#         model.dist = cumsum(model.dist)
#         init_agents!(model)
#         return model
#     else
#         error("Given env '$(env)' is not implemented in the model.
#             Please verify the integrity of the provided `env` value.")
#     end
# end

function init_model(; seed::UInt32, env, properties...)
    if env in ("Complex", "Rational") #Remove this element?
        space = GridSpace((10,10), periodic = true, metric = :euclidean )  # TODO: Investigate this
        model = ABM(
            Trader, 
            space; # Where to add 'State' structure? If not anywhere then add to Trader?
            properties = ModelProperties(; env, properties...), 
            scheduler = Schedulers.randomly, # TODO: Investigate this
            rng = MersenneTwister(seed) # TODO: Investigate this
        )
        model.dist = cumsum(model.dist)
        init_state!(model)
        init_agents!(model)
        return model
    else
        error("Given env '$(env)' is not implemented in the model.
            Please verify the integrity of the provided `env` value.")
    end
end

"""
Initialize market state.
"""

function init_state!(model)
    dividend = Vector{Float64}(undef, 0)
    init_dividend = model.d̄
    price = Vector{Float64}(undef, 0)
    init_price = init_dividend / model.r
    state = State(
        t = 1
        price = push!(price, init_price)
        dividend = push!(dividend, init_dividend)
        trading_volume = Vector{Any}(undef, 0)
        volatility = Vector{Any}(undef, 0)
        technical_activity = Vector{Any}(undef, 0)
        bit1 = 0
        bit2 = 0
        bit3 = 0
        bit4 = 0
        bit5 = 0
        bit6 = 0
        bit7 = 0
        bit8 = 0
        bit9 = 0
        bit10 = 0
        bit11 = 1
        bit12 = 0
    )
    # Initialization period, generate historical dividend and prices
    while state.t <= model.initialization_t
        state.dividend = dividend_process(model.d̄, model.ρ, state.dividend, model.σ_ε)
        state.price = push!(state.price, (last(state.dividend) / model.r))
        state.t += 1
    end

    # generate first state bit vector sequence
    bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10 = update_market_vector(state.price, state.dividend)

end

"""
Initialize and add agents.
"""
# function init_agents!(model)
#     base_wage = model.ω * model.τ
#     theoretical_max_output = 
#         (model.τ * (1 - model.κ)) ^ (1 - model.κ) * (model.τ * model.κ) ^ model.κ
#     for id in 1:model.numagents
#         a = Employee(
#             id = id, 
#             pos = (1,1),
#             time_cooperation = model.τ/3,
#             time_shirking = model.τ/3,
#             status = InVaNo.init_status(id, model.numagents, model.dist, model.groups)
#         )

#         a.time_individual = model.τ - a.time_cooperation - a.time_shirking
#         a.δ = InVaNo.init_delta(a.status)
#         # at t_0 both norms are equal for each agent
#         a.norm_coop = a.time_cooperation
#         a.norm_shirk = a.time_shirking
#         # at t_0 mean_coop is equal to every agent's time_cooperation
#         InVaNo.update_output!(a, model.κ, a.time_cooperation)
#         InVaNo.update_realised_output!(a, theoretical_max_output)
#         InVaNo.update_realised_output_max!(a, a.output)
#         InVaNo.update_rewards!(a, model.μ, model.λ, base_wage, a.output)

#         add_agent_single!(a, model)
#     end
#     for agent in allagents(model)
#         InVaNo.find_peers!(agent, model)
#     end
#     return model
# end

# ## Stepping

function init_agents!(model) #init_state has to come before this
    T = model.warm_up_t + model.recorded_t # Total sim time
    GA_frequency = T / model.k # num times GA is invoked across total simulation
    n = Int(GA_frequency / model.k_var) # scaling factor for consistent k range over time
    δ_dist_1 = repeat(Vector(((model.k - (model.k_var/2)) + 1) : (model.k - 1)), n)
    δ_dist_2 = repeat([model.k, model.k], n)
    δ_dist_3 = repeat(Vector((model.k + 1) : ((model.k + (model.k_var/2)) - 1)), n)
    δ_dist = vcat(δ_dist_1, δ_dist_2, δ_dist_3)
    for id in 1:model.numagents # Why are some properties included in `Trader` and others aren't, distinction?
        a = Trader(
            id = id, 
            pos = (1,1),
            time_cooperation = model.τ/3,
            time_shirking = model.τ/3,
            status = InVaNo.init_status(id, model.numagents, model.dist, model.groups)
            predictors = evolution.init_predictors(model.num_predictors)
        )
        a.relative_cash = model.init_cash
        a.predict_acc = Vector{Any}(undef, 0) # Should I change all these from `Any` to `Float` 
        a.fitness_j = Vector{Any}(undef, 0)
        a.δ = evolution.init_learning(N,δ_dist)
        a.expected_pd = evolution.update_exp!(a.predictors, state.price, state.dividend)
        # a.demand_xi = evolution.get_demand!(X...)
        # a.σ_i = Vector{Any}(undef, 0)
        
        # add lines that do initial price formation process?
        #evolution...
        
        a.time_individual = model.τ - a.time_cooperation - a.time_shirking
        a.δ = InVaNo.init_delta(a.status)
        # at t_0 both norms are equal for each agent
        a.norm_coop = a.time_cooperation
        a.norm_shirk = a.time_shirking
        # at t_0 mean_coop is equal to every agent's time_cooperation

        # Notice how this isn't attached to indv agent, just uses the indv agent... right? 
        InVaNo.update_output!(a, model.κ, a.time_cooperation)
        InVaNo.update_realised_output!(a, theoretical_max_output)
        InVaNo.update_realised_output_max!(a, a.output)
        InVaNo.update_rewards!(a, model.μ, model.λ, base_wage, a.output)

        add_agent_single!(a, model) # Where does this function come from?
    end
    for agent in allagents(model)
        InVaNo.find_peers!(agent, model)
    end
    return model
end

## Stepping

"""
Define what happens in the model.
"""
function model_step!(model)
    scheduled_agents = (model[id] for id in model.scheduler(model))

    for agent in scheduled_agents
        InVaNo.update_norm_coop!(agent, model)
        InVaNo.update_norm_shirk!(agent, model)
        InVaNo.update_phi!(agent, model.Σ)
        InVaNo.update_gamma!(agent)
        if model.μ == 1.0
            InVaNo.update_rho!(agent, model.λ)
        else
            agent.ρ = 0.0
        end
    end

    for agent in scheduled_agents
        if rand(model.rng, Bool)
            InVaNo.spend_time_shirking!(agent, model.τ, model.rng)
            residual_τ = model.τ - agent.time_shirking
            InVaNo.spend_time_cooperation!(agent, residual_τ, model.rng)
        else
            InVaNo.spend_time_cooperation!(agent, model.τ, model.rng)
            residual_τ = model.τ - agent.time_cooperation
            InVaNo.spend_time_shirking!(agent, residual_τ, model.rng)
        end
        InVaNo.spend_time_individual!(agent, model.τ)
        InVaNo.update_deviations!(agent)
    end

    OGO = (model.τ * (1 - model.κ)) ^ (1 - model.κ) * (model.τ * model.κ) ^ model.κ
    max_output = maximum(agent.output for agent in allagents(model))
    for agent in scheduled_agents
        all_other_ids = filter(x -> x != agent.id, collect(allids(model)))
        mean_coop = mean(model[id].time_cooperation for id in all_other_ids)
        InVaNo.update_output!(agent, model.κ, mean_coop)
        InVaNo.update_realised_output!(agent, OGO)
        InVaNo.update_realised_output_max!(agent, max_output)
    end

    base_wage = model.ω * model.τ
    mean_output = mean(agent.output for agent in allagents(model))
    for agent in scheduled_agents
        InVaNo.update_rewards!(agent, model.μ, model.λ, base_wage, mean_output)
    end

    InVaNo.update_gini_index!(model)

    return model
end
