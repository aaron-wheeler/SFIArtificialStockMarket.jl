include("data_struct.jl") 
include("evolution.jl")

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

function init_model(; seed::UInt32, properties...)
    space = GridSpace((10,10), periodic = true, metric = :euclidean )  # TODO: Investigate this
    model = ABM(
        Trader, 
        space; 
        properties = ModelProperties(; properties...), 
        scheduler = Schedulers.randomly, # TODO: Investigate this
        rng = MersenneTwister(seed) # Is this used anywhere in simulation? dividend_process?
    )
    init_state!(model)
    init_agents!(model)
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
    model.trading_volume = Vector{Any}(undef, 0)
    model.volatility = Vector{Any}(undef, 0)
    model.technical_activity = Vector{Any}(undef, 0)
    
    # Initialization period, generate historical dividend and prices
    while model.t <= model.initialization_t
        model.dividend = dividend_process(model.d̄, model.ρ, model.dividend, model.σ_ε)
        model.price = push!(model.price, (last(model.dividend) / model.r))
        model.t += 1
    end

    # generate first state bit vector sequence
    bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10 = update_market_vector(model.price, model.dividend) # Have to append model.X to each bit?
    model.state_vector = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9, bit10, bit11, bit12]

    return model
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


function init_agents!(model)
    T = model.warm_up_t + model.recorded_t # Total sim time
    GA_frequency = Int(T / model.k) # num times GA is invoked across total simulation
    # MAY NEED ERROR MESSAGE TO SPECIFY THAT T/K MUST BE AN INTEGER**
    n = Int(GA_frequency / model.k_var) # scaling factor for consistent k range over time
    δ_dist_1 = repeat(Vector(((model.k - (model.k_var/2)) + 1) : (model.k - 1)), n)
    δ_dist_2 = repeat([model.k, model.k], n)
    δ_dist_3 = repeat(Vector((model.k + 1) : ((model.k + (model.k_var/2)) - 1)), n)
    δ_dist = vcat(δ_dist_1, δ_dist_2, δ_dist_3)
    for id in 1:model.numagents # Properties included in `Trader` here are ones that don't have default value in data_struct.jl or may be user changed later
        a = Trader(
            id = id, 
            pos = (1,1),
            relative_cash = model.init_cash,
            σ_i = model.σ_pd
        )
        a.predictors = evolution.init_predictors(model.num_predictors, model.σ_pd)
        a.δ, a.predict_acc, a.fitness_j = evolution.init_learning(GA_frequency, δ_dist, model.σ_pd, model.C, model.num_predictors, a.predictors)
        a.active_predictors, a.forecast = evolution.match_predictors(a.id, model.num_predictors, a.predictors, model.state_vector, a.predict_acc, a.fitness_j)

        # # Lines from prev ABM....?        
        # a.time_individual = model.τ - a.time_cooperation - a.time_shirking
        # a.δ = InVaNo.init_delta(a.status)
        # # at t_0 both norms are equal for each agent
        # a.norm_coop = a.time_cooperation
        # a.norm_shirk = a.time_shirking
        # # at t_0 mean_coop is equal to every agent's time_cooperation

        # # Notice how this isn't attached to indv agent, just uses the indv agent... 
        # InVaNo.update_output!(a, model.κ, a.time_cooperation)
        # InVaNo.update_realised_output!(a, theoretical_max_output)
        # InVaNo.update_realised_output_max!(a, a.output)
        # InVaNo.update_rewards!(a, model.μ, model.λ, base_wage, a.output)

        add_agent_single!(a, model) # Where does this function come from? Agents.jl?
    end

    # # Notice how this function works agentwise for all agents in sim 
    # for agent in allagents(model)
    #     InVaNo.find_peers!(agent, model)
    # end
    return model
end

## Stepping

# """
# Define what happens in the model.
# """
# function model_step!(model)
#     scheduled_agents = (model[id] for id in model.scheduler(model))

#     for agent in scheduled_agents
#         InVaNo.update_norm_coop!(agent, model)
#         InVaNo.update_norm_shirk!(agent, model)
#         InVaNo.update_phi!(agent, model.Σ)
#         InVaNo.update_gamma!(agent)
#         if model.μ == 1.0
#             InVaNo.update_rho!(agent, model.λ)
#         else
#             agent.ρ = 0.0
#         end
#     end

#     for agent in scheduled_agents
#         if rand(model.rng, Bool)
#             InVaNo.spend_time_shirking!(agent, model.τ, model.rng)
#             residual_τ = model.τ - agent.time_shirking
#             InVaNo.spend_time_cooperation!(agent, residual_τ, model.rng)
#         else
#             InVaNo.spend_time_cooperation!(agent, model.τ, model.rng)
#             residual_τ = model.τ - agent.time_cooperation
#             InVaNo.spend_time_shirking!(agent, residual_τ, model.rng)
#         end
#         InVaNo.spend_time_individual!(agent, model.τ)
#         InVaNo.update_deviations!(agent)
#     end

#     OGO = (model.τ * (1 - model.κ)) ^ (1 - model.κ) * (model.τ * model.κ) ^ model.κ
#     max_output = maximum(agent.output for agent in allagents(model))
#     for agent in scheduled_agents
#         all_other_ids = filter(x -> x != agent.id, collect(allids(model)))
#         mean_coop = mean(model[id].time_cooperation for id in all_other_ids)
#         InVaNo.update_output!(agent, model.κ, mean_coop)
#         InVaNo.update_realised_output!(agent, OGO)
#         InVaNo.update_realised_output_max!(agent, max_output)
#     end

#     base_wage = model.ω * model.τ
#     mean_output = mean(agent.output for agent in allagents(model))
#     for agent in scheduled_agents
#         InVaNo.update_rewards!(agent, model.μ, model.λ, base_wage, mean_output)
#     end

#     InVaNo.update_gini_index!(model)

#     return model
# end


# function model_step!(model)
#     scheduled_agents = (model[id] for id in model.scheduler(model))

#     # Collect demands of all individual agents and return aggregate forecast matrix `expected_xi`
#     expected_xi = zeros(Float64, 4, 0)
    
#     for agent in scheduled_agents
#         expected_xi = hcat(expected_xi, agent.forecast) # Have to add `forecast` to Agent struct? Investigate This.

#         # InVaNo.update_norm_coop!(agent, model)
#         # InVaNo.update_norm_shirk!(agent, model)
#         # InVaNo.update_phi!(agent, model.Σ)
#         # InVaNo.update_gamma!(agent)
#         # if model.μ == 1.0
#         #     InVaNo.update_rho!(agent, model.λ)
#         # else
#         #     agent.ρ = 0.0
#         # end
#     end

#     # Price formation mechanism here, get_demand()
#     # Order execution mechanism here, get_trades()

#     OGO = (model.τ * (1 - model.κ)) ^ (1 - model.κ) * (model.τ * model.κ) ^ model.κ
#     max_output = maximum(agent.output for agent in allagents(model))

#     # Calculate and update individual agent financial rewards
#     base_wage = model.ω * model.τ
#     mean_output = mean(agent.output for agent in allagents(model))
#     for agent in scheduled_agents
#         InVaNo.update_rewards!(agent, model.μ, model.λ, base_wage, mean_output)
#     end

#     # Update agent forecasting metrics 
#     for agent in scheduled_agents

#         # all_other_ids = filter(x -> x != agent.id, collect(allids(model)))
#         # mean_coop = mean(model[id].time_cooperation for id in all_other_ids)
#         # InVaNo.update_output!(agent, model.κ, mean_coop)
#         # InVaNo.update_realised_output!(agent, OGO)
#         # InVaNo.update_realised_output_max!(agent, max_output)
#     end


#     # **SAVING GA STUFF FOR AFTER INTEGRATION TESTING
#     # Check recombination status for individual agent, and if true, then undergo GA 
#     # for agent in scheduled_agents

#     #     # if rand(model.rng, Bool)
#     #     #     InVaNo.spend_time_shirking!(agent, model.τ, model.rng)
#     #     #     residual_τ = model.τ - agent.time_shirking
#     #     #     InVaNo.spend_time_cooperation!(agent, residual_τ, model.rng)
#     #     # else
#     #     #     InVaNo.spend_time_cooperation!(agent, model.τ, model.rng)
#     #     #     residual_τ = model.τ - agent.time_cooperation
#     #     #     InVaNo.spend_time_shirking!(agent, residual_τ, model.rng)
#     #     # end
#     #     # InVaNo.spend_time_individual!(agent, model.τ)
#     #     # InVaNo.update_deviations!(agent)
#     # end

#     return model
# end
