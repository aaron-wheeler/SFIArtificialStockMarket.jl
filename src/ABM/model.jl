include("structs.jl") 
include("InVaNo.jl")

## Initialisation

"""
    `init_model(; seed, env, properties...) → ABM`

Create ABM model with given `seed`, `env`, and other `properties`.
"""
function init_model(; seed::UInt32, env, properties...)
    if env in ("Global", "Neighbours", "Random")
        space = GridSpace((10,10), periodic = true, metric = :euclidean )
        model = ABM(
            Employee, 
            space; 
            properties = ModelProperties(; env, properties...), 
            scheduler = Schedulers.randomly,
            rng = MersenneTwister(seed)
        )
        model.dist = cumsum(model.dist)
        init_agents!(model)
        return model
    else
        error("Given env '$(env)' is not implemented in the model.
            Please verify the integrity of the provided `env` value.")
    end
end

"""
Initialize and add agents.
"""
function init_agents!(model)
    base_wage = model.ω * model.τ
    theoretical_max_output = 
        (model.τ * (1 - model.κ)) ^ (1 - model.κ) * (model.τ * model.κ) ^ model.κ
    for id in 1:model.numagents
        a = Employee(
            id = id, 
            pos = (1,1),
            time_cooperation = model.τ/3,
            time_shirking = model.τ/3,
            status = InVaNo.init_status(id, model.numagents, model.dist, model.groups)
        )

        a.time_individual = model.τ - a.time_cooperation - a.time_shirking
        a.δ = InVaNo.init_delta(a.status)
        # at t_0 both norms are equal for each agent
        a.norm_coop = a.time_cooperation
        a.norm_shirk = a.time_shirking
        # at t_0 mean_coop is equal to every agent's time_cooperation
        InVaNo.update_output!(a, model.κ, a.time_cooperation)
        InVaNo.update_realised_output!(a, theoretical_max_output)
        InVaNo.update_realised_output_max!(a, a.output)
        InVaNo.update_rewards!(a, model.μ, model.λ, base_wage, a.output)

        add_agent_single!(a, model)
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
