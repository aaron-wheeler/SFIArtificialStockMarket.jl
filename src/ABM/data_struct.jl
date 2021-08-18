"""
Define model parameters.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
Base.@kwdef mutable struct ModelProperties
    numagents::Int = 100
    κ::Float64 = 0.5
    ω::Float64 = 1.0
    τ::Float64 = 10.0
    scenario::String
    μ::Float64
    λ::Float64
    Σ::Float64
    env::String
    num_peers::Int = 8
    dist::NTuple{4, Float64} = (0.25, 0.25, 0.25, 0.25)
    groups::NTuple{4, Symbol} = (:C, :O, :SE, :ST)
    h::Float64 = 0.1
    gini_index::Float64 = 0.0
end

"""
Define parameter structure of scenarios.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
struct Scenario
    name::String
    μ::Float64
    λ::Float64
    Σ::Float64
end

"""
Define base structure of agents and their properties.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
Base.@kwdef mutable struct Employee <: AbstractAgent
    id::Int
    pos::Dims{2}
    time_cooperation::Float64
    time_shirking::Float64
    time_individual::Float64 = 0.0
    reward::Float64 = 0.0
    deviation_norm_shirk::Float64 = 0.0
    deviation_norm_coop::Float64 = 0.0
    γ::Float64 = 0.0
    δ::Float64 = 0.0
    ϕ::Float64 = 0.0
    ρ::Float64 = 0.0
    status::Symbol
    norm_coop::Float64 = 0.0
    norm_shirk::Float64 = 0.0
    peers::Vector{Int} = []
    output::Float64 = 0.0
    realised_output::Float64 = 0.0
    realised_output_max::Float64 = 0.0
end
