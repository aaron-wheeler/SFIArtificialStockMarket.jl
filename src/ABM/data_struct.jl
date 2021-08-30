"""
Define model parameters.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
Base.@kwdef mutable struct ModelProperties
    num_agents::Int = 25
    λ::Float64 = 0.5
    predictors::Int = 100
    bit1::Int
    bit2::Int
    bit3::Int
    bit4::Int
    bit5::Int
    bit6::Int
    bit7::Int
    bit8::Int
    bit9::Int
    bit10::Int
    bit11::Int
    bit12::Int
    warm_up_t::Int
    κ::Int
    regime::String
    num_shares::Int = 25
    r::Float64 = 0.1
    ρ::Float64 = 0.95
    d̄::Float64 = 10.0
    ε::Float64
    σ_ε::Float64 = 0.0743
    σ_pd::Float64 = 4.0
    M::Float64 = 0.0
    C::Float64 = 0.005
    init_cash::Float64 = 20000.0
end

"""
Define parameter structure of states.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
struct State # This must be a mutable struct filled with vectors like one below... 
    bit1::Int
    bit2::Int
    bit3::Int
    bit4::Int
    bit5::Int
    bit6::Int
    bit7::Int
    bit8::Int
    bit9::Int
    bit10::Int
    bit11::Int
    bit12::Int
    price::Float64 
    dividend::Float64 
    volume::Float64 
    volatility::Float64 
    technical_activity::Float64 
end

"""
Define base structure of agents and their properties.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
Base.@kwdef mutable struct Trader <: AbstractAgent # Investigate what this line means 
    id::Int
    relative_cash::Float64 = init_cash #(**To do: Investigate this, set equal to zero?**)
    pos::Dims{2}
    predict_acc::Vector{Float64} = []
    fitness_j::Vector{Float64} = []
    expected_pd::Vector{Float64} = []
    demand_xi::Int = 0 #(**To do: Investigate this, should this be a float?**)
    σ_i::Float64 = σ_pd #(**To do: Investigate this, set equal to zero? Should this be vector?**)
    δ::Float64 = 0.0
    a::Vector{Float64} = []
    b::Vector{Float64} = []
    JX::Float64 = 0.0 #(**To do: Investigate this)
    τ::Int = 0
    s::Vector{Int} = []
end
