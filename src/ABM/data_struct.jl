"""
Define model parameters.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
Base.@kwdef mutable struct ModelProperties
    num_agents::Int
    λ::Float64 = 0.5
    num_predictors::Int = 100
    t::Int = 1
    state_vector::Vector{Int} = Vector{Int}(undef, 12)
    price::Vector{Float64} = []
    dividend::Vector{Float64} = Vector{Float64}(undef, 2)
    trading_volume::Float64 = 0.0
    volatility::Float64 = 0.0
    initialization_t::Int = 499
    generalization_t::Int = 4000
    k::Int
    pGAcrossover::Float64
    τ::Int
    r::Float64 = 0.1
    ρ::Float64 = 0.95
    d̄::Float64 = 10.0
    σ_ε::Float64 = 0.0743
    σ_pd::Float64 = 4.0
    a_min::Float64 = 0.7
    a_max::Float64 = 1.2
    b_min::Float64 = -10.0
    b_max::Float64 = 19.002
    C::Float64 = 0.005
    price_min::Float64 = 0.01
    price_max::Float64 = 200.0
    init_cash::Float64 = 20000.0
    eta::Float64 = 0.0005
    min_excess::Float64 = 0.01
    max_rounds::Int = 20
    trade_restriction::Float64 = 10.0
    short_restriction::Float64 = -5.0
    cash_restriction::Float64 = -2000.0
    num_elimination::Int = 20
    pcond_mut::Float64 = 0.03
    pparam_mut_long::Float64 = 0.2
    pparam_mut_short::Float64 = 0.2
    percent_mut_short::Float64 = 0.05
    mdf_price::Float64 = 0.0
    mdf_dividend::Float64 = 0.0
    track_bits::Bool
    frac_bits_set::Float64 = 0.0
    frac_bits_fund::Float64 = 0.0
    frac_bits_tech::Float64 = 0.0
end

"""
Define base structure of agents and their properties.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
Base.@kwdef mutable struct Trader <: AbstractAgent
    id::Int
    pos::Dims{2}
    relative_cash::Float64
    relative_holdings::Float64 = 1.0
    relative_wealth::Float64
    predictors::Vector{Any} = []
    predict_acc::Vector{Float64} = []
    fitness_j::Vector{Float64} = []
    chosen_j::Int = 100 
    demand_xi::Float64 = 0.0
    active_predictors::Vector{Int} = []
    forecast::Vector{Any} = []
    active_j_records::Matrix{Int} = zeros(Int, 0, 0)
end
