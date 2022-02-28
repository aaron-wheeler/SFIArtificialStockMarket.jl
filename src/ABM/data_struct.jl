"""
Define model parameters.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
Base.@kwdef mutable struct ModelProperties
    num_agents::Int = 25
    N::Int = 25
    λ::Float64 = 0.5
    num_predictors::Int = 100
    t::Int = 1
    # bit1::Int = 0
    # bit2::Int = 0
    # bit3::Int = 0
    # bit4::Int = 0
    # bit5::Int = 0
    # bit6::Int = 0
    # bit7::Int = 0
    # bit8::Int = 0
    # bit9::Int = 0
    # bit10::Int = 0
    # bit11::Int = 1
    # bit12::Int = 0
    state_vector::Vector{Int} = []
    price::Vector{Float64} = []
    dividend::Vector{Float64} = []
    trading_volume::Vector{Int} = []
    volatility::Vector{Float64} = []
    technical_activity::Vector{Int} = []
    initialization_t::Int = 499
    generalization_t::Int = 4000
    warm_up_t::Int = 250000
    recorded_t::Int = 10000
    k::Int # = 250 for complex regime
    pGAcrossover::Float64 # = 0.1 for complex regime
    τ::Int # = 75 for complex regime
    num_shares::Int = 25
    r::Float64 = 0.1
    ρ::Float64 = 0.95
    d̄::Float64 = 10.0
    # ε::Float64
    σ_ε::Float64 = 0.0743
    σ_pd::Float64 = 4.0
    a_min::Float64 = 0.7
    a_max::Float64 = 1.2
    b_min::Float64 = -10.0
    b_max::Float64 = 19.002
    # δ_dist::Vector{Int} = [] # **TODO: Remove this?**
    k_var::Int = 40
    C::Float64 = 0.005
    price_min::Float64 = 0.01 # from 2008 textbook
    price_max::Float64 = 200.0 # from 2008 textbook
    # price_min::Float64 = 0.001 # from original source code
    # price_max::Float64 = 99999.0 # from original source code
    init_cash::Float64 = 20000.0
    trade_restriction::Float64 = 10.0
    short_restriction::Float64 = -5.0
    cash_restriction::Float64 = -2000.0
    itermax::Int = 500
    num_elimination::Int = 20
    pcond_mut::Float64 = 0.03
    pparam_mut_long::Float64 = 0.2
    pparam_mut_short::Float64 = 0.2
    percent_mut_short::Float64 = 0.05
    mdf_price::Float64 = 0.0 # for data collection and plotting
    mdf_dividend::Float64 = 0.0 # for data collection and plotting
    mdf_trading_volume::Int = 0 # for data collection and plotting
    mdf_volatility::Float64 = 0.0 # for data collection and plotting
    frac_bits_set::Float64 = 0.0 # for data collection and plotting
    frac_bits_fund::Float64 = 0.0 # for data collection and plotting
    frac_bits_tech::Float64 = 0.0 # for data collection and plotting
end

# """
# Define parameter structure of dynamic market state.

# Please use `ABM/README.md` as a reference for what each field of this struct does.
# """
# Base.@kwdef mutable struct State  
#     t::Int # Make this exclusive to ModelProperties?
#     bit1::Int
#     bit2::Int
#     bit3::Int
#     bit4::Int
#     bit5::Int
#     bit6::Int
#     bit7::Int
#     bit8::Int
#     bit9::Int
#     bit10::Int
#     bit11::Int
#     bit12::Int
#     price::Vector{Float64} = []
#     dividend::Vector{Float64} = []
#     trading_volume::Vector{Int} = []
#     volatility::Vector{Float64} = []
#     technical_activity::Vector{Int} = []
# end

"""
Define base structure of agents and their properties.

Please use `ABM/README.md` as a reference for what each field of this struct does.
"""
Base.@kwdef mutable struct Trader <: AbstractAgent # Investigate what this line means 
    id::Int
    pos::Dims{2}
    relative_cash::Float64
    relative_holdings::Int = 1
    relative_wealth::Float64
    predictors::Vector{Any} = []
    predict_acc::Vector{Float64} = []
    fitness_j::Vector{Float64} = []
    expected_pd::Vector{Float64} = [] # Remove this? Is there one of these for each predictor? Why a vector?
    demand_xi::Int = 0 # Remove this?
    δ::Vector{Int} = []
    active_predictors::Vector{Int} = []
    forecast::Vector{Any} = []
    active_j_records::Matrix{Int} = zeros(Int, 0, 0)

    # specific to predictor, remove?
    σ_i::Float64
end
