# Install and precompile packages
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# # Start workers (for parallel computing)
# using Distributed
# addprocs(4)

# # Set up package environment on workers (for parallel computing)
# @everywhere begin
#     using Pkg
#     Pkg.activate(".")
# end

# Load packages on master process
using DataFrames
using CSV

# # Load packages on workers (for parallel computing)
# @everywhere begin
#     using Agents
#     using Statistics: mean
#     using Random
# end

# Load packages (for serial computing)
using Agents
using Statistics
using Random
using SFIArtificialStockMarket

# # Load model libraries on workers (for parallel computing)
# @everywhere cd("src/ABM")
# @everywhere include("data_struct.jl")
# @everywhere include("model.jl")


# Load model libraries (for serial computing)
include("data_struct.jl") 
include("model.jl")

## Define and run model
"""
Create model, let it run, wrangle data
"""
function let_it_run()
    # Number of model ensembles to run and their random seeds
    num_model_ensembles = 1
    seeds = rand(UInt32, num_model_ensembles)

    # Setup parameters
    properties = (
        num_agents = 25,
        k = 250,
        pGAcrossover = 0.1,
        τ = 75,
        track_bits = false
    )

    # Agent data to collect
    adata = [:relative_cash, :relative_holdings, :relative_wealth, :chosen_j]

    # Model data to collect
    if properties[:track_bits] == false
        mdata = [:t, :mdf_price, :mdf_dividend, :trading_volume, :volatility]
    else
        mdata = [:t, :mdf_price, :mdf_dividend, :trading_volume, :volatility, :frac_bits_set, :frac_bits_fund, :frac_bits_tech]
    end

    models = [init_model(; seed, properties...) for seed in seeds] 

    # Collect data (ensemble simulation for multiple random seeded models)
    pre_SS_t = 250000 # number of time steps to warm up and reach steady state 
    recorded_t = 10000 # time steps recorded once steady state is reached

    model_runs = pre_SS_t + recorded_t # total numder of time steps in model
    steady_state = collect(pre_SS_t:model_runs) # time steps where data is collected and stored locally

    adf, mdf = ensemblerun!(models, dummystep, model_step!, model_runs;
        adata = adata, mdata = mdata, when = steady_state, when_model = steady_state, parallel = false)

    # Create save path
    savepath = mkpath("../../Data/ABMs/SFI/test")

    # Save agent data
    CSV.write("$(savepath)/adf_test.csv", adf)

    # Save model data
    CSV.write("$(savepath)/mdf_test.csv", mdf)
end

Random.seed!(44801)
@time let_it_run()
println("Simulation Complete")
