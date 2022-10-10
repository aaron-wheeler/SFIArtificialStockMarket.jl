using SFIArtificialStockMarket, Random

Random.seed!(44801)
N_agents = 25
warm_up_t = 250000 
SS_t = 10000

@time adf, mdf = SFI_run(N_agents, warm_up_t, SS_t)
# @time adf, mdf = SFI_run(N_agents, warm_up_t, SS_t; save_to_disk=true)

# include("test/example_run.jl")