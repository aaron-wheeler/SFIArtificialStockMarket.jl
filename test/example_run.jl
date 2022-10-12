using SFIArtificialStockMarket, Random

Random.seed!(44801)
N_agents = 25
risk_tol = 0.5
warm_up_t = 250000 
SS_t = 10000
k = 250
horizon = 75

@time adf, mdf = SFI_run(N_agents, risk_tol, warm_up_t, SS_t, k, horizon; print_msg=true);
# @time adf, mdf = SFI_run(N_agents, risk_tol, warm_up_t, SS_t, k, horizon; save_to_disk=true, print_msg=true);

# include("test/example_run.jl")