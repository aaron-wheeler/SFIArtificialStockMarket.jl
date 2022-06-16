using Documenter
using SFIArtificialStockMarket

makedocs(
    sitename = "SFIArtificialStockMarket",
    format = Documenter.HTML(),
    modules = [SFIArtificialStockMarket]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/aaron-wheeler/SFIArtificialStockMarket.jl.git"
)
