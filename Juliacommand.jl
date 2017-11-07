### JULIA Initialization: run from Dockerfile

# from 'which jupyter' in command line
ENV["JUPYTER"] = "/onenow/conda/bin//jupyter"

# from 'which python' in command line
ENV["PYTHON"] = "/onenow/conda/envs/junogym/bin/python"

# add julia to Jupyter
Pkg.add("IJulia")

### General Julia packages
# Call python from julia
Pkg.add("PyCall")
# plot
Pkg.add("PyPlot")
# data
Pkg.add("CSV")
Pkg.add("RDatasets")
Pkg.add("DataFrames")


#### Statistics Julia packages
Pkg.add("Distributions")
Pkg.add("BayesNets")
Pkg.add("PGFPlots")
Pkg.add("Interact")

### Markov Decision Process Julia packages
# install the POMDPs.jl interface
## Clone
# POMPDs
Pkg.add("POMDPs")
using POMDPs # POMDPs.jl
POMDP.add("MCTS")
POMDPs.add_all()
# UPDATE all
Pkg.update()