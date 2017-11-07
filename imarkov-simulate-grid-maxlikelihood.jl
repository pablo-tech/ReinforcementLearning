#################################
#MARKOV DECISION PROCESS: SIMULATE
#https://github.com/JuliaPOMDP/POMDPToolbox.jl#simulators
#################################
# include("Juliacommand.jl");
using POMDPs
using POMDPToolbox


#### GRID SIMULATION (MAX LIKELIHOOD): state transitions

# TODO


### GRID SIMULATION (MAX LIKELIHOOD): total reward from initial state

# TODO

### HISTORY REPLAY: stepwise from previously computed history
function replayHistory(hist::MDPHistory)
    println("\n\n")
    for (s, a, sp) in eachstep(hist, "s,a,sp")
        @printf("REPLAY s: %-26s  a: %-6s  s': %-26s\n", s, a, sp)
    end
end


########
#TOOLS: https://github.com/JuliaPOMDP/POMDPs.jl#tools
########

# TODO

########
#PERFORMANCE: https://github.com/JuliaPOMDP/POMDPs.jl#performance-benchmarks
########

# TODO


