#################################
#MARKOV DECISION PROCESS: SIMULATE
#https://github.com/JuliaPOMDP/POMDPToolbox.jl#simulators
#################################
# include("Juliacommand.jl");
using POMDPs
using POMDPToolbox


#### GRID SIMULATION (PROBABILISTIC): state transitions
# https://github.com/JuliaPOMDP/POMDPToolbox.jl/blob/master/src/simulators/sim.jl
function runSimulation(mdp::ProbabilisticGrid, startingState::ProbabilisticState, maxSteps::Int)
    println("\n\n")
    sim(mdp, startingState, max_steps=maxSteps) do current_State
        action = :right     # the policy
        println("SIMULATION: startingState is: $current_State....Action: $action")
        return action # code above calculates action `a` based on `s` - this is the policy
    end;
end

### GRID SIMULATION (PROBABILISTIC): total reward from initial state
function getTotalReward(mdp::ProbabilisticGrid, s::ProbabilisticState, maxSteps::Int, policy::ValueIterationPolicy)
    println("\n\n")
    recorder = HistoryRecorder(max_steps=maxSteps)
    history = simulate(recorder, mdp, policy, s)
    println("TOTAL discounted reward: $(discounted_reward(history))")
    return history
end

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


