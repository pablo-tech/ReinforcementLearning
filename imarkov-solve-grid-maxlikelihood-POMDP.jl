#################################
#SOLUTION FOR MARKOV DECISION PROBLEM: GRID (MAX LIKELIHOOD)
#IMPORTANT: experimental use of POMDP.jl
#################################
# SOLVERS: for MDP
# https://github.com/JuliaPOMDP/POMDPs.jl#mdp-solvers
# http://juliapomdp.github.io/POMDPs.jl/stable/
# MODELS:
# https://github.com/JuliaPOMDP/POMDPModels.jl Grid World, Tiger, Crying Baby, Random, Mountain Car, Inverted Pendulum, T-Maze
using POMDPs
using DiscreteValueIteration


##################################MDP SOLVERS: https://github.com/JuliaPOMDP/POMDPs.jl#mdp-solvers
#Discrete value iteration https://github.com/JuliaPOMDP/DiscreteValueIteration.jl
#Montecarlo Tree Search https://github.com/JuliaPOMDP/MCTS.jl
#################################


########
#GENERATIVE MAX LIKELIHOOD
########

### VALUE ITERATION: DISCRETE SOLVER
# DiscreteValueIteration (JuliaPOMDP/DiscreteValueIteration.jl)
# initializes the Solver type
function getUtilitySolver(mdp::MaxlikelihoodGrid)
    # max_iterations: maximum number of iterations value iteration runs for (default is 100)
    # belres: the value of Bellman residual used in the solver (defualt is 1e-3)
    solver = ValueIterationSolver(max_iterations=100, belres=1e-3)
    return solver
end


### POLICY SOLVER: value iteration discrete
#POLICY SOLVER: DiscreteValueIteration (JuliaPOMDP/DiscreteValueIteration.jl)
# initializes the policy type
function getPolicySolver(mdp::MaxlikelihoodGrid)
    return ValueIterationPolicy(mdp)
end


### POLICY SOLUTION: DiscreteValueIteration (JuliaPOMDP/DiscreteValueIteration.jl)
# if verbose=false, the text output will be supressed (false by default)
## instructions:
# 1) get the policy
# 2) s = create_state(mdp) # this can be any valid state
# 3) a = action(polciy, s) # returns the optimal action for state s
## example:
# s = MaxlikelihoodState(9,2)
# a = action(planner, s)
#
# runs value iterations
function getPolicySolution(mdp::MaxlikelihoodGrid)
    solver = getUtilitySolver(mdp)
    policy = getPolicySolver(mdp)
    planner = solve(solver, mdp, policy, verbose=true);
    return policy
end

### MONTECARLO TREE SEARCH (MCTSSolver)

# TODO


### PLANNER: TREE (D3Tree)

# TODO




