#################################
#SOLUTION FOR MARKOV DECISION PROBLEM: GRID (PROBABILISTIC)
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
#EXPLICIT PROBABILISTIC
########

### VALUE ITERATION: DISCRETE SOLVER
# DiscreteValueIteration (JuliaPOMDP/DiscreteValueIteration.jl)
# initializes the Solver type
function getUtilitySolver(mdp::ProbabilisticGrid)
    # max_iterations: maximum number of iterations value iteration runs for (default is 100)
    # belres: the value of Bellman residual used in the solver (defualt is 1e-3)
    solver = ValueIterationSolver(max_iterations=100, belres=1e-3)
    return solver
end


### POLICY SOLVER: value iteration discrete
#POLICY SOLVER: DiscreteValueIteration (JuliaPOMDP/DiscreteValueIteration.jl)
# initializes the policy type
function getPolicySolver(mdp::ProbabilisticGrid)
    return ValueIterationPolicy(mdp)
end


### POLICY SOLUTION: DiscreteValueIteration (JuliaPOMDP/DiscreteValueIteration.jl)
# if verbose=false, the text output will be supressed (false by default)
## instructions:
# 1) get the policy
# 2) s = create_state(mdp) # this can be any valid state
# 3) a = action(polciy, s) # returns the optimal action for state s
## example:
# s = ProbabilisticState(9,2)
# a = action(planner, s)
#
# runs value iterations
function getPolicySolution(mdp::ProbabilisticGrid)
    solver = getUtilitySolver(mdp)
    policy = getPolicySolver(mdp)
    planner = solve(solver, mdp, policy, verbose=true);
    return policy
end

### MONTECARLO TREE SEARCH (MCTSSolver)
# ERROR: LoadError: HttpParser not properly installed. Please run
# Pkg.build("HttpParser")
#
# # initialize the solver with hyper parameters
# # n_iterations: the number of iterations that each search runs for
# # depth: the depth of the tree (how far away from the current state the algorithm explores)
# # exploration constant: this is how much weight to put into exploratory actions.
# # A good rule of thumb is to set the exploration constant to what you expect the upper bound on your average expected reward to be.
# solver = MCTSSolver(n_iterations=1000,
#                     depth=20,
#                     exploration_constant=10.0,
#                     enable_tree_vis=true)
#
# # initialize the planner by calling the `solve` function. For online solvers, the
# planner = solve(solver, mdp)


### PLANNER: TREE (D3Tree)
# Pkg.build("HttpParser")
# ERROR: LoadError: HttpParser not properly installed. Please run
#
# using D3Trees
#
# # first, run the planner on the state
# s = state_hist(hist)[1]
# a = action(planner, s)
#
# # show the tree (click the node to expand)
# D3Tree(planner, s)




