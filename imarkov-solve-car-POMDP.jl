#################################
#SOLUTION FOR MARKOV DECISION PROBLEM: SMALL
#IMPORTANT: experimental use of POMDP.jl
#################################
# SOLVERS: for MDP
# https://github.com/JuliaPOMDP/POMDPs.jl#mdp-solvers
# http://juliapomdp.github.io/POMDPs.jl/stable/
# MODELS:
# https://github.com/JuliaPOMDP/POMDPModels.jl Grid World, Tiger, Crying Baby, Random, Mountain Car, Inverted Pendulum, T-Maze
using POMDPs
using DiscreteValueIteration


#################################
#REINFORCEMENT SOLVERS: https://github.com/JuliaPOMDP/POMDPs.jl#reinforcement-learning
#Tabular Learning: Q-Learning, SARSA, SARSA lambda https://github.com/JuliaPOMDP/TabularTDLearning.jl
#################################


########
#UNCERTAIN WORLD: T() and R() not given, explore in data
########
#Pkg.add("POMDPs")
#import POMDPs
#POMDPs.add("TabularTDLearning")
using TabularTDLearning
using POMDPModels

# https://github.com/JuliaPOMDP/TabularTDLearning.jl
#mdp = GridWorld()

### SOLVER: use Q-Learning
function getQlearningSolver(mdp::MountainCar)
    solver = QLearningSolver(mdp, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
    return solver
end

#policy = solve(solver, mdp)
# Use SARSA
#solver = SARSASolver(mdp, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
#policy = solve(solver, mdp)
# Use SARSA lambda
#solver = SARSALambdaSolver(mdp, learning_rate=0.1, lambda=0.9, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
#policy = solve(solver, mdp)




