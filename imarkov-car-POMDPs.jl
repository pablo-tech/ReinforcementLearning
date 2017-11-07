#################################
#MARKOV DECISION PROCESS: CAR
#NOTE: EXPERIMENTAL use of POMPDs.jl for this problem
#################################
# APPROACH: define the world by extending MDP{}, and solve by implementing POMDPs.* methods
include("imarkov-car.jl");

########
# WORLD TUPLE: (STATE, ACTION, TRANSITION, REWARD)
########
using POMDPs
importall POMDPs


### STATE: count
# implement POMDPs.n_states
POMDPs.n_states(mdp::MountainCar) = 0 # TODO getMaxlikelihoodNumberOfStates(mdp)

### ACTIONS: itemize
# implement POMDPs.actions
POMDPs.actions(mdp::MountainCar) = getCarActions;

### ACTIONS: total
# implement POMDPs.n_actions
POMDPs.n_actions(mdp::MountainCar) = 0 # TODO length(gridMaxlikelihoodActions())



