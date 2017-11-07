#################################
#MARKOV DECISION PROCESS: GRID
#NOTE: EXPERIMENTAL use of POMPDs.jl to better understand this problem/solution
#################################
# APPROACH: define the world by extending MDP{}, and solve by implementing POMDPs.* methods
include("imarkov-grid-probabilistic.jl");


########
# WORLD TUPLE: (STATE, ACTION, TRANSITION, REWARD)
# NOTE: can be run in two modes
#   a) Probabilistic mode: given Transition and Rewards probability functions.  See Decisions Under Uncertainty 4.2.5
#   b) MAX LIKELYHOOD modes: calculated Transition and Reward probability functions.  See Decisions Under Uncertainty 5.2
########
using POMDPs
importall POMDPs


########
# GRID: PROBABILISTIC EXPLICIT
########

### STATE: DONE
# implement POMDPs.isterminal
function POMDPs.isterminal(mdp::ProbabilisticGrid, state::ProbabilisticState)
    return isProbabilisticTerminal(mdp, state)
end

### STATE: define
# implement POMDPs.states
function POMDPs.states(mdp::ProbabilisticGrid)
    return getProbabilisticStateSpace(mdp)
end

### STATE: index
# implement POMDPs.state_index
function POMDPs.state_index(mdp::ProbabilisticGrid, state::ProbabilisticState)
    return getProbabilisticStateIndex(mdp,state)
end

### STATE: count
# implement POMDPs.n_states
POMDPs.n_states(mdp::ProbabilisticGrid) = getProbabilisticNumberOfStates(mdp)

### DISCOUNT (Probabilistic):
# implement POMDPs.discount
POMDPs.discount(mdp::ProbabilisticGrid) = getProbabilisticDiscountFactor(mdp);

### ACTION: index
# POMDPs.action_index
function POMDPs.action_index(mdp::ProbabilisticGrid, act::Symbol)
    actionIndex = getProbabilisticActionsIndexDict()
    return actionIndex[act]
end

### ACTIONS: itemize
# implement POMDPs.actions
POMDPs.actions(mdp::ProbabilisticGrid) = gridProbabilisticActions;

### total actions
# implement POMDPs.n_actions
POMDPs.n_actions(mdp::ProbabilisticGrid) = length(gridProbabilisticActions)

### TRANSITION: configure the transition model
# implement POMDPs.transition
function POMDPs.transition(mdp::ProbabilisticGrid, state::ProbabilisticState, action::Symbol)
    return ProbabilisticTransitionModel(mdp, state, action)
end

### REWARD: utility of a (state,action,statePrime) datapoint in a world
# implement POMDPs.reward
function POMDPs.reward(mdp::ProbabilisticGrid, state::ProbabilisticState, action::Symbol, statePrime::ProbabilisticState)
    return getProbabilisticReward(mdp.reward_states, mdp.reward_values, state)
end




