#################################
#MARKOV DECISION PROCESS: GRID
#NOTE: EXPERIMENTAL use of POMPDs.jl to better understand this problem/solution
#################################
# APPROACH: define the world by extending MDP{}, and solve by implementing POMDPs.* methods
include("imarkov-grid-maxlikelihood.jl");


########
# WORLD TUPLE: (STATE, ACTION, TRANSITION, REWARD)
# NOTE: can be run in two modes
#   a) Probabilistic mode: given Transition and Rewards probability functions.  See Decisions Under Uncertainty 4.2.5
#   b) MAX LIKELYHOOD modes: calculated Transition and Reward probability functions.  See Decisions Under Uncertainty 5.2
########
using POMDPs
importall POMDPs


########
# GRID: MAX LIKELIHOOD
########

### STATE: DONE
# implement POMDPs.isterminal
function POMDPs.isterminal(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState)
    return isMaxlikelihoodTerminal(mdp, state)
end

### STATE: define
# implement POMDPs.states
function POMDPs.states(mdp::MaxlikelihoodGrid)
    return getMaxlikelihoodStateSpace(mdp)
end

### STATE: index
# implement POMDPs.state_index
function POMDPs.state_index(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState)
    return getMaxlikelihoodStateIndex(mdp,state)
end

### STATE: count
# implement POMDPs.n_states
POMDPs.n_states(mdp::MaxlikelihoodGrid) = getMaxlikelihoodNumberOfStates(mdp)

### DISCOUNT (Maxlikelihood):
# implement POMDPs.discount
POMDPs.discount(mdp::MaxlikelihoodGrid) = getMaxlikelihoodDiscountFactor(mdp);

### ACTION: index
# POMDPs.action_index
function POMDPs.action_index(mdp::MaxlikelihoodGrid, act::Symbol)
    actionIndex = getMaxlikelihoodActionsIndexDict()
    return actionIndex[act]
end

### ACTIONS: itemize
# implement POMDPs.actions
POMDPs.actions(mdp::MaxlikelihoodGrid) = gridMaxlikelihoodActions;


### total actions
# implement POMDPs.n_actions
POMDPs.n_actions(mdp::MaxlikelihoodGrid) = length(gridMaxlikelihoodActions())

### TRANSITION: configure the transition model
# implement POMDPs.transition
function POMDPs.transition(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState, action::Symbol)
    return MaxlikelihoodTransitionModel(mdp, state, action)
end

### REWARD: utility of a (state,action,statePrime) datapoint in a world
# implement POMDPs.reward
function POMDPs.reward(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState, action::Symbol, statePrime::MaxlikelihoodState)
    return getMaxlikelihoodReward(mdp.reward_states, mdp.reward_values, state)
end



