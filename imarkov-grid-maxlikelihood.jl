########
#WORLD (GENERATIVE: MAX LIKELYHOOOD): as a Markov Decision Process
# NOTE: this world is completely observable. Transition() and Reward() functions are determined from the dataset
########
using POMDPs
# importall POMDPs
using RDatasets, DataFrames

########
# WORLD TUPLE: (STATE, ACTION, TRANSITION, REWARD)
# NOTE: can be run in two modes
#     a) DETERMINISTIC mode, see sister notebook: given Transition and Rewards probability functions.  See Decisions Under Uncertainty 4.2.5
#   *(b) MAX LIKELIHOOD modes, in this notebook: estimated Transition and Reward probability functions.  See Decisions Under Uncertainty 5.2
########
using POMDPs
# importall POMDPs


### STATE (MAX LIKELIHOOD): class, with objects/values estimated from the dataset
struct MaxlikelihoodState
    x::Int64                    # x position
    y::Int64                    # y position
    done::Bool                  # are we in a terminal state?
end

### ACTION (MAX LIKELIHOOD): class, with objects/values estimated from the dataset
# 1:left, 2:right, 3:up, 4:down
gridMaxlikelihoodActions = [:left, :right, :up, :down];


### WORLD (DETERMINISTIC): class
# grid mdp extends MDP{}
type MaxlikelihoodGrid <: MDP{MaxlikelihoodState, Symbol}   # MDP{StateType, ActionType} is parametrized by the State and the Action
    size_x::Int64                                           # x size of the grid
    size_y::Int64                                           # y size of the grid
    discount_factor::Float64                                # disocunt factor
    data::DataFrame                                         # loaded from file, as sampled
end


### TRANSITION (MAX LIKELIHOOD): Transition() determined from the dataset
# Return a distribution of neighbors, and a distribution of their associated probabilities

# https://github.com/JuliaStats/Distributions.jl/blob/master/src/multivariate/dirichletmultinomial.jl

# SparseCat(values,probabilities): sparse categorical distribution.  https://github.com/JuliaPOMDP/POMDPToolbox.jl/blob/master/src/distributions/sparse_cat.jl
function maxLikelihoodTransitionModel(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState, action::Symbol)

    mdp.dataset

    if isDone(state,mdp)
        return SparseCat([MaxlikelihoodState(state.x, state.y, true)], [1.0])
    end

    stateDistribution = MaxlikelihoodState[]
    probabilityDistribution = Float64[]

    actionIndexDict = getActionsIndexDict()
    neighborsDict = getNextCoordinatesDictionary(state)

    @printf("==> AT %s ACTION %s HAS: \n\t DISTRIBUTION OF STATES IS %s \n\t PROBABILITY DISTRIBUTION IS %s \n", state, action, stateDistribution, probabilityDistribution)
    return SparseCat(stateDistribution, probabilityDistribution)
end

### REWARD (MAX LIKELIHOOD): determined from the dataset
# add reward if state_now is in reward states
function getReward(reward_states, reward_values, state_now::MaxlikelihoodState)
    reward = 0.0
    if state_now.done
        reward = 0.0
    end
    for i = 1:length(reward_states)
        if isSamePosition(state_now, reward_states[i])
            reward += reward_values[i]
        end
    end
    return reward
end


### WORLD (MAX LIKELIHOOD): factory
# class constructor with default values
function MaxlikelihoodGrid(;sx::Int64=10,                                       # size_x
                    sy::Int64=10,                                               # size_y
                    discount_factor::Float64=0.9,                               # discount factor
                    data::DataFrame=dataframe)                                  # as loaded from file: default none
    return MaxlikelihoodGrid(sx, sy, discount_factor, data)
end



### STATE: DONE
# implement POMDPs.isterminal
function isMaxlikelihoodTerminal(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState)
    # TODO
    # return isMaxlikelihoodTerminal(mdp, state)
    return 1
end

### STATE: define
# implement POMDPs.states
function getMaxlikelihoodStateSpace(mdp::MaxlikelihoodGrid)
    # TODO
    # return getMaxlikelihoodStateSpace(mdp)
    return 1
end

### STATE: index
# implement POMDPs.state_index
function getMaxlikelihoodStateIndex(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState)
    # TODO
    # return getMaxlikelihoodStateIndex(mdp,state)
    return 1
end

### STATE: count
# implement POMDPs.n_states
function getMaxlikelihoodNumberOfStates(mdp::MaxlikelihoodGrid)
    # TODO
    # return getMaxlikelihoodNumberOfStates(mdp)
    return 1
end

### DISCOUNT (Maxlikelihood):
# implement POMDPs.discount
function getMaxlikelihoodDiscountFactor(mdp::MaxlikelihoodGrid)
    # TODO
    # return getMaxlikelihoodDiscountFactor(mdp)
    return 1
end

### ACTION: index
# POMDPs.action_index
function getMaxlikelihoodActionsIndexDict(mdp::MaxlikelihoodGrid, act::Symbol)
    # TODO
    # actionIndex = getMaxlikelihoodActionsIndexDict()
    # return actionIndex[act]
    return 1
end

### TRANSITION: configure the transition model
# implement POMDPs.transition
function MaxlikelihoodTransitionModel(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState, action::Symbol)
    # TODO
    # return MaxlikelihoodTransitionModel(mdp, state, action)
    return 1
end

### REWARD: utility of a (state,action,statePrime) datapoint in a world
# implement POMDPs.reward
function getMaxlikelihoodReward(mdp::MaxlikelihoodGrid, state::MaxlikelihoodState, action::Symbol, statePrime::MaxlikelihoodState)
    # TODO
    # return getMaxlikelihoodReward(mdp.reward_states, mdp.reward_values, state)
    return 1
end