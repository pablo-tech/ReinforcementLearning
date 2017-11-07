#################################
#MARKOV DECISION PROCESS: MOUNTAIN CAR
#################################
using POMDPs
# importall POMDPs
using TabularTDLearning
using POMDPModels
using RDatasets, DataFrames


### STATE: class.  Tuple (position, velocity)
# State measurements are given by integers with 500 possible position values and 100 possible velocity values (50,000 possible state measurements).
# 1+pos+500*vel gives the integer corresponding to a state with position pos and velocity vel.
# END: when the goal (the flag) is reached at the top of the hill
struct CarState
    index::Int64                           # 1+pos+500*vel
    #position::Int64                       # position
    #velocity::Int64                       # velocity
    done::Bool                             # are we in a terminal state?
end

### ACTION: class
# There are 7 actions that represent different amounts of acceleration
getCarActions = [1,2,3,4,5,6,7]

getNumCarActions = 7
# n_actions(mdp::MountainCar) = 7

### ACTION: index
function getCarActionIndex(mdp::MountainCar, action::Int64)
    return action
end


### WORLD (MOUNTAIN CAR): class
type MountainCar <: MDP{CarState, Symbol}  # MDP{StateType, ActionType} is parametrized by the State and the Action
  jackpot::Float64                              # reached the flag the top of the hill
  data::DataFrames.DataFrame                    # the dataset to solve for
  stateSpace::Vector{CarState}                  # the states
  discount::Float64
end

### WORLD (MOUNTAIN CAR): factory
# class constructor with default values
function MountainCar(;jackpot::Float64=0.0,                         # have not hit jackpot by default
                    data::DataFrames.DataFrame=dataframe,           # from file
                    stateSpace::Vector{CarState}=stateSpace,        # CarState[]
                    discount::Float64=1.0)                          # no discount
    return MountainCar(jackpot, data, stateSpace, discount)
end

### STATE: is terminal?
function isCarTerminal(mdp::MountainCar, state::CarState)   # vs s::Int64
    return state.done
end

### STATE: get states
function getCarStates(mdp::MountainCar)
    return mdp.stateSpace
end

### STATE: get states
function getNumCarStates(mdp::MountainCar)
    return length(getCarStates(mdp))
    #dataset = mdp.data
    #columnNames = names(dataset)
    #stateKey = columnNames[1]
    #numRows = length(dataset[stateKey])
    #return numRows
end


### STATE: get state index
function getCarStateIndex(mdp::MountainCar, state::CarState)
    return state.index
end

