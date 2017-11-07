#################################
#MARKOV DECISION PROCESS: MYSTERY!
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
struct MysteryState
    index::Int64                           # 1+pos+500*vel
    #position::Int64                       # position
    #velocity::Int64                       # velocity
    done::Bool                             # are we in a terminal state?
end

### ACTION: class
# There are 7 actions that represent different amounts of acceleration
# getMysteryActions = [1,2,3,4,5,6,7]
getMysteryActions = 1:125

getNumCarActions = 7
# n_actions(mdp::MysteryWorld) = 7

### ACTION: index
# 125 actions
#function getMysteryActionIndex(mdp::MysteryWorld, action::Int64)
#    return action
#end


### WORLD (MYSTERY): class
type MysteryWorld <: MDP{MysteryState, Symbol}  # MDP{StateType, ActionType} is parametrized by the State and the Action
  jackpot::Float64                              # reached the flag the top of the hill
  data::DataFrames.DataFrame                    # the dataset to solve for
  stateSpace::Vector{MysteryState}                  # the states
  discount::Float64
end

### WORLD (MYSTERY): factory
# class constructor with default values
function MysteryWorld(;jackpot::Float64=0.0,                         # have not hit jackpot by default
                    data::DataFrames.DataFrame=dataframe,           # from file
                    stateSpace::Vector{MysteryState}=stateSpace,        # MysteryState[]
                    discount::Float64=1.0)                          # no discount
    return MysteryWorld(jackpot, data, stateSpace, discount)
end

### STATE: is terminal?
function isMysteryTerminal(mdp::MysteryWorld, state::MysteryState)   # vs s::Int64
    return state.done
end

### STATE: get states
function getMysteryStates(mdp::MysteryWorld)
    return mdp.stateSpace
end

### STATE: get states
# TODO: get from length of stateSpace
#function getNumMysteryStates(mdp::MysteryWorld)
#    dataset = mdp.data
#    columnNames = names(dataset)
#    stateKey = columnNames[1]
#    numRows = length(dataset[stateKey])
#    return numRows
#end

### STATE: get state index
function getMysteryStateIndex(mdp::MysteryWorld, state::MysteryState)
    return state.index
end