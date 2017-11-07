########
#WORLD (EXPLICIT: PROBABILISTIC): as a Markov Decision Process
# NOTE: this world is completely observable. Transition() and Reward() functions are given.
########
using POMDPs
# importall POMDPs


########
# WORLD TUPLE: (STATE, ACTION, TRANSITION, REWARD)
# NOTE: can be run in two modes
#   *(a) PROBABILISTIC mode, in this notebook: given Transition and Rewards probability functions.  See Decisions Under Uncertainty 4.2.5
#     b) MAX LIKELIHOOD modes, see sister notebook: estimated Transition and Reward probability functions.  See Decisions Under Uncertainty 5.2
########
using POMDPs
# importall POMDPs


### STATE: (PROBABILISTIC): class
struct ProbabilisticState
    x::Int64                        # x position
    y::Int64                        # y position
    done::Bool                      # are we in a terminal state?
end

### ACTION (PROBABILISTIC): class
gridProbabilisticActions = [:up, :down, :left, :right];

### WORLD (PROBABILISTIC): class
# grid mdp extends MDP{}
type ProbabilisticGrid <: MDP{ProbabilisticState, Symbol}   # MDP{StateType, ActionType} is parametrized by the State and the Action
    size_x::Int64                                           # x size of the grid
    size_y::Int64                                           # y size of the grid
    reward_states::Vector{ProbabilisticState}               # the states in which agent recieves reward
    reward_values::Vector{Float64}                          # reward values for those states
    tprob::Float64                                          # probability of transitioning to the desired state
    discount_factor::Float64                                # disocunt factor
end

### WORLD (PROBABILISTIC): factory
# class constructor with default values
function ProbabilisticGrid(;
                    sx::Int64=10,                                               # size_x
                    sy::Int64=10,                                               # size_y
                    rs::Vector{ProbabilisticState}=[ProbabilisticState(4,3),    # reward states:
                        ProbabilisticState(4,6), ProbabilisticState(9,3),
                        ProbabilisticState(8,8)],
                    rv::Vector{Float64}=rv = [-10.,-5,10,3],                    # reward values
                    tprob::Float64=0.7,                                         # probability of transition
                    discount_factor::Float64=0.9)                               # discount factor
    return ProbabilisticGrid(sx, sy, rs, rv, tprob, discount_factor)
end

### WORLD (PROBABILISTIC)
function getProbabilisticDiscountFactor(mdp::ProbabilisticGrid)
    mdp.discount_factor
end

### TRANSITION (PROBABILISTIC): with known/probabilistic Transition() probability
# Return a distribution of neighbors, and a distribution of their associated probabilities
# SparseCat(values,probabilities): sparse categorical distribution.  https://github.com/JuliaPOMDP/POMDPToolbox.jl/blob/master/src/distributions/sparse_cat.jl
function ProbabilisticTransitionModel(mdp::ProbabilisticGrid, state::ProbabilisticState, action::Symbol)
    if isDone(state,mdp)
        return SparseCat([ProbabilisticState(state.x, state.y, true)], [1.0])
    end

    stateDistribution = ProbabilisticState[]
    probabilityDistribution = Float64[]

    actionIndexDict = getProbabilisticActionsIndexDict()
    neighborsDict = getNextCoordinatesDictionary(state)

    for a in gridProbabilisticActions
        possibleNextState = getNextState(state, a)
        if a==action                                                                                    # (s, intended a)
            if isInGrid(mdp, possibleNextState)
                push!(stateDistribution, possibleNextState)
                push!(probabilityDistribution, mdp.tprob)
            else
                return SparseCat([ProbabilisticState(state.x, state.y)], [1.0])                             # end of life
            end
        elseif isInGrid(mdp, possibleNextState)
            inBoundCount = countNeighborsInGrid(neighborsDict, mdp)                                     # number of neighbors in grid
            if inBoundCount > 1     # beyond the intended action
                individualProbability = getIndividualUnintendedProbability(inBoundCount, mdp)           # (s, not intended a with s' in grid)
                push!(stateDistribution, possibleNextState)
                push!(probabilityDistribution, individualProbability)
            end
        end
    end

    @printf("==> AT %s ACTION %s HAS: \n\t DISTRIBUTION OF STATES IS %s \n\t PROBABILITY DISTRIBUTION IS %s \n", state, action, stateDistribution, probabilityDistribution)
    return SparseCat(stateDistribution, probabilityDistribution)
end

### REWARD (PROBABILISTIC):
# add reward if state_now is in reward states
function getProbabilisticReward(reward_states, reward_values, state_now::ProbabilisticState)
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


########
#STATE (PROBABILISTIC)
########

### STATE: terminal?
function isProbabilisticTerminal(mdp::ProbabilisticGrid, state::ProbabilisticState)
    return state.done
end

### STATE: factory
# every possible (x,y,done) tuple
function getProbabilisticStateSpace(mdp::ProbabilisticGrid)
    stateSpace = ProbabilisticState[]
        for done = 0:1, y = 1:mdp.size_y, x = 1:mdp.size_x
            push!(stateSpace, ProbabilisticState(x,y,done))
            # println("(", x, ",", y, done, ")")
        end
    return stateSpace
end

### STATE FACTORY: factory. By default state is not terminal
ProbabilisticState(x::Int64, y::Int64) = ProbabilisticState(x,y,false)

### STATE MATCH: Same position for two states?
isSamePosition(s1::ProbabilisticState, s2::ProbabilisticState) = s1.x == s2.x && s1.y == s2.y

### STATE COUNT
function getProbabilisticNumberOfStates(mdp::ProbabilisticGrid)
    return 2*mdp.size_x*mdp.size_y
end

### STATE INDEX: linear
# sub2ind((10, 10), x, y) to find the integer state index from the x and y coordinates for a 10x10 grid
function getProbabilisticStateIndex(mdp::ProbabilisticGrid, state::ProbabilisticState)
    stateDone = Int(state.done + 1)
    return sub2ind((mdp.size_x, mdp.size_y, 2), state.x, state.y, stateDone)
end

########
#ACTION (PROBABILISTIC)
########

function getProbabilisticActionsIndexDict()
    return Dict(:right=>1, :left=>2, :down=>3, :up=>4)
end

function getProbabilisticActionIndex()
    actionIndex = getProbabilisticActionsIndexDict()
    return actionIndex[act]
end

########
#TRANSITION (PROBABILISTIC)
########

### TRANSITION: is coordinate in the grid
function isInGrid(mdp::ProbabilisticGrid,x::Int64,y::Int64)
    isIn = false
    if 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y
        isIn = true
    end
    # @printf("x,y is isIn? %s %d %d\n", isIn? "true" : "false", x, y)
    return isIn
end

### TRANSITION: is state in grid
isInGrid(mdp::ProbabilisticGrid, state::ProbabilisticState) = isInGrid(mdp, state.x, state.y);

### TRANSITION: how many neighbors in grid
function countNeighborsInGrid(neighbors::Dict, mdp::ProbabilisticGrid)
    total = sum(isInGrid(mdp, neighbor) for neighbor in values(neighbors))
    # @printf("Number of neighbors in grid? %d \n", total)
    return total
end

### TRANSITION: get any of all plausible next states
function getNeighborToRight(state::ProbabilisticState, done::Bool)
    return ProbabilisticState(state.x+1, state.y, done)
end

function getNeighborToLeft(state::ProbabilisticState, done::Bool)
    return ProbabilisticState(state.x-1, state.y, done)
end

function getNeighborDown(state::ProbabilisticState, done::Bool)
    return ProbabilisticState(state.x, state.y-1, done)
end

function getNeighborUp(state::ProbabilisticState, done::Bool)
    return ProbabilisticState(state.x, state.y+1, done)
end

### TRANSITION: action -> nextState, regardless of grid boundaries
function getNextCoordinatesDictionary(state::ProbabilisticState)
    next = Dict(:right=>getNeighborToRight(state,false), :left=>getNeighborToLeft(state,false), :down=>getNeighborDown(state,false), :up=>getNeighborUp(state,false))
    # print("ALL NEXT COORDINATES: ", next, "...right COORDINATE: " , next[:right], "\n")
    return next
end

### TRANSITION: next state from current and action
function getNextState(state::ProbabilisticState, action::Symbol)
    stateOptions = getNextCoordinatesDictionary(state)
    nextState = stateOptions[action]
    #@printf("STATE %s ACTION %s => %s\n", state, action, nextState)
    return nextState
end

### TRANSITION: probability
# returns array for probability in all possible direction, wether in or out of grid
function getNewProbabilityDistribion()
    defaultTransitionProbability = 0.0
    numberOfNeighbors = 4
    probability = fill(defaultTransitionProbability, numberOfNeighbors)
    return probability
end

### TRANSITION: definition of done
function isDone(state::ProbabilisticState, mdp::ProbabilisticGrid)
    if state.done
        return true
    elseif state in mdp.reward_states
        return true
    end
    return false
end

### PROBABILITY: for not intended directions [assumes success of intended direction, else end of game]
function getIndividualUnintendedProbability(inBoundNeighbors::Int, mdp::ProbabilisticGrid)
    sharedProbability = 1.0 - mdp.tprob             # probability of the intended direction
    divideAmong = inBoundNeighbors - 1              # deduct the success of the intendded direction
    individualProbability = sharedProbability/divideAmong
    # @printf("intended direction probability: %f to divide among %f and thus shared probability is %f \n", mdp.tprob, divideAmong, individualProbability)
    return individualProbability
end


########
#REWARD (PROBABILISTIC)
########

# reward states and values are defined in the world

