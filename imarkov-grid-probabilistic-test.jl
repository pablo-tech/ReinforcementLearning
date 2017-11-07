#################################
#TEST MARKOV DECISION PROCESS: GRID (EXPLICIT: PROBABILISTIC)
#BONUS: probabilistic GRID problem solved with POMDPs.jl
#################################
include("imarkov-grid-probabilistic.jl");
include("imarkov-grid-probabilistic-POMDPs.jl");
include("imarkov-solve-grid-probabilistic-ONENOW.jl");
include("imarkov-solve-grid-probabilistic-POMDP.jl");
include("imarkov-simulate-grid-probabilistic.jl");
using POMDPs
importall POMDPs


########
#CONFIG: configurable probability distribution
########

function getIntendedProbability()
    return 0.7
end

function getSizeX()
    return 10
end

function getSizeY()
    return 10
end


########
#PROBLEM: WORLD
########

mdp = ProbabilisticGrid()  # default constructor
mdp.tprob = getIntendedProbability()
mdp.size_x = getSizeX()
mdp.size_y = getSizeY()

@printf("sizeX: %d \n", mdp.size_x)


########
#POPMD.jl: REQUIREMENTS
########

# METHOD REQUIREMENTS: Returns a list of the following functions required by the solver for your problem
function getImplementationRequirements(mdp::ProbabilisticGrid)

    @requirements_info getUtilitySolver(mdp) mdp

    #INFO: POMDPs.jl requirements for solve(::ValueIterationSolver, ::Union{POMDPs.MDP,POMDPs.POMDP}) and dependencies. ([✔] = implemented correctly; [X] = missing)
    #
    # For solve(::ValueIterationSolver, ::Union{POMDPs.MDP,POMDPs.POMDP}):
    # [✔] discount(::ProbabilisticGrid)
    # [✔] n_states(::ProbabilisticGrid)
    # [✔] n_actions(::ProbabilisticGrid)
    # [✔] transition(::ProbabilisticGrid, ::ProbabilisticState, ::Symbol)
    # [✔] reward(::ProbabilisticGrid, ::ProbabilisticState, ::Symbol, ::ProbabilisticState)
    # [✔] state_index(::ProbabilisticGrid, ::ProbabilisticState)
    # [✔] action_index(::ProbabilisticGrid, ::Symbol)
    # [✔] actions(::ProbabilisticGrid, ::ProbabilisticState)
    # [✔] iterator(::Array)
    # [✔] iterator(::Array)
    # [✔] iterator(::SparseCat)
    # [✔] pdf(::SparseCat, ::ProbabilisticState)
    # For ordered_states(::Union{POMDPs.MDP,POMDPs.POMDP}) (in solve(::ValueIterationSolver, ::Union{POMDPs.MDP,POMDPs.POMDP})):
    # [✔] states(::ProbabilisticGrid)
    # For ordered_actions(::Union{POMDPs.MDP,POMDPs.POMDP}) (in solve(::ValueIterationSolver, ::Union{POMDPs.MDP,POMDPs.POMDP})):
    # [✔] actions(::ProbabilisticGrid)

end

getImplementationRequirements(mdp)



########
#STATE
########

# a state
s = ProbabilisticState(9,2)
@printf("x: %d \n", s.x)

# reward states
@printf("reward states: %s \n", mdp.reward_states)

# state space
for state in getProbabilisticStateSpace(mdp);
    #@printf("state space: %s \n", state)
end
@printf("State space size: %d \n", length(getProbabilisticStateSpace(mdp)))

# first state
@printf("fist state %s \n", getProbabilisticStateSpace(mdp)[1])

## hasInboundTransition: true for all in grid, except (0,0)
isInGrid(mdp, 0, 0)
isInGrid(mdp, 3, 3)
isInGrid(mdp, 10, 10)
isInGrid(mdp, 11, 11)


########
#ACTION
########

actionProbabilisticIndexDic = getProbabilisticActionsIndexDict()
for action in gridProbabilisticActions
    index = actionProbabilisticIndexDic[action]
    @printf("ACTION: %s...INDEX: %d\n", action, index)
end


########
#SIMULATION
########

startFrom = ProbabilisticState(4,1)
maxSteps = 10
#  mdp.tprob = 1
runSimulation(mdp, startFrom, maxSteps)


########
#POLICY
########

policy = getPolicySolution(mdp)

### EVALUATE: policy at (8,3)
s = ProbabilisticState(8,3)
a = action(policy, s)
@printf("\n\n FROM POLICY => s:%s BEST a:%s\n", s, a)


########
#REWARD: with POMDPs.jl SOLVER
########

startFrom = ProbabilisticState(4,1)
maxSteps = 100
history = getTotalReward(mdp, startFrom, maxSteps, policy)


########
#REPLAY: with POMDPs.jl SOLVER
########

replayHistory(history)


########
#REWARD: with OWN SOLVER
########

# TODO


########
#OUTPUT
########

# print action for every state
for s in states(mdp)
    println("POLICY (s->a)... (State:$s->Action:$(action(policy, s)))")
end

### EVALUATE: policy at (9,2)
s = ProbabilisticState(9,2)
a = action(policy, s)
@printf("\n\n FROM POLICY => s:%s BEST a:%s\n", s, a)

# create_state not defined
# s = create_state(mdp) # this can be any valid state
# a = action(policy, s) # returns the optimal action for state s