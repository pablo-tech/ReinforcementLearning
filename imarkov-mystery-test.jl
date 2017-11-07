########
#WORLD (REINFORCED LEARNING): as a Markov Decision Process
#NOTE: this world is completely observable. Model uncertainty: Transition() and Reward() not known.
#Use the dataset to build the model and make decisions
#IMPORTANT: reinforcement learning problem solved with OWN solvers.  Some experiments POMDPs.jl done as well.
########
include("imarkov-io.jl");
include("imarkov-mystery.jl");
include("imarkov-mystery-POMDPs.jl");
include("imarkov-solve-mystery-ONENOW.jl");
include("imarkov-solve-mystery-POMDP.jl");
include("imarkov-simulate-mystery.jl");
using POMDPs
# importall POMDPs
using TabularTDLearning
using POMDPModels
using RDatasets, DataFrames





########
#CONFIG: configurable probability distribution
########

# Meant to be a challenge, and blindly applying q-learning or Sarsa is sufficient.
#
# * MDP with 10101010 states
# * 125 actions
# * discount factor of 0.95
# * Each line in the CSV file represents a transition from state (s) to (sp) (along with the associated reward (r)), when taking an action (a)
# * You might not have data for every state. Neither are all states equally important. Use your best judgement in handling this
#
# But that is not likely to yield a good policy.  Look at the state indices and their transitions. There's a structure to the large problem.


function getDiscount()
    return 1    # no discount
end

function getFileName()
    return "large.csv"
end

function getData()
    return getDataset(getFileName())
end

check(getFileName(), 10101010)

dataset = getData()

# "s","a","r","sp"
columnNames = names(dataset)
stateKey = columnNames[1]
actionKey = columnNames[2]
rewardKey = columnNames[3]
statePrimeKey = columnNames[4]
numRows = length(dataset[stateKey])
println("COLUMN NAMES ", stateKey, " ", actionKey, " ", rewardKey, " ", statePrimeKey, " x ", numRows)


########
#PROBLEM: WORLD
########

mdp = MysteryWorld(getDiscount(), dataset, MysteryState[], getDiscount())

@printf("discount: %f \n", mdp.discount)



########
#POPMD.jl: REQUIREMENTS EXPERIMENT
########

# METHOD REQUIREMENTS: Returns a list of the following functions to be implemented for your MDP
function getImplementationRequirements()

    @requirements_info QLearningSolver() MysteryWorld()

end

# getImplementationRequirements()


########
#STATE
########

# a state
#s = ProbabilisticState(9,2)
#@printf("x: %d \n", s.x)

# reward states
#@printf("reward states: %s \n", mdp.reward_states)

# state space
#for state in getProbabilisticStateSpace(mdp);
#    #@printf("state space: %s \n", state)
#end
#@printf("State space size: %d \n", length(getProbabilisticStateSpace(mdp)))

# first state
#@printf("fist state %s \n", getProbabilisticStateSpace(mdp)[1])

## hasInboundTransition: true for all in grid, except (0,0)
#isInGrid(mdp, 0, 0)
#isInGrid(mdp, 3, 3)
#isInGrid(mdp, 10, 10)
#isInGrid(mdp, 11, 11)


########
#ACTION
########

#actionProbabilisticIndexDic = getProbabilisticActionsIndexDict()
#for action in gridProbabilisticActions
#    index = actionProbabilisticIndexDic[action]
#    @printf("ACTION: %s...INDEX: %d\n", action, index)
#end


########
#SIMULATION
########

#startFrom = ProbabilisticState(4,1)
#maxSteps = 10
#  mdp.tprob = 1
#runSimulation(mdp, startFrom, maxSteps)


########
#DATASET VS POLICY
########

### INPUT size
# $ wc -l large.csv
# 1000001 large.csv

### OUTPUT size
# Each output file should contain an action for every possible state in the problem
# large.policy  => 10101010

# TODO: output file should be same name ".policy"

numActions = length(getMysteryActions)
numStates = 10101010

########
#POLICY
########


qTable = zeros(numStates,numActions)
visitCount = zeros(numStates,numActions)

solver = getSarsaLambdaSolver(mdp)

qTable = solve(solver, mdp)


### EVALUATE: policy at (8,3)
#s = ProbabilisticState(8,3)
#a = action(policy, s)
#printf("\n\n FROM POLICY => s:%s BEST a:%s\n", s, a)


########
#OUTPUT
########

output_filename = "large.policy"

try
    rm(output_filename)
except
    println("could not delete inexistent file: " * output_filename)
end

out_file = open(output_filename, "w")

for stateId in 1:10101010
    try
        policyId = getArgmaxPolicy(qTable, stateId)
        println(out_file, policyId)
        #@printf("%d ACTION RECOMMENDED FOR s=%d \n", policyId, stateId)
    catch e
        println(out_file, 3)
        #@printf("NO SOUP FOR YOU! s=%d %s\n", stateId, e)
    end
end

close(out_file)


########
#REWARD: with POMDPs.jl SOLVER
########

#startFrom = ProbabilisticState(4,1)
#maxSteps = 100
#history = getTotalReward(mdp, startFrom, maxSteps, policy)


########
#REPLAY: with POMDPs.jl SOLVER
########

#replayHistory(history)


########
#REWARD: with OWN SOLVER
########

# TODO

