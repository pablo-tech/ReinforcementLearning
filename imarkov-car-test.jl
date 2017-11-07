########
#WORLD (REINFORCED LEARNING): as a Markov Decision Process
#NOTE: this world is completely observable. Model uncertainty: Transition() and Reward() not known.
#Use the dataset to build the model and make decisions
#IMPORTANT: reinforcement learning problem solved with OWN solvers.  Some experiments POMDPs.jl done as well.
########
include("imarkov-io.jl");
include("imarkov-car.jl");
include("imarkov-car-POMDPs.jl");
include("imarkov-solve-car-ONENOW.jl");
include("imarkov-solve-car-POMDP.jl");
include("imarkov-simulate-car.jl");
using POMDPs
# importall POMDPs
using RDatasets, DataFrames
# DataFrames.DataFrame



########
#CONFIG: configurable probability distribution
########

# * Discretized State measurements are given by integers with 500 possible position values and 100 possible velocity values (50,000 possible state measurements).
# * 1+pos+500*vel is the formula that was used to discretize to integer states, utilizing position pos and velocity vel.
# * There are 7 actions that represent different amounts of acceleration.
# * This problem is undiscounted, but ends when the goal (the flag) is reached. Note that, since the discrete state measurements are calculated after the simulation, the data in medium.csv does not quite satisfy the Markov property.


function getDiscount()
    return 1.0      # no discount
end

function getJackpot()
    return 0.0      # haven't hit it yet
end

function getFileName()
    return "medium.csv"
end

function getData()
    return getDataset(getFileName())
end

## dataset
dataset = getData()
check("medium.csv", 50000)

# "s","a","r","sp"
columnNames = names(dataset)
stateKey = columnNames[1]
actionKey = columnNames[2]
rewardKey = columnNames[3]
statePrimeKey = columnNames[4]
numRows = length(dataset[stateKey])
println("COLUMN NAMES ", stateKey, " ", actionKey, " ", rewardKey, " ", statePrimeKey, " x ", numRows)


for rowNum in 1:10  # first row is header

    row = dataset[rowNum:rowNum,:]  # just the one row, every column

    state = row[stateKey]
    action = row[actionKey]
    reward = row[rewardKey]
    statePrime = row[statePrimeKey]

    print("s=", state, " a=", action, " reward=", reward, " statePrime=", statePrime, "\n")

    b = (123)[1]+2
    println("PARSING ROW ", b)

    for columnName in columnNames
        ## PROCESS ROW: reinforced learning
        # print("(", columnName, ",", row, ")")
    end
end

#COLUMN NAMES s a r sp x 100000
#s=[24715] a=[1] reward=[-225] statePrime=[24214]
#s=[24214] a=[1] reward=[-225] statePrime=[23713]
#s=[23713] a=[1] reward=[-225] statePrime=[23212]
#s=[23212] a=[1] reward=[-225] statePrime=[22711]
#s=[22711] a=[1] reward=[-225] statePrime=[22709]
#s=[22709] a=[1] reward=[-225] statePrime=[22207]
#s=[22207] a=[1] reward=[-225] statePrime=[21705]
#s=[21705] a=[1] reward=[-225] statePrime=[21703]
#s=[21703] a=[1] reward=[-225] statePrime=[21200]
#s=[21200] a=[1] reward=[-225] statePrime=[21197]




########
#PROBLEM: WORLD
########

# stateSpace = CarState[numStates]
mdp = MountainCar(getJackpot(), getData(), CarState[], getDiscount())

@printf("jackpot: %f \n", mdp.jackpot)


########
#POPMD.jl: REQUIREMENTS EXPERIMENT
########

# METHOD REQUIREMENTS: Returns a list of the following functions required by the solver for your problem
function getImplementationRequirements(mdp::MountainCar)

    @requirements_info getQlearningSolver(mdp) mdp

end

# getImplementationRequirements(mdp)


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
# $ wc -l medium.csv
#  100001 medium.csv

### OUTPUT size
# Each output file should contain an action for every possible state in the problem
# medium.policy => 50000

# TODO: output file should be same name ".policy"

numActions = length(getCarActions)
numStates = 50000


########
#POLICY
########

qTable = zeros(numStates,numActions)
visitCount = zeros(numStates,numActions)

solver = getSarsaLambdaSolver(mdp)

qTable = solve(solver, mdp)  #



### EVALUATE: policy at (8,3)
#s = ProbabilisticState(8,3)
#a = action(policy, s)
#@printf("\n\n FROM POLICY => s:%s BEST a:%s\n", s, a)



########
#OUTPUT
########


output_filename = "medium.policy"

try
    rm(output_filename)
except
    println("could not delete inexistent file: " * input_filename)
end

out_file = open(output_filename, "w")

for stateId in 1:50000
    try
        policyId = getArgmaxPolicy(qTable, stateId)
        println(out_file, policyId)
        @printf("%d ACTION RECOMMENDED FOR s=%d \n", policyId, stateId)
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






