#################################
#TEST MARKOV DECISION PROCESS: GRID (GENERATIVE: MAX LIKELYHOOD)
#NOTE: probabilistic GRID problem solved with OWN solvers.  Some experiments POMDPs.jl done as well.
#################################
include("imarkov-io.jl");
include("imarkov-grid-maxlikelihood.jl");
include("imarkov-grid-maxlikelihood-POMDPs.jl");
include("imarkov-solve-grid-maxlikelihood-ONENOW.jl");
include("imarkov-solve-grid-maxlikelihood-POMDP.jl");
include("imarkov-simulate-grid-maxlikelihood.jl");
using POMDPs
importall POMDPs
# using RDatasets, DataFrames



########
#CONFIG: configurable probability distribution
########

function getDiscountFactor()
    return 0.95
end

function getSizeX()
    return 10
end

function getSizeY()
    return 10
end

function getFileName()
    return "small.csv"
end

check(getFileName(), 100)

function getData()
    return getDataset(getFileName())
end

dataset = getData()

########
#PROBLEM: WORLD
########

mdp = MaxlikelihoodGrid(getSizeX(), getSizeY(), getDiscountFactor(), getData())
#mdp.discount_factor = getDiscountFactor()
#mdp.size_x = getSizeX()
#mdp.size_y = getSizeY()
#mdp.data = getDataset(getFileName())

#@printf("sizeX: %d \n", mdp.size_x)


########
#POPMD.jl: REQUIREMENTS EXPERIMENT
########

# METHOD REQUIREMENTS: Returns a list of the following functions required by the solver for your problem
# mdp::MaxlikelihoodGrid(getSizeX(), getSizeY(), getDiscountFactor(), dataset)
function getImplementationRequirements(mdp::MaxlikelihoodGrid)

    @requirements_info getUtilitySolver(mdp) mdp
    # @requirements_info ValueIterationSolver() MaxlikelihoodGrid()

    # https://github.com/JuliaPOMDP/DiscreteValueIteration.jl
    # discount(::MDP)
    # n_states(::MDP)
    # n_actions(::MDP)
    # transition(::MDP, ::State, ::Action)
    # reward(::MDP, ::State, ::Action, ::State)
    # state_index(::MDP, ::State)
    # action_index(::MDP, ::Action)
    # actions(::MDP, ::State)
    # iterator(::ActionSpace)
    # iterator(::StateSpace)
    # iterator(::StateDistribution)
    # pdf(::StateDistribution, ::State)
    # states(::MDP)
    # actions(::MDP)
end


# getImplementationRequirements(mdp)




########
#DATASET VS. POLICY
########

### INPUT size
# $ wc -l small.csv
#   50001 small.csv

### OUTPUT size
# Each output file should contain an action for every possible state in the problem
# small.policy => 100

# TODO: output file should be same name ".policy"


numStates = 100     # TODO
numActions = 4      # TODO

#"s","a","r","sp"
columnNames = names(dataset)
stateKey = columnNames[1]
actionKey = columnNames[2]
rewardKey = columnNames[3]
statePrimeKey = columnNames[4]
numRows = length(dataset[stateKey])
println("COLUMN NAMES ", stateKey, " ", actionKey, " ", rewardKey, " ", statePrimeKey, " x ", numRows)


########
#POLICY
########

qTable = zeros(numStates,numActions)
roTable = zeros(numStates,numActions)
visitCount = zeros(numStates,numActions)

solver = getMaxLikelihoodSolver(mdp)

qTable = solve(solver, mdp)


########
#OUTPUT
########


output_filename = "small.policy"

try
    rm(output_filename)
except
    println("could not delete inexistent file: " * input_filename)
end

out_file = open(output_filename, "w")

for stateId in 1:100
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

