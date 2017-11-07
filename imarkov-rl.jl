#################################
# REINFORCED LEARNING: see README and Instructions
#################################
## IMPORTANT: first run the julia configuration
#include("Juliacommand.jl");
include("imarkov-io.jl");
## worlds second
include("imarkov-grid-probabilistic.jl");
include("imarkov-grid-maxlikelihood.jl");
include("imarkov-car.jl");
include("imarkov-mystery.jl");


########
#DATASET:
########

### INPUT size
# $ wc -l small.csv
#   50001 small.csv
# $ wc -l medium.csv
#  100001 medium.csv
# $ wc -l large.csv
# 1000001 large.csv

### OUTPUT size
# Each output file should contain an action for every possible state in the problem
# small.policy => 100
# medium.policy => 50000
# large.policy  => 10101010

# TODO: output file should be same name ".policy"


########
#SMALL (GRID: MAX LIKELIHOOD)
########
# SET TO THE TUNE OF: "Bombay's a grid, Delhi Swings" by Midival Punditz, a Delhi-based Asian underground band

#mdp = MaxlikelihoodGrid()  # default constructor
#mdp.discount_factor
# mdp.tprob =

# columns: state, action, reward, state prime ("s","a","r","sp")
# small.policy should contain 100 lines

#input("small.csv")
#output("small.csv")
check("small.csv", 100)


########
#MEDIUM (MOUNTAIN CAR)
########
# SET TO THE TUNE OF: "I Get Around," Beach Boys


# MountainCarContinuous-v0 environment from Open AI Gym with altered parameters.
# State measurements are given by integers with 500 possible position values and 100 possible velocity values
# (50,000 possible state measurements).
# 1+pos+500*vel gives the integer corresponding to a state with position pos and velocity vel. There are 7 actions
# that represent different amounts of acceleration.
# This problem is undiscounted, but ends when the goal (the flag) is reached.
# Note that, since the discrete state measurements are calculated after the simulation, the data in medium.csv
# does not quite satisfy the Markov property

# animated: http://web.stanford.edu/class/aa228/projects/2/out.gif

# columns: state, action, reward, state prime ("s","a","r","sp")
# medium.policy should contain 50000 lines

#input("medium.csv")
#output("medium.csv")
check("medium.csv", 50000)


########
#LARGE (MYSTERY)
########
# SET TO THE TUNE OF: "LA Woman," The Doors

# MDP with 10101010 states and 125 actions, with a discount factor of 0.95.

# columns: state, action, reward, state prime ("s","a","r","sp")
# large.policy should contain 10101010 lines

#input("large.csv")
#output("large.csv")
check("large.csv", 10101010)


### REFERENCES
# POMDPs.jl: A Framework for Sequential Decision Making under Uncertainty http://www.jmlr.org/papers/v18/16-300.html