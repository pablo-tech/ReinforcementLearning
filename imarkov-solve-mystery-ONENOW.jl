#################################
#SOLUTION FOR MARKOV DECISION PROBLEM: MYSTERY
#IMPORTANT: with OWN solvers
#################################





########
#SARSA LAMBDA
########

### SASA LAMBDA solver: class
# TODO: mutable struct SARSALambdaSolver <: Solver
mutable struct SARSALambdaSolverONENOW
    learning_rate::Float64
    lambda::Float64
    n_episodes::Int64
    max_episode_length::Int64
    eval_every::Int64
    n_eval_traj::Int64
    Q_vals::Matrix{Float64}             # (state,action) matrix.  Initialize with zeros(numRows,numColumns)
    eligibility::Matrix{Float64}
    # TODO: exploration_policy::Policy
end

function getSarsaLambdaSolver(mdp::MysteryWorld;
            learning_rate::Float64=0.1,
            lambda::Float64=0.9,
            n_episodes::Int64=5000,
            max_episode_length::Int64=50,
            eval_every::Int64=50,
            n_eval_traj::Int64=100,
            Q_vals::Matrix{Float64}=qTable,                         # qTable
            eligibility::Matrix{Float64}=visitCount)                # visit count N
    return SARSALambdaSolverONENOW(learning_rate, lambda, n_episodes, max_episode_length, eval_every, n_eval_traj, qTable, visitCount)
end

# SARSA LAMBDA: policy
function create_policy(solver::SARSALambdaSolverONENOW, mdp::Union{MDP,POMDP})
    # TODO: add Policy
    # return solver.exploration_policy.val
end


# SARSA LAMBDA: solve
# Credit is backpropagated exponentially to the (s,a) tuples leading to the goal
# Decisions Under Uncertainty: Algorithm 5.4
# https://github.com/JuliaPOMDP/TabularTDLearning.jl/blob/master/src/sarsa_lambda.jl
function solve(solver::SARSALambdaSolverONENOW, mdp::MysteryWorld, policy=create_policy(solver, mdp))

    #rng = solver.exploration_policy.uni.rng
    Q = solver.Q_vals
    ecounts = solver.eligibility
    #exploration_policy = solver.exploration_policy
    #sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)

    dataset = mdp.data

    numberOfEpisodes = solver.n_episodes
    numberOfEpisodes = convert(Integer, length(dataset[stateKey])/2)
    # numberOfEpisodes = length(dataset[stateKey])

    totalReward = 0


    # MAX EPISODES
    for i = 1:numberOfEpisodes             # An episode h ∈ H is a *sequence* of quadruples (s, a, o, r) of state s ∈ S

            try
                # DATASET
                row = dataset[i:i,:]  # just the one row, every column
                nextRow = dataset[i+1:i+1,:]

                s,a = (row[stateKey])[1], (row[actionKey])[1]
                # s = initial_state(mdp, rng)
                # a = action(exploration_policy, s)

                # MAX EPISODE LENGTH: number of passes over the dataset
                t = 0
                isTerminal = false
                maxEpisodeLengthTest = 5                                                  # solver.max_episode_length && !isTerminal
                while t<maxEpisodeLengthTest

                    # DATASET
                    sp, r = (row[statePrimeKey])[1], (row[rewardKey])[1] # observe results of action
                    totalReward = totalReward + r

                    # generation alternative:
                    # sp, r = generate_sr(mdp, s, a, rng); ap = action(exploration_policy, sp)

                    ap =  (nextRow[actionKey])[1]           # from suitable exploration strategy
                    #print("s=", s, " a=", a, " r=", r, " sp=", sp, " ap=", ap, "\n")

                    # INDEX: state and action
                    si = s; ai = a; spi = sp; api = ap
                    # si = getMysteryStateIndex(mdp, s); ai = getCarActionIndex(mdp, a);
                    # spi = getMysteryStateIndex(mdp, sp); api = getCarActionIndex(mdp, ap)

                    # DELTA: DAU 5.26
                    delta = r
                    try
                        delta = delta + mdp.discount * Q[spi,api] - Q[si,ai]
                    catch
                        delta = r
                    end

                    # VISIT COUNT: DAU 5.7
                    ecounts[si,ai] = 1.0      # non-zero starting pseudocount
                    try
                        ecounts[si,ai] += 1.0
                    catch e
                        # @printf("CAUGHT EXCEPTION AT ECOUNTS %s", e) # , e
                        ecounts[si,ai] = 1.0      # non-zero starting pseudocount
                    end

                    # PROPAGATE: all s, all a
                    minPropagationThreshold = 10            # 15     # 5000
                    if abs(delta)>minPropagationThreshold        # something to propagate
                        print("BACKPROPAGATING1:", "s=", (row[stateKey])[1], " a=", (row[actionKey])[1], " reward=", (row[rewardKey])[1], " statePrime=", (row[statePrimeKey])[1], "\n")
                        for es in 1:10101010   # TODO: getNumMysteryStates(mdp)
                            for ea in 1:125     # TODO: getNumMysteryActions
                                esi, eai = es, ea
                                #esi, eai = getMysteryStateIndex(mdp, es), getCarActionIndex(mdp, ea)
                                Q[esi,eai] += solver.learning_rate * delta * ecounts[esi,eai]
                                ecounts[esi,eai] *= mdp.discount * solver.lambda                    # exponential decay
                                if abs(Q[esi,eai])>0
                                    print("BACKPROPAGATING2: ", "esi=", esi, " eai=", eai, " r=", r, " delta=", delta, " Q[esi,eai]=", Q[esi,eai], " ecounts[esi,eai]=", ecounts[esi,eai], " ", now(), "\n")
                                end
                            end
                        end
                    end

                    s, a = sp, ap

                    t += 1

                end

                if i % solver.eval_every == 0
                #    r_tot = 0.0
                #    for traj in 1:solver.n_eval_traj
                #        r_tot += simulate(sim, mdp, policy, initial_state(mdp, rng))
                #    end
                     @printf("Through episode %d with *cummunlative* returns: %d, current row %s...at %s\n", i, totalReward, row, now())
                #    println("Through episode $i, Returns: $(r_tot/solver.n_eval_traj)")
                end

            catch e
                # @printf("CAUGHT EXCEPTION AT ROW %d", i) # , e
                @printf("CAUGHT EXCEPTION AT ROW %d: %s", i, e)
            end

    end

    @printf("QTABLE TO SCREEN, WATCH OUT! %s", Q)
    return Q                # Q[s,a]
    # return policy
end



### POLICY: get the best action for a given state, given a qTable
function getArgmaxPolicy(qTable::Matrix{Float64}, mysteryStateId::Int64)
    argMaxAction=[]
    bestQvalue = -100000        # minus infinity, looking for better
    for actionId in 1:125       # TODO: get from mdp
        coordinateValue = qTable[mysteryStateId, actionId]
        if coordinateValue>bestQvalue
            bestQvalue=coordinateValue
            push!(argMaxAction, actionId)
        end
    end
    try
        return pop!(argMaxAction)
    except
        return 2 # random action
    end
end





