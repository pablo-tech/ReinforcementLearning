#################################
#SOLUTION FOR MARKOV DECISION PROBLEM: GRID (MAX LIKELIHOOD)
#IMPORTANT: with OWN solvers
#################################



########
#MAX LIKELIHOOD Q
########


### MAX LIKELIHOOD LAMBDA solver: class (DAU 5.2)
# TODO: mutable struct SARSALambdaSolver <: Solver
mutable struct MaxLikelihoodSolverONENOW
    learning_rate::Float64
    lambda::Float64
    n_episodes::Int64
    max_episode_length::Int64
    eval_every::Int64
    n_eval_traj::Int64
    Q_vals::Matrix{Float64}                 # (state,action) matrix.  Initialize with zeros(numRows,numColumns)
    Ro::Matrix{Float64}                     # (state,action) matrix.  Initialize with zeros(numRows,numColumns)
    eligibility::Matrix{Float64}
    # TODO: exploration_policy::Policy
end

function getMaxLikelihoodSolver(mdp::MaxlikelihoodGrid;
            learning_rate::Float64=0.1,
            lambda::Float64=0.9,
            n_episodes::Int64=5000,
            max_episode_length::Int64=50,
            eval_every::Int64=50,
            n_eval_traj::Int64=100,
            Q_vals::Matrix{Float64}=qTable,                          # qTable: Rewards(si,aj) * Transition(si,aj).  Over all s'
            Ro::Matrix{Float64}=roTable,                             # roTable: Total rewards to date.  Over all s'
            eligibility::Matrix{Float64}=visitCount)                 # visit count N
    return MaxLikelihoodSolverONENOW(learning_rate, lambda, n_episodes, max_episode_length, eval_every, n_eval_traj, qTable, roTable, visitCount)
end

# MAX LIKELILHOOD: policy
function create_policy(solver::MaxLikelihoodSolverONENOW, mdp::Union{MDP,POMDP})
    # TODO: add Policy
    # return solver.exploration_policy.val
end


# MAX LIKELIHOOD: solve
## Cummulative: aggregated over all s'
# Ro(s,a)=R+r cummunlative reward plus increment
# N(s,a): Counts in a q matrix
# Dynamic
# R(s,a)=Ro(s,a)/N(s,a)  reward probability
# T(s,a)=N(s,ai)/N(s,a) ai over i
## Calculation
# And Q(s,a) = R(s,a)/N
# Then argmax the best a for s, iterating over columns for that row.  Which has greater R*T?
# Credit is backpropagated as
##
# Decisions Under Uncertainty: Algorithm 5.4
# https://github.com/JuliaPOMDP/TabularTDLearning.jl/blob/master/src/sarsa_lambda.jl
function solve(solver::MaxLikelihoodSolverONENOW, mdp::MaxlikelihoodGrid, policy=create_policy(solver, mdp))

    #rng = solver.exploration_policy.uni.rng
    Q = solver.Q_vals
    Ro = solver.Ro
    ecounts = solver.eligibility

    #exploration_policy = solver.exploration_policy
    #sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)

    dataset = mdp.data


    #totalReward = 0

    numberOfEpisodes = length(dataset[stateKey])
    # numberOfEpisodes = solver.n_episodes

    # MAX EPISODES
    for i = 1:numberOfEpisodes             # An episode h ∈ H is a *sequence* of quadruples (s, a, o, r) of state s ∈ S

        try
            # DATASET
            row = dataset[i:i,:]  # just the one row, every column
            # print("s=", (row[stateKey])[1], " a=", (row[actionKey])[1], " reward=", (row[rewardKey])[1], " statePrime=", (row[statePrimeKey])[1], "\n")
            nextRow = dataset[i+1:i+1,:]

            s,a = (row[stateKey])[1], (row[actionKey])[1]
            # s = initial_state(mdp, rng)
            # a = action(exploration_policy, s)

            # MAX EPISODE LENGTH: number of passes over the dataset: back propagation or exploration
            t = 0
            isTerminal = false
            maxEpisodeLengthTest = 5                                                  # solver.max_episode_length && !isTerminal
            while t<maxEpisodeLengthTest

                #### NO GOING OVER EPISODES: JUST UPDATES
                # DATASET
                sp, r = (row[statePrimeKey])[1], (row[rewardKey])[1] # observe results of action
                # generation alternative:
                #sp, r = generate_sr(mdp, s, a, rng); ap = action(exploration_policy, sp)

                # next time reward!!!
                rp = (nextRow[rewardKey])[1]           # from suitable exploration strategy
                ap = (nextRow[actionKey])[1]
                #print("s=", s, " a=", a, " r=", r, " sp=", sp, " ap=", ap, "\n")

                # INDEX: state and action
                si = s; ai = a;
                spi = sp; api = ap
                # si = getCarStateIndex(mdp, s); ai = getCarActionIndex(mdp, a);
                # spi = getCarStateIndex(mdp, sp); api = getCarActionIndex(mdp, ap)

                # DELTA: DAU 5.26
                #delta = r
                #try
                #    delta = delta + mdp.discount * Q[spi,api] - Q[si,ai]
                #catch
                #    delta = r
                #end

                ### UPDATE: LOCAL
                # VISIT COUNT
                ecounts[si,ai] += 1.0      # non-zero starting pseudocount
                #try
                #    ecounts[si,ai] += 1.0
                #catch e
                #    # @printf("CAUGHT EXCEPTION AT ECOUNTS %s", e) # , e
                #    ecounts[si,ai] = 1.0      # non-zero starting pseudocount
                #end

                ### Ro REWARD sum: DAU 5.2
                #Ro[si,ai] = 1.0      # non-zero starting pseudocount
                #try
                    Ro[si,ai] = Ro[si,ai] + r + rp          # TODO: hack to take into account next state
                #catch e
                    # @printf("CAUGHT EXCEPTION AT REWARD COUNT %s", e) # , e
                #    Ro[si,ai] = 1.0      # non-zero starting pseudocount
                #end

                TotalVisitCount = 0
                for aix in 1:length(gridMaxlikelihoodActions)
                    TotalVisitCount = TotalVisitCount + ecounts[si,aix]
                end

                ### THUS UPDATE qValue
                #qValue(s,ai) <- Ro(s,ai)*T(s'|s,ai)
                #where T(s'|s,ai) <- N(s,a,s')/N(s,a)
                Q[si,ai] = Ro[si,ai] * ecounts[si,ai]/TotalVisitCount

                # PROPAGATE: all s, all a
                #minPropagationThreshold = 10 # 1000
                #if abs(delta)>minPropagationThreshold        # something to propagate
                #    for es in 1:50000   # TODO: getNumCarStates(mdp)
                #        for ea in 1:getNumCarActions
                #            esi, eai = es, ea
                #            #esi, eai = getCarStateIndex(mdp, es), getCarActionIndex(mdp, ea)
                #            Q[esi,eai] += solver.learning_rate * delta * ecounts[esi,eai]
                #            ecounts[esi,eai] *= mdp.discount * solver.lambda                    # exponential decay
                #            if abs(Q[esi,eai])>0
                #                print("BACKPROPAGATING: ", "esi=", esi, " eai=", eai, " r=", r, " delta=", delta, " Q[esi,eai]=", Q[esi,eai], " ecounts[esi,eai]=", ecounts[esi,eai], " ", now(), "\n")
                #            end
                #        end
                #    end
                #end

                s, a = sp, ap

                t += 1

                # end

                if i % solver.eval_every == 0
                #    r_tot = 0.0
                #    for traj in 1:solver.n_eval_traj
                #        r_tot += simulate(sim, mdp, policy, initial_state(mdp, rng))
                #    end
                     @printf("Through episode %d with *cummunlative* returns: %d, current row %s...at %s\n", i, Ro[si,ai], row, now())
                #    println("Through episode $i, Returns: $(r_tot/solver.n_eval_traj)")
                end
            end

        catch e
            # @printf("CAUGHT EXCEPTION AT ROW %d", i) # , e
            @printf("CAUGHT EXCEPTION AT ROW %d: %s", i, e)
        end

    end

    @printf("RO TABLE TO SCREEN, WATCH OUT! %s", Ro)
    @printf("Ncounts TABLE TO SCREEN, WATCH OUT! %s", ecounts)
    @printf("qValue TABLE TO SCREEN, WATCH OUT! %s", Q)

    return Q                # Q[s,a]
    # return policy
end


### POLICY: get the best action for a given state, given a qTable

            # FOR POLICY:
            # R(s,a)=Ro(s,ai)/N(s,ai)  reward average probability
            # T(s,ai)=N(s,ai)/N(s,a) ai over all i
            ## Calculation
            # And Q(s,a) = R(s,a)/N
            #
            #Reward[si,ai] = Ro[si,ai]/ecounts[si,ai]]        # Average reward at (s,a)
            #
            #totalTransitions =0
            #for aix in 1:getNumCarActions
            #    totalTansitions = totalTransitions + ecounts[si,aix]
            #
            ## UPDATED NEIGHBORS: transitions
                # for aix in 1:getNumCarActions

            # TRANSITION COUNT: DAU 5.2
            #T[si,ai] = 1.0      # non-zero starting pseudocount
            #try
            #    T[si,ai] += 1.0
            #catch e
            #    # @printf("CAUGHT EXCEPTION AT ECOUNTS %s", e) # , e
            #    T[si,ai] = 1.0      # non-zero starting pseudocount
            #end
            #
            #TransitionProbability_si_ai = ecounts[si,ai]]/totalTransitions
            #Q[s,a] =


function getArgmaxPolicy(qTable::Matrix{Float64}, carStateId::Int64)
    argMaxAction=[]
    bestQvalue = -100000        # minus infinity, looking for better
    for actionId in 1:length(gridMaxlikelihoodActions)
        coordinateValue = qTable[carStateId, actionId]
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


### REFERENCES
#https://github.com/JuliaPOMDP/POMDPs.jl#reinforcement-learning
#Tabular Learning: Q-Learning, SARSA, SARSA lambda https://github.com/JuliaPOMDP/TabularTDLearning.jl
#https://github.com/JuliaPOMDP/POMDPs.jl#mdp-solvers
#http://juliapomdp.github.io/POMDPs.jl/stable/
#https://github.com/JuliaPOMDP/POMDPModels.jl Grid World, Tiger, Crying Baby, Random, Mountain Car, Inverted Pendulum, T-Maze




