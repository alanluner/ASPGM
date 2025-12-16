include("example_oracles.jl")
include("TestingUtilities.jl")
include("../ASPGM.jl")
include("../AlternateMethods/ACFGM.jl")
include("../AlternateMethods/AdaNAG.jl")
include("../AlternateMethods/AdGD.jl")
include("../AlternateMethods/LBFGS.jl")
include("../AlternateMethods/NAGF.jl")
include("../AlternateMethods/OBL.jl")
include("../AlternateMethods/OSGMR.jl")
include("../AlternateMethods/UFGM.jl")
include("plot_Performance.jl")


using Random



# ---------- Settings ----------

# Problem classes
problemTypes = defaultProblemTypes()        # :LogSumExp,:LSReg,:LogReg,:SquaredRelu,:QuarticReg,:CubicReg

# Problem dimensions: x ∈ ℝ^d, A ∈ ℝ^{4d x d}
dimArray = [1000, 2000, 4000]


# Random seed for reproducibility
randSeedNum = 23
randSeed = Xoshiro(randSeedNum)

# Determine computation budget for each problem instance in terms of oracle calls or run time (seconds). If both are nonzero, then method will run until BOTH conditions are satisfied
maxIter = 250
runTime = 0.1

# Number of copies of each oracle setup to use - this produces numCopies*4 oracles per dimension and problemType
numCopies = 2   

# Determine whether or not to calculate x_* and f_* for each problem instance
findTrueSolutions = true                  

# If nonempty, save results to file
ResultsSaveFile = []

# List of first-order methods to test
methods = [ASPGM(5,5), ASPGM(1,1), BSPGM(7), LBFGS(10,:BackTracking), LBFGS(10,:HagerZhang), UFGM(), OBL(), AdaNAG(), ACFGM(), AdGD(), NAGF(), OSGMR(0.1)]

# ---------- End Settings ----------


methodsInit = copy(methods) # Create copy for saving to results file. This is necessary because .jld2 cannot handle the ASPGM object once Mosek is initialized.

numDims = size(dimArray,1)
numProblemTypes = length(problemTypes)
totalProblems = numDims*numProblemTypes*numCopies*4

print("\nTotal Problems: ", totalProblems)

functionValData = Array{Vector{Float64}}(undef, length(methods), totalProblems)
nullStepData = Array{Vector{Int64}}(undef, length(methods), totalProblems)
timeData = Array{Vector{Float64}}(undef, length(methods), totalProblems)

xStarData = Array{Vector{Float64}}(undef, totalProblems)
fStarData = Vector{Float64}(undef, totalProblems)


global idxCtr = 0
for (dIdx,d) in enumerate(dimArray)
    print("\n\n------------Dimension: ",d,"-----------\n")

    for (probIdx, problemType) in enumerate(problemTypes)
        print("\n---Problem Type: ", problemType,"---\n") 

        # Generate oracles - performed in batches to prevent memory issues for large dimensions
        print("\n\tGenerating Oracles...")
        qList, x0List = generateOracles(randSeed, [problemType], [d]; numCopies = numCopies)

        M = length(qList)
        indices = idxCtr+1 : idxCtr+M

        # Calculate true solutions x_* and f_*
        if findTrueSolutions
            print("\n\tCalculating Solutions...")
            fStarList, xStarList = getSolutions(qList, x0List)
            fStarData[indices] .= fStarList
            xStarData[indices] .= xStarList
        end

        # Run each method
        print("\n\tRunning Methods...")
        for (i, met) in enumerate(methods)
            print("\n\tMethod: ", methodTitle(met),"\n\t")
            
            # Loop through problem instances in this batch
            for j = 1:M
                q = qList[j]

                # Convert to smart oracle
                q2 = SmartOracle(q)

                x0 = x0List[j]

                vals, _ = runMethod(met, q2, x0; oracleCalls = oracleCalls, runTime = runTime)

                functionValData[i,indices[j]] = q2.vals
                timeData[i,indices[j]] = q2.times .- q2.times[1]
            end
        end

        global idxCtr
        idxCtr += M

        if findTrueSolutions
            jldsave(ResultsSaveFile; functionValData, timeData, xStarData, fStarData, methodsInit, dimArray, oracleCalls, runTime, problemTypes, randSeed, randSeedNum)
        else
            jldsave(ResultsSaveFile; functionValData, timeData, xStarData, fStarData, methodsInit, dimArray, oracleCalls, runTime, problemTypes, randSeed, randSeedNum)
        end

    end
end

totalProblems = size(functionValData,2)
targetRelAccuracies = [1e-4, 1e-7, 1e-10]

numberSolved, times = getSummaryData(functionValData, timeData, fStarData, targetRelAccuracies, maxIter)

p = plotOracleAndTime(numberSolved, times, targetRelAccuracies, methods; file=[], colors=[])