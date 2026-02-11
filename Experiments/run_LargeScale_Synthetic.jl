include("example_oracles.jl")
include("TestingUtilities.jl")
include("../ASPGM.jl")
include("../ASPGM11.jl")
include("../AlternateMethods/ACFGM.jl")
include("../AlternateMethods/AdaNAG.jl")
include("../AlternateMethods/AdGD.jl")
include("../AlternateMethods/LBFGS.jl")
include("../AlternateMethods/NAGF.jl")
include("../AlternateMethods/OBL.jl")
include("../AlternateMethods/OSGMB.jl")
include("../AlternateMethods/UFGM.jl")
include("plot_Performance.jl")


using Logging, LineSearches, Random, LinearAlgebra

BLAS.set_num_threads(1)

# ---------- Settings ----------

# Problem classes
problemTypes = defaultProblemTypes()        # :LogSumExp,:LSReg,:LogReg,:SquaredRelu,:QuarticReg,:CubicReg

# Problem dimensions: x ∈ ℝ^d, A ∈ ℝ^{4d x d}
dimArray = [1000, 2000, 4000, 8000]


# Random seed for reproducibility
randSeedNum = 41
randSeed = Xoshiro(randSeedNum)

# Determine computation budget for each problem instance in terms of oracle calls or run time (seconds). If both are nonzero, then method will run until BOTH conditions are satisfied
oracleCalls = 500
runTime = 60

# Number of copies of each oracle setup to use - this produces numCopies*4 oracles per dimension and problemType
numCopies = 2   

# Determine whether or not to calculate x_* and f_* for each problem instance
findTrueSolutions = true                  

# If nonempty, save results to file
ResultsSaveFile = "Results_Synthetic.jld2"

# List of first-order methods to test
methods = [ASPGM(5,5), ASPGM11(), BSPGM(7), LBFGS(10,:BackTracking), LBFGS(10,:HagerZhang), UFGM(), OBL(), AdaNAG(), ACFGM(), AdGD(), NAGF(), OSGMB()]

# ---------- End Settings ----------


methodsInit = methodTitle.(methods) # Store list of methods for saving to results file

numDims = size(dimArray,1)
numProblemTypes = length(problemTypes)
totalProblems = numDims*numProblemTypes*numCopies*4

print("\nTotal Problems: ", totalProblems)

functionValData = Array{Vector{Float64}}(undef, length(methods), totalProblems)
nullStepData = Array{Vector{Int64}}(undef, length(methods), totalProblems)
timeData = Array{Vector{Float64}}(undef, length(methods), totalProblems)

xStarData = Array{Vector{Float64}}(undef, totalProblems)
fStarData = Vector{Float64}(undef, totalProblems)

# --- Warm-up problem to compile each method ---
print("\nWarming up methods...")
A_tmp = rand(10, 5)
b_tmp = rand(10)
q_tmp = SmartOracle(LSRegOracle(A_tmp, b_tmp))
x0_tmp = zeros(5)

for met in methods
    # Run briefly just to trigger compilation
    runMethod(met, q_tmp, x0_tmp; oracleCalls = 5, runTime = 0)
end
print("\nWarm-up complete.\n")
# --- End warm-up ---

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

                GC.gc() # garbage collection to reset

                # Run method
                vals, _ = runMethod(met, q2, x0; oracleCalls = oracleCalls, runTime = runTime)

                # Save function value data and time data
                functionValData[i,indices[j]] = q2.vals
                timeData[i,indices[j]] = q2.times .- q2.times[1]
            end
        end

        global idxCtr
        idxCtr += M

        # Save off results
        if !isempty(ResultsSaveFile)
            if findTrueSolutions
                jldsave(ResultsSaveFile; functionValData, timeData, xStarData, fStarData, methodsInit, dimArray, oracleCalls, runTime, problemTypes, randSeed, randSeedNum)
            else
                jldsave(ResultsSaveFile; functionValData, timeData, xStarData, fStarData, methodsInit, dimArray, oracleCalls, runTime, problemTypes, randSeed, randSeedNum)
            end
        end

    end
end

totalProblems = size(functionValData,2)
targetRelAccuracies = [1e-4, 1e-7, 1e-10]

# Process data for plotting
numberSolved, times = getSummaryData(functionValData, timeData, fStarData, targetRelAccuracies, oracleCalls)

# Plot results
p = plotOracleAndTime(numberSolved, times, targetRelAccuracies, methods; file=[], endTime = 60)