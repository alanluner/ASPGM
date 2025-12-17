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

# Problems instances and their classes
probs, types, sources = problemList_LIBSVM()         # Ex. problemList_LIBSVM(), problemList_LP()

# Determine computation budget for each problem instance in terms of oracle calls or run time (seconds). If both are nonzero, then method will run until BOTH conditions are satisfied
oracleCalls = 500
runTime = 0

# Determine whether or not to calculate x_* and f_* for each problem instance
findTrueSolutions = true                  # If true, calculate x_* and f_*

# Save results to file
ResultsSaveFile = []

# List of first-order methods to test
methods = [ASPGM(5,5), ASPGM(1,1), BSPGM(7), LBFGS(10,:BackTracking), LBFGS(10,:HagerZhang), UFGM(), OBL(), AdaNAG(), ACFGM(), AdGD(), NAGF(), OSGMR(0.1)]

# ---------- End Settings ----------


methodsInit = copy(methods) # Create copy for saving to results file. This is necessary because .jld2 cannot handle the ASPGM object once Mosek is initialized.

# Parse files and build oracles
print("\nParsing Files...")
x0List = []
qList = []
for i in eachindex(probs)
    print(i)
    prob = probs[i]
    problemType = types[i]
    source = sources[i]

    if source == :LIBSVM
        file = "Experiments/LIBSVM/"*string(prob)*".txt"
        A, b = parse_LIBSVM_file(file)
    elseif source == :LPFeas
        file = "Experiments/LPFeas/"*string(prob)*".txt"
        A, b = parse_LP_file(file)
    end

    (m,n) = size(A)

    if problemType == :LogReg
        x0 = zeros(n)
        q = LogRegOracle(A, b, 1/m)
    elseif problemType == :LSReg
        x0 = zeros(n)
        q = LSRegOracle(A, b)
    elseif problemType == :LogSumExp
        x0 = zeros(n)
        A = vcat(A, zeros(1, size(A,2)))
        b = vcat(b, 0.0)
        q = LogSumExpOracle(A, b)
    elseif problemType == :SquaredRelu
        x0 = zeros(n)
        q = SquaredReluOracle(A, b)
    end

    push!(qList, q)
    push!(x0List, x0)
end

if findTrueSolutions         
    print("\nFinding True Solutions...")
    fStarData, xStarData = getSolutions(qList, x0List)
end

totalProblems = length(qList)

functionValData = Array{Vector{Float64}}(undef, length(methods), totalProblems)
timeData = Array{Vector{Float64}}(undef, length(methods), totalProblems)

print("\nRunning Methods...\n\n")
for (i, met) in enumerate(methods)

    print("\nMethod: ", methodTitle(met),"\n")
    for (j, q) in enumerate(qList)
        print(j)

        # Convert to smart oracle
        q2 = SmartOracle(q)

        x0 = x0List[j]

        vals, _ = runMethod(met, q2, x0; oracleCalls = oracleCalls, runTime = runTime)

        functionValData[i,j] = q2.vals
        timeData[i,j] = q2.times .- q2.times[1]

    end

end

if !isempty(ResultsSaveFile)
    if findTrueSolutions
        jldsave(ResultsSaveFile; functionValData, timeData, xStarData, fStarData, methodsInit, dimArray, oracleCalls, runTime, probs, types)
    else
        jldsave(ResultsSaveFile; functionValData, timeData, methodsInit, dimArray, oracleCalls, runTime, probs, types)
    end
end

totalProblems = size(functionValData,2)
targetRelAccuracies = [1e-4, 1e-7, 1e-10]

numberSolved, times = getSummaryData(functionValData, timeData, fStarData, targetRelAccuracies, oracleCalls)

p = plotOracleAndTime(numberSolved, times, targetRelAccuracies, methods; file=[], colors=[])





