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

# Problems instances and their classes
dir = "Experiments/LIBSVM/LSReg"
problemType = :LSReg        # :LogReg, :LSReg, :LogSumExp, :SquaredRelu
fileType = :LIBSVM          # :LIBSVM, :LPFeas


# Determine computation budget for each problem instance in terms of oracle calls or run time (seconds). If both are nonzero, then method will run until BOTH conditions are satisfied
oracleCalls = 500
runTime = 60     

# Determine whether or not to calculate x_* and f_* for each problem instance
findTrueSolutions = true          

# Save results to file
ResultsSaveFile = "Results_Real.jld2"

# List of first-order methods to test
methods = [ASPGM(5,5), ASPGM11(), BSPGM(7), LBFGS(10,:BackTracking), LBFGS(10,:HagerZhang), UFGM(), OBL(), AdaNAG(), ACFGM(), AdGD(), NAGF(), OSGMB()]

# ---------- End Settings ----------


methodList = methodTitle.(methods) # Create copy for saving to results file. This is necessary because .jld2 cannot handle the ASPGM object once Mosek is initialized.

# Parse files and build oracles
print("\nParsing Files...")
x0List = []
qList = []
i = 0
for file in readdir(dir)
    global i += 1
    print(i)
    filepath = joinpath(dir, file)

    if fileType == :LIBSVM
        A, b = parse_LIBSVM_file(filepath)
    elseif fileType == :LP
        A, b = parse_LP_file(filepath)
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

# --- Find true solutions ---
if findTrueSolutions         
    print("\nFinding True Solutions...")
    fStarData, xStarData = getSolutions(qList, x0List)
end

# --- Warm-up problem to compile each method ---
print("\nWarming up methods...")
A_tmp = rand(10, 5)
b_tmp = rand(10)
q_tmp = SmartOracle(LSRegOracle(A_tmp, b_tmp))
x0_tmp = zeros(5)

for met in methods
    runMethod(met, q_tmp, x0_tmp; oracleCalls = 5, runTime = 0)
end
print("\nWarm-up complete.\n")
# --- End warm-up ---

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

        GC.gc() # garbage collection to reset

        # Run method
        vals, _ = runMethod(met, q2, x0; oracleCalls = oracleCalls, runTime = runTime)

        # Save function value data and time data
        functionValData[i,j] = q2.vals
        timeData[i,j] = q2.times .- q2.times[1]

    end

end

# --- Save off results --- #
if !isempty(ResultsSaveFile)
    if findTrueSolutions
        jldsave(ResultsSaveFile; functionValData, timeData, xStarData, fStarData, methodList, oracleCalls, runTime, probs, types)
    else
        jldsave(ResultsSaveFile; functionValData, timeData, methodList, oracleCalls, runTime, probs, types)
    end
end

totalProblems = size(functionValData,2)
targetRelAccuracies = [1e-4, 1e-7, 1e-10]

# Process data for plotting
numberSolved, times = getSummaryData(functionValData, timeData, fStarData, targetRelAccuracies, oracleCalls)

# Plot results
p = plotOracleAndTime(numberSolved, times, targetRelAccuracies, methods; file=[], endTime = 60)





