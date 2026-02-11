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

using Random

# ---------- Settings ----------

# Problems instances and their classes
file = "Experiments/LIBSVM/LogReg/duke.txt"
problemType = :LogReg       # :LogReg, :LSReg, :LogSumExp, :SquaredRelu
fileType = :LIBSVM          # :LIBSVM, :LPFeas

# Determine computation budget in terms of oracle calls or run time (seconds). If both are nonzero, then method will run until BOTH conditions are satisfied
oracleCalls = 250
runTime = 0

# List of first-order methods to test
methods = [ASPGM(5,5), ASPGM11(), BSPGM(7), LBFGS(10,:BackTracking), LBFGS(10,:HagerZhang), UFGM(), OBL(), AdaNAG(), ACFGM(), AdGD(), NAGF(), OSGMB()]

# ---------- End Settings ----------


# Parse files and build oracles
print("\nParsing Files...")

if fileType == :LIBSVM
    A, b = parse_LIBSVM_file(file)
elseif fileType == :LPFeas
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

# --- Find true solution ---
print("\nFinding True Solution...")
fStar, xStar, _ = getSolution(q, x0)

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


# additional data - format: [f, ||g||, guarantee(method), Delta, tau, L]
addlData = Array{Vector{Float64}}(undef, length(methods), 6)

functionValData = Array{Vector{Float64}}(undef, length(methods))
timeData = Array{Vector{Float64}}(undef, length(methods))

for (j, met) in enumerate(methods)

    print("\n------",methodTitle(met),"------\n")

    # Convert to smart oracle
    q2 = SmartOracle(q)

    GC.gc() # garbage collection to reset

    _, _, thisData = runMethod(met, q2, x0; oracleCalls = oracleCalls, runTime = runTime, saveDetails = true)
    
    addlData[j,1] = q2.vals
    if !isempty(thisData)
        k = size(thisData,2)
        for i=1:k
            addlData[j,i] = thisData[:,i]
        end
    else
        addlData[j,2] = zeros(size(q2.vals))
        addlData[j,3] = zeros(size(q2.vals))
        addlData[j,4] = zeros(size(q2.vals))
    end

    functionValData[j] = q2.vals
    timeData[j] = q2.times

end


plotSingleInstance(functionValData, timeData, methods, fStar; endTime = 5, oracleCalls = oracleCalls)

# plotDetails(addlData, methods, xStar, fStar, x0)


    
