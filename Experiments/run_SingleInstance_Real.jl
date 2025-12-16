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
prob = :coloncancer               # LogReg Problems - :coloncancer,:duke,:gisette,:leukemia,:madelon,
                            # LSReg Problems - :bodyfat,:eunite,:pyrim,:triazines,:YearPrediction
                            # LP Problems (LogSumExp or SquaredRelu) - :brazil3,:chromatic,:ex10,:graph40,:qap15,:rmine15,:savsched1,:scpm1,:setcover,:supportcase10

problemType = :LogReg       # :LogReg, :LSReg, :LogSumExp, :SquaredRelu

source = getProblemSource(prob)            # :LIBSVM, :LPFeas

# Determine computation budget in terms of oracle calls or run time (seconds). If both are nonzero, then method will run until BOTH conditions are satisfied
oracleCalls = 500
runTime = 0

# Save results to file
ResultsSaveFile = []

# List of first-order methods to test
methods = [ASPGM(5,5), ASPGM(1,1), BSPGM(7), LBFGS(10,:BackTracking), LBFGS(10,:HagerZhang), UFGM(), OBL(), AdaNAG(), ACFGM(), AdGD(), NAGF(), OSGMR(0.1)]

# ---------- End Settings ----------


# Parse files and build oracles
print("\nParsing Files...")

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
    x0 = randn(n)
    A = vcat(A, zeros(1, size(A,2)))
    b = vcat(b, 0.0)
    q = LogSumExpOracle(A, b)
elseif problemType == :SquaredRelu
    x0 = randn(n)
    q = SquaredReluOracle(A, b)
end

    
print("\nFinding True Solution...")
fStar, xStar, _ = getSolution(q, x0)


data = Array{Vector{Float64}}(undef, length(methods), 6)

for (j, met) in enumerate(methods)

    print("\n------",methodTitle(met),"------\n")

    # Convert to smart oracle
    q2 = SmartOracle(q)

    _, _, thisData = runMethod(met, q2, x0; oracleCalls = oracleCalls, runTime = runTime, saveDetails = true)
    
    data[j,1] = q2.vals
    if !isempty(thisData)
        k = size(thisData,2)
        for i=1:k
            data[j,i] = thisData[:,i]
        end
    else
        data[j,2] = zeros(size(q2.vals))
        data[j,3] = zeros(size(q2.vals))
        data[j,4] = zeros(size(q2.vals))
    end

end



plotDetails(data, methods, xStar, fStar, x0)


    
