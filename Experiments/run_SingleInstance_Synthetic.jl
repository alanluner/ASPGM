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

# Problem class
problemType = :LogSumExp       # :LSReg, :LogReg, :LogSumExp :SquaredRelu, :CubicReg, :QuarticReg      or specific hard problem instances :HardA, :HardB, :HardC

# Problem dimensions: x ∈ ℝ^d, A ∈ ℝ^{mxd}
m = 2000 
d = 500

# Random seed for reproducibility
rng = Xoshiro(64)

# Determine distribution used to generate random matrices for oracle
randMode = 2            # 1, 2, 3, 4 - See TestingUtilities for descriptions

# Determine computation budget in terms of oracle calls or run time (seconds). If both are nonzero, then method will run until BOTH conditions are satisfied
oracleCalls = 500
runTime = 0

# List of first-order methods to test
methods = [ASPGM(5,5), ASPGM(1,1), BSPGM(7), LBFGS(10,:BackTracking), LBFGS(10,:HagerZhang), UFGM(), OBL(), AdaNAG(), ACFGM(), AdGD(), NAGF(), OSGMR(0.1)]

# ---------- End Settings ----------


# Build oracle for random problem instance, or specific hard instance
q, x0 = buildOracle(problemType, rng, randMode, m, d)


# Find true minimum
# If using hard instance, minimum/minimizer have explicit solutions, otherwise use a solver
if problemType == :HardA
    xStar = 2*collect(length(x0):-1:1) / (length(x0) + 1)
    fStar, _ = q(xStar)
elseif problemType == :HardB
    xStar = zeros(size(x0))
    fStar = 0.0
elseif problemType == :HardC
    xStar = 1 ./ (1:length(x0))
    fStar = 0.0
else
    fStar, xStar, _ = getSolution(q, x0)
end


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


    
