using JuMP, Mosek, MosekTools, LinearAlgebra, Parameters
import MathOptInterface as MOI

#========================================================================
This code implements the Adaptive Subgame Perfect Gradient Method (ASPGM). At each iteration, ASPGM makes a momentum-type update,
optimized dynamically based on a (limited) memory/bundle of past first-order information. It is linesearch-free, parameter-free, and adaptive.

See https://github.com/alanluner/ASPGM for implementation details.

See https://arxiv.org/pdf/2510.21617 for the paper that introduces the method.
========================================================================#
#========================================================================
Notes:
- This implementation is written to be highly performant for large-scale problems. To accomplish this, we repeatedly use pre-allocated vectors, in-place vector calculation,
views, and more. Any future updates should be careful in their vector handling to maintain this behavior. In particular, any changes to the pre-allocated vectors must be done very carefully.
========================================================================#

# Options for ASPGM
@with_kw mutable struct options_ASPGM
    mode::Symbol = :A                           # :A - ASPGM (with restarting and preconditioning)   [ vs :B - BSPGM (no restarting or preconditioning) ]
    memorySize_SP::Int64 = 5                    # Memory size for subgame perfect memory
    memorySize_QN::Int64 = 5                    # Memory size for quasi-newton memory
    restartFactor::Float64 = 2.0                # (:A mode only) Value > 1 that determines restart condition on τ_n. Larger value results in less frequent restarts (Default: 2)
    minEpochSize::Int64 = 20                    # (:A mode only) Minimum number of iterations in an epoch before allowing a restart. Set to 0 to ignore (Default: 20)
    maxEpochSize::Union{Int64,Float64} = 100    # (:A mode only) Maximum number of iterations in epoch before forcing a restart. Set to Inf to ignore (Default: 100)
    g_tol::Float64 = -1.0                       # Terminate early when ||g_n||_inf <= g_tol. Set to -1 to ignore
    guarantee_tol::Float64 = -1.0               # (:B mode only) Terminate early when convergence guarantee (L_n/τ_n) is less than guarantee_tol. Set to -1 to ignore
end

# Flags used throughout the algorithm for branching behavior
mutable struct flags
    flag_NullStep::Bool           # Flag that previous step was null step (i.e., τ_{n-1} = 0)
    flag_FinalStep::Bool          # Set to true once phi exceeds restart value. Then update will use last-step correction until next serious step, triggering restart
    flag_DisablePrec::Bool        # Flag to disable preconditioning going forward. This is sometimes necessary to avoid numerical errors in preconditioning when very close to the solution.
    flag_Terminate::Bool          # Flag to terminate algorithm once desired tolerance is met (either via g_tol or guarantee_tol)
end

flags() = flags(false, false, false, false)


mutable struct ASPGM
    # ------------- Initial values ------------- #
    x0::Vector{Float64}                     # Starting point for this epoch
    f0::Float64                             # Stores initial f(x0) for this epoch
    # ------------- Current iterate data ------------- #
    x::Vector{Float64}                      # Iterate x_n
    value::Float64                          # Function value f_n
    gradient::Vector{Float64}               # Gradient g_n
    L::Float64                              # Smoothness estimate
    mu::Float64                             # Strong convexity estimate
    newDelta::Float64                       # Error parameter Δ_n
    newTau::Float64                         # Convergence guarantee τ_n
    newL::Float64                           # Smoothness estimate for next iterate L_{n+1}
    # ------------- Pre-Allocation ------------- #
    scratchVec1::Vector{Float64}            # Pre-allocated vector for temporary calculations. Should NOT be used as input or output to any functions in this routine (native functions are ok)
    scratchVec2::Vector{Float64}            # Pre-allocated vector for temporary calculations. Should NOT be used as input or output to any functions in this routine (native functions are ok)
    inputVec::Vector{Float64}               # Pre-allocated vector to be used as input argument for functions in this routine. Should be used as argument immediately after its value is set.
    outputVec::Vector{Float64}              # Pre-allocated vector to be used as output argument (modified in-place) for functions in this routine. Its value should be used immediately after receiving output.
    # ------------- Index Tracking ------------- #
    startIdx::Int64                         # First index of active memory
    nextIdx::Int64                          # Index that next iterate will be stored in
    # ------------- Memory Storage - Subgame Perfect ------------- #
    Z::Matrix{Float64}                      # Past auxiliary sequence data z_{n-k} to z_{n-1}. We will use a different scaling than in the paper [ L_i(z_{i+1} - x_0) rather than L_i/L_n(z_{i+1} - x_0) ].
    G::Matrix{Float64}                      # Past gradient data g_{n-k} to g_{n-1}. We will use a different scaling than in the paper [ g_i rather than g_i/L_n ].
    X::Matrix{Float64}                      # Past iterate data x_{n-k} to x_{n-1}
    F::Vector{Float64}                      # Past function value data f_{n-k} to f_{n-1}
    ZG_MAT::Matrix{Float64}                 # Efficient storage matrix for [Z'Z  -Z'G; -G'Z  G'G] for SOCP subproblem
    zprime::Vector{Float64}                 # Optimal auxiliary step z' as determined by subproblem
    psi_prev::Float64                       # ψ_{n-1} = τ_{n-1} - τ'
    taus::Vector{Float64}                   # (τ_{n-k}, ..., τ_{n-1})
    h::Vector{Float64}                      # (h_{n-k}, ..., h_{n-1})
    w::Vector{Float64}                      # (w_{n-k}, ..., w_{n-1})
    LPrev::Vector{Float64}                  # (L_{n-k}, ..., L_{n-1})
    Deltas::Vector{Float64}                 # (Δ_{n-k}, ..., Δ_{n-1})
    # ------------- Preconditioning Memory Storage ------------- #
    S::Matrix{Float64}                      # Quasi-Newton matrix formed from iterate differences: (x_{n-t+1}-x_{n-t}, ..., x_{n}-x_{n-1})  
    Y::Matrix{Float64}                      # Quasi-Newton matrix formed from gradient differences: (g_{n-t+1}-g_{n-t}, ..., x_{n}-x_{n-1})  
    PrecLU::Union{Nothing, LU{Float64, Matrix{Float64}, Vector{Int64}}}       # LU decomposition of 2t x 2t matrix used for preconditioning (see ApplyHessianApprox), it is saved off to remove redundant calculations
    theta::Float64                          # Scalar value used for precondntioning (see ApplyHessianApprox), it is saved off to remove redundant calculations
    # ------------- Metadata ------------- #
    iteration::Int64                        # total iteration number
    epochIter::Int64                        # iteration number within this epoch
    numEpochs::Int64                        # number of epochs
    oracleCtr::Int64                        # number of oracle calls
    model::Union{Nothing, Model}            # Mosek optimization model
    flags::flags              # Set of flags for branching behavior according to the algorithm
    # ------------- Implementation Options ------------- #
    options::options_ASPGM            # (See below)
end


# --------Method Constructors-------- #

ASPGM(k, t) = ASPGM(Float64[],0.0,
                            Float64[],0.0,Float64[],0.0,0.0,0.0,0.0,0.0,
                            Float64[],Float64[],Float64[],Float64[],
                            0,0,
                            Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),Float64[],Array{Float64}(undef, 0, 0),Float64[],0.0,Float64[],Float64[],Float64[],Float64[],Float64[],
                            Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),nothing,0.0,
                            0,0,0,0,nothing,flags(),
                            options_ASPGM(:A, k, t, 2.0, 20, 100, -1.0, -1.0))

ASPGM() = ASPGM(5,5)

ASPGM(opts) = ASPGM(Float64[],0.0,
                            Float64[],0.0,Float64[],0.0,0.0,0.0,0.0,0.0,
                            Float64[],Float64[],Float64[],Float64[],
                            0,0,
                            Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),Float64[],Array{Float64}(undef, 0, 0),Float64[],0.0,Float64[],Float64[],Float64[],Float64[],Float64[],
                            Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),nothing,0.0,
                            0,0,0,0,nothing,flags(),
                            opts)

BSPGM(k) = ASPGM(Float64[],0.0,
                            Float64[],0.0,Float64[],0.0,0.0,0.0,0.0,0.0,
                            Float64[],Float64[],Float64[],Float64[],
                            0,0,
                            Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),Float64[],Array{Float64}(undef, 0, 0),Float64[],0.0,Float64[],Float64[],Float64[],Float64[],Float64[],
                            Array{Float64}(undef, 0, 0),Array{Float64}(undef, 0, 0),nothing,0.0,
                            0,0,0,0,nothing,flags(),
                            options_ASPGM(:B, k, 0, -1.0, -1, Inf, -1.0, -1.0))

BSPGM() = BSPGM(5)

# ----------------------------------- #

# Run ASPGM for a certain amount of time/oracle calls

# Arguments:
# - method: ASPGM object 
# - oracle: first-order oracle that returns function and gradient information: f,g = oracle(x)
# - x0: x0
# - oracleCalls/runTime: computation budget in terms of oracle calls or run time (seconds). If both are nonzero, then method will run until BOTH conditions are satisfied
# - saveDetails: determines if additional problem data (gradient norm, L, tau, etc.) are saved off. Note that function value data will already be obtained from the SmartOracle logging. Not recommended for large-scale testing.
#
# Returns: 
# - x: ''best'' iterate - iterate (in memory) with the lowest function value at stopping time
# - val: function value of x
# - metaData: additional data from each oracle call - gradient norm, guarantee, Δ, τ, L
#
# Example:
# function oracle(x)
#   return 1/2*norm(x)^2, x
#
# runMethod(ASPGM(5,5), oracle, zeros(d); oracleCalls = 500)
#
@views function runMethod(method::ASPGM, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)
    
    initialize(method, oracle, x0)

    exit = false
    metaData = Vector{Float64}[]
    t0 = time()

    if saveDetails
        push!(metaData, [method.value, norm(method.gradient), guarantee(method), method.newDelta, method.newTau, method.L])
    end

    while !exit
        ctr1 = method.oracleCtr
        update(method, oracle)
        ctr2 = method.oracleCtr

        # If we did multiple oracle calls during this iteration (usually in the case of a restart), then save off data multiple times - once for each oracle call
        if saveDetails
            currentData = [method.value, norm(method.gradient), guarantee(method), method.newDelta, method.newTau, method.L]
            for i=1:ctr2-ctr1
                push!(metaData, currentData)
            end
        end

        t = time() - t0

        # Check for terminate flag, or if sufficient time/oracleCalls
        if method.flags.flag_Terminate
            exit = true
        elseif (method.oracleCtr > oracleCalls)&&(t >= runTime)
            exit = true
        end

    end

    # Return x with best function value
    Fval, best_idx = findmin(method.F)
    if method.value < Fval
        # If current iterate is best, return current iterate
        val = method.value
        x = method.x + method.x0
    else
        # If past iterate in memory is best, return that iterate
        val = Fval
        x = method.X[:,best_idx] + method.x0
    end

    # Convert from list of vectors to matrix
    if !isempty(metaData)
        metaData = reduce(vcat, transpose.(metaData))
    end

    return x, val, metaData
end


# --- Initialize variables of the method and validate settings ---
function initialize(method::ASPGM, oracle, x0::Vector{Float64})

    # Validate settings
    validate(method)

    d = length(x0)
    k = method.options.memorySize_SP
    t = method.options.memorySize_QN

    # Save initial values
    method.x0 = copy(x0)
    method.gradient = zeros(d)
    method.value = oracle(method.gradient, method.x0)
    method.f0 = method.value
    method.x = zeros(d)      # We implement a shift so that we can assume that x0 = 0

    # Allocate memory for scratch vectors
    method.scratchVec1 = zeros(d)
    method.scratchVec2 = zeros(d)
    method.inputVec = zeros(d)
    method.outputVec = zeros(d)

    # Set tracker indices
    method.startIdx = 0     # Initially set startIdx to 0, since no indices are active
    method.nextIdx = 1

    # Initialize various storage vectors and matrices
    method.Z = zeros(d, k)
    method.G = zeros(d, k)
    method.X = zeros(d, k)
    method.F = Inf*ones(k)
    method.ZG_MAT = zeros(2k, 2k)
    method.zprime = zeros(d)
    method.psi_prev = 1.0    
    method.taus = zeros(k)   
    method.h = zeros(k)
    method.w = zeros(k)
    method.LPrev = zeros(k)
    method.Deltas = zeros(k)

    method.newTau = 1.0     # Set τ_0 = 1
    method.newDelta = 0.0   # Set Δ_0 = 0

    # Initialize quasi-Newton matrices
    method.S = zeros(d,t)
    method.Y = zeros(d,t)
    method.PrecLU = nothing
    method.theta = 1.0

    # Iteration counters
    method.iteration = 0
    method.epochIter = 0
    method.numEpochs = 1
    method.oracleCtr = 1 # Count initial oracle call on x0

    # Instantiate a single Mosek instance for solving our subproblem at each step
    # Clearing it out and reusing at each step is faster than building a new one every iteration.
    if method.options.memorySize_SP > 1
        method.model = Model(Mosek.Optimizer)
        set_optimizer_attribute(method.model, "MSK_DPAR_SEMIDEFINITE_TOL_APPROX", 1e-12)
    end

    method.flags = flags()

    # Initialize strong convexity estimate μ
    method.mu = Inf     

    # Initialize smoothness estimate L0
    LInit = getLInit(method, oracle)
    method.L = LInit
    method.newL = LInit

end

# --- Validate settings ---
function validate(method::ASPGM)
    opts = method.options
    if (opts.mode==:A)
        if opts.memorySize_QN > opts.memorySize_SP
            @warn "Quasi-Newton memory (memorySize_QN) cannot be larger than subgame perfect memory (memorySize_SP). Reducing memorySize_QN to " * string(opts.memorySize_SP)
            opts.memorySize_QN = opts.memorySize_SP
        end
        
        if opts.restartFactor <= 1
            @warn "Invalid value for restartFactor. Must be larger than 1. Updating value to default: 2"
            opts.restartFactor = 2.0
        end
        
        if opts.maxEpochSize <= opts.minEpochSize
            @warn "Invalid values for minEpochSize and maxEpochSize. Setting to default values: minEpochSize = 20, maxEpochSize = 100"
            opts.minEpochSize = 20
            opts.maxEpochSize = 100
        end

        if opts.minEpochSize < opts.memorySize_QN
            @warn "Minimum epoch size cannot be less than Quasi-Newton memory (memorySize_QN). Increasing minEpochSize to " * string(opts.memorySize_QN)
            opts.minEpochSize = opts.memorySize_QN
        end

        if (opts.guarantee_tol > 0)
            @warn "Note that for mode :A, the setting guarantee_tol is ignored."
        end


    elseif (opts.mode == :B)
        if (opts.minEpochSize > 0) || (opts.maxEpochSize < Inf) || (opts.restartFactor > 0)
            @warn "Note that for mode :B, the settings minEpochSize, maxEpochSize, and restartFactor are ignored."
        end
    end

end


# --- Perform single iteration of ASPGM/BSPGM ---
function update(method::ASPGM, oracle)

    opts = method.options
    flags = method.flags
    k = opts.memorySize_SP

    # Sanity check for bad behavior (NaNs, Infs) in gradients
    if any(!isfinite, method.gradient)
        if isnothing(method.PrecLU)
            # NaNs/Infs can arise from numerical error in preconditioning when close to the solution
            # If that is the case, disable preconditioning going forward and restart
            flags.flag_DisablePrec = true
            
            # If our current iterate x has bad behavior, return to previous iterate
            if any(!isfinite, method.x)
                idx = getActiveIndices(method)[end]
                method.x .= method.X[:,idx]
            end

            restart(method, oracle)
        else
            # If NaNs/Infs came from standard gradient calculation, terminate because something is wrong
            @warn "NaNs/Infinite values encountered in gradient calculation. Terminating at iter "*string(method.iteration)*"."
            flags.flag_Terminate = true
        end
        
        return
    end

    # If τ is full, τ_{n-k-1} (taus[startIdx]) is the only nonzero component of τ (i.e. idx_s = startIdx), and this was a null step (i.e. τ_n = 0), then:
        # If k>1, we replace at index (startIdx+1) (corresponding to τ_{n-k}) instead of startIdx
        # Else if k==1, we ignore the new data
    # This ensures that the nonzero τ_{n-k-1} will be preserved. See Remark 1 in the paper.
    pushNewData = true
    idx_s = getLastSuccessIdx(method)
    isMemFull = (method.startIdx == method.nextIdx)
    if flags.flag_NullStep && isMemFull && (idx_s == method.startIdx)
        if (k > 1)
            # Replace at index after startIdx
            replaceIdx = mod1(method.startIdx+1, k)
            shiftData(method, method.startIdx, replaceIdx)
            pushNewData = true
        else
            pushNewData = false
        end
    end

    # --- Update variables with new data ---
    if pushNewData
        # Update τ with τ_{n-1}
        push_data_vec!(method.taus, method.newTau, method.nextIdx)

        # Update Δ with Δ_{n-1}
        push_data_vec!(method.Deltas, method.newDelta, method.nextIdx)

        # Update LPrev with L_{n-1}
        push_data_vec!(method.LPrev, method.L, method.nextIdx) # Note: At this point, method.L stores L_{n-1}

        # Update G with g_{n-1}
        push_data_mat!(method.G, method.gradient, method.nextIdx)

        # Update Z with L_{n-1}*z_n
        if flags.flag_NullStep
            # If last step was null step, set z_n = x0 (but with the same shift applied, so z_n = 0)
            method.inputVec .= 0
            push_data_mat!(method.Z, method.inputVec, method.nextIdx) 
        else
            # Otherwise, use standard update zL = L_{n-1}*z_n = L_{n-1}*z' - ψ_{n-1}*g_{n-1}
            @. method.inputVec = method.L*method.zprime - method.psi_prev*method.gradient
            push_data_mat!(method.Z, method.inputVec, method.nextIdx)     
        end

        # -----Calculate new Z/G dot products-----
        # Calculate ⟨Z, zL⟩_B, ⟨G, zL⟩_B by first calculating v = B^{-1}'*zL, then Z'*v, G'*v
        ApplyHessianApprox!(method.outputVec, method.inputVec, method)  # outputVec = B^{-1}'*zL
        vec_zi_z = method.Z'*method.outputVec   #⟨Z, zL⟩_B
        vec_gi_z = method.G'*method.outputVec   #⟨G, zL⟩_B
        # Then calculate ⟨Z, g⟩_B, ⟨G, g⟩_B by first calculating v = B^{-1}'*g, then Z'*v, G'*v
        ApplyHessianApprox!(method.outputVec, method.gradient, method)  # outputVec = B^{-1}'*g
        vec_zi_g = method.Z'*method.outputVec   #⟨Z, g⟩_B
        vec_gi_g = method.G'*method.outputVec   #⟨G, g⟩_B

        # Update [Z'Z  -Z'G; -G'Z  G'G] matrix
        push_data_ZG_MAT!(method.ZG_MAT, vec_zi_z, vec_zi_g, vec_gi_z, vec_gi_g, method.nextIdx)

        # Update w with f_{n-1} - ⟨g_{n-1}, x_{n-1}⟩
        push_data_vec!(method.w, method.value - dot_B(method.gradient, method.x, method), method.nextIdx)

        # Update h with tau_{n-1}*(f_{n-1} - 1/(2L_{n-1})*||g_{n-1}||^2 + L_{n-1}/2*||z_n||^2)
        # We already calculated these norms, so we can pull from the vectors above: ||g_{n-1}||^2 = vec_gi_g[nextIdx] and ||z_n||^2 = vec_zi_z[nextIdx]/L{n-1}^2
        push_data_vec!(method.h, method.taus[method.nextIdx]*(method.value - 1/(2*method.L)*vec_gi_g[method.nextIdx]) + 1/(2*method.L)*vec_zi_z[method.nextIdx], method.nextIdx)

        # Update F with f_{n-1}
        push_data_vec!(method.F, method.value, method.nextIdx)

        # Update X with x_{n-1}
        push_data_mat!(method.X, method.x, method.nextIdx)


        # Update index counters
        if (method.startIdx == method.nextIdx) || (method.startIdx == 0)
            # If we were previously at full memory, increment both indices (also used if this is the first iteration)
            method.startIdx = mod1(method.startIdx + 1, k)
            method.nextIdx = mod1(method.nextIdx + 1, k)
        else
            # Otherwise, only increment nextIdx
            method.nextIdx = mod1(method.nextIdx + 1, k)
        end
    end

    # Now update method.L to use L_n for generating the next step
    method.L = method.newL

    # --- Generate next iterate --- 
    if (k==1)
        idx_m, phi, method.psi_prev, tau, deltaSum, success = generateNextIterate_Mem1!(method.x, method.zprime, method)
    else
        idx_m, phi, method.psi_prev, tau, deltaSum, success, model = generateNextIterate!(method.x, method.zprime, method)
    end


    # Calculate f, g for our new iterate
    @. method.inputVec = method.x + method.x0       # Add back shift when calling oracle, use working vector
    method.value = preconditionedOracle!(method.gradient, method, oracle, method.inputVec)      # Call oracle and save results into method.value and method.gradient
    method.oracleCtr += 1

    # Calculate L for Q_{m,n} (and additional values for later)
    LReq, wExpression, distSq = calculateL(method, idx_m)

    if LReq <= method.L
        # This means Q_{m,n} >= 0, so L_n is valid and the new hypothesis U_n is valid
        
        flags.flag_NullStep = false # Set null step flag to false

        # psi_prev and zprime were already saved to method

        method.newL = method.L  # Set L_{n+1} = L_n

    else
        # This means Q_{m,n} < 0, so L_n is not valid

        # This iteration was a null step
        flags.flag_NullStep = true 

        # Increment L_{n+1} for the next iteration
        method.newL = max(LReq, 2*method.L) 
        
        # Set τ_n and Δ_n so that U_n = 0 and hypothesis holds vacuously
        tau = 0.0      
        deltaSum = 0.0 

        # psi_prev and zprime were saved to method, but will be ignored because we will force z_n = x_0 in the next update step

    end

    # Update μ estimate (regardless of whether this iteration was null step or serious step)
    muEst = calculateMu(wExpression, distSq)
    method.mu = min(method.mu, muEst)

    # Save τ_n and Δ_n
    method.newTau = tau
    method.newDelta = deltaSum

    # Iteration counts
    method.iteration += 1
    method.epochIter += 1

    # --- Restart Options (ASPGM) ---
    if opts.mode == :A
        # If final step flag is set, and this iteration was serious step, then restart
        if (!flags.flag_NullStep)&&(flags.flag_FinalStep)
            restart(method, oracle)
        else
            # Otherwise, check if restart condition is hit. If so, set flag for the next iteration
            #   Restart conditions:
            #       1)  Serious step
            #       2)  At least XX iterations in this epoch
            #       3)  τ_n >= c*[ L_n/mu + Δ_n/(f_0 - f_n) ], typically with c = 2
            condition = (!flags.flag_NullStep)&&(method.epochIter >= opts.minEpochSize)&&( tau >= opts.restartFactor*(method.L/method.mu + method.newDelta/(method.f0 - method.value) ) )

            if condition||(method.epochIter >= opts.maxEpochSize)
                flags.flag_FinalStep = true     # Set flag to restart on next serious step
            end
        end
    end

    # --- Check guarantee for early terminate (BSPGM) ---
    if opts.mode == :B
        if guarantee(method) < opts.guarantee_tol
            print("\nReached desired guarantee: (f(x_n) - f(x_0))/(1/2||x_0-x_*||^2) <= ", guarantee(method))
            flags.flag_Terminate = true
        end
    end

    return

end

# --- Generate x_n by solving subproblem ---
@views function generateNextIterate!(dest_x::Vector{Float64}, dest_zprime::Vector{Float64}, method::ASPGM)

    v_m, idx_m = get_v_m(method)

    # Find optimal values of ρ and γ
    rho, gamma, delta, success, status, model = optimize_rho_gamma(method, v_m)

    # Set ϕ_n equal to subproblem optimal value (equivalently: τ')
    phi = dot(rho, method.taus) + sum(gamma)

    # Set ψ_n according to standard OBL Update (equivalently: τ_n - τ')
    if method.flags.flag_FinalStep
        psi = sqrt(phi)
    else
        psi = (1+sqrt(1+8*phi))/2
    end

    # Set z' (dest_zprime), τ_n, x_n (dest_x), and Δ_n using subproblem solution

    mul!(method.scratchVec1, method.Z, rho)     # scratchVec1 = Z*rho
    mul!(method.scratchVec2, method.G, gamma)   # scratchVec2 = G*gamma

    @. dest_zprime = 1/method.L*method.scratchVec1 - 1/method.L*method.scratchVec2
    tau = phi + psi
    @. dest_x = psi/tau * dest_zprime + phi/tau * method.X[:,idx_m] - phi/(tau*method.L)*method.G[:,idx_m]
    deltaSum = dot(rho, method.Deltas) + delta

    return idx_m, phi, psi, tau, deltaSum, success, model

end

# --- If memory size is 1, then we solve the subproblem exactly to generate x_n ---
@views function generateNextIterate_Mem1!(dest_x::Vector{Float64}, dest_zprime::Vector{Float64}, method::ASPGM)

    L_n = method.L
    L_minus = method.LPrev[end]
    tau_minus = method.taus[end]

    gammaVals = zeros(4)
    rhoVals = zeros(4)
    objVals = zeros(4)
    
    # Note: We can use idx = end because this is memory size 1
    dot_z_g_minus = 1/L_minus*dot_B(method.Z[:,end], method.G[:,end], method)
    dot_g_x_minus = dot_B(method.G[:,end], method.X[:,end], method)
    z_minus_normSq = 1/L_minus^2*normSq_B(method.Z[:,end], method)      
    g_minus_normSq = normSq_B(method.G[:,end], method)  

    # Save off reused values
    v1 = L_minus^2/(2*L_n)*z_minus_normSq
    v2 = tau_minus*(1/L_minus - 1/L_n)*1/2*g_minus_normSq - L_minus/2*z_minus_normSq
    delta = L_n*tau_minus*(1/L_minus^2 - 1/L_n^2)*1/2*g_minus_normSq

    # Case 1: γ = 0     
    rho = (-v2 + sqrt(abs(v2^2 + 4*v1*delta)))/(2*v1)       # discriminant will always be nonnegative, so add abs to avoid numerical error
    if rho >= 0
        gammaVals[1] = 0
        rhoVals[1] = rho
        objVals[1] = tau_minus*rho
    end

    # Case 2: ρ = 0
    u1 = 1/(2*L_n)*g_minus_normSq
    u2 = dot_g_x_minus - u1
    gamma = (-u2 + sqrt(abs(u2^2 + 4*u1*delta)))/(2*u1)     # discriminant will always be nonnegative, so add abs to avoid numerical error
    if gamma >= 0
        gammaVals[2] = gamma
        rhoVals[2] = 0
        objVals[2] = gamma
    end

    # Case 3: Tangency
    w = tau_minus*1/L_n*g_minus_normSq + L_minus/L_n*dot_z_g_minus
    r = tau_minus*dot_g_x_minus - tau_minus*1/(2*L_minus)*g_minus_normSq + L_minus/2*z_minus_normSq
    s = L_minus^2/L_n*z_minus_normSq + tau_minus*L_minus/L_n*dot_z_g_minus

    if s != 0
        a = w^2/s^2*v1 + 1/(2*L_n)*g_minus_normSq - w/s*L_minus/L_n*dot_z_g_minus
        b = 2*w*r/s^2*v1 + w/s*v2 + dot_g_x_minus - 1/(2*L_n)*g_minus_normSq - r/s*L_minus/L_n*dot_z_g_minus
        c = r^2/s^2*v1 + r/s*v2 - delta

        discr = b^2 - 4*a*c
        if (discr >= 0)&&(a!=0)     # Note if a=0, then gamma = 0/0 so we do not include as possible solution
            gamma = (-b + sqrt(b^2 - 4*a*c))/(2*a) 
            rho = (w*gamma + r)/s
            if (gamma >= 0)&&(rho >= 0)
                gammaVals[3] = gamma
                rhoVals[3] = rho
                objVals[3] = tau_minus*rho + gamma
            end

            gamma = (-b - sqrt(b^2 - 4*a*c))/(2*a)
            rho = (w*gamma + r)/s
            if (gamma >= 0)&&(rho >= 0)
                gammaVals[4] = gamma
                rhoVals[4] = rho
                objVals[4] = tau_minus*rho + gamma
            end
        end
    end

    # Find the best option
    _, idx = findmax(objVals)
    gamma = gammaVals[idx]
    rho = rhoVals[idx]

    # Set ϕ_n equal to subproblem optimal value (equivalently: τ')
    phi = rho*tau_minus + gamma

    # Set ψ_n according to standard OBL Update (equivalently: τ_n - τ')
    if method.flags.flag_FinalStep
        psi = sqrt(phi)
    else
        psi = (1+sqrt(1+8*phi))/2
    end

    success = true
    idx_m = 1

    # Set z', τ_n, x_n, and Δ_n using subproblem solution
    @. dest_zprime = 1/L_n*rho*method.Z[:,end] - 1/L_n*gamma*method.G[:,end]
    tau = phi + psi
    @. dest_x = psi/tau*dest_zprime + phi/tau*method.X[:,end] - phi/(tau*L_n)*method.G[:,end]
    deltaSum = rho*method.Deltas[end] + delta

    return idx_m, phi, psi, tau, deltaSum, success
end

# Calculate index m = argmin_i v_i and its corresponding value v_m. Where v_i = f_i - 1/(2L_i)||g_i||^2
function get_v_m(method)
    V = method.F .- 1/(2*method.L)*diag(method.ZG_MAT)[length(method.taus)+1:end]    # Pull ||g_i||^2 from (the second half of) the diagonal of ZG_MAT
    idx = (method.taus .== 0)
    V[idx] .= Inf    # Set invalid indices to inf so they are excluded
    v_m, idx_m = findmin(V)   # Calculate m and v_m

    return v_m, idx_m
end

# --- Solve subproblem to find optimal ρ, γ, τ ---
function optimize_rho_gamma(method::ASPGM, v_m::Float64)
    
    k = method.options.memorySize_SP

    rhocoeffs = method.h - v_m*method.taus
    gammacoeffs = method.w .- v_m
    
    # Get index of last serious step
    idx_s = getLastSuccessIdx(method)

    # Safeguard to prevent huge jumps in rho and gamma - we enforce rho,gamma < 1/tol
    tol = min(1e-8, 1e-3/method.taus[idx_s]) # This allows new tau to be very large if the previous one was very large, but prevents huge jumps

    # Calculate δ = L_n τ_s(1/L_s^2 - 1/L_n^2)*1/2||g_s||^2
    delta = method.L*method.taus[idx_s]*(1/method.LPrev[idx_s]^2 - 1/method.L^2)*1/2*method.ZG_MAT[k+idx_s, k+idx_s]     #Note: ZG_MAT[k+idx_s,k+idx_s] = ||g_s||^2

    # Prep variables to pass into SOCP 
    A = 1/method.L*method.ZG_MAT               # Rescale to account for different Z and G definitions
    b = vcat(rhocoeffs, gammacoeffs)
    c = vcat(method.taus, ones(k))
    d = delta

    fixedIdxRho = findall(method.taus == 0)        # For any i where τ_i = 0 (including inactive indices), we will fix ρ_i = 0
    fixedIdxGamma = k .+ setdiff(1:k, getActiveIndices(method))  # For any inactive index i, we will fix γ_i = 0
    fixedIdx = vcat(fixedIdxRho, fixedIdxGamma)

    # Pass into solver
    y, success, status, model = solveSOCP(method, A, b, c, d, fixedIdx, method.model; tol=tol)

    rho, gamma = clean_solution(success, y, method.taus, method.L, method.LPrev, idx_s)

    return rho, gamma, delta, success, status, model   
end

# --- Solve SOCP given by ---
# max ⟨c, y⟩
# s.t. 1/2*y'*A*y <= ⟨b, y⟩ + d
#      y >= 0
#      y[fixedIdx] = 0      # Corresponds to fixing ρ_i = 0 when τ_i = 0, and fixing γ_i when i is an inactive index
#
function solveSOCP(method::ASPGM, A::Matrix{Float64}, b::Vector{Float64}, c::Vector{Float64}, d::Float64, fixedIdx::Vector{Int64}, model::Model; tol::Float64=1e-8)
    K = length(b)

    # Check for invalid A matrix, if so return placeholder values
    if (sum(isnan.(A)) > 0)||(sum(isinf.(A)) > 0)
        success = false
        return zeros(size(b)), 0.0, success, termination_status(model), model
    end

    # If A is very large, rescale the problem so that Mosek can handle it better
    if maximum(A) > 1e9
        rescaleFactor = maximum(A)/1e3
        A = A./rescaleFactor
        b = b./rescaleFactor^2
        d = d/rescaleFactor^2
    end
    
    # A is PSD but not exactly - add shift so condition number is good enough for Mosek
    lams = real(eigvals(A))     # take real part to account for numerical error
    minEval = minimum(lams)
    shift = 1e-10 + (minEval < 0)*abs(minEval)*abs(maximum(lams))
    A = A + shift*Matrix(I,K,K)

    # Clear out Mosek model, and repopulate it. This is faster than constructing a new one every time
    empty!(model)
    set_silent(model)

    # Include tolerance for numerical stability
    @variable(model, 1/tol >= y[1:K] >= 0)

    # Force ρ_i = 0 when τ_i = 0, since the U_i hypothesis is vacuous in that case anyway
    # Force γ_i = 0 for any inactive indices
    for i in fixedIdx
        fix(y[i], 0; force=true)
    end

    @constraint(model, 1/2*y'*A*y <= b'*y + d)

    @objective(model, Max, dot(c, y))

    try
        optimize!(model)
    catch e
        # If optimization fails, return placeholder values
        success = false
        return zeros(size(y)), success, termination_status(model), model
    end

    if termination_status(model) == MOI.INFEASIBLE
        success = false
    else
        success = true
    end

    return JuMP.value.(y), success, termination_status(model), model
end

# --- Check validity of our solution - if not valid, revert to standard induction for this step ---
function clean_solution(success, y, taus, L, LPrev, s_idx)
    k = length(taus)
    rho = max.(y[1:k], 0)
    gamma = max.(y[k+1:end], 0)

    # If the solution to our subproblem is worse than τ_s, then just use standard induction: τ' = τ_s, ρ_s = L_n/L_s, γ = 0
    if isnan(dot(rho, taus) + sum(gamma)) || (dot(rho, taus) + sum(gamma) < taus[s_idx]) || (!success)
        rho .= 0
        rho[s_idx] = L/LPrev[s_idx]
        gamma .= 0
    end

    return rho, gamma
end

# --- Restart method at current iterate (start new epoch), re-initialize method variables accordingly ---
# This requires several instances of "un-transforming" gradients. Recall that the gradients stored in method.G, etc. are the gradients with respect to 
# the ⟨⋅,⋅⟩_B dot product. To build a new preconditioner (among other things), we have to convert to the true Euclidean gradient - i.e. "un-transforming"
@views function restart(method::ASPGM, oracle)

    opts = method.options
    k = opts.memorySize_SP
    t = opts.memorySize_QN

    @. method.x0 = method.x + method.x0     # Set new x0, and add back x0 shift from previous epoch
    
    # We have to un-transform the G's before we can build our new preconditioner Y
    for i = 1:k
        ApplyHessianApprox!(method.outputVec, method.G[:,i], method)
        method.G[:,i] .= method.outputVec
    end
    # No transformation necessary for the X's

    # Similarly, we will have to un-transform the gradient (and later re-transform it to match our new preconditioner)
    method.value = method.value     # Use value unchanged
    ApplyHessianApprox!(method.outputVec, method.gradient, method) # First, un-transform from the old S,Y
    
    # Reset quasi-Newton memory storage S and Y 
    fill!(method.S, 0.0)
    fill!(method.Y, 0.0)
    method.PrecLU = nothing
    method.theta = 1.0

    # If preconditioning is still enabled, set new S and Y (and M)
    if !method.flags.flag_DisablePrec
        precIdxs = getActiveIndices(method)[end-t+1 : end]

        # Push over new values to preconditioner S: x_{n-t+1}-x_{n-t}, ..., x_n-x_{n-1}, and Y: g_{n-t+1}-g_{n-t}, ..., g_n-g_{n-1}
        for (trueIdx,storageIdx) in enumerate(precIdxs)
            if trueIdx == t
                @. method.S[:,trueIdx] = method.x - method.X[:,storageIdx]              # x_n - x_{n-1}
                @. method.Y[:,trueIdx] = method.outputVec - method.G[:,storageIdx]      # g_n (un-transformed) - g_{n-1}
            else
                nextStorageIdx = mod1(storageIdx+1, k)
                @. method.S[:,trueIdx] = method.X[:,nextStorageIdx] - method.X[:,storageIdx]       # x_j - x_{j-1}
                @. method.Y[:,trueIdx] = method.G[:,nextStorageIdx] - method.G[:,storageIdx]       # g_j - g_{j-1}
            end
        end

        # Calculate and save off LU matrix decomposition for efficient preconditioning calculations
        method.PrecLU, method.theta = buildPrecLU(method)    
    else
        # Otherwise, if preconditioning is disabled recalculate gradient without preconditioner (since S, Y are empty)
        # (We need to recalculate because method.gradient must have had NaNs/Infs, so we cannot just un-transform)
        _ = preconditionedOracle!(method.outputVec, method, oracle, method.x0)
        method.oracleCtr += 1
    end

    # Re-transform our un-transformed gradient (gTemp), using the new preconditioners S, Y
    method.inputVec .= method.outputVec     # Transfer gTemp from output to input
    ApplyInverseHessianApprox!(method.gradient, method.inputVec, method)

    # Save new initial values for this epoch
    method.f0 = method.value
    fill!(method.x, 0.0)              # Implement shift for this epoch so that we can assume that x0 = 0

    # Set tracker indices
    method.startIdx = 0
    method.nextIdx = 1

    # Initialize various storage vectors and matrices
    fill!(method.Z, 0.0)
    fill!(method.G, 0.0)
    fill!(method.X, 0.0)
    fill!(method.F, Inf)
    fill!(method.ZG_MAT, 0.0)
    fill!(method.zprime, 0.0)
    method.psi_prev = 1.0    
    fill!(method.taus, 0.0)
    fill!(method.h, 0.0)
    fill!(method.w, 0.0)
    fill!(method.LPrev, 0.0)
    fill!(method.Deltas, 0.0)

    method.newTau = 1.0
    method.newDelta = 0.0

    method.epochIter = 0
    method.numEpochs += 1

    # Reset specific flags
    method.flags.flag_FinalStep = false
    method.flags.flag_NullStep = false

    method.mu = Inf

    LInit = getLInit(method, oracle)
    method.L = LInit
    method.newL = LInit

end

# --- Calculate initial value L_0 for an epoch, based on perturbing x0 and measuring smoothness ---
function getLInit(method::ASPGM, oracle)

    # y = method.x0 - 1e-4*g_0/||g_0||
    c = 1e-4/norm(method.gradient)
    @. method.inputVec = method.x0 - c*method.gradient

    # Calculate fy and gy (stored in outputVec)
    fy = preconditionedOracle!(method.outputVec, method, oracle, method.inputVec)
    method.oracleCtr += 1

    #---We then follow the logic of calculateL, but avoiding additional allocations---#
    # LEst = ||g_x - g_y||^2 / (2 * (f_y - f_x - <g_x, y - x>))

    # Denominator: (f_y - f_x - <g_x, y - x>))
    @. method.inputVec = -c*method.gradient         # inputVec = y - x
    denom = fy - method.value - dot_B(method.gradient, method.inputVec, method)     # Calculate denominator separately - we can reuse it in calculating mu to save us a dot product

    # Numerator: ||g_x - g_y||^2
    @. method.inputVec = method.outputVec - method.gradient
    gDiffNormSq = normSq_B(method.inputVec, method)

    LEst = gDiffNormSq/(2*denom)
    LEst = abs(LEst)    # Sanity check

    # Special handling for 0/0 cases
    L_default = 0.01
    if isnan(LEst)||isinf(LEst)
        LEst = L_default
    end

    return LEst

end

# --- Calculate smallest L value such that cocoercivity holds (Q_{m,n}(L) = 0) ---
#       L = ||g_x - g_y||^2 / (2 * (f_y - f_x - <g_x, y - x>))
@views function calculateL(method::ASPGM, idx_m::Int64)

    @. method.inputVec = method.gradient - method.G[:,idx_m]
    gDiffNormSq = normSq_B(method.inputVec, method)

    @. method.inputVec = method.X[:,idx_m] - method.x
    denom = method.F[idx_m] - method.value - dot_B(method.gradient, method.inputVec, method)     # Calculate denominator separately - we can reuse it in calculating mu to save us a dot product
    distSq = normSq_B(method.inputVec, method)      # Calculate ||x_m-x_n||^2 now for use later - since x_m-x_n is already calculated

    LEst = gDiffNormSq/(2*denom)
    LEst = abs(LEst)    # Sanity check

    # Special handling for 0/0 cases.
    # Recall Q = denom - gDiffNormSq/2L
    if isnan(LEst)
        LEst = method.L # If NaN, then gDiffNormSq = denom = 0, so Q = 0 >= 0. L is unchanged
    elseif isinf(LEst)
        LEst = 2*method.L # If Inf, then gDiffNormSq > 0, denom = 0, so Q < 0. Increase L
    end

    return LEst, denom, distSq

end

# --- Calculate largest μ value such that strong convexity holds. This leverages calculations that have already been made in calculateL ---
function calculateMu(wExpression::Float64, distSq::Float64)

    muEst = 2*wExpression/distSq
    muEst = abs(muEst) # Sanity check

    return muEst
end

# --- Preconditioned dot product, using S,Y as quasi-Newton preconditioners (via the precalculated PrecLU) ---
function dot_B(v::Union{Vector{Float64}, SubArray{Float64, 1, Matrix{Float64}}}, w::Union{Vector{Float64}, SubArray{Float64, 1, Matrix{Float64}}}, method::ASPGM)
    if isnothing(method.PrecLU)
        return dot(v, w)    # If PrecLU preconditioner is not set, return standard dot product
    end

    # --Follow the same logic as ApplyHessianApprox--

    t = size(method.S,2)

    vec = vcat(method.theta*(method.S' * v), method.Y'*v)

    # M should not be singular, but even if it was then we would have set PrecLU = nothing, and returned v
    q = method.PrecLU \ vec

    mul!(method.scratchVec1, method.S, q[1:t])          # Store S*q[1:t] in scratchVec1
    mul!(method.scratchVec2, method.Y, q[t+1:end])      # Store Y*q[t+1:end] in scratchVec2

    # res = θ*v'*w - ([θ*S Y]*q)'*w
    return method.theta*dot(v,w) - method.theta*dot(method.scratchVec1, w) - dot(method.scratchVec2, w)

end

# --- Preconditioned norm squared, using S,Y as quasi-Newton preconditioners ---
function normSq_B(v::Union{Vector{Float64}, SubArray{Float64, 1, Matrix{Float64}}}, method::ASPGM)
    return abs(dot_B(v, v, method))     # Take absolute value in case of numerical error
end

# --- Build matrix M for preconditioner calculations, then take its LU decomposition and save it off; this speeds up repeated Mx=y solves ---
@views function buildPrecLU(method::ASPGM)

    t = method.options.memorySize_QN

    theta = dot(method.Y[:, end], method.Y[:, end]) / dot(method.S[:, end], method.Y[:, end])

    D = diagm([dot(method.S[:, i], method.Y[:, i]) for i in 1:t])
    T = zeros(t, t)
    for i in 1:t
        for j in 1:i-1
            T[i, j] = dot(method.S[:, i], method.Y[:, j])
        end
    end

    M = zeros(2t, 2t)
    M[1:t, 1:t] = theta*(method.S'*method.S)
    M[1:t, t+1:2t] = T
    M[t+1:2t, 1:t] = T'
    M[t+1:2t, t+1:2t] = -D


    if any(!isfinite, M)
        PrecLU = nothing    # Check for Infs/NaNs
    else
        PrecLU = lu(M; check=false)
        if !issuccess(PrecLU)       # M should not be singular, but if it is due to numerical issues, set PrecLU = nothing to bypass preconditioning
            PrecLU = nothing
        end
    end

    return PrecLU, theta

end

# --- Calculate B^{-1} v where B^{-1} is the qausi-Newton Hessian approximation formed using S and Y. Follows the standard method of Byrd, Nocedal and Schnabel (1994) ---
# The LU decomposition PrecLU is calculated and saved off ahead of time.
function ApplyHessianApprox!(dest::Vector{Float64}, v::Union{Vector{Float64}, SubArray{Float64, 1, Matrix{Float64}}}, method::ASPGM)

    # If no preconditioner is set, return v
    if isnothing(method.PrecLU)
        dest .= v
        return
    end

    t = size(method.S,2)

    vec = vcat(method.theta*(method.S' * v), method.Y'*v)

    # LU should not be singular, but even if it was then we would have set PrecLU = nothing, and returned v
    q = method.PrecLU \ vec

    mul!(method.scratchVec1, method.S, q[1:t])  # Store S*q[1:t] in scratchVec1
    mul!(method.scratchVec2, method.Y, q[t+1:end])    # Store Y*q[t+1:end] in scratchVec2

    # dest = θ*v - ([θ*S Y]*q)
    @. dest = method.theta * v - (method.theta * method.scratchVec1 + method.scratchVec2)

    return
end

# --- Calculate B*v where B is the qausi-Newton inverse Hessian approximation formed using S and Y. Follows the standard two-loop method of Nocedal (1980) ---
# The vector B*v is written into dest
@views function ApplyInverseHessianApprox!(dest::Vector{Float64}, v::Vector{Float64}, method::ASPGM)

    if isnothing(method.PrecLU)
        dest .= v
        return
    end

    t = size(method.S,2)
    alpha = zeros(t)
    eta = zeros(t)

    dest .= v

    for i = t:-1:1
        eta[i] = 1.0 / dot(method.Y[:,i], method.S[:,i])
        alpha[i] = eta[i] * dot(method.S[:,i], dest)

        @. dest = dest - alpha[i] * method.Y[:,i]  
    end

    @. dest = 1/method.theta*dest

    for i = 1:t
        beta = eta[i] * dot(method.Y[:,i], dest)
        @. dest = dest + (alpha[i] - beta)*method.S[:,i]
    end

    return
end

# --- Calls function/gradient oracle then applies preconditioning to the returned gradient. Also checks if ||g|| < g_tol ---
# The function value is returned and the preconditioned gradient is written into dest_g
function preconditionedOracle!(dest_g::Vector{Float64}, method::ASPGM, oracle, x::Vector{Float64})

    if isnothing(method.PrecLU)
        val = oracle(dest_g, x)
        gNorm = norm(dest_g, Inf)
        if gNorm < method.options.g_tol
            print("\nGradient tolerance met: ||g_n||_inf = ", gNorm)
            method.flags.flag_Terminate = true
        end
        return val
    else
        val = oracle(method.outputVec, x)
        gNorm = norm(method.outputVec, Inf)
        if gNorm < method.options.g_tol
            print("\nGradient tolerance met: ||g_n||_inf = ", gNorm)
            method.flags.flag_Terminate = true
        end
        # Transfer output to input
        method.inputVec .= method.outputVec
        ApplyInverseHessianApprox!(dest_g, method.inputVec, method)
        return val
    end

end

# --- Return the indices that have active memory storage ---
function getActiveIndices(method::ASPGM)
    if method.startIdx==0
        return nothing
    end
    k = method.options.memorySize_SP
    # Cycle from startIdx to mod(nextIdx-1,k)
    return mod1.(method.startIdx : (method.startIdx + mod(method.nextIdx - method.startIdx - 1, k)), k)
end

# --- Return the index of the last iteration that was not a null step. Represents idx_s in paper ---
function getLastSuccessIdx(method)
    if method.startIdx == 0
        return nothing
    end
    indices = getActiveIndices(method)
    j = findlast(method.taus[indices] .!= 0)
    if isnothing(j)
        return nothing
    else
        return indices[j]
    end
end


# --- Add v into a vector x, replacing data in nextIdx ---
function push_data_vec!(x::Vector{Float64},v::Float64,idx::Int64)
    x[idx] = v
    return
end

# --- Push column vector v into a matrix X, taking the tail columns if the size grows over k ---
function push_data_mat!(X::Matrix{Float64},v::Vector{Float64},idx::Int64)
    X[:,idx] .= v
    return
end

# --- Update ZG_MAT = [Z'Z  -Z'G; -G'Z  G'G] with new rows and columns ---
function push_data_ZG_MAT!(MAT::Matrix{Float64}, vec_zi_z::Vector{Float64}, vec_zi_g::Vector{Float64}, vec_gi_z::Vector{Float64}, vec_gi_g::Vector{Float64}, idx::Int64)

    k = Int(size(MAT,1)/2)

    # Update Z'Z block
    MAT[1:k, idx] .= vec_zi_z
    MAT[idx, 1:k] .= vec_zi_z

    # Update G'G block
    MAT[k+1:end, k+idx] .= vec_gi_g
    MAT[idx+k, k+1:end] .= vec_gi_g

    # Update -Z'G
    MAT[1:k, k+idx] .= -vec_zi_g
    MAT[idx, k+1:end] .= -vec_gi_z

    # Update -G'Z
    MAT[k+1:end, idx] .= -vec_gi_z
    MAT[k+idx, 1:k] .= -vec_zi_g

    return
end

# --- Copy data in our stored variables from one index to another ---
#     This is used before pushing new data in the special case that τ is full and τ_{n-k-1} (taus[1]) is the only nonzero component of τ. See Remark 1 in the paper ---
#     After performing this shift, the source index data will be replaced with the new incoming data in update()
@views function shiftData(method::ASPGM, copyIdx::Int64, replaceIdx::Int64)
    method.G[:,replaceIdx] .= method.G[:,copyIdx]
    method.Z[:,replaceIdx] .= method.Z[:,copyIdx]
    method.X[:,replaceIdx] .= method.X[:,copyIdx]
    method.F[replaceIdx] = method.F[copyIdx]
    method.w[replaceIdx] = method.w[copyIdx]
    method.h[replaceIdx] = method.h[copyIdx]
    method.LPrev[replaceIdx] = method.LPrev[copyIdx]
    method.taus[replaceIdx] = method.taus[copyIdx]
    method.Deltas[replaceIdx] = method.Deltas[copyIdx]

    shiftData_ZG_MAT!(method.ZG_MAT, copyIdx, replaceIdx)

end

# --- Helper function for shifting data in ZG_MAT. See above ---
@views function shiftData_ZG_MAT!(MAT::Matrix{Float64}, copyIdx::Int64, replaceIdx::Int64)

    k = Int(size(MAT,1)/2)

    # --- Copy over the row and column for each block ---

    # ZZ Block
    MAT[1:k, replaceIdx] .= MAT[1:k, copyIdx]
    MAT[replaceIdx, 1:k] .= MAT[copyIdx, 1:k]

    # GG Block
    MAT[k+1:end, k+replaceIdx] .= MAT[k+1:end, k+copyIdx]
    MAT[k+replaceIdx, k+1:end] .= MAT[k+copyIdx, k+1:end]

    # ZG Block
    MAT[1:k, k+replaceIdx] .= MAT[1:k, k+copyIdx]
    MAT[replaceIdx, k+1:end] .= MAT[copyIdx, k+1:end]

    # GZ Block
    MAT[k+1:end, replaceIdx] .= MAT[k+1:end, copyIdx]
    MAT[k+replaceIdx, 1:k] .= MAT[k+copyIdx, 1:k]

    return
end

# --- Convergence guarantee on (f(x_n) - f(x_0))/(1/2||x_0-x_*||^2) for ASPGM/BSPGM. Mostly applicable for BSPGM. For ASPGM, the value is with respect to the preconditioner B and the last restart iterate ---
function guarantee(method::ASPGM)
    if method.newTau > 0
        return method.L/(method.newTau)
    else
        idx_s = findlast(method.taus .!= 0)
        return method.LPrev[idx_s]/(method.taus[idx_s])
    end
end

# --- Method title ---
function methodTitle(method::ASPGM)
    if method.options.mode == :B
        return "BSPGM-"*string(method.options.memorySize_SP)
    else
        mainStr = "ASPGM"
        return mainStr * "-" * string(method.options.memorySize_SP) * "-" * string(method.options.memorySize_QN)
    end
end
