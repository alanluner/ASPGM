using JuMP, Mosek, MosekTools
import MathOptInterface as MOI

#========================================================================
This code implements the Adaptive Subgame Perfect Gradient Method (ASPGM). At each iteration, ASPGM makes a momentum-type update,
optimized dynamically based on a (limited) memory/bundle of past first-order information. It is linesearch-free, parameter-free, and adaptive.

See https://github.com/alanluner/ASPGM for implementation details.

See https://arxiv.org/pdf/2510.21617 for the paper that introduces the method.
========================================================================#

mutable struct ASPGM
    # ------------- Initial values ------------- #
    x0                  # Starting point for this epoch
    f0                  # Stores initial f(x0) for this epoch
    # ------------- Current iterate data ------------- #
    x                   # Iterate x_n
    value               # Function value f_n
    gradient            # Gradient g_n
    L                   # Smoothness estimate
    mu                  # Strong convexity estimate
    newDelta            # Error parameter Δ_n
    newTau              # Convergence guarantee τ_n
    newL                # Smoothness estimate for next iterate L_{n+1}
    # ------------- Memory Storage - Subgame Perfect ------------- #
    Z                   # Past auxiliary sequence data z_{n-k} to z_{n-1}. We will use a different scaling than in the paper [ L_i(z_{i+1} - x_0) rather than L_i/L_n(z_{i+1} - x_0) ].
    G                   # Past gradient data g_{n-k} to g_{n-1}. We will use a different scaling than in the paper [ g_i rather than g_i/L_n ].
    X                   # Past iterate data x_{n-k} to x_{n-1}
    F                   # Past function value data f_{n-k} to f_{n-1}
    MAT                 # Efficient storage matrix for [Z'Z  -Z'G; -G'Z  G'G] for SOCP subproblem
    zprime              # Optimal auxiliary step z' as determined by subproblem
    psi_prev            # ψ_{n-1} = τ_{n-1} - τ'
    taus                # (τ_{n-k}, ..., τ_{n-1})
    h                   # (h_{n-k}, ..., h_{n-1})
    w                   # (w_{n-k}, ..., w_{n-1})
    LPrev               # (L_{n-k}, ..., L_{n-1})
    Δ                   # (Δ_{n-k}, ..., Δ_{n-1})
    # ------------- Preconditioning Memory Storage ------------- #
    S                   # Quasi-Newton matrix formed from iterate differences: (x_{n-t+1}-x_{n-t}, ..., x_{n}-x_{n-1})  
    Y                   # Quasi-Newton matrix formed from gradient differences: (g_{n-t+1}-g_{n-t}, ..., x_{n}-x_{n-1})  
    # ------------- Metadata ------------- #
    iteration           # total iteration number
    epochIter           # iteration number within this epoch
    numEpochs           # number of epochs
    oracleCtr           # number of oracle calls
    model               # Mosek optimization model
    flags               # Set of flags for branching behavior according to the algorithm
    # ------------- Implementation Options ------------- #
    options             # (See below)
end

# Options for ASPGM
mutable struct options_ASPGM
    mode                # :A - ASPGM (with restarting and preconditioning)   [ vs :B - BSPGM (no restarting or preconditioning) ]
    memorySize_SP       # Memory size for subgame perfect memory
    memorySize_QN       # Memory size for quasi-newton memory
    restartFactor       # Value > 1 that determines restart condition on τ_n. Larger value results in less frequent restarts (Default: 2)
    minEpochSize        # Minimum number of iterations in an epoch before allowing a restart. Set to 0 to ignore (Default: 20)
    maxEpochSize        # Maximum number of iterations in epoch before forcing a restart. Set to Inf to ignore (Default: 100)
    g_tol               # Terminate early when ||g_n||_inf <= g_tol. Set to -1 to ignore
end

# Options for BSPGM
mutable struct options_BSPGM
    mode                # :B - BSPGM (no restarting or preconditioning)   [ vs :A - ASPGM (with restarting and preconditioning) ]
    memorySize_SP       # Memory size for subgame perfect memory
    guarantee_tol       # Terminate early when convergence guarantee (L_n/τ_n) is less than guarantee_tol. Set to -1 to ignore
    g_tol               # Terminate early when ||g_n||_inf <= g_tol. Set to -1 to ignore.
end

options_ASPGM() = options_ASPGM(:A, 5, 5, 2.0, 20, 100, -1)

options_BSPGM() = options_BSPGM(:B, 5, -1, -1)

mutable struct flags
    flag_NullStep           # Flag that previous step was null step (i.e., τ_{n-1} = 0)
    flag_FinalStep          # Set to true once phi exceeds restart value. Then update will use last-step correction until next serious step, triggering restart
    flag_Terminate          # Flag to terminate algorithm once desired tolerance is met (either via g_tol or guarantee_tol)
end

flags() = flags(false, false, false)


# --------Method Constructors-------- #

ASPGM(k, t) = ASPGM(missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,
                    missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,
                    options_ASPGM(:A, k, t, 2.0, 20, 100, -1))

ASPGM() = ASPGM(5,5)

BSPGM(k) = ASPGM(missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,
                missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,
                options_BSPGM(:B, k, -1, -1))

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
# run(ASPGM(5,5), oracle, zeros(d); oracleCalls = 500)
#
function runMethod(method::ASPGM, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    initialize(method, oracle, x0)
    
    x = x0

    exit = false
    metaData = []
    i = 0
    t0 = time()

    if saveDetails
        metaData = [method.value  norm(method.gradient)  guarantee(method)  method.newDelta  method.newTau  method.L]
    end

    while !exit
        ctr1 = method.oracleCtr
        x = update(method, oracle)
        ctr2 = method.oracleCtr

        # If we did multiple oracle calls during this iteration (usually in the case of a restart), then save off data multiple times - once for each oracle call
        if saveDetails
            for i=1:ctr2-ctr1
                metaData = vcat(metaData, [method.value  norm(method.gradient)  guarantee(method)  method.newDelta  method.newTau  method.L])
            end
        end

        t = time() - t0

        if method.flags.flag_Terminate
            exit = true
        elseif (method.oracleCtr > oracleCalls)&&(t >= runTime)
            exit = true
        end

    end

    # Return x with best function value
    if (isempty(method.F)) || method.value < minimum(method.F)
        x = method.x
        val = method.value
    else
        val, best_idx = findmin(method.F)
        x = method.X[:,best_idx]
    end

    return x, val, metaData
end


# --- Initialize variables of the method and validate settings ---
function initialize(method::ASPGM, oracle, x0::Vector{Float64})

    # Validate settings
    validate(method)

    # Save initial values
    method.x0 = x0
    method.value, method.gradient = oracle(x0)
    method.f0 = method.value
    method.x = x0 - x0 # This implements a shift so that we can assume that x0 = 0

    # Initialize various storage vectors and matrices
    method.Z = Array{Float64}(undef, length(x0), 0)
    method.G = Array{Float64}(undef, length(x0), 0)
    method.X = Array{Float64}(undef, length(x0), 0)
    method.F = []
    method.MAT = Array{Float64}(undef, 0, 0)
    method.zprime = x0 - x0
    method.psi_prev = 1.0    
    method.taus = []      
    method.h = []
    method.w = []
    method.LPrev = []
    method.Δ = []

    method.newTau = 1.0     # Set τ_0 = 1
    method.newDelta = 0.0   # Set Δ_0 = 0

    # Initialize quasi-Newton matrices
    method.S = Array{Float64}(undef, length(x0), 0)
    method.Y = Array{Float64}(undef, length(x0), 0)

    # Iteration counters
    method.iteration = 0
    method.epochIter = 0
    method.numEpochs = 1
    method.oracleCtr = 1 # Count initial oracle call on x0

    # Instantiate a single Mosek instance for solving our subproblem at each step
    # Clearing it out and reusing at each step is faster than building a new one every iteration.
    method.model = Model(Mosek.Optimizer)
    set_optimizer_attribute(method.model, "MSK_DPAR_SEMIDEFINITE_TOL_APPROX", 1e-12)

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
            opts.restartFactor = 2
        end
        
        if opts.maxEpochSize <= opts.minEpochSize
            @warn "Invalid values for minEpochSize and maxEpochSize. Setting to default values: minEpochSize = 20, maxEpochSize = 100"
            opts.minEpochSize = 20
            opts.maxEpochSize = 100
        end
    end
end

# --- Perform single iteration of ASPGM/BSPGM ---
function update(method::ASPGM, oracle)

    opts = method.options
    flags = method.flags
    k = opts.memorySize_SP

    # Sanity check for bad behavior in gradients. If infinite values or NaNs, terminate algorithm
    if sum(isnan.(method.gradient)) + sum(isinf.(method.gradient)) > 0
        @warn "NaNs/Infinite values encountered in gradient. Terminating."
        flags.flag_Terminate = true
        return method.x + method.x0
    end

    # If τ is full, τ_{n-k-1} (taus[1]) is the only nonzero component of τ, and the new τ_n = 0, then:
        # If k>1, we remove index 2 (corresponding to τ_{n-k}) from our data storage before pushing on new data
        # Else if k==1, we ignore the new data
    # This ensures that the nonzero τ_{n-k-1} will be preserved. See Remark 1 in the paper.
    pushNewData = true
    if (method.newTau==0)&&(findlast(method.taus .!= 0) == 1)&&(length(method.taus) == k)
        if k > 1
            purgeIndex(method, 2)
            pushNewData = true
        else
            pushNewData = false
        end
    end

    # --- Update variables with new data ---
    if pushNewData
        # Update τ with τ_{n-1}
        method.taus = push_over(method.taus, method.newTau, k)

        # Update Δ with Δ_{n-1}
        method.Δ = push_over(method.Δ, method.newDelta, k)

        # Update LPrev with L_{n-1}
        method.LPrev = push_over(method.LPrev, method.L, k) # Note: At this point, method.L stores L_{n-1}

        # Update G with g_{n-1}
        method.G = push_over_mat(method.G, method.gradient, k)

        # Determine z_n
        if flags.flag_NullStep
            z = method.x0 - method.x0  # If last step was null step,  set z_n = x0 (but with the same shift applied, so z_n = 0)
        else
            z = method.zprime - method.psi_prev/method.L * method.gradient  # Otherwise, use standard update z_n = z' - ψ_{n-1}/L_{n-1} g_{n-1}
        end
        
        # Update Z with L_{n-1}*(z_n - x_0)
        method.Z = push_over_mat(method.Z, method.L*z, k)

        # Calculate new Z/G dot products
        vec_z = ApplyHessianApprox(method.L*z, method.S, method.Y)
        vec_zi_z = method.Z'*vec_z
        vec_gi_z = method.G'*vec_z
        vec_g = ApplyHessianApprox(method.gradient, method.S, method.Y)
        vec_zi_g = method.Z'*vec_g
        vec_gi_g = method.G'*vec_g

        # Build new [Z'Z  -Z'G; -G'Z  G'G] matrix for efficient storage for subproblem
        method.MAT = build_new_MAT(method.MAT, opts.memorySize_SP, vec_zi_z, vec_zi_g, vec_gi_z, vec_gi_g)

        # Update w with f_{n-1} - ⟨g_{n-1}, x_{n-1}⟩
        method.w = push_over(method.w, method.value - dot_B(method.gradient, method.x, method.S, method.Y), k)

        # Update h with tau_{n-1}*(f_{n-1} - 1/(2L_{n-1})*||g_{n-1}||^2 + L_{n-1}/2*||z_n||^2)
        # We already calculated these norms, so we can pull from the vectors above: ||g_{n-1}||^2 = vec_gi_g[end] and ||z_n||^2 = vec_zi_z[end]/L{n-1}^2
        method.h = push_over(method.h, method.taus[end]*(method.value - 1/(2*method.L)*vec_gi_g[end]) + 1/(2*method.L)*vec_zi_z[end], k)

        # Update F with f_{n-1}
        method.F = push_over(method.F, method.value, k)

        # Update X with x_{n-1}
        method.X = push_over_mat(method.X, method.x, k)
    end

    # Now update method.L to use L_n for generating the next step
    method.L = method.newL

    xTest, idx_m, phi, psi, tau, zprime, deltaSum, success, model = generateNextIterate(method)

    # Calculate f, g for our new test point
    fTest, gTest = preconditionedOracle(method, oracle, xTest + method.x0) # Add back shift when calling oracle
    method.oracleCtr += 1

    # Calculate L for Q_{m,n}
    LReq, wExpression = calculateL(xTest, fTest, gTest, method.X[:,idx_m], method.F[idx_m], method.G[:,idx_m], method.S, method.Y, method.L)

    if LReq <= method.L
        # This means Q_{m,n} >= 0, so L_n is valid and the new hypothesis U_n is valid
        
        flags.flag_NullStep = false # Set null step flag to false

        # Save data to method
        method.zprime = zprime  # Used to generate z_n in the next iteration
        method.psi_prev = psi   # Used to generate z_n in the next iteration

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

        # Do not update zprime or psi_prev because we will force z_n = x_0 in the next update step

    end

    # Update μ estimate (regardless of whether this iteration was null step or serious step)
    muEst = calculateMu(wExpression, xTest, method.X[:,idx_m], method.S, method.Y)
    method.mu = min(method.mu, muEst)

    # Save x_n, f_n, g_n
    method.x = xTest
    method.value = fTest
    method.gradient = gTest

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
            #       3)  τ' >= c*[ L_n/mu + Δ_n/(f_0 - f_n) ], typically with c = 2
            condition = (!flags.flag_NullStep)&&(method.epochIter >= opts.minEpochSize)&&( phi >= opts.restartFactor*(method.L/method.mu + method.newDelta/(method.f0 - method.value) ) )

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

    return method.x + method.x0     # Add back x0 shift when returning x_n

end

# --- Generate x_n by solving subproblem ---
function generateNextIterate(method::ASPGM)

    # Calculate index m, but only include valid indices (where τ is nonzero)
    idx = (method.taus .== 0)
    V = method.F .- 1/(2*method.L)*diag(method.MAT)[length(method.taus)+1:end]    # Pull ||g_i||^2 from (the second half of) the diagonal of MAT
    V[idx] .= Inf    # Set invalid indices to inf so they are excluded
    v_m, idx_m = findmin(V)   # Calculate m and v_m

    rho, gamma, delta, success, status, model = optimize_rho_gamma(method, v_m)

    # Set ϕ_n equal to subproblem optimal value (equivalently: τ')
    phi = dot(rho, method.taus) + sum(gamma)

    # Set ψ_n according to standard OBL Update (equivalently: τ_n - τ')
    if method.flags.flag_FinalStep
        psi = sqrt(phi)
    else
        psi = (1+sqrt(1+8*phi))/2
    end

    # Set z', τ_n, x_n, and Δ_n using subproblem solution
    zprime = 1/method.L * method.Z * rho - 1/method.L * method.G * gamma  
    tau = phi + psi
    x = (psi * zprime + phi * (method.X[:,idx_m] - method.G[:,idx_m]/method.L))/tau
    deltaSum = dot(rho, method.Δ) + delta

    return x, idx_m, phi, psi, tau, zprime, deltaSum, success, model

end

# --- Solve subproblem to find optimal ρ, γ, τ ---
function optimize_rho_gamma(method, v_m)
    
    l = length(method.taus)

    rhocoeffs = method.h - v_m*method.taus
    gammacoeffs = method.w .- v_m
    
    # Get index of last serious step
    idx_s = findlast(method.taus .!= 0)

    # Safeguard to prevent huge jumps in rho and gamma - we enforce rho,gamma < 1/tol
    tol = min(1e-8, 1e-3/method.taus[idx_s]) # This allows new tau to be very large if the previous one was very large, but prevents huge jumps

    # Calculate δ = L_n τ_s(1/L_s^2 - 1/L_n^2)*1/2||g_s||^2
    delta = method.L*method.taus[idx_s]*(1/method.LPrev[idx_s]^2 - 1/method.L^2)*1/2*method.MAT[l+idx_s, l+idx_s]     #Note: MAT[l+idx_s,l+idx_s] = ||g_s||^2

    # Prep variables to pass into SOCP 
    A = 1/method.L*method.MAT               # Rescale to account for different Z and G definitions
    b = vcat(rhocoeffs, gammacoeffs)
    c = vcat(method.taus, ones(l))
    d = delta
    fixedIdx = findall(method.taus == 0)        # For any i where τ_i = 0, we will fix ρ_i = 0

    y, success, status, model = solveSOCP(A, b, c, d, fixedIdx, method.model; tol=tol)

    rho, gamma = clean_solution(success, y, method.taus, method.L, method.LPrev, idx_s)

    return rho, gamma, delta, success, status, model   
end

# --- Solve SOCP given by ---
# max ⟨c, y⟩
# s.t. 1/2*y'*A*y <= ⟨b, y⟩ + d
#      y >= 0
#      y[fixedIdx] = 0      # Corresponds to fixing ρ_i = 0 when τ_i = 0
function solveSOCP(A, b, c, d, fixedIdx, model; tol=1e-8)
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
    lams = eigvals(A)
    minEval = minimum(lams)
    shift = 1e-10 + (minEval < 0)*abs(minEval)*abs(maximum(lams))
    A = A + shift*Matrix(I,K,K)

    # Clear out Mosek model, and repopulate it. This is faster than constructing a new one every time
    empty!(model)
    set_silent(model)

    # Include tolerance for numerical stability
    @variable(model, 1/tol >= y[1:K] >= 0)

    # Force ρ_i = 0 when τ_i = 0, since the U_i hypothesis is vacuous in that case anyway
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

    return value.(y), success, termination_status(model), model
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
function restart(method::ASPGM, oracle)

    opts = method.options
    t = opts.memorySize_QN

    x0 = method.x + method.x0

    # We already calculated value and gradient for the current x, but we have to re-transform the gradient to match our new preconditioner.
    method.value = method.value     # Use value unchanged
    gradientTemp = ApplyHessianApprox(method.gradient, method.S, method.Y) # First, un-transform from the old S,Y
    
    # We have to un-transform the G's to build our new preconditioner Y
    GTemp = zeros(size(method.G))
    for i in axes(GTemp,2)
        GTemp[:,i] = ApplyHessianApprox(method.G[:,i], method.S, method.Y)
    end
    # No transformation necessary for the X's
    
    # Reset quasi-Newton memory storage S and Y 
    method.S = Array{Float64}(undef, length(method.x), 0)
    method.Y = Array{Float64}(undef, length(method.x), 0)

    # Push over new values to preconditioner S: x_{n-t+1}-x_{n-t}, ..., x_n-x_{n-1}
    XDiffs = hcat(method.X[:,2:end], method.x) .- method.X
    method.S = push_over_mat(method.S, XDiffs, t)

    # Push over new (un-transformed) values to preconditioner Y: g_{n-t+1}-g_{n-t}, ..., g_n-g_{n-1}
    GDiffs = hcat(GTemp[:,2:end], gradientTemp) .- GTemp
    method.Y = push_over_mat(method.Y, GDiffs, t)

    # Retransform gradient with the new preconditioners S, Y
    method.gradient = ApplyInverseHessianApprox(gradientTemp, method.S, method.Y) 

    # Save new initial values for this epoch
    method.x0 = x0
    method.f0 = method.value
    method.x = x0 - x0

    # Initialize various storage vectors and matrices
    method.Z = Array{Float64}(undef, length(x0), 0)
    method.G = Array{Float64}(undef, length(x0), 0)
    method.X = Array{Float64}(undef, length(x0), 0)
    method.F = []
    method.MAT = Array{Float64}(undef, 0, 0)
    method.zprime = x0 - x0
    method.psi_prev = 1.0    
    method.taus = []      
    method.h = []
    method.w = []
    method.LPrev = []
    method.Δ = []

    method.newTau = 1.0
    method.newDelta = 0.0

    method.epochIter = 0
    method.numEpochs += 1

    method.flags = flags()

    method.mu = Inf

    LInit = getLInit(method, oracle)
    method.L = LInit
    method.newL = LInit

end

# --- Calculate initial value L_0 for an epoch, based on perturbing x0 and measuring smoothness ---
function getLInit(method::ASPGM, oracle)

    y = method.x0 + 1e-4*randn(length(method.x0))
    fy, gy = preconditionedOracle(method, oracle, y)
    LInit, _ = calculateL(method.x0, method.value, method.gradient, y, fy, gy, method.S, method.Y, 0.01)    # Set default value L=0.01 in case of numerical error
    method.oracleCtr += 1

    return LInit

end

# --- Calculate smallest L value such that cocoercivity holds (Q_{m,n}(L) = 0) ---
function calculateL(x_n::Vector{Float64}, f_n::Float64, g_n::Vector{Float64},  x_m::Vector{Float64}, f_m::Float64, g_m::Vector{Float64}, S::Matrix{Float64}, Y::Matrix{Float64}, L::Float64)

    denom = f_m - f_n - dot_B(g_n, x_m - x_n, S, Y)     # Calculate denominator separately - we can reuse it in calculating mu to save us a dot product
    gDiffNorm = norm_B(g_n - g_m, S, Y)^2
    LEst = gDiffNorm/(2*denom)
    LEst = abs(LEst)    # Sanity check

    # Special handling for 0/0 cases. Then we can check Q_{m,n} directly
    if isnan(LEst)||isinf(LEst)
        Q = denom - gDiffNorm/(2L)
        if (Q >= 0)
            LEst = L    # If Q_{m,n} >= 0, then reuse current value of L.
        else
            LEst = 2*L  # Else if Q_{m,n} < 0, double L.
        end
    end

    return LEst, denom

end

# --- Calculate largest μ value such that strong convexity holds ---
function calculateMu(wExpression::Float64, x_n::Vector{Float64}, x_m::Vector{Float64}, S::Matrix{Float64}, Y::Matrix{Float64})

    muEst = 2*wExpression/norm_B(x_n - x_m, S, Y)^2
    muEst = abs(muEst) # Sanity check

    return muEst
end

# --- Preconditioned dot product, using S,Y as quasi-Newton preconditioners ---
function dot_B(v, w, S::Matrix{Float64}, Y::Matrix{Float64})
    if isempty(S)
        return dot(v, w)    # If S,Y are empty, return standard dot product
    else
        return dot(v, ApplyHessianApprox(w, S, Y))  # If S,Y are populated, return ⟨v, B w⟩ where B is the qausi-Newton preconditioner formed using S and Y
    end
end

# --- Preconditioned norm, using S,Y as quasi-Newton preconditioners ---
function norm_B(v::Vector, S::Matrix{Float64}, Y::Matrix{Float64})
    return sqrt(abs(dot_B(v, v, S, Y)))     # Take absolute value in case of numerical error
end

# --- Calculate B^{-1} v where B^{-1} is the qausi-Newton Hessian approximation formed using S and Y. Follows the standard method of Byrd, Nocedal and Schnabel (1994) ---
function ApplyHessianApprox(v::Vector{Float64}, S::Matrix{Float64}, Y::Matrix{Float64})

    if isempty(S)
        return v
    end

    t = size(S, 2)

    theta = dot(Y[:, end], Y[:, end]) / dot(S[:, end], Y[:, end])

    D = diagm([dot(S[:, i], Y[:, i]) for i in 1:t])
    T = zeros(t, t)
    for i in 1:t
        for j in 1:i-1
            T[i, j] = dot(S[:, i], Y[:, j])
        end
    end

    M = zeros(2t, 2t)
    M[1:t, 1:t] = theta*(S'*S)
    M[1:t, t+1:2t] = T
    M[t+1:2t, 1:t] = T'
    M[t+1:2t, t+1:2t] = -D

    vec = vcat(theta*(S' * v), Y'*v)

    q = zeros(size(vec))
    try
        q = M \ vec
    catch
        # M should not be singular, but add handling in case of numerical error
        return v
    end

    r = theta * v - (theta * S * q[1:t] + Y * q[t+1:end])
    return r
end

# --- Calculate B v where B is the qausi-Newton inverse Hessian approximation formed using S and Y. Follows the standard two-loop method of Nocedal (1980) ---
function ApplyInverseHessianApprox(v::Vector{Float64}, S::Matrix{Float64}, Y::Matrix{Float64})

    if isempty(S)
        return v
    end

    t = size(S,2)
    alpha = zeros(t)
    eta = zeros(t)

    q = v

    for i = t:-1:1
        eta[i] = 1.0 / dot(Y[:,i], S[:,i])
        alpha[i] = eta[i] * dot(S[:,i], q)
        q = q - alpha[i] * Y[:,i]
    end

    r = dot(S[:,end], Y[:,end]) / dot(Y[:,end], Y[:,end]) * q

    for i = 1:t
        beta = eta[i] * dot(Y[:,i], r)
        r = r + S[:,i] * (alpha[i] - beta)
    end

    return r
end

# --- Calls function/gradient oracle then applies preconditioning to the returned gradient. Also checks if ||g|| < g_tol ---
function preconditionedOracle(method::ASPGM, oracle, x::Vector{Float64})

    if isempty(method.S)
        value, gradient = oracle(x)
        gNorm = norm(gradient, Inf)
        if gNorm < method.options.g_tol
            print("\nGradient tolerance met: ||g_n||_inf = ", gNorm)
            method.flags.flag_Terminate = true
        end
        return value, gradient
    else
        value, gTemp = oracle(x)
        gNorm = norm(gTemp, Inf)
        if gNorm < method.options.g_tol
            print("\nGradient tolerance met: ||g_n||_inf = ", gNorm)
            method.flags.flag_Terminate = true
        end
        gradient = ApplyInverseHessianApprox(gTemp, method.S, method.Y)
        return value, gradient
    end

end

# --- Push v into a vector x, taking the tail if the size grows over k ---
function push_over(x,v,k)
    push!(x, v)
    if length(x) > k
        return x[end-k+1:end]
    else
        return x
    end
end

# --- Push column vector v into a matrix X, taking the tail columns if the size grows over k ---
function push_over_mat(X,v,k)
    X = hcat(X, v)
    if size(X)[end] > k
        X = X[:, end-k+1:end]
    end
    return X
end

# --- Push vector v into the bottom row and right column of a symmetric matrix X, taking the tail principal submatrix if the size grows over k ---
#
#     X  --->  X[2:end, 2:end]   v[1:end-1]
#              v[1:end-1]'       v[end]
#
function push_over_principal_submatrix_sym(X,v,k)
    if size(X,2)==0
        X = v*Matrix(I,1,1)
        return X
    elseif size(X)[end] >= k
        X = X[2:end,2:end]
    end
    X = hcat(vcat(X, v[1:end-1]'), v)
    return X
end

# --- Append vectors to the outer layer of a matrix X, taking the tail principal submatrix if the size grows over k ---
#
#     X  --->  X[2:end, 2:end]   v_right
#              v_bottom'          v_corner
#
function push_over_principal_submatrix(X,v_right,v_bottom,v_corner,k)
    if size(X,2)==0
        X = v_corner*Matrix(I,1,1)
        return X
    elseif size(X)[end] >= k
        X = X[2:end,2:end]
    end
    X = hcat(vcat(X, v_bottom), vcat(v_right, v_corner))
    return X
end

# --- Update MAT = [Z'Z  -Z'G; -G'Z  G'G] with new values. This is done by appending to the different submatrices ---
function build_new_MAT(MAT, k, vec_zi_z, vec_zi_g, vec_gi_z, vec_gi_g)
    i = Int(size(MAT,1)/2)

    ZZ = MAT[1:i,1:i]
    ZZ = push_over_principal_submatrix_sym(ZZ, vec_zi_z, k)

    GG = MAT[i+1:2i, i+1:2i]
    GG = push_over_principal_submatrix_sym(GG, vec_gi_g, k)

    GZ = -MAT[i+1:2i, 1:i]
    GZ = push_over_principal_submatrix(GZ, vec_gi_z[1:end-1], vec_zi_g[1:end-1]', vec_gi_z[end], k)

    ZG = GZ'

    MAT = vcat(hcat(ZZ, -ZG), hcat(-GZ, GG))

    return MAT

end

# --- Remove specified index from our stored variables. This is used in the special case that τ is full and τ_{n-k-1} (taus[1]) is the only nonzero component of τ. See Remark 1 in the paper ---
function purgeIndex(method::ASPGM, removeIdx::Int64)

    idx = setdiff(1:size(method.G,2), removeIdx)

    method.G = method.G[:,idx]
    method.Z = method.Z[:,idx]
    method.X = method.X[:,idx]
    method.w = method.w[idx]
    method.h = method.h[idx]
    method.F = method.F[idx]
    method.LPrev = method.LPrev[idx]
    method.taus = method.taus[idx]
    method.Δ = method.Δ[idx]

    idx = setdiff(1:size(method.MAT,2), [removeIdx, removeIdx + size(method.G,2)])
    method.MAT = method.MAT[idx, idx]

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
