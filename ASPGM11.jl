#========================================================================
This routine is specifically for ASPGM with memory size 1. It uses an explicit calculation to determine the next iterate, rather than solving an SOCP subproblem.
========================================================================#

#========================================================================
This code implements the Adaptive Subgame Perfect Gradient Method (ASPGM11). At each iteration, ASPGM11 makes a momentum-type update,
optimized dynamically based on a (limited) memory/bundle of past first-order information. It is linesearch-free, parameter-free, and adaptive.

See https://github.com/alanluner/ASPGM for implementation details.

See https://arxiv.org/pdf/2510.21617 for the paper that introduces the method.
========================================================================#
#========================================================================
Notes:
- This implementation is written to be highly performant for large-scale problems. To accomplish this, we repeatedly use pre-allocated vectors, in-place vector calculation,
views, and more. Any future updates should be careful in their vector handling to maintain this behavior. In particular, any changes to the pre-allocated vectors must be done very carefully.
========================================================================#

using Parameters

# Options for ASPGM11
@with_kw mutable struct options_ASPGM11
    mode::Symbol = :A                           # :A - ASPGM11 (with restarting and preconditioning)   [ vs :B - BSPGM (no restarting or preconditioning) ]
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

mutable struct ASPGM11
    # ------------- Initial values ------------- #
    x0::Vector{Float64}                 # Starting point for this epoch
    f0::Float64                         # Stores initial f(x0) for this epoch
    # ------------- Current iterate data ------------- #
    x::Vector{Float64}                  # Iterate x_n
    value::Float64                      # Function value f_n
    gradient::Vector{Float64}           # Gradient g_n
    L::Float64                          # Smoothness estimate
    mu::Float64                         # Strong convexity estimate
    newDelta::Float64                   # Error parameter Δ_n
    newTau::Float64                     # Convergence guarantee τ_n
    newL::Float64                       # Smoothness estimate for next iterate L_{n+1}
    # ------------- Pre-Allocation ------------- #
    inputVec::Vector{Float64}           # Pre-allocated vector to be used as input argument for functions in this routine. Should be used as argument immediately after its value is set.
    outputVec::Vector{Float64}          # Pre-allocated vector to be used as output argument (modified in-place) for functions in this routine. Its value should be used immediately after receiving output.
    # ------------- Memory Storage - Subgame Perfect ------------- #
    z_prev::Vector{Float64}             # Past auxiliary sequence z_{n-1}
    g_prev::Vector{Float64}             # Past gradient g_{n-1}
    x_prev::Vector{Float64}             # Past iterate x_{n-1}
    f_prev::Float64                     # Past function value f_{n-1}
    zprime::Vector{Float64}             # Optimal auxiliary step z' as determined by subproblem
    psi_prev::Float64                   # ψ_{n-1} = τ_{n-1} - τ'
    tau_prev::Float64                   # τ_{n-1}
    L_prev::Float64                     # L_{n-1}
    Delta_prev::Float64                 # Δ_{n-1}
    # ------------- Preconditioning Memory Storage ------------- #
    S::Vector{Float64}                  # Quasi-Newton matrix formed from iterate differences: (x_{n-t+1}-x_{n-t}, ..., x_{n}-x_{n-1})  
    Y::Vector{Float64}                  # Quasi-Newton matrix formed from gradient differences: (g_{n-t+1}-g_{n-t}, ..., x_{n}-x_{n-1})  
    M::Union{Nothing,Matrix{Float64}}   # 2x2 matrix used for preconditioning (see ApplyHessianApprox), it is saved off to remove redundant calculations
    theta::Float64                      # Scalar value used for precondntioning (see ApplyHessianApprox), it is saved off to remove redundant calculations
    # ------------- Metadata ------------- #
    iteration::Int64                    # total iteration number
    epochIter::Int64                    # iteration number within this epoch
    numEpochs::Int64                    # number of epochs
    oracleCtr::Int64                    # number of oracle calls
    flags::flags          # Set of flags for branching behavior according to the algorithm
    # ------------- Implementation Options ------------- #
    options::options_ASPGM11      # (See below)
end


# --------Method Constructors-------- #

ASPGM11() = ASPGM11(Float64[],0.0,
                                Float64[],0.0,Float64[],0.0,0.0,0.0,0.0,0.0,
                                Float64[],Float64[],
                                Float64[],Float64[],Float64[],0.0,Float64[],0.0,0.0,0.0,0.0,
                                Float64[],Float64[],nothing,0.0,
                                0,0,0,0,flags(),
                                options_ASPGM11(:A, 2.0, 20, 100, -1, -1))

BSPGM1() = ASPGM11(Float64[],0.0,
                                Float64[],0.0,Float64[],0.0,0.0,0.0,0.0,0.0,
                                Float64[],Float64[],
                                Float64[],Float64[],Float64[],0.0,Float64[],0.0,0.0,0.0,0.0,
                                Float64[],Float64[],nothing,0.0,
                                0,0,0,0,flags(),
                                options_ASPGM11(:B, -1.0, -1, Inf, -1, -1))

# ----------------------------------- #

# Run ASPGM11 for a certain amount of time/oracle calls

# Arguments:
# - method: ASPGM11 object 
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
# runMethod(ASPGM11(), oracle, zeros(d); oracleCalls = 500)
#
function runMethod(method::ASPGM11, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

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
    if method.value < method.f_prev
        x = method.x + method.x0
        val = method.value
    else
        x = method.x_prev + method.x0
        val = method.f_prev
    end

    # Convert from list of vectors to matrix
    if !isempty(metaData)
        metaData = reduce(vcat, transpose.(metaData))
    end

    return x, val, metaData
end


# --- Initialize variables of the method and validate settings ---
function initialize(method::ASPGM11, oracle, x0::Vector{Float64})

    # Validate settings
    validate(method)

    d = length(x0)

    # Save initial values
    method.x0 = copy(x0)
    method.gradient = zeros(d)
    method.value = oracle(method.gradient, method.x0)
    method.f0 = method.value
    method.x = zeros(d) # We implement a shift so that we can assume that x0 = 0

    # Allocate memory for scratch vector
    method.inputVec = zeros(d)
    method.outputVec = zeros(d)

    # Initialize various storage vectors and matrices
    method.z_prev = zeros(d)
    method.g_prev = zeros(d)
    method.x_prev = zeros(d)
    method.f_prev = Inf
    method.zprime = zeros(d)
    method.psi_prev = 1.0    
    method.tau_prev = 0.0      
    method.L_prev = 0.0
    method.Delta_prev = 0.0

    method.newTau = 1.0     # Set τ_0 = 1
    method.newDelta = 0.0   # Set Δ_0 = 0

    # Initialize quasi-Newton matrices
    method.S = zeros(d)
    method.Y = zeros(d)
    method.M = nothing
    method.theta = 1.0

    # Iteration counters
    method.iteration = 0
    method.epochIter = 0
    method.numEpochs = 1
    method.oracleCtr = 1 # Count initial oracle call on x0

    method.flags = flags()

    # Initialize strong convexity estimate μ
    method.mu = Inf     

    # Initialize smoothness estimate L0
    LInit = getLInit(method, oracle)
    method.L = LInit
    method.newL = LInit

end

# --- Validate settings ---
function validate(method::ASPGM11)
    opts = method.options
    if (opts.mode==:A)
        
        if opts.restartFactor <= 1
            @warn "Invalid value for restartFactor. Must be larger than 1. Updating value to default: 2"
            opts.restartFactor = 2.0
        end
        
        if opts.maxEpochSize <= opts.minEpochSize
            @warn "Invalid values for minEpochSize and maxEpochSize. Setting to default values: minEpochSize = 20, maxEpochSize = 100"
            opts.minEpochSize = 20
            opts.maxEpochSize = 100
        end
    end
end

# --- Perform single iteration of ASPGM11/BSPGM11 ---
function update(method::ASPGM11, oracle)

    opts = method.options
    flags = method.flags

    # Sanity check for bad behavior (NaNs, Infs) in gradients
    if any(!isfinite, method.gradient)
        if isnothing(method.M)
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

    # If this was a null step (i.e. τ_n = 0), then we ignore the new data
    # This ensures that the nonzero τ_{n-1} will be preserved. See Remark 1 in the paper.
    pushNewData = !flags.flag_NullStep

    # --- Update variables with new data ---
    if pushNewData
        # Update τ_{n-1}
        method.tau_prev = method.newTau

        # Update Δ_{n-1}
        method.Delta_prev = method.newDelta

        # Update L_{n-1}
        method.L_prev = method.L # Note: At this point, method.L stores L_{n-1}

        # Update g_{n-1}
        method.g_prev .= method.gradient

        # Update z_n
        if flags.flag_NullStep
            method.z_prev .= 0  # If last step was null step,  set z_n = x0 (but with the same shift applied, so z_n = 0)
        else
            @. method.z_prev = method.zprime - method.psi_prev/method.L*method.gradient  # Otherwise, use standard update z_n = z' - ψ_{n-1}/L_{n-1} g_{n-1}
        end

        # Update f_{n-1}
        method.f_prev = method.value

        # Update x_{n-1}
        method.x_prev .= method.x
    end

    # Now update method.L to use L_n for generating the next step
    method.L = method.newL

    # --- Generate next iterate --- 
    idx_m, phi, method.psi_prev, tau, deltaSum, success = generateNextIterate_Mem1!(method.x, method.zprime, method)

    # Calculate f, g for our new iterate
    @. method.inputVec = method.x + method.x0      # Add back shift when calling oracle, use working vector
    method.value = preconditionedOracle!(method.gradient, method, oracle, method.inputVec)    # Call oracle and save results into method.value and method.gradient
    method.oracleCtr += 1

    # Calculate L for Q_{m,n} (and additional values for later)
    LReq, wExpression, distSq = calculateL(method) 

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

# --- If memory size is 1, then we solve the subproblem exactly to generate x_n ---
function generateNextIterate_Mem1!(dest_x::Vector{Float64}, dest_zprime::Vector{Float64}, method::ASPGM11)

    L_n = method.L
    L_minus = method.L_prev
    tau_minus = method.tau_prev

    gammaVals = zeros(4)
    rhoVals = zeros(4)
    objVals = zeros(4)

    dot_z_g_minus = dot_B(method.z_prev, method.g_prev, method)
    dot_g_x_minus = dot_B(method.g_prev, method.x_prev, method)
    z_minus_normSq = normSq_B(method.z_prev, method)      
    g_minus_normSq = normSq_B(method.g_prev, method)                

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
    @. dest_zprime = L_minus/L_n*rho*method.z_prev - 1/L_n*gamma*method.g_prev
    tau = phi + psi
    @. dest_x = psi/tau*dest_zprime + phi/tau*method.x_prev - phi/(tau*L_n)*method.g_prev
    deltaSum = rho*method.Delta_prev + delta

    return idx_m, phi, psi, tau, deltaSum, success
end

# --- Restart method at current iterate (start new epoch), re-initialize method variables accordingly ---
# This requires several instances of "un-transforming" gradients. Recall that the gradients stored in method.G, etc. are the gradients with respect to 
# the ⟨⋅,⋅⟩_B dot product. To build a new preconditioner (among other things), we have to convert to the true Euclidean gradient - i.e. "un-transforming"
function restart(method::ASPGM11, oracle)

    @. method.x0 = method.x + method.x0        # Set new x0, and add back x0 shift from previous epoch

    # We have to un-transform the g_prev before we can build our new preconditioner Y
    ApplyHessianApprox!(method.g_prev, method.g_prev, method)
    # No transformation necessary for the X's

    # Similarly, we will have to un-transform the gradient (and later re-transform it to match our new preconditioner)
    method.value = method.value     # Use value unchanged
    ApplyHessianApprox!(method.outputVec, method.gradient, method) # First, un-transform from the old S,Y
    
    # Reset quasi-Newton memory storage S and Y
    fill!(method.S, 0.0)
    fill!(method.Y, 0.0)
    method.M = nothing
    method.theta = 1.0

    # If preconditioning is still enabled, set new S and Y (and M)
    if !method.flags.flag_DisablePrec
        # Pass new value to preconditioner S: x_n-x_{n-1}
        @. method.S = method.x - method.x_prev

        # Pass new (un-transformed) value to preconditioner Y: g_n-g_{n-1}
        @. method.Y = method.outputVec - method.g_prev

        # Calculate and save off M and theta for efficient preconditioning calculations
        method.M, method.theta = buildPrecM(method)
    else
        # Otherwise, if preconditioning is disabled recalculate gradient without preconditioner (since S, Y are empty)
        # (We need to recalculate because method.gradient must have had NaNs/Infs, so we cannot just un-transform)
        _ = preconditionedOracle!(method.outputVec, method, oracle, method.x0)
        method.oracleCtr += 1
    end

    # Re-transform our un-transformed gradient (gTemp), using the new preconditioners S, Y
    method.inputVec .= method.outputVec         # Transfer gTemp from output to input
    ApplyInverseHessianApprox!(method.gradient, method.inputVec, method) 

    # Save new initial values for this epoch
    method.f0 = method.value
    fill!(method.x, 0.0)              # Implement shift for this epoch so that we can assume that x0 = 0

    # Initialize various storage vectors and matrices
    fill!(method.z_prev, 0.0)
    fill!(method.g_prev, 0.0)
    fill!(method.x_prev, 0.0)
    method.f_prev = Inf
    fill!(method.zprime, 0.0)
    method.psi_prev = 1.0    
    method.tau_prev = 0.0      
    method.L_prev = 0.0
    method.Delta_prev = 0.0

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
function getLInit(method::ASPGM11, oracle)

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
function calculateL(method::ASPGM11)

    @. method.inputVec = method.gradient - method.g_prev
    gDiffNormSq = normSq_B(method.inputVec, method)

    @. method.inputVec = method.x_prev - method.x
    denom = method.f_prev - method.value - dot_B(method.gradient, method.inputVec, method)     # Calculate denominator separately - we can reuse it in calculating mu to save us a dot product
    distSq = normSq_B(method.inputVec, method)    # Calculate ||x_m-x_n||^2 now for use later - since x_m-x_n is already calculated

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

# --- Preconditioned dot product, using S,Y as quasi-Newton preconditioners (via the precalculated M) ---
function dot_B(v::Vector{Float64}, w::Vector{Float64}, method::ASPGM11)
    if isnothing(method.M)
        return dot(v, w)    # If M preconditioner is not set, return standard dot product
    end

    m11 = method.M[1,1]
    m22 = method.M[2,2]

    if m11 * m22 == 0
        return dot(v, w)
    end
    
    vw = dot(v, w)

    # Reuse dot products if v == w
    vS = dot(v, method.S)
    vY = dot(v, method.Y)
    if v === w
        Sw = vS
        Yw = vY
    else
        Sw = dot(method.S, w)
        Yw = dot(method.Y, w)
    end

    # res = θ*v'*w - ([θ*S Y]*q)'*w
    return method.theta * vw - (method.theta^2 / m11) * vS * Sw - (1.0 / m22) * vY * Yw
end

# --- Preconditioned norm squared, using S,Y as quasi-Newton preconditioners ---
function normSq_B(v::Vector{Float64}, method::ASPGM11)
    return abs(dot_B(v, v, method))     # Take absolute value in case of numerical error
end

# --- Build matrix M for preconditioner calculations and save it off; this speeds up repeated Mx=y solves ---
function buildPrecM(method)
    yy = dot(method.Y, method.Y)
    sy = dot(method.S, method.Y)
    ss = dot(method.S, method.S)
    theta = yy / sy

    if !isfinite(theta)
        return nothing, theta
    else
        M = [theta*ss  0; 0  -sy]
    end

    return M, theta
end

# --- Calculate B^{-1} v where B^{-1} is the qausi-Newton Hessian approximation formed using S and Y. Follows the standard method of Byrd, Nocedal and Schnabel (1994) ---
# The matrix M is calculated and saved off ahead of time.
function ApplyHessianApprox!(dest::Vector{Float64}, v::Vector{Float64}, method::ASPGM11)

    # If no preconditioner is set, return v
    if isnothing(method.M)
        dest .= v
        return
    end

    vec = [method.theta*dot(method.S,v), dot(method.Y,v)]

    if method.M[1,1]*method.M[2,2] == 0
        return v
    end

    # Set q = M^{-1}*vec
    q1 = vec[1] / method.M[1,1]
    q2 = vec[2] / method.M[2,2]

    # dest = θ*v - ([θ*S Y]*q)
    @. dest = method.theta * v - (method.theta * q1 * method.S + q2 * method.Y)
    return
end

# --- Calculate B*v where B is the qausi-Newton inverse Hessian approximation formed using S and Y. Follows the standard two-loop method of Nocedal (1980) ---
# The vector B*v is written into dest
function ApplyInverseHessianApprox!(dest::Vector{Float64}, v::Vector{Float64}, method::ASPGM11)

    if isnothing(method.M)
        dest .= v
        return
    end

    dest .= v

    eta = -1.0 / method.M[2,2]
    alpha = eta * dot(method.S, dest)
    @. dest = dest - alpha*method.Y

    @. dest = 1/method.theta*dest

    beta = eta * dot(method.Y, dest)
    @. dest = dest + (alpha-beta)*method.S

    return
end

# --- Calls function/gradient oracle then applies preconditioning to the returned gradient. Also checks if ||g|| < g_tol ---
# The function value is returned and the preconditioned gradient is written into dest_g
function preconditionedOracle!(dest_g::Vector{Float64}, method::ASPGM11, oracle, x::Vector{Float64})

    if isnothing(method.M)
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

# --- Convergence guarantee on (f(x_n) - f(x_0))/(1/2||x_0-x_*||^2) for ASPGM11/BSPGM. Mostly applicable for BSPGM. For ASPGM11, the value is with respect to the preconditioner B and the last restart iterate ---
function guarantee(method::ASPGM11)
    if method.newTau > 0
        return method.L/(method.newTau)
    else
        return method.L_prev/(method.tau_prev)
    end
end

# --- Method title ---
function methodTitle(method::ASPGM11)
    if method.options.mode == :B
        return "BSPGM-1"
    else
        return "ASPGM-1-1"
    end
end
