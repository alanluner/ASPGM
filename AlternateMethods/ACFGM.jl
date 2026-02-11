include("fom_interface.jl")

# This code is adapted from the original implementation of the Auto-Conditioned Fast Gradient Method (AC-FGM) by Tianjiao Li and Guanghui Lan.
# See their original implementation at https://github.com/tli432/AC-FGM-Implementation
# See their accompanying paper at https://arxiv.org/pdf/2310.10082

mutable struct ACFGM <: FOM
    beta::Float64
    alpha::Float64
end

ACFGM() = ACFGM(1-sqrt(6)/3, 0.1)

function runMethod(method::ACFGM, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    metaData = Vector{Float64}[]
    exit = false

    x = copy(x0)
    z = copy(x)
    y = copy(x)
    g = zeros(size(x))
    x_new = copy(x)
    g_new = zeros(size(x))
    tmp = zeros(size(x))

    beta = method.beta
    alpha = method.alpha

    f = oracle(g, x)
    if saveDetails
        push!(metaData, [f, norm(g)])
    end

    gNorm = norm(g)
    c = 1e-4/gNorm
    @. y = x0 - c*g
    f_new = oracle(g_new, y)
    if saveDetails
        push!(metaData, [f_new, norm(g_new)])
    end

    @. tmp = g - g_new
    L0 = dot(tmp, tmp) / (2 * (f_new - f + 1e-4*gNorm))
    if isnan(L0)||(L0==0)
        L0 = 0.01
    end
    
    # Determine eta_1 based on L_0
    L_new = L0
    eta = 1 / (2.5 * L_new)
    initial_eta = eta

    @. x_new = x - eta*g
    f_new = oracle(g_new, x_new)
    if saveDetails
        push!(metaData, [f_new, norm(g_new)])
    end

    copyto!(x, x_new)
    copyto!(z, x_new)
    L_est = L_new
    f = f_new
    copyto!(g, g_new)
    copyto!(y, x)

    eta = initial_eta

    tau_old = 0.0
    tau = 1.0
    i = 2
    
    while !exit

        if i == 2
            eta = min((1 - beta)*eta, 1/(4*L_est))
            tau_old = 0.0; tau = 1.0
        else
            if L_est > 0
                eta = min(4eta/3, (tau_old + 1)/tau * eta, tau/(4*L_est))
            end
            tau_old = tau
            tau = tau + 2*(1 - alpha)*eta*L_est/tau + alpha/2
        end

        @. z = y - eta*g
        @. y = (1-beta)*y + beta*z
        @. x_new = 1/(1+tau)*z + tau/(1+tau)*x

        f_new = oracle(g_new, x_new)
        if saveDetails
            push!(metaData, [f_new, norm(g_new)])
        end

        @. tmp = g_new - g
        gDiffNorm = norm(tmp)
        if gDiffNorm == 0
            L_est = 0.0
        else
            @. tmp = x - x_new
            denom = 2*(f - f_new - dot(g_new, tmp))
            L_est = gDiffNorm^2 / denom
            if L_est < 0
                L_est = 0.0
            end
        end

        if f_new > f
            tau_old = 0.0
            tau = 1.0
            eta = initial_eta
            copyto!(y, x_new)
        end

        f = f_new
        copyto!(g, g_new)
        copyto!(x, x_new)

        i = i+1
        t = time() - t0

        if (i > oracleCalls)&&(t >= runTime)
            exit = true
        end

    end

    # Convert from list of vectors to matrix
    if !isempty(metaData)
        metaData = reduce(vcat, transpose.(metaData))
    end

    return x, f, metaData
end

function methodTitle(method::ACFGM)
    return "ACFGM"
end