include("fom_interface.jl")

# This code is adapted from the original implementation of the Auto-Conditioned Fast Gradient Method (AC-FGM) by Tianjiao Li and Guanghui Lan.
# See their original implementation at https://github.com/tli432/AC-FGM-Implementation
# See their accompanying paper at https://arxiv.org/pdf/2310.10082

mutable struct ACFGM <: FOM
    L0
    beta
    alpha
end

ACFGM(L0) = ACFGM(L0, 1-sqrt(6)/3, 0.1)

ACFGM() = ACFGM(nothing, 1-sqrt(6)/3, 0.1)

function runMethod(method::ACFGM, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    metaData = []
    exit = false

    x = copy(x0)
    z = copy(x)
    y = copy(x)

    L0 = method.L0
    beta = method.beta
    alpha = method.alpha

    f, g = oracle(x)
    if saveDetails
        metaData = [f  norm(g)]
    end

    if isnothing(L0)
        y = x0 + 1e-4*randn(length(x0))
        fy,gy = oracle(y)
        if saveDetails
            metaData = vcat(metaData, [fy  norm(gy)])
        end
        L0 = dot(g - gy, g - gy) / (2 * (fy - f - dot(g, y-x0)))
        if isnan(L0)||(L0==0)
            L0 = 0.01
        end
    end
    
    # Determine eta_1 based on L_0
    L_new = L0
    η = 1 / (2.5 * L_new)
    initial_eta = η

    x_new = x - η*g
    f_new, g_new = oracle(x_new)
    if saveDetails
        metaData = vcat(metaData, [f_new  norm(g_new)])
    end

    x = x_new
    z = x_new # y stays same
    L_est = L_new
    f = f_new
    g = g_new

    η = initial_eta

    τ_old = 0.0
    τ = 1.0
    i = 2
    
    while !exit

        if i == 2
            η = min((1 - beta)*η, 1/(4*L_est))
            τ_old = 0.0; τ = 1.0
        else
            if L_est > 0
                η = min(4η/3, (τ_old + 1)/τ * η, τ/(4*L_est))
            end
            τ_old = τ
            τ = τ + 2*(1 - alpha)*η*L_est/τ + alpha/2
        end

        z = y - η*g
        y = (1 - beta)*y + beta*z
        x_new = (z + τ*x) / (1 + τ)
        f_new, g_new = oracle(x_new)
        if saveDetails
            metaData = vcat(metaData, [f_new  norm(g_new)])
        end

        if norm(g_new .- g) == 0
            L_est = 0.0
        else
            denom = 2*(f - f_new - dot(g_new, x - x_new))
            L_est = norm(g_new - g)^2 / denom
            if L_est < 0
                L_est = 0.0
            end
        end

        if f_new > f
            τ_old = 0.0
            τ = 1.0
            η = initial_eta
            y = x_new
        end

        f = f_new
        g = g_new
        x = x_new

        i = i+1
        t = time() - t0

        if (i > oracleCalls)&&(t >= runTime)
            exit = true
        end

    end

    return x, f, metaData
end

function methodTitle(method::ACFGM)
    return "ACFGM"
end