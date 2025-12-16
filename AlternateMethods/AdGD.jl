include("fom_interface.jl")

mutable struct AdGD <: FOM
end


function runMethod(method::AdGD, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    i = 0
    exit = false
    metaData = []

    x_prev = x0
    θ = 1.0 / 3

    y = x0 + 1e-4*randn(length(x0))
    fy,gy = oracle(y)
    f0,g0 = oracle(x0)
    L0 = dot(g0 - gy, g0 - gy) / (2 * (fy - f0 - dot(g0, y-x0)))
    if isnan(L0)||(L0==0)
        L0 = 0.01
    end

    if saveDetails
        metaData = [f0  norm(g0)]
        metaData = vcat(metaData, [fy  norm(gy)])
    end

    α_prev = 1/L0
    g_prev = g0
    x_prev = x0

    x = x0 - 1/L0*g0
    f,g = oracle(x)
    if saveDetails
        metaData = vcat(metaData, [f  norm(g)])
    end

    while !exit
        L = norm(g - g_prev) / norm(x - x_prev)
        α = min(sqrt(2 / 3 + θ) * α_prev, α_prev / sqrt(max(2 * α_prev^2 * L^2 - 1, 0)))

        θ = α / α_prev
        
        x_prev = x
        g_prev = g
        α_prev = α

        x = x - α * g
        
        f, g = oracle(x)
        if saveDetails
            metaData = vcat(metaData, [f  norm(g)])
        end

        i = i+1
        t = time() - t0

        if (i > oracleCalls)&&(t >= runTime)
            exit = true
        end
    end

    return x, f, metaData
end


function methodTitle(method::AdGD)
    return "AdGD"
end