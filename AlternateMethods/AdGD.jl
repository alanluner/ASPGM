include("fom_interface.jl")

# This code implements the improved adaptive gradient descent (AdGD) method by Yura Malitsky and Konstantin Mischchenko.
# See their paper at https://arxiv.org/abs/2308.02261

mutable struct AdGD <: FOM
end


function runMethod(method::AdGD, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    i = 0
    exit = false
    metaData = Vector{Float64}[]

    theta = 1.0 / 3

    x = copy(x0)
    g = zeros(size(x0))
    x_prev = copy(x)
    g_prev = zeros(size(x0))
    tmp = zeros(size(x0))


    f = oracle(g, x)

    gNorm = norm(g)
    c = 1e-4
    @. tmp = x - c/gNorm*g
    f_prev = oracle(g_prev, tmp)

    @. tmp = g - g_prev
    L0 = dot(tmp, tmp) / (2 * (f_prev - f + c*gNorm))
    if isnan(L0)||(L0==0)
        L0 = 0.01
    end

    if saveDetails
        push!(metaData, [f, norm(g)])
        push!(metaData, [f_prev, norm(g_prev)])
    end

    a_prev = 1/L0
    copyto!(g_prev, g)
    copyto!(x_prev, x)

    @. x = x_prev - a_prev*g_prev
    f = oracle(g, x)

    if saveDetails
        push!(metaData, [f, norm(g)])
    end

    while !exit

        @. tmp = g - g_prev
        gDiffNorm = norm(tmp)

        @. tmp = x - x_prev
        xDiffNorm = norm(tmp)

        L = gDiffNorm / xDiffNorm
        a = min(sqrt(2 / 3 + theta) * a_prev, a_prev / sqrt(max(2 * a_prev^2 * L^2 - 1, 0)))

        theta = a / a_prev
        
        copyto!(x_prev, x)
        copyto!(g_prev, g)
        a_prev = a

        @. x = x - a * g
        
        f = oracle(g, x)
        if saveDetails
            push!(metaData, [f, norm(g)])
        end

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


function methodTitle(method::AdGD)
    return "AdGD"
end