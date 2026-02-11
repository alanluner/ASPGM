include("fom_interface.jl")

# This code implements the NAG-Free method by Joao Cavalcanti, Laurent Lessard and Ashia Wilson
# See their paper at https://arxiv.org/abs/2506.13033

mutable struct NAGF <: FOM
end


function runMethod(method::NAGF, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    i = 0
    exit = false
    metaData = Vector{Float64}[]

    x = copy(x0)

    g = zeros(size(x))
    y = zeros(size(x))
    y_new = zeros(size(x))
    x_new = zeros(size(x))
    g_new = zeros(size(x))
    tmp = zeros(size(x))

    f = oracle(g, x)

    tmp .= 1e-6*rand(length(x))
    xDiffNorm = norm(tmp)
    @. y = x + tmp
    
    f_new = oracle(g_new, y)

    if saveDetails
        push!(metaData, [f, norm(g)])
        push!(metaData, [f_new, norm(g_new)])
    end

    @. tmp = g - g_new
    gDiffNorm = norm(tmp)

    L0 = gDiffNorm / xDiffNorm
    m0 = L0

    L = L0
    m = m0

    copyto!(y_new, y)
    copyto!(x_new, x)
    f_new = f
    copyto!(g_new, g)

    while !exit
        @. y_new = x - g/L

        @. tmp = y_new - y
        c = (sqrt(L)-sqrt(m))/(sqrt(L) + sqrt(m))
        @. x_new = y_new + c*tmp

        f_new = oracle(g_new, x_new)

        if saveDetails
            push!(metaData, [f_new, norm(g_new)])
        end

        @. tmp = g_new - g
        gDiffNorm = norm(tmp)

        @. tmp = x_new - x
        xDiffNorm = norm(tmp)

        c = gDiffNorm/xDiffNorm
        L = max(L,c)
        m = min(m,c)

        copyto!(x, x_new)
        copyto!(y, y_new)
        copyto!(g, g_new)
        
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
    
    return x_new, f_new, metaData
end


function methodTitle(method::NAGF)
    return "NAGF"
end