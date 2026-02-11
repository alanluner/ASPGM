include("fom_interface.jl")

# This code implements the OSGM-Best method by Ya-Chi Chu, Wenzhi Gao, Yinyu Ye, and Madeleine Udell
# See their paper at https://arxiv.org/pdf/2509.11007

mutable struct OSGMB <: FOM
end


function runMethod(method::OSGMB, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    i = 0
    exit = false
    metaData = Vector{Float64}[]

    x = copy(x0)
    g = zeros(size(x))
    x_tmp = zeros(size(x))
    g_tmp = zeros(size(x))
    x_lookahead = zeros(size(x))
    g_lookahead = zeros(size(x))
    x_diff = zeros(size(x))
    P = zeros(length(x0))
    g_P = zeros(size(x))
    g_phi = zeros(size(x))
    v = zeros(size(x))


    f = oracle(g, x)

    gNorm = norm(g)
    c = 1e-4
    @. v = x0 - c/gNorm*g
    f_tmp = oracle(g_tmp, v)

    @. v = g - g_tmp
    gDiffNormSq = dot(v,v)
    L = gDiffNormSq / (2 * (f_tmp - f + 1e-4*gNorm))
    L = abs(L) # Sanity check

    omega = 0.0
    tau = 1/2*L^2
    beta = 0.95
    eta_P = 1/L
    eta_B = min(1.0, L)

    if saveDetails
        push!(metaData, [f, norm(g)])
        push!(metaData, [f_tmp, norm(g_tmp)])
    end

    x_prev = copy(x)
    phi_prev = Inf

    gNormSq = dot(g, g)

    while !exit

        @. x_diff = x - x_prev
        x_diff_normSq = dot(x_diff, x_diff)

        @. v = P*g
        @. v = v - beta*x_diff

        @. x_tmp = x - v
        f_tmp = oracle(g_tmp, x_tmp)

        @. g_phi = g_tmp - omega*v

        c = 1/(gNormSq + tau/2*x_diff_normSq)
        @. g_P = -c*g_phi*g
        g_B = c*dot(g_phi, x_diff)

        c1 = (1-omega/(L+omega))
        c2 = omega/(L+omega)
        c3 = -1/(L+omega)
        @. x_lookahead = c1*x_tmp + c2*x_prev + c3*g_tmp

        f_lookahead = oracle(g_lookahead, x_lookahead)

        @. v = x_lookahead - x
        new_xDiffSq = dot(v,v)
        phi_lookahead = f_lookahead + omega/2*new_xDiffSq

        nullStep = true
        if phi_lookahead <= phi_prev
            nullStep = false
        end

        @. P = P - eta_P*g_P
        beta = beta - eta_B*g_B

        P .= min.(max.(0, P), 1e5/L)
        beta = min(max(0, beta), 0.9995)

        copyto!(x_prev, x)

        if !nullStep
            phi_prev = phi_lookahead
            copyto!(x, x_lookahead)
            copyto!(g, g_lookahead)
            f = f_lookahead
            gNormSq = dot(g, g)
        end

        if saveDetails
            push!(metaData, [f_tmp, norm(g_tmp)])
            push!(metaData, [f_lookahead, norm(g_lookahead)])
        end

        i = i+2
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


function methodTitle(method::OSGMB)
    return "OSGMB"
end
