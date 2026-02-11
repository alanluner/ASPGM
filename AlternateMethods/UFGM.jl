include("fom_interface.jl")

# This code implements the Universal Fast Gradient Method (UFGM) method by Alexander Gasnikov and Yurii Nesterov
# See their paper at https://arxiv.org/abs/1604.05275

mutable struct UFGM <: FOM
end

function runMethod(method::UFGM, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)
    
    t0 = time()
    i = 0
    exit = false
    metaData = Vector{Float64}[]

    x = copy(x0)
    g = zeros(size(x))
    y = zeros(size(x))
    x_new = zeros(size(x))
    g_new = zeros(size(x))
    y_new = zeros(size(x))
    v = zeros(size(x))
    dual_phi = zeros(size(x))
    tmp = zeros(size(x))

    f = oracle(g, x)

    gNorm = norm(g)
    c = 1e-4
    @. x_new = x - c/gNorm*g
    f_new = oracle(g_new, x_new)

    @. tmp = g - g_new
    L = dot(tmp, tmp) / (2 * (f_new - f + c*gNorm))
    if isnan(L)||(L<=0)
        L = 0.01
    end

    if saveDetails
        push!(metaData, [f, norm(g)])
        push!(metaData, [f_new, norm(g_new)])
    end

    A_new = 0.0
    a_new = 0.0
    L_new = 0.0
    a = 0.0
    A = 0.0
    f_y = 0.0

    copyto!(y, x)

    while !exit
        @. v = x0 - dual_phi
        for j = 0:50
            L_new = 2^j * L
            a_new = (1+sqrt(abs(1+4*L_new*A)))/(2*L_new)
            A_new = A + a_new

            tau = a_new / A_new
            @. x_new = tau * v + (1-tau) * y
            f_x = oracle(g_new, x_new)

            @. tmp = v - a_new * g_new
            @. y_new = tau * tmp + (1-tau) * y

            f_y = oracle(tmp, y_new)

            if saveDetails
                push!(metaData, [f_x, norm(g_new)])
                push!(metaData, [f_y, norm(tmp)])
            end

            @. tmp = y_new - x_new
            val1 = f_y - f_x - dot(g_new, tmp)
            val2 = L_new/2 * dot(tmp, tmp)

            i = i+2

            if val1 <= val2
                break
            end
        end

        copyto!(x, x_new)
        copyto!(y, y_new)
        A = A_new
        L = L_new/2
        a = a_new
        @. dual_phi = dual_phi + a * g_new

        t = time() - t0

        if (i > oracleCalls)&&(t >= runTime)
            exit = true
        end

    end

    # Convert from list of vectors to matrix
    if !isempty(metaData)
        metaData = reduce(vcat, transpose.(metaData))
    end
    
    return y, f_y, metaData
end


function methodTitle(method::UFGM)
    return "UFGM"
end