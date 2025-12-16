include("fom_interface.jl")

mutable struct NAGF <: FOM
end


function runMethod(method::NAGF, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    i = 0
    exit = false
    metaData = []

    y = x0 + 1e-6*rand(length(x0))
    
    f0, g0 = oracle(x0)
    fy, gy = oracle(y)

    if saveDetails
        metaData = [f0  norm(g0)]
        metaData = vcat(metaData, [fy  norm(gy)])
    end

    L0 = norm(g0 - gy)/norm(x0 - y)
    m0 = L0

    L = L0
    m = m0

    x = copy(x0)
    g = copy(g0)
    f = f0

    y_new = copy(y)
    x_new = copy(x)
    f_new = f
    g_new = copy(g)

    while !exit
        y_new = x - g/L
        x_new = y_new + (sqrt(L)-sqrt(m))/(sqrt(L) + sqrt(m))*(y_new - y)
        f_new, g_new = oracle(x_new)
        if saveDetails
            metaData = vcat(metaData, [f_new  norm(g_new)])
        end
        c = norm(g_new - g)/norm(x_new - x)
        L = max(L,c)
        m = min(m,c)

        x = x_new
        y = y_new
        g = g_new
        
        i = i+1
        t = time() - t0

        if (i > oracleCalls)&&(t >= runTime)
            exit = true
        end
    end
    
    return x_new, f_new, metaData
end


function methodTitle(method::NAGF)
    return "NAGF"
end