include("fom_interface.jl")

mutable struct OBL <: FOM
    value
    gradient
    x
    z
    L
    k
    Delta
    oracleCtr
end

OBL() = OBL(missing, missing, missing, missing, missing, missing, missing, missing)

function runMethod(method::OBL, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    i = 0
    exit = false
    metaData = []

    y = x0 + 1e-4*randn(length(x0))
    f0,g0 = oracle(x0)
    fy,gy = oracle(y)
    LEst = dot(g0 - gy, g0 - gy) / (2 * (fy - f0 - dot(g0, y-x0)))
    LEst = abs(LEst) # Sanity check
    
    initialize(method, x0, f0, g0, LEst)

    if saveDetails
        metaData = [f0  norm(g0)  guarantee(method)  method.Delta  getTau(method)  method.L]
        metaData = vcat(metaData, [fy  norm(gy)  guarantee(method)  method.Delta  getTau(method)  method.L])
    end
    
    x = x0

    while !exit
        ctr1 = method.oracleCtr
        x = update(method, oracle)
        ctr2 = method.oracleCtr

        # If we did multiple oracle calls during this iteration (backtracking), then save off data multiple times - once for each oracle call
        if saveDetails
            for i=1:ctr2-ctr1
                metaData = vcat(metaData, [method.value  norm(method.gradient)  guarantee(method)  method.Delta  getTau(method)  method.L])
            end
        end

        t = time() - t0

        if (method.oracleCtr > oracleCalls)&&(t >= runTime)
            exit = true
        end
    end

    return method.x, method.value, metaData

end

function initialize(method::OBL, x0::Vector{Float64}, val::Float64, grad::Vector{Float64}, L::Float64)
    method.value = val
    method.gradient = grad
    method.x = x0
    method.z = x0
    method.L = L
    method.k = 0
    method.Delta = 0
    method.oracleCtr = 2 # Start at 2 since we already called oracle twice
end

function update(method::OBL, oracle::Oracle)

    LPrev = copy(method.L)

    # Sanity check
    if method.L >= 1e10
        @warn "L estimate has gotten too large. Returning current iterate" maxlog=1
        method.oracleCtr += 1
        return method.x
    end

    while method.L < 1e10

        y = method.x - 1/method.L * method.gradient
        z = method.z - (method.k + 1)/method.L * method.gradient
        xTest = (1-2/(method.k + 3))*y + 2/(method.k + 3)*z

        fTest, gTest = oracle(xTest)
        method.oracleCtr += 1

        if isNewIterateValid(xTest, fTest, gTest, method.x, method.value, method.gradient, method.L)

            delta = method.L*(method.k+1)*(method.k+2)/2*(1/LPrev^2 - 1/method.L^2)*1/2*norm(method.gradient)^2
            method.Delta = method.L/LPrev * method.Delta  +  delta

            method.z = z
            method.x = xTest
            method.value = fTest
            method.gradient = gTest
            
            break

        else
            method.L = 2*method.L
        end

    end

    method.k += 1

    return method.x
end


function getTau(method::OBL)
    return (method.k+1)*(method.k+2)/2
end

function guarantee(method::OBL)
    return 2*method.L/((method.k+1)*(method.k+2))
end

function methodTitle(method::OBL)
    return "OBL"
end

function isNewIterateValid(x_N::Vector{Float64}, f_N::Float64, g_N::Vector{Float64},  x_NMinus1::Vector{Float64}, f_NMinus1::Float64, g_NMinus1::Vector{Float64}, L::Float64)

    Q = f_NMinus1 - f_N - dot(g_N, x_NMinus1 - x_N) - norm(g_N - g_NMinus1)^2/(2L)

    return (Q>=0)

end