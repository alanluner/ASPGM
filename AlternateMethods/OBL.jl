include("fom_interface.jl")

# This code implements the Optimized Backtracking Linesearch (OBL-F) method by Chanwoo Park and Ernest Ryu
# See their paper at https://arxiv.org/abs/2110.11035

mutable struct OBL <: FOM
    x::Vector{Float64}
    g::Vector{Float64}
    f::Float64
    z::Vector{Float64}
    xTest::Vector{Float64}
    fTest::Float64
    gTest::Vector{Float64}
    zTest::Vector{Float64}
    L::Float64
    k::Int64
    Delta::Float64
    oracleCtr::Int64
    tmp::Vector{Float64}
end

OBL() = OBL(Float64[], Float64[], 0.0, Float64[], Float64[], 0.0, Float64[], Float64[], 0.0, 0, 0.0, 0, Float64[])

function runMethod(method::OBL, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    exit = false
    metaData = Vector{Float64}[]

    # ------------ Initialize -----------
    method.x = copy(x0)
    method.g = zeros(size(method.x))
    method.z = copy(method.x)
    method.xTest = zeros(size(method.x))
    method.gTest = zeros(size(method.x))
    method.zTest = zeros(size(method.x))
    method.tmp = zeros(size(method.x))

    method.f = oracle(method.g, method.x)

    gNorm = norm(method.g)
    c = 1e-4
    @. method.tmp = method.x - c/gNorm*method.g
    method.fTest = oracle(method.gTest, method.tmp)

    @. method.tmp = method.g - method.gTest
    gDiffNormSq = dot(method.tmp, method.tmp)
    LEst = gDiffNormSq / (2 * (method.fTest - method.f + 1e-4*gNorm))
    LEst = abs(LEst) # Sanity check

    method.L = LEst
    method.k = 0
    method.Delta = 0.0
    method.oracleCtr = 2    # Start at 2 since we already called oracle twice

    # ------------ End Initialize -----------

    if saveDetails
        push!(metaData, [method.f, norm(method.g), guarantee(method), method.Delta, getTau(method), method.L])
        push!(metaData, [method.fTest, norm(method.gTest), guarantee(method), method.Delta, getTau(method), method.L])
    end

    while !exit
        ctr1 = method.oracleCtr
        update(method, oracle)
        ctr2 = method.oracleCtr

        # If we did multiple oracle calls during this iteration (backtracking), then save off data multiple times - once for each oracle call
        if saveDetails
            for i=1:ctr2-ctr1
                currentData = [method.f, norm(method.g), guarantee(method), method.Delta, getTau(method), method.L]
                push!(metaData, currentData)
            end
        end

        t = time() - t0

        if (method.oracleCtr > oracleCalls)&&(t >= runTime)
            exit = true
        end
    end

    # Convert from list of vectors to matrix
    if !isempty(metaData)
        metaData = reduce(vcat, transpose.(metaData))
    end

    return method.x, method.f, metaData

end

function update(method::OBL, oracle)

    LPrev = copy(method.L)

    # Sanity check
    if method.L >= 1e20
        str = "L estimate has gotten too large. Returning current iterate. "*string(method.k)
        @warn str maxlog=1
        method.oracleCtr += 1
        method.f = oracle(method.g, method.x)
        return
    end

    while method.L < 1e20

        @. method.zTest = method.z - (method.k + 1)/method.L*method.g

        @. method.tmp = method.x - 1/method.L * method.g

        @. method.xTest = (1-2/(method.k + 3))*method.tmp + 2/(method.k + 3)*method.zTest

        method.fTest = oracle(method.gTest, method.xTest)
        method.oracleCtr += 1

        @. method.tmp = method.g - method.gTest
        gDiffNormSq = dot(method.tmp, method.tmp)

        @. method.tmp = method.x - method.xTest
        Q = method.f - method.fTest - dot(method.gTest, method.tmp) - gDiffNormSq/(2*method.L)

        if Q + 1e-12 >= 0

            if method.L == LPrev
                delta = 0
            else
                gNormSq = dot(method.g, method.g)
                delta = method.L*(method.k+1)*(method.k+2)/2*(1/LPrev^2 - 1/method.L^2)*1/2*gNormSq
            end
            method.Delta = method.L/LPrev * method.Delta  +  delta

            copyto!(method.z, method.zTest)
            copyto!(method.x, method.xTest)
            method.f = method.fTest
            copyto!(method.g, method.gTest)
            
            break

        else
            method.L = 2*method.L
        end

    end

    method.k += 1

    return
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