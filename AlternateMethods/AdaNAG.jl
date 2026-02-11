include("fom_interface.jl")

# This code implements the AdaNAG method by Jaewook Suh and Shiqian Ma.
# See their paper at https://arxiv.org/pdf/2505.11670

mutable struct AdaNAG <: FOM
end


Tau(k) = ((k+2)+12)/12

Alpha(k) = 1/2*(Tau(k+1)-1)^2/Tau(k)^2

Ak(k) = ( k==-1 ? 0 : Alpha(k+1)*Tau(k+1)*(Tau(k+1)-1) )

B0 = Alpha(0)^2*Tau(0)^2*((Tau(0)-1)^2/(Alpha(-1)*Tau(-1)^2) - 1)
Bk(k) = ( k==0 ? B0 : Alpha(k)^2*Tau(k)^2*((Tau(k)-1)^2/(Alpha(k-1)*Tau(k-1)^2) - 1) )

function runMethod(method::AdaNAG, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    metaData = Vector{Float64}[]
    exit = false

    x = copy(x0)
    g = zeros(size(x))
    xOld = copy(x)
    gOld = zeros(size(x))
    y = zeros(size(x0))
    z = copy(x)
    tmp = zeros(size(x0))

    i = 0

    f = oracle(g, x)
    
    fOld = f


    gNorm = norm(g)
    c = 1e-4
    @. y = x0 - c/gNorm*g
    fOld = oracle(gOld, y)

    @. tmp = g - gOld
    L = dot(tmp, tmp) / (2 * (fOld - f + c*gNorm))
    if isnan(L)||(L<=0)
        L = 0.01
    end

    r = 27/(2*(12+3)*(2*12^2+8*12+17))
    s = Ak(0)*r/(Alpha(0)*Tau(0)*Alpha(1)*L)

    if saveDetails
        push!(metaData, [f, norm(g), guarantee(i,s)])
        push!(metaData, [fOld, norm(gOld), guarantee(i,s)])
    end

    while !exit

        @. y = x - s*g

        c = Alpha(i)*Tau(i)*s
        @. z = z - c*g

        fOld = f
        copyto!(gOld, g)
        copyto!(xOld, x)

        c1 = (1-1/Tau(i+1))
        c2 = 1/Tau(i+1)
        @. x = c1*y + c2*z
        f = oracle(g, x)

        if saveDetails
            push!(metaData, [f, norm(g), guarantee(i,s)])
        end

        @. tmp = g - gOld
        gDiffNormSq = norm(tmp)^2

        @. tmp = x - xOld
        L = -(1/2*gDiffNormSq)/(f - fOld - dot(g, tmp))

        if isnan(L)||L<=0
            s = (Ak(i-1)+Alpha(i)*Tau(i))/(Ak(i))*s
        else
            s = min((Ak(i-1)+Alpha(i)*Tau(i))/(Ak(i))*s, 1/(Ak(i)/Bk(i) + (Bk(i+1)+Alpha(i+1)^2*Tau(i+1)^2)/Ak(i))*1/L)
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

function guarantee(i, s)
    return 2*144*(i+15)/((i+3)*(i+4)^2)*1/s
end

function methodTitle(method::AdaNAG)
    return "AdaNAG"
end