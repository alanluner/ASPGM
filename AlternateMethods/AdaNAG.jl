include("fom_interface.jl")

mutable struct AdaNAG <: FOM
end


Tau(k) = ((k+2)+12)/12

Alpha(k) = 1/2*(Tau(k+1)-1)^2/Tau(k)^2

Ak(k) = ( k==-1 ? 0 : Alpha(k+1)*Tau(k+1)*(Tau(k+1)-1) )

B0 = Alpha(0)^2*Tau(0)^2*((Tau(0)-1)^2/(Alpha(-1)*Tau(-1)^2) - 1)
Bk(k) = ( k==0 ? B0 : Alpha(k)^2*Tau(k)^2*((Tau(k)-1)^2/(Alpha(k-1)*Tau(k-1)^2) - 1) )

function runMethod(method::AdaNAG, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    t0 = time()
    metaData = []
    exit = false

    x = x0
    i = 0

    f, g = oracle(x)
    fOld = f
    gOld = copy(g)
    xOld = copy(x)
    z = copy(x)


    y = x0 + 1e-4*randn(length(x0))
    fy,gy = oracle(y)
    L = dot(g - gy, g - gy) / (2 * (fy - f - dot(g, y-x0)))
    if isnan(L)||(L<=0)
        L = 0.01
    end


    r = 27/(2*(12+3)*(2*12^2+8*12+17))
    s = Ak(0)*r/(Alpha(0)*Tau(0)*Alpha(1)*L)

    if saveDetails
        metaData = [f  norm(g)  guarantee(i,s)]
        metaData = vcat(metaData, [fy  norm(gy)  guarantee(i,s)])
    end

    while !exit

        y = x - s*g
        z = z - s*Alpha(i)*Tau(i)*g

        fOld = f
        gOld = copy(g)
        xOld = copy(x)

        x = (1-1/Tau(i+1))*y + 1/Tau(i+1)*z
        f,g = oracle(x)
        if saveDetails
            metaData = vcat(metaData, [f  norm(g)  guarantee(i,s)])
        end


        L = -(1/2*norm(g - gOld)^2)/(f - fOld + g'*(xOld - x))

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

    return x, f, metaData
end

function guarantee(i, s)
    return 2*144*(i+15)/((i+3)*(i+4)^2)*1/s
end

function methodTitle(method::AdaNAG)
    return "AdaNAG"
end