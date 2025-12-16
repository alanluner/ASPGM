include("fom_interface.jl")

mutable struct OSGMR <: FOM
    eta
end

OSGMR() = OSGMR(0.01)



function runMethod(method::OSGMR, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)
    
    t0 = time()
    i = 0
    exit = false
    metaData = []

    P0 = zeros(length(x0))

    f0, g0 = oracle(x0)

    if saveDetails
        metaData = [f0  norm(g0)]
    end

    x = x0
    g = g0
    f = f0
    P = P0

    V = zeros(size(x0))

    fStar = min(-1, f0 - 2*abs(f0))

    while !exit

        xTest = x - P.*g
        
        fNew, gNew = oracle(xTest)
        if saveDetails
            metaData = vcat(metaData, [fNew  norm(gNew)])
        end

        if fNew < fStar
            fStar = fNew - min(5*(fStar - fNew), 1)
        end

        # We are ultimately taking the diagonal, so we don't need a full matrix
        vec = -gNew.*g/(f-fStar)
        
        V .+= vec.^2

        epsilon = 1e-6
        steps = method.eta*(epsilon .+ V).^(-1/2)

        P = P - steps.*vec

        if fNew < f
            x = xTest
            f = fNew
            g = gNew
        end

        i = i+1
        t = time() - t0

        if (i > oracleCalls)&&(t >= runTime)
            exit = true
        end
    end

    return x, f, metaData

end


function methodTitle(method::OSGMR)
    return "OSGMR"
end


