include("fom_interface.jl")

# This implementation of UFGM is adapted from the UFGM implementation in https://github.com/tli432/AC-FGM-Implementation

mutable struct UFGM <: FOM
end

function runMethod(method::UFGM, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)
    
    t0 = time()
    i = 0
    exit = false
    metaData = []

    x = copy(x0)
    f, g = oracle(x)

    A = 0
    dual_phi = zeros(length(x0))
    const_psi = 0

    y = x0 + 1e-4*randn(length(x0))
    f_y,g_y = oracle(y)
    L0 = dot(g - g_y, g - g_y) / (2 * (f_y - f - dot(g, y-x0)))
    if isnan(L0)||(L0==0)
        L0 = 0.01
    end

    if saveDetails
        metaData = [f  norm(g)]
        metaData = vcat(metaData, [f_y  norm(g_y)])
    end

    L = L0

    while !exit
        v = x0 - dual_phi
        x_new = x0
        y_new = x0
        A_new = A
        a_new = 0
        L_new = L
        g_new = g
        f_y = f
        for j = 0:50
            L_new = 2^j * L
            a_new = (1+sqrt(abs(1+4*L_new*A)))/(2*L_new)
            A_new = A + a_new
            tau = a_new / A_new
            x_new = tau * v + (1-tau) * y
            f_x, g_new = oracle(x_new)
            x_hat_new = v - a_new * g_new
            y_new = tau * x_hat_new + (1-tau) * y
            f_y, g_y = oracle(y_new)
            tmp1 = f_y - f_x - dot(g_new, y_new - x_new)
            tmp2 = L_new/2 * (norm(y_new-x_new))^2

            i = i+2

            if saveDetails
                metaData = vcat(metaData, [f_x  norm(g_new)])
                metaData = vcat(metaData, [f_y  norm(g_y)])
            end

            if tmp1 <= tmp2  + 1e-15 * f_y
                break
            end
        end

        x = x_new
        y = y_new
        A = A_new
        L = L_new/2
        a = a_new
        dual_phi += a * g_new
        const_psi += a
        obj = f_y

        t = time() - t0

        if (i > oracleCalls)&&(t >= runTime)
            exit = true
        end

    end
    
    return y, f_y, metaData
end


function methodTitle(method::UFGM)
    return "UFGM"
end