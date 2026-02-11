include("fom_interface.jl")


using Optim, LineSearches, NLSolversBase

mutable struct LBFGS <: FOM
    mem_size::Int64
    linesearchMode

    LBFGS(mem_size) = new(mem_size, LineSearches.BackTracking())

    function LBFGS(mem_size, LSMode)
        if LSMode == :StrongWolfe
            return new(mem_size, LineSearches.StrongWolfe())
        elseif LSMode == :HagerZhang
            return new(mem_size, LineSearches.HagerZhang())
        else
            return new(mem_size, LineSearches.BackTracking())
        end
    end

end


function runMethod(method::LBFGS, oracle, x0::Vector{Float64}; oracleCalls = 500, runTime = 0, saveDetails = false)

    function f(x)
        return fOracle(oracle, x)
    end

    function g!(dest_g, x)
        return gOracle!(dest_g, oracle, x)
    end

    if oracleCalls > 0
        if runTime > 0
            # For LBFGS, we can't use both runTime and oracleCalls, so we will just use the runTime limit
            res = Optim.optimize(f, g!, x0, Optim.LBFGS(;m=method.mem_size, linesearch = method.linesearchMode), Optim.Options(time_limit = runTime, g_abstol=1e-16))
        else
            # Otherwise, use oracleCalls limit (via g_calls_limit)
            res = Optim.optimize(f, g!, x0, Optim.LBFGS(;m=method.mem_size, linesearch = method.linesearchMode), Optim.Options(g_calls_limit = oracleCalls, g_abstol=1e-16))
        end
    else
        # Otherwise, use runTime (via time_limit)
        res = Optim.optimize(f, g!, x0, Optim.LBFGS(;m=method.mem_size, linesearch = method.linesearchMode), Optim.Options(time_limit = runTime, g_abstol=1e-16))
    end

    return Optim.minimizer(res), Optim.minimum(res), []
end

function methodTitle(method::LBFGS)
    if typeof(method.linesearchMode).name.name == :BackTracking
        str = "BT"
    else
        str = "HZ"
    end
    return "LBFGS-"*str
end

