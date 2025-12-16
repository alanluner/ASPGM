using BlockDiagonals, SparseArrays

# Abstract superclass for first order oracles
# Should implement
# (f(x), ∇f(x)) = (q::Oracle)(x::Vector{Float64})
abstract type Oracle end

# Quadratic
# f(x) = 1/2 x'*A*x + b'*x + c
struct quadratic <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
    c::Float64
end

function (q::quadratic)(x::Vector{Float64})
    y = q.A*x
    f = 1/2*x'*y + q.b'*x + q.c
    g = y + q.b
    return f, g
end

# Least Squares Regression
# f(x) = 1/2 ||Ax-b||_2^2
struct LSRegOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
end

function (q::LSRegOracle)(x::Vector{Float64})
    y = q.A*x-q.b
    f = 1/2*norm(y)^2
    g = q.A'*(y)

    return f,g
end


# Logistic Regression
# f(x) = ∑ log(1+exp(c_i ⋅ ⟨a_i, x⟩))  +  η/2 ||x||_2^2
struct LogRegOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    c::Vector{Float64}
    eta::Float64
end

function (q::LogRegOracle)(x::Vector{Float64})
    y = q.A*x
    retg = zeros(length(y))
    f = 0
    for i in 1:length(q.c)
        f += max(0,q.c[i]*y[i]) + log(exp(-1*max(0,q.c[i]*y[i]))+exp(q.c[i]*y[i] - max(0,q.c[i]*y[i])))
        retg[i] = q.c[i] / (1 + exp(-1*q.c[i] * y[i]))
    end
    f = f + q.eta/2*norm(x)^2
    g = q.A'*retg + q.eta*x

    return f,g
end

# Log-Sum-Exp
# f(x) = log( 1 + ∑ exp(⟨a_i, x⟩ - b_i) )
struct LogSumExpOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
end

function (q::LogSumExpOracle)(x::Vector{Float64})
    y = q.A*x - q.b
    max_y = maximum(y)
    f = max_y + log(sum(exp.(y .- max_y)))
    exp_shifted = exp.( y .- max_y)
    g = q.A'*exp_shifted/sum(exp_shifted)

    return f,g
end

# Squared Relu Regression
# f(x) = ∑ max(⟨a_i, x⟩ - b_i, 0)^2
struct SquaredReluOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
end

function (q::SquaredReluOracle)(x::Vector{Float64})
    y = q.A*x - q.b
    f = 1/2*sum((max.(y, zeros(length(q.b)))).^2)
    g = q.A'*max.(y, zeros(length(q.b)))
    return f, g
end

# Cubic Regression
# f(x) = ⟨b, x⟩ + 1/2*||A*x||_2^2 + η/6*||x||_2^3
struct CubicRegOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
    eta::Float64
end

function (q::CubicRegOracle)(x::Vector{Float64})
    y = q.A*x
    f = q.b'*x + 1/2*norm(y)^2 + (q.eta/6)*norm(x)^3
    g = q.b + q.A'*y + (q.eta/2)*norm(x)*x

    return f,g
end

# Quartic Regression
# f(x) = 1/4 ||Ax-b||_4^4
struct QuarticRegOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
end

function (q::QuarticRegOracle)(x::Vector{Float64})
    y = q.A*x - q.b
    f = norm(y, 4)^4
    g = 4*q.A'*(y.^3)
    return f,g
end


# "Wrapper" oracle that stores the function value each time the oracle is called.
# This also accounts for the behavior of L-BFGS in which it calls for f(x) and ∇f(x) independently, not leveraging them at the same time.
# For example, when L-BFGS begins a linesearch, it will call for f(x) at the current iterate x_n, even though the gradient ∇f(x_n) was already calculated.
# In our framework where f(x) and ∇f(x) are calculated simultaneously, this should only result in one oracle call. The special handling of SmartOracle ensures that is the case.

mutable struct SmartOracle <: Oracle
    q::Oracle
    f
    g
    x
    vals       # Saves off the value each time the specific type of oracle is called
    times
    oracleCalls
end

SmartOracle(q) = SmartOracle(q, [], [], [], [], [], 0)


function (q::SmartOracle)(x::Vector{Float64})

    f, g = q.q(x)
    q.oracleCalls += 1
    q.x = copy(x)
    q.f = f
    q.g = copy(g)
    push!(q.times, time())
    push!(q.vals, q.f)

    return (q.f, q.g)

end

function fOracle(q::SmartOracle, x::Vector{Float64})
    #If evaluation point x is the same as previous evaluation point, reuse previous data
    if x == q.x 
        f = q.f

    # Else, evaluate new f and g, and save off in case of future repeats
    else
        (f, g) = q.q(x)     
        q.oracleCalls += 1
        q.x = copy(x)
        q.f = f
        q.g = copy(g)
        push!(q.times, time())
        push!(q.vals, q.f)
    end

    return q.f

end

function gOracle(q::SmartOracle, x::Vector{Float64})
    #If evaluation point x is the same as previous data, reuse previous data
    if x == q.x
        g = q.g

    # Else, evaluate new f and g, and save off in case of future repeats
    else
        (f, g) = q.q(x)
        q.oracleCalls += 1
        q.x = copy(x)
        q.f = f
        q.g = copy(g)
        push!(q.times, time())
        push!(q.vals, q.f)
    end

    return q.g

end