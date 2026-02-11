using BlockDiagonals, SparseArrays, LinearAlgebra

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

function (q::quadratic)(dest_g::Vector{Float64}, x::Vector{Float64})
    mul!(dest_g, q.A, x)
    f = 1/2*dot(x, dest_g) + dot(q.b, x) + q.c
    @. dest_g = dest_g + q.b
    return f
end

# Least Squares Regression
# f(x) = 1/2 ||Ax-b||_2^2
struct LSRegOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
    tmp::Vector{Float64}
end

LSRegOracle(A,b) = LSRegOracle(A,b,zeros(size(b)))

function (q::LSRegOracle)(dest_g::Vector{Float64}, x::Vector{Float64})
    mul!(q.tmp, q.A, x)
    @. q.tmp = q.tmp - q.b
    f = 1/2*dot(q.tmp, q.tmp)
    mul!(dest_g, transpose(q.A), q.tmp)

    return f
end


# Logistic Regression
# f(x) = ∑ log(1+exp(c_i ⋅ ⟨a_i, x⟩))  +  η/2 ||x||_2^2
struct LogRegOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    c::Vector{Float64}
    eta::Float64
    tmp::Vector{Float64}
end

LogRegOracle(A,c,eta) = LogRegOracle(A,c,eta,zeros(size(c)))

function (q::LogRegOracle)(dest_g::Vector{Float64}, x::Vector{Float64})
    mul!(q.tmp, q.A, x)
    f = 0
    @inbounds for i in eachindex(q.c)
        t = q.c[i] * q.tmp[i]
        r = max(0, t)
        f += r + log( exp(-r) + exp(t - r))
        q.tmp[i] = q.c[i] / (1 + exp(-t))
    end

    f = f + 0.5*q.eta*dot(x,x)

    mul!(dest_g, transpose(q.A), q.tmp)
    @. dest_g = dest_g + q.eta*x

    return f
end

# Log-Sum-Exp
# f(x) = log( 1 + ∑ exp(⟨a_i, x⟩ - b_i) )
struct LogSumExpOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
    tmp::Vector{Float64}
end

LogSumExpOracle(A,b) = LogSumExpOracle(A,b,zeros(size(b)))

function (q::LogSumExpOracle)(dest_g::Vector{Float64}, x::Vector{Float64})
    mul!(q.tmp, q.A, x)
    @. q.tmp = q.tmp - q.b
    max_y = maximum(q.tmp)

    s = 0.0
    @inbounds for i in eachindex(q.tmp)
        v = exp(q.tmp[i] - max_y)
        q.tmp[i] = v
        s += v
    end

    f = max_y + log(s)

    mul!(dest_g, transpose(q.A), q.tmp)
    @. dest_g = dest_g/s

    return f
end

# Squared Relu Regression
# f(x) = ∑ max(⟨a_i, x⟩ - b_i, 0)^2
struct SquaredReluOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
    tmp::Vector{Float64}
end

SquaredReluOracle(A,b) = SquaredReluOracle(A,b,zeros(size(b)))

function (q::SquaredReluOracle)(dest_g::Vector{Float64}, x::Vector{Float64})
    mul!(q.tmp, q.A, x)
    @. q.tmp = q.tmp - q.b

    q.tmp .= max.(q.tmp, 0)

    f = 1/2*dot(q.tmp, q.tmp)

    mul!(dest_g, transpose(q.A), q.tmp)

    return f
end

# Cubic Regression
# f(x) = ⟨b, x⟩ + 1/2*||A*x||_2^2 + η/6*||x||_2^3
struct CubicRegOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
    eta::Float64
    tmp::Vector{Float64}
end

CubicRegOracle(A,b,eta) = CubicRegOracle(A,b,eta,zeros(size(A,1)))

function (q::CubicRegOracle)(dest_g::Vector{Float64}, x::Vector{Float64})
    mul!(q.tmp, q.A, x)
    yNormSq = dot(q.tmp, q.tmp)
    xNorm = norm(x)

    f = dot(q.b, x) + 1/2*yNormSq + (q.eta/6)*xNorm^3

    mul!(dest_g, transpose(q.A), q.tmp)
    @. dest_g = dest_g + q.b + (q.eta/2)*xNorm*x

    return f
end

# Quartic Regression
# f(x) = 1/4 ||Ax-b||_4^4
struct QuarticRegOracle <: Oracle
    A::Union{Matrix{Float64}, SparseMatrixCSC{Float64,Int64}}
    b::Vector{Float64}
    tmp::Vector{Float64}
end

QuarticRegOracle(A, b) = QuarticRegOracle(A, b, zeros(size(b)))

function (q::QuarticRegOracle)(dest_g::Vector{Float64}, x::Vector{Float64})
    mul!(q.tmp, q.A, x)
    @. q.tmp = q.tmp - q.b

    f = 1/4*norm(q.tmp, 4)^4

    @. q.tmp = q.tmp.^3

    mul!(dest_g, transpose(q.A), q.tmp)

    return f
end


# "Wrapper" oracle that stores the function value each time the oracle is called.
# This also accounts for the behavior of L-BFGS in which it calls for f(x) and ∇f(x) independently, not leveraging them at the same time.
# For example, when L-BFGS begins a linesearch, it will call for f(x) at the current iterate x_n, even though the gradient ∇f(x_n) was already calculated.
# In our framework where f(x) and ∇f(x) are calculated simultaneously, this should only result in one oracle call. The special handling of SmartOracle ensures that is the case.

mutable struct SmartOracle <: Oracle
    orac::Oracle                # Oracle
    x::Vector{Float64}          # Stores iterate from previous oracle call
    f::Float64                  # Stores value from previous oracle call
    g::Vector{Float64}          # Stores gradient from previous oracle call
    vals::Vector{Float64}       # Saves off the value at each oracle call
    times::Vector{Float64}      # Saves off the time at each oracle call
    oracleCalls::Int64          # Total oracle calls
end

SmartOracle(q) = SmartOracle(q, -Inf*ones(size(q.A,2)), 0.0, zeros(size(q.A,2)), Float64[], Float64[], 0)


function (q::SmartOracle)(dest_g::Vector{Float64}, x::Vector{Float64})

    q.f = q.orac(dest_g, x)
    q.oracleCalls += 1

    copyto!(q.x, x)
    copyto!(q.g, dest_g)

    push!(q.times, time())
    push!(q.vals, q.f)

    return q.f

end

# Oracle that calculates f(x) and ∇f(x), and returns f(x). ∇f(x) is saved off for any future oracle calls for this same point x
function fOracle(q::SmartOracle, x::Vector{Float64})
    #If evaluation point x is the same as previous evaluation point, reuse previous data
    if x == q.x 
        f = q.f

    # Else, evaluate new f and g, and save off in case of future repeats
    else
        f = q.orac(q.g, x)
        q.oracleCalls += 1
        copyto!(q.x, x)
        q.f = f
        # g is already stored

        push!(q.times, time())
        push!(q.vals, q.f)
    end

    return q.f

end

# Oracle that calculates f(x) and ∇f(x), and returns ∇f(x) (in place). f(x) is saved off for any future oracle calls for this same point x
function gOracle!(dest_g::Vector{Float64}, q::SmartOracle, x::Vector{Float64})
    #If evaluation point x is the same as previous data, reuse previous data
    if x == q.x
        copyto!(dest_g, q.g)

    # Else, evaluate new f and g, and save off in case of future repeats
    else
        f = q.orac(dest_g, x)
        q.oracleCalls += 1
        copyto!(q.x, x)
        q.f = f
        copyto!(q.g, dest_g)
        
        push!(q.times, time())
        push!(q.vals, q.f)
    end

    return nothing

end