include("example_oracles.jl")

using JLD2, QPSReader, LinearAlgebra, Optim

function generateOracles(rng, problemTypes, dimArray; numCopies = 1)

    oracles = []
    x0s = []
    for i in axes(dimArray,1)
        print(i)
        d = dimArray[i]
        m = 4*d
        for problemType in problemTypes
            for k=1:numCopies # Determine how many instances per setting

                q1, x1 = buildOracle(problemType, rng, 1, m, d)
                q2, x2 = buildOracle(problemType, rng, 2, m, d)
                q3, x3 = buildOracle(problemType, rng, 3, m, d)
                q4, x4 = buildOracle(problemType, rng, 4, m, d)

                append!(oracles, [q1, q2, q3, q4])
                append!(x0s, [x1, x2, x3, x4])
            end
        end

    end

    return oracles, x0s

end


function getSolutions(oracles, x0s; file = [])
    xStarList = []
    fStarList = []
    for (i,q) in enumerate(oracles)
        print(i)
        x0 = x0s[i]
        fStar, xStar, success = getSolution(q, x0; maxIter = 25000)
        if !success
            print(" --Max iterations reached.")
        end
        append!(fStarList, fStar)
        append!(xStarList, [xStar])
    end

    if !isempty(file)
        jldsave(file; xStarList, fStarList)
    end

    return fStarList, xStarList
end

function getSolution(oracle, start::Vector{Float64}; maxIter::Int64 = 10000, printout = false)
    res = Optim.optimize((x -> oracle(x)[1]), (x -> oracle(x)[2]), start, Optim.LBFGS(), Optim.Options(g_abstol=1e-12, show_trace=printout, show_every=500, iterations=maxIter); inplace=false)
    success = Optim.converged(res)
    return Optim.minimum(res), Optim.minimizer(res), success
end


function buildOracle(type, rng, mode, m, d)

    if type == :LSReg
        q, x0 = build_LSReg(rng, mode, m, d)
    elseif type == :LogReg
        q, x0 = build_LogReg(rng, mode, m, d)
    elseif type == :LogSumExp
        q, x0 = build_LogSumExp(rng, mode, m, d)
    elseif type == :SquaredRelu
        q, x0 = build_SquaredRelu(rng, mode, m, d)
    elseif type == :CubicReg
        q, x0 = build_CubicReg(rng, mode, m, d)
    elseif type == :QuarticReg
        q, x0 = build_QuarticReg(rng, mode, m, d)
    elseif type == :HardA
        q, x0 = build_HardInstanceA(d)
    elseif type == :HardB
        q, x0 = build_HardInstanceB(d)
    elseif type == :HardC
        q, x0 = build_HardInstanceC(d)
    end

    return q, x0
end

function build_LSReg(rng, mode, m, d)
    A = randMat(rng, mode, m, d)
    b = randn(rng, m)

    x0 = zeros(d)
    q = LSRegOracle(A, b)

    return q, x0
end

function build_LogReg(rng, mode, m, d)
    c = 2*round.(rand(rng, m)) .- 1
    A = randMat(rng, mode, m, d)

    x0 = zeros(d)
    q = LogRegOracle(A,c,1/m)

    return q, x0
end

function build_LogSumExp(rng, mode, m, d)
    A = randMat(rng,mode, m, d)
    b = randn(rng, m)

    A = vcat(A, zeros(d)')
    b = vcat(b, 0)

    x0 = zeros(d)
    q = LogSumExpOracle(A, b)

    return q, x0
end


function build_SquaredRelu(rng, mode, m, d)
    A = randMat(rng, mode, m, d)
    b = randn(rng, m)

    x0 = zeros(d)
    q = SquaredReluOracle(A,b)

    return q, x0
end

function build_CubicReg(rng, mode, m, d)
    A = randMat(rng,mode,m,d)
    b = randn(rng, d)

    x0 = zeros(d)
    q = CubicRegOracle(A, b, 1/m)

    return q, x0
end

function build_QuarticReg(rng, mode, m, d)
    A = randMat(rng, mode, m, d)
    b = randn(rng, m)
    x0 = zeros(d)
    q = QuarticRegOracle(A,b)
    
    return q, x0
end

function build_HardInstanceA(d)
    A = 1/2*diagm(-1 => -1/2*ones(d-1),  0 => ones(d),  1 => -1/2*ones(d-1))
    b = zeros(d)
    b[1] = -1/2

    x0 = zeros(d)
    q = quadratic(A, b, 0.0)

    return q, x0
end

function build_HardInstanceB(d)
    lams = (sin.(pi*(1:d)./(2*d))).^2
    A = diagm(lams)

    x0 = 1 ./ lams
    q = quadratic(A, zeros(d), 0.0)

    return q, x0
end

function build_HardInstanceC(d)
    A = 1.0*diagm(1:d)
    b = ones(d)
    x0 = zeros(d)

    q = LSRegOracle(A, b)

    return q, x0
end



# Generate random matrix A according to 4 possible distributions:
# 1: σ_i ~ 𝒰(1, √κ), κ = 100
# 2: σ_i ~ 𝒰(1, √κ), κ = 10000
# 3: σ_1, ..., σ_{9d/10} ~ 𝒰(1, 1.1),     σ_{9d/10+1}, ..., σ_d ~ 𝒰(0.9√κ, √κ), κ = 100
# 4: σ_1, ..., σ_{9d/10} ~ 𝒰(1, 1.1),     σ_{9d/10+1}, ..., σ_d ~ 𝒰(0.9√κ, √κ), κ = 10000

function randMat(rng, mode, m, d)

    if mode in [1,3]
        maxSV = 1e2
    else
        maxSV = 1e4
    end

    # Uniform distribution
    if mode in [1,2]

        # Uniformly distributed over [1, maxSV] so that condition number is approximately maxSV
        sv = 1.0 .+ (maxSV - 1.0)*rand(rng, min(m,d))

    # Skewed distribution
    elseif mode in [3,4]

        # Take 10% between 0.9*maxSV and maxSV
        # And 90% between [1,1.1]
        # So the condition number is again approximately maxSV
        k = Int(floor(min(m,d)/10))
        sv1 = 0.9*maxSV .+ (maxSV*0.1)*rand(rng, k)
        sv2 = 1.0 .+ (10-1)*rand(rng,min(m,d)-k)
        sv = vcat(sv1, sv2)
        sv = sort(sv)
    end 

    M = randn(rng,m,d)
    U,~,V = svd(M)
    
    S = diagm(sqrt.(sv))

    # Account for different output cases for svd
    if m>d
        A = U*S
    else
        A = S*V'
    end

    return A

end


function defaultProblemTypes()
    return [
        :LogSumExp, 
        :LSReg,
        :LogReg, 
        :SquaredRelu, 
        :QuarticReg, 
        :CubicReg
    ]
end

function problemList_LIBSVM()
    probs = [:coloncancer,:duke,:gisette,:leukemia,:madelon,
            :bodyfat,:eunite,:pyrim,:triazines,:YearPrediction]
    types = [:LogReg,:LogReg,:LogReg,:LogReg,:LogReg,
             :LSReg,:LSReg,:LSReg, :LSReg, :LSReg]
    sources = [:LIBSVM, :LIBSVM, :LIBSVM, :LIBSVM, :LIBSVM,
            :LIBSVM, :LIBSVM, :LIBSVM, :LIBSVM, :LIBSVM]

    return probs, types, sources
end


function problemList_LP()
    probs = [:brazil3,:chromatic,:ex10,:graph40,:qap15,:rmine15,:savsched1,:scpm1,:setcover,:supportcase10,
             :brazil3,:chromatic,:ex10,:graph40,:qap15,:rmine15,:savsched1,:scpm1,:setcover,:supportcase10]
    types = [:LogSumExp,:LogSumExp,:LogSumExp,:LogSumExp,:LogSumExp,:LogSumExp,:LogSumExp,:LogSumExp,:LogSumExp,:LogSumExp,
             :SquaredRelu,:SquaredRelu,:SquaredRelu,:SquaredRelu,:SquaredRelu,:SquaredRelu,:SquaredRelu,:SquaredRelu,:SquaredRelu,:SquaredRelu]
    sources = [:LP, :LP, :LP, :LP, :LP,
               :LP, :LP, :LP, :LP, :LP]

    return probs, types, sources
end

function getProblemSource(prob)
    if prob in [:brazil3,:chromatic,:ex10,:graph40,:qap15,:rmine15,:savsched1,:scpm1,:setcover,:supportcase10]
        return :LPFeas
    elseif prob in [:coloncancer,:duke,:gisette,:leukemia,:madelon,:bodyfat,:eunite,:pyrim,:triazines,:YearPrediction]
        return :LIBSVM
    end
end


function parse_LIBSVM_file(filename::String)
    labels = Float64[]
    row_indices = Int[]
    col_indices = Int[]
    values = Float64[]

    max_feature = 0
    line_num = 0

    open(filename, "r") do file
        for line in eachline(file)
            line_num += 1
            parts = split(strip(line))

            label = parse(Float64, parts[1])
            push!(labels, label)

            for part in parts[2:end]
                idx_val = split(part, ":")
                idx = parse(Int, idx_val[1])
                val = parse(Float64, idx_val[2])

                push!(row_indices, line_num)
                push!(col_indices, idx)
                push!(values, val)

                max_feature = max(max_feature, idx)
            end
        end
    end

    m = line_num
    d = max_feature

    A = sparse(row_indices, col_indices, values, m, d)

    # Convert to dense matrix
    A = Array(A)
    b = collect(labels)

    return A, b
end


# Ax <= b

function parse_LP_file(filename::String)
    
    data = readqps(filename)

    m = max(maximum(data.arows),length(data.ucon),length(data.lcon))
    d = max(maximum(data.acols),length(data.uvar), length(data.lvar))

    A1 = sparse(data.arows, data.acols, data.avals, m, d)
    A2 = copy(A1)

    b1 = data.ucon
    b2 = data.lcon

    idx = findall(.!isinf.(b1))
    b1 = b1[idx]
    A1 = A1[idx,:]

    idx = findall(.!isinf.(b2))
    b2 = b2[idx]
    A2 = A2[idx,:]

    A = vcat(A1, -A2)
    b = vcat(b1, -b2)

    uBound = data.uvar
    lBound = data.lvar

    idx = findall(.!isinf.(uBound))
    if length(idx) > 0
        uBound = uBound[idx]
        I_u = sparse(1:length(idx), idx, 1.0, length(idx), d )
        A = vcat(A, I_u)
        b = vcat(b, uBound)
    end

    idx = findall(.!isinf.(lBound))
    if length(idx) > 0
        lBound = lBound[idx]
        I_l = sparse(1:length(idx), idx, 1.0, length(idx), d )
        A = vcat(A, -I_l)
        b = vcat(b, -lBound)
    end

    return A, b
end


