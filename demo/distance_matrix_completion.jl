"""
	matrix_completion : example from Bazinga.jl

    Given some observed entries, find a complete matrix with minimum rank:

	minimize      g(X)
	subject to    X_{i,j} = Z_{i,j}  ∀(i,j) ∈ observations

	Reformulation as a constrained structured problem in the form

	minimize     f(X) + g(X)
	subject to   c(X) in D

	where
	    f(x) = 0
	    g(x) = rank(X)
	    c(x) = X_{i,j}[:] - Z_{i,j}[:]
	    D    = 0
"""

using LinearAlgebra
using Bazinga
using ProximalAlgorithms
using ProximalOperators

using Random
Random.seed!(123456)

function sampledDistanceMatrix(n, nobs, l)
    T = Float64
    X = randn(T, n, l)
    D = zeros(T, n, n)
    for i = 1:n
        for j = i+1:n
            D[i,j] = sum( (X[i,:] - X[j,:]).^2 )
            D[j,i] = D[i,j]
        end
    end
    @info "Distance matrix D has rank $(rank(D))"

    idx = randperm(n * n)[1:nobs]
    sort!(idx)
    #idx = LinearIndices(D)[idx]
    #d = D[idx]
    IJ = CartesianIndices(D)[idx]
    Iobs = zeros(Int, nobs)
    Jobs = zeros(Int, nobs)
    for k = 1:nobs
        Iobs[k] = IJ[k][1]
        Jobs[k] = IJ[k][2]
    end
    Vobs = D[IJ]
    return Iobs, Jobs, Vobs
end

function push_out_to_data(data, id, out)
    push!(
        data,
        (
            id = id,
            iters = out[3],
            subiters = out[4],
            runtime = out[5],
            cviolation = out[7],
            rank = rank(out[1]),
        ),
    )
    return nothing
end

###################################################################################
# problem definition
###################################################################################

struct ConstraintDMC <: SmoothFunction
    ny::Int
    nobs::Int
    iobs::Vector
    jobs::Vector
    vobs::Vector
    nsym::Int
    isym::Vector
    jsym::Vector
end
function ConstraintDMC(n::Int, iobs::Vector, jobs::Vector, vobs::Vector)
    nobs = length(vobs)
    nsym = Int(n*(n-1)/2)
    ny = nobs + nsym
    # indices for  symmetry constraints
    isym = Vector{Int}(undef,nsym)
    jsym = Vector{Int}(undef,nsym)
    #lidx = LinearIndices((n,n))
    #idxsym = Vector{Int}(undef,nsym)
    #idxsymt = Vector{Int}(undef,nsym)
    k = 0
    for i = 1:n
        for j = i+1:n
            k += 1
            isym[k] = i
            jsym[k] = j
            #idxsymt[k] = lidx[i,j]
            #idxsym[k] = lidx[j,i]
        end
    end
    return ConstraintDMC(ny, nobs, iobs, jobs, vobs, nsym, isym ,jsym)
end
function Bazinga.eval!(cB::Vector, c::ConstraintDMC, B::Matrix)
    # observations
    for k = 1:c.nobs
        i = c.iobs[k]
        j = c.jobs[k]
        cB[k] = B[i,i] + B[j,j] - B[i,j] - B[j,i] - c.vobs[k]
    end
    #...cx[1:c.nobs] .= X[c.idxobs] - c.valobs
    # symmetry
    for k = 1:c.nsym
        i = c.isym[k]
        j = c.jsym[k]
        cB[c.nobs+k] = B[i,j] - B[j,i]
    end
    #...cB[c.nobs+1:c.nobs+c.nsym] .= B[c.idxsym] - B[c.idxsymt]
    return nothing
end
function Bazinga.jtprod!(jtv::Matrix, c::ConstraintDMC, B::Matrix, v::Vector)
    jtv .= 0.0
    # observations
    for k = 1:c.nobs
        i = c.iobs[k]
        j = c.jobs[k]
        jtv[i,i] += v[k]
        jtv[j,j] += v[k]
        jtv[i,j] -= v[k]
        jtv[j,i] -= v[k]
    end
    # symmetry
    for k = 1:c.nsym
        i = c.isym[k]
        j = c.jsym[k]
        jtv[i,j] += v[c.nobs+k]
        jtv[j,i] -= v[c.nobs+k]
    end
    #...jtv[c.idxsym] .-= v[c.nobs+1:c.nobs+c.nsym]
    #...jtv[c.idxsymt] .+= v[c.nobs+1:c.nobs+c.nsym]
    return nothing
end

################################################################################
# solve problems
################################################################################
using DataFrames
using CSV
using Statistics

problem_name = "dmc"

T = Float64
n = 10 # 50
nobs = 30 # Int(floor(n*n*0.2)) # 150
l = 3 # 3
pSchatten = 0.5

ntests = 10

prob_f = Bazinga.Zero()
prob_g_nuclear = ProximalOperators.NuclearNorm()
prob_g_schatten = Bazinga.SchattenNormLpPower(pSchatten)
prob_g_rank = Bazinga.Rank()
prob_D = Bazinga.ZeroSet()

subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = ProximalAlgorithms.LBFGS(5),
    maxit = 10_000,
    freq = 10_000,
    minimum_gamma = eps(T);
    kwargs...,
)
solver(g, c, X0, y0; kwargs...) = Bazinga.alps(
    prob_f,
    g,
    c,
    prob_D,
    X0,
    y0,
    verbose = true,
    subsolver = subsolver;
    kwargs...,
)

data = Dict{Symbol,DataFrame}()
keys = [:nuclear, :rank, :schatten, :schattenrank, :nuclearrank]
for k in keys
    data[k] = DataFrame()
end

for id = 1:ntests

    Iobs, Jobs, Vobs = sampledDistanceMatrix(n, nobs, l)
    prob_c = ConstraintDMC(n, Iobs, Jobs, Vobs)
    nx = n*n
    ny = prob_c.ny
    prob_X0 = randn(T, n, n)
    prob_y0 = zeros(T, ny)

    # Schatten, nuclear norm, rank
    for (prob_g, data) in [(prob_g_schatten, data[:schatten]), (prob_g_nuclear, data[:nuclear]), (prob_g_rank, data[:rank])]
        out = solver(prob_g, prob_c, prob_X0, prob_y0)
        push_out_to_data(data, id, out)
    end

    # Schatten + rank
    out = solver(prob_g_schatten, prob_c, prob_X0, prob_y0)
    out = solver(prob_g_rank, prob_c, out[1], out[2])
    push_out_to_data(data[:schattenrank], id, out)

    # nuclear + rank
    out = solver(prob_g_nuclear, prob_c, prob_X0, prob_y0)
    out = solver(prob_g_rank, prob_c, out[1], out[2])
    push_out_to_data(data[:nuclearrank], id, out)

end

filename = problem_name
filepath = joinpath(@__DIR__, "results", filename)

for k in keys
    CSV.write(filepath * "_" * String(k) * ".csv", data[k], header = true)
end

for k in keys
    @info uppercase(String(k))
    @info "       rank: min $(minimum(data[k].rank)), max $(maximum(data[k].rank)), median  $(median(data[k].rank))"
    @info " cviolation: min $(minimum(data[k].cviolation)), max $(maximum(data[k].cviolation)), median  $(median(data[k].cviolation))"
    @info "inner iters: min $(minimum(data[k].subiters)), max $(maximum(data[k].subiters)), median  $(median(data[k].subiters))"
end
