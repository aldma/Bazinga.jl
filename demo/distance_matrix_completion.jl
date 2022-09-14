"""
	distance_matrix_completion : example from Bazinga.jl

    Given some observed entries, find a complete matrix with minimum rank:

	minimize      g(X)
	subject to    X_{i,i} + X_{j,j} - X_{i,j} - X_{j,i} = D_{i,j}   ∀(i,j) ∈ observations
                  X_{i,j} = X_{j,i}                                 ∀ i,j, j < i

	Reformulation as a constrained structured problem in the form

	minimize     f(X) + g(X)
	subject to   c(X) in D

	where
	    f(x) = 0
	    c(x) = {observations and symmetry constraints}
	    D    = {0}
    and g ∈ {rank, Schatten-p-norm, nuclear norm}
"""

using LinearAlgebra
using Bazinga
using ProximalAlgorithms
using DataFrames
using CSV
using Statistics
using Random

rng = Xoshiro(123)

function sampledDistanceMatrix(N, nobs, l)
    T = Float64
    X = randn(rng, T, N, l)
    D = zeros(T, N, N)
    for i = 1:N
        for j = i+1:N
            D[i, j] = sum((X[i, :] - X[j, :]) .^ 2)
            D[j, i] = D[i, j]
        end
    end
    @info "Distance matrix D has rank $(rank(D))"

    idx = randperm(rng, N * N)[1:nobs]
    sort!(idx)
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
    X = Bazinga.check_and_reshape_as_matrix(out[1])
    push!(
        data,
        (
            id = id,
            iters = out[3],
            subiters = out[4],
            runtime = out[5],
            cviolation = out[7],
            rank = rank(X),
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
function ConstraintDMC(N::Int, iobs::Vector, jobs::Vector, vobs::Vector)
    nobs = length(vobs)
    nsym = Int(N * (N - 1) / 2)
    ny = nobs + nsym
    # indices for  symmetry constraints
    isym = Vector{Int}(undef, nsym)
    jsym = Vector{Int}(undef, nsym)
    k = 0
    for i = 1:N
        for j = i+1:N
            k += 1
            isym[k] = i
            jsym[k] = j
        end
    end
    return ConstraintDMC(ny, nobs, iobs, jobs, vobs, nsym, isym, jsym)
end
function Bazinga.eval!(cB::Vector, c::ConstraintDMC, B::Matrix)
    # observations
    for k = 1:c.nobs
        i = c.iobs[k]
        j = c.jobs[k]
        cB[k] = B[i, i] + B[j, j] - B[i, j] - B[j, i] - c.vobs[k]
    end
    # symmetry
    for k = 1:c.nsym
        i = c.isym[k]
        j = c.jsym[k]
        cB[c.nobs+k] = B[i, j] - B[j, i]
    end
    return nothing
end
function Bazinga.jtprod!(jtv::Matrix, c::ConstraintDMC, B::Matrix, v::Vector)
    jtv .= 0.0
    # observations
    for k = 1:c.nobs
        i = c.iobs[k]
        j = c.jobs[k]
        jtv[i, i] += v[k]
        jtv[j, j] += v[k]
        jtv[i, j] -= v[k]
        jtv[j, i] -= v[k]
    end
    # symmetry
    for k = 1:c.nsym
        i = c.isym[k]
        j = c.jsym[k]
        jtv[i, j] += v[c.nobs+k]
        jtv[j, i] -= v[c.nobs+k]
    end
    return nothing
end
function Bazinga.eval!(cB::Vector, c::ConstraintDMC, x::Vector)
    X = Bazinga.check_and_reshape_as_matrix(x)
    Bazinga.eval!(cB, c, X)
    return nothing
end
function Bazinga.jtprod!(jtv::Matrix, c::ConstraintDMC, x::Vector, v::Vector)
    X = Bazinga.check_and_reshape_as_matrix(x)
    Bazinga.jtprod!(jtv, c, X, v)
    return nothing
end
function Bazinga.jtprod!(jtv::Vector, c::ConstraintDMC, x::Vector, v::Vector)
    X = Bazinga.check_and_reshape_as_matrix(x)
    Y = Bazinga.check_and_reshape_as_matrix(jtv)
    Bazinga.jtprod!(Y, c, X, v)
    jtv .= Y[:]
    return nothing
end

################################################################################
# solve problems
################################################################################

problem_name = "dist_matrix_completion"
basepath = joinpath(@__DIR__, "results", problem_name)

T = Float64
N = 20 # 10, 20
nsym = Int(N * (N - 1) / 2)
nobs = Int(floor((N * N - nsym) / 3))
l = 5
pSchatten = 0.5

ntests = 30

prob_f = Bazinga.Zero()
prob_g_nuclear = Bazinga.NuclearNorm()
prob_g_schatten = Bazinga.SchattenNormLpPower(pSchatten)
prob_g_rank = Bazinga.Rank()
prob_D = Bazinga.ZeroSet()

subsolver_directions = ProximalAlgorithms.LBFGS(5)
subsolver_minimum_gamma = 1e-32
subsolver_maxit = 1_000_000
subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = subsolver_directions,
    maxit = subsolver_maxit,
    freq = subsolver_maxit,
    minimum_gamma = subsolver_minimum_gamma;
    kwargs...,
)
solver(g, c, x0, y0; kwargs...) = Bazinga.als(
    prob_f,
    g,
    c,
    prob_D,
    x0,
    y0,
    verbose = true,
    subsolver = subsolver;
    subsolver_maxit = subsolver_maxit,
    kwargs...,
)

data = Dict{Symbol,DataFrame}()
keys = [:nuclear, :rank, :schatten, :schattenrank, :nuclearrank]
for k in keys
    data[k] = DataFrame()
end

for id = 1:ntests
    @info "===== Problem $(id) of $(ntests) ====="
    Iobs, Jobs, Vobs = sampledDistanceMatrix(N, nobs, l)
    prob_c = ConstraintDMC(N, Iobs, Jobs, Vobs)
    nx = N * N
    ny = prob_c.ny
    prob_x0 = randn(rng, T, N * N)
    prob_y0 = zeros(T, ny)

    # rank
    @info "RANK:"
    out = solver(prob_g_rank, prob_c, prob_x0, prob_y0)
    push_out_to_data(data[:rank], id, out)

    # Schatten
    @info "SCHATTEN NORM:"
    out = solver(prob_g_schatten, prob_c, prob_x0, prob_y0)
    push_out_to_data(data[:schatten], id, out)
    # Schatten + rank
    @info "SCHATTEN NORM + RANK:"
    out = solver(prob_g_rank, prob_c, out[1], out[2])
    push_out_to_data(data[:schattenrank], id, out)

    # nuclear
    @info "NUCLEAR NORM:"
    out = solver(prob_g_nuclear, prob_c, prob_x0, prob_y0)
    push_out_to_data(data[:nuclear], id, out)
    # nuclear + rank
    @info "NUCLEAR NORM + RANK:"
    out = solver(prob_g_rank, prob_c, out[1], out[2])
    push_out_to_data(data[:nuclearrank], id, out)

end

filename = "N" * string(N)
filepath = joinpath(basepath, filename)

for k in keys
    CSV.write(filepath * "_" * String(k) * ".csv", data[k], header = true)
end

for k in keys
    @info uppercase(String(k))
    @info "    rank: min $(minimum(data[k].rank)), max $(maximum(data[k].rank)), median  $(median(data[k].rank))"
    @info "subiters: min $(minimum(data[k].subiters)), max $(maximum(data[k].subiters)), median  $(median(data[k].subiters))"
end
