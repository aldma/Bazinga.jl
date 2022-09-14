"""
	portfolio : example from Bazinga.jl

    Given some observed entries, find a complete matrix with minimum rank:

	minimize      g(X)
	subject to    X_{i,i} + X_{j,j} - X_{i,j} - X_{j,i} = D_{i,j}   ∀(i,j) ∈ observations
                  X_{i,j} = X_{j,i}                                 ∀ i,j, j < i

	Reformulation as a constrained structured problem in the form

	minimize     f(X) + g(X)
	subject to   c(X) in D

	where
	    f(x) = 1/2 x' Q x
        g(x) = alpha ||x||_p^p + indicator_[0, u](x)
	    c(x) = [mu' x; e' x]
	    D    = [rho, ∞) × {1}
"""

using LinearAlgebra
using Bazinga
using ProximalAlgorithms
using ProximalOperators
using DataFrames
using CSV
using Statistics

###################################################################################
# problem definition
###################################################################################
struct SmoothCostP <: Bazinga.ProximableFunction
    Q::Matrix
end
function (f::SmoothCostP)(x)
    return 0.5 * dot(x, f.Q, x)
end
function Bazinga.gradient!(dfx, f::SmoothCostP, x)
    mul!(dfx, f.Q, x)
    return 0.5 * dot(x, dfx)
end

struct ConstraintP <: SmoothFunction
    mu::Vector
end
function Bazinga.eval!(cx::Vector, c::ConstraintP, x::Vector)
    cx[1] = dot(c.mu, x)
    cx[2] = sum(x)
    return nothing
end
function Bazinga.jtprod!(jtv::Vector, c::ConstraintP, x::Vector, v::Vector)
    jtv .= c.mu .* v[1] .+ v[2]
    return nothing
end

struct SetP <: ClosedSet
    rho::Real
end
function Bazinga.proj!(z, D::SetP, cx)
    z .= cx
    z[1] = max(D.rho, cx[1])
    z[2] = 1
    return nothing
end

################################################################################
# load problem data
################################################################################
function read_FG_problem_data(foldername)
    T = Float64
    out = CSV.read(joinpath(foldername, "Q.csv"), DataFrame, header = false)
    Q = T.(Tables.matrix(out))
    out = CSV.read(joinpath(foldername, "rho.csv"), DataFrame, header = false)
    rho = T.(Tables.matrix(out))[:][1] # scalar
    out = CSV.read(joinpath(foldername, "mu.csv"), DataFrame, header = false)
    mu = T.(Tables.matrix(out))[:] # vector
    out = CSV.read(joinpath(foldername, "ub.csv"), DataFrame, header = false)
    ub = T.(Tables.matrix(out))[:] # vector

    nx = length(mu)
    @assert length(rho) == 1
    @assert length(ub) == nx
    @assert size(Q, 1) == nx
    @assert size(Q, 2) == nx

    return Q, rho, mu, ub
end

function push_out_to_data(data, id, out, Q)
    x = out[1]
    objectiveq = 0.5 * dot(x, Q, x)
    push!(
        data,
        (
            id = id,
            iters = out[3],
            subiters = out[4],
            runtime = out[5],
            cviolation = out[7],
            objectiveq = objectiveq,
            nnz = sum(x .> 0),
        ),
    )
    return nothing
end

################################################################################
# solve problems
################################################################################

problem_name = "portfolio"
attribu_name = "dim200"
basepath = joinpath(@__DIR__, "results", problem_name)

basefoldername = joinpath(@__DIR__, "portfolio_data", attribu_name)
problems = readdir(basefoldername)
nproblems = length(problems)

T = Float64
data_a = T(100) # 1, 10, 100
pnorm = 0.5

subsolver_directions = ProximalAlgorithms.LBFGS(5)
subsolver_maxit = 1_000
subsolver_minimum_gamma = 1e-32
subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = subsolver_directions,
    maxit = subsolver_maxit,
    freq = subsolver_maxit,
    minimum_gamma = subsolver_minimum_gamma;
    kwargs...,
)

data = Dict{Symbol,DataFrame}()
keys = [:l0, :l1, :l10, :lp, :lp0] # :l1p, :l1p0
for k in keys
    data[k] = DataFrame()
end

for id in eachindex(problems)
    # load problem data
    foldername = joinpath(basefoldername, problems[id])
    data_Q, data_rho, data_mu, data_ub = read_FG_problem_data(foldername)

    n = length(data_mu)

    prob_f = SmoothCostP(data_Q)
    prob_c = ConstraintP(data_mu)
    prob_D = SetP(data_rho)
    prob_g_l1 = Bazinga.NormL1Box(data_a, u = data_ub)
    prob_g_lp = Bazinga.NormLpPowerBox(pnorm, data_a, u = data_ub)
    prob_g_l0 = Bazinga.NormL0Box(data_a, u = data_ub)

    # build solver
    solver(g, X0, y0; kwargs...) = Bazinga.als(
        prob_f,
        g,
        prob_c,
        prob_D,
        X0,
        y0,
        verbose = true,
        subsolver = subsolver;
        kwargs...,
    )

    prob_X0 = ones(T, n)
    prob_y0 = zeros(T, 2)

    @info "===== L0 ====="
    out = solver(prob_g_l0, prob_X0, prob_y0)
    push_out_to_data(data[:l0], id, out, data_Q)
    nnz0 = sum(out[1] .> 0)

    @info "===== Lp ====="
    out = solver(prob_g_lp, prob_X0, prob_y0)
    push_out_to_data(data[:lp], id, out, data_Q)
    nnzP = sum(out[1] .> 0)

    @info "===== Lp + L0 ====="
    out = solver(prob_g_l0, out[1], out[2])
    push_out_to_data(data[:lp0], id, out, data_Q)
    nnzP0 = sum(out[1] .> 0)

    @info "===== L1 ====="
    out = solver(prob_g_l1, prob_X0, prob_y0)
    push_out_to_data(data[:l1], id, out, data_Q)
    nnz1 = sum(out[1] .> 0)

    @info "===== L1 + L0 ====="
    out = solver(prob_g_l0, out[1], out[2])
    push_out_to_data(data[:l10], id, out, data_Q)
    nnz10 = sum(out[1] .> 0)

    #=@info "===== L1 ====="
    out = solver(prob_g_l1, prob_X0, prob_y0)

    @info "===== L1 + Lp ====="
    out = solver(prob_g_lp, out[1], out[2])
    push_out_to_data(data[:l1p], id, out, data_Q)
    nnz1P = sum(out[1] .> 0)

    @info "===== L1 + Lp + L0 ====="
    out = solver(prob_g_l0, out[1], out[2])
    push_out_to_data(data[:l1p0], id, out, data_Q)
    nnz1P0 = sum(out[1] .> 0)=#

    @info "nnz:" nnz0 nnzP nnzP0 nnz1 nnz10 #nnz1P nnz1P0

end

filename = attribu_name * "_a" * string(Int(data_a))
filepath = joinpath(basepath, filename)

for k in keys
    CSV.write(filepath * "_" * String(k) * ".csv", data[k], header = true)
end

for k in keys
    @info uppercase(String(k))
    @info "       nnz: min $(minimum(data[k].nnz)), max $(maximum(data[k].nnz)), median  $(median(data[k].nnz))"
    @info " cviolation: min $(minimum(data[k].cviolation)), max $(maximum(data[k].cviolation)), median  $(median(data[k].cviolation))"
    @info "inner iters: min $(minimum(data[k].subiters)), max $(maximum(data[k].subiters)), median  $(median(data[k].subiters))"
end
