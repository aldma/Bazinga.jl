"""
	basis pursuit : example from Bazinga.jl

	Original formulations:

	minimize      | x |_1
	subject to    A x = b

	Reformulation as a constrained structured problem in the form

	minimize     f(x) + g(x)
	subject to   c(x) in D

	where
	    f(x) = Zero()
	    g(x) = NormL1(x)
	    c(x) = A x - b
	    D    = ZeroSet()

    References:
    [BPC11]     Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
                S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein
                Foundations and Trends in Machine Learning, 3(1):1-122, 2011.
                https://web.stanford.edu/~boyd/papers/admm/
"""

using ProximalOperators
using LinearAlgebra
using Bazinga
using ProximalAlgorithms

using Random
using SparseArrays

################################################################################
# problem definition
################################################################################
struct ConstraintBasisPursuit <: SmoothFunction
    A::AbstractMatrix
    b::AbstractVector
end
function Bazinga.eval!(cx, c::ConstraintBasisPursuit, x)
    cx .= c.A * x - c.b
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintBasisPursuit, x, v)
    jtv .= c.A' * v
    return nothing
end

# iter
problem_name = "basispursuit"

T = Float64
rng = MersenneTwister(0)
nx = 100
ny = 20
A = randn(rng, T, (ny, nx))
xtrue = sprandn(rng, T, nx, 0.1)
b = A * xtrue

f = ProximalOperators.Zero()
g = ProximalOperators.NormL1()
g0 = ProximalOperators.NormL0()
c = ConstraintBasisPursuit(A, b)
D = Bazinga.ZeroSet()

# solver setup
subsolver_directions = ProximalAlgorithms.LBFGS(5)
subsolver_maxit = 100_000
subsolver_minimum_gamma = eps(T)
subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = subsolver_directions,
    maxit = subsolver_maxit,
    freq = subsolver_maxit,
    minimum_gamma = subsolver_minimum_gamma;
    kwargs...,
)
solver(f, g, c, D, x0, y0) = Bazinga.alps(
    f,
    g,
    c,
    D,
    x0,
    y0,
    verbose = true,
    subsolver = subsolver,
    subsolver_maxit = subsolver_maxit,
)


out0 = solver(f, g0, c, D, zeros(T, nx), zeros(T, ny))
xsol0 = out0[1]
ysol0 = out0[2]

out1 = solver(f, g, c, D, zeros(T, nx), zeros(T, ny))
xsol1 = out1[1]
ysol1 = out1[2]

out0warm = solver(f, g0, c, D, xsol1, ysol1)
xsol0warm = out0warm[1]
ysol0warm = out0warm[2]

function (c::ConstraintBasisPursuit)(x)
    cx = similar(c.b)
    eval!(cx, c, x)
    return cx
end

objectiveFun0(x) = f(x) + g0(x)
cviolationFun(x) = norm(c(x), Inf)

print_info(x, y) = begin
    obj = objectiveFun0(x)
    cvl = cviolationFun(x)
    @info "obj = $(obj), cviol = $(cvl)"
    return nothing
end
print_info(x) = print_info(x, 0)

print_info(xtrue)
print_info(xsol0)
print_info(xsol0warm)
