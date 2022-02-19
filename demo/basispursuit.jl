"""
	basis pursuit : example from Bazinga.jl

    Academic example of mathematical program with vanishing constraints (MPVC),
    see Section 9.5.1 [Hoh09].

	Original formulations:

	minimize      \| x \|_1
	subject to    A x = b

	Reformulation as a constrained structured problem in the form

	minimize     f(x) + g(x)
	subject to   c(x) in D

	where
	    f(x) = 0
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
problem_name = "basis_pursuit"

T = Float64
rng = MersenneTwister(0)
nx = 30
ny = 10
A = randn(rng, T, (ny,nx))
xtrue = sprandn(rng, T, nx, 0.1)
b = A * xtrue

f = ProximalOperators.Zero()
g = ProximalOperators.NormL1()
c = ConstraintBasisPursuit(A, b)
D = Bazinga.ZeroSet()

# solver setup
subsolver_directions = ProximalAlgorithms.LBFGS(5)
subsolver_maxit = 1_000_000
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
out = solver(f, g, c, D, ones(T, nx), zeros(T, ny))
