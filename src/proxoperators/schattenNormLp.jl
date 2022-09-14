# Schatten-p-norm to the power p (times a constant), for 0 < p < 1

using LinearAlgebra
using SparseArrays

"""
    SchattenNormLpPower(p;alpha=1)

With a nonnegative scalar `alpha`, return the Schatten-p-norm
```math
f(X) = alpha \\|X\\|_p^p.
```
"""
struct SchattenNormLpPower{T} <: Bazinga.ProximableFunction
    p::T
    alpha::T
    function SchattenNormLpPower{T}(p::T, alpha::T) where {T}
        if p <= 0
            error("p must be positive")
        elseif p >= 1
            error("p must be smaller than one")
        end
        if alpha < 0
            error("alpha must be nonnegative")
        end
        new(p, alpha)
    end
end

SchattenNormLpPower(p::T; alpha::T = T(1)) where {T} = SchattenNormLpPower{T}(p, alpha)

function (f::SchattenNormLpPower{T})(X) where {T<:Real}
    return f.alpha * sum(svdvals(X) .^ f.p)
end

function prox!(Y::Matrix, f::SchattenNormLpPower{<:Real}, X::Matrix, gamma::Number)
    svdec = svd(X)
    sigmas = svdec.S
    solve_lp_quasi_norm_subproblem!(sigmas, f.p, f.alpha, gamma)
    Y .= svdec.U * spdiagm(sigmas) * svdec.Vt
    return f.alpha * sum(sigmas .^ f.p)
end

# vectorized input
function (f::SchattenNormLpPower)(x::AbstractVector)
    X = check_and_reshape_as_matrix(x)
    return f(X)
end

function Bazinga.prox!(y::AbstractVector, f::SchattenNormLpPower, x::AbstractVector, gamma)
    X = check_and_reshape_as_matrix(x)
    Y = check_and_reshape_as_matrix(y)
    fy = prox!(Y, f, X, gamma)
    y .= Y[:]
    return fy
end


#============================================================================#
function solve_lp_quasi_norm_subproblem(x::Real, p::Real, a::Real, gamma::Real)
    # min    a ||z||_p^p + 1/(2 gamma) (z-x)^2
    # wrt    z
    # s.t.   z ≥ 0
    #
    # with   x ∈ ℜ, a > 0, p ∈ (0,1), gamma > 0
    T = eltype(x)
    iter = 0
    maxiter = 1_000
    epsilon = 1e-14

    alpha = a * gamma
    if x <= 0
        z = T(0)
    else
        zbar = (1 / (alpha * p * (1 - p)))^(1 / (p - 2))
        psi_zbar = zbar + alpha * p * zbar^(p - 1)
        if psi_zbar >= x
            z = T(0)
        else
            # apply Newton's method
            z = T(zbar) + 0.1 # perturbation to the right!
            while iter < maxiter
                dphi_z = z - x + alpha * p * z^(p - 1)
                if abs(dphi_z) <= epsilon
                    break
                end
                iter += 1
                ddphi_z = 1 + alpha * p * (p - 1) * z^(p - 2)
                z -= dphi_z / ddphi_z
            end
            # test for global minimum
            phi_0 = 0.5 * x^2
            phi_z = 0.5 * (z - x)^2 + alpha * z^p
            if phi_0 <= phi_z
                z = T(0)
            end
        end
    end
    return z
end

function solve_lp_quasi_norm_subproblem!(x::Array, p::Real, a::Real, gamma::Real)
    map!(el -> solve_lp_quasi_norm_subproblem(el, p, a, gamma), x, x)
    return nothing
end
