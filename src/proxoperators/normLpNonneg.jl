# Lp-norm to the power p (times a constant), for 0 < p < 1

using LinearAlgebra
using SparseArrays

"""
    NormLpPowerNonneg(p;alpha=1)

With a nonnegative scalar `alpha`, return the Lp-quasi-norm
```math
f(x) = alpha \\|x\\|_p^p := alpha sum_i |x_i|^p.
```
"""
struct NormLpPowerNonneg{T}
    p::T
    alpha::T
    function NormLpPowerNonneg{T}(p::T, alpha::T) where {T}
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

NormLpPowerNonneg(p::T; alpha::T = T(1)) where {T} = NormLpPowerNonneg{T}(p, alpha)

function (f::NormLpPowerNonneg{T})(x) where {T<:Real}
    return f.alpha * sum(x .^ f.p)
end

function prox!(y::Vector, f::NormLpPowerNonneg{<:Real}, x::Vector, gamma::Number)
    y .= x
    solve_lp_quasi_norm_subproblem_nonneg!(y, f.p, f.alpha, gamma)
    return f.alpha * sum(y .^ f.p)
end



function solve_lp_quasi_norm_subproblem_nonneg(x::Real, p::Real, a::Real, gamma::Real)
    # min    a ||z||_p^p + 1/(2 gamma) (z-x)^2
    # wrt    z
    # s.t.   z ≥ 0
    #
    # with   x ∈ ℜ, a > 0, p ∈ (0,1), gamma > 0
    T = eltype(x)
    iter = 0
    maxiter = 1_000
    epsilon = 1e-12

    alpha = a * gamma
    if x <= 0
        return T(0)
    end
    zbar = (1 / (alpha * p * (1 - p)))^(1 / (p - 2))
    psi_zbar = zbar + alpha * p * zbar^(p - 1)
    if psi_zbar >= x
        return T(0)
    end
    # apply Newton's method
    z = T(zbar) + T(1) # perturbation to the right!
    while iter < maxiter
        dphi_z = z - x + alpha * p * z^(p - 1)
        if abs(dphi_z) <= epsilon
            break
        end
        iter += 1
        if iter >= maxiter
            @warn "max iter prox lp, $(abs(dphi_z))"
        end
        ddphi_z = 1 + alpha * p * (p - 1) * z^(p - 2)
        z -= dphi_z / ddphi_z
    end
    # test for global minimum
    phi_0 = 0.5 * x^2
    phi_z = 0.5 * (z - x)^2 + alpha * z^p
    if phi_0 <= phi_z
        return T(0)
    end
    return z
end

function solve_lp_quasi_norm_subproblem_nonneg!(x::Array, p::Real, a::Real, gamma::Real)
    map!(el -> solve_lp_quasi_norm_subproblem_nonneg(el, p, a, gamma), x, x)
    return nothing
end
