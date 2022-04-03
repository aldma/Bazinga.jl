
"""
    NormLpPowerBox(p;alpha=1, u)

With a nonnegative scalar `alpha`, return the Lp-quasi-norm
```math
f(x) = alpha \\|x\\|_p^p := alpha sum_i |x_i|^p.
```
"""
struct NormLpPowerBox{T}
    p::T
    alpha::T
    u::Vector{T}
    n::Int
end

function NormLpPowerBox(p::T, alpha::T = 1; u::Vector{T}) where {T}
    if p <= 0
        error("p must be positive")
    elseif p >= 1
        error("p must be smaller than one")
    end
    if alpha < 0
        error("alpha must be nonnegative")
    end
    if any(u .< 0)
        error("vector u must have nonnegative entries")
    end
    n = length(u)
    return NormLpPowerBox(p, alpha, u, n)
end

function (f::NormLpPowerBox{T})(x) where {T<:Real}
    return f.alpha * sum(x .^ f.p)
end

function prox!(y::Vector, f::NormLpPowerBox{<:Real}, x::Vector, gamma::Number)
    y .= x
    for i in eachindex(x)
        y[i] = solve_lp_quasi_norm_subproblem_box(x[i], f.p, f.alpha, gamma, f.u[i])
    end
    return f.alpha * sum(y .^ f.p)
end



function solve_lp_quasi_norm_subproblem_box(x::Real, p::Real, a::Real, gamma::Real, u::Real)
    # min    a ||z||_p^p + 1/(2 gamma) (z-x)^2
    # wrt    z
    # s.t.   0 ≤ z ≤ u
    #
    # with   x ∈ ℜ, a > 0, p ∈ (0,1), gamma > 0, u ≥ 0
    T = eltype(x)
    iter = 0
    maxiter = 1_000
    epsilon = 1e-12

    alpha = a * gamma
    if x <= 0 || u == 0
        return T(0)
    end
    zbar = (1 / (alpha * p * (1 - p)))^(1 / (p - 2))
    psi_zbar = zbar + alpha * p * zbar^(p - 1)
    if psi_zbar >= x
        return T(0)
    end
    # apply Newton's method
    z = T(zbar) + 0.1 # perturbation to the right!
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
    if z > u
        phi_u = 0.5 * (u - x)^2 + alpha * u^p
        if phi_u < phi_0
            return u
        else
            return T(0)
        end
    else
        return z
    end
end
