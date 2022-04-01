# rank (times a constant)

"""
    Rank(λ=1)
Return the rank
```math
f(X) = λ\\cdot\\mathrm{rank}(X) = λ\\cdot\\mathrm{nnz}σ(X),
```
where `λ` is a positive parameter and ``σ(X)`` are the singular values of matrix ``X``.
"""
struct Rank{R} <: Bazinga.ProximableFunction
    lambda::R
    function Rank{R}(lambda::R) where {R}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

Rank(lambda::R = 1.0) where {R} = Rank{R}(lambda)

function (f::Rank)(X)
    return f.lambda * rank(X)
end

function Bazinga.prox!(Y, f::Rank, X, gamma)
    F = svd(X)
    over = F.S .> sqrt(2 * gamma * f.lambda)
    mul!(Y, F.U, Diagonal(F.S .* over) * F.Vt)
    return f.lambda * sum(over)
end
