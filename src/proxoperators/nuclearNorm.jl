# nuclear norm (times a constant)

"""
    NuclearNorm(λ=1)
Return the nuclear norm
```math
f(X) = λ \\cdot \\|X\\|_* = λ ∑_i σ_i(X),
```
where `λ` is a positive parameter and ``σ(X)`` are the singular values of matrix ``X``.
"""
struct NuclearNorm{R} <: Bazinga.ProximableFunction
    lambda::R
    function NuclearNorm{R}(lambda::R) where {R}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

NuclearNorm(lambda::R = 1.0) where {R} = NuclearNorm{R}(lambda)

function (f::NuclearNorm)(X)
    F = svd(X)
    return f.lambda * sum(F.S)
end

function Bazinga.prox!(Y, f::NuclearNorm, X, gamma)
    F = svd(X)
    F.S .= max.(0, F.S .- f.lambda * gamma) # overwrite
    mul!(Y, F.U, Diagonal(F.S) * F.Vt) # in-place
    return f.lambda * sum(F.S)
end

# vectorized input
function (f::NuclearNorm)(x::AbstractVector)
    X = check_and_reshape_as_matrix(x)
    return f(X)
end

function Bazinga.prox!(y::AbstractVector, f::NuclearNorm, x::AbstractVector, gamma)
    X = check_and_reshape_as_matrix(x)
    Y = check_and_reshape_as_matrix(y)
    fy = prox!(Y, f, X, gamma)
    y .= Y[:]
    return fy
end
