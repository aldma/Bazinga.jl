# L1 norm (times a constant, or weighted)

"""
    NormL1Nonneg(λ=1)

With a nonnegative scalar parameter λ, return the ``L_1`` norm
```math
f(x) = λ\\cdot∑_i|x_i|.
```
"""
struct NormL1Nonneg{T}
    lambda::T
    function NormL1Nonneg{T}(lambda::T) where {T}
        if !(eltype(lambda) <: Real)
            error("λ must be real")
        end
        if any(lambda .< 0)
            error("λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

NormL1Nonneg(lambda::R = 1) where {R} = NormL1Nonneg{R}(lambda)

(f::NormL1Nonneg)(x) = f.lambda * norm(x, 1)

function prox!(y, f::NormL1Nonneg, x::AbstractArray{<:Real}, gamma)
    @assert length(y) == length(x)
    n1y = eltype(x)(0)
    gl = gamma * f.lambda
    @inbounds @simd for i in eachindex(x)
        if x[i] >= gl
            y[i] = x[i] - gl
            n1y += y[i]
        else
            y[i] = 0
        end
    end
    return f.lambda * n1y
end

function prox_naive(f::NormL1Nonneg, x, gamma)
    y = max.(0, x .- gamma .* f.lambda)
    return y, norm(f.lambda .* y, 1)
end
