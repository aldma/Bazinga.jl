# L1 norm (times a constant) with a box constraint

"""
    NormL1Box(λ=1)

With a nonnegative scalar parameter λ, return the ``L_1`` norm
```math
f(x) = λ\\cdot∑_i|x_i|.
```
"""
struct NormL1Box{T}
    lambda::T
    u::Vector{T}
    n::Int
end

function NormL1Box(lambda::T = 1; u::Vector{T}) where {T}
    if lambda < 0
        error("parameter λ must be nonnegative")
    end
    if any(u .< 0)
        error("vector u must have nonnegative entries")
    end
    n = length(u)
    return NormL1Box(lambda, u, n)
end

(f::NormL1Box)(x) = f.lambda * norm(x, 1)

function prox!(y, f::NormL1Box, x::AbstractArray{<:Real}, gamma)
    @assert length(y) == length(x)
    n1y = eltype(x)(0)
    gl = gamma * f.lambda
    @inbounds @simd for i in eachindex(x)
        y[i] = max(0, min(x[i] - gl, f.u[i]))
        n1y += y[i]
    end
    return f.lambda * n1y
end

function prox_naive(f::NormL1Box, x, gamma)
    y = max.(0, min.(x .- gamma .* f.lambda, f.u))
    return y, f.lambda * norm(y, 1)
end
