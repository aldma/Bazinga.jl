# L0 pseudo-norm (times a constant) with indicator [0, ub]

"""
    NormL0Box(λ=1)

Return the ``L_0`` pseudo-norm function
```math
f(x) = λ\\cdot\\mathrm{nnz}(x) + indicator_[0,u](x)
```
for a nonnegative parameter `λ` and vector `u` with nonnegative entries.
"""
struct NormL0Box{T}
    lambda::T
    u::Vector{T}
    n::Int
end

function NormL0Box(lambda::T = 1; u::Vector{T}) where {T}
    if lambda < 0
        error("parameter λ must be nonnegative")
    end
    if any(u .< 0)
        error("vector u must have nonnegative entries")
    end
    n = length(u)
    return NormL0Box(lambda, u, n)
end

function (f::NormL0Box)(x)
    return f.lambda * real(eltype(x))(count(!iszero, x))
end

function prox!(y, f::NormL0Box, x, gamma)
    countnzy = real(eltype(x))(0)
    gl2 = gamma * f.lambda
    for i in eachindex(x)
        if f.u[i] == 0
            y[i] = 0
        else
            if x[i] > sqrt(gl2)
                if x[i] > f.u[i] # > 0
                    if x[i]^2 > gl2 + (f.u[i] - x[i])^2
                        y[i] = x[i]
                        countnzy += 1
                    else
                        y[i] = 0
                    end
                else
                    y[i] = x[i] # > 0
                    countnzy += 1
                end
            else
                y[i] = 0
            end
        end
    end
    return f.lambda * countnzy
end
