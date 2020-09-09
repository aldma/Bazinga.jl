# SCALING
#
# PART OF BAZINGA.jl

using LinearAlgebra

"""
    objgradscaling()
returns a scaling factor `σ` for an objective function `f` at a given point `x`
such that the gradient of `F := σ f` at `x` does not exceed `scaling_grad_max` or,
if requested, is `scaling_grad_target`, whenever allowed by `scaling_min_value`.
"""
function objgradscaling(
    grad_f_x::Tx,
    grad_target::Maybe{R},
    grad_max::R,
    min_value::R,
) where {R<:Real,Tx<:AbstractVector{R}}
    max_grad_f = norm(grad_f_x, Inf)
    if grad_target === nothing
        s = min(R(1), grad_max / max_grad_f)
    else
        if max_grad_f == 0
            s = R(1)
            if warnings
                @warn "Gradient of smooth objective function is zero at starting point.
                Cannot determine scaling factor based on `scaling_grad_target` option."
            end
        else
            s = grad_target / max_grad_f
        end
    end
    return max(s, min_value)
end
