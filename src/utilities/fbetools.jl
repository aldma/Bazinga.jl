function f_model(
    f_x::R,
    grad_f_x::AbstractArray{R},
    res::AbstractArray{R},
    gamma::R,
) where {R<:Real}
    return f_x - dot(grad_f_x, res) + (0.5 / gamma) * norm(res)^2
end
