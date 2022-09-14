# default dual safeguard
function default_dual_safeguard!(y)
    y .= max.(-1e20, min.(y, 1e20))
    return nothing
end

function default_dual_safeguard!(y, cx)
    default_dual_safeguard!(y)
    return nothing
end

# default penalty parameter
function default_penalty_parameter!(mu, cx, proj_cx, objx)
    mu .= max.(1.0, 0.5 * (cx .- proj_cx) .^ 2) ./ max(1.0, objx)
    mu .*= 0.1
    mu .= max.(1e-8, min.(mu, 1e8))
    return nothing
end
