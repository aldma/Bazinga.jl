"""
Given positive penalty parameters `mu` and Lagrange multipliers `y`,

   AugLagFun(f, c, D, mu, y, x)

returns a function ´al_f´ which represents

  L(x) = f(x) + 1/(2 mu) dist_D^2( c(x) + mu y ) - (mu/2) ||y||^2

"""
mutable struct AugLagFun{Tf} <: ProximableFunction where {Tf}
    f::Tf                     # original problem
    c::SmoothFunction
    D::ClosedSet
    mu::AbstractVector              # penalty parameters
    y::AbstractVector               # dual variable estimate
    # additional storage
    muy::AbstractVector            # mu .* y
    musqy::Real                    # 0.5 * sum( mu .* y.^2 )
    cx::AbstractVector             # c(x)
    s::AbstractVector              # z ∈ proj_D( cx + mu * y )
    yupd::AbstractVector           # yupd := ( cx + mu * y - s ) / mu
    fx::Real                       # f at x
    dfx::AbstractArray             # gradient_f at x
    jtv::AbstractArray             # jtv := dcx' * yupd
end

"""
    Constructor
    AugLagFun(f, c, D, mu, y, x)
"""
function AugLagFun(f, c, D, mu, y, x)
    if any(mu .<= 0)
        error("parameters `mu` must be positive")
    else
        muy = mu .* y
        musqy = 0.5 * sum(muy .* y)
        AugLagFun(
            f,
            c,
            D,
            mu,
            y,
            muy,
            musqy,
            similar(y),
            similar(y),
            similar(y),
            zero(eltype(x)),
            similar(x),
            similar(x),
        )
    end
end
"""
    al( x )
"""
function (al::AugLagFun)(x)
    eval!(al.cx, al.c, x)                  # cx
    al.yupd .= al.cx .+ al.muy           # cx + mu * y           (temporary)
    proj!(al.s, al.D, al.yupd)            # s ∈ proj_D( cx + mu * y )
    al.yupd .-= al.s                      # cx + mu * y - s       (temporary)
    lx = 0.5 * sum((al.yupd) .^ 2 ./ al.mu)  # 1/(2 mu) dist_D^2( cx + mu * y )
    al.yupd ./= al.mu                      # yupd := (cx + mu * y - s) / mu
    al.fx = al.f(x)                        # f(x)
    lx += al.fx                            # + fx
    lx -= al.musqy                         #      - (mu/2) ||y||^2
    return lx
end
"""
    lx = gradient!( dlx, al, x )
"""
function gradient!(dlx, al::AugLagFun, x)
    eval!(al.cx, al.c, x)                  # cx
    al.yupd .= al.cx .+ al.muy           # cx + mu .* y          (temporary)
    proj!(al.s, al.D, al.yupd)            # s ∈ proj_D( cx + mu .* y )
    al.yupd .-= al.s                      # cx + mu .* y - s      (temporary)
    lx = 0.5 * sum((al.yupd) .^ 2 ./ al.mu)  # 1/(2 mu) dist_D^2( cx + mu * y )
    al.yupd ./= al.mu                      # yupd := (cx + mu .* y - s) ./ mu
    al.fx = gradient!(al.dfx, al.f, x)    # fx, dfx
    lx += al.fx  # + fx
    lx -= al.musqy                         #      - (mu/2) ||y||^2
    jtprod!(al.jtv, al.c, x, al.yupd)     # jtv ← dcx' * yupd
    dlx .= al.dfx .+ al.jtv    # dfx + dcx' * yu
    return lx
end

"""
    AugLagUpdate!( al, mu, y )
"""
function AugLagUpdate!(al::AugLagFun, mu, y)
    if any(mu .<= 0)
        error("parameters `mu` must be positive")
    else
        al.mu .= mu
        al.y .= y
        al.muy .= al.mu .* al.y
        al.musqy = 0.5 * sum(al.muy .* al.y)
    end
    return nothing
end
