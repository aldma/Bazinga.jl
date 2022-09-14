"""
Given positive penalty parameters `μ` and Lagrange multipliers `y`,

   AugLagFunSlack(f, c, D, mu, y, x)

returns a function ´al_f´ which represents the smooth part of

  L_μ(x,s,y) := f(x) + g(x) + δ_D(s) + 1/(2 μ) || c(x) + μ y - s ||^2 - (μ/2) ||y||^2

namely

  al_f(x,s) := f(x) + 1/(2 μ) || c(x) + μ y - s ||^2 - (μ/2) ||y||^2

"""
mutable struct AugLagFunSlack{Tf} <: ProximableFunction where {Tf}
    f::Tf                       # smooth cost function
    c::SmoothFunction           # constraints function
    nx::Integer                 # number of (primal) variables
    ny::Integer                 # number of constraints
    mu::AbstractVector          # penalty parameters
    y::AbstractVector           # dual variable estimate
    # additional storage
    muy::AbstractVector         # mu .* y
    musqy::Real                 # 0.5 * sum( mu .* y.^2 )
    cx::AbstractVector          # c(x)
    yupd::AbstractVector        # yupd := ( cx + mu * y - s ) / mu
    dfx::AbstractArray          # gradient_f at x
    jtv::AbstractArray          # jtv := dcx' * yupd
end
"""
    Constructor
    F = AugLagFunSlack(f, c, mu, y, x)
"""
function AugLagFunSlack(f, c, mu, y, x)
    if any(mu .<= 0)
        error("parameters `mu` must be positive")
    end
    muy = mu .* y
    musqy = 0.5 * sum(muy .* y)
    AugLagFunSlack(
        f,
        c,
        length(x),
        length(y),
        mu,
        y,
        muy,
        musqy,
        similar(y),
        similar(y),
        similar(x),
        similar(x),
    )
end

"""
    Fxs = F( xs )
"""
function (F::AugLagFunSlack)(xs)
    if length(xs) != F.nx + F.ny
        error("wrong length of passed argument xs")
    end
    x = xs[1:F.nx]                  # x
    s = xs[F.nx+1:F.nx+F.ny]        # s
    fx = F.f(x)                     # f(x)
    eval!(F.cx, F.c, x)             # c(x)
    F.yupd .= F.cx .+ F.muy .- s    # cx + mu * y - s (temporary)
    Fxs = 0.5 * sum((F.yupd) .^ 2 ./ F.mu)  # 1/(2 mu) dist_D^2( cx + mu * y )
    Fxs += fx                       # + fx
    Fxs -= F.musqy                  # - (mu/2) ||y||^2
    F.yupd .= F.y .+ (F.cx .- s) ./ F.mu # yupd := (cx + mu * y - s) / mu
    return Fxs
end

"""
    Fxs = gradient!( dFxs, F, xs )
"""
function gradient!(dFxs, F::AugLagFunSlack, xs)
    if length(xs) != F.nx + F.ny
        error("wrong length of passed argument xs")
    end
    if length(dFxs) != F.nx + F.ny
        error("wrong length of passed argument dFxs")
    end
    x = xs[1:F.nx]                  # x
    s = xs[F.nx+1:F.nx+F.ny]        # s
    fx = gradient!(F.dfx, F.f, x)   # fx, dfx
    eval!(F.cx, F.c, x)             # cx
    Fxs = 0.5 * sum((F.cx .+ F.muy .- s) .^ 2 ./ F.mu)  # 1/(2 mu) dist_D^2( cx + mu * y )
    Fxs += fx                       # + fx
    Fxs -= F.musqy                  # - (mu/2) ||y||^2
    F.yupd .= F.y .+ (F.cx .- s) ./ F.mu # yupd := (cx + mu * y - s) / mu
    jtprod!(F.jtv, F.c, x, F.yupd)  # dcx' * yupd
    dFxs[1:F.nx] .= F.dfx .+ F.jtv  # dfx + jtv
    dFxs[F.nx+1:F.nx+F.ny] .= -F.yupd # - yupd
    return Fxs
end

"""
    AugLagUpdate!( F, mu, y )
"""
function AugLagUpdate!(F::AugLagFunSlack, mu, y)
    if any(mu .<= 0)
        error("parameters `mu` must be positive")
    end
    if length(y) != F.ny
        error("wrong length of passed argument y")
    end
    F.mu .= mu
    F.y .= y
    F.muy .= F.mu .* F.y
    F.musqy = 0.5 * sum(F.muy .* F.y)
    return nothing
end

############################################################################################

mutable struct NonsmoothCostFunSlack{Tg} <: ProximableFunction where {Tg}
    g::Tg
    D::ClosedSet
    nx::Integer
    ny::Integer
    gamma::Real                     # proximal stepsize
    gz::Real                        # g at z, z being the proximal update
end
"""
    Constructor
    G = NonsmoothCostFunSlack(g, D, nx, ny)
"""
function NonsmoothCostFunSlack(g, D, nx, ny)
    NonsmoothCostFunSlack(g, D, nx, ny, 0.0, 0.0)
end
"""
    Gz = prox!( z, G, x, gamma )
"""
function prox!(z, G::NonsmoothCostFunSlack, xs, gamma)
    if length(xs) != G.nx + G.ny
        error("wrong length of passed argument xs")
    end
    if length(z) != G.nx + G.ny
        error("wrong length of passed argument z")
    end
    G.gamma = gamma
    x = xs[1:G.nx]
    s = xs[G.nx+1:G.nx+G.ny]
    zx = similar(x)
    gz = prox!(zx, G.g, x, gamma)
    z[1:G.nx] .= zx
    G.gz = gz
    zs = similar(s)
    proj!(zs, G.D, s)
    z[G.nx+1:G.nx+G.ny] .= zs
    return gz
end
