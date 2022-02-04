using ProximalOperators
using LinearAlgebra
using Bazinga

###################################################################################
# problem definition
###################################################################################
struct SmoothCostOBST <: ProximalOperators.ProximableFunction
    N::Int
end
function (f::SmoothCostOBST)(x)
    return 0.5 * norm(x[1:2*f.N], 2)^2 - sum(x[f.N+1:2*f.N])
end
function ProximalOperators.gradient!(dfx, f::SmoothCostOBST, x)
    dfx .= 0.0
    dfx[1:f.N] .= x[1:f.N]
    dfx[f.N+1:2*f.N] .= x[f.N+1:2*f.N] .- 1.0
    return 0.5 * norm(x[1:2*f.N], 2)^2 - sum(x[f.N+1:2*f.N])
end

struct NonsmoothCostOBST <: ProximalOperators.ProximableFunction
    N::Int
end
function ProximalOperators.prox!(y, g::NonsmoothCostOBST, x, gamma)
    y .= max.(0, x)
    for i = 1:f.N
        if x[2*f.N+i] > x[f.N+i]
            y[f.N+i] = 0
            y[2*f.N+i] = x[2*f.N+i]
        else
            y[f.N+i] = x[f.N+i]
            y[2*f.N+i] = 0
        end
    end
    return zero(eltype(x))
end

struct ConstraintOBST <: SmoothFunction
    N::Int
    A::AbstractMatrix
end
function ConstraintOBST(N)
    T = Float64
    dv = 2 * ones(T,N)
    ev = -1 * ones(T,N-1)
    A = SymTridiagonal(dv, ev)
    return ConstraintOBST(N, A)
end
function Bazinga.eval!(cx, c::ConstraintOBST, x)
    cx .= x[1:c.N] .+ c.A * x[c.N+1:2*c.N] .- x[2*c.N+1:3*c.N]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintOBST, x, v)
    jtv[1:c.N] .= v
    jtv[c.N+1:2*c.N] .= c.A * v # A = A'
    jtv[2*c.N+1:3*c.N] .= - v
    return nothing
end

struct SetOBST <: ClosedSet end
function Bazinga.proj!(z, D::SetOBST, x)
    z .= 0.0
    return nothing
end

## formulation with (xx, xy) only
struct NonsmoothCostOBSTxy <: ProximalOperators.ProximableFunction
    N::Int
end
function ProximalOperators.prox!(y, g::NonsmoothCostOBSTxy, x, gamma)
    y .= max.(0, x)
    return zero(eltype(x))
end
struct ConstraintOBSTxy <: SmoothFunction
    N::Int
    A::AbstractMatrix
end
function ConstraintOBSTxy(N)
    c = ConstraintOBST(N)
    return ConstraintOBSTxy(N, c.A)
end
function Bazinga.eval!(cx, c::ConstraintOBSTxy, x)
    cx[1:c.N] .= x[1:c.N] .+ c.A * x[c.N+1:2*c.N]
    cx[c.N+1:2*c.N] .= x[c.N+1:2*c.N]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintOBSTxy, x, v)
    jtv[1:c.N] .= v[1:c.N]
    jtv[c.N+1:2*c.N] .= c.A * v[1:c.N] + v[c.N+1:2*c.N] # A = A'
    return nothing
end
struct SetOBSTxy <: ClosedSet
    N::Int
end
function Bazinga.proj!(z, D::SetOBSTxy, x)
    z .= max.(0, x)
    for i = 1:D.N
        if z[i] > z[i+D.N]
            z[i+D.N] = 0
        else
            z[i] = 0
        end
    end
    return nothing
end

## iter
N = 64 # discretization intervals
T = Float64
TOL = T(1e-6)

f = SmoothCostOBST( N )
g = NonsmoothCostOBST( N )
c = ConstraintOBST( N )
D = SetOBST()

x0 = ones(T,3*N)
y0 = zeros(T,N)

out = Bazinga.alps(f, g, c, D, x0, y0, tol = TOL, verbose=true)

xsol = out[1]
ysol = out[2]
tot_it = out[3]
tot_inner_it = out[4]



g_xy = NonsmoothCostOBSTxy( N )
c_xy = ConstraintOBSTxy( N )
D_xy = SetOBSTxy( N )
x0_xy = x0[1:2*N]
y0_xy = zeros(T,2*N)

out_xy = Bazinga.alps(f, g_xy, c_xy, D_xy, x0_xy, y0_xy, tol = TOL, verbose=true)

xsol_xy = out_xy[1]
ysol_xy = out_xy[2]
tot_it_xy = out_xy[3]
tot_inner_it_xy = out_xy[4]
