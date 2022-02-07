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
problem_name = "obstacle"
N = 64 # discretization intervals
tol = 1e-4 # tolerance
sub_maxit = 10_000 # subsolver max iterations

filename = problem_name
filepath = joinpath(@__DIR__, "results", filename)

data = DataFrame()

T = Float64
f = SmoothCostOBST( N )
if problem_name == "obstacle"
    g = NonsmoothCostOBST( N )
    c = ConstraintOBST( N )
    D = SetOBST()
    x0 = ones(T,3*N)
    y0 = zeros(T,N)
else
    g = NonsmoothCostOBSTxy( N )
    c = ConstraintOBSTxy( N )
    D = SetOBSTxy( N )
    x0 = ones(T,2*N)
    y0 = zeros(T,2*N)
end

out = Bazinga.alps(f, g, c, D, x0, y0, tol=tol, subsolver_maxit = sub_maxit)

xsol = out[1]
objx = f(xsol)
cx = similar(y0)
eval!(cx, c, xsol)
px = similar(y0)
proj!(px, D, cx)
distcx = norm(cx - px, 2)
push!(data, (N=N, objective = objx, distcx = distcx, iter=out[3], sub_iter=out[4], time=out[5]))

CSV.write(filepath * ".csv", data, header = false)
