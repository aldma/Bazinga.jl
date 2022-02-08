"""
	obstacle : example from Bazinga.jl

    Example of optimal control of a discretized obstacle problem, yielding a
    mathematical program with complementarity constraints (MPCC),
    see Section 7.4 [HMW21].

    References:
    [HMW21]     Harder, Mehlitz, Wachsmuth, "Reformulation of the M-Stationarity
                Conditions as a System of Discontinuous Equations and Its Solution
                by a Semismooth Newton Method",
                SIAM Journal on Optimization, 2021,
                DOI: 10.1137/20M1321413.
"""

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
    for i = 1:g.N
        if x[2*g.N+i] > x[g.N+i]
            y[g.N+i] = 0
            y[2*g.N+i] = x[2*g.N+i]
        else
            y[g.N+i] = x[g.N+i]
            y[2*g.N+i] = 0
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

## reduced formulation, without slack variables
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

################################################################################
# grid of discretizations and tolerances
################################################################################
using DataFrames
using CSV

problem_name = "obstacle_xy"
Nvec = [32; 64; 128] # discretization intervals
TOLvec = [1e-3; 1e-4; 1e-5] # tolerance
sub_maxit = 10_000 # subsolver max iterations

filename = problem_name
filepath = joinpath(@__DIR__, "results", filename)

data = DataFrame()

T = Float64

for N in Nvec

    f = SmoothCostOBST( N )
    if problem_name == "obstacle_xy"
        g = NonsmoothCostOBSTxy( N )
        c = ConstraintOBSTxy( N )
        D = SetOBSTxy( N )
        x0 = ones(T,2*N)
        y0 = zeros(T,2*N)
    else
        g = NonsmoothCostOBST( N )
        c = ConstraintOBST( N )
        D = SetOBST()
        x0 = ones(T,3*N)
        y0 = zeros(T,N)
    end

    for TOL in TOLvec

        out = Bazinga.alps(f, g, c, D, x0, y0, tol=TOL, subsolver_maxit = sub_maxit)

        xsol = out[1]
        objx = f(xsol)
        cx = similar(y0)
        eval!(cx, c, xsol)
        px = similar(y0)
        proj!(px, D, cx)
        distcx = norm(cx - px, 2)

        push!(data, (N=N,
                     tol=TOL,
                     objective = objx,
                     cviolation = distcx,
                     iters=out[3],
                     sub_iters=out[4],
                     runtime=out[5],
                     ))
    end
end

CSV.write(filepath * ".csv", data, header = false)
