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

#using ProximalOperators
using LinearAlgebra
using Bazinga
using ProximalAlgorithms

###################################################################################
# problem definition
###################################################################################
struct SmoothCostObstacleL2 <: Bazinga.ProximableFunction
    N::Int
end
function (f::SmoothCostObstacleL2)(x)
    return 0.5 * norm(x[1:2*f.N], 2)^2 - sum(x[f.N+1:2*f.N])
end
function Bazinga.gradient!(dfx, f::SmoothCostObstacleL2, x)
    dfx .= 0.0
    dfx[1:f.N] .= x[1:f.N]
    dfx[f.N+1:2*f.N] .= x[f.N+1:2*f.N] .- 1.0
    return 0.5 * norm(x[1:2*f.N], 2)^2 - sum(x[f.N+1:2*f.N])
end

struct SmoothCostObstacleL1 <: Bazinga.ProximableFunction
    N::Int
end
function (f::SmoothCostObstacleL1)(x)
    return 0.5 * sum(x[f.N+1:2*f.N] .^ 2) - sum(x[f.N+1:2*f.N])
end
function Bazinga.gradient!(dfx, f::SmoothCostObstacleL1, x)
    dfx .= 0.0
    dfx[f.N+1:2*f.N] .= x[f.N+1:2*f.N] .- 1.0
    return 0.5 * sum(x[f.N+1:2*f.N] .^ 2) - sum(x[f.N+1:2*f.N])
end

struct NonsmoothCostObstacleL2 <: Bazinga.ProximableFunction
    N::Int
end
function Bazinga.prox!(y, g::NonsmoothCostObstacleL2, x, gamma)
    y .= max.(0, x)
    for i = 1:g.N
        if y[g.N+i] > y[2*g.N+i]
            y[2*g.N+i] = 0
        else
            y[g.N+i] = 0
        end
    end
    return zero(eltype(x))
end

struct NonsmoothCostObstacleL1 <: Bazinga.ProximableFunction
    N::Int
end
function Bazinga.prox!(y, g::NonsmoothCostObstacleL1, x, gamma)
    y[1:g.N] .= max.(0, x[1:g.N] .- gamma)
    y[g.N+1:3*g.N] .= max.(0, x[g.N+1:3*g.N])
    for i = 1:g.N
        if y[g.N+i] > y[2*g.N+i]
            y[2*g.N+i] = 0
        elseif y[g.N+i] < y[2*g.N+i]
            y[g.N+i] = 0
        else # set-valued case
            y[g.N+i] = 0
            #y[2*g.N+i] = 0
        end
    end
    return sum(y[1:g.N])
end

struct NonsmoothCostObstacleRedL1 <: Bazinga.ProximableFunction
    N::Int
end
function Bazinga.prox!(y, g::NonsmoothCostObstacleRedL1, x, gamma)
    y[1:g.N] .= max.(0, x[1:g.N] .- gamma)
    y[g.N+1:2*g.N] .= x[g.N+1:2*g.N]
    #y[g.N+1:2*g.N] .= max.(0, x[g.N+1:2*g.N])
    return sum(y[1:g.N])
end

struct ConstraintObstacle <: SmoothFunction
    N::Int
    A::AbstractMatrix
end
function ConstraintObstacle(N)
    T = Float64
    dv = 2 * ones(T, N)
    ev = -1 * ones(T, N - 1)
    A = SymTridiagonal(dv, ev)
    return ConstraintObstacle(N, A)
end
function Bazinga.eval!(cx, c::ConstraintObstacle, x)
    cx .= x[1:c.N] .+ c.A * x[c.N+1:2*c.N] .- x[2*c.N+1:3*c.N]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintObstacle, x, v)
    jtv[1:c.N] .= v
    jtv[c.N+1:2*c.N] .= c.A * v # A = A'
    jtv[2*c.N+1:3*c.N] .= -v
    return nothing
end

struct SetObstacle <: ClosedSet end
function Bazinga.proj!(z, D::SetObstacle, x)
    z .= 0.0
    return nothing
end

struct NonsmoothCostObstacleL2Red <: Bazinga.ProximableFunction
    N::Int
end
function Bazinga.prox!(y, g::NonsmoothCostObstacleL2Red, x, gamma)
    y .= max.(0, x)
    return zero(eltype(x))
end

struct ConstraintObstacleRed <: SmoothFunction
    N::Int
    A::AbstractMatrix
end
function ConstraintObstacleRed(N)
    c = ConstraintObstacle(N)
    return ConstraintObstacleRed(N, c.A)
end
function Bazinga.eval!(cx, c::ConstraintObstacleRed, x)
    cx[1:c.N] .= x[1:c.N] + c.A * x[c.N+1:2*c.N]
    cx[c.N+1:2*c.N] .= x[c.N+1:2*c.N]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintObstacleRed, x, v)
    jtv[1:c.N] .= v[1:c.N]
    jtv[c.N+1:2*c.N] .= c.A * v[1:c.N] + v[c.N+1:2*c.N] # A = A'
    return nothing
end

struct SetObstacleRed <: ClosedSet
    N::Int
end
function Bazinga.proj!(z, D::SetObstacleRed, x)
    # complementarity constraint
    for i = 1:D.N
        Bazinga.project_onto_CC_set!(@view(z[[i, i + D.N]]), x[[i, i + D.N]])
    end
    #=z .= max.(0, x)
    for i = 1:D.N
        if z[i] > z[i+D.N]
            z[i+D.N] = 0
        elseif z[i] < z[i+D.N]
            z[i] = 0
        else # set-valued case
            z[i] = 0
            #z[i+D.N] = 0
        end
    end=#
    return nothing
end

################################################################################
# grid of discretizations and tolerances
################################################################################
using DataFrames
using CSV

problem_name = "obstacle_l1" # obstacle_l1, obstacle_l1red
Nvec = [16; 32; 48; 64] # discretization intervals
TOLvec = 10 .^ collect(range(-3, -5, length = 9)) # tolerance

filename = problem_name
filepath = joinpath(@__DIR__, "results", filename)

data = DataFrame()

T = Float64

subsolver_directions = ProximalAlgorithms.LBFGS(5)
subsolver_maxit = 1_000_000
subsolver_minimum_gamma = eps(T)
subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = subsolver_directions,
    maxit = subsolver_maxit,
    freq = subsolver_maxit,
    minimum_gamma = subsolver_minimum_gamma;
    kwargs...,
)
solver(f, g, c, D, x0, y0; kwargs...) = Bazinga.alps(
    f,
    g,
    c,
    D,
    x0,
    y0,
    verbose = true,
    subsolver = subsolver,
    subsolver_maxit = subsolver_maxit;
    kwargs...,
)

for N in Nvec
    @info "N = $(N)"
    if problem_name == "obstacle_l2"
        f = SmoothCostObstacleL2(N)
        g = NonsmoothCostObstacleL2(N)
        c = ConstraintObstacle(N)
        D = SetObstacle()
        nx = 3 * N
        ny = N
    elseif problem_name == "obstacle_l2red"
        f = SmoothCostObstacleL2(N)
        g = NonsmoothCostObstacleL2Red(N)
        c = ConstraintObstacleRed(N)
        D = SetObstacleRed(N)
        nx = 2 * N
        ny = 2 * N
    elseif problem_name == "obstacle_l1"
        f = SmoothCostObstacleL1(N)
        g = NonsmoothCostObstacleL1(N)
        c = ConstraintObstacle(N)
        D = SetObstacle()
        nx = 3 * N
        ny = N
    elseif problem_name == "obstacle_l1red"
        f = SmoothCostObstacleL1(N)
        g = NonsmoothCostObstacleRedL1(N)
        c = ConstraintObstacleRed(N)
        D = SetObstacleRed(N)
        nx = 2 * N
        ny = 2 * N
    else
        @error "Unknown problem"
    end

    for TOL in TOLvec
        @info "TOL = $(TOL)"

        out = solver(f, g, c, D, 2 .* ones(T, nx), zeros(T, ny), tol = TOL)

        xsol = out[1]
        objx = f(xsol)
        cx = zeros(T, ny)
        eval!(cx, c, xsol)
        px = zeros(T, ny)
        proj!(px, D, cx)
        distcx = norm(cx - px, 2)

        push!(
            data,
            (
                N = N,
                tol = TOL,
                objective = objx,
                cviolation = distcx,
                iters = out[3],
                sub_iters = out[4],
                runtime = out[5],
            ),
        )
    end
end

CSV.write(filepath * ".csv", data, header = true)
