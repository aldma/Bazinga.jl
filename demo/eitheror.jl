"""
	eitheror : example from Bazinga.jl

	Mathematical program with either-or constraints, from [KMS18].

	Original formulations:

	minimize      (x1 - 8)^2 + (x2 + 3)^2
	subject to    x1 - 2 x2 + 4 ≤ 0    or    x1 - 2 ≤ 0
	              x1^2 - 4 x2 ≤ 0      or    (x1 - 3)^2 + (x2 - 1)^2 - 10 ≤ 0

	Reformulations as constrained structured problems in the form

	minimize     f(x) + g(x)
	subject to   c(x) in D

	(i) implicit formulation
        name = 'eitheror'
        x    = [x1, x2]
	    f(x) = (x1 - 8)^2 + (x2 + 3)^2
	    g(x) = 0
	    c(x) = [ - x1 + 2 x2 - 4                ]
               [ - x1 + 2                       ]
       	       [ - x1^2 + 4 x2                  ]
       	       [ - (x1 - 3)^2 - (x2 - 1)^2 + 10 ]
        D    = { c | (c1,c2) in Dor, (c3,c4) in Dor }
        nx   = 2
        ny   = 4 
	with Dor given by
	    Deor  = { (a,b) | either a >= 0  or b >= 0 } <<< inclusive or
        Dxor  = { (a,b) |        a >= 0 xor b >= 0 } <<< exclusive or

    (ii) slack variables for all constraints
        name = 'eitheror_fullslack'
        x    = [x1, x2, s1, s2, s3, s4]
        f(x) = (x1 - 8)^2 + (x2 + 3)^2
        g(x) = Indicator_Dor(s1, s2) + Indicator_Dor(s3, s4)
        c(x) = [ - x1 + 2 x2 - 4                - s1 ]
               [ - x1 + 2                       - s2 ]
               [ - x1^2 + 4 x2                  - s3 ]
               [ - (x1 - 3)^2 - (x2 - 1)^2 + 10 - s4 ]
        D    = (0, 0, 0, 0)
        nx   = 6
        ny   = 4

    References:
    [KMS18]     Kanzow, Mehlitz, Steck, "Relaxation schemes for mathematical
                programs with switching constraints" (2019).
"""

using LinearAlgebra
using Bazinga
using ProximalAlgorithms

###################################################################################
# problem definition
###################################################################################
struct SmoothCostOR <: Bazinga.ProximableFunction
    c::AbstractVector
end
function (f::SmoothCostOR)(x)
    return norm(x[1:2] - f.c, 2)^2
end
function Bazinga.gradient!(dfx, f::SmoothCostOR, x)
    dfx .= 0
    dfx[1:2] .= 2 * (x[1:2] - f.c)
    return norm(x[1:2] - f.c, 2)^2
end

struct NonsmoothCostOR <: Bazinga.ProximableFunction end
function Bazinga.prox!(z, g::NonsmoothCostOR, x, gamma)
    z .= x
    return zero(eltype(x))
end

struct NonsmoothCostEORfullslack <: Bazinga.ProximableFunction end
function Bazinga.prox!(z, g::NonsmoothCostEORfullslack, x, gamma)
    z .= x
    Bazinga.project_onto_EITHEROR_set!(@view(z[3:4]), x[3:4])
    Bazinga.project_onto_EITHEROR_set!(@view(z[5:6]), x[5:6])
    return zero(eltype(x))
end

struct NonsmoothCostXORfullslack <: Bazinga.ProximableFunction end
function Bazinga.prox!(z, g::NonsmoothCostXORfullslack, x, gamma)
    z .= x
    Bazinga.project_onto_XOR_set!(@view(z[3:4]), x[3:4])
    Bazinga.project_onto_XOR_set!(@view(z[5:6]), x[5:6])
    return zero(eltype(x))
end

struct ConstraintOR <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintOR, x)
    cx[1] = 2 * x[2] - x[1] - 4
    cx[2] = 2 - x[1]
    cx[3] = 4 * x[2] - x[1]^2
    cx[4] = 10 - (x[1] - 3)^2 - (x[2] - 1)^2
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintOR, x, v)
    jtv[1] = -v[1] - v[2] - 2 * x[1] * v[3] + 2 * (3 - x[1]) * v[4]
    jtv[2] = 2 * v[1] + 4 * v[3] + 2 * (1 - x[2]) * v[4]
    return nothing
end

struct ConstraintORfullslack <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintORfullslack, x)
    cx[1] = 2 * x[2] - x[1] - 4 - x[3]
    cx[2] = 2 - x[1] - x[4]
    cx[3] = 4 * x[2] - x[1]^2 - x[5]
    cx[4] = 10 - (x[1] - 3)^2 - (x[2] - 1)^2 - x[6]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintORfullslack, x, v)
    jtv[1] = -v[1] - v[2] - 2 * x[1] * v[3] + 2 * (3 - x[1]) * v[4]
    jtv[2] = 2 * v[1] + 4 * v[3] + 2 * (1 - x[2]) * v[4]
    jtv[3:6] .= -v[1:4]
    return nothing
end

struct SetEOR <: ClosedSet end
function Bazinga.proj!(z, D::SetEOR, x)
    Bazinga.project_onto_EITHEROR_set!(@view(z[1:2]), x[1:2])
    Bazinga.project_onto_EITHEROR_set!(@view(z[3:4]), x[3:4])
    return nothing
end
struct SetXOR <: ClosedSet end
function Bazinga.proj!(z, D::SetXOR, x)
    Bazinga.project_onto_XOR_set!(@view(z[1:2]), x[1:2])
    Bazinga.project_onto_XOR_set!(@view(z[3:4]), x[3:4])
    return nothing
end

# tets setup
problem_name = "eitheror" # eitheror, xor, *_fullslack
solver_name = "alps" # alps, als
subsolver_name = "lbfgs" # lbfgs, noaccel, broyden, anderson

T = Float64
f = SmoothCostOR(T.([8; -3]))
if problem_name == "eitheror"
    g = NonsmoothCostOR()
    c = ConstraintOR()
    D = SetEOR()
    nx = 2
    ny = 4
elseif problem_name == "eitheror_fullslack"
    g = NonsmoothCostEORfullslack()
    c = ConstraintORfullslack()
    D = Bazinga.ZeroSet()
    nx = 6
    ny = 4
elseif problem_name == "xor"
    g = NonsmoothCostOR()
    c = ConstraintOR()
    D = SetXOR()
    nx = 2
    ny = 4
elseif problem_name == "xor_fullslack"
    g = NonsmoothCostXORfullslack()
    c = ConstraintORfullslack()
    D = Bazinga.ZeroSet()
    nx = 6
    ny = 4
else
    @error "Unknown problem name"
end

subsolver_directions = if subsolver_name == "noaccel"
    ProximalAlgorithms.NoAcceleration()
elseif subsolver_name == "broyden"
    ProximalAlgorithms.Broyden()
elseif subsolver_name == "anderson"
    ProximalAlgorithms.AndersonAcceleration(5)
elseif subsolver_name == "lbfgs"
    ProximalAlgorithms.LBFGS(5)
else
    @error "Unknown subsolver name"
end
subsolver_maxit = 1_000
subsolver_minimum_gamma = eps(T)
subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = subsolver_directions,
    maxit = subsolver_maxit,
    freq = subsolver_maxit,
    minimum_gamma = subsolver_minimum_gamma;
    kwargs...,
)
if solver_name == "alps"
    solver(f, g, c, D, x0, y0) = Bazinga.alps(
        f,
        g,
        c,
        D,
        x0,
        y0,
        verbose = false,
        tol = 1e-8,
        subsolver = subsolver,
        subsolver_maxit = subsolver_maxit,
    )
elseif solver_name == "als"
    solver(f, g, c, D, x0, y0) = Bazinga.als(
        f,
        g,
        c,
        D,
        x0,
        y0,
        verbose = false,
        tol = 1e-8,
        subsolver = subsolver,
        subsolver_maxit = subsolver_maxit,
    )
else
    @error "Unknown solver name"
end

@info "Problem " * problem_name
@info "Solver  " * solver_name * "(" * subsolver_name * ")"
_ = solver(f, g, c, D, zeros(T, nx), zeros(T, ny)) # warm up

################################################################################
# grid of starting points
################################################################################
using Plots
using DataFrames
using Printf
using CSV
using Statistics

filename = problem_name * "_" * solver_name * "_" * subsolver_name
filepath = joinpath(@__DIR__, "results", filename)

xmin = -4.0
xmax = 8.0

data = DataFrame()

xgrid = [(i, j) for i = xmin:0.25:xmax, j = xmin:0.25:xmax];
xgrid = xgrid[:];
ntests = length(xgrid)

for i = 1:ntests
    if problem_name in ["eitheror", "xor"]
        x0 = [xgrid[i][1]; xgrid[i][2]]
    elseif problem_name in ["eitheror_fullslack", "xor_fullslack"]
        x0 = [
            xgrid[i][1]
            xgrid[i][2]
            2 * xgrid[i][2] - xgrid[i][1] - 4
            2 - xgrid[i][1]
            4 * xgrid[i][2] - xgrid[i][1]^2
            10 - (xgrid[i][1] - 3)^2 - (xgrid[i][2] - 1)^2
        ]
    end
    y0 = zeros(T, ny)

    out = solver(f, g, c, D, x0, y0)

    xsol = out[1]
    push!(
        data,
        (
            id = i,
            xinit_1 = x0[1],
            xinit_2 = x0[2],
            xfinal_1 = xsol[1],
            xfinal_2 = xsol[2],
            iters = out[3],
            sub_iters = out[4],
            runtime = out[5],
        ),
    )

end

CSV.write(filepath * ".csv", data, header = false)

@info " sample points: $(ntests)"
@info "    iterations: max $(maximum(data.iters)), median  $(median(data.iters))"
@info "sub iterations: max $(maximum(data.sub_iters)), median  $(median(data.sub_iters))"
@info "       runtime: max $(maximum(data.runtime)), median  $(median(data.runtime))"

################################################################################
# plot results
################################################################################

# minimizers
x_22 = T.([2; -2])
x_44 = T.([4; 4])

# counters
global c_22 = 0
global c_44 = 0
global c_un = 0

# plot feasible set
pts = [(0.0, 0.0)]
rad = sqrt(10.0)
for i = 1:50
    xx = 2.0 * i / 50
    yy = 1.0 - sqrt(10.0 - (xx - 3)^2)
    append!(pts, [(xx, yy)])
end
append!(pts, [(2, 3)])
append!(pts, [(4, 4)])
for i = 1:50
    xx = 4 + (4 * i / 50)
    yy = (xx^2) / 4
    append!(pts, [(xx, yy)])
end
append!(pts, [(-4, 8)])
for i = 0:50
    xx = -4 * (1 - i / 50)
    yy = (xx^2) / 4
    append!(pts, [(xx, yy)])
end
feasset = Shape(pts)
hplt = plot(feasset, color = plot_color(:grey, 0.4), linewidth = 0, legend = false)
xlims!(xmin, xmax)
ylims!(xmin, xmax)

tolx = 1e-6 # approx tolerance

for i = 1:ntests
    xi = [data[i, 2]; data[i, 3]]
    xf = [data[i, 4]; data[i, 5]]

    if norm(xf - x_22) <= tolx
        global c_22 += 1
        scatter!(
            hplt,
            [xi[1]],
            [xi[2]],
            color = :blue,
            marker = :circle,
            markerstrokewidth = 0,
            legend = false,
        )
    elseif norm(xf - x_44) <= tolx
        global c_44 += 1
        scatter!(
            hplt,
            [xi[1]],
            [xi[2]],
            color = :red,
            marker = :diamond,
            markerstrokewidth = 0,
            legend = false,
        )
    else
        global c_un += 1
        @printf "(%6.4f,%6.4f) from (%6.4f,%6.4f)\n" xf[1] xf[2] xi[1] xi[2]
    end
end

@info " global (2,-2): $(c_22)/$(ntests) ($(100*c_22/ntests)%)"
@info "  local (4, 4): $(c_44)/$(ntests) ($(100*c_44/ntests)%)"
if c_un > 0
    @info "        others: $(c_un)/$(ntests) ($(100*c_un/ntests)%)"
end

savefig(filepath * ".pdf")

# store some statistics
stats = DataFrame()
push!(
        stats,
        (
            npoints = ntests,
            iters_max = maximum(data.iters),
            iters_median = median(data.iters),
            subiters_max = maximum(data.sub_iters),
            subiters_median = median(data.sub_iters),
            runtime_max = maximum(data.runtime),
            runtime_median = median(data.runtime),
            global_nabs = c_22,
            global_nrel = 100*c_22/ntests,
            local_nabs = c_44,
            local_nrel = 100*c_44/ntests,
            unkwn_nabs = c_un,
            unkwn_nrel = 100*c_un/ntests,
        ),
    )
CSV.write(filepath * "_stats" * ".csv", stats, header = true)