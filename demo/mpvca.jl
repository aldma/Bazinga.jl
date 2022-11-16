"""
	mpvca : example from Bazinga.jl

    Academic example of mathematical program with vanishing constraints (MPVC),
    see Section 9.5.1 [Hoh09].

	Original formulations:

	minimize      4*x1 + 2*x2
	subject to    x1 >= 0
	              x2 >= 0
	              x1 > 0   =>   x1 + x2 - 5*sqrt(2) >= 0
	              x2 > 0   =>   x1 + x2 - 5         >= 0

	Reformulations as constrained structured problems in the form

	minimize     f(x) + g(x)
	subject to   c(x) in D

	(i) implicit formulation
        name = 'mpvca'
        x    = [x1, x2]
	    f(x) = 4*x1 + 2*x2
	    g(x) = Indicator_Box[0,+infty](x)
	    c(x) = [ x1                  ]
	           [ x1 + x2 - 5*sqrt(2) ]
       	       [ x2                  ]
	           [ x1 + x2 - 5         ]
	    D    = { c | (c1,c2) in Dvc, (c3,c4) in Dvc }
        nx   = 2
        ny   = 4
	with
	    Dvc  = { (a,b) | a >= 0, a*b >= 0 }
             = { (a,b) | a = 0 } ∪ { (a,b) | a >= 0, b >= 0 }

    (ii) slack variables for all constraints
        name = 'mpvca_fullslack'
        x    = [x1, x2, s1, s2, s3, s4]
        f(x) = 4*x1 + 2*x2
        g(x) = Indicator_Box[0,+infty](x1, x2) + Indicator_Dvc(s1, s2) + Indicator_Dvc(s3, s4)
        c(x) = [ x1                  - s1 ]
               [ x1 + x2 - 5*sqrt(2) - s2 ]
               [ x2                  - s3 ]
               [ x1 + x2 - 5         - s4 ]
        D    = (0, 0, 0, 0)
        nx   = 6
        ny   = 4

    (iii) slack variables for nontrivial constraints
        name = 'mpvca_slack'
        x    = [x1, x2, s1, s2]
        f(x) = 4*x1 + 2*x2
        g(x) = Indicator_Dvc(x1, s1) + Indicator_Dvc(x2, s2)
        c(x) = [ x1 + x2 - 5*sqrt(2) - s1 ]
               [ x1 + x2 - 5         - s2 ]
        D    = (0, 0)
        nx   = 4
        ny   = 2

    References:
    [Hoh09]     Hoheisel, "Mathematical Programs with Vanishing Constraints",
                PhD thesis, University of Würzburg, 2009.
"""

using LinearAlgebra
using Bazinga
using ProximalAlgorithms

################################################################################
# problem definition
################################################################################
struct SmoothCostMPVCA <: Bazinga.ProximableFunction
    c::AbstractVector
end
function (f::SmoothCostMPVCA)(x)
    return dot(f.c, x[1:2])
end
function Bazinga.gradient!(dfx, f::SmoothCostMPVCA, x)
    dfx .= 0
    dfx[1:2] .= f.c
    return dot(f.c, x[1:2])
end

struct NonsmoothCostMPVCA <: Bazinga.ProximableFunction end
function Bazinga.prox!(z, g::NonsmoothCostMPVCA, x, gamma)
    z .= max.(0, x)
    return zero(eltype(x))
end

struct NonsmoothCostMPVCAslack <: Bazinga.ProximableFunction end
function Bazinga.prox!(z, g::NonsmoothCostMPVCAslack, x, gamma)
    Bazinga.project_onto_VC_set!(@view(z[[1, 3]]), x[[1, 3]])
    Bazinga.project_onto_VC_set!(@view(z[[2, 4]]), x[[2, 4]])
    return zero(eltype(x))
end

struct NonsmoothCostMPVCAfullslack <: Bazinga.ProximableFunction end
function Bazinga.prox!(z, g::NonsmoothCostMPVCAfullslack, x, gamma)
    z[1:2] .= max.(0, x[1:2])
    Bazinga.project_onto_VC_set!(@view(z[3:4]), x[3:4])
    Bazinga.project_onto_VC_set!(@view(z[5:6]), x[5:6])
    return zero(eltype(x))
end

struct ConstraintMPVCA <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintMPVCA, x)
    cx .= [x[1]; x[1] + x[2] - 5.0 * sqrt(2.0); x[2]; x[1] + x[2] - 5.0]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintMPVCA, x, v)
    jtv .= [v[1] + v[2] + v[4]; v[2] + v[3] + v[4]]
    return nothing
end

struct ConstraintMPVCAslack <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintMPVCAslack, x)
    cx .= [x[1] + x[2] - 5.0 * sqrt(2.0) - x[3]; x[1] + x[2] - 5.0 - x[4]]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintMPVCAslack, x, v)
    jtv .= [v[1] + v[2]; v[1] + v[2]; -v[1]; -v[2]]
    return nothing
end

struct ConstraintMPVCAfullslack <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintMPVCAfullslack, x)
    cx .= [
        x[1] - x[3]
        x[1] + x[2] - 5.0 * sqrt(2.0) - x[4]
        x[2] - x[5]
        x[1] + x[2] - 5.0 - x[6]
    ]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintMPVCAfullslack, x, v)
    jtv .= [v[1] + v[2] + v[4]; v[2] + v[3] + v[4]; -v[1]; -v[2]; -v[3]; -v[4]]
    return nothing
end

struct SetMPVCA <: ClosedSet end
function Bazinga.proj!(z, D::SetMPVCA, cx)
    Bazinga.project_onto_VC_set!(@view(z[1:2]), cx[1:2])
    Bazinga.project_onto_VC_set!(@view(z[3:4]), cx[3:4])
    return nothing
end

# test setup
problem_name = "mpvca" # mpvca, *_slack, *_fullslack
solver_name = "alps" # alps, als
subsolver_name = "lbfgs" # lbfgs

T = Float64
f = SmoothCostMPVCA(T.([4; 2]))
if problem_name == "mpvca"
    g = NonsmoothCostMPVCA()
    c = ConstraintMPVCA()
    D = SetMPVCA()
    nx = 2
    ny = 4
elseif problem_name == "mpvca_slack"
    g = NonsmoothCostMPVCAslack()
    c = ConstraintMPVCAslack()
    D = Bazinga.ZeroSet()
    nx = 4
    ny = 2
elseif problem_name == "mpvca_fullslack"
    g = NonsmoothCostMPVCAfullslack()
    c = ConstraintMPVCAfullslack()
    D = Bazinga.ZeroSet()
    nx = 6
    ny = 4
else
    error("Unknown problem name")
end

# solver setup
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

xmin = -5.0
xmax = 20.0

data = DataFrame()

xgrid = [(i, j) for i = xmin:0.5:xmax, j = xmin:0.5:xmax];
xgrid = xgrid[:];
ntests = length(xgrid)

for i = 1:ntests
    if problem_name == "mpvca"
        x0 = [xgrid[i][1]; xgrid[i][2]]
    elseif problem_name == "mpvca_slack"
        x0 = [
            xgrid[i][1]
            xgrid[i][2]
            xgrid[i][1] + xgrid[i][2] - 5 * sqrt(2)
            xgrid[i][1] + xgrid[i][2] - 5
        ]
    elseif problem_name == "mpvca_fullslack"
        x0 = [
            xgrid[i][1]
            xgrid[i][2]
            xgrid[i][1]
            xgrid[i][1] + xgrid[i][2] - 5 * sqrt(2)
            xgrid[i][2]
            xgrid[i][1] + xgrid[i][2] - 5
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

CSV.write(filepath * ".csv", data, header = true)

@info " sample points: $(ntests)"
@info "    iterations: max $(maximum(data.iters)), median  $(median(data.iters))"
@info "sub iterations: max $(maximum(data.sub_iters)), median  $(median(data.sub_iters))"
@info "       runtime: max $(maximum(data.runtime)), median  $(median(data.runtime))"

################################################################################
# plot results
################################################################################

# minimizers
x_00 = T.([0; 0])
x_05 = T.([0; 5])

# counters
global c_00 = 0
global c_05 = 0
global c_un = 0

# plot feasible set
feasset = Shape([
    (0.0, xmax),
    (0.0, 5 * sqrt(2)),
    (5 * sqrt(2), 0.0),
    (xmax, 0.0),
    (xmax, xmax),
    (0.0, xmax),
])
hplt = plot(feasset, color = plot_color(:grey, 0.4), linewidth = 0, legend = false)
plot!(
    hplt,
    [0, 0],
    [5, 5 * sqrt(2)],
    color = plot_color(:grey, 0.4),
    linewidth = 5,
    legend = false,
)
scatter!(
    hplt,
    [0],
    [0],
    color = plot_color(:grey, 0.4),
    marker = :circle,
    markerstrokewidth = 5,
    legend = false,
)
xlims!(xmin, xmax)
ylims!(xmin, xmax)

tolx = 1e-6 # approx tolerance

for i = 1:ntests
    xi = [data[i, 2]; data[i, 3]]
    xf = [data[i, 4]; data[i, 5]]

    if norm(xf - x_00) <= tolx
        global c_00 += 1
        scatter!(
            hplt,
            [xi[1]],
            [xi[2]],
            color = :black,
            marker = :circle,
            markerstrokewidth = 0,
            legend = false,
        )
    elseif norm(xf - x_05) <= tolx
        global c_05 += 1
        scatter!(
            hplt,
            [xi[1]],
            [xi[2]],
            color = :red,
            marker = :cross,
            markerstrokewidth = 0,
            legend = false,
        )
    else
        global c_un += 1
        @printf "(%6.4f,%6.4f) from (%6.4f,%6.4f)\n" xf[1] xf[2] xi[1] xi[2]
    end
end

@info "  global (0,0): $(c_00)/$(ntests) ($(100*c_00/ntests)%)"
@info "   local (0,5): $(c_05)/$(ntests) ($(100*c_05/ntests)%)"
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
            global_nabs = c_00,
            global_nrel = 100*c_00/ntests,
            local_nabs = c_05,
            local_nrel = 100*c_05/ntests,
            unkwn_nabs = c_un,
            unkwn_nrel = 100*c_un/ntests,
        ),
    )
CSV.write(filepath * "_stats" * ".csv", stats, header = true)