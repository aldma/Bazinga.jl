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

	Reformulation as a constrained structured problem in the form

	minimize     f(x) + g(x)
	subject to   c(x) in D

	where
	    f(x) = 4*x1 + 2*x2
	    g(x) = IndBox[0,+infty](x)
	    c(x) = [ x1                  ]
	           [ x1 + x2 - 5*sqrt(2) ]
       	       [ x2                  ]
	           [ x1 + x2 - 5         ]
	    D    = { c | (c1,c2) in Dvc, (c3,c4) in Dvc }
	with
	    Dvc  = { (a,b) | a >= 0, a*b >= 0 }
             = { (a,b) | a = 0 } ∪ { (a,b) | a >= 0, b >= 0 }

    References:
    [Hoh09]     Hoheisel, "Mathematical Programs with Vanishing Constraints",
                PhD thesis, University of Würzburg, 2009.
"""

using ProximalOperators
using LinearAlgebra
using Bazinga
using ProximalAlgorithms

################################################################################
# problem definition
################################################################################
struct SmoothCostMPVCA <: ProximalOperators.ProximableFunction
    c::AbstractVector
end
function (f::SmoothCostMPVCA)(x)
    return dot(f.c, x[1:2])
end
function ProximalOperators.gradient!(dfx, f::SmoothCostMPVCA, x)
    dfx .= 0
    dfx[1:2] .= f.c
    return dot(f.c, x[1:2])
end

struct NonsmoothCostMPVCA <: ProximalOperators.ProximableFunction end
function ProximalOperators.prox!(z, g::NonsmoothCostMPVCA, x, gamma)
    z .= max.(0, x)
    return zero(eltype(x))
end

struct NonsmoothCostMPVCAslack <: ProximalOperators.ProximableFunction end
function ProximalOperators.prox!(z, g::NonsmoothCostMPVCAslack, x, gamma)
    Bazinga.project_onto_VC_set!(@view(z[[1, 3]]), x[[1, 3]])
    Bazinga.project_onto_VC_set!(@view(z[[2, 4]]), x[[2, 4]])
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

struct SetMPVCA <: ClosedSet end
function Bazinga.proj!(z, D::SetMPVCA, cx)
    Bazinga.project_onto_VC_set!(@view(z[1:2]), cx[1:2])
    Bazinga.project_onto_VC_set!(@view(z[3:4]), cx[3:4])
    return nothing
end

# iter
problem_name = "mpvca" # mpvca, mpvca_slack

T = Float64
f = SmoothCostMPVCA(T.([4; 2]))
if problem_name == "mpvca_slack"
    g = NonsmoothCostMPVCAslack()
    c = ConstraintMPVCAslack()
    D = Bazinga.ZeroSet()
    nx = 4
    ny = 2
else
    g = NonsmoothCostMPVCA()
    c = ConstraintMPVCA()
    D = SetMPVCA()
    nx = 2
    ny = 4
end

# solver setup
subsolver_directions = ProximalAlgorithms.LBFGS(5)
subsolver_maxit = 1_000
subsolver_minimum_gamma = eps(T)
subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = subsolver_directions,
    maxit = subsolver_maxit,
    freq = subsolver_maxit,
    minimum_gamma = subsolver_minimum_gamma;
    kwargs...,
)
solver(f, g, c, D, x0, y0) = Bazinga.alps(
    f,
    g,
    c,
    D,
    x0,
    y0,
    verbose = true,
    subsolver = subsolver,
    subsolver_maxit = subsolver_maxit,
)
_ = solver(f, g, c, D, ones(T, nx), zeros(T, ny))

################################################################################
# grid of starting points
################################################################################
using Plots
using DataFrames
using Printf
using CSV
using Statistics

filename = problem_name
filepath = joinpath(@__DIR__, "results", filename)

xmin = -5.0
xmax = 20.0

data = DataFrame()

xgrid = [(i, j) for i = xmin:1.0:xmax, j = xmin:1.0:xmax];
xgrid = xgrid[:];
ntests = length(xgrid)

for i = 1:ntests
    if problem_name == "mpvca_slack"
        x0 = [
            xgrid[i][1]
            xgrid[i][2]
            xgrid[i][1] + xgrid[i][2] - 5 * sqrt(2)
            xgrid[i][1] + xgrid[i][2] - 5
        ]
    else
        x0 = [xgrid[i][1]; xgrid[i][2]]
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

tolx = 1e-3 # approx tolerance

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

savefig(filepath * ".pdf")
