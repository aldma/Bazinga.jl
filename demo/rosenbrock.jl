"""
	rosenbrock : example from Bazinga.jl

    Academic example of involving a nonsmooth Rosenbrock-like objective function
    and either-or constraints.

	Original formulations:

	minimize      10 * (x2 + 1 - (x1 + 1)^2)^2 + |x1|
	subject to    x2 <= - x1   or   x2 >= x1

	Reformulation as a constrained structured problem in the form

	minimize     f(x) + g(x)
	subject to   c(x) in D

	where
	    f(x) = 10 * (x2 + 1 - (x1 + 1)^2)^2
	    g(x) = |x1|
	    c(x) = [ - x1 - x2 ]
	           [ - x1 + x2 ]
	    D    = Deo
	with
	    Deo  = { (a,b) | a >= 0 } ∪ { (a,b) | b >= 0 }
"""

using LinearAlgebra
using Bazinga
using ProximalAlgorithms

###################################################################################
# problem definition
###################################################################################
struct SmoothCostRosenbrock <: Bazinga.ProximableFunction
    w::Real
end
function (f::SmoothCostRosenbrock)(x)
    return f.w * (x[2] + 1 - (x[1] + 1)^2)^2
end
function Bazinga.gradient!(dfx, f::SmoothCostRosenbrock, x)
    tmp = x[2] + 1 - (x[1] + 1)^2
    dfx[1] = -4 * f.w * tmp * (x[1] + 1)
    dfx[2] = 2 * f.w * tmp
    return f.w * tmp^2
end

struct NonsmoothCostRosenbrock <: Bazinga.ProximableFunction
    lambda::Real
end
function Bazinga.prox!(y, g::NonsmoothCostRosenbrock, x, gamma)
    gl = gamma * g.lambda
    y[1] = if abs(x[1]) <= gl
        0.0
    else
        sign(x[1]) * (abs(x[1]) - gl)
    end
    y[2] = x[2]
    return g.lambda * abs(y[1])
end

struct ConstraintRosenbrock <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintRosenbrock, x)
    cx .= [-x[1] - x[2]; x[2] - x[1]]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintRosenbrock, x, v)
    jtv .= [-v[1] - v[2]; v[2] - v[1]]
    return nothing
end

struct SetRosenbrock <: ClosedSet end
function Bazinga.proj!(z, D::SetRosenbrock, cx)
    Bazinga.project_onto_EITHEROR_set!(z, cx)
    return nothing
end

################################################################################
# problems and solvers
################################################################################
problem_name = "rosenbrock"
solver_name = "als" # alps, alps
subsolver_name = "lbfgs" # noaccel, lbfgs

T = Float64

rosenbrock_w = T(10)
rosenbrock_l = T(1)
f = SmoothCostRosenbrock(rosenbrock_w)
g = NonsmoothCostRosenbrock(rosenbrock_l)
c = ConstraintRosenbrock()
D = SetRosenbrock()
nx = 2
ny = 2

subsolver_directions = if subsolver_name == "noaccel"
    ProximalAlgorithms.NoAcceleration()
elseif subsolver_name == "lbfgs"
    ProximalAlgorithms.LBFGS(5)
else
    @error "Unknown acceleration"
end

subsolver_maxit = 1_000_000
subsolver_minimum_gamma = eps(T)
subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = subsolver_directions,
    maxit = subsolver_maxit,
    freq = subsolver_maxit,
    minimum_gamma = subsolver_minimum_gamma;
    kwargs...,
)
solver(f, g, c, D, x0, y0) = Bazinga.als(
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

################################################################################
# grid of starting points
################################################################################
using Plots
using DataFrames
using Printf
using CSV
using Statistics

filename = solver_name * "_" * subsolver_name * "_" * string(subsolver_maxit)
filepath = joinpath(@__DIR__, "results", problem_name, filename)

xmin = -5.0
xmax = 5.0

data = DataFrame()

xgrid = [(i, j) for i = xmin:1.0:xmax, j = xmin:1.0:xmax];
xgrid = xgrid[:];
ntests = length(xgrid)

for i = 1:ntests
    @info "Problem $(i) of $(ntests)"
    x0 = [xgrid[i][1]; xgrid[i][2]]
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
x00 = T.([0; 0])

# counters
global c00 = 0
global cun = 0

# plot feasible set
feasset =
    Shape([(0.0, 0.0), (xmax, xmin), (xmin, xmin), (xmin, xmax), (xmax, xmax), (0.0, 0.0)])
hplt = plot(feasset, color = plot_color(:grey, 0.4), linewidth = 0, legend = false)
xlims!(xmin, xmax)
ylims!(xmin, xmax)
# plot contour lines
xvector = xmin:0.02:xmax
yvector = xmin:0.02:xmax
rosenbrockFunction(x, y) = begin
    rosenbrock_w * (y + 1 - (x + 1)^2)^2 + rosenbrock_l * abs(x)
end
Xmatrix = repeat(reshape(xvector, 1, :), length(yvector), 1)
Ymatrix = repeat(yvector, 1, length(xvector))
Zmatrix = map(rosenbrockFunction, Xmatrix, Ymatrix)
lvls = [0; 10 .^ collect(range(-6, log10(maximum(Zmatrix)), length = 49))]
contour!(hplt, xvector, yvector, Zmatrix, levels = lvls)


tolx = 1e-3 # approx tolerance

for i = 1:ntests
    xi = [data[i, 2]; data[i, 3]]
    xf = [data[i, 4]; data[i, 5]]

    if norm(xf - x00) <= tolx
        global c00 += 1
        scatter!(
            hplt,
            [xi[1]],
            [xi[2]],
            color = :blue,
            marker = :circle,
            markerstrokewidth = 0,
            legend = false,
        )
    else
        global cun += 1
        @printf "(%6.4f,%6.4f) from (%6.4f,%6.4f)\n" xf[1] xf[2] xi[1] xi[2]
    end
end

savefig(filepath * ".pdf")
