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
using Plots
using DataFrames
using Printf
using CSV
using Statistics

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
function run_rosenbrock_test(problem_name, solver_name, subsolver_name)
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
    elseif subsolver_name == "broyden"
        ProximalAlgorithms.Broyden()
    elseif subsolver_name == "anderson"
        ProximalAlgorithms.AndersonAcceleration(5)
    elseif subsolver_name == "lbfgs"
        ProximalAlgorithms.LBFGS(5)
    else
        @error "Unknown subsolver name"
    end
    subsolver_maxit = 1_000_000_000
    subsolver_minimum_gamma = 1e-32
    subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
        directions = subsolver_directions,
        maxit = subsolver_maxit,
        freq = subsolver_maxit,
        minimum_gamma = subsolver_minimum_gamma;
        kwargs...,
    )
    solver_base = if solver_name == "alps"
        Bazinga.alps
    elseif solver_name == "als"
        Bazinga.als
    else
        @error "Unknown solver name"
    end
    y0 = zeros(T, ny) # dual guess
    solver(f, g, c, D, x0, y0) = solver_base(
        f,
        g,
        c,
        D,
        x0,
        y0,
        tol = 1e-8,
        inner_tol = 1.0,
        verbose = false,
        subsolver = subsolver,
        subsolver_maxit = subsolver_maxit,
    )

    ################################################################################
    # grid of starting points
    ################################################################################
    filename = problem_name * "_" * solver_name * "_" * subsolver_name
    filepath = joinpath(@__DIR__, "results", filename)

    xmin = -5.0
    xmax = 5.0

    data = DataFrame()

    xgrid = [(i, j) for i = xmin:0.25:xmax, j = xmin:0.25:xmax] # 1.0
    xgrid = xgrid[:]
    ntests = length(xgrid)

    @info "Rosenbrock $(ntests) problems"
    for i = 1:ntests
        #@info "Problem $(i) of $(ntests)"
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
                is_solved = out[6] == :first_order ? 1 : 0,
            ),
        )

    end

    CSV.write(filepath * ".csv", data, header = true)

    ################################################################################
    # plot results
    ################################################################################

    # minimizers
    x00 = T.([0; 0])

    # counters
    global c00 = 0
    global cun = 0

    # plot feasible set
    feasset = Shape([
        (0.0, 0.0),
        (xmax, xmin),
        (xmin, xmin),
        (xmin, xmax),
        (xmax, xmax),
        (0.0, 0.0),
    ])
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


    tolx = 1e-6 # approx tolerance

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

    # store some statistics
    nsolved = sum(data.is_solved)
    stats = DataFrame()
    push!(
        stats,
        (
            npoints = ntests,
            nsolved = nsolved,
            iters_fivenum = five_num_summary(data.iters),
            subiters_fivenum = five_num_summary(data.sub_iters),
            runtime_fivenum = five_num_summary(data.runtime),
            global_nabs = c00,
            global_nrel = 100 * c00 / ntests,
            unkwn_nabs = cun,
            unkwn_nrel = 100 * cun / ntests,
        ),
    )
    CSV.write(filepath * "_stats" * ".csv", stats, header = true)

    @info " sample points: $(ntests)  (solved $(nsolved))"
    @info "    iterations: $(stats.iters_fivenum)"
    @info "sub iterations: $(stats.subiters_fivenum)"
    @info "       runtime: $(stats.runtime_fivenum)"

    return data, stats
end

function five_num_summary(data)
    return quantile(data, [0.01, 0.25, 0.50, 0.75, 0.99])
end

problem_name = "rosenbrock"
solver_names = ["als", "alps"] # alps, alps
subsolver_name = "lbfgs" # noaccel, lbfgs

data = Dict{Symbol,DataFrame}()
stats = Dict{Symbol,DataFrame}()

for solver_name in solver_names
    @info solver_name

    k = Symbol(solver_name)

    data[k], stats[k] = run_rosenbrock_test(problem_name, solver_name, subsolver_name)
end

hplt = plot([1e-4, 1.0], [1e-4, 1.0], legend = false)
plot!(xscale = :log10, yscale = :log10, minorgrid = true)
scatter!(data[:als].runtime, data[:alps].runtime)
xlabel!("runtime ALS")
ylabel!("runtime ALPS")

filename = problem_name * "_" * subsolver_name
filepath = joinpath(@__DIR__, "results", filename)
dataplot = DataFrame()
for i = 1:length(data[:als].runtime)
    push!(
        dataplot,
        (runtime_als = data[:als].runtime[i], runtime_alps = data[:alps].runtime[i]),
    )
end
CSV.write(filepath * "_plot" * ".csv", dataplot, header = true)
