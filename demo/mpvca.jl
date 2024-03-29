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
using Plots
using DataFrames
using Printf
using CSV
using Statistics

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

# test execution function
function run_mpvca_test(problem_name, solver_name, subsolver_name)
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
        @error "Unknown problem name"
    end

    # setup solver and subsolver
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
    subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
        minimum_gamma = eps(T),
        directions = subsolver_directions,
        maxit = subsolver_maxit,
        freq = subsolver_maxit;
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
    solver(x0) = solver_base(
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

    @info "Problem " * problem_name
    @info "Solver  " * solver_name * "(" * subsolver_name * ")"

    # warm up the solver
    _ = solver(zeros(T, nx))

    ################################################################################
    # grid of starting points
    ################################################################################
    filename = problem_name * "_" * solver_name * "_" * subsolver_name
    filepath = joinpath(@__DIR__, "results", filename)

    xmin = -5.0
    xmax = 20.0

    data = DataFrame()

    xgrid = [(i, j) for i = xmin:0.5:xmax, j = xmin:0.5:xmax]
    xgrid = xgrid[:]
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

        # solve instance
        out = solver(x0)

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

    nsolved = sum(data.is_solved)

    ################################################################################
    # plot results
    ################################################################################

    # minimizers
    x_glb = T.([0; 0])
    x_lcl = T.([0; 5])

    # counters
    global c_glb = 0
    global c_lcl = 0
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

        if norm(xf - x_glb) <= tolx
            global c_glb += 1
            scatter!(
                hplt,
                [xi[1]],
                [xi[2]],
                color = :blue,
                marker = :circle,
                markerstrokewidth = 0,
                legend = false,
            )
        elseif norm(xf - x_lcl) <= tolx
            global c_lcl += 1
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

    @info " global (0, 0): $(c_glb)/$(ntests) ($(100*c_glb/ntests)%)"
    @info "  local (0, 5): $(c_lcl)/$(ntests) ($(100*c_lcl/ntests)%)"
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
            nsolved = nsolved,
            iters_fivenum = five_num_summary(data.iters),
            subiters_fivenum = five_num_summary(data.sub_iters),
            runtime_fivenum = five_num_summary(data.runtime),
            global_nabs = c_glb,
            global_nrel = 100 * c_glb / ntests,
            local_nabs = c_lcl,
            local_nrel = 100 * c_lcl / ntests,
            unkwn_nabs = c_un,
            unkwn_nrel = 100 * c_un / ntests,
        ),
    )
    CSV.write(filepath * "_stats" * ".csv", stats, header = true)

    @info " sample points: $(ntests)  (solved $(nsolved))"
    @info "    iterations: $(stats.iters_fivenum)"
    @info "sub iterations: $(stats.subiters_fivenum)"
    @info "       runtime: $(stats.runtime_fivenum)"
    return stats
end

function five_num_summary(data)
    return quantile(data, [0.01, 0.25, 0.50, 0.75, 0.99])
end

# test setup
problem_names = ["mpvca", "mpvca_slack", "mpvca_fullslack"]
solver_names = ["als", "alps"]
subsolver_names = ["lbfgs"] #["lbfgs","noaccel"]

solver_name = "alps"
for problem_name in problem_names
    for subsolver_name in subsolver_names
        run_mpvca_test(problem_name, solver_name, subsolver_name)
    end
end
solver_name = "als"
problem_name = "mpvca"
for subsolver_name in subsolver_names
    run_mpvca_test(problem_name, solver_name, subsolver_name)
end
