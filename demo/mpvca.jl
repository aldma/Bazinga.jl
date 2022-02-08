"""
	mpvca : example from Bazinga.jl

    Academic example of mathematical program with vanishing constraints (MPVC),
    see Section 9.5.1 [Hoh09].

	Original formulations:

	minimize      4*x1 + 2*x2
	subject to    x1                       >= 0
	              x2                       >= 0
	              (x1 + x2 - 5*sqrt(2))*x1 >= 0
	              (x1 + x2 - 5        )*x2 >= 0

	Reformulation as a constrained structured problem in the form

	minimize     f(x) + g(x)
	subject to   c(x) in S

	where
	    f(x) = 4*x1 + 2*x2
	    g(x) = IndBox[0,+infty](x)
	    c(x) = [ x1                  ]
	           [ x2                  ]
	           [ x1 + x2 - 5*sqrt(2) ]
	           [ x1 + x2 - 5         ]
	    S   = { c | (c1,c3) in SV, (c2,c4) in SV }
	with
	    SV  = { (a,b) | a >= 0, a*b >= 0 }

    References:
    [Hoh09]     Hoheisel, "Mathematical Programs with Vanishing Constraints",
                PhD thesis, University of Würzburg, 2009.
"""

using ProximalOperators
using LinearAlgebra
using Bazinga

###################################################################################
# problem definition
###################################################################################
struct SmoothCostMPVCA <: ProximalOperators.ProximableFunction
    c::AbstractVector
end
function (f::SmoothCostMPVCA)(x)
    return dot(f.c, x)
end
function ProximalOperators.gradient!(dfx, f::SmoothCostMPVCA, x)
    dfx .= f.c
    return dot(f.c, x)
end

struct NonsmoothCostMPVCA <: ProximalOperators.ProximableFunction end
function ProximalOperators.prox!(y, g::NonsmoothCostMPVCA, x, gamma)
    y .= max.(0, x)
    return zero(eltype(x))
end

struct ConstraintMPVCA <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintMPVCA, x)
    cx .= [x[1]; x[1]+x[2]-5*sqrt(2); x[2]; x[1]+x[2]-5]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintMPVCA, x, v)
    jtv .= [v[1]+v[2]+v[4]; v[2]+v[3]+v[4]]
    return nothing
end

struct SetMPVCA <: ClosedSet end
function Bazinga.proj!(z, D::SetMPVCA, x)
    Bazinga.project_onto_VC_set!(@view(z[1:2]), @view(x[1:2]))
    Bazinga.project_onto_VC_set!(@view(z[3:4]), @view(x[3:4]))
    return nothing
end

# iter
T = Float64
f = SmoothCostMPVCA( T.([4; 2]))
g = NonsmoothCostMPVCA()
c = ConstraintMPVCA()
D = SetMPVCA()

x0 = ones(T,2)
y0 = zeros(T,4)

out = Bazinga.alps(f, g, c, D, x0, y0, verbose=true)

###############################################################################
###############################################################################
###############################################################################
using DataFrames
using Printf
using Plots
using CSV

problem_name = "mpvca"
filename = problem_name * "_grid"
filepath = joinpath(@__DIR__, "results", filename)

xmin = -5.0
xmax = 20.0

data = DataFrame()

xgrid = [(i, j) for i = xmin:0.25:xmax, j = xmin:0.25:xmax];
xgrid = xgrid[:];
ntests = length(xgrid)

for i = 1:ntests
    x0 = [xgrid[i][1]; xgrid[i][2]]
    y0 = zeros(T,4)

    out = Bazinga.alps(f, g, c, D, x0, y0)

    @printf "."
    if mod(i, 50) == 0
        @printf "\n"
    end

    xsol = out[1]
    push!(data, (id = i, xi_1 = x0[1], xi_2 = x0[2], xf_1 = xsol[1], xf_2 = xsol[2], iter=out[3], sub_iter=out[4], time=out[5]))

end
@printf "\n"

CSV.write(filepath * ".csv", data, header=false)

################################################################################
tolx = 1e-3

global c_00 = 0
global c_05 = 0
global c_11 = 0
global c_un = 0

feasset = Shape([(0.0,xmax),(0.0,5*sqrt(2)),(5*sqrt(2),0.0),(xmax,0.0),(xmax,xmax),(0.0,xmax)])
hplt = plot(feasset, color=plot_color(:grey,0.4), linewidth=0, legend = false)
xlims!(xmin,xmax)
ylims!(xmin,xmax)

for i in 1:ntests
    xi = [data[i,2]; data[i,3]]
    xf = [data[i,4]; data[i,5]]

    if norm(xf) <= tolx
        global c_00 += 1
        scatter!(hplt, [xi[1]], [xi[2]], color=:blue, marker=:circle, markerstrokewidth=0, legend = false)
    elseif norm(xf - [0.0;5.0]) <= tolx
        global c_05 += 1
        scatter!(hplt, [xi[1]], [xi[2]], color=:red, marker=:diamond, markerstrokewidth=0, legend = false)
    elseif xf[1] >= 0 && xf[2] >= 0 && xf[1]+xf[2] <= 3
        global c_11 += 1
        scatter!(hplt, [xi[1]], [xi[2]], color=:green, marker=:star5, markerstrokewidth=0, legend = false)
    else
        global c_un += 1
    end
end

savefig(filepath * ".pdf")
