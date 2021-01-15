"""
	eitheror : example from Bazinga.jl

	Mathematical program with either-or constraints, from [KMS18].

	Original formulations:

	minimize      (x1 - 8)^2 + (x2 + 3)^2
	subject to    x1 - 2 x2 + 4 ≤ 0    or    x1 - 2 ≤ 0
	              x1^2 - 4 x2 ≤ 0      or    (x1 - 3)^2 + (x2 - 1)^2 - 10 ≤ 0

	Reformulation as a constrained structured problem in the form

	minimize     f(x)
	subject to   c(x) in S

	where
	    f(x) = (x1 - 8)^2 + (x2 + 3)^2
	    c(x) = [ x1 - 2 x2 + 4               ]
       	       [ x1^2 - 4 x2                 ]
   	           [ x1 - 2                      ]
       	       [(x1 - 3)^2 + (x2 - 1)^2 - 10 ]
	    S    = { (a,b) | a ≤ 0  or  b ≤ 0 }^2

    References:
    [KMS18]     Kanzow, Mehlitz, Steck, "Relaxation schemes for mathematical
                programs with switching constraints" (2019).
                ....
                arXiv:1809.02388v1 [math.OC] 7 Sep 2018
"""

using Bazinga, OptiMo
using Random, LinearAlgebra
using DataFrames, CSV
using Printf, Plots

###################################################################################
# problem definition
###################################################################################
mutable struct EITHEROR <: AbstractOptiModel
    meta::OptiModelMeta
end

function EITHEROR()
    name = "eitheror"
    meta = OptiModelMeta(2, 4, x0 = zeros(Float64, 2), name = name)
    return EITHEROR(meta)
end

# necessary methods:
# obj, grad!: cons!, jprod!, jtprod!, proj!, prox!, objprox!
function OptiMo.obj(prob::EITHEROR, x::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    return (x[1] - 8)^2 + (x[2] + 3)^2
end

function OptiMo.grad!(prob::EITHEROR, x::AbstractVector, dfx::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    dfx .= [2 * (x[1] - 8); 2 * (x[2] + 3)]
    return nothing
end

function OptiMo.cons!(prob::EITHEROR, x::AbstractVector, cx::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    OptiMo.@lencheck prob.meta.ncon cx
    cx .= [
        x[1] - 2 * x[2] + 4
        x[1]^2 - 4 * x[2]
        x[1] - 2
        (x[1] - 3)^2 + (x[2] - 1)^2 - 10
    ]
    return nothing
end

function OptiMo.jprod!(
    prob::EITHEROR,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    OptiMo.@lencheck prob.meta.nvar x v
    OptiMo.@lencheck prob.meta.ncon Jv
    Jv .= [
        v[1] - 2 * v[2]
        2 * x[1] * v[1] - 4 * v[2]
        v[1]
        2 * (x[1] - 3) * v[1] + 2 * (x[2] - 1) * v[2]
    ]
    return nothing
end

function OptiMo.jtprod!(
    prob::EITHEROR,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    OptiMo.@lencheck prob.meta.nvar x Jtv
    OptiMo.@lencheck prob.meta.ncon v
    Jtv .= [
        v[1] + 2 * x[1] * v[2] + v[3] + 2 * (x[1] - 3) * v[4]
        -2 * v[1] - 4 * v[2] + 2 * (x[2] - 1) * v[4]
    ]
    return nothing
end

function OptiMo.prox!(prob::EITHEROR, x::AbstractVector, a::Real, z::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x
    return nothing
end

function OptiMo.objprox!(prob::EITHEROR, x::AbstractVector, a::Real, z::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x
    return 0.0
end

function OptiMo.proj!(prob::EITHEROR, cx::AbstractVector, px::AbstractVector)
    OptiMo.@lencheck prob.meta.ncon cx px
    if cx[1] > 0 && cx[3] > 0
        if cx[1] > cx[3]
            px[1] = cx[1]
            px[3] = 0
        else
            px[1] = 0
            px[3] = cx[3]
        end
    else
        px[1] = cx[1]
        px[3] = cx[3]
    end
    if cx[2] > 0 && cx[4] > 0
        if cx[2] > cx[4]
            px[2] = cx[2]
            px[4] = 0
        else
            px[2] = 0
            px[4] = cx[4]
        end
    else
        px[2] = cx[2]
        px[4] = cx[4]
    end
    #if cx[1] > 0 && cx[3] > 0
    # ...
    return nothing
end

# problem build
problem = EITHEROR()

foldername = "/home/albertodm/Documents/Bazinga.jl/"
filename = problem.meta.name * "_grid_rev"

# solver build
solver =
    Bazinga.ALPX(max_iter = 50, max_sub_iter = 1000, verbose = false, subsolver = :zerofpr)

R = eltype(problem.meta.x0)
nvar = problem.meta.nvar

xmin = -4.0
xmax = 8.0

data = DataFrame()

xgrid = [(i, j) for i = xmin:0.25:xmax, j = xmin:0.25:xmax];
xgrid = xgrid[:];
ntests = length(xgrid)

x0 = [xgrid[1][1]; xgrid[1][2]]
out = solver(problem, x0 = x0)

for i = 1:ntests
    x0 = [xgrid[i][1]; xgrid[i][2]]

    out = solver(problem, x0 = x0)

    @printf "."
    if mod(i, 50) == 0
        @printf "\n"
    end

    push!(
        data,
        (
            id = i,
            xi_1 = x0[1],
            xi_2 = x0[2],
            xf_1 = out.x[1],
            xf_2 = out.x[2],
            iter = out.iterations,
            time = out.time,
            solved = out.status == :first_order ? 1 : 0,
        ),
    )

end
@printf "\n"

CSV.write(foldername * "demo/data/" * filename * ".csv", data, header = false)

################################################################################
tolx = 1e-3

global c22 = 0
global c44 = 0
global cun = 0

pyplot()

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
    xx = 4 + (xmax - 4) * i / 50
    yy = (xx^2) / 4
    append!(pts, [(xx, yy)])
end
append!(pts, [(xmin, xmax)])
for i = 0:50
    xx = xmin * (1 - i / 50)
    yy = (xx^2) / 4
    append!(pts, [(xx, yy)])
end
feasset = Shape(pts)
hplt = plot(feasset, color = plot_color(:grey, 0.4), linewidth = 0, legend = false)
xlims!(xmin, xmax)
ylims!(xmin, xmax)

for i = 1:ntests
    xi = [data[i, 2]; data[i, 3]]
    xf = [data[i, 4]; data[i, 5]]

    if norm(xf - [2.0; -2.0], Inf) <= tolx
        global c22 += 1
        scatter!(
            hplt,
            [xi[1]],
            [xi[2]],
            color = :blue,
            marker = :circle,
            markerstrokewidth = 0,
            legend = false,
        )
    elseif norm(xf - [4.0; 4.0], Inf) <= tolx
        global c44 += 1
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
        global cun += 1
        @printf "(%6.4f,%6.4f) from (%6.4f,%6.4f)\n" xf[1] xf[2] xi[1] xi[2]
    end
end
savefig(foldername * "demo/data/" * filename * ".pdf")
