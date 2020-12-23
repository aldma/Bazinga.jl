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
 		          x1 + x2 - 3              >= 0 (ncon == 5)

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
		       [ x1 + x2 - 3         ] (ncon == 5)
	    S   = { c | (c1,c3) in SV, (c2,c4) in SV, c5 >= 0 }
	with
	    SV  = { (a,b) | a >= 0, a*b >= 0 }

    References:
    [Hoh09]     Hoheisel, "Mathematical Programs with Vanishing Constraints",
                PhD thesis, University of Würzburg, 2009.
"""

push!(LOAD_PATH, "/home/albertodm/Documents/OptiMo.jl/src/")
push!(LOAD_PATH, "/home/albertodm/Documents/Bazinga.jl/src/")

using Bazinga, OptiMo
using Random, LinearAlgebra
using DataFrames, CSV
using Printf, Plots

###################################################################################
# problem definition
###################################################################################
mutable struct MPVCA <: AbstractOptiModel
    meta::OptiModelMeta
end

function MPVCA(; ncon::Int = 4)
    @assert 4 <= ncon <= 5
    name = "mpvca" * "_$ncon"
    meta = OptiModelMeta(2, ncon, x0 = [5.0; 5.5], name = name)
    return MPVCA(meta)
end

# necessary methods:
# obj, grad!: cons!, jprod!, jtprod!, proj!, prox!, objprox!
function OptiMo.obj(prob::MPVCA, x::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    return 4 * x[1] + 2 * x[2]
end

function OptiMo.grad!(prob::MPVCA, x::AbstractVector, dfx::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    dfx .= [4; 2]
    return nothing
end

function OptiMo.cons!(prob::MPVCA, x::AbstractVector, cx::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    OptiMo.@lencheck prob.meta.ncon cx
    cx[1:4] .= [x[1]; x[2]; x[1] + x[2] - 5.0 * sqrt(2.0); x[1] + x[2] - 5.0]
    if prob.meta.ncon > 4
        cx[5] = x[1] + x[2] - 3.0
    end
    return nothing
end

function OptiMo.jprod!(
    prob::MPVCA,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    OptiMo.@lencheck prob.meta.nvar x v
    OptiMo.@lencheck prob.meta.ncon Jv
    v12 = v[1] + v[2]
    if prob.meta.ncon > 4
        Jv .= [v[1]; v[2]; v12; v12; v12]
    else
        Jv .= [v[1]; v[2]; v12; v12]
    end
    return nothing
end

function OptiMo.jtprod!(
    prob::MPVCA,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    OptiMo.@lencheck prob.meta.nvar x Jtv
    OptiMo.@lencheck prob.meta.ncon v
    if prob.meta.ncon > 4
        v345 = v[3] + v[4] + v[5]
    else
        v345 = v[3] + v[4]
    end
    Jtv .= [v[1] + v345; v[2] + v345]
    return nothing
end

function OptiMo.prox!(prob::MPVCA, x::AbstractVector, a::Real, z::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x z
    z .= max.(0.0, x)
    return nothing
end

function OptiMo.objprox!(prob::MPVCA, x::AbstractVector, a::Real, z::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x z
    OptiMo.prox!(prob, x, a, z)
    return 0.0
end

function OptiMo.proj!(prob::MPVCA, cx::AbstractVector, px::AbstractVector)
    OptiMo.@lencheck prob.meta.ncon cx px
    # vanishing constraint
    px[1] = max(0.0, cx[1])
    px[3] = cx[3]
    if (px[1] + px[3] < 0)
        px[1] = 0.0
    else
        px[3] = max(px[3], 0.0)
    end
    # vanishing constraint
    px[2] = max(0.0, cx[2])
    px[4] = cx[4]
    if (px[2] + px[4] < 0)
        px[2] = 0.0
    else
        px[4] = max(px[4], 0.0)
    end
    # inequality (ncon == 5)
    if prob.meta.ncon > 4
        px[5] = max(0.0, cx[5])
    end
    return nothing
end

###############################################################################
###############################################################################
###############################################################################
#=function MPVCA()
    ncon = 4
    name = "mpvca" * "_5g"
    meta = OptiModelMeta(2, ncon, x0 = [5.0; 5.0], name = name)
    return MPVCA(meta)
end
function OptiMo.prox!(prob::MPVCA, x::AbstractVector, a::Real, z::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x z
    c = 3.0
    @assert(c ≥ 0.0)
    if x[1] ≥ c || x[2] ≥ c
        z .= max.(0.0, x)
    elseif x[1] + x[2] ≥ c
        z .= max.(0.0, x)
    elseif x[2] - x[1] ≥ c
        z .= [0.0; c]
    elseif x[1] - x[2] ≥ c
        z .= [c; 0.0]
    else
        z[1] = min(c, (c+x[1]-x[2])/2.0)
        z[2] = min(c, (c-x[1]+x[2])/2.0)
    end
    return nothing
end
=#




###############################################################################
###############################################################################
###############################################################################
foldername = "/home/albertodm/Documents/Bazinga.jl/demo/data/"

# problem build
#problem = MPVCA(ncon = 4)
problem = MPVCA(ncon = 5)
#problem = MPVCA()

# solver build
solver =
    Bazinga.ALPX(max_iter = 10, max_sub_iter = 1000, verbose = false, subsolver = :zerofpr)

R = eltype(problem.meta.x0)
nvar = problem.meta.nvar

xmin = -5.0
xmax = 20.0

data = DataFrame()

xgrid = [(i, j) for i = xmin:0.5:xmax, j = xmin:0.5:xmax];
xgrid = xgrid[:];
ntests = length(xgrid)

for i = 1:ntests
    x0 = [xgrid[i][1]; xgrid[i][2]]

    out = solver(problem, x0 = x0)

    @printf "."
    if mod(i, 50) == 0
        @printf "\n"
    end

    push!(data, (id = i, xi_1 = x0[1], xi_2 = x0[2], xf_1 = out.x[1], xf_2 = out.x[2], iter=out.iterations, time=out.time))

end
@printf "\n"

filename = problem.meta.name * "_grid"
CSV.write(foldername * filename * ".csv", data, header=false)

################################################################################
tolx = 1e-3

global c_00 = 0
global c_05 = 0
global c_11 = 0
global c_un = 0

pyplot()

feasset = Shape([(0.0,xmax),(0.0,5*sqrt(2)),(5*sqrt(2),0.0),(xmax,0.0),(xmax,xmax),(0.0,xmax)])
hplt = plot(feasset, color=plot_color(:grey,0.4), linewidth=0, legend = false)
xlims!(xmin,xmax)
ylims!(xmin,xmax)

for i in 1:ntests
    xi = [data[i,2]; data[i,3]]
    xf = [data[i,4]; data[i,5]]

    if norm(xf,Inf) <= tolx
        global c_00 += 1
        scatter!(hplt, [xi[1]], [xi[2]], color=:blue, marker=:circle, markerstrokewidth=0, legend = false)
    elseif norm(xf - [0.0;5.0],Inf) <= tolx
        global c_05 += 1
        scatter!(hplt, [xi[1]], [xi[2]], color=:red, marker=:diamond, markerstrokewidth=0, legend = false)
    elseif xf[1] >= 0 && xf[2] >= 0 && xf[1]+xf[2] <= 3
        global c_11 += 1
        scatter!(hplt, [xi[1]], [xi[2]], color=:green, marker=:star5, markerstrokewidth=0, legend = false)
    else
        global c_un += 1
    end
end

savefig("/home/albertodm/Documents/Bazinga.jl/demo/data/" * filename * ".pdf")
