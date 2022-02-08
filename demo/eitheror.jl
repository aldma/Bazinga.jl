"""
	eitheror : example from Bazinga.jl

	Mathematical program with either-or constraints, from [KMS18].

	Original formulations:

	minimize      (x1 - 8)^2 + (x2 + 3)^2
	subject to    x1 - 2 x2 + 4 ≤ 0    or    x1 - 2 ≤ 0
	              x1^2 - 4 x2 ≤ 0      or    (x1 - 3)^2 + (x2 - 1)^2 - 10 ≤ 0

	Reformulation as a constrained structured problem in the form

	minimize     f(x) + g(x)
	subject to   c(x) in D

	where
	    f(x) = (x1 - 8)^2 + (x2 + 3)^2
	    g(x) = IndBox[-10,10](x)
	    c(x) = [ 2 x2 - x1 - 4                ]
               [ 2 - x1                       ]
       	       [ 4 x2 - x1^2                  ]
       	       [ 10 - (x1 - 3)^2 - (x2 - 1)^2 ]
        D    = { c | (c1,c2) in Deo, (c3,c4) in Deo }
	with
	    Deo  = { (a,b) | a >= 0 or b >= 0 }

    References:
    [KMS18]     Kanzow, Mehlitz, Steck, "Relaxation schemes for mathematical
                programs with switching constraints" (2019).
"""

using ProximalOperators
using LinearAlgebra
using Bazinga

###################################################################################
# problem definition
###################################################################################
struct SmoothCostOR <: ProximalOperators.ProximableFunction
    c::AbstractVector
end
function (f::SmoothCostOR)(x)
    return norm(x - f.c, 2)^2
end
function ProximalOperators.gradient!(dfx, f::SmoothCostOR, x)
    dfx .= 2 * (x - f.c)
    return norm(x - f.c, 2)^2
end

struct NonsmoothCostOR <: ProximalOperators.ProximableFunction end
function ProximalOperators.prox!(y, g::NonsmoothCostOR, x, gamma)
    y .= max.(-10, min.(x, 10))
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
    jtv[1] = - v[1] - v[2] - 2 * x[1] * v[3] + 2 * (3 - x[1]) * v[4]
    jtv[2] = 2 * v[1] + 4 * v[3] + 2 * (1 - x[2]) * v[4]
    return nothing
end

struct SetEITHEROR <: ClosedSet end
function Bazinga.proj!(z, D::SetEITHEROR, x)
    Bazinga.project_onto_EITHEROR_set!(@view(z[1:2]), @view(x[1:2]))
    Bazinga.project_onto_EITHEROR_set!(@view(z[3:4]), @view(x[3:4]))
    return nothing
end
struct SetXOR <: ClosedSet end
function Bazinga.proj!(z, D::SetXOR, x)
    Bazinga.project_onto_XOR_set!(@view(z[1:2]), @view(x[1:2]))
    Bazinga.project_onto_XOR_set!(@view(z[3:4]), @view(x[3:4]))
    return nothing
end

# iter
problem_name = "xor"
T = Float64
f = SmoothCostOR( T.([8; -3]) )
g = NonsmoothCostOR()
c = ConstraintOR()
if problem_name == "xor"
    D = SetXOR()
else
    D = SetEITHEROR()
end

x0 = ones(T,2)
y0 = zeros(T,4)

out = Bazinga.alps(f, g, c, D, x0, y0, verbose=true)

################################################################################
# grid of starting points
################################################################################
using DataFrames
using Plots
using CSV

filename = problem_name * "_grid"
filepath = joinpath(@__DIR__, "results", filename)

xmin = -4.0
xmax =  8.0

data = DataFrame()

xgrid = [(i, j) for i = xmin:0.25:xmax, j = xmin:0.25:xmax];
xgrid = xgrid[:];
ntests = length(xgrid)

for i = 1:ntests
    x0 = [xgrid[i][1]; xgrid[i][2]]
    y0 = zeros(T,4)

    out = Bazinga.alps(f, g, c, D, x0, y0)

    xsol = out[1]
    push!(data, (id = i,
                 xinit_1 = x0[1],
                 xinit_2 = x0[2],
                 xfinal_1 = xsol[1],
                 xfinal_2 = xsol[2],
                 iters=out[3],
                 sub_iters=out[4],
                 runtime=out[5],
                 ))

end

CSV.write(filepath * ".csv", data, header = false)

################################################################################
# plot results
################################################################################

# minimizers
x_22 = T.([2; -2])
x_44 = T.([4; 4])

# counters
global c22 = 0
global c44 = 0
global cun = 0

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

tolx = 1e-3 # approx tolerance

for i = 1:ntests
    xi = [data[i, 2]; data[i, 3]]
    xf = [data[i, 4]; data[i, 5]]

    if norm(xf - x_22) <= tolx
        global c22 += 1
        scatter!(hplt, [xi[1]], [xi[2]], color=:blue, marker=:circle, markerstrokewidth=0, legend = false)
    elseif norm(xf - x_44) <= tolx
        global c44 += 1
        scatter!(hplt, [xi[1]], [xi[2]], color=:red, marker=:diamond, markerstrokewidth=0, legend = false)
    else
        global cun += 1
        @printf "(%6.4f,%6.4f) from (%6.4f,%6.4f)\n" xf[1] xf[2] xi[1] xi[2]
    end
end

savefig(filepath * ".pdf")
