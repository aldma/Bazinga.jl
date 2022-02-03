"""
	disjp : example from Bazinga.jl

	Mathematical program with disjunctive constraints, from [Meh19].

	Original formulations:

	minimize      f(x)
	subject to    x ∈ X1 ∪ X2
    with          f(x) := (x1 - 1)^2 + (x2 - 2)^2 + (x3 + 2)^2
                  X1 := { (x1,x2,x3) | x1 ≥ 4, x1 + (x2 - 2)^2 + (x3 + 2)^2 ≥ 5 }
                  X2 := { (x1,x2,x3) | x1^2 + x2^2 ≤ x3, (x1 - 1)^2 + x2^2 + x3 ≥ 1, x2 ≤ 0 }

	Reformulation as a constrained structured problem in the form

	minimize     f(x)
	subject to   c(x) in S

	where
	    c(x) := [ x1 - 4                           ]
       	        [ x1 + (x2 - 2)^2 + (x3 + 2)^2 - 5 ]
   	            [ x3 - x1^2 - x2^2                 ]
       	        [ (x1 - 1)^2 + x2^2 + x3 - 1       ]
                [ - x2                             ]
	    S    := { (c12,c345) | c12 ≥ 0  or  c345 ≥ 0 }

    References:
    [Meh19]     Mehlitz, "A comparison of first-order methods for the numerical
                solution of or-constrained optimization problems" (2019).
                ....
                arXiv:1905.01893v1 [math.OC] 6 May 2019
"""

using ProximalOperators
using LinearAlgebra
using Bazinga

###################################################################################
# problem definition
###################################################################################
struct SmoothCostDISJP <: ProximalOperators.ProximableFunction end
function (f::SmoothCostDISJP)(x)
    return (x[1]-1)^2 + (x[2]-2)^2 + (x[3]+2)^2
end
function ProximalOperators.gradient!(dfx, f::SmoothCostDISJP, x)
    dfx[1] = 2*(x[1]-1)
    dfx[2] = 2*(x[2]-2)
    dfx[3] = 2*(x[3]+2)
    return (x[1]-1)^2 + (x[2]-2)^2 + (x[3]+2)^2
end

struct NonsmoothCostDISJP <: ProximalOperators.ProximableFunction end
function ProximalOperators.prox!(y, g::NonsmoothCostDISJP, x, gamma)
    y .= x
    return zero(eltype(x))
end

struct ConstraintDISJP <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintDISJP, x)
    cx .= [x[1]-4;
            x[1]+(x[2]-2)^2+(x[3]+2)^2-5;
            x[3]-x[1]^2-x[2]^2;
            (x[1]-1)^2+x[2]^2+x[3]-1;
            -x[2]]
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintDISJP, x, v)
    jtv[1] = v[1]+v[2]-2*x[1]*v[3]+2*(x[1]-1)*v[4]
    jtv[2] = 2*(x[2]-2)*v[2]-2*x[2]*v[3]+2*x[2]*v[4]-v[5]
    jtv[3] = 2*(x[3]+2)*v[2]+v[3]+v[4]
    return nothing
end

struct SetDISJP <: ClosedSet end
function Bazinga.proj!(z, D::SetDISJP, x)
    z .= max.(0, x)
    d1 = norm(z[1:2]-x[1:2], 2)
    d2 = norm(z[3:5]-x[3:5], 2)
    if d1 > d2
        z[1:2] .= x[1:2]
    else
        z[3:5] .= x[3:5]
    end
    return nothing
end

# iter
T = Float64
f = SmoothCostDISJP()
g = NonsmoothCostDISJP()
c = ConstraintDISJP()
D = SetDISJP()

x0 = ones(T,3)
y0 = zeros(T,5)

out = Bazinga.alps(f, g, c, D, x0, y0)
