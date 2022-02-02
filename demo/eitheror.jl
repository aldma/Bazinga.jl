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

using Revise
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
T = Float64
f = SmoothCostOR( T.([8; -3]))
g = ProximalOperators.Zero()
c = ConstraintOR()
D = SetEITHEROR()
#D = SetXOR()

x0 = ones(T,2)
y0 = ones(T,4)

out = Bazinga.alps(f, g, c, D, x0, y0)
