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

using Bazinga, OptiMo
using Random, LinearAlgebra
using Printf

###################################################################################
# problem definition
###################################################################################
mutable struct DISJP <: AbstractOptiModel
  meta::OptiModelMeta
end

function DISJP()
    name = "DisjProgExample"
    nvar = 3
    ncon = 5
    meta = OptiModelMeta( nvar, ncon, x0=ones(Float64,nvar), name=name)
    return DISJP( meta )
end

# necessary methods:
# obj, grad!: cons!, jprod!, jtprod!, proj!, prox!, objprox!
function OptiMo.obj( prob::DISJP, x::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    return (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] + 2)^2
end

function OptiMo.grad!( prob::DISJP, x::AbstractVector, dfx::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    dfx .= [2*(x[1] - 1); 2*(x[2] - 2); 2*(x[3] + 2)]
    return nothing
end

function OptiMo.cons!( prob::DISJP, x::AbstractVector, cx::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    OptiMo.@lencheck prob.meta.ncon cx
    cx .= [ x[1] - 4;
            x[1] + (x[2] - 2)^2 + (x[3] + 2)^2 - 5;
            x[3] - x[1]^2 - x[2]^2;
            (x[1] - 1)^2 + x[2]^2 + x[3] - 1;
            - x[2] ]
    return nothing
end

function OptiMo.jprod!( prob::DISJP, x::AbstractVector, v::AbstractVector, Jv::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x v
    OptiMo.@lencheck prob.meta.ncon Jv
    Jv .= [ v[1];
            v[1] + 2*(x[2]-2)*v[2] + 2*(x[3]+2)*v[3];
            -2*x[1]*v[1] - 2*x[2]*v[2] + v[3];
            2*(x[1]-1)*v[1] + 2*x[2]*v[2] + v[3];
            -v[2] ]
    return nothing
end

function OptiMo.jtprod!( prob::DISJP, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x Jtv
    OptiMo.@lencheck prob.meta.ncon v
    Jtv .= [ v[1] + v[2] - 2*x[1]*v[3];
             2*(x[2]-2)*v[2] - 2*x[2]*v[3] + 2*x[2]*v[4] - v[5];
             2*(x[3]+2)*v[2] + v[3] + v[4] ]
    return nothing
end

function OptiMo.prox!( prob::DISJP, x::AbstractVector, a::Real, z::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x
    return nothing
end

function OptiMo.objprox!( prob::DISJP, x::AbstractVector, a::Real, z::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x
    return 0.0
end

function OptiMo.proj!( prob::DISJP, cx::AbstractVector, px::AbstractVector )
    OptiMo.@lencheck prob.meta.ncon cx px
    px .= cx
    if any(cx[1:2] .< 0) && any(cx[3:5] .< 0)
        px .= max.(cx, 0)
        if norm(px[1:2]-cx[1:2], 2) < norm(px[3:5]-cx[3:5], 2)
            px[3:5] .= cx[3:5]
        else
            px[1:2] .= cx[1:2]
        end
    end
    return nothing
end

###################################################################################
# problem build
###################################################################################
problem = DISJP()

###################################################################################
# solver build
###################################################################################
solver = Bazinga.ALPX( max_iter=50,
                       max_sub_iter=1000,
                       tol_optim=1e-6,
                       tol_cviol=1e-8,
                       verbose=true,
                       subsolver=:zerofpr,
                       subsolver_verbose=false )

out = solver( problem )
print( out )
x = out.x;

#=
R = eltype( problem.meta.x0 )
nvar = problem.meta.nvar

global c22 = 0
global c44 = 0
global cun = 0

xmin = -5.0
xmax = +5.0

ntests = 1e+3
atol = 1e-4
ndots = round(Int,cbrt(ntests^2))

for i=1:ntests

    x0 = xmin .+ (xmax - xmin) .* rand(R,nvar);

    out = solver( problem, x0=x0 )

    x = out.x;

    if norm(x - [2; -2], Inf) <= atol
        global c22 += 1
    elseif norm(x - [4; 4], Inf) <= atol
        global c44 += 1
    else
        global cun += 1
        @printf "(%6.4f,%6.4f) from (%6.4f,%6.4f)\n" x[1] x[2] x0[1] x0[2]
    end

    @printf "."
    if mod(i,ndots) == 0
        @printf "\n"
    end

end
=#
