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

push!(LOAD_PATH,"/home/alberto/Documents/optimo/src");
push!(LOAD_PATH,"/home/alberto/Documents/bazinga/src");

using Bazinga, OptiMo
using Random, LinearAlgebra
using Printf

###################################################################################
# problem definition
###################################################################################
mutable struct EITHEROR <: AbstractOptiModel
  meta::OptiModelMeta
end

function EITHEROR()
    name = "EitherOrExample"
    meta = OptiModelMeta( 2, 4, x0=zeros(Float64,2), name=name)
    return EITHEROR( meta )
end

# necessary methods:
# obj, grad!: cons!, jprod!, jtprod!, proj!, prox!, objprox!
function OptiMo.obj( prob::EITHEROR, x::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    return (x[1] - 8)^2 + (x[2] + 3)^2
end

function OptiMo.grad!( prob::EITHEROR, x::AbstractVector, dfx::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    dfx .= [2*(x[1] - 8); 2*(x[2] + 3)]
    return nothing
end

function OptiMo.cons!( prob::EITHEROR, x::AbstractVector, cx::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    OptiMo.@lencheck prob.meta.ncon cx
    cx .= [ x[1] - 2*x[2] + 4;
            x[1]^2 - 4*x[2];
            x[1] - 2;
            (x[1]-3)^2 + (x[2]-1)^2 - 10 ]
    return nothing
end

function OptiMo.jprod!( prob::EITHEROR, x::AbstractVector, v::AbstractVector, Jv::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x v
    OptiMo.@lencheck prob.meta.ncon Jv
    Jv .= [ v[1] - 2*x[2]*v[2];
            2*x[1]*v[1] - 4*v[2];
            v[1];
            2*(x[1]-3)*v[1] + 2*(x[2]-1)*v[2] ]
    return nothing
end

function OptiMo.jtprod!( prob::EITHEROR, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x Jtv
    OptiMo.@lencheck prob.meta.ncon v
    Jtv .= [ v[1] + 2*x[1]*v[2] + v[3] + 2*(x[1]-3)*v[4];
             -2*x[2]*v[1] - 4*v[2]+2*(x[2]-1)*v[4] ]
    return nothing
end

function OptiMo.prox!( prob::EITHEROR, x::AbstractVector, a::Real, z::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x
    return nothing
end

function OptiMo.objprox!( prob::EITHEROR, x::AbstractVector, a::Real, z::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x
    return 0.0
end

function OptiMo.proj!( prob::EITHEROR, cx::AbstractVector, px::AbstractVector )
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
    return nothing
end
#=function proj_eitheror_constr!( c1,c2,p1,p2 )
    # cx <= 0  or  cy <= 0
    i_out = (c1 .> 0.0) .& (c2 .> 0.0)
    i_y2z = (c1 .> c2)
    p1 .= c1
    p2 .= c2
    p1[i_out .& .!i_y2z] .= 0.0
    p2[i_out .&  i_y2z] .= 0.0
    return nothing
end
using Random
R=Float64
n=100
c=randn(R,n);
d=randn(R,n);
for i=1:100
    a=randn(R,n)
    b=randn(R,n)
    proj_eitheror_constr!( a,b,c,d )
    @assert all((c .<= 0) .| (d .<= 0))
end
@printf "Passed"=#

###################################################################################
# problem build
###################################################################################
problem = EITHEROR()

###################################################################################
# solver build
###################################################################################
solver = Bazinga.ALPX( max_iter=50,
                       max_sub_iter=1000,
                       tol_optim=1e-6,
                       tol_cviol=1e-8,
                       verbose=false,
                       subsolver=:panoc,
                       subsolver_verbose=false )

#out = solver( problem )
#print( out )
#x = out.x;

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
