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

push!(LOAD_PATH,"/home/albertodm/Documents/optimo/src");
push!(LOAD_PATH,"/home/albertodm/Documents/bazinga.jl/src");

using Bazinga, OptiMo
using Random, LinearAlgebra
using Printf

###################################################################################
# problem definition
###################################################################################
mutable struct MPVCA <: AbstractOptiModel
  meta::OptiModelMeta
end

function MPVCA(; ncon::Int=4)
    @assert 4 <= ncon <= 5
    name = "AcademicMPVC"*"-$ncon"
    meta = OptiModelMeta(2, ncon, x0=[5.0; 5.5], name=name)
    return MPVCA( meta )
end

# necessary methods:
# obj, grad!: cons!, jprod!, jtprod!, proj!, prox!, objprox!
function OptiMo.obj( prob::MPVCA, x::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    return 4*x[1] + 2*x[2]
end

function OptiMo.grad!( prob::MPVCA, x::AbstractVector, dfx::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    dfx .= [4; 2]
    return nothing
end

function OptiMo.cons!( prob::MPVCA, x::AbstractVector, cx::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    OptiMo.@lencheck prob.meta.ncon cx
    cx[1:4] .= [x[1]; x[2]; x[1] + x[2] - 5.0*sqrt(2.0); x[1] + x[2] - 5.0]
    if prob.meta.ncon > 4
        cx[5] = x[1] + x[2] - 3.0
    end
    return nothing
end

function OptiMo.jprod!( prob::MPVCA, x::AbstractVector, v::AbstractVector, Jv::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x v
    OptiMo.@lencheck prob.meta.ncon Jv
    v12 = v[1] + v[2]
    if prob.meta.ncon > 4
        Jv .= [ v[1]; v[2]; v12; v12; v12 ]
    else
        Jv .= [ v[1]; v[2]; v12; v12 ]
    end
    return nothing
end

function OptiMo.jtprod!( prob::MPVCA, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector )
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

function OptiMo.prox!( prob::MPVCA, x::AbstractVector, a::Real, z::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x z
    z .= max.( 0.0, x )
    return nothing
end

function OptiMo.objprox!( prob::MPVCA, x::AbstractVector, a::Real, z::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x z
    z .= max.( 0.0, x )
    return 0.0
end

function OptiMo.proj!( prob::MPVCA, cx::AbstractVector, px::AbstractVector )
    OptiMo.@lencheck prob.meta.ncon cx px
    # vanishing constraint
    px[1] = max( 0.0, cx[1] )
    px[3] = cx[3]
    if (px[1] + px[3] < 0)
        px[1] = 0.0
    else
        px[3] = max( px[3], 0.0 )
    end
    # vanishing constraint
    px[2] = max( 0.0, cx[2] )
    px[4] = cx[4]
    if (px[2] + px[4] < 0)
        px[2] = 0.0
    else
        px[4] = max( px[4], 0.0 )
    end
    # inequality (ncon == 5)
    if prob.meta.ncon > 4
        px[5] = max( 0.0, cx[5] )
    end
    return nothing
end

###################################################################################
# problem build
###################################################################################
problem = MPVCA( ncon=5 )

###################################################################################
# solver build
###################################################################################
solver = Bazinga.ALPX( max_iter=10,
                       max_sub_iter=1000,
                       verbose=false,
                       subsolver=:panoc )

R = eltype( problem.meta.x0 )
nvar = problem.meta.nvar

global c00 = 0
global c05 = 0
global c11 = 0
global cun = 0

xmin = -5.0
xmax = 20.0

ntests = 1e+2
ndots = round(Int,cbrt(ntests^2))

for i=1:ntests

    x0 = xmin .+ (xmax - xmin) .* rand(R,nvar);

    out = solver( problem, x0=x0 )

    x = out.x;

    if norm(x,Inf) <= 1e-4
        global c00 += 1
        #@printf "0,0 \n"
        #scatter!(plt, x0[1], x0[2],color=:orangered)
    elseif norm(x-[0.0;5.0],Inf) <= 1e-4
        global c05 += 1
        #@printf "5,0 from (%4.2f,%4.2f)\n" x0[1] x0[2]
        #scatter!(plt, x0[1], x0[2],color=:steelblue)
    elseif x[1] >= 0 && x[2] >= 0 && x[1]+x[2] <= 3
        global c11 += 1
        #@printf "1,1 from (%4.2f,%4.2f)\n" x0[1] x0[2]
        #scatter!(plt, x0[1], x0[2],color=:steelblue)
    else
        global cun += 1
        @printf "(%6.4f,%6.4f) from (%6.4f,%6.4f)\n" x[1] x[2] x0[1] x0[2]
        #scatter!(plt, [x0[1],x[1]], [x0[2],x[2]],color=:green)
    end

    @printf "."
    if mod(i,ndots) == 0
        @printf "\n"
    end

end

#@printf "\n\n"
#x0 = [3.8665; 5.5150]
#out = solver( problem, x0=x0 )
