"""
	openp : example from Bazinga.jl

	Parametric constrained minimization of the Rosenbrock function, from [SFP20].

	Original formulations:

	minimize      sum( p2 ( x[i+1] - x[i]^2 ) + ( p1 - x[i] )^2 ) for i=1:4
	subject to    || x || <= p4
	              p3 sin( x1 ) = cos( x2 + x3 )
	              x3 + x4 <= p5
    with          p = [1, 50, 1.5, 0.73, 0.2]

	Reformulation as a constrained structured problem in the form

	minimize     f(x) + g(x)
	subject to   c(x) in S

	where
	    f(x) = sum( p2 ( x[i+1] - x[i]^2 )^2 + ( p1 - x[i] )^2 ) for i=1:4
	    g(x) = IndBall2[radius=p4](x)
	    c(x) = [ p3 sin( x1 ) - cos( x2 + x3 ) ]
	           [ x3 + x4 - p5                  ]
	    S    = { c | c1 = 0, c2 <= 0 }

    References:
    [SFP20]     Sopasakis, Fresk, Patrinos, "OpEn: Code Generation for Embedded
                Nonconvex Optimization" (2020).
                21st IFAC World Congress: Proceedings
                arXiv:2003.00292v1 [math.OC] 29 Feb 2020
"""

foldername = "/home/alberto/Documents/";
push!(LOAD_PATH, foldername * "OptiMo.jl/src");
push!(LOAD_PATH, foldername * "Bazinga.jl/src");

using Bazinga, OptiMo
using Random, LinearAlgebra
using Printf

###################################################################################
# problem definition
###################################################################################
mutable struct OPENP <: AbstractOptiModel
  meta::OptiModelMeta
  p::Vector
end

function OPENP(; p::Vector=Vector([1.0, 50.0, 1.5, 0.73, 0.2]))
    @assert length(p) == 5
    @assert 0 < p[4]
    name = "OpEnParamExample"
    meta = OptiModelMeta( 5, 2, x0=zeros(Float64, 5), name=name)
    return OPENP( meta, p )
end

# necessary methods:
# obj, grad!: cons!, jprod!, jtprod!, proj!, prox!, objprox!
function OptiMo.obj( prob::OPENP, x::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    fx = 0
    for i=1:4
        fx += prob.p[2] * (x[i+1] - x[i]^2)^2 + (prob.p[1] - x[i])^2
    end
    return fx
end

function OptiMo.grad!( prob::OPENP, x::AbstractVector, dfx::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    dfx[1] = - 4.0 * prob.p[2] * x[1] * (x[2] - x[1]^2) - 2.0 * (prob.p[1] - x[1])
    dfx[2] = 2.0 * prob.p[2] * (x[2] - x[1]^2) - 4.0 * prob.p[2] * x[2] * (x[3] - x[2]^2) - 2.0 * (prob.p[1] - x[2])
    dfx[3] = 2.0 * prob.p[2] * (x[3] - x[2]^2) - 4.0 * prob.p[2] * x[3] * (x[4] - x[3]^2) - 2.0 * (prob.p[1] - x[3])
    dfx[4] = 2.0 * prob.p[2] * (x[4] - x[3]^2) - 4.0 * prob.p[2] * x[4] * (x[5] - x[4]^2) - 2.0 * (prob.p[1] - x[4])
    dfx[5] = 2.0 * prob.p[2] * (x[5] - x[4]^2)
    return nothing
end

function OptiMo.cons!( prob::OPENP, x::AbstractVector, cx::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x
    OptiMo.@lencheck prob.meta.ncon cx
    cx.= [ prob.p[3] * sin( x[1] ) - cos( x[2] + x[3] ); x[3] + x[4] - prob.p[5] ]
    return nothing
end

function OptiMo.jprod!( prob::OPENP, x::AbstractVector, v::AbstractVector, Jv::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x v
    OptiMo.@lencheck prob.meta.ncon Jv
    Jv .= [ prob.p[3] * cos( x[1] ) * v[1] + sin( x[2] + x[3] ) * ( v[2] + v[3] ); v[3] + v[4] ]
    return nothing
end

function OptiMo.jtprod!( prob::OPENP, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x Jtv
    OptiMo.@lencheck prob.meta.ncon v
    sx23v1 = sin( x[2] + x[3] ) * v[1]
    Jtv .= [ prob.p[3] * cos( x[1] ) * v[1]; sx23v1; sx23v1 + v[2]; v[2]; 0.0]
    return nothing
end

function OptiMo.prox!( prob::OPENP, x::AbstractVector, a::Real, z::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x .* min( 1.0, prob.p[4] / norm( x, 2 ) )
    return nothing
end

function OptiMo.objprox!( prob::OPENP, x::AbstractVector, a::Real, z::AbstractVector )
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x .* min( 1.0, prob.p[4] / norm( x, 2 ) )
    return 0.0
end

function OptiMo.proj!( prob::OPENP, cx::AbstractVector, px::AbstractVector )
    OptiMo.@lencheck prob.meta.ncon cx px
    px .= [ 0.0; min( 0.0, cx[2] ) ]
    return nothing
end

###################################################################################
# problem build
###################################################################################
p0 = [1.0; 50.0; 1.5; 0.73; 0.2]
problem = OPENP( p=p0 )

###################################################################################
# solver build
###################################################################################
VERBOSE = false

# warm up
solver = Bazinga.ALPX( tol_optim=1e-5, # OpEn 1e-5
                       tol_cviol=1e-4,
                       verbose=VERBOSE )
out = solver( problem )
print( out )
@printf "\n\n\n"

# OpEn setup
solver = Bazinga.ALPX( tol_optim=1e-5, # OpEn 1e-5
                       tol_cviol=1e-4, # OpEn 1e-4
                       eps_init=1e-4,  # OpEn 1e-4
                       mu_down=0.2,    # OpEn 0.2
                       verbose=false )
mu0 = 1e-3 * ones(Float64, problem.meta.ncon) # OpEn 1e-3
out = solver( problem, mu=mu0 )
print( out )

# ALPX setup, OpEn initial penalty
solver = Bazinga.ALPX( tol_optim=1e-5, # OpEn 1e-5
                       tol_cviol=1e-4, # OpEn 1e-4
                       verbose=VERBOSE )
mu0 = 1e-3 * ones(Float64, problem.meta.ncon) # OpEn 1e-3
out = solver( problem, mu=mu0 )
print( out )

# ALPX setup, OpEn tolerances
solver = Bazinga.ALPX( tol_optim=1e-5, # OpEn 1e-5
                       tol_cviol=1e-4, # OpEn 1e-4
                       verbose=VERBOSE )
out = solver( problem )
print( out )

# ALPX setup
solver = Bazinga.ALPX( verbose=VERBOSE )
out = solver( problem )
print( out )
