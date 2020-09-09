# PANOC problem : tiny nonconvex QP
#
# minimize      Quadratic[Q,q](x) + IndBox[l,u](x)
#
# where
#    Q = [-0.5, 0.0; 0.0, 1.0]
#    q = [0.3; 0.5]
#    l = [-1.0; -1.0]
#    u = [1.0; 1.0]

using OptiMo

# Type
mutable struct PPNCVXQP <: AbstractOptiModel
  meta::OptiModelMeta
end

# Constructor
function PPNCVXQP()
  meta = OptiModelMeta(2, 0, name="Nonconvex QP (tiny)")
  return PPNCVXQP( meta )
end

##########################################################
# Necessary methods:
# obj, grad!, prox!, objprox!
##########################################################
function OptiMo.obj(prob::PPNCVXQP, x::AbstractVector)
    # f(x)
    return 0.5 * x[2]^2 - 0.25 * x[1]^2 + 0.3 * x[1] + 0.5 * x[2]
end

function OptiMo.grad!(prob::PPNCVXQP, x::AbstractVector, dfx::AbstractVector)
    # df/dx(x)
    dfx[1] = - 0.5 * x[1] + 0.3
    dfx[2] = x[2] + 0.5
    return nothing
end

function OptiMo.prox!(prob::PPNCVXQP, x::AbstractVector, a::Real, z::AbstractVector)
    # prox_{a g}( x )
    z[1] = max(-1.0, min(x[1], 1.0))
    z[2] = max(-1.0, min(x[2], 1.0))
    return nothing
end

function OptiMo.objprox!(prob::PPNCVXQP, x::AbstractVector, a::Real, z::AbstractVector)
    # prox_{a g}( x )
    # g(z)
    z[1] = max(-1.0, min(x[1], 1.0))
    z[2] = max(-1.0, min(x[2], 1.0))
    return 0.0
end
