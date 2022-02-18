module Bazinga

using LinearAlgebra
using ProximalOperators
import ProximalOperators: prox!, gradient!
using ProximalAlgorithms

abstract type SmoothFunction <: ProximableFunction end
abstract type ClosedSet <: ProximableFunction end

export SmoothFunction
export eval!, jtprod, jtprod!
export ClosedSet
export proj, proj!, dist, dist!

ClosedSet(f::ProximableFunction) = begin
    if !(ProximalOperators.is_set(f))
        @error "$(f) is not a set!"
    end
    return IndicatorSet(f)
end

# projections
include("projections/zeroSet.jl")
include("projections/freeSet.jl")
include("projections/indicatorSet.jl")
include("projections/orConstraints.jl")
include("projections/vanishingConstraints.jl")

# utilities
include("utilities/auglagfun.jl")
include("utilities/nonsmoothcostfun.jl")

# algorithms
include("algorithms/alps.jl")

# projection mapping
function proj(f::ClosedSet, x)
    y = similar(x)
    proj!(y, f, x)
    return y
end

proj!

# distance function
function dist(f::ClosedSet, x, p::Real=2)
    y = similar(x)
    return dist!(y, f, x)
end

function dist!(y, f::ClosedSet, x, p::Real=2)
    proj!(y, f, x)
    return norm(x - y, p)
end

# function evaluation (in-place)
function eval!(fx, f::ProximableFunction, x)
    return nothing
end

# Jacobian-vector product
function jtprod(f::ProximableFunction, x, v)
    jtv = similar(x)
    jtprod!(jtv, f, x, v)
    return jtv
end

function jtprod!(jtv, f::ProximableFunction, x, v)
    dfx, _ = gradient(f, x)
    jtv .= dfx' * v
    return nothing
end

end # module
