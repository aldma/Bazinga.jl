module Bazinga

using LinearAlgebra
import ProximalCore: prox, prox!, gradient, gradient!
using ProximalAlgorithms

abstract type ProximableFunction end
abstract type SmoothFunction end
abstract type ClosedSet end

export SmoothFunction
export eval!, jtprod, jtprod!
export ClosedSet
export proj, proj!, dist, dist!
export ProximableFunction
export prox, prox!, gradient, gradient!

ClosedSet(f) = IndicatorSet(f)

# projections
include("projections/zeroSet.jl")
include("projections/freeSet.jl")
include("projections/indicatorSet.jl")
include("projections/orConstraints.jl")
include("projections/vanishingConstraints.jl")
include("projections/complementarityConstraints.jl")

include("proxoperators/zero.jl")
include("proxoperators/rank.jl")
include("proxoperators/schattenNormLp.jl")
include("proxoperators/normL1Nonneg.jl")
include("proxoperators/normL1Box.jl")
include("proxoperators/normLpNonneg.jl")
include("proxoperators/normLpBox.jl")
include("proxoperators/normL0Box.jl")

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
function dist(f::ClosedSet, x, p::Real = 2)
    y = similar(x)
    return dist!(y, f, x)
end

function dist!(y, f::ClosedSet, x, p::Real = 2)
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
