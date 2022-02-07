mutable struct NonsmoothCostFun <: ProximalOperators.ProximableFunction
    g::ProximableFunction
    gamma::Real                     # proximal stepsize
    gz::Real                        # g at z, z being the proximal update
end

"""
    Constructor
    NonsmoothCostFun(g)
"""
function NonsmoothCostFun(g)
    NonsmoothCostFun( g, 0.0, 0.0 )
end
"""
    gz = prox!( z, g, x, gamma )
"""
function ProximalOperators.prox!( z, g::NonsmoothCostFun, x, gamma)
    gz = ProximalOperators.prox!(z, g.g, x, gamma)
    g.gamma = gamma
    g.gz = gz
    return gz
end
