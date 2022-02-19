mutable struct NonsmoothCostFun{Tg} <: ProximableFunction where {Tg}
    g::Tg
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
function prox!( z, g::NonsmoothCostFun, x, gamma)
    gz = prox!(z, g.g, x, gamma)
    g.gamma = gamma
    g.gz = gz
    return gz
end
