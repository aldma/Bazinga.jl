module Bazinga

const Maybe{T} = Union{T,Nothing}

# compatibility
include("utilities/compat.jl")

# utilities
include("utilities/fbetools.jl")
include("utilities/lbfgs.jl")
include("utilities/restore.jl")
include("utilities/scaling.jl")

# algorithms
include("algorithms/zerofpr.jl")
include("algorithms/panoc.jl")
include("algorithms/alpx.jl")

# TODO pxal : proximal augmented lagrangian
#                     optimize   f(x) + g(x)   s.t.   c(x) ∈ S
#             by considering the problem
#                     optimize   f(x) + h(x)
#             where
#                     h(x) = g(x) + Ind[S](c(x))
#             and whose proximal operator
#                     prox_{a h}(u) = argmin_x { 2 a h(x) + ||x-u||^2 }
#             corresponds to
#                     optimize   q(x) + g(x)   s.t.   c(x) ∈ S
#             where
#                     q(x) = ||x-u||^2 / (2 a)
#             Thus, it can be evaluated by calling, e.g., `alpx`.

end # module
