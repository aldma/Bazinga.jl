# identity function

struct IdentityFunction <: SmoothFunction end

function Bazinga.eval!(fx, f::IdentityFunction, x)
    fx .= x
    return nothing
end

function Bazinga.jtprod!(jtv, f::IdentityFunction, x, v)
    jtv .= v
    return nothing
end
