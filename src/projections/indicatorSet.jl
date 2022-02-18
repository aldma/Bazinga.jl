# indicator set
# closed set from indicator function

struct IndicatorSet <: ClosedSet
    f::ProximableFunction
end

function Bazinga.proj!(z, f::IndicatorSet, x)
    ProximalOperators.prox!(z, f.f, x)
    return nothing
end
