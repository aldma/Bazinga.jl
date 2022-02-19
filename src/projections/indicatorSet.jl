# indicator set
# closed set from indicator function

struct IndicatorSet{P} <: ClosedSet where {P}
    f::P
end

function proj!(z, f::IndicatorSet, x)
    prox!(z, f.f, x)
    return nothing
end
