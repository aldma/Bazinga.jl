# free set

"""
    free set
    {x}
"""

struct FreeSet <: ClosedSet end

is_set(f::FreeSet) = true

function prox!(y, f::FreeSet, x, args...)
    proj!(y, f, x)
    return eltype(x)(0)
end

function proj!(y, f::FreeSet, x)
    y .= x
    return nothing
end
