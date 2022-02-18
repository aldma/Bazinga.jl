# zero set

"""
    zero set
    {0}
"""

struct ZeroSet <: ClosedSet end

is_set(f::ZeroSet) = true

function prox!(y, f::ZeroSet, x, args...)
    proj!(y, f, x)
    return eltype(x)(0)
end

function proj!(y, f::ZeroSet, x)
    y .= eltype(x)(0)
    return nothing
end
