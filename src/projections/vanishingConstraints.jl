# vanishing constraint

export VanishingConstraint

"""
    vanishing constraint
    x[1] >= 0   and   x[1] * x[2:end] >= 0
"""

struct VanishingConstraint <: ClosedSet end

is_set(f::VanishingConstraint) = true

function prox!(y, f::VanishingConstraint, x, args...)
    proj!(y, f, x)
    return eltype(x)(0)
end

function proj!(y, f::VanishingConstraint, x)
    if length(x) > 2
        @error "implemented only for dim=2"
    end
    project_onto_VC_set!(y, x)
    return nothing
end

function project_onto_VC_set!(z, x)
    z .= 0
    if x[1] <= 0
        z[2] = x[2]
    else
        if x[2] >= 0
            z .= x
        else # x1 > 0 and x2 < 0
            if x[1] + x[2] > 0
                z[1] = x[1]
            elseif x[1] + x[2] < 0
                z[2] = x[2]
            else # x1 + x2 = 0, set-valued case
                #z[1] = x[1]
                z[2] = x[2]
            end
        end
    end
    return nothing
end
