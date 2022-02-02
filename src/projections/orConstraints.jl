# or constraints

"""
    either-or constraint (inclusive logical disjuction)
    x[1] >= 0   or   x[2] >= 0
"""
function project_onto_EITHEROR_set!(z, x)
    z .= x
    if x[1] < 0 && x[2] < 0
        if x[1] > x[2]
            z[1] = 0
        else
            z[2] = 0
        end
    end
    return nothing
end

"""
    xor constraint (exclusive logical disjuction)
    (x[1] >= 0   and   x[2] <= 0)   or   (x[1] <= 0   and   x[2] >= 0)
    closure of (x[1] >= 0   xor   x[2] >= 0)
"""
function project_onto_XOR_set!(z, x)
    z .= x
    if x[1] * x[2] > 0
        if x[1] > x[2]
            z[1] = max(0, x[1])
            z[2] = min(0, x[2])
        else
            z[1] = min(0, x[1])
            z[2] = max(0, x[2])
        end
    end
    return nothing
end
