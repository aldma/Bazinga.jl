# complementarity constraint

"""
    complementarity constraint
    x[1], x[2] >= 0   and   x[1] * x[2] = 0
"""

function project_onto_CC_set!(z, x)
    if x[1] > 0 && x[2] > 0
        z .= x
        if x[2] > x[1]
            z[1] = 0
        else
            z[2] = 0
        end
    else
        z .= max.(0, x)
    end
    return nothing
end
