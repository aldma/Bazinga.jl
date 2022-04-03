# zero

"""
    Zero()

returns the zero function
```math
f(x) = 0.
```
"""
struct Zero <: Bazinga.ProximableFunction end

function (f::Zero)(x)
    return 0.0
end

function Bazinga.gradient!(dfx, f::Zero, x)
    dfx .= 0.0
    return 0.0
end

function Bazinga.prox!(y, f::Zero, x, gamma::Number)
    y .= x
    return 0.0
end
