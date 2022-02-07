# Bazinga.jl

A toolbox for constrained structured optimization in Julia.

## Why?
-----------
Robust, easy to use, yet efficient and fast numerical optimization algorithms designed to find (local) solutions of mathematical optimization problems of the form

```
   minimize        f(x) + g(x)
   with respect to x ∈ Rⁿ
   subject to      c(x) ∈ D
```
where ```f(x): Rⁿ --> R``` is a smooth objective function (locally Lipschitz continuous gradient), ```g(x): Rⁿ --> R ∪ ∞``` is a proximable objective function, ```c(x): Rⁿ --> Rᵐ``` are smooth constraint functions, and ```D ⊆ Rᵐ``` is a nonempty closed set.

The problem terms are accessed through some oracles:

* ```f```: function value, gradient
* ```g```: proximal mapping, function value at proximal point
* ```c```: function value, Jacobian-vector product
* ```D```: projection mapping

[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) provides first-order primitives (gradient and proximal mapping) for modelling problems and giving access to the oracles of interest.

License
----------
The Bazinga.jl package is licensed under the MIT License. We provide this program in the hope that it may be useful to others, and we would very much like to hear about your experience with it. If you found it helpful, we encourage you to [get in touch](mailto:aldmarchi@gmail.com) with us.
