# Bazinga.jl
Toolbox for constrained structured optimization in [Julia](https://julialang.org/)

Robust, easy to use, yet efficient and fast numerical optimization algorithms designed to find (local) solutions of mathematical optimization problems of the form

```
   min     f(x) + g(x)
  x ∈ Rⁿ
   s.t.    c(x) ∈ S
```
where ```f(x): Rⁿ --> R``` is a smooth objective function, ```g(x): Rⁿ --> R ∪ ∞``` is a proximable objective function, ```c(x): Rⁿ --> Rᵐ``` are the constraint functions, and ```S ⊆ Rᵐ``` is a closed set.

[OptiMo](https://github.com/aldma/OptiMo.jl) provides a tool for modeling problems in this form and giving access to the oracles of interest.

See also [ProximalAlgorithms.jl](https://github.com/kul-forbes/ProximalAlgorithms.jl), the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, and [ALGENCAN](https://www.ime.usp.br/~egbirgin/tango/codes.php#algencan).
