# Bazinga.jl

A toolbox for constrained structured optimization in Julia.

## Synopsis
This package contains optimization algorithms designed to find (local) minimizers of mathematical problems of the form
```
   minimize        f(x) + g(x)
   with respect to x ∈ Rⁿ
   subject to      c(x) ∈ D
```
where ``f`` and ``c`` have locally Lipschitz-continuous gradient, ``g`` is proper and lower semi-continuous, and ``D`` is a nonempty closed set.
All these terms can be nonconvex.

The problem terms are accessed through some oracles:

* ```f```: function value f(x) and gradient ∇f(x)
* ```g```: proximal mapping prox_g(x) and function value at proximal point g(z)
* ```c```: function value c(x) and Jacobian-vector product ∇c(x)ᵀv
* ```D```: projection mapping proj_D(v)

[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) provides first-order primitives (gradient and proximal mapping) for modelling problems and giving access to the oracles of interest. The ``PANOCplus`` solver offered by [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) is adopted to solve subproblems arising in the augmented Lagrangian framework.

## Citing

If you are using Bazinga for your work or research, we encourage you to

* Cite our work,
* Put a star on this repository.

We are looking forward to hearing your success stories with Bazinga! Please [share them with us](mailto:aldmarchi@gmail.com).


## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/aldma/Bazinga.jl/issues). All types of issues are welcome including bug reports, typos, feature requests and so on.
