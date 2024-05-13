# Bazinga.jl

A toolbox for constrained structured optimization in Julia.

## Synopsis
This package contains optimization algorithms designed to find (local) minimizers of mathematical problems of the form
```
   minimize        f(x) + g(x)
   over            x ∈ Rⁿ
   subject to      c(x) ∈ D
```
where ``f`` and ``c`` have locally Lipschitz-continuous gradient, ``g`` is proper and lower semi-continuous, and ``D`` is a nonempty closed set.
All these terms can be nonconvex, but ``g`` and ``D`` should be prox-friendly.

The problem terms are accessed through some oracles:

* ```f```: function value f(x) and gradient ∇f(x)
* ```g```: proximal mapping prox_g(x) and function value at proximal point g(z)
* ```c```: function value c(x) and Jacobian-vector product ∇c(x)ᵀv
* ```D```: projection mapping proj_D(v)

[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) provides first-order primitives (gradient and proximal mapping) for modelling problems and giving access to the oracles of interest. The ``PANOCplus`` solver offered by [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) is adopted as default to solve subproblems arising in the augmented Lagrangian framework.

## Citing

If you are using Bazinga for your work or research, we encourage you to cite our research papers, where you can find more details on the mathematical background.
```
   @article{demarchi2023constrained,
      author   = {De~Marchi, Alberto and Jia, Xiaoxi and Kanzow, Christian and Mehlitz, Patrick},
      title    = {Constrained composite optimization and augmented {L}agrangian methods},
      journal  = {Mathematical Programming},
      year     = {2023},
      month    = {9},
      volume   = {201},
      number   = {1},
      pages    = {863--896},
      doi      = {10.1007/s10107-022-01922-4},
}
   @article{demarchi2024implicit,
      author   = {De~Marchi, Alberto},
      title    = {Implicit augmented {L}agrangian and generalized optimization},
      journal  = {Journal of Applied and Numerical Optimization},
      year     = {2024},
      pages    = {291--320},
      volume   = {6},
      number   = {2},
      doi      = {10.23952/jano.6.2024.2.08},
}
```
We are looking forward to hearing your success stories with Bazinga! Please [share them with us](mailto:aldmarchi@gmail.com).

## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/aldma/Bazinga.jl/issues). All types of issues are welcome including bug reports, typos, feature requests and so on.
