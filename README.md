# Bazinga.jl
Toolbox for constrained structured optimization in [Julia](https://julialang.org/)

Why?
-----------
Robust, easy to use, yet efficient and fast numerical optimization algorithms designed to find (local) solutions of mathematical optimization problems of the form

```
   min     f(x) + g(x)
  x ∈ Rⁿ
   s.t.    c(x) ∈ S
```
where ```f(x): Rⁿ --> R``` is a smooth objective function, ```g(x): Rⁿ --> R ∪ ∞``` is a proximable objective function, ```c(x): Rⁿ --> Rᵐ``` are smooth constraint functions, and ```S ⊆ Rᵐ``` is a closed set.

The problem terms are accessed through some oracles:

* ```f```: function value, gradient
* ```g```: proximal operator, function value at proximal point
* ```c```: function value, Jacobian-vector product
* ```S```: projection operator, distance function

[OptiMo.jl](https://github.com/aldma/OptiMo.jl) provides a modeling language for problems in this form, giving access to the oracles of interest.

License
----------
The Bazinga.jl package is licensed under the MIT License. We provide this program in the hope that it may be useful to others, and we would very much like to hear about your experience with it. If you found it helpful, we encourage you to [get in touch](mailto:aldmarchi@gmail.com) with us.
