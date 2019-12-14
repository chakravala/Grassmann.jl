# Quick start (G2)

Import the `Grassmann` package and instantiate a two-dimensional algebra (G2),

```Julia
julia> using Grassmann

julia> @basis ℝ^2
(⟨++⟩, v, v₁, v₂, v₁₂)

julia> v1*v2 # geometric product
v₁₂

julia> v1|v2 # inner product
0v

julia> v1∧v2 # exterior product
v₁₂
```

## Reflection

```Julia
julia> a = v1+v2
1v₁ + 1v₂

julia> n = v1
v₁

julia> -n*a/n # reflect a in hyperplane normal to n
0.0 - 1.0v₁ + 1.0v₂
```

## Rotation

```Julia
julia> R = exp(π/4*v12)
0.7071067811865476 + 0.7071067811865475v₁₂

julia> R*v1*~R
0.0 + 2.220446049250313e-16v₁ - 1.0v₂
```
