# Quick start (G2)

Import the `Grassmann` package and instantiate a two-dimensional algebra (G2),

```@repl ga
using Grassmann
@basis ℝ^2
v1*v2 # geometric product
v1|v2 # inner product
v1∧v2 # exterior product
```

## Reflection

```@example ga
a = v1+v2
n = v1
-n*a/n # reflect a in hyperplane normal to n
```

## Rotation

```@repl ga
R = exp(π/4*v12)
~R*v1*R
```
