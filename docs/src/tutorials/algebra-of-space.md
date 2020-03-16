# The Algebra of Space (G3)

This notebook is an adaptation from the [clifford](https://clifford.readthedocs.io/en/latest/TheAlgebraOfSpaceG3.html) python documentation.

Import `Grassmann` and instantiate a three dimensional geometric algebra

```@repl ga
using Grassmann
basis"3"
```

Given a three dimensional GA with the orthonormal basis ``v_i\cdot v_j = \delta_{ij}``, the basis consists of scalars, three vectors, three bivectors, and a trivector.
```math
\{\underbrace{v}_{\text{scalar}},\qquad\underbrace{v_1,v_2,v_3}_{\text{vectors}},\qquad\underbrace{v_{12},v_{23},v_{13}}_{\text{bivectors}},\qquad\underbrace{v_{123}}_{\text{trivector}}\}
```
The `@basis` macro declares the algebra and assigns the `SubManifold` elements to local variables. The `Basis` can also be assigned to `G3` as
```@repl ga
G3 = Λ(3)
```
You may wish to explicitly assign the blades to variables like so,
```julia
e1 = G3.v1
e2 = G3.v2
# etc ...
```
Or, if you're lazy you can use the macro with different local names
```@repl ga
@basis ℝ^3 E e
e3, e123
```

## Basics

The basic products are available

```@repl ga
v1 * v2 # geometric product
v1 | v2 # inner product
v1 ∧ v2 # exterior product
v1 ∧ v2 ∧ v3 # even more exterior products
```

Multivectors can be defined in terms of the basis blades. For example, you can construct a rotor as a sum of a scalar and a bivector, like so
```@repl ga
θ = π/4
R = cos(θ) + sin(θ)*v23
R = exp(θ*v23)
```
You can also mix grades without any reason
```@repl ga
A = 1 + 2v1 + 3v12 + 4v123
```
The reversion operator is accomplished with the tilde `~` in front of the `MultiVector` on which it acts
```@repl ga
~A
```
Taking a projection into a specific `grade` of a `MultiVector` is usually written ``\langle A\rangle_n`` and can be done using the soft brackets, like so
```@repl ga
A(0)
A(1)
A(2)
```
Using the reversion and grade projection operators, we can define the magnitude of `A` as ``|A|^2 = \langle\tilde A A\rangle``
```@repl ga
~A*A
scalar(ans)
```
This is done in the `abs` and `abs2` operators
```@repl ga
abs2(A)
scalar(ans)
```
The dual of a multivector `A` can be defined as ``\tilde AI``, where `I` is the pseudoscalar for the geometric algebra. In `G3`, the dual of a vector is a bivector:
```@repl ga
a = 1v1 + 2v2 + 3v3
⋆a
```

## Reflections

Reflecting a vector ``c`` about a normalized vector ``n`` is pretty simple, ``c\mapsto -ncn``
```@repl ga
c = v1+v2+v3 # a vector
n = v1 # the reflector
-n*c*n # reflect a in hyperplane normal to n
```
Because we have the `inv` available, we can equally well reflect in un-normalized vectors using ``a\mapsto n^{-1}an``
```@repl ga
a = v1+v2+v3 # the vector
n = 3v1 # the reflector
inv(n)*a*n
n\a*n
```
Reflections can also be made with respect to the hyperplane normal to the vector, in which case the formula is negated.

## Rotations

A vector can be rotated using the formula ``a\mapsto \tilde R aR``, where `R` is a rotor. A rotor can be defined by multiple reflections, ``R = mn`` or by a plane and an angle ``R = e^{\theta B/2}``.
For example,
```@repl ga
R = exp(π/4*v12)
~R*v1*R
```
Maybe we want to define a function which can return rotor of some angle ``\theta`` in the ``v_{12}``-plane, ``R_{12} = e^{\theta v_{12}/2}``
```@example ga
R12(θ) = exp(θ/2*v12)
nothing # hide
```
And use it like this
```@repl ga
R = R12(π/2)
a = v1+v2+v3
~R*a*R
```
You might as well make the angle argument a bivector, so that you can control the plane of rotation as well as the angle
```@example ga
R_B(B) = exp(B/2)
nothing # hide
```
Then you could do
```@repl ga
Rxy = R_B(π/4*v12)
Ryz = R_B(π/5*v23)
```
or
```@repl ga
R_B(π/6*(v23+v12))
```
Maybe you want to define a function which returns a *function* that enacts a specified rotation, ``f(B) = a\mapsto e^{B/2}\\ae^{B/2}``.
This just saves you having to write out the sandwich product, which is nice if you are cascading a bunch of rotors, like so
```@example ga
R_factory(B) = (R = exp(B/2); a -> ~R*a*R)
Rxy = R_factory(π/3*v12)
Ryz = R_factory(π/3*v23)
Rxz = R_factory(π/3*v13)
nothing # hide
```
Then you can do things like
```@repl ga
R = R_factory(π/6*(v23+v12)) # this returns a function
R(a) # which acts on a vector
Rxy(Ryz(Rxz(a)))
```
To make cascading a sequence of rotations as concise as possible, we could define a function which takes a list of bivectors ``A,B,C,...``, and enacts the sequence of rotations which they represent on some vector ``x``.
```@repl ga
R_seq(args...) = (R = prod(exp.(args./2)); a -> ~R*a*R)
R = R_seq(π/2*v23, π/2*v12, v1)
R(v1)
```

## Barycentric Coordinates

We can find the barycentric coordinates of a point in a triangle using area ratios.
```@repl ga
function barycoords(p, a, b, c)
  ab = b-a
  ca = a-c
  bc = c-b
  A = -ab∧ca
  (bc∧(p-b)/A, ca∧(p-c)/A, ab∧(p-a)/A)
end
barycoords(0.25v1+0.25v2, 0v1, 1v1, 1v2)
```

```
