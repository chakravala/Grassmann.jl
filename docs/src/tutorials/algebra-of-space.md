# The Algebra of Space (G3)

This notebook is an adaptation from the [clifford](https://clifford.readthedocs.io/en/latest/TheAlgebraOfSpaceG3.html) python documentation.

Import `Grassmann` and instantiate a three dimensional geometric algebra

```Julia
julia> using Grassmann

julia> basis"3"
(⟨+++⟩, v, v₁, v₂, v₃, v₁₂, v₁₃, v₂₃, v₁₂₃)
```

Given a three dimensional GA with the orthonormal basis ``v_i\cdot v_j = \delta_{ij}``, the basis consists of scalars, three vectors, three bivectors, and a trivector.
```math
\{\underbrace{v}_{\mbox{scalar}},\qquad\underbrace{v_{1},v_{2},v_{3}}_{\mbox{vectors}},\qquad\underbrace{v_{12},v_{23},v_{13}}_{\mbox{bivectors}},\qquad\underbrace{v_{123}}_{\mbox{trivector}}\}
```
The `@basis` macro declares the algebra and assigns the `Basis` elements to local variables. The `Grassmann.Algebra` can also be assigned to `G3` as
```Julia
julia> G3 = Λ(3)
Grassmann.Algebra{⟨+++⟩,8}(v, v₁, v₂, v₃, v₁₂, v₁₃, v₂₃, v₁₂₃)
```
You may wish to explicitly assign the blades to variables like so,
```Julia
e1 = G3.v1
e2 = G3.v2
# etc ...
```
Or, if you're lazy you can use the macro with different local names
```Julia
julia> @basis ℝ^3 E e
(⟨+++⟩, v, v₁, v₂, v₃, v₁₂, v₁₃, v₂₃, v₁₂₃)

julia> e3, e123
(v₃, v₁₂₃)
```

## Basics

The basic products are available

```Julia
julia> v1*v2 # geometric product
v₁₂

julia> v1|v2 # inner product
0v

julia> v1 ∧ v2 # exterior product
v₁₂

julia> v1 ∧ v2 ∧ v3 # even more exterior products
v₁₂₃
```

Multivectors can be defined in terms of the basis blades. For example, you can construct a rotor as a sum of a scalar and a bivector, like so
```Julia
julia> θ = π/4
0.7853981633974483

julia> R = cos(θ) - sin(θ)*v23
0.7071067811865476 - 0.7071067811865475v₂₃

julia> R = ~exp(θ*v23)
0.7071067811865476 - 0.7071067811865475v₂₃
```
You can also mix grades without any reason
```Julia
julia> A = 1 + 2v1 + 3v12 + 4v123
1 + 2v₁ + 3v₁₂ + 4v₁₂₃
```
The reversion operator is accomplished with the tilde `~` in front of the `MultiVector` on which it acts
```Julia
julia> ~A
1 + 2v₁ - 3v₁₂ - 4v₁₂₃
```
Taking a projection into a specific `grade` of a `MultiVector` is usually written ``\langle A\rangle_n`` and can be done using the soft brackets, like so
```Julia
julia> A(0)
1v

julia> A(1)
2v₁ + 0v₂ + 0v₃

julia> A(2)
3v₁₂ + 0v₁₃ + 0v₂₃
```
Using the reversion and grade projection operators, we can define the magnitude of `A` as ``|A|^2 = \langle{\tilde A A\rangle``
```Julia
julia> ~A*A
30 + 4v₁ + 12v₂ + 24v₃

julia> scalar(ans)
30v
```
This is done in the `abs` and `abs2` operators
```Julia
julia> abs2(A)
30 + 4v₁ + 12v₂ + 24v₃

julia> scalar(ans)
30v
```
The dual of a multivector `A` can be defined as ``\tilde AI``, where `I` is the pseudoscalar for the geometric algebra. In `G3`, the dual of a vector is a bivector:
```Julia
julia> a = 1v1 + 2v2 + 3v3
1v₁ + 2v₂ + 3v₃

julia> ⋆a
3v₁₂ - 2v₁₃ + 1v₂₃
```

## Reflections

Reflecting a vector ``c`` about a normalized vector ``n`` is pretty simple, ``c\mapsto -ncn``
```Julia
julia> c = v1+v2+v3 # a vector
1v₁ + 1v₂ + 1v₃

julia> n = v1 # the reflector
v₁

julia> -n*c*n # reflect a in hyperplane normal to n
0.0 - 1.0v₁ + 1.0v₂ + 1.0v₃
```
Because we have the `inv` available, we can equally well reflect in un-normalized vectors using ``a\mapsto n^{-1}an``
```Julia
julia> a = v1+v2+v3 # the vector
1v₁ + 1v₂ + 1v₃

julia> n = 3v1 # the reflector
3v₁

julia> inv(n)*a*n
0.0 + 1.0v₁ - 1.0v₂ - 1.0v₃

julia> n\a*n
0.0 + 1.0v₁ - 1.0v₂ - 1.0v₃
```
Reflections can also be made with respect to the hyperplane normal to the vector, in which case the formula is negated.

## Rotations

A vector can be rotated using the formula ``a\mapsto \tilde R aR``, where `R` is a rotor. A rotor can be defined by multiple reflections, ``R = mn`` or by a plane and an angle ``R = e^{\theta B/2}``.
For example,
```Julia
julia> R = exp(π/4*v12)
0.7071067811865476 + 0.7071067811865475v₁₂

julia> ~R*v1*R
0.0 + 2.220446049250313e-16v₁ + 1.0v₂
```
Maybe we want to define a function which can return rotor of some angle ``\theta`` in the ``v_{12}``-plane, ``R_{12} = e^{\theta v_{12}/2}``
```Julia
R12(θ) = exp(θ/2*v12)
```
And use it like this
```Julia
julia> R = R12(π/2)
0.7071067811865476 + 0.7071067811865475v₁₂

julia> a = v1+v2+v3
1v₁ + 1v₂ + 1v₃

julia> ~R*a*R
0.0 - 0.9999999999999997v₁ + 1.0v₂ + 1.0v₃
```
You might as well make the angle argument a bivector, so that you can control the plane of rotation as well as the angle
```Julia
R_B(B) = exp(B/2)
```
Then you could do
```Julia
julia> Rxy = R_B(π/4*v12)
0.9238795325112867 + 0.3826834323650898v₁₂

julia> Ryz = R_B(π/5*v23)
0.9510565162951535 + 0.3090169943749474v₂₃
```
or
```Julia
julia> R_B(π/6*(v23+v12))
0.9322404424570728 + 0.25585909935689327v₁₂ + 0.25585909935689327v₂₃
```
Maybe you want to define a function which returns a *function* that enacts a specified rotation, ``f(B) = a\mapsto e^{B/2}\\ae^{B/2}``.
This just saves you having to write out the sandwich product, which is nice if you are cascading a bunch of rotors, like so
```Julia
R_factory(B) = (R = exp(B/2); a -> ~R*a*R)
Rxy = R_factory(π/3*v12)
Ryz = R_factory(π/3*v23)
Rxz = R_factory(π/3*v13)
```
Then you can do things like
```Julia
julia> R = R_factory(π/6*(v23+v12)) # this returns a function
#7 (generic function with 1 method)

julia> R(a) # which acts on a vector
0.0 + 0.5229556000177233v₁ + 0.7381444851051178v₂ + 1.4770443999822769v₃ + 2.7755575615628914e-17v₁₂₃

julia> Rxy(Ryz(Rxz(a)))
0.0 + 0.40849364905389035v₁ - 0.6584936490538903v₂ + 1.5490381056766584v₃
```
To make cascading a sequence of rotations as concise as possible, we could define a function which takes a list of bivectors ``A,B,C,...``, and enacts the sequence of rotations which they represent on some vector ``x``.
```Julia
julia> R_seq(args...) = (R = prod(exp.(args./2)); a -> ~R*a*R)
R_seq (generic function with 1 method)

julia> R = R_seq(π/2*v23, π/2*v12, v1)
#11 (generic function with 1 method)

julia> R(v1)
2.220446049250313e-16 + 3.469446951953614e-16v₁ + 0.9999999999999996v₂ - 1.3877787807814457e-17v₃ - 5.551115123125783e-17v₂₃ + 2.7755575615628914e-17v₁₂₃
```
