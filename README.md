<p align="center">
  <img src="./docs/src/assets/logo.png" alt="Grassmann.jl"/>
</p>

# Grassmann.jl

*⟨Grassmann-Clifford-Hestenes-Taylor⟩ differential geometric algebra of hyper-dual multivector forms*

[![Build Status](https://travis-ci.org/chakravala/Grassmann.jl.svg?branch=master)](https://travis-ci.org/chakravala/Grassmann.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/c36u0rgtm2rjcquk?svg=true)](https://ci.appveyor.com/project/chakravala/grassmann-jl)
[![Coverage Status](https://coveralls.io/repos/chakravala/Grassmann.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chakravala/Grassmann.jl?branch=master)
[![codecov.io](http://codecov.io/github/chakravala/Grassmann.jl/coverage.svg?branch=master)](http://codecov.io/github/chakravala/Grassmann.jl?branch=master)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Liberapay patrons](https://img.shields.io/liberapay/patrons/chakravala.svg)](https://liberapay.com/chakravala)

The [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) package provides tools for doing computations based on multi-linear algebra, differential geometry, and spin groups using the extended tensor algebra known as Grassmann-Clifford-Hestenes-Taylor geometric algebra.
The primary operations are `∧, ∨, ⋅, *, ×, ⋆, ', ~` (which are the outer, regressive, inner, geometric, and cross products along with the Hodge star, adjoint, and multivector reversal operations).
Any operations are highly extensible with high dimensional support for up to 62 indices and staged caching / precompilation, where the code generation enables the fairly automated task of making more definitions and computational effort depends on the sparsity.
The [DirectSum.jl](https://github.com/chakravala/DirectSum.jl) multivector parametric type polymorphism is based on tangent bundle vector spaces and conformal projective geometry to make the dispatch highly extensible for many applications.
Additionally, interoperability between different sub-algebras is enabled by [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl), on which the type system is built.

  * [Design, code generation](#design-code-generation)
	 * [Requirements](#requirements)
  * [Direct-sum yields VectorSpace parametric type polymorphism ⨁](#direct-sum-yields-vectorspace-parametric-type-polymorphism-)
	 * [Tangent bundle](#tangent-bundle)
	 * [Interoperability for TensorAlgebra{V}](#interoperability-for-tensoralgebrav)
  * [Generating elements and geometric algebra Λ(V)](#generating-elements-and-geometric-algebra-λv)
	 * [Approaching ∞ dimensions with SparseAlgebra and ExtendedAlgebra](#approaching--dimensions-with-sparsealgebra-and-extendedalgebra)
	 * [Null-basis of the conformal split](#null-basis-of-the-conformal-split)
	 * [Differential forms and Taylor's tangent algebra](#differential-forms-and-taylors-tangent-algebra)
  * [Constructing linear transformations from mixed tensor product ⊗](#constructing-linear-transformations-from-mixed-tensor-product-)
  * [Importing the generators of the Leech lattice](#importing-the-generators-of-the-leech-lattice)
  * [Symbolic coefficients by declaring an alternative scalar algebra](#symbolic-coefficients-by-declaring-an-alternative-scalar-algebra)
  * [References](#references)

#### Design, code generation

The design of [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) is based on the `TensorAlgebra` abstract type system interoperability from [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl) with a `VectorSpace` parameter from [DirectSum.jl](https://github.com/chakravala/DirectSum.jl).
Abstract tangent vector space type operations happen at compile-time, resulting in a differential conformal geometric algebra of hyper-dual multivector forms.

The abstract nature of the product algebra code generation enables one to automate the extension of the product operations to any specific number field type (including symbolic coefficients with [Reduce.jl](https://github.com/chakravala/Reduce.jl) or SymPy.jl), by taking advantage of Julia's type system.
With the type system, it is possible to construct mixed tensor products from the mixed tangent vector basis and its dual basis, such as bivector elements of Lie groups.
`Grassmann` can be used to study unitary groups used in quantum computing by building efficient computational representations of their algebras.
Applicability of the Grassmann computational package not only maps to quantum computing, but has the potential of impacting countless other engineering and scientific computing applications.
It can be used to work with automatic differentiation and differential geometry, algebraic forms and invariant theory, electric circuits and wave scattering, spacetime geometry and relativity, computer graphics and photogrammetry, and much more.

Due to the abstract generality of the product algebra code generation, it is possible to extend the `Grassmann` library to include additional high performance products with few extra definitions.
Operations on ultra-sparse representations for very high dimensional algebras will be gaining further performance enhancements in future updates, while the standard lower dimensional algebras already are highly performant and optimized.
Thanks to the design of the product algebra code generation, any additional optimizations to the type stability will automatically enhance all the different products simultaneously.
Likewise, any new product formulas will be able to quickly gain from the setup of all of the existing optimizations.

The *Grassmann.jl* package and its accompanying support packages provide an extensible platform for high performance computing with geometric algebra at high dimensions.
This enables the usage of many different types of `TensorAlgebra` along with various `VectorSpace` parameters and interoperability for a wide range of scientific and research applications.

### Requirements

Availability of this package and its subpackages can be automatically handled with the Julia package manager; however, when the `master` branch is used it is possible that some of the dependencies also require a development branch before the release. This may include (but is not limited to) the following packages:

This requires a merged version of `ComputedFieldTypes` at https://github.com/vtjnash/ComputedFieldTypes.jl

Interoperability of `TensorAlgebra` with other packages is automatically enabled by [DirectSum.jl](https://github.com/chakravala/DirectSum.jl) and [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl).

## Direct-sum yields `VectorSpace` parametric type polymorphism ⨁

The *DirectSum.jl* package is a work in progress providing the necessary tools to work with arbitrary dual `VectorSpace` elements with optional origin.
Due to the parametric type system for the generating `VectorSpace`, the Julia compiler can fully preallocate and often cache values efficiently ahead of run-time.
Although intended for use with the *Grassmann.jl* package, `DirectSum` can be used independently.

Let `N` be the dimension of a `VectorSpace{N}`.
The metric signature of the `Basis{V,1}` elements of a vector space `V` can be specified with the `V"..."` constructor by using `+` and `-` to specify whether the `Basis{V,1}` element of the corresponding index squares to `+1` or `-1`.
E.g., `V"+++"` constructs a positive definite 3-dimensional `VectorSpace`.
```Julia
julia> ℝ^3 == V"+++" == vectorspace(3)
true
```
The direct sum operator `⊕` can be used to join spaces (alternatively `+`), and `'` is an involution which toggles a dual vector space with inverted signature.
```Julia
julia> V = ℝ'⊕ℝ^3
⟨-+++⟩

julia> V'
⟨+---⟩'

julia> W = V⊕V'
⟨-++++---⟩*
```
The direct sum of a `VectorSpace` and its dual `V⊕V'` represents the full mother space `V*`.
```Julia
julia> collect(V) # all vector basis elements
Grassmann.Algebra{⟨-+++⟩,16}(v, v₁, v₂, v₃, v₄, v₁₂, v₁₃, v₁₄, v₂₃, v₂₄, v₃₄, v₁₂₃, v₁₂₄, v₁₃₄, ...)

julia> collect(V') # all covector basis elements
Grassmann.Algebra{⟨+---⟩',16}(w, w¹, w², w³, w⁴, w¹², w¹³, w¹⁴, w²³, w²⁴, w³⁴, w¹²³, w¹²⁴, w¹³⁴, ...)

julia> collect(W) # all mixed basis elements
Grassmann.Algebra{⟨-++++---⟩*,256}(v, v₁, v₂, v₃, v₄, w¹, w², w³, w⁴, v₁₂, v₁₃, v₁₄, v₁w¹, v₁w², ...
```
Compile-time type operations make code optimization easier.
Additionally to the direct-sum operation, several others operations are supported, such as `∪,∩,⊆,⊇` for set operations.
```Julia
julia> ℝ+ℝ' ⊇ vectorspace(1)
true

julia> ℝ ∩ ℝ' == vectorspace(0)
true

julia> ℝ ∪ ℝ' == ℝ+ℝ'
true
```
**Note**. Although some of the operations sometimes result in the same value as shown in the above examples, the `∪` and `+` are entirely different operations in general.

It is possible to specify an arbitrary `DiagonalForm` for the metric of the basis elements with `V"1,2,3"` or `V"-1,1,1,1"`, for example.
Additionally, special projective geometry elements can be specified, such as `∞` at the first index, i.e. `V"∞+++"` or the *origin* by using `∅` subsequently (`V"∅+++"` or `V"∞∅+++"`). The projective geometry elements are still experimental on this release.

### Tangent bundle

The `tangent` map takes `V` to its tangent space and can be applied repeatedly or specified `tangent(V,order)` for higher.

```Julia
julia> V = tangent(ℝ^3)
⟨+++₁⟩

julia> V'
⟨---¹⟩'

julia> V+V'
⟨+++---₁¹⟩*
```

More information about `DirectSum` is available  at https://github.com/chakravala/DirectSum.jl

### Interoperability for `TensorAlgebra{V}`

*AbstractTensors.jl* provides the abstract interoperability between tensor algebras having differing `VectorSpace` parameters. The great thing about it is that the `VectorSpace` unions and intersections are handled separately in a different package and the actual tensor implementations are handled separately also. This enables anyone who wishes to be interoperable with `TensorAlgebra` to build their own subtypes in their own separate package with interoperability automatically possible between it all, provided the guidelines are followed.

The key to making the whole interoperability work is that each `TensorAlgebra` subtype shares a `VectorSpace` parameter (with all `isbitstype` parameters), which contains all the info needed at compile time to make decisions about conversions. So other packages need only use the vector space information to decide on how to convert based on the implementation of a type. If external methods are needed, they can be loaded by `Requires` when making a separate package with `TensorAlgebra` interoperability.

Since `VectorSpace` choices are fundamental to `TensorAlgebra` operations, the universal interoperability between `TensorAlgebra{V}` elements with different associated `VectorSpace` choices is naturally realized by applying the `union` morphism to operations.
Some of the method names like `+,-,\otimes,\times,\cdot,*` for `TensorAlgebra` elements are shared across different packages, with interoperability.

Additionally, a universal unit volume element can be specified in terms of `LinearAlgebra.UniformScaling`, which is independent of `V` and has its interpretation only instantiated by the context of the `TensorAlgebra{V}` element being operated on.
The universal interoperability of `LinearAlgebra.UniformScaling` as a pseudoscalar element which takes on the `VectorSpace` form of any other `TensorAlgebra` element is handled globally.
This enables the usage of `I` from `LinearAlgebra` as a universal pseudoscalar element.

By importing the `AbstractTensors` module, the *Reduce.jl* package is able to correctly bypass operations on `TensorAlgebra` elements to the correct methods within the scope of the `Reduce.Algebra` module.
This requires no additional overhead for the `Grassmann` or `Reduce` packages, because the `AbstractTensors` interoperability interface enables separate precompilation of both.
Additionally, the `VectorSpace` interoperability also enables more arbitrary inputs.

## Generating elements and geometric algebra Λ(V)

By virtue of Julia's multiple dispatch on the field type `T`, methods can specialize on the `Dimension{N}` and `Grade{G}` and `VectorSpace{N,D,O}` via the `TensorAlgebra` subtypes, such as `Basis{V,G}`, `SValue{V,G,B,T}`, `MValue{V,G,B,T}`, `SBlade{T,V,G}`, `MBlade{T,V,G}`, `MultiVector{T,V}`, and `MultiGrade{V}` types.

The elements of the `Algebra` can be generated in many ways using the `Basis` elements created by the `@basis` macro,
```Julia
julia> using Grassmann; @basis ℝ'⊕ℝ^3 # equivalent to basis"-+++"
(⟨-+++⟩, v, v₁, v₂, v₃, v₄, v₁₂, v₁₃, v₁₄, v₂₃, v₂₄, v₃₄, v₁₂₃, v₁₂₄, v₁₃₄, v₂₃₄, v₁₂₃₄)
```
As a result of this macro, all of the `Basis{V,G}` elements generated by that `VectorSpace` become available in the local workspace with the specified naming.
The first argument provides signature specifications, the second argument is the variable name for the `VectorSpace`, and the third and fourth argument are the the prefixes of the `Basis` vector names (and covector basis names). By default, `V` is assigned the `VectorSpace` and `v` is the prefix for the `Basis` elements.
```Julia
julia> V # Minkowski spacetime
⟨-+++⟩

julia> typeof(V) # dispatch by vector space
VectorSpace{4,0,0x0000000000000001}

julia> typeof(v13) # extensive type info
Basis{⟨-+++⟩,2,0x0000000000000005}

julia> v13∧v2 # exterior tensor product
-1v₁₂₃

julia> ans^2 # applies geometric product
1v

julia> @btime h = 2v1+v3 # vector element
  37.794 ns (3 allocations: 80 bytes)
2v₁ + 0v₂ + 1v₃ + 0v₄

julia> @btime $h⋅$h # inner product
  105.948 ns (2 allocations: 160 bytes)
-3v
```
It is entirely possible to assign multiple different bases with different signatures without any problems. In the following command, the `@basis` macro arguments are used to assign the vector space name to `S` instead of `V` and basis elements to `b` instead of `v`, so that their local names do not interfere:
```Julia
julia> @basis "++++" S b;

julia> let k = (b1+b2)-b3
           for j ∈ 1:9
               k = k*(b234+b134)
               println(k)
       end end
0 + 1v₁₄ + 1v₂₄ + 2v₃₄
0 - 2v₁ - 2v₂ + 2v₃
0 - 2v₁₄ - 2v₂₄ - 4v₃₄
0 + 4v₁ + 4v₂ - 4v₃
0 + 4v₁₄ + 4v₂₄ + 8v₃₄
0 - 8v₁ - 8v₂ + 8v₃
0 - 8v₁₄ - 8v₂₄ - 16v₃₄
0 + 16v₁ + 16v₂ - 16v₃
0 + 16v₁₄ + 16v₂₄ + 32v₃₄
```
Alternatively, if you do not wish to assign these variables to your local workspace, the versatile `Grassmann.Algebra{N}` constructors can be used to contain them, which is exported to the user as the method `Λ(V)`,
```Julia
julia> G3 = Λ(3) # equivalent to Λ(V"+++"), Λ(ℝ^3), Λ.V3
Grassmann.Algebra{⟨+++⟩,8}(v, v₁, v₂, v₃, v₁₂, v₁₃, v₂₃, v₁₂₃)

julia> G3.v13 * G3.v12
v₂₃
```
It is possible to assign the **quaternion** generators `i,j,k` with
```Julia
julia> i,j,k = hyperplanes(ℝ^3)
3-element Array{SValue{⟨+++⟩,2,B,Int64} where B,1}:
 -1v₂₃
 1v₁₃
 -1v₁₂

julia> @btime i^2, j^2, k^2, i*j*k
  158.925 ns (5 allocations: 112 bytes)
(-1v, -1v, -1v, -1v)

julia> @btime -(j+k) * (j+k)
  176.233 ns (8 allocations: 240 bytes)
2

julia> @btime -(j+k) * i
  111.394 ns (6 allocations: 192 bytes)
0 - 1v₁₂ - 1v₁₃
```
Alternatively, another representation of the quaternions is
```Julia
julia> basis"--"
(⟨--⟩, v, v₁, v₂, v₁₂)

julia> v1^2, v2^2, v12^2, v1*v2*v12
(-1v, -1v, -1v, -1v)
```
The parametric type formalism in `Grassmann` is highly expressive to enable the pre-allocation of geometric algebra computations for specific sparse-subalgebras, including the representation of rotational groups, Lie bivector algebras, and affine projective geometry.

### Approaching ∞ dimensions with `SparseAlgebra` and `ExtendedAlgebra`

It is possible to reach `Basis` elements up to `N=62` indices with `TensorAlgebra` having higher maximum dimensions than supported by Julia natively.
```Julia
julia> Λ(62)
Grassmann.ExtendedAlgebra{⟨++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++⟩,4611686018427387904}(v, ..., v₁₂₃₄₅₆₇₈₉₀abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ)

julia> Λ(62).v32a87Ng
-1v₂₃₇₈agN
```
The 62 indices require full alpha-numeric labeling with lower-case and capital letters. This now allows you to reach up to `4,611,686,018,427,387,904` dimensions with Julia `using Grassmann`. Then the volume element is
```Julia
v₁₂₃₄₅₆₇₈₉₀abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
```
However, Julia is only able to allocate full `MultiVector` for `N≤22`, with sparse operations only available at higher dimension.

While `Grassmann.Algebra{V}` is a container for the `TensorAlgebra` generators of `V`, the `Grassmann.Algebra` is only cached for `N≤8`.
For a `VectorSpace{N}` of dimension `8<N≤22`, the `Grassmann.SparseAlgebra` type is used.

```Julia
julia> Λ(22)
Grassmann.SparseAlgebra{⟨++++++++++++++++++++++⟩,4194304}(v, ..., v₁₂₃₄₅₆₇₈₉₀abcdefghijkl)
```
This is the largest `SparseAlgebra` that can be generated with Julia, due to array size limitations.

To reach higher dimensions, for `N>22` the `Grassmann.ExtendedAlgebra` type is used.
It is suficient to work with a 64-bit representation (which is the default). And it turns out that with 62 standard keyboard characters, this fits nicely. Since 22 is the limit for the largest fully representable `MultiVector` with Julia, having a 64-bit representation still lets you extend to 44 generating `Basis` elements if you suddenly want to decide to have a dual vector space also.
```Julia
julia> V = ℝ^22
⟨++++++++++++++++++++++⟩

julia> Λ(V+V')
Grassmann.ExtendedAlgebra{⟨++++++++++++++++++++++----------------------⟩*,17592186044416}(v, ..., v₁₂₃₄₅₆₇₈₉₀abcdefghijklw¹²³⁴⁵⁶⁷⁸⁹⁰ABCDEFGHIJKL)
```
Currently, up to `N=62` is supported with alpha-numeric indexing. This is due to the defaults of the bit depth from the computer, so if you are 32-bit it is more limited.

At 22 dimensions and lower, you have better caching, and 8 dimensions or less have extra caching.
Thus, the largest Hilbert space that is fully reachable has 4,194,304 dimensions, but we can still reach out to 4,611,686,018,427,387,904 dimensions with the `ExtendedAlgebra` built in.
This is approximately `2^117` times smaller than the order of the Monster group. It is still feasible to extend to a further super-extended 128-bit representation using the `UInt128` type (but this will require further modifications of internals and helper functions.
To reach into infinity even further, it is theoretically possible to construct ultra-extensions also using dictionaries.
Full `MultiVector` elements are not representable when `ExtendedAlgebra` is used, but the performance of the `Basis` and sparse elements should be just as fast as for lower dimensions for the current `SubAlgebra` and `TensorAlgebra` types.
The sparse representations are a work in progress to be improved with time.

In order to work with a `TensorAlgebra{V}`, it is necessary for some computations to be cached. This is usually done automatically when accessed.
```Julia
julia> Λ(7) + Λ(7)'
Grassmann.SparseAlgebra{⟨+++++++-------⟩*,16384}(v, ..., v₁₂₃₄₅₆₇w¹²³⁴⁵⁶⁷)
```
One way of declaring the cache for all 3 combinations of a `VectorSpace{N}` and its dual is to ask for the sum `Λ(V) + Λ(V)'`, which is equivalent to `Λ(V⊕V')`, but this does not initialize the cache of all 3 combinations unlike the former.

The staging of the precompilation and caching is designed so that a user can smoothly transition between very high dimensional and low dimensional algebras in a single session, with varying levels of extra caching and optimizations.

### Null-basis of the conformal split

Declaring an additional *null-basis* is done by specifying it in the string constructor with `∞` at the first index (i.e. `S"∞+++"`).
Likewise, an optional *origin* can be declared by `∅` subsequently (i.e. `S"∅+++"` or `S"∞∅+++"`).
These two basis elements will be interpreted in the type system such that they propagate under transformations when combining a mixed index sets (provided the `Signature` is compatible).

```Julia
julia> using Grassmann; @basis S"∞∅++"
(⟨∞∅++⟩, v, v∞, v∅, v₁, v₂, v∞∅, v∞₁, v∞₂, v∅₁, v∅₂, v₁₂, v∞∅₁, v∞∅₂, v∞₁₂, v∅₁₂, v∞∅₁₂)

julia> v∞^2, v∅^2, v1^2, v2^2
(0v, 0v, v, v)

julia> v∞ ⋅ v∅
-1v

julia> v∞∅^2
v

julia> v∞∅ * v∞, v∞∅ * v∅
(-1v∞, v∅)

julia> v∞ * v∅, v∅ * v∞
(-1 + 1v∞∅, -1 - 1v∞∅)
```
The index number `N` of the `VectorSpace` corresponds to the total number of generator elements. However, even though `S"∞∅+++"` is of type `VectorSpace{5,3}` with `5` generator elements, it can be internally recognized in the direct sum algebra as being an embedding of a 3-index `VectorSpace{3,0}` with additional encoding of the null-basis (origin and point at infinity) in the parameter `M` of the `VectorSpace{N,M}` type.

### Differential forms and Taylor's tangent algebra

The chain rule is encoded into `Grassmann` algebra when a `tangent` bundle is used, demonstrated here symbolically with `Reduce` by using the dual number definition:
```Julia
julia> using Grassmann, Reduce
Reduce (Free CSL version, revision 4590), 11-May-18 ...

julia> @mixedbasis tangent(ℝ^1)
(⟨+-₁¹⟩*, v, v₁, w¹, ϵ₁, ∂¹, v₁w¹, v₁ϵ₁, v₁∂¹, w¹ϵ₁, w¹∂¹, ϵ₁∂¹, v₁w¹ϵ₁, v₁w¹∂¹, v₁ϵ₁∂¹, w¹ϵ₁∂¹, v₁w¹ϵ₁∂¹)

julia> a,b = :x*v1 + :dx*ϵ1, :y*v1 + :dy*ϵ1
(xv₁ + dxϵ₁, yv₁ + dyϵ₁)

julia> a * b
x * y + (dy * x - dx * y)v₁ϵ₁
```
Higher order and multivariable Taylor numbers are also supported, although the implementation in this release is still experimental.

## Constructing linear transformations from mixed tensor product ⊗

Groups such as SU(n) can be represented with the dual Grassmann’s exterior product algebra, generating a `2^(2n)`-dimensional mother algebra with geometric product from the `n`-dimensional vector space and its dual vector space. The product of the vector basis and covector basis elements form the `n^2`-dimensional bivector subspace of the full `(2n)!/(2(2n−2)!)`-dimensional bivector sub-algebra.
The package `Grassmann` is working towards making the full extent of this number system available in Julia by using static compiled parametric type information to handle sparse sub-algebras, such as the (1,1)-tensor bivector algebra.

Note that `Λ(3)` gives the vector basis, and `Λ(3)'` gives the covector basis:
```Julia
julia> Λ(3)
Grassmann.Algebra{⟨+++⟩,8}(v, v₁, v₂, v₃, v₁₂, v₁₃, v₂₃, v₁₂₃)

julia> Λ(3)'
Grassmann.Algebra{⟨---⟩',8}(w, w¹, w², w³, w¹², w¹³, w²³, w¹²³)
```
The following command yields a local 2D vector and covector basis,
```Julia
julia> mixedbasis"2"
(⟨++--⟩*, v, v₁, v₂, w¹, w², v₁₂, v₁w¹, v₁w², v₂w¹, v₂w², w¹², v₁₂w¹, v₁₂w², v₁w¹², v₂w¹², v₁₂w¹²)

julia> w1+2w2
1w¹ + 2w²

julia> ans(v1+v2)
3v
```
The sum `w1+2w2` is interpreted as a covector element of the dual vector space, which can be evaluated as a linear functional when a vector argument is input.
Using these in the workspace, it is possible to use the Grassmann exterior `∧`-tensor product operation to construct elements `ℒ` of the (1,1)-bivector subspace of linear transformations from the `Grade{2}` algebra.
```Julia
julia> ℒ = (v1+2v2)∧(3w1+4w2)
0v₁₂ + 3v₁w¹ + 4v₁w² + 6v₂w¹ + 8v₂w² + 0w¹²
```
The element `ℒ` is a linear form which can take `Grade{1}` vectors as input,
```Julia
julia> ℒ(v1+v2)
7v₁ + 14v₂ + 0w¹ + 0w²

julia> L = [1,2] * [3,4]'; L * [1,1]
2-element Array{Int64,1}:
  7
 14
```
which is a computation equivalent to a matrix computation.

The `TensorAlgebra` evalution is still a work in progress, and the API and implementation may change as more features and algebraic operations and product structure are added.

## Importing the generators of the Leech lattice

In the example below, we define a constant `Leech` which can be used to obtain linear combinations of the Leech lattice,
```Julia
julia> using Grassmann

julia> generator = [8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0;
       2 2 2 2 0 0 0 0 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0;
       4 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0;
       2 2 0 0 2 2 0 0 2 2 0 0 2 2 0 0 0 0 0 0 0 0 0 0;
       2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 0 0 0 0 0 0 0 0;
       2 0 0 2 2 0 0 2 2 0 0 2 2 0 0 2 0 0 0 0 0 0 0 0;
       4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0;
       2 0 2 0 2 0 0 2 2 2 0 0 0 0 0 0 2 2 0 0 0 0 0 0;
       2 0 0 2 2 2 0 0 2 0 2 0 0 0 0 0 2 0 2 0 0 0 0 0;
       2 2 0 0 2 0 2 0 2 0 0 2 0 0 0 0 2 0 0 2 0 0 0 0;
       0 2 2 2 2 0 0 0 2 0 0 0 2 0 0 0 2 0 0 0 2 0 0 0;
       0 0 0 0 0 0 0 0 2 2 0 0 2 2 0 0 2 2 0 0 2 2 0 0;
       0 0 0 0 0 0 0 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0;
       -3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

julia> const E24,W24 = Λ(24), ℝ^24+(ℝ^24)';

julia> const Leech = SBlade{Float64,W24}(generator./sqrt(8));

julia> typeof(Leech)
SBlade{Float64,⟨++++++++++++++++++++++++------------------------⟩*,2,1128}

julia> ndims(vectorspace(Leech))
48
```
The `Leech` generator matrix is contained in the 1128-dimensional bivector subalgebra of the space with 48 indices.
```Julia
julia> Leech(E24.v1)
2.82842712474619v₁ + 0.0v₂ + 0.0v₃ + 0.0v₄ + 0.0v₅ + 0.0v₆ + 0.0v₇ + 0.0v₈ + 0.0v₉ + 0.0v₀ + 0.0va + 0.0vb + 0.0vc + 0.0vd + 0.0ve + 0.0vf + 0.0vg + 0.0vh + 0.0vi + 0.0vj + 0.0vk + 0.0vl + 0.0vm + 0.0vn + 0.0w¹ + 0.0w² + 0.0w³ + 0.0w⁴ + 0.0w⁵ + 0.0w⁶ + 0.0w⁷ + 0.0w⁸ + 0.0w⁹ + 0.0w⁰ + 0.0wA + 0.0wB + 0.0wC + 0.0wD + 0.0wE + 0.0wF + 0.0wG + 0.0wH + 0.0wI + 0.0wJ + 0.0wK + 0.0wL + 0.0wM + 0.0wN

julia> Leech(E24.v2)
1.414213562373095v₁ + 1.414213562373095v₂ + 0.0v₃ + 0.0v₄ + 0.0v₅ + 0.0v₆ + 0.0v₇ + 0.0v₈ + 0.0v₉ + 0.0v₀ + 0.0va + 0.0vb + 0.0vc + 0.0vd + 0.0ve + 0.0vf + 0.0vg + 0.0vh + 0.0vi + 0.0vj + 0.0vk + 0.0vl + 0.0vm + 0.0vn + 0.0w¹ + 0.0w² + 0.0w³ + 0.0w⁴ + 0.0w⁵ + 0.0w⁶ + 0.0w⁷ + 0.0w⁸ + 0.0w⁹ + 0.0w⁰ + 0.0wA + 0.0wB + 0.0wC + 0.0wD + 0.0wE + 0.0wF + 0.0wG + 0.0wH + 0.0wI + 0.0wJ + 0.0wK + 0.0wL + 0.0wM + 0.0wN

julia> Leech(E24.v3)
1.414213562373095v₁ + 0.0v₂ + 1.414213562373095v₃ + 0.0v₄ + 0.0v₅ + 0.0v₆ + 0.0v₇ + 0.0v₈ + 0.0v₉ + 0.0v₀ + 0.0va + 0.0vb + 0.0vc + 0.0vd + 0.0ve + 0.0vf + 0.0vg + 0.0vh + 0.0vi + 0.0vj + 0.0vk + 0.0vl + 0.0vm + 0.0vn + 0.0w¹ + 0.0w² + 0.0w³ + 0.0w⁴ + 0.0w⁵ + 0.0w⁶ + 0.0w⁷ + 0.0w⁸ + 0.0w⁹ + 0.0w⁰ + 0.0wA + 0.0wB + 0.0wC + 0.0wD + 0.0wE + 0.0wF + 0.0wG + 0.0wH + 0.0wI + 0.0wJ + 0.0wK + 0.0wL + 0.0wM + 0.0wN

...
```
Then a `TensorAlgebra` evaluation of `Leech` at an `Integer` linear combination would be
```Julia
julia> Leech(E24.v1 + 2*E24.v2)
5.65685424949238v₁ + 2.82842712474619v₂ + 0.0v₃ + 0.0v₄ + 0.0v₅ + 0.0v₆ + 0.0v₇ + 0.0v₈ + 0.0v₉ + 0.0v₀ + 0.0va + 0.0vb + 0.0vc + 0.0vd + 0.0ve + 0.0vf + 0.0vg + 0.0vh + 0.0vi + 0.0vj + 0.0vk + 0.0vl + 0.0vm + 0.0vn + 0.0w¹ + 0.0w² + 0.0w³ + 0.0w⁴ + 0.0w⁵ + 0.0w⁶ + 0.0w⁷ + 0.0w⁸ + 0.0w⁹ + 0.0w⁰ + 0.0wA + 0.0wB + 0.0wC + 0.0wD + 0.0wE + 0.0wF + 0.0wG + 0.0wH + 0.0wI + 0.0wJ + 0.0wK + 0.0wL + 0.0wM + 0.0wN

julia> ans⋅ans
39.99999999999999v

julia> Leech(E24.v2 + E24.v5)
2.82842712474619v₁ + 1.414213562373095v₂ + 0.0v₃ + 0.0v₄ + 0.0v₅ + 0.0v₆ + 0.0v₇ + 0.0v₈ + 0.0v₉ + 0.0v₀ + 1.414213562373095va + 0.0vb + 0.0vc + 0.0vd + 0.0ve + 0.0vf + 0.0vg + 0.0vh + 0.0vi + 0.0vj + 0.0vk + 0.0vl + 0.0vm + 0.0vn + 0.0w¹ + 0.7071067811865475w² + 1.414213562373095w³ + 1.414213562373095w⁴ + 0.0w⁵ + 0.0w⁶ + 0.0w⁷ + 0.0w⁸ + 0.0w⁹ + 0.0w⁰ + 0.0wA + 0.0wB + 0.0wC + 0.0wD + 0.0wE + 0.0wF + 0.0wG + 0.0wH + 0.0wI + 0.0wJ + 0.0wK + 0.0wL + 0.0wM + 0.0wN

julia> ans⋅ans
7.499999999999998v
```
The `Grassmann` package is designed to smoothly handle high-dimensional bivector algebras with headroom to spare. Although some of these calculations may have an initial delay, repeated calls are fast due to built-in caching and pre-compilation.

In future updates, more emphasis will be placed on increased type-stability with more robust sparse output allocation in the computational graph and minimal footprint but maximal type-stability for intermediate results and output.

## Symbolic coefficients by declaring an alternative scalar algebra

Due to the abstract generality of the code generation of the `Grassmann` product algebra, it is easily possible to extend the entire set of operations to other kinds of scalar coefficient types.
```Julia
julia> using GaloisFields,Grassmann

julia> const F = GaloisField(7)
𝔽₇

julia> basis"2"
(⟨++⟩, v, v₁, v₂, v₁₂)

julia> @btime F(3)*v1
  21.076 ns (2 allocations: 32 bytes)
3v₁

julia> @btime inv($ans)
  26.636 ns (0 allocations: 0 bytes)
5v₁
```
By default, the coefficients are required to be `<:Number`. However, if this does not suit your needs, alternative scalar product algebras can be specified with
```Julia
generate_product_algebra(SymField,:(Sym.:*),:(Sym.:+),:(Sym.:-),:svec)
```
where `SymField` is the desired scalar field and `Sym` is the scope which contains the scalar field algebra for `SymField`.

With the use of `Requires` it is feasible to automatically enable symbolic scalar computation with [Reduce.jl](https://github.com/chakravala/Reduce.jl), e.g.
```Julia
julia> using Reduce, Grassmann
Reduce (Free CSL version, revision 4590), 11-May-18 ...
```
Additionally, due to the interoperability of the `AbstractTensors` package, the `Reduce` package automatically bypasses mixed symbolic operations with `TensorAlgebra` elements within the `Reduce.Algebra` scope to the correct methods.
```Julia
julia> basis"2"
(⟨++⟩, v, v₁, v₂, v₁₂)

julia> (:a*v1 + :b*v2) ⋅ (:c*v1 + :d*v2)
(a * c + b * d)v

julia> (:a*v1 + :b*v2) ∧ (:c*v1 + :d*v2)
0.0 + (a * d - b * c)v₁₂

julia> (:a*v1 + :b*v2) * (:c*v1 + :d*v2)
a * c + b * d + (a * d - b * c)v₁₂
```
If these compatibility steps are followed, then `Grassmann` will automatically declare the product algebra to use the `Reduce.Algebra` symbolic field operation scope.

```Julia
julia> using Reduce,Grassmann; basis"4"
Reduce (Free CSL version, revision 4590), 11-May-18 ...
(⟨++++⟩, v, v₁, v₂, v₃, v₄, v₁₂, v₁₃, v₁₄, v₂₃, v₂₄, v₃₄, v₁₂₃, v₁₂₄, v₁₃₄, v₂₃₄, v₁₂₃₄)

julia> P,Q = :px*v1 + :py*v2 + :pz* v3 + v4, :qx*v1 + :qy*v2 + :qz*v3 + v4
(pxv₁ + pyv₂ + pzv₃ + 1.0v₄, qxv₁ + qyv₂ + qzv₃ + 1.0v₄)

julia> P∧Q
0.0 + (px * qy - py * qx)v₁₂ + (px * qz - pz * qx)v₁₃ + (px - qx)v₁₄ + (py * qz - pz * qy)v₂₃ + (py - qy)v₂₄ + (pz - qz)v₃₄

julia> R = :rx*v1 + :ry*v2 + :rz*v3 + v4
rxv₁ + ryv₂ + rzv₃ + 1.0v₄

julia> P∧Q∧R
0.0 + ((px * qy - py * qx) * rz - ((px * qz - pz * qx) * ry - (py * qz - pz * qy) * rx))v₁₂₃ + (((px * qy - py * qx) + (py - qy) * rx) - (px - qx) * ry)v₁₂₄ + (((px * qz - pz * qx) + (pz - qz) * rx) - (px - qx) * rz)v₁₃₄ + (((py * qz - pz * qy) + (pz - qz) * ry) - (py - qy) * rz)v₂₃₄
```

It should be straight-forward to easily substitute any other extended algebraic operations and fields; issues with questions or pull-requests to that end are welcome.

## References
* C. Doran, D. Hestenes, F. Sommen, and N. Van Acker, [Lie groups as spin groups](http://geocalc.clas.asu.edu/pdf/LGasSG.pdf), J. Math Phys. (1993)
* David Hestenes, [Universal Geometric Algebra](http://lomont.org/Math/GeometricAlgebra/Universal%20Geometric%20Algebra%20-%20Hestenes%20-%201988.pdf), Pure and Applied (1988)
* Peter Woit, [Clifford algebras and spin groups](http://www.math.columbia.edu/~woit/LieGroups-2012/cliffalgsandspingroups.pdf), Lecture Notes (2012)
