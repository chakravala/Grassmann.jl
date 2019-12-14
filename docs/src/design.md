# `TensorAlgebra` design, `Manifold` code generation

Mathematical foundations and definitions specific to the [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) implementation provide an extensible platform for computing with geometric algebra at high dimensions, along with the accompanying support packages. 
The design is based on the `TensorAlgebra` abstract type interoperability from [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl) with a `VectorBundle` parameter from [DirectSum.jl](https://github.com/chakravala/DirectSum.jl).
Abstract tangent vector space type operations happen at compile-time, resulting in a differential conformal geometric algebra of hyper-dual multivector forms.

The nature of the geometric algebra code generation enables one to easily extend the abstract product operations to any specific number field type (including differential operators with [Leibniz.jl](https://github.com/chakravala/Leibniz.jl) or symbolic coefficients with [Reduce.jl](https://github.com/chakravala/Reduce.jl)), by making use of Julia's type system. Mixed tensor products with their coefficients are constructed from these operations to work with bivector elements of Lie groups.

* **DirectSum.jl**: Abstract tangent bundle vector space types (unions, intersections, sums, etc.)
* **AbstractTensors.jl**: Tensor algebra abstract type interoperability with vector bundle parameter
* **Grassmann.jl**: ⟨Leibniz-Grassmann-Clifford-Hestenes⟩ differential geometric algebra of multivector forms
* **Leibniz.jl**: Derivation operator algebras for tensor fields
* **Reduce.jl**: Symbolic parser generator for Julia expressions using REDUCE algebra term rewriter

Mathematics of `Grassmann` can be used to study unitary groups used in quantum computing by building efficient computational representations of their algebras.
Applicability of the Grassmann computational package not only maps to quantum computing, but has the potential of impacting countless other engineering and scientific computing applications.
It can be used to work with automatic differentiation and differential geometry, algebraic forms and invariant theory, electric circuits and wave scattering, spacetime geometry and relativity, computer graphics and photogrammetry, and much more.

Thus, computations involving fully general rotational algebras and Lie bivector groups are possible with a full trigonometric suite.
Conformal geometric algebra is possible with the Minkowski plane ``v_{\infty\emptyset}``, based on the null-basis.
In general, multivalued quantum logic is enabled by the ``\wedge,\vee,\star`` Grassmann lattice.
Mixed-symmetry algebra with *Leibniz.jl* and *Grassmann.jl*, having the geometric algebraic product chain rule, yields automatic differentiation and Hodge-DeRahm co/homology  as unveiled by Grassmann.
Most importantly, the Dirac-Clifford product yields generalized Hodge-Laplacian and the Betti numbers with Euler characteristic `χ`.

Due to the abstract generality of the product algebra code generation, it is possible to extend the `Grassmann` library to include additional high performance products with few extra definitions.
Operations on ultra-sparse representations for very high dimensional algebras will be gaining further performance enhancements in future updates, along with hybrid optimizations for low-dimensional algebra code generation.
Thanks to the design of the product algebra code generation, any additional optimizations to the type stability will automatically enhance all the different products simultaneously.
Likewise, any new product formulas will be able to quickly gain from the setup of all of the existing optimizations.

The *Grassmann.jl* package and its accompanying support packages provide an extensible platform for high performance computing with geometric algebra at high dimensions.
This enables the usage of many different types of `TensorAlgebra` along with various `VectorBundle` parameters and interoperability for a wide range of scientific and research applications.

## Direct-sum yields `VectorBundle` parametric type polymorphism ⨁

The *DirectSum.jl* package is a work in progress providing the necessary tools to work with an arbitrary `Manifold` specified by an encoding.
Due to the parametric type system for the generating `VectorBundle`, the Julia compiler can fully preallocate and often cache values efficiently ahead of run-time.
Although intended for use with the *Grassmann.jl* package, `DirectSum` can be used independently.

Let `N` be the rank of a `Manifold{N}`.
The type `VectorBundle{N,P,g,ν,μ}` uses *byte-encoded* data available at pre-compilation, where
`P` specifies the basis for up and down projection,
`g` is a bilinear form that specifies the metric of the space,
and `μ` is an integer specifying the order of the tangent bundle (i.e. multiplicity limit of Leibniz-Taylor monomials). Lastly, `ν` is the number of tangent variables.

The metric signature of the `Basis{V,1}` elements of a vector space `V` can be specified with the `V"..."` constructor by using `+` and `-` to specify whether the `Basis{V,1}` element of the corresponding index squares to `+1` or `-1`.
For example, `S"+++"` constructs a positive definite 3-dimensional `VectorBundle`.
```Julia
julia> ℝ^3 == V"+++" == vectorspace(3)
true
```
It is also possible to specify an arbitrary `DiagonalForm` having numerical values for the basis with degeneracy `D"1,1,1,0"`, although the `Signature` format has a more compact representation.
Further development will result in more metric types.

Declaring an additional plane at infinity is done by specifying it in the string constructor with `∞` at the first index (i.e. Riemann sphere `S"∞+++"`). The hyperbolic geometry can be declared by `∅` subsequently (i.e. Minkowski spacetime `S"∅+++"`).
Additionally, the *null-basis* based on the projective split for confromal geometric algebra would be specified with `∞∅` initially (i.e. 5D CGA `S"∞∅+++"`). These two declared basis elements are interpreted in the type system.

The `tangent` map takes `V` to its tangent space and can be applied repeatedly for higher orders, such that `tangent(V,μ,ν)` can be used to specify `μ` and `ν`.
```Julia
julia> V = tangent(ℝ^3)
⟨+++₁⟩

julia> V'
⟨---¹⟩'

julia> V+V'
⟨+++---₁¹⟩*
```
The direct sum operator `⊕` can be used to join spaces (alternatively `+`), and the dual space functor `'` is an involution which toggles a dual vector space with inverted signature.
```Julia
julia> V = ℝ'⊕ℝ^3
⟨-+++⟩

julia> V'
⟨+---⟩'

julia> W = V⊕V'
⟨-++++---⟩*
```
The direct sum of a `VectorBundle` and its dual `V⊕V'` represents the full mother space `V*`.
```Julia
julia> collect(V) # all vector basis elements
Grassmann.Algebra{⟨-+++⟩,16}(v, v₁, v₂, v₃, v₄, v₁₂, v₁₃, v₁₄, v₂₃, v₂₄, v₃₄, v₁₂₃, v₁₂₄, v₁₃₄, ...)

julia> collect(V') # all covector basis elements
Grassmann.Algebra{⟨+---⟩',16}(w, w¹, w², w³, w⁴, w¹², w¹³, w¹⁴, w²³, w²⁴, w³⁴, w¹²³, w¹²⁴, w¹³⁴, ...)

julia> collect(W) # all mixed basis elements
Grassmann.Algebra{⟨-++++---⟩*,256}(v, v₁, v₂, v₃, v₄, w¹, w², w³, w⁴, v₁₂, v₁₃, v₁₄, v₁w¹, v₁w², ...
```
In addition to the direct-sum operation, several other operations are supported, such as `∪,∩,⊆,⊇` for set operations.
Due to the design of the `VectorBundle` dispatch, these operations enable code optimizations at compile-time provided by the bit parameters.
```Julia
julia> ℝ+ℝ' ⊇ vectorspace(1)
true

julia> ℝ ∩ ℝ' == vectorspace(0)
true

julia> ℝ ∪ ℝ' == ℝ+ℝ'
true
```
**Remark**. Although some of the operations like `∪` and `⊕` are similar and sometimes result in the same values, the `union` and `sum` are entirely different operations in general.

Calling manifolds with sets of indices constructs the subspace representations.
Given `M(s::Int...)` one can encode `SubManifold{length(s),M,s}` with induced orthogonal space, such that computing unions of submanifolds is done by inspecting the parameter `s`.
Operations on `Manifold` types is automatically handled at compile time.

More information about `DirectSum` is available  at https://github.com/chakravala/DirectSum.jl

## Interoperability for `TensorAlgebra{V}`

The `AbstractTensors` package is intended for universal interoperability of the abstract `TensorAlgebra` type system.
All `TensorAlgebra{V}` subtypes have type parameter `V`, used to store a `VectorBundle` value obtained from *DirectSum.jl*.
By itself, this package does not impose any specifications or structure on the `TensorAlgebra{V}` subtypes and elements, aside from requiring `V` to be a `VectorBundle`.
This means that different packages can create tensor types having a common underlying `VectorBundle` structure.

The key to making the whole interoperability work is that each `TensorAlgebra` subtype shares a `VectorBundle` parameter (with all `isbitstype` parameters), which contains all the info needed at compile time to make decisions about conversions. So other packages need only use the vector space information to decide on how to convert based on the implementation of a type. If external methods are needed, they can be loaded by `Requires` when making a separate package with `TensorAlgebra` interoperability.

Since `VectorBundle` choices are fundamental to `TensorAlgebra` operations, the universal interoperability between `TensorAlgebra{V}` elements with different associated `VectorBundle` choices is naturally realized by applying the `union` morphism to operations.
Some of the method names like `+,-,\otimes,\times,\cdot,*` for `TensorAlgebra` elements are shared across different packages, with interoperability.

Additionally, a universal unit volume element can be specified in terms of `LinearAlgebra.UniformScaling`, which is independent of `V` and has its interpretation only instantiated by the context of the `TensorAlgebra{V}` element being operated on.
The universal interoperability of `LinearAlgebra.UniformScaling` as a pseudoscalar element which takes on the `VectorBundle` form of any other `TensorAlgebra` element is handled globally.
This enables the usage of `I` from `LinearAlgebra` as a universal pseudoscalar element.
