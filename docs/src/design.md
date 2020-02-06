# `TensorAlgebra` design, `Manifold` code generation

Mathematical foundations and definitions specific to the [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) implementation provide an extensible platform for computing with geometric algebra at high dimensions, along with the accompanying support packages. 
The design is based on the `TensorAlgebra` abstract type interoperability from [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl) with a `TensorBundle` parameter from [DirectSum.jl](https://github.com/chakravala/DirectSum.jl).
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
Most importantly, the Dirac-Clifford product yields generalized Hodge-Laplacian and the Betti numbers with Euler characteristic ``\chi``.

Due to the abstract generality of the product algebra code generation, it is possible to extend the `Grassmann` library to include additional high performance products with few extra definitions.
Operations on ultra-sparse representations for very high dimensional algebras will be gaining further performance enhancements in future updates, along with hybrid optimizations for low-dimensional algebra code generation.
Thanks to the design of the product algebra code generation, any additional optimizations to the type stability will automatically enhance all the different products simultaneously.
Likewise, any new product formulas will be able to quickly gain from the setup of all of the existing optimizations.

The *Grassmann.jl* package and its accompanying support packages provide an extensible platform for high performance computing with geometric algebra at high dimensions.
This enables the usage of many different types of `TensorAlgebra` along with various `TensorBundle` parameters and interoperability for a wide range of scientific and research applications.

## DirectSum yields `TensorBundle` parametric type polymorphism

[![DOI](https://zenodo.org/badge/169765288.svg)](https://zenodo.org/badge/latestdoi/169765288)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/chakravala/DirectSum.jl)](https://github.com/chakravala/DirectSum.jl/releases)
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/chakravala/DirectSum.jl/latest?label=new%20commits)](https://github.com/chakravala/DirectSum.jl/commits)
[![Build Status](https://travis-ci.org/chakravala/DirectSum.jl.svg?branch=master)](https://travis-ci.org/chakravala/DirectSum.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/ipaggdeq2f1509pl?svg=true)](https://ci.appveyor.com/project/chakravala/directsum-jl)

The *DirectSum.jl* package is a work in progress providing the necessary tools to work with an arbitrary `Manifold` specified by an encoding.
Due to the parametric type system for the generating `TensorBundle`, the Julia compiler can fully preallocate and often cache values efficiently ahead of run-time.
Although intended for use with the *Grassmann.jl* package, `DirectSum` can be used independently.

Let ``M = T^\mu V`` be a `TensorBundle{n}<:Manifold{n}` of rank ``n``,
```math
T^\mu V = (n,\mathbb P,g,\nu,\mu), \qquad \mathbb P \subseteq\langle v_\infty,v_\emptyset\rangle, \qquad g :V\times V\rightarrow\mathbb K
```
The type `TensorBundle{n,ℙ,g,ν,μ}` uses *byte-encoded* data available at pre-compilation, where
``\mathbb P`` specifies the basis for up and down projection,
``g`` is a bilinear form that specifies the metric of the space,
and ``\mu`` is an integer specifying the order of the tangent bundle (i.e. multiplicity limit of Leibniz-Taylor monomials). Lastly, ``\nu`` is the number of tangent variables.
The dual space functor ``'``
is an involution which toggles a dual vector space with inverted signature with property ``V' = \text{Hom}(V,\mathbb K)`` and having `SubManifold` generators
```math
\langle v_1,\dots,v_{n-\nu},\partial_1,\dots,\partial_\nu\rangle=M\leftrightarrow M' = \langle w_1,\dots,w_{n-\nu},\epsilon_1,\dots,\epsilon_\nu\rangle
```
where ``v_i,w_i`` are a basis for the vectors and covectors, while ``\partial_j,\epsilon_j`` are a basis for differential operators and tensor fields.

The metric signature of the `SubManifold{V,1}` elements of a vector space ``V`` can be specified with the `V"..."` constructor by using ``+`` and ``-`` to specify whether the `SubManifold{V,1}` element of the corresponding index squares to ``+1`` or ``-1``.
For example, `S"+++"` constructs a positive definite 3-dimensional `TensorBundle`.
```@setup ds
using DirectSum
```
```@repl ds
ℝ^3 == V"+++" == Manifold(3)
```
It is also possible to specify an arbitrary `DiagonalForm` having numerical values for the basis with degeneracy `D"1,1,1,0"`, although the `Signature` format has a more compact representation.
Further development will result in more metric types.

Declaring an additional plane at infinity is done by specifying it in the string constructor with ``\infty`` at the first index (i.e. Riemann sphere `S"∞+++"`). The hyperbolic geometry can be declared by ``\emptyset`` subsequently (i.e. Minkowski spacetime `S"∅+++"`).
Additionally, the *null-basis* based on the projective split for confromal geometric algebra would be specified with `∞∅` initially (i.e. 5D CGA `S"∞∅+++"`). These two declared basis elements are interpreted in the type system.

The index number ``n`` of the `TensorBundle` corresponds to the total number of generator elements. However, even though `V"∞∅+++"` is of type `TensorBundle{5,3}` with ``5`` generator elements, it can be internally recognized in the direct sum algebra as being an embedding of a 3-index `TensorBundle{3,0}` with additional encoding of the null-basis (origin and point at infinity) in the parameter ``\mathbb P`` of the `TensorBundle{n,ℙ}` type.

The `tangent` map takes ``V`` to its tangent space and can be applied repeatedly for higher orders, such that `tangent(V,μ,ν)` can be used to specify ``\mu`` and ``\nu``.
```@repl ds
V = tangent(ℝ^3)
tangent(V')
V⊕V'
```
The direct sum operator ``\oplus`` can be used to join spaces (alternatively ``+``), and the dual space functor ``'`` is an involution which toggles a dual vector space with inverted signature.
```@repl ds
V = ℝ'⊕ℝ^3
V'
W = V⊕V'
```
The direct sum of a `TensorBundle` and its dual ``V\oplus V'`` represents the full mother space ``V*``.
```@repl ds
collect(V) # all SubManifold vector basis elements
collect(SubManifold(V')) # all covector basis elements
collect(SubManifold(W)) # all mixed basis elements
```
In addition to the direct-sum operation, several other operations are supported, such as ``\cup,\cap,\subseteq,\supseteq`` for set operations.
Due to the design of the `TensorBundle` dispatch, these operations enable code optimizations at compile-time provided by the bit parameters.
```@repl ds
ℝ⊕ℝ' ⊇ Manifold(1)
ℝ ∩ ℝ' == Manifold(0)
ℝ ∪ ℝ' == ℝ⊕ℝ'
```
**Remark**. Although some of the operations like ``\cup`` and ``\oplus`` are similar and sometimes result in the same values, the `union` and `⊕` are entirely different operations in general.
```math
\bigcup T^{\mu_i}V_i = \left(|\mathbb P|+\max\{n_i-|\mathbb P_i|\}_i,\, \bigcup \mathbb P_i,\, \cup g_i,\, \max\{\mu_i\}_i\right)
```
```math
\bigoplus T^{\mu_i}V_i = \left(|\mathbb P|+\sum (n_i-|\mathbb P_i|),\, \bigcup \mathbb P_i,\, \oplus_i g_i,\,\max\{\mu_i\}_i\right)
```
Calling manifolds with sets of indices constructs the subspace representations.
Given `M(s::Int...)` one can encode `SubManifold{length(s),M,s}` with induced orthogonal space ``Z``, such that computing unions of submanifolds is done by inspecting the parameter ``s\in V\subseteq W`` and ``s\notin Z``.
```@repl ds
(ℝ^5)(3,5)
dump(ans)
```
Here, calling a `Manifold` with a set of indices produces a `SubManifold` representation.
```math
T^eV \subset T^\mu W \iff \exists Z\in\text{Vect}_{\mathbb K}(T^e(V\oplus Z) = T^{e\leq \mu}W,\,V\perp Z).
```
Operations on `Manifold` types is automatically handled at compile time.

To help provide a commonly shared and readable indexing to the user, some extended dual index print methods with full alphanumeric characters (62+2) are provided:
```@repl ds
DirectSum.printindices(stdout,DirectSum.indices(UInt(2^62-1)),false,"v")
DirectSum.printindices(stdout,DirectSum.indices(UInt(2^62-1)),false,"w")
```
An application of this is in the `Grasmann` package, where dual indexing is used.

More information about `DirectSum` is available  at [https://github.com/chakravala/DirectSum.jl](https://github.com/chakravala/DirectSum.jl)

## Interoperability for `TensorAlgebra{V}`

[![DOI](https://zenodo.org/badge/169811826.svg)](https://zenodo.org/badge/latestdoi/169811826)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/chakravala/AbstractTensors.jl)](https://github.com/chakravala/AbstractTensors.jl/releases)
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/chakravala/AbstractTensors.jl/latest?label=new%20commits)](https://github.com/chakravala/AbstractTensors.jl/commits)
[![Build Status](https://travis-ci.org/chakravala/AbstractTensors.jl.svg?branch=master)](https://travis-ci.org/chakravala/AbstractTensors.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/yey8huk505h4b81u?svg=true)](https://ci.appveyor.com/project/chakravala/abstracttensors-jl)

The `AbstractTensors` package is intended for universal interoperability of the abstract `TensorAlgebra` type system.
All `TensorAlgebra{V}` subtypes have type parameter ``V``, used to store a `TensorBundle` value obtained from *DirectSum.jl*.
By itself, this package does not impose any specifications or structure on the `TensorAlgebra{V}` subtypes and elements, aside from requiring ``V`` to be a `TensorBundle`.
This means that different packages can create tensor types having a common underlying `TensorBundle` structure.
For example, this is mainly used in *Grassmann.jl* to define various `SubAlgebra`, `TensorTerm` and `TensorMixed` types, each with subtypes. Externalizing the abstract type helps extend the dispatch to other packages.

The key to making the whole interoperability work is that each `TensorAlgebra` subtype shares a `TensorBundle` parameter (with all `isbitstype` parameters), which contains all the info needed at compile time to make decisions about conversions. So other packages need only use the vector space information to decide on how to convert based on the implementation of a type. If external methods are needed, they can be loaded by `Requires` when making a separate package with `TensorAlgebra` interoperability.

Since `TensorBundle` choices are fundamental to `TensorAlgebra` operations, the universal interoperability between `TensorAlgebra{V}` elements with different associated `TensorBundle` choices is naturally realized by applying the `union` morphism to operations,
e.g. ``\bigwedge :\Lambda^{p_1}V_1\times\dots\times\Lambda^{p_g}V_g \rightarrow \Lambda^{\sum_kp_k}\bigcup_k V_k``.
Some of the method names like ``+,-,*,\otimes,\circledast,\odot,\boxtimes,\star`` for `TensorAlgebra` elements are shared across different packages, with interoperability.
```julia
function op(::TensorAlgebra{V},::TensorAlgebra{V}) where V
    # well defined operations if V is shared
end # but what if V ≠ W in the input types?

function op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W}
    VW = V ∪ W        # VectorSpace type union
    op(VW(a),VW(b))   # makes call well-defined
end # this option is automatic with interop(a,b)

# alternatively for evaluation of forms, VW(a)(VW(b))
```
Suppose we are dealing with a new subtype in another project, such as
```@example at
using AbstractTensors, DirectSum
struct SpecialTensor{V} <: TensorAlgebra{V} end
a = SpecialTensor{ℝ}()
b = SpecialTensor{ℝ'}()
nothing # hide
```
To define additional specialized interoperability for further methods, it is necessary to define dispatch that catches well-defined operations for equal `TensorBundle` choices and a fallback method for interoperability, along with a `TensorBundle` morphism:
```@example at
(W::Signature)(s::SpecialTensor{V}) where V = SpecialTensor{W}() # conversions
op(a::SpecialTensor{V},b::SpecialTensor{V}) where V = a # do some kind of operation
op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W} = interop(op,a,b) # compat
nothing # hide
```
which should satisfy (using the ``\cup`` operation as defined in `DirectSum`)
```@repl at
op(a,b) |> Manifold == Manifold(a) ∪ Manifold(b)
```
Thus, interoperability is simply a matter of defining one additional fallback method for the operation and also a new form `TensorBundle` compatibility morphism.

Additionally, a universal unit volume element can be specified in terms of `LinearAlgebra.UniformScaling`, which is independent of ``V`` and has its interpretation only instantiated by the context of the `TensorAlgebra{V}` element being operated on.
The universal interoperability of `LinearAlgebra.UniformScaling` as a pseudoscalar element which takes on the `TensorBundle` form of any other `TensorAlgebra` element is handled globally.
This enables the usage of ``I`` from `LinearAlgebra` as a universal pseudoscalar element.
```julia
(W::Signature)(s::UniformScaling) = ones(ndims(W)) # interpret a unit pseudoscalar
op(a::TensorAlgebra{V},b::UniformScaling) where V = op(a,V(b)) # right pseudoscalar
op(a::UniformScaling,b::TensorAlgebra{V}) where V = op(V(a),b) # left pseudoscalar
```
Utility methods such as `scalar, involute, norm, norm2, unit, even, odd` are also defined.

To support a generalized interface for `TensorAlgebra` element evaluation, a similar compatibility interface is constructible.
```@example at
(a::SpecialTensor{V})(b::SpecialTensor{V}) where V = a # conversion of some form
(a::SpecialTensor{W})(b::SpecialTensor{V}) where {V,W} = interform(a,b) # compat
nothing # hide
```
which should satisfy (using the ``\cup`` operation as defined in `DirectSum`)
```@repl at
b(a) |> Manifold == Manifold(a) ∪ Manifold(b)
```
The purpose of the `interop` and `interform` methods is to help unify the interoperability of `TensorAlgebra` elements.

More information about `DirectSum` is available  at [https://github.com/chakravala/AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl)
