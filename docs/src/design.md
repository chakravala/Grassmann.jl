# `TensorAlgebra{V}` design and code generation

Mathematical foundations and definitions specific to the [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) implementation provide an extensible platform for computing with a universal language for finite element methods based on a discrete manifold bundle. 
Tools built on these foundations enable computations based on multi-linear algebra and spin groups using the geometric algebra known as Grassmann algebra or Clifford algebra.
This foundation is built on a [DirectSum.jl](https://github.com/chakravala/DirectSum.jl) parametric type system for tangent bundles and vector spaces generating the algorithms for local tangent algebras in a global context.
With this unifying mathematical foundation, it is possible to improve efficiency of multi-disciplinary research using geometric tensor calculus by relying on universal mathematical principles.

* **AbstractTensors.jl**: Tensor algebra abstract type interoperability setup
* **DirectSum.jl**: Tangent bundle, vector space and `Submanifold` definition
* **Grassmann.jl**: ⟨Grassmann-Clifford-Hodge⟩ multilinear differential geometric algebra

## Direct sum parametric type polymorphism

[![DOI](https://zenodo.org/badge/169765288.svg)](https://zenodo.org/badge/latestdoi/169765288)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/chakravala/DirectSum.jl)](https://github.com/chakravala/DirectSum.jl/releases)
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/chakravala/DirectSum.jl/latest?label=new%20commits)](https://github.com/chakravala/DirectSum.jl/commits)
[![Build status](https://ci.appveyor.com/api/projects/status/ipaggdeq2f1509pl?svg=true)](https://ci.appveyor.com/project/chakravala/directsum-jl)

The *DirectSum.jl* package is a work in progress providing the necessary tools to work with an arbitrary `Manifold` specified by an encoding.
Due to the parametric type system for the generating `TensorBundle`, the Julia compiler can fully preallocate and often cache values efficiently ahead of run-time.
Although intended for use with the *Grassmann.jl* package, `DirectSum` can be used independently.

Let ``M = T^\mu V`` be a ``\mathbb{K}``-module of rank ``n``, then an instance for
``T^\mu V`` can be the tuple ``(n,\mathbb{P},g,\nu,\mu)`` with ``\mathbb{P}\subseteq \langle v_\infty,v_\emptyset\rangle`` specifying the presence of the projective basis and ``g:V\times V\rightarrow\mathbb{K}`` is a metric tensor specification.
The type `TensorBundle{n,```\mathbb{P}```,g,```\nu```,```\mu```}` encodes this information as *byte-encoded* data available at pre-compilation,
where ``\mu`` is an integer specifying the order of the tangent bundle (i.e. multiplicity limit of the Leibniz-Taylor monomials).
Lastly, ``\nu`` is the number of tangent variables.
```math
\langle v_1,\dots,v_{n-\nu},\partial_1,\dots,\partial_\nu\rangle=M\leftrightarrow M' = \langle w_1,\dots,w_{n-\nu},\epsilon_1,\dots,\epsilon_\nu\rangle
```
where ``v_i`` and ``w_i`` are bases for the vectors and covectors, while ``\partial_i`` and ``\epsilon_j`` are bases for differential operators and scalar functions.
The purpose of the `TensorBundle` type is to specify the ``\mathbb{K}``-module basis at compile time.
When assigned in a workspace, `V = Submanifold(::TensorBundle)` is used.

The metric signature of the `Submanifold{V,1}` elements of a vector space ``V`` can be specified with the `V"..."` by using ``+`` or ``-`` to specify whether the `Submanifold{V,1}` element of the corresponding index squares to ``+1`` or ``-1``.
For example, `S"+++"` constructs a positive definite 3-dimensional `TensorBundle`, so constructors such as `S"..."` and `D"..."` are convenient.
```@setup ds
using DirectSum
```
```@repl ds
ℝ^3 == V"+++" == Manifold(3)
```
It is also possible to change the diagonal scaling, such as with `D"1,1,1,0"`, although the `Signature` format has a more compact representation if limited to ``+1`` and ``-1``.
It is also possible to change the diagonal scaling, such as with `D"0.3,2.4,1"`.
Fully general `MetricTensor` as a type with non-diagonal components requires a matrix, e.g. `MetricTensor([1 2; 2 3])`.

Declaring an additional point at infinity is done by specifying it in the string constructor with ``\infty`` at the first index (i.e. Riemann sphere `S"∞+++"`).
The hyperbolic geometry can be declared by ``\emptyset`` subsequently (i.e. hyperbolic projection `S"∅+++"`).
Additionally, the *null-basis* based on the projective split for conformal geometric algebra would be specified with `S"∞∅+++"`.
These two declared basis elements are interpreted in the type system.
The `tangent(V,μ,ν)`  map can be used to specify ``\mu`` and ``\nu``.

To assign `V = Submanifold(::TensorBundle)` along with associated basis
elements of the `DirectSum.Basis` to the local Julia session workspace, it is typical to use `Submanifold` elements created by the `@basis` macro,
```@repl ds
using Grassmann; @basis S"-++" # macro or basis"-++"
```
the macro `@basis V` delcares a local basis in Julia.
As a result of this macro, all `Submanifold{V,G}` elements generated with `M::TensorBundle` become available in the local workspace with the specified naming arguments.
The first argument provides signature specifications, the second argument is the variable name for ``V`` the ``\mathbb{K}``-module, and the third and fourth argument are prefixes of the `Submanifold` vector names (and covector names).
Default is ``V`` assigned `Submanifold{M}` and ``v`` is prefix for the `Submanifold{V}`.

It is entirely possible to assign multiple different bases having different signatures without any problems.
The `@basis` macro arguments are used to assign the vector space name to ``V`` and the basis elements to ``v_i``, but other assigned names can be chosen so that their local names don't interfere:
If it is undesirable to assign these variables to a local workspace, the versatile constructs of `DirectSum.Basis{V}` can be used to contain or access them, which is exported to the user as the method `DirectSum.Basis(V)`.
```@repl ds
DirectSum.Basis(V)
```
`V(::Int...)` provides a convenient way to define a `Submanifold` by using integer indices to reference specific direct sums within the ambient space ``V``.
```@repl ds
(ℝ^5)(3,5)
dump(ans)
```
Here, calling a `Manifold` with a set of indices produces a `Submanifold` representation.

The direct sum operator ``\oplus`` can be used to join spaces (alternatively ``+``), and the dual space functor ``'`` is an involution which toggles a dual vector space with inverted signature.
```@repl ds
V = ℝ'⊕ℝ^3
V'
W = V⊕V'
```
The direct sum of a `TensorBundle` and its dual ``V\oplus V'`` represents the full mother space ``V*``.
```@repl ds
collect(V) # all Submanifold vector basis elements
collect(Submanifold(V')) # all covector basis elements
collect(Submanifold(W)) # all mixed basis elements
```
In addition to the direct-sum operation, several other operations are supported, such as ``\cup,\cap,\subseteq,\supseteq`` for set operations.
Due to the design of the `TensorBundle` dispatch, these operations enable code optimizations at compile-time provided by the bit parameters.
```@repl ds
ℝ⊕ℝ' ⊇ Manifold(1)
ℝ ∩ ℝ' == Manifold(0)
ℝ ∪ ℝ' == ℝ⊕ℝ'
```
Operations on `Manifold` types is automatically handled at compile time.

More information about `DirectSum` is available  at [https://github.com/chakravala/DirectSum.jl](https://github.com/chakravala/DirectSum.jl)

## Higher dimensions with `SparseBasis` and `ExtendedBasis`

In order to work with a `TensorAlgebra{V}`, it is necessary for some computations to be cached. This is usually done automatically when accessed.
```julia
julia> Λ(7) ⊕ Λ(7)'
DirectSum.SparseBasis{⟨+++++++-------⟩*,16384}(v, ..., v₁₂₃₄₅₆₇w¹²³⁴⁵⁶⁷)
```
One way of declaring the cache for all 3 combinations of a `TensorBundle{N}` and its dual is to ask for the sum `Λ(V) + Λ(V)'`, which is equivalent to `Λ(V⊕V')`, but this does not initialize the cache of all 3 combinations unlike the former.

Staging of precompilation and caching is designed so that a user can smoothly transition between very high dimensional and low dimensional algebras in a single session, with varying levels of extra caching and optimizations.
The parametric type formalism in `Grassmann` is highly expressive and enables pre-allocation of geometric algebra computations involving specific sparse subalgebras, including the representation of rotational groups.

It is possible to reach elements with up to ``N=62`` vertices from a `TensorAlgebra` having higher maximum dimensions than supported by Julia natively.
```@repl ds
Λ(62)
Λ(62).v32a87Ng
```
The 62 indices require full alpha-numeric labeling with lower-case and capital letters. This now allows you to reach up to ``4,611,686,018,427,387,904`` dimensions with Julia `using Grassmann`. Then the volume element is
```@example
using DirectSum # hide
DirectSum.printindices(stdout,DirectSum.indices(UInt(2^62-1))) # hide
```
Full `Multivector` allocations are only possible for ``N\leq22``, but sparse operations are also available at higher dimensions.
While `DirectSum.Basis{V}` is a container for the `TensorAlgebra` generators of ``V``, the `Basis` is only cached for ``N\leq8``.
For the range of dimensions ``8<N\leq22``, the `SparseBasis` type is used.
```julia
julia> Λ(22)
DirectSum.SparseBasis{⟨++++++++++++++++++++++⟩,4194304}(v, ..., v₁₂₃₄₅₆₇₈₉₀abcdefghijkl)
```
This is the largest `SparseBasis` that can be generated with Julia, due to array size limitations.

To reach higher dimensions with ``N>22``, the `DirectSum.ExtendedBasis` type is used.
It is suficient to work with a 64-bit representation (which is the default). And it turns out that with 62 standard keyboard characters, this fits.
```@repl ds
V = ℝ^22
Λ(V+V')
```
At 22 dimensions and lower there is better caching, with further extra caching for 8 dimensions or less.
Thus, the largest Hilbert space that is fully reachable has 4,194,304 dimensions, but we can still reach out to 4,611,686,018,427,387,904 dimensions with the `ExtendedBasis` built in.
It is still feasible to extend to a further super-extended 128-bit representation using the `UInt128` type (but this will require further modifications of internals and helper functions.
To reach into infinity even further, it is theoretically possible to construct ultra-extensions also using dictionaries.
Full `Multivector` elements are not representable when `ExtendedBasis` is used, but the performance of the `Basis` and sparse elements should be just as fast as for lower dimensions for the current `SubAlgebra` and `TensorAlgebra` types.
The sparse representations are a work in progress to be improved with time.

## Interoperability for `TensorAlgebra{V}`

[![DOI](https://zenodo.org/badge/169811826.svg)](https://zenodo.org/badge/latestdoi/169811826)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/chakravala/AbstractTensors.jl)](https://github.com/chakravala/AbstractTensors.jl/releases)
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/chakravala/AbstractTensors.jl/latest?label=new%20commits)](https://github.com/chakravala/AbstractTensors.jl/commits)
[![Build status](https://ci.appveyor.com/api/projects/status/yey8huk505h4b81u?svg=true)](https://ci.appveyor.com/project/chakravala/abstracttensors-jl)

The `AbstractTensors` package is intended for universal interoperation of the abstract `TensorAlgebra` type system.
All `TensorAlgebra{V}` subtypes have type parameter ``V``, used to store a `Submanifold{M}` value, which is parametrized by ``M`` the `TensorBundle` choice.
This means that different tensor types can have a commonly shared underlying ``\mathbb{K}``-module parametric type expressed by defining `V::Submanifold{M}`.
Each `TensorAlgebra` subtype must be accompanied by a corresponding `TensorBundle` parameter, which is fully static at compile time.
Due to the parametric type system for the ``\mathbb{K}``-module types, the compiler can fully pre-allocate and often cache.

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

Additionally, a universal unit volume element can be specified in terms of `LinearAlgebra.UniformScaling`, which is independent of ``V`` and has its interpretation only instantiated by context of `TensorAlgebra{V}` elements being operated on.
Interoperability of `LinearAlgebra.UniformScaling` as a pseudoscalar element which takes on the `TensorBundle` form of any other `TensorAlgebra` element is handled globally.
This enables the usage of `I` from `LinearAlgebra` as a universal pseudoscalar element defined at every point ``x`` of a `Manifold`, which is mathematically denoted by ``I = I(x)`` and specified by the ``g(x)`` bilinear tensor field of ``TM``.

More information about `AbstractTensors` is available  at [https://github.com/chakravala/AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl)
