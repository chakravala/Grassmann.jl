# Dyadic tensor product ⊗

Dyadic tensors are represented with the Grassmann’s exterior product algebra or nested `Chain{V,1,Chain{V,1}}` elements, generating a ``2^{2n}``-dimensional mother algebra with the direct sum of the ``n``-dimensional vector space and its dual vector space.
The product of the vector basis and covector basis elements form the ``n^2``-dimensional bivector subspace of the full ``\frac{(2n)!}{2(2n−2)!}``-dimensional bivector sub-algebra.
The package `Grassmann` is working towards making the full extent of this number system available in Julia by using static compiled parametric type information to handle sparse sub-algebras, such as the dyadic (1,1)-tensor bivector algebra or mother algebra.

## Nested dyadic algebra

In this algebra, it's possible to compute on a mesh of arbitrary 5 dimensional conformal geometric algebra simplices, which can be represented by a bundle of  nested dyadic tensors.

```@repl ga5
using Grassmann, StaticArrays; basis"+-+++"
value(rand(Chain{V,1,Chain{V,1}}))
A = Chain{V,1}(rand(SMatrix{5,5}))
```

Additionally, in Grassmann.jl we prefer the nested usage of pure `ChainBundle` parametric types for large re-usable global cell geometries, from which local dyadics can be selected.

Programming the `A\b` method is straight forward with some Julia language metaprogramming and Grassmann.jl by first instantiating some Cramer symbols

```@repl ga5
Base.@pure function Grassmann.Cramer(N::Int)
    x,y = SVector{N}([Symbol(:x,i) for i ∈ 1:N]),SVector{N}([Symbol(:y,i) for i ∈ 1:N])
    xy = [:(($(x[1+i]),$(y[1+i])) = ($(x[i])∧t[$(1+i)],t[end-$i]∧$(y[i]))) for i ∈ 1:N-1]
    return x,y,xy
end
```

These are exterior product variants of the Cramer determinant symbols ($N!$ times $N$-simplex hypervolumes), which can be combined to directly solve a linear system:

```@repl ga5
@generated function Base.:\(t::Chain{V,1,<:Chain{V,1}},v::Chain{V,1}) where V
    N = ndims(V)-1 # paste this into the REPL for faster eval
    x,y,xy = Grassmann.Cramer(N)
    mid = [:($(x[i])∧v∧$(y[end-i])) for i ∈ 1:N-1]
    out = Expr(:call,:SVector,:(v∧$(y[end])),mid...,:($(x[end])∧v))
    return Expr(:block,:((x1,y1)=(t[1],t[end])),xy...,
        :(Chain{V,1}(getindex.($(Expr(:call,:./,out,:(t[1]∧$(y[end])))),1))))
end
```

Which results in the following highly efficient `@generated` code for solving the linear system,

```Julia
(x1, y1) = (t[1], t[end])
(x2, y2) = (x1 ∧ t[2], t[end - 1] ∧ y1)
(x3, y3) = (x2 ∧ t[3], t[end - 2] ∧ y2)
(x4, y4) = (x3 ∧ t[4], t[end - 3] ∧ y3)
Chain{V, 1}(getindex.(SVector(v ∧ y4, (x1 ∧ v) ∧ y3, (x2 ∧ v) ∧ y2, (x3 ∧ v) ∧ y1, x4 ∧ v) ./ (t[1] ∧ y4), 1))
```

Benchmarks with that algebra indicate a $3\times$ faster performance than `SMatrix` for applying `A\b` to bundles of dyadic elements.

```Julia
julia> @btime $(rand(SMatrix{5,5},10000)).\Ref($(SVector(1,2,3,4,5)));
  2.588 ms (29496 allocations: 1.44 MiB)

julia> @btime $(Chain{V,1}.(rand(SMatrix{5,5},10000))).\$(v1+2v2+3v3+4v4+5v5);
  808.631 μs (2 allocations: 390.70 KiB)

julia> @btime $(SMatrix(A))\$(SVector(1,2,3,4,5))
  150.663 ns (0 allocations: 0 bytes)
5-element SArray{Tuple{5},Float64,1,5} with indices SOneTo(5):
 -4.783720495603508
  6.034887114999602
  1.017847212237964
  6.379374861538397
 -4.158116538111051

julia> @btime $A\$(v1+2v2+3v3+4v4+5v5)
  72.405 ns (0 allocations: 0 bytes)
-4.783720495603519v₁ + 6.034887114999605v₂ + 1.017847212237964v₃ + 6.379374861538393v₄ - 4.1581165381110505v₅
```

Such a solution is not only more efficient than Julia's [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) method for `SMatrix`, but is also useful to minimize allocations in Grassmann.jl finite element assembly.

Similarly, the Cramer symbols can also be manipulated to invert the linear system or for determining whether a point is within a simplex.

```@setup ga3
using StaticArrays
```

```@repl ga3
using Grassmann; basis"3"
T = Chain{V,1}(Chain(v1),v1+v2,v1+v3)
barycenter(T) ∈ T, (v1+v2+v3) ∈ T
```

Of course, there are multiple equivalent ways of computing the same results using the `⋅` and `:` dyadic products.

```@repl ga3
T\barycenter(T) == inv(T)⋅barycenter(T)
sqrt(T:T) == norm(SMatrix(T))
```
It is possible to generate a [Makie.jl](https://github.com/JuliaPlots/Makie.jl) `streamplot` diagrams with the `Grassmann.Cramer` method for interpolated data of finite element solutions.


## Mother algebra formalism

Note that `Λ(3)` gives the vector basis, and `Λ(3)'` gives the covector basis:
```@setup ga
using Grassmann
```
```@repl ga
Λ(3)
Λ(3)'
```
The following command yields a local 2D vector and covector basis,
```@repl ga
mixedbasis"2"
w1+2w2
ans(v1+v2)
```
The sum `w1+2w2` is interpreted as a covector element of the dual vector space, which can be evaluated as a linear functional when a vector argument is input.
Using these in the workspace, it is possible to use the Grassmann exterior ``\wedge``-tensor product operation to construct elements `ℒ` of the dyadic (1,1)-bivector subspace of linear transformations from the mother algebra.
```@repl ga
ℒ = (v1+2v2)∧(3w1+4w2)
```
The element `ℒ` is a linear form which can be evaluated,
```@repl ga
ℒ(v1+v2)
L = [1,2] * [3,4]'; L * [1,1]
```
which is a computation equivalent to a matrix computation.

The `TensorAlgebra` evalution is still a work in progress, and the API and implementation may change as more features and algebraic operations and product structure are added.

### Importing the Leech lattice generator

In the example below, we define a constant `Leech` which can be used to obtain linear combinations of the Leech lattice,
```julia
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

julia> const E24,W24 = Λ(24), ℝ^24⊕(ℝ^24)';

julia> const Leech = Chain{SubManifold(W24),Float64}(generator./sqrt(8));

julia> typeof(Leech)
Chain{⟨++++++++++++++++++++++++------------------------⟩*,2,Float64,1128}

julia> ndims(Manifold(Leech))
48
```
The `Leech` generator matrix is contained in the 1128-dimensional bivector subalgebra of the space with 48 indices.
```julia
julia> Leech(E24.v1)
2.82842712474619v₁ + 0.0v₂ + 0.0v₃ + 0.0v₄ + 0.0v₅ + 0.0v₆ + 0.0v₇ + 0.0v₈ + 0.0v₉ + 0.0v₀ + 0.0va + 0.0vb + 0.0vc + 0.0vd + 0.0ve + 0.0vf + 0.0vg + 0.0vh + 0.0vi + 0.0vj + 0.0vk + 0.0vl + 0.0vm + 0.0vn + 0.0w¹ + 0.0w² + 0.0w³ + 0.0w⁴ + 0.0w⁵ + 0.0w⁶ + 0.0w⁷ + 0.0w⁸ + 0.0w⁹ + 0.0w⁰ + 0.0wA + 0.0wB + 0.0wC + 0.0wD + 0.0wE + 0.0wF + 0.0wG + 0.0wH + 0.0wI + 0.0wJ + 0.0wK + 0.0wL + 0.0wM + 0.0wN

julia> Leech(E24.v2)
1.414213562373095v₁ + 1.414213562373095v₂ + 0.0v₃ + 0.0v₄ + 0.0v₅ + 0.0v₆ + 0.0v₇ + 0.0v₈ + 0.0v₉ + 0.0v₀ + 0.0va + 0.0vb + 0.0vc + 0.0vd + 0.0ve + 0.0vf + 0.0vg + 0.0vh + 0.0vi + 0.0vj + 0.0vk + 0.0vl + 0.0vm + 0.0vn + 0.0w¹ + 0.0w² + 0.0w³ + 0.0w⁴ + 0.0w⁵ + 0.0w⁶ + 0.0w⁷ + 0.0w⁸ + 0.0w⁹ + 0.0w⁰ + 0.0wA + 0.0wB + 0.0wC + 0.0wD + 0.0wE + 0.0wF + 0.0wG + 0.0wH + 0.0wI + 0.0wJ + 0.0wK + 0.0wL + 0.0wM + 0.0wN

julia> Leech(E24.v3)
1.414213562373095v₁ + 0.0v₂ + 1.414213562373095v₃ + 0.0v₄ + 0.0v₅ + 0.0v₆ + 0.0v₇ + 0.0v₈ + 0.0v₉ + 0.0v₀ + 0.0va + 0.0vb + 0.0vc + 0.0vd + 0.0ve + 0.0vf + 0.0vg + 0.0vh + 0.0vi + 0.0vj + 0.0vk + 0.0vl + 0.0vm + 0.0vn + 0.0w¹ + 0.0w² + 0.0w³ + 0.0w⁴ + 0.0w⁵ + 0.0w⁶ + 0.0w⁷ + 0.0w⁸ + 0.0w⁹ + 0.0w⁰ + 0.0wA + 0.0wB + 0.0wC + 0.0wD + 0.0wE + 0.0wF + 0.0wG + 0.0wH + 0.0wI + 0.0wJ + 0.0wK + 0.0wL + 0.0wM + 0.0wN

...
```
Then a `TensorAlgebra` evaluation of `Leech` at an `Integer` linear combination would be
```julia
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
