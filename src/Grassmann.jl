module Grassmann

#   This file is part of Grassmann.jl
#   It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com
#                                              ___
#                 /\                  _____   / _ \
#   __   __      /  \      __   __   (_____) | | | |
#  / / _ \ \    / /\ \    / / _ \ \   _____  | | | |
# | |_/ \_| |  / /  \ \  | |_/ \_| | (_____) | |_| |
#  \___^___/  /_/    \_\  \___^___/           \___/

using SparseArrays, ComputedFieldTypes
using AbstractTensors, Leibniz, DirectSum, Requires
import AbstractTensors: Values, Variables, FixedVector, clifford, hodge, wedge, vee

export ⊕, ℝ, @V_str, @S_str, @D_str, Manifold, Submanifold, Signature, DiagonalForm, value
export @basis, @basis_str, @dualbasis, @dualbasis_str, @mixedbasis, @mixedbasis_str, Λ
export ℝ0, ℝ1, ℝ2, ℝ3, ℝ4, ℝ5, ℝ6, ℝ7, ℝ8, ℝ9, mdims, tangent, metric, antimetric
export hodge, wedge, vee, complement, dot, antidot, istangent, Values

import Base: @pure, ==, isapprox
import Base: print, show, getindex, setindex!, promote_rule, convert, adjoint
import DirectSum: V0, ⊕, generate, basis, getalgebra, getbasis, dual, Zero, One, Zero, One
import Leibniz: hasinf, hasorigin, dyadmode, value, pre, vsn, metric, mdims, gdims
import Leibniz: bit2int, indexbits, indices, diffvars, diffmask, hasconformal
import Leibniz: symmetricmask, indexstring, indexsymbol, combo, digits_fast
import DirectSum: metrichash, antimetric, signbool

import AbstractTensors: valuetype, scalar, isscalar, trivector, istrivector, ⊗, complement
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume
import AbstractTensors: wedgedot_metric, contraction_metric, log_metric

## cache

import Leibniz: algebra_limit, sparse_limit, cache_limit, fill_limit, gdimsall, spincumsum
import Leibniz: binomial, binomial_set, binomsum, binomsum_set, lowerbits, expandbits
import Leibniz: bladeindex, basisindex, indexbasis, indexbasis_set, loworder, intlog
import Leibniz: antisum, antisum_set, anticumsum, antiindex, spinindex, binomcumsum
import Leibniz: promote_type, mvec, svec, intlog, insert_expr, supermanifold

include("multivectors.jl")
include("parity.jl")
include("algebra.jl")
include("products.jl")
include("composite.jl")
include("forms.jl")

## fundamentals

export cayley, hyperplanes, points, TensorAlgebra

@pure hyperplanes(V::Manifold) = map(n -> UniformScaling{Bool}(false) * getbasis(V, 1 << n), 0:rank(V)-1-diffvars(V))

for M ∈ (:Signature, :DiagonalForm)
    @eval (::$M)(::S) where {S<:SubAlgebra{V}} where {V} = Multivector{V,Int}(ones(Int, 1 << mdims(V)))
end

points(f::F, r=-2π:0.0001:2π) where {F<:Function} = vector.(f.(r))

export 𝕚, 𝕛, 𝕜
const 𝕚, 𝕛, 𝕜 = hyperplanes(ℝ3)

using Leibniz
import Leibniz: ∇, Δ, d, ∂
export ∇, Δ, ∂, d, δ, ↑, ↓, up, down, nabla, laplacian, boundary, codifferential, exterior

#generate_products(:(Leibniz.Operator),:svec)
for T ∈ (:(Chain{V}), :(Multivector{V}))
    @eval begin
        *(a::Derivation, b::$T) where {V} = V(a) * b
        *(a::$T, b::Derivation) where {V} = a * V(b)
    end
end
⊘(x::T, y::Derivation) where {T<:TensorAlgebra{V}} where {V} = x ⊘ V(y)
⊘(x::Derivation, y::T) where {T<:TensorAlgebra{V}} where {V} = V(x) ⊘ y

@pure function (V::Signature{N})(d::Leibniz.Derivation{T,O}) where {N,T,O}
    (O < 1 || diffvars(V) == 0) && (return Chain{V,1,Int}(d.v.λ * ones(Values{N,Int})))
    G, D, C = grade(V), diffvars(V) == 1, isdyadic(V)
    G2 = (C ? Int(G / 2) : G) - 1
    ∇ = sum([getbasis(V, 1 << (D ? G : k + G)) * getbasis(V, 1 << k) for k ∈ 0:G2])
    isone(O) && (return ∇)
    x = (∇ ⋅ ∇)^div(isodd(O) ? O - 1 : O, 2)
    isodd(O) ? sum([(x * getbasis(V, 1 << (k + G))) * getbasis(V, 1 << k) for k ∈ 0:G2]) : x
end

@pure function (M::Submanifold{W,N})(d::Leibniz.Derivation{T,O}) where {W,N,T,O}
    V = isbasis(M) ? W : M
    (O < 1 || diffvars(V) == 0) && (return Chain{V,1,Int}(d.v.λ * ones(Values{N,Int})))
    G, D, C = grade(V), diffvars(V) == 1, isdyadic(V)
    G2 = (C ? Int(G / 2) : G) - 1
    ∇ = sum([getbasis(V, 1 << (D ? G : k + G)) * getbasis(V, 1 << k) for k ∈ 0:G2])
    isone(O) && (return ∇)
    x = (∇ ⋅ ∇)^div(isodd(O) ? O - 1 : O, 2)
    isodd(O) ? sum([(x * getbasis(V, 1 << (k + G))) * getbasis(V, 1 << k) for k ∈ 0:G2]) : x
end

"""
    ∂(ω)

Boundary / interior derivative acting on tensor algebra elements.

Given a tensor `ω` on a manifold `V`, the operator contracts `ω` with the
vector‑valued gradient `∇`, lowering its grade by one:

```julia
∂(ω) == ω ⋅ ∇
```

The operator is nilpotent (`∂^2 == 0`) and constitutes the algebraic dual of
the exterior derivative `d`.  A specialised **generated** method is provided
for the common case of a 1‑chain whose coefficients are themselves
1‑chains.
"""
@generated ∂(ω::Chain{V,1,<:Chain{W,1}}) where {V,W} = :(∧(ω) ⋅ $(Λ(W).v1))
∂(ω::T) where {T<:TensorAlgebra} = ω ⋅ Manifold(ω)(∇)

"""
    d(ω)

Exterior derivative. It raises the grade of `ω` by wedging the gradient with
`ω`:

```julia
d(ω) == ∇ ∧ ω
```

`d` is nilpotent (`d^2 == 0`) and, together with `δ`, forms the de Rham
cochain complex (`Δ = d⋅δ + δ⋅d`).
"""
d(ω::T) where {T<:TensorAlgebra} = Manifold(ω)(∇) ∧ ω

"""
    δ(ω)

Codifferential — the adjoint of the exterior derivative.  In Grassmann it is
defined as the negative of the boundary operator:

```julia
δ(ω) == -∂(ω)
```

`δ` lowers the grade by one and satisfies `δ^2 == 0`.
"""
δ(ω::T) where {T<:TensorAlgebra} = -∂(ω)

"""
    boundary_rank(t)

Return the **rank** of the boundary operator evaluated on `t`.

The argument `t` can be any element of the tensor algebra.  Internally the
function counts how many *graded dimensions* of `t` survive after applying
the boundary operator `∂` and clips them so they cannot exceed the original
graded dimensions.

The result is a `Values` object whose entries `rₖ` satisfy `0 ≤ rₖ ≤ dₖ` with
`d = count_gdims(t)`.
"""
function boundary_rank(t, d=count_gdims(t))
    out = count_gdims(∂(t))
    out[1] = 0
    for k ∈ 2:length(out)-1
        @inbounds out[k] = min(out[k], d[k+1])
    end
    return Values(out)
end

"""
    boundary_null(t)

Dimension of the **nullspace** (kernel) of the boundary operator on `t`.

For each grade `k` the function returns `dₖ₊₁ - rₖ` where
`d = count_gdims(t)` and `r = boundary_rank(t)`.  Intuitively, this measures
how many `k`‑chains are **boundaries of (k+1)‑chains** and therefore vanish
under `∂`.
"""
function boundary_null(t)
    d = count_gdims(t)
    r = boundary_rank(t, d)
    l = length(d)
    out = zeros(Variables{l,Int})
    for k ∈ 1:l-1
        @inbounds out[k] = d[k+1] - r[k]
    end
    return Values(out)
end

"""
    betti(::TensorAlgebra)

Compute the Betti numbers.
"""
function betti(t::T) where {T<:TensorAlgebra}
    d = count_gdims(t)
    r = boundary_rank(t, d)
    l = length(d) - 1
    out = zeros(Variables{l,Int})
    for k ∈ 1:l
        @inbounds out[k] = d[k+1] - r[k] - r[k+1]
    end
    return Values(out)
end

raw"""
    ↑(ω::TensorAlgebra)

Conformal **up**-projection. For a Euclidean or projective multivector `ω` that
lives in a manifold `V`, `↑` embeds the point into the conformal geometric
algebra (CGA)

```math
\mathcal{C}(V)=V \oplus \operatorname{span}\{\mathbf v_{\infty},\;\mathbf v_{\emptyset}\},
```

by constructing a *null* vector whose inner product reproduces Euclidean
squared distances. In conventional CGA notation the map is

```math
x \;\mapsto\; x + \tfrac{1}{2}\lVert x\rVert^{2}\,\mathbf v_{\infty} + \mathbf v_{\emptyset},
```

so that

```math
\uparrow(x) \cdot \uparrow(y) = -\tfrac{1}{2}\lVert x-y\rVert^{2}.
```

If `V` already contains the required null directions the argument is returned
unchanged.

This method is a *generated function* so the explicit algebraic form is
specialised at compile time for the concrete `Manifold` carried by `ω`.

Extra arities expose lower-level building blocks:

  • `↑(ω, b)` — use an explicit null basis element `b` (typically `v∞` or `v∅`).
  • `↑(ω, p, m)` — parameterise the conformal split with point-like part `p` and Minkowski part `m`.

See also [`↓`](@ref), the inverse **down**-projection.
"""
@generated function ↑(ω::T) where {T<:TensorAlgebra}
    V = Manifold(ω)
    T <: Submanifold && !isbasis(ω) && (return Leibniz.supermanifold(V))
    !(hasinf(V) || hasorigin(V)) && (return :ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        :((($G.v∞ * (one(valuetype(ω)) / 2)) * ((~ω) ⋅ ω) + $G.v∅) + ω)
        #:((($G.v∞*(one(valuetype(ω))/2))*ω^2+$G.v∅)+ω)
    else
        quote
            ω2 = (~ω) ⋅ ω # ω^2
            iω2 = inv(ω2 + 1)
            (hasinf($V) ? $G.v∞ : $G.v∅) * ((ω2 - 1) * iω2) + (2iω2) * ω
        end
    end
end
#↑(ω::ChainBundle) = ω
function ↑(ω, b)
    ω2 = (~ω) ⋅ ω # ω^2
    iω2 = inv(ω2 + 1)
    (2iω2) * ω + ((ω2 - 1) * iω2) * b
end
function ↑(ω, p, m)
    ω2 = scalar((~ω) ⋅ ω) # ω^2
    iω2 = inv(ω2 + 1)
    (2iω2) * ω + ((ω2 - 1) * iω2) * p + ((ω2 + 1) * iω2) * m
end

raw"""
    ↓(ω::TensorAlgebra)

Conformal **down**-projection, the left inverse of [`↑`](@ref). It projects a null CGA multivector
back to its Euclidean component by eliminating the null basis directions `v∞` and `v∅`.

If the ambient manifold does not contain a conformal split the operator acts as an identity, or
(for `Submanifold`s) returns the appropriate slice.
"""
@generated function ↓(ω::T) where {T<:TensorAlgebra}
    V, M = Manifold(ω), T <: Submanifold && !isbasis(ω)
    !(hasinf(V) || hasorigin(V)) && (return M ? V(2:mdims(V)) : :ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        M && (return ω(3:mdims(V)))
        #:(inv(one(valuetype(ω))*$G.v∞∅)*($G.v∞∅∧ω)/(-ω⋅$G.v∞))
        :((($G.v∞∅ ∧ ω) ⋅ inv(one(valuetype(ω)) * ~$G.v∞∅)) / (-ω ⋅ $G.v∞))
    else
        M && (return V(2:mdims(V)))
        quote
            b = hasinf($V) ? $G.v∞ : $G.v∅
            (~(ω ∧ b) ⋅ b) / (1 - b ⋅ ω) # ((ω∧b)*b)/(1-b⋅ω)
        end
    end
end
#↓(ω::ChainBundle) = ω(list(2,mdims(ω)))
↓(ω, b) = (~(b ∧ ω) ⋅ b) / (1 - ω ⋅ b) # ((b∧ω)*b)/(1-ω⋅b)
↓(ω, ∞, ∅) = (m = ∞ ∧ ∅; ((m ∧ ω) ⋅ ~inv(m)) / (-ω ⋅ ∞)) #(m=∞∧∅;inv(m)*(m∧ω)/(-ω⋅∞))

## skeleton / subcomplex

export skeleton, 𝒫, collapse, subcomplex, chain, path

absym(t) = abs(t)
absym(t::Submanifold) = t
absym(t::T) where {T<:TensorTerm{V,G}} where {V,G} = Single{V,G}(absym(value(t)), basis(t))
absym(t::Chain{V,G,T}) where {V,G,T} = Chain{V,G}(absym.(value(t)))
absym(t::Multivector{V,T}) where {V,T} = Multivector{V}(absym.(value(t)))

collapse(a, b) = a ⋅ absym(∂(b))

function chain(t::S, ::Val{T}=Val{true}()) where {S<:TensorTerm{V}} where {V,T}
    N, B, v = mdims(V), UInt(basis(t)), value(t)
    C = symmetricmask(V, B, B)[1]
    G = count_ones(C)
    G < 2 && (return t)
    out, ind = zeros(mvec(N, 2, Int)), indices(C, N)
    if T || G == 2
        setblade!(out, G == 2 ? v : -v, bit2int(indexbits(N, [ind[1], ind[end]])), Val{N}())
    end
    for k ∈ 2:G
        setblade!(out, v, bit2int(indexbits(N, ind[[k - 1, k]])), Val{N}())
    end
    return Chain{V,2}(out)
end
path(t) = chain(t, Val{false}())

@inline (::Leibniz.Derivation)(x::T, v=Val{true}()) where {T<:TensorAlgebra} = skeleton(x, v)
𝒫(t::T) where {T<:TensorAlgebra} = Δ(t, Val{false}())
subcomplex(x::S, v=Val{true}()) where {S<:TensorAlgebra} = Δ(absym(∂(x)), v)
function skeleton(x::S, v::Val{T}=Val{true}()) where {S<:TensorTerm{V}} where {V,T}
    B = UInt(basis(x))
    count_ones(symmetricmask(V, B, B)[1]) > 0 ? absym(x) + skeleton(absym(∂(x)), v) : (T ? Zero(V) : absym(x))
end
function skeleton(x::Chain{V}, v::Val{T}=Val{true}()) where {V,T}
    N, G, g = mdims(V), rank(x), 0
    ib = indexbasis(N, G)
    for k ∈ 1:binomial(N, G)
        if !iszero(x.v[k]) && (!T || count_ones(symmetricmask(V, ib[k], ib[k])[1]) > 0)
            g += skeleton(Single{V,G}(x.v[k], getbasis(V, ib[k])), v)
        end
    end
    return g
end
function skeleton(x::Multivector{V}, v::Val{T}=Val{true}()) where {V,T}
    N, g = mdims(V), 0
    for i ∈ 0:N
        R = binomsum(N, i)
        ib = indexbasis(N, i)
        for k ∈ 1:binomial(N, i)
            if !iszero(x.v[k+R]) && (!T || count_ones(symmetricmask(V, ib[k], ib[k])[1]) > 0)
                g += skeleton(Single{V,i}(x.v[k+R], getbasis(V, ib[k])), v)
            end
        end
    end
    return g
end

# mesh

export column, columns

column(t, i=1) = getindex.(value(t), i)
columns(t, i=1, j=mdims(Manifold(t))) = column.(Ref(value(t)), list(i, j))

rows(a::T) where {T<:AbstractMatrix} = getindex.(Ref(a), list(1, 3), :)

function pointset(e)
    mdims(Manifold(e)) == 1 && (return column(e))
    out = Int[]
    for i ∈ value(e)
        for k ∈ value(i)
            k ∉ out && push!(out, k)
        end
    end
    return out
end

export scalarfield, vectorfield, chainfield, rectanglefield # rectangle

function pointfield end
const vectorfield = pointfield # deprecate ?

chainfield(t, V=Manifold(t), W=V) = p -> V(vector(↓(↑((V ∪ Manifold(t))(p)) ⊘ t)))
function scalarfield(t, ϕ::T) where {T<:AbstractVector}
    M = Manifold(t)
    V = Manifold(M)
    P -> begin
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return ((Pi\P)⋅Chain{V,1}(ϕ[ti]))[1])
        end
        return 0.0
    end
end
function chainfield(t, ϕ::T) where {T<:AbstractVector}
    M = Manifold(t)
    V = Manifold(M)
    z = mdims(V) ≠ 4 ? Chain{V,1}(1.0, 0.0, 0.0) : Chain{V,1}(1.0, 0.0, 0.0, 0.0)
    P -> begin
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return (Pi \ P) ⋅ Chain{V,1}(ϕ[ti]))
        end
        return z
    end
end

function rectangle(p, nx=100, ny=nx)
    px, py = column(p, 2), column(p, 3)
    x = range(minimum(px), maximum(px), length=nx)
    y = range(minimum(py), maximum(py), length=ny)
    z = x' .+ im * y
    Chain{Manifold(p),1}.(1.0, real.(z), imag.(z))
end
rectanglefield(t, ϕ, nx=100, ny=nx) = chainfield(t, ϕ).(rectangle(points(t), nx, ny))

generate_products()
generate_products(Complex)
generate_products(Rational{BigInt}, :svec)
for Big ∈ (BigFloat, BigInt)
    generate_products(Big, :svec)
    generate_products(Complex{Big}, :svec)
end
generate_products(SymField, :svec, :($Sym.:∏), :($Sym.:∑), :($Sym.:-), :($Sym.conj))
function generate_derivation(m, t, d, c)
    @eval derive(n::$(:($m.$t)), b) = $m.$d(n, $m.$c(indexsymbol(Manifold(b), UInt(b))))
end
function generate_algebra(m, t, d=nothing, c=nothing)
    generate_products(:($m.$t), :svec, :($m.:*), :($m.:+), :($m.:-), :($m.conj), true)
    generate_inverses(m, t)
    !isnothing(d) && generate_derivation(m, t, d, c)
end
function generate_symbolic_methods(mod, symtype, methods_noargs, methods_args)
    for method ∈ methods_noargs
        @eval begin
            local apply_symbolic(x) = map(v -> typeof(v) == $mod.$symtype ? $mod.$method(v) : v, x)
            $mod.$method(x::T) where {T<:TensorGraded} = apply_symbolic(x)
            $mod.$method(x::T) where {T<:TensorMixed} = apply_symbolic(x)
        end
    end
    for method ∈ methods_args
        @eval begin
            local apply_symbolic(x, args...) = map(v -> typeof(v) == $mod.$symtype ? $mod.$method(v, args...) : v, x)
            $mod.$method(x::T, args...) where {T<:TensorGraded} = apply_symbolic(x, args...)
            $mod.$method(x::T, args...) where {T<:TensorMixed} = apply_symbolic(x, args...)
        end
    end
end

function __init__()
    @require Reduce = "93e0c654-6965-5f22-aba9-9c1ae6b3c259" begin
        *(a::Reduce.RExpr, b::Submanifold{V}) where {V} = Single{V}(a, b)
        *(a::Submanifold{V}, b::Reduce.RExpr) where {V} = Single{V}(b, a)
        *(a::Reduce.RExpr, b::Multivector{V,T}) where {V,T} = Multivector{V}(broadcast(Reduce.Algebra.:*, Ref(a), b.v))
        *(a::Multivector{V,T}, b::Reduce.RExpr) where {V,T} = Multivector{V}(broadcast(Reduce.Algebra.:*, a.v, Ref(b)))
        #*(a::Reduce.RExpr,b::MultiGrade{V}) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,Ref(a),b.v))
        #*(a::MultiGrade{V},b::Reduce.RExpr) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,a.v,Ref(b)))
        ∧(a::Reduce.RExpr, b::Reduce.RExpr) = Reduce.Algebra.:*(a, b)
        ∧(a::Reduce.RExpr, b::B) where {B<:TensorTerm{V,G}} where {V,G} = Single{V,G}(a, b)
        ∧(a::A, b::Reduce.RExpr) where {A<:TensorTerm{V,G}} where {V,G} = Single{V,G}(b, a)
        Leibniz.extend_field(Reduce.RExpr)
        parsym = (parsym..., Reduce.RExpr)
        for T ∈ (:RExpr, :Symbol, :Expr)
            @eval *(a::Reduce.$T, b::Chain{V,G,Any}) where {V,G} = (a * One(V)) * b
            @eval *(a::Chain{V,G,Any}, b::Reduce.$T) where {V,G} = a * (b * One(V))
            generate_inverses(:(Reduce.Algebra), T)
            generate_derivation(:(Reduce.Algebra), T, :df, :RExpr)
            #generate_algebra(:(Reduce.Algebra),T,:df,:RExpr)
        end
    end
    @require Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7" begin
        generate_algebra(:Symbolics, :Num)
        generate_symbolic_methods(:Symbolics, :Num, (:expand,), (:simplify, :substitute))
        *(a::Symbolics.Num, b::Multivector{V}) where {V} = Multivector{V}(a * b.v)
        *(a::Multivector{V}, b::Symbolics.Num) where {V} = Multivector{V}(a.v * b)
        *(a::Symbolics.Num, b::Chain{V,G}) where {V,G} = Chain{V,G}(a * b.v)
        *(a::Chain{V,G}, b::Symbolics.Num) where {V,G} = Chain{V,G}(a.v * b)
        *(a::Symbolics.Num, b::Single{V,G,B,T}) where {V,G,B,T<:Real} = Single{V}(a, b)
        *(a::Single{V,G,B,T}, b::Symbolics.Num) where {V,G,B,T<:Real} = Single{V}(b, a)
        Base.iszero(a::Single{V,G,B,Symbolics.Num}) where {V,G,B} = false
        isfixed(::Type{Symbolics.Num}) = true
        for op ∈ (:+, :-)
            for Term ∈ (:TensorGraded, :TensorMixed)
                @eval begin
                    $op(a::T, b::Symbolics.Num) where {T<:$Term} = $op(a, b * One(Manifold(a)))
                    $op(a::Symbolics.Num, b::T) where {T<:$Term} = $op(a * One(Manifold(b)), b)
                end
            end
        end
    end
    @require SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6" begin
        generate_algebra(:SymPy, :Sym, :diff, :symbols)
        generate_symbolic_methods(:SymPy, :Sym, (:expand, :factor, :together, :apart, :cancel), (:N, :subs))
        for T ∈ (Chain{V,G,SymPy.Sym} where {V,G},
            Multivector{V,SymPy.Sym} where {V},
            Single{V,G,SymPy.Sym} where {V,G})
            SymPy.collect(x::T, args...) = map(v -> typeof(v) == SymPy.Sym ? SymPy.collect(v, args...) : v, x)
        end
    end
    @require SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8" begin
        generate_algebra(:SymEngine, :Basic, :diff, :symbols)
        generate_symbolic_methods(:SymEngine, :Basic, (:expand, :N), (:subs, :evalf))
    end
    @require AbstractAlgebra = "c3fe647b-3220-5bb0-a1ea-a7954cac585d" generate_algebra(:AbstractAlgebra, :SetElem)
    @require GaloisFields = "8d0d7f98-d412-5cd4-8397-071c807280aa" generate_algebra(:GaloisFields, :AbstractGaloisField)
    @require LightGraphs = "093fc24a-ae57-5d10-9952-331d41423f4d" begin
        function LightGraphs.SimpleDiGraph(x::T, g=LightGraphs.SimpleDiGraph(rank(V))) where {T<:TensorTerm{V}} where {V}
            ind = (signbit(value(x)) ? reverse : identity)(indices(basis(x)))
            rank(x) == 2 ? LightGraphs.add_edge!(g, ind...) : LightGraphs.SimpleDiGraph(∂(x), g)
            return g
        end
        function LightGraphs.SimpleDiGraph(x::Chain{V}, g=LightGraphs.SimpleDiGraph(rank(V))) where {V}
            N, G = mdims(V), rank(x)
            ib = indexbasis(N, G)
            for k ∈ 1:binomial(N, G)
                if !iszero(x.v[k])
                    B = symmetricmask(V, ib[k], ib[k])[1]
                    count_ones(B) ≠ 1 && LightGraphs.SimpleDiGraph(x.v[k] * getbasis(V, B), g)
                end
            end
            return g
        end
        function LightGraphs.SimpleDiGraph(x::Multivector{V}, g=LightGraphs.SimpleDiGraph(rank(V))) where {V}
            N = mdims(V)
            for i ∈ 2:N
                R = binomsum(N, i)
                ib = indexbasis(N, i)
                for k ∈ 1:binomial(N, i)
                    if !iszero(x.v[k+R])
                        B = symmetricmask(V, ib[k], ib[k])[1]
                        count_ones(B) ≠ 1 && LightGraphs.SimpleDiGraph(x.v[k+R] * getbasis(V, B), g)
                    end
                end
            end
            return g
        end
    end
    #=@require GraphPlot="a2cc645c-3eea-5389-862e-a155d0052231"
    @require Compose="a81c6b42-2e10-5240-aca2-a61377ecd94b" begin
        import LightGraphs, GraphPlot, Cairo
        viewer = Base.Process(`$(haskey(ENV,"VIEWER") ? ENV["VIEWER"] : "xdg-open") simplex.pdf`,Ptr{Nothing}())
        function Compose.draw(img,x::T,l=layout=GraphPlot.circular_layout) where T<:TensorAlgebra
            Compose.draw(img,GraphPlot.gplot(LightGraphs.SimpleDiGraph(x),layout=l,nodelabel=collect(1:mdims(Manifold(x)))))
        end
        function graph(x,n="simplex.pdf",l=GraphPlot.circular_layout)
            cmd = `$(haskey(ENV,"VIEWER") ? ENV["VIEWER"] : "xdg-open") $n`
            global viewer
            viewer.cmd == cmd && kill(viewer)
            Compose.draw(Compose.PDF(n,16Compose.cm,16Compose.cm),x,l)
            viewer = run(cmd,(devnull,stdout,stderr),wait=false)
        end
    end=#
    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        StaticArrays.SMatrix(m::Chain{V,G,<:Chain{W,G}}) where {V,W,G} = StaticArrays.SMatrix{binomial(mdims(W), G),binomial(mdims(V), G)}(vcat(value.(value(m))...))
        Chain(m::StaticArrays.SMatrix{N,N}) where {N} = Chain{Submanifold(N),1}(m)
        Chain{V,G}(m::StaticArrays.SMatrix{N,N}) where {V,G,N} = Chain{V,G}(Chain{V,G}.(getindex.(Ref(m), :, StaticArrays.SVector{N}(1:N))))
        Chain{V,G,<:Chain{W,G}}(m::StaticArrays.SMatrix{M,N}) where {V,W,G,M,N} = Chain{V,G}(Chain{W,G}.(getindex.(Ref(m), :, StaticArrays.SVector{N}(1:N))))
    end
    @require Meshes = "eacbb407-ea5a-433e-ab97-5258b1ca43fa" begin
        Meshes.Point(t::Values) = Meshes.Point(Tuple(t.v))
        Meshes.Point(t::Variables) = Meshes.Point(Tuple(t.v))
        Base.convert(::Type{Meshes.Point}, t::T) where {T<:TensorTerm{V}} where {V} = Meshes.Point(value(Chain{V,valuetype(t)}(vector(t))))
        Base.convert(::Type{Meshes.Point}, t::T) where {T<:TensorTerm{V,0}} where {V} = Meshes.Point(zeros(valuetype(t), mdims(V))...)
        Base.convert(::Type{Meshes.Point}, t::T) where {T<:TensorAlgebra} = Meshes.Point(value(vector(t)))
        Base.convert(::Type{Meshes.Point}, t::Chain{V,G,T}) where {V,G,T} = G == 1 ? Meshes.Point(value(vector(t))) : Meshes.Point(zeros(T, mdims(V))...)
        Meshes.Point(t::T) where {T<:TensorAlgebra} = convert(Meshes.Point, t)
        pointpair(p, V) = Pair(Meshes.Point.(V.(value(p)))...)
        @pure ptype(::Meshes.Point{N,T} where {N}) where {T} = T
        export pointfield
        pointfield(t, V=Manifold(t), W=V) = p -> Meshes.Point(V(vector(↓(↑((V ∪ Manifold(t))(Chain{W,1,ptype(p)}(p.data))) ⊘ t))))
        function pointfield(t, ϕ::T) where {T<:AbstractVector}
            M = Manifold(t)
            V = Manifold(M)
            z = mdims(V) ≠ 4 ? Meshes(0.0, 0.0) : Meshes.Point(0.0, 0.0, 0.0)
            p -> begin
                P = Chain{V,1}(one(ptype(p)), p.data...)
                for i ∈ 1:length(t)
                    ti = value(t[i])
                    Pi = Chain{V,1}(M[ti])
                    P ∈ Pi && (return Meshes.Point((Pi \ P) ⋅ Chain{V,1}(ϕ[ti])))
                end
                return z
            end
        end
    end
    @require GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326" begin
        GeometryBasics.Point(t::Values) = GeometryBasics.Point(Tuple(t.v))
        GeometryBasics.Point(t::Variables) = GeometryBasics.Point(Tuple(t.v))
        Base.convert(::Type{GeometryBasics.Point}, t::T) where {T<:TensorTerm{V}} where {V} = convert(GeometryBasis.Point, Chain(t))
        Base.convert(::Type{GeometryBasics.Point}, t::T) where {T<:TensorTerm{V,0}} where {V} = GeometryBasics.Point(zeros(valuetype(t), mdims(V))...)
        Base.convert(::Type{GeometryBasics.Point}, t::T) where {T<:TensorAlgebra} = GeometryBasics.Point(value(vector(t)))
        Base.convert(::Type{GeometryBasics.Point}, t::T) where {T<:Couple} = GeometryBasics.Point(t.v.re, t.v.im)
        Base.convert(::Type{GeometryBasics.Point}, t::T) where {T<:Phasor} = GeometryBasics.Point(t.v.re, t.v.im)
        Base.convert(::Type{GeometryBasics.Point}, t::Chain{V,G,T}) where {V,G,T} = GeometryBasics.Point(value(t))
        GeometryBasics.Point(t::T) where {T<:TensorAlgebra} = convert(GeometryBasics.Point, t)
        pointpair(p, V) = Pair(GeometryBasics.Point.(V.(value(p)))...)
        @pure ptype(::GeometryBasics.Point{N,T} where {N}) where {T} = T
        export pointfield
        pointfield(t, V=Manifold(t), W=V) = p -> GeometryBasics.Point(V(vector(↓(↑((V ∪ Manifold(t))(Chain{W,1,ptype(p)}(p.data))) ⊘ t))))
        function pointfield(t, ϕ::T) where {T<:AbstractVector}
            M = Manifold(t)
            V = Manifold(M)
            z = mdims(V) ≠ 4 ? GeometryBasics(0.0, 0.0) : GeometryBasics.Point(0.0, 0.0, 0.0)
            p -> begin
                P = Chain{V,1}(one(ptype(p)), p.data...)
                for i ∈ 1:length(t)
                    ti = value(t[i])
                    Pi = Chain{V,1}(M[ti])
                    P ∈ Pi && (return GeometryBasics.Point((Pi \ P) ⋅ Chain{V,1}(ϕ[ti])))
                end
                return z
            end
        end
    end
    @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        Makie.convert_arguments(P::Makie.PointBased, a::Vector{<:Chain}) = Makie.convert_arguments(P, Makie.Point.(a))
        Makie.convert_single_argument(a::Chain) = convert_arguments(P, Point(a))
        Makie.arrows(p::Vector{<:Chain{V}}, v; args...) where {V} = Makie.arrows(GeometryBasics.Point.(↓(V).(p)), GeometryBasics.Point.(value(v)); args...)
        Makie.arrows!(p::Vector{<:Chain{V}}, v; args...) where {V} = Makie.arrows!(GeometryBasics.Point.(↓(V).(p)), GeometryBasics.Point.(value(v)); args...)
        Makie.lines(p::Vector{<:TensorAlgebra}; args...) = Makie.lines(GeometryBasics.Point.(p); args...)
        Makie.lines!(p::Vector{<:TensorAlgebra}; args...) = Makie.lines!(GeometryBasics.Point.(p); args...)
        Makie.lines(p::Vector{<:TensorTerm}; args...) = Makie.lines(value.(p); args...)
        Makie.lines!(p::Vector{<:TensorTerm}; args...) = Makie.lines!(value.(p); args...)
        Makie.lines(p::Vector{<:Chain{V,G,T,1} where {V,G,T}}; args...) = Makie.lines(getindex.(p, 1); args...)
        Makie.lines!(p::Vector{<:Chain{V,G,T,1} where {V,G,T}}; args...) = Makie.lines!(getindex.(p, 1); args...)
    end
    @require UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228" begin
        vandermonde(x::Chain, y, V, grid) = vandermonde(value(x), y, V, grid)
        function vandermonde(x, y, V, grid) # grid=384
            coef, xp, yp = vandermondeinterp(x, y, V, grid)
            p = UnicodePlots.scatterplot(x, value(y)) # overlay points
            display(UnicodePlots.lineplot!(p, xp, yp)) # plot polynomial
            println("||ϵ||: ", norm(approx.(x, Ref(value(coef))) .- value(y)))
            return coef # polynomial coefficients
        end
    end
end

#   ____  ____    ____   _____  _____ ___ ___   ____  ____   ____
#  /    T|    \  /    T / ___/ / ___/|   T   T /    T|    \ |    \
# Y   __j|  D  )Y  o  |(   \_ (   \_ | _   _ |Y  o  ||  _  Y|  _  Y
# |  T  ||    / |     | \__  T \__  T|  \_/  ||     ||  |  ||  |  |
# |  l_ ||    \ |  _  | /  \ | /  \ ||   |   ||  _  ||  |  ||  |  |
# |     ||  .  Y|  |  | \    | \    ||   |   ||  |  ||  |  ||  |  |
# l___,_jl__j\_jl__j__j  \___j  \___jl___j___jl__j__jl__j__jl__j__j

include("aliases.jl")

end # module
