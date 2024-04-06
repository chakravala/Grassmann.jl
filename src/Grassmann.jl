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
export hodge, wedge, vee, complement, dot, antidot

import Base: @pure, ==, isapprox
import Base: print, show, getindex, setindex!, promote_rule, convert, adjoint
import DirectSum: V0, ⊕, generate, basis, getalgebra, getbasis, dual, Zero, One, Zero, One
import Leibniz: hasinf, hasorigin, dyadmode, value, pre, vsn, metric, mdims
import Leibniz: bit2int, indexbits, indices, diffvars, diffmask
import Leibniz: symmetricmask, indexstring, indexsymbol, combo, digits_fast
import DirectSum: antimetric

import Leibniz: hasconformal, hasinf2origin, hasorigin2inf
import AbstractTensors: valuetype, scalar, isscalar, trivector, istrivector, ⊗, complement
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume

## cache

import Leibniz: algebra_limit, sparse_limit, cache_limit, fill_limit
import Leibniz: binomial, binomial_set, binomsum, binomsum_set, lowerbits, expandbits
import Leibniz: bladeindex, basisindex, indexbasis, indexbasis_set, loworder, intlog
import Leibniz: promote_type, mvec, svec, intlog, insert_expr, supermanifold

include("multivectors.jl")
include("parity.jl")
include("algebra.jl")
include("products.jl")
include("composite.jl")
include("forms.jl")

## fundamentals

export cayley, hyperplanes, points, TensorAlgebra

cayley(x) = (y=Vector(Λ(x).b); y*transpose(y))

@pure hyperplanes(V::Manifold) = map(n->UniformScaling{Bool}(false)*getbasis(V,1<<n),0:rank(V)-1-diffvars(V))

for M ∈ (:Signature,:DiagonalForm)
    @eval (::$M)(::S) where S<:SubAlgebra{V} where V = Multivector{V,Int}(ones(Int,1<<mdims(V)))
end

points(f::F,r=-2π:0.0001:2π) where F<:Function = vector.(f.(r))

export 𝕚,𝕛,𝕜
const 𝕚,𝕛,𝕜 = hyperplanes(ℝ3)

using Leibniz
import Leibniz: ∇, Δ, d, ∂
export ∇, Δ, ∂, d, δ, ↑, ↓

#generate_products(:(Leibniz.Operator),:svec)
for T ∈ (:(Chain{V}),:(Multivector{V}))
    @eval begin
        *(a::Derivation,b::$T) where V = V(a)*b
        *(a::$T,b::Derivation) where V = a*V(b)
    end
end
⊘(x::T,y::Derivation) where T<:TensorAlgebra{V} where V = x⊘V(y)
⊘(x::Derivation,y::T) where T<:TensorAlgebra{V} where V = V(x)⊘y

@pure function (V::Signature{N})(d::Leibniz.Derivation{T,O}) where {N,T,O}
    (O<1||diffvars(V)==0) && (return Chain{V,1,Int}(d.v.λ*ones(Values{N,Int})))
    G,D,C = grade(V),diffvars(V)==1,isdyadic(V)
    G2 = (C ? Int(G/2) : G)-1
    ∇ = sum([getbasis(V,1<<(D ? G : k+G))*getbasis(V,1<<k) for k ∈ 0:G2])
    isone(O) && (return ∇)
    x = (∇⋅∇)^div(isodd(O) ? O-1 : O,2)
    isodd(O) ? sum([(x*getbasis(V,1<<(k+G)))*getbasis(V,1<<k) for k ∈ 0:G2]) : x
end

@pure function (M::Submanifold{W,N})(d::Leibniz.Derivation{T,O}) where {W,N,T,O}
    V = isbasis(M) ? W : M
    (O<1||diffvars(V)==0) && (return Chain{V,1,Int}(d.v.λ*ones(Values{N,Int})))
    G,D,C = grade(V),diffvars(V)==1,isdyadic(V)
    G2 = (C ? Int(G/2) : G)-1
    ∇ = sum([getbasis(V,1<<(D ? G : k+G))*getbasis(V,1<<k) for k ∈ 0:G2])
    isone(O) && (return ∇)
    x = (∇⋅∇)^div(isodd(O) ? O-1 : O,2)
    isodd(O) ? sum([(x*getbasis(V,1<<(k+G)))*getbasis(V,1<<k) for k ∈ 0:G2]) : x
end

@generated ∂(ω::Chain{V,1,<:Chain{W,1}}) where {V,W} = :(∧(ω)⋅$(Λ(W).v1))
∂(ω::T) where T<:TensorAlgebra = ω⋅Manifold(ω)(∇)
d(ω::T) where T<:TensorAlgebra = Manifold(ω)(∇)∧ω
δ(ω::T) where T<:TensorAlgebra = -∂(ω)

function boundary_rank(t,d=gdims(t))
    out = gdims(∂(t))
    out[1] = 0
    for k ∈ 2:length(out)-1
        @inbounds out[k] = min(out[k],d[k+1])
    end
    return Values(out)
end

function boundary_null(t)
    d = gdims(t)
    r = boundary_rank(t,d)
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
function betti(t::T) where T<:TensorAlgebra
    d = gdims(t)
    r = boundary_rank(t,d)
    l = length(d)-1
    out = zeros(Variables{l,Int})
    for k ∈ 1:l
        @inbounds out[k] = d[k+1] - r[k] - r[k+1]
    end
    return Values(out)
end

@generated function ↑(ω::T) where T<:TensorAlgebra
    V = Manifold(ω)
    T<:Submanifold && !isbasis(ω) && (return Leibniz.supermanifold(V))
    !(hasinf(V)||hasorigin(V)) && (return :ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        :((($G.v∞*(one(valuetype(ω))/2))*((~ω)⋅ω)+$G.v∅)+ω)
        #:((($G.v∞*(one(valuetype(ω))/2))*ω^2+$G.v∅)+ω)
    else
        quote
            ω2 = (~ω)⋅ω # ω^2
            iω2 = inv(ω2+1)
            (hasinf($V) ? $G.v∞ : $G.v∅)*((ω2-1)*iω2) + (2iω2)*ω
        end
    end
end
↑(ω::ChainBundle) = ω
function ↑(ω,b)
    ω2 = (~ω)⋅ω # ω^2
    iω2 = inv(ω2+1)
    (2iω2)*ω + ((ω2-1)*iω2)*b
end
function ↑(ω,p,m)
    ω2 = scalar((~ω)⋅ω) # ω^2
    iω2 = inv(ω2+1)
    (2iω2)*ω + ((ω2-1)*iω2)*p + ((ω2+1)*iω2)*m
end

@generated function ↓(ω::T) where T<:TensorAlgebra
    V,M = Manifold(ω),T<:Submanifold && !isbasis(ω)
    !(hasinf(V)||hasorigin(V)) && (return M ? V(2:mdims(V)) : :ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        M && (return ω(3:mdims(V)))
        #:(inv(one(valuetype(ω))*$G.v∞∅)*($G.v∞∅∧ω)/(-ω⋅$G.v∞))
        :((($G.v∞∅∧ω)⋅inv(one(valuetype(ω))*~$G.v∞∅))/(-ω⋅$G.v∞))
    else
        M && (return V(2:mdims(V)))
        quote
            b = hasinf($V) ? $G.v∞ : $G.v∅
            (~(ω∧b)⋅b)/(1-b⋅ω) # ((ω∧b)*b)/(1-b⋅ω)
        end
    end
end
↓(ω::ChainBundle) = ω(list(2,mdims(ω)))
↓(ω,b) = (~(b∧ω)⋅b)/(1-ω⋅b) # ((b∧ω)*b)/(1-ω⋅b)
↓(ω,∞,∅) = (m=∞∧∅;((m∧ω)⋅~inv(m))/(-ω⋅∞)) #(m=∞∧∅;inv(m)*(m∧ω)/(-ω⋅∞))

## skeleton / subcomplex

export skeleton, 𝒫, collapse, subcomplex, chain, path

absym(t) = abs(t)
absym(t::Submanifold) = t
absym(t::T) where T<:TensorTerm{V,G} where {V,G} = Single{V,G}(absym(value(t)),basis(t))
absym(t::Chain{V,G,T}) where {V,G,T} = Chain{V,G}(absym.(value(t)))
absym(t::Multivector{V,T}) where {V,T} = Multivector{V}(absym.(value(t)))

collapse(a,b) = a⋅absym(∂(b))

function chain(t::S,::Val{T}=Val{true}()) where S<:TensorTerm{V} where {V,T}
    N,B,v = mdims(V),UInt(basis(t)),value(t)
    C = symmetricmask(V,B,B)[1]
    G = count_ones(C)
    G < 2 && (return t)
    out,ind = zeros(mvec(N,2,Int)), indices(C,N)
    if T || G == 2
        setblade!(out,G==2 ? v : -v,bit2int(indexbits(N,[ind[1],ind[end]])),Val{N}())
    end
    for k ∈ 2:G
        setblade!(out,v,bit2int(indexbits(N,ind[[k-1,k]])),Val{N}())
    end
    return Chain{V,2}(out)
end
path(t) = chain(t,Val{false}())

@inline (::Leibniz.Derivation)(x::T,v=Val{true}()) where T<:TensorAlgebra = skeleton(x,v)
𝒫(t::T) where T<:TensorAlgebra = Δ(t,Val{false}())
subcomplex(x::S,v=Val{true}()) where S<:TensorAlgebra = Δ(absym(∂(x)),v)
function skeleton(x::S,v::Val{T}=Val{true}()) where S<:TensorTerm{V} where {V,T}
    B = UInt(basis(x))
    count_ones(symmetricmask(V,B,B)[1])>0 ? absym(x)+skeleton(absym(∂(x)),v) : (T ? Zero(V) : absym(x))
end
function skeleton(x::Chain{V},v::Val{T}=Val{true}()) where {V,T}
    N,G,g = mdims(V),rank(x),0
    ib = indexbasis(N,G)
    for k ∈ 1:binomial(N,G)
        if !iszero(x.v[k]) && (!T || count_ones(symmetricmask(V,ib[k],ib[k])[1])>0)
            g += skeleton(Single{V,G}(x.v[k],getbasis(V,ib[k])),v)
        end
    end
    return g
end
function skeleton(x::Multivector{V},v::Val{T}=Val{true}()) where {V,T}
    N,g = mdims(V),0
    for i ∈ 0:N
        R = binomsum(N,i)
        ib = indexbasis(N,i)
        for k ∈ 1:binomial(N,i)
            if !iszero(x.v[k+R]) && (!T || count_ones(symmetricmask(V,ib[k],ib[k])[1])>0)
                g += skeleton(Single{V,i}(x.v[k+R],getbasis(V,ib[k])),v)
            end
        end
    end
    return g
end

# mesh

initpoints(P::T) where T<:AbstractVector = Chain{ℝ2,1}.(1.0,P)
initpoints(P::T) where T<:AbstractRange = Chain{ℝ2,1}.(1.0,P)
@generated function initpoints(P,::Val{n}=Val(size(P,1))) where n
    Expr(:.,:(Chain{$(Submanifold(n+1)),1}),
         Expr(:tuple,1.0,[:(P[$k,:]) for k ∈ 1:n]...))
end

function initpointsdata(P,E,N::Val{n}=Val(size(P,1))) where n
    p = ChainBundle(initpoints(P,N)); l = list(1,n)
    p,[Chain{↓(p),1}(Int.(E[l,k])) for k ∈ 1:size(E,2)]
end

function initmeshdata(P,E,T,N::Val{n}=Val(size(P,1))) where n
    p,e = initpointsdata(P,E,N); l = list(1,n+1)
    t = [Chain{p,1}(Int.(T[l,k])) for k ∈ 1:size(T,2)]
    return p,ChainBundle(e),ChainBundle(t)
end

export pointset, edges, facets, adjacency, column, columns

column(t,i=1) = getindex.(value(t),i)
columns(t,i=1,j=mdims(Manifold(t))) = column.(Ref(value(t)),list(i,j))

rows(a::T) where T<:AbstractMatrix = getindex.(Ref(a),list(1,3),:)

function pointset(e)
    mdims(Manifold(e)) == 1 && (return column(e))
    out = Int[]
    for i ∈ value(e)
        for k ∈ value(i)
            k ∉ out && push!(out,k)
        end
    end
    return out
end

antiadjacency(t::ChainBundle,cols=columns(t)) = (A = sparse(t,cols); A-transpose(A))
adjacency(t,cols=columns(t)) = (A = sparse(t,cols); A+transpose(A))
function SparseArrays.sparse(t,cols=columns(t))
    np,N = length(points(t)),mdims(Manifold(t))
    A = spzeros(Int,np,np)
    for c ∈ combo(N,2)
        A += @inbounds sparse(cols[c[1]],cols[c[2]],1,np,np)
    end
    return A
end

edges(t,cols::Values) = edges(t,adjacency(t,cols))
function edges(t,adj=adjacency(t))
    mdims(t) == 2 && (return t)
    N = mdims(Manifold(t)); M = points(t)(list(N-1,N)...)
    f = findall(x->!iszero(x),LinearAlgebra.triu(adj))
    [Chain{M,1}(Values{2,Int}(@inbounds f[n].I)) for n ∈ 1:length(f)]
end

function facetsinterior(t::Vector{<:Chain{V}}) where V
    N = mdims(Manifold(t))-1
    W = V(list(2,N+1))
    N == 0 && (return [Chain{W,1}(list(2,1))],Int[])
    out = Chain{W,1,Int,N}[]
    bnd = Int[]
    for i ∈ t
        for w ∈ Chain{W,1}.(Leibniz.combinations(sort(value(i)),N))
            j = findfirst(isequal(w),out)
            isnothing(j) ? push!(out,w) : push!(bnd,j)
        end
    end
    return out,bnd
end
facets(t) = faces(t,Val(mdims(Manifold(t))-1))
facets(t,h) = faces(t,h,Val(mdims(Manifold(t))-1))
faces(t,v::Val) = faces(value(t),v)
faces(t,h,v,g=identity) = faces(value(t),h,v,g)
faces(t::Tuple,v,g=identity) = faces(t[1],t[2],v,g)
function faces(t::Vector{<:Chain{V}},::Val{N}) where {V,N}
    N == mdims(V) && (return t)
    N == 2 && (return edges(t))
    W = V(list(2,N+1))
    N == 1 && (return Chain{W,1}.(pointset(t)))
    N == 0 && (return Chain{W,1}(list(2,1)))
    out = Chain{W,1,Int,N}[]
    for i ∈ value(t)
        for w ∈ Chain{W,1}.(DirectSum.combinations(sort(value(i)),N))
            w ∉ out && push!(out,w)
        end
    end
    return out
end
function faces(t::Vector{<:Chain{V}},h,::Val{N},g=identity) where {V,N}
    W = V(list(1,N))
    N == 0 && (return [Chain{W,1}(list(1,N))],Int[sum(h)])
    out = Chain{W,1,Int,N}[]
    bnd = Int[]
    vec = zeros(Variables{mdims(V),Int})
    val = N+1==mdims(V) ? ∂(Manifold(points(t))(list(1,N+1))(I)) : ones(Values{binomial(mdims(V),N)})
    for i ∈ 1:length(t)
        vec[:] = @inbounds value(t[i])
        par = DirectSum.indexparity!(vec)
        w = Chain{W,1}.(DirectSum.combinations(par[2],N))
        for k ∈ 1:binomial(mdims(V),N)
            j = findfirst(isequal(w[k]),out)
            v = h[i]*(par[1] ? -val[k] : val[k])
            if isnothing(j)
                push!(out,w[k])
                push!(bnd,g(v))
            else
                bnd[j] += g(v)
            end
        end
    end
    return out,bnd
end

∂(t::ChainBundle) = ∂(value(t))
∂(t::Values{N,<:Tuple}) where N = ∂.(t)
∂(t::Values{N,<:Vector}) where N = ∂.(t)
∂(t::Tuple{Vector{<:Chain},Vector{Int}}) = ∂(t[1],t[2])
∂(t::Vector{<:Chain},u::Vector{Int}) = (f=facets(t,u); f[1][findall(x->!iszero(x),f[2])])
∂(t::Vector{<:Chain}) = mdims(t)≠3 ? (f=facetsinterior(t); f[1][setdiff(1:length(f[1]),f[2])]) : edges(t,adjacency(t).%2)
#∂(t::Vector{<:Chain}) = (f=facets(t,ones(Int,length(t))); f[1][findall(x->!iszero(x),f[2])])

skeleton(t::ChainBundle,v) = skeleton(value(t),v)
@inline (::Leibniz.Derivation)(x::Vector{<:Chain},v=Val{true}()) = skeleton(x,v)
@generated skeleton(t::Vector{<:Chain{V}},v) where V = :(faces.(Ref(t),Ref(ones(Int,length(t))),$(Val.(list(1,mdims(V)))),abs))
#@generated skeleton(t::Vector{<:Chain{V}},v) where V = :(faces.(Ref(t),$(Val.(list(1,mdims(V))))))

export scalarfield, vectorfield, chainfield, rectanglefield # rectangle

function pointfield end; const vectorfield = pointfield # deprecate ?

chainfield(t,V=Manifold(t),W=V) = p->V(vector(↓(↑((V∪Manifold(t))(p))⊘t)))
function scalarfield(t,ϕ::T) where T<:AbstractVector
    M = Manifold(t)
    V = Manifold(M)
    P->begin
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return ((Pi\P)⋅Chain{V,1}(ϕ[ti]))[1])
        end
        return 0.0
    end
end
function chainfield(t,ϕ::T) where T<:AbstractVector
    M = Manifold(t)
    V = Manifold(M)
    z = mdims(V) ≠ 4 ? Chain{V,1}(1.0,0.0,0.0) : Chain{V,1}(1.0,0.0,0.0,0.0)
    P->begin
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return (Pi\P)⋅Chain{V,1}(ϕ[ti]))
        end
        return z
    end
end

function rectangle(p,nx=100,ny=nx)
    px,py = column(p,2),column(p,3)
    x = range(minimum(px),maximum(px),length=nx)
    y = range(minimum(py),maximum(py),length=ny)
    z = x' .+ im*y
    Chain{Manifold(p),1}.(1.0,real.(z),imag.(z))
end
rectanglefield(t,ϕ,nx=100,ny=nx) = chainfield(t,ϕ).(rectangle(points(t),nx,ny))

generate_products()
generate_products(Complex)
generate_products(Rational{BigInt},:svec)
for Big ∈ (BigFloat,BigInt)
    generate_products(Big,:svec)
    generate_products(Complex{Big},:svec)
end
generate_products(SymField,:svec,:($Sym.:∏),:($Sym.:∑),:($Sym.:-),:($Sym.conj))
function generate_derivation(m,t,d,c)
    @eval derive(n::$(:($m.$t)),b) = $m.$d(n,$m.$c(indexsymbol(Manifold(b),UInt(b))))
end
function generate_algebra(m,t,d=nothing,c=nothing)
    generate_products(:($m.$t),:svec,:($m.:*),:($m.:+),:($m.:-),:($m.conj),true)
    generate_inverses(m,t)
    !isnothing(d) && generate_derivation(m,t,d,c)
end
function generate_symbolic_methods(mod, symtype, methods_noargs, methods_args)
    for method ∈ methods_noargs
        @eval begin
            local apply_symbolic(x) = map(v -> typeof(v) == $mod.$symtype ? $mod.$method(v) : v, x)
            $mod.$method(x::T) where T<:TensorGraded = apply_symbolic(x)
            $mod.$method(x::T) where T<:TensorMixed = apply_symbolic(x)
        end
    end
    for method ∈ methods_args
        @eval begin
            local apply_symbolic(x, args...) = map(v -> typeof(v) == $mod.$symtype ? $mod.$method(v, args...) : v, x)
            $mod.$method(x::T, args...) where T<:TensorGraded = apply_symbolic(x, args...)
            $mod.$method(x::T, args...) where T<:TensorMixed = apply_symbolic(x, args...)
        end
    end
end

function __init__()
    @require Reduce="93e0c654-6965-5f22-aba9-9c1ae6b3c259" begin
        *(a::Reduce.RExpr,b::Submanifold{V}) where V = Single{V}(a,b)
        *(a::Submanifold{V},b::Reduce.RExpr) where V = Single{V}(b,a)
        *(a::Reduce.RExpr,b::Multivector{V,T}) where {V,T} = Multivector{V}(broadcast(Reduce.Algebra.:*,Ref(a),b.v))
        *(a::Multivector{V,T},b::Reduce.RExpr) where {V,T} = Multivector{V}(broadcast(Reduce.Algebra.:*,a.v,Ref(b)))
        #*(a::Reduce.RExpr,b::MultiGrade{V}) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,Ref(a),b.v))
        #*(a::MultiGrade{V},b::Reduce.RExpr) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,a.v,Ref(b)))
        ∧(a::Reduce.RExpr,b::Reduce.RExpr) = Reduce.Algebra.:*(a,b)
        ∧(a::Reduce.RExpr,b::B) where B<:TensorTerm{V,G} where {V,G} = Single{V,G}(a,b)
        ∧(a::A,b::Reduce.RExpr) where A<:TensorTerm{V,G} where {V,G} = Single{V,G}(b,a)
        Leibniz.extend_field(Reduce.RExpr)
        parsym = (parsym...,Reduce.RExpr)
        for T ∈ (:RExpr,:Symbol,:Expr)
            @eval *(a::Reduce.$T,b::Chain{V,G,Any}) where {V,G} = (a*One(V))*b
            @eval *(a::Chain{V,G,Any},b::Reduce.$T) where {V,G} = a*(b*One(V))
            generate_inverses(:(Reduce.Algebra),T)
            generate_derivation(:(Reduce.Algebra),T,:df,:RExpr)
            #generate_algebra(:(Reduce.Algebra),T,:df,:RExpr)
        end
    end
    @require Symbolics="0c5d862f-8b57-4792-8d23-62f2024744c7" begin
        generate_algebra(:Symbolics,:Num)
        generate_symbolic_methods(:Symbolics,:Num, (:expand,),(:simplify,:substitute))
        *(a::Symbolics.Num,b::Multivector{V}) where V = Multivector{V}(a*b.v)
        *(a::Multivector{V},b::Symbolics.Num) where V = Multivector{V}(a.v*b)
        *(a::Symbolics.Num,b::Chain{V,G}) where {V,G} = Chain{V,G}(a*b.v)
        *(a::Chain{V,G},b::Symbolics.Num) where {V,G} = Chain{V,G}(a.v*b)
        *(a::Symbolics.Num,b::Single{V,G,B,T}) where {V,G,B,T<:Real} = Single{V}(a,b)
        *(a::Single{V,G,B,T},b::Symbolics.Num) where {V,G,B,T<:Real} = Single{V}(b,a)
        Base.iszero(a::Single{V,G,B,Symbolics.Num}) where {V,G,B} = false
        isfixed(::Type{Symbolics.Num}) = true
        for op ∈ (:+,:-)
            for Term ∈ (:TensorGraded,:TensorMixed)
                @eval begin
                    $op(a::T,b::Symbolics.Num) where T<:$Term = $op(a,b*One(Manifold(a)))
                    $op(a::Symbolics.Num,b::T) where T<:$Term = $op(a*One(Manifold(b)),b)
                end
            end
        end
    end
    @require SymPy="24249f21-da20-56a4-8eb1-6a02cf4ae2e6" begin
        generate_algebra(:SymPy,:Sym,:diff,:symbols)
        generate_symbolic_methods(:SymPy,:Sym, (:expand,:factor,:together,:apart,:cancel), (:N,:subs))
        for T ∈ (   Chain{V,G,SymPy.Sym} where {V,G},
                    Multivector{V,SymPy.Sym} where V,
                    Single{V,G,SymPy.Sym} where {V,G} )
            SymPy.collect(x::T, args...) = map(v -> typeof(v) == SymPy.Sym ? SymPy.collect(v, args...) : v, x)
        end
    end
    @require SymEngine="123dc426-2d89-5057-bbad-38513e3affd8" begin
        generate_algebra(:SymEngine,:Basic,:diff,:symbols)
        generate_symbolic_methods(:SymEngine,:Basic, (:expand,:N), (:subs,:evalf))
    end
    @require AbstractAlgebra="c3fe647b-3220-5bb0-a1ea-a7954cac585d" generate_algebra(:AbstractAlgebra,:SetElem)
    @require GaloisFields="8d0d7f98-d412-5cd4-8397-071c807280aa" generate_algebra(:GaloisFields,:AbstractGaloisField)
    @require LightGraphs="093fc24a-ae57-5d10-9952-331d41423f4d" begin
        function LightGraphs.SimpleDiGraph(x::T,g=LightGraphs.SimpleDiGraph(rank(V))) where T<:TensorTerm{V} where V
           ind = (signbit(value(x)) ? reverse : identity)(indices(basis(x)))
           rank(x) == 2 ? LightGraphs.add_edge!(g,ind...) : LightGraphs.SimpleDiGraph(∂(x),g)
           return g
        end
        function LightGraphs.SimpleDiGraph(x::Chain{V},g=LightGraphs.SimpleDiGraph(rank(V))) where V
            N,G = mdims(V),rank(x)
            ib = indexbasis(N,G)
            for k ∈ 1:binomial(N,G)
                if !iszero(x.v[k])
                    B = symmetricmask(V,ib[k],ib[k])[1]
                    count_ones(B) ≠1 && LightGraphs.SimpleDiGraph(x.v[k]*getbasis(V,B),g)
                end
            end
            return g
        end
        function LightGraphs.SimpleDiGraph(x::Multivector{V},g=LightGraphs.SimpleDiGraph(rank(V))) where V
           N = mdims(V)
           for i ∈ 2:N
                R = binomsum(N,i)
                ib = indexbasis(N,i)
                for k ∈ 1:binomial(N,i)
                    if !iszero(x.v[k+R])
                        B = symmetricmask(V,ib[k],ib[k])[1]
                        count_ones(B) ≠ 1 && LightGraphs.SimpleDiGraph(x.v[k+R]*getbasis(V,B),g)
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
    @require StaticArrays="90137ffa-7385-5640-81b9-e52037218182" begin
        StaticArrays.SMatrix(m::Chain{V,G,<:Chain{W,G}}) where {V,W,G} = StaticArrays.SMatrix{binomial(mdims(W),G),binomial(mdims(V),G)}(vcat(value.(value(m))...))
        DyadicChain(m::StaticArrays.SMatrix{N,N}) where N = Chain{Submanifold(N),1}(m)
        Chain{V,G}(m::StaticArrays.SMatrix{N,N}) where {V,G,N} = Chain{V,G}(Chain{V,G}.(getindex.(Ref(m),:,StaticArrays.SVector{N}(1:N))))
        Chain{V,G,Chain{W,G}}(m::StaticArrays.SMatrix{M,N}) where {V,W,G,M,N} = Chain{V,G}(Chain{W,G}.(getindex.(Ref(m),:,StaticArrays.SVector{N}(1:N))))
        Base.exp(A::Chain{V,G,<:Chain{V,G}}) where {V,G} = Chain{V,G}(exp(StaticArrays.SMatrix(A)))
        Base.log(A::Chain{V,G,<:Chain{V,G}}) where {V,G} = Chain{V,G}(log(StaticArrays.SMatrix(A)))
        LinearAlgebra.eigvals(A::Chain{V,G,<:Chain{V,G}}) where {V,G} = Chain(Values{binomial(mdims(V),G)}(LinearAlgebra.eigvals(StaticArrays.SMatrix(A))))
        LinearAlgebra.eigvecs(A::Chain{V,G,<:Chain{V,G}}) where {V,G} = Chain(Chain.(Values{binomial(mdims(A),G)}.(getindex.(Ref(LinearAlgebra.eigvecs(StaticArrays.SMatrix(A))),:,list(1,binomial(mdims(A),G))))))
        function LinearAlgebra.eigen(A::Chain{V,G,<:Chain{V,G}}) where {V,G}
            E,N = eigen(StaticArrays.SMatrix(A)),binomial(mdims(V),G)
            e = Chain(Chain.(Values{N}.(getindex.(Ref(E.vectors),:,list(1,N)))))
            Proj(e,Chain(Values{N}(E.values)))
        end
    end
    @require Meshes = "eacbb407-ea5a-433e-ab97-5258b1ca43fa" begin
        Meshes.Point(t::Values) = Meshes.Point(Tuple(t.v))
        Meshes.Point(t::Variables) = Meshes.Point(Tuple(t.v))
        Base.convert(::Type{Meshes.Point},t::T) where T<:TensorTerm{V} where V = Meshes.Point(value(Chain{V,valuetype(t)}(vector(t))))
        Base.convert(::Type{Meshes.Point},t::T) where T<:TensorTerm{V,0} where V = Meshes.Point(zeros(valuetype(t),mdims(V))...)
        Base.convert(::Type{Meshes.Point},t::T) where T<:TensorAlgebra = Meshes.Point(value(vector(t)))
        Base.convert(::Type{Meshes.Point},t::Chain{V,G,T}) where {V,G,T} = G == 1 ? Meshes.Point(value(vector(t))) : Meshes.Point(zeros(T,mdims(V))...)
        Meshes.Point(t::T) where T<:TensorAlgebra = convert(Meshes.Point,t)
        pointpair(p,V) = Pair(Meshes.Point.(V.(value(p)))...)
        function initmesh(m::Meshes.SimpleMesh{N}) where N
            c,f = Meshes.vertices(m),m.topology.connec
            s = N+1; V = Submanifold(ℝ^s) # s
            n = length(f[1].indices)
            p = ChainBundle([Chain{V,1}(Values{s,Float64}(1.0,k.coords...)) for k ∈ c])
            M = s ≠ n ? p(list(s-n+1,s)) : p
            t = ChainBundle([Chain{M,1}(Values{n,Int}(k.indices)) for k ∈ f])
            return (p,ChainBundle(∂(t)),t)
        end
        @pure ptype(::Meshes.Point{N,T} where N) where T = T
        export pointfield
        pointfield(t,V=Manifold(t),W=V) = p->Meshes.Point(V(vector(↓(↑((V∪Manifold(t))(Chain{W,1,ptype(p)}(p.data)))⊘t))))
        function pointfield(t,ϕ::T) where T<:AbstractVector
            M = Manifold(t)
            V = Manifold(M)
            z = mdims(V) ≠ 4 ? Meshes(0.0,0.0) : Meshes.Point(0.0,0.0,0.0)
            p->begin
                P = Chain{V,1}(one(ptype(p)),p.data...)
                for i ∈ 1:length(t)
                    ti = value(t[i])
                    Pi = Chain{V,1}(M[ti])
                    P ∈ Pi && (return Meshes.Point((Pi\P)⋅Chain{V,1}(ϕ[ti])))
                end
                return z
            end
        end
    end
    @require GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326" begin
        GeometryBasics.Point(t::Values) = GeometryBasics.Point(Tuple(t.v))
        GeometryBasics.Point(t::Variables) = GeometryBasics.Point(Tuple(t.v))
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorTerm{V} where V = GeometryBasics.Point(value(Chain{V,valuetype(t)}(vector(t))))
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorTerm{V,0} where V = GeometryBasics.Point(zeros(valuetype(t),mdims(V))...)
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorAlgebra = GeometryBasics.Point(value(vector(t)))
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:Couple = GeometryBasics.Point(t.v.re,t.v.im)
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:Phasor = GeometryBasics.Point(t.v.re,t.v.im)
        Base.convert(::Type{GeometryBasics.Point},t::Chain{V,G,T}) where {V,G,T} = G == 1 ? GeometryBasics.Point(value(vector(t))) : GeometryBasics.Point(zeros(T,mdims(V))...)
        GeometryBasics.Point(t::T) where T<:TensorAlgebra = convert(GeometryBasics.Point,t)
        pointpair(p,V) = Pair(GeometryBasics.Point.(V.(value(p)))...)
        function initmesh(m::GeometryBasics.Mesh)
            c,f = GeometryBasics.coordinates(m),GeometryBasics.faces(m)
            s = size(eltype(c))[1]+1; V = Submanifold(ℝ^s) # s
            n = size(eltype(f))[1]
            p = ChainBundle([Chain{V,1}(Values{s,Float64}(1.0,k...)) for k ∈ c])
            M = s ≠ n ? p(list(s-n+1,s)) : p
            t = ChainBundle([Chain{M,1}(Values{n,Int}(k)) for k ∈ f])
            return (p,ChainBundle(∂(t)),t)
        end
        @pure ptype(::GeometryBasics.Point{N,T} where N) where T = T
        export pointfield
        pointfield(t,V=Manifold(t),W=V) = p->GeometryBasics.Point(V(vector(↓(↑((V∪Manifold(t))(Chain{W,1,ptype(p)}(p.data)))⊘t))))
        function pointfield(t,ϕ::T) where T<:AbstractVector
            M = Manifold(t)
            V = Manifold(M)
            z = mdims(V) ≠ 4 ? GeometryBasics(0.0,0.0) : GeometryBasics.Point(0.0,0.0,0.0)
            p->begin
                P = Chain{V,1}(one(ptype(p)),p.data...)
                for i ∈ 1:length(t)
                    ti = value(t[i])
                    Pi = Chain{V,1}(M[ti])
                    P ∈ Pi && (return GeometryBasics.Point((Pi\P)⋅Chain{V,1}(ϕ[ti])))
                end
                return z
            end
        end
    end
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        Makie.convert_arguments(P::Makie.PointBased, a::Vector{<:Chain}) = Makie.convert_arguments(P, Makie.Point.(a))
        Makie.convert_arguments(P::Makie.PointBased, a::ChainBundle) = Makie.convert_arguments(P, value(a))
        Makie.convert_single_argument(a::Chain) = convert_arguments(P,Point(a))
        Makie.arrows(p::ChainBundle{V},v;args...) where V = Makie.arrows(value(p),v;args...)
        Makie.arrows!(p::ChainBundle{V},v;args...) where V = Makie.arrows!(value(p),v;args...)
        Makie.arrows(p::Vector{<:Chain{V}},v;args...) where V = Makie.arrows(GeometryBasics.Point.(↓(V).(p)),GeometryBasics.Point.(value(v));args...)
        Makie.arrows!(p::Vector{<:Chain{V}},v;args...) where V = Makie.arrows!(GeometryBasics.Point.(↓(V).(p)),GeometryBasics.Point.(value(v));args...)
        Makie.scatter(p::ChainBundle,x;args...) = Makie.scatter(submesh(p)[:,1],x;args...)
        Makie.scatter!(p::ChainBundle,x;args...) = Makie.scatter!(submesh(p)[:,1],x;args...)
        Makie.scatter(p::Vector{<:Chain},x;args...) = Makie.scatter(submesh(p)[:,1],x;args...)
        Makie.scatter!(p::Vector{<:Chain},x;args...) = Makie.scatter!(submesh(p)[:,1],x;args...)
        Makie.scatter(p::ChainBundle;args...) = Makie.scatter(submesh(p);args...)
        Makie.scatter!(p::ChainBundle;args...) = Makie.scatter!(submesh(p);args...)
        Makie.scatter(p::Vector{<:Chain};args...) = Makie.scatter(submesh(p);args...)
        Makie.scatter!(p::Vector{<:Chain};args...) = Makie.scatter!(submesh(p);args...)
        Makie.lines(p::ChainBundle;args...) = Makie.lines(value(p);args...)
        Makie.lines!(p::ChainBundle;args...) = Makie.lines!(value(p);args...)
        Makie.lines(p::Vector{<:TensorAlgebra};args...) = Makie.lines(GeometryBasics.Point.(p);args...)
        Makie.lines!(p::Vector{<:TensorAlgebra};args...) = Makie.lines!(GeometryBasics.Point.(p);args...)
        Makie.lines(p::Vector{<:TensorTerm};args...) = Makie.lines(value.(p);args...)
        Makie.lines!(p::Vector{<:TensorTerm};args...) = Makie.lines!(value.(p);args...)
        Makie.lines(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines(getindex.(p,1);args...)
        Makie.lines!(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines!(getindex.(p,1);args...)
        Makie.linesegments(e::ChainBundle;args...) = Makie.linesegments(value(e);args...)
        Makie.linesegments!(e::ChainBundle;args...) = Makie.linesegments!(value(e);args...)
        Makie.linesegments(e::Vector{<:Chain};args...) = (p=points(e); Makie.linesegments(pointpair.(p[e],↓(Manifold(p)));args...))
        Makie.linesegments!(e::Vector{<:Chain};args...) = (p=points(e); Makie.linesegments!(pointpair.(p[e],↓(Manifold(p)));args...))
        Makie.wireframe(t::ChainBundle;args...) = Makie.linesegments(edges(t);args...)
        Makie.wireframe!(t::ChainBundle;args...) = Makie.linesegments!(edges(t);args...)
        Makie.wireframe(t::Vector{<:Chain};args...) = Makie.linesegments(edges(t);args...)
        Makie.wireframe!(t::Vector{<:Chain};args...) = Makie.linesegments!(edges(t);args...)
        Makie.mesh(t::ChainBundle;args...) = Makie.mesh(points(t),t;args...)
        Makie.mesh!(t::ChainBundle;args...) = Makie.mesh!(points(t),t;args...)
        Makie.mesh(t::Vector{<:Chain};args...) = Makie.mesh(points(t),t;args...)
        Makie.mesh!(t::Vector{<:Chain};args...) = Makie.mesh!(points(t),t;args...)
        function Makie.mesh(p::ChainBundle,t;args...)
            if mdims(p) == 2
                sm = submesh(p)[:,1]
                Makie.lines(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh(submesh(p),array(t);args...)
            end
        end
        function Makie.mesh!(p::ChainBundle,t;args...)
            if mdims(p) == 2
                sm = submesh(p)[:,1]
                Makie.lines!(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh!(submesh(p),array(t);args...)
            end
        end
    end
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" begin
        UnicodePlots.scatterplot(p::ChainBundle,x;args...) = UnicodePlots.scatterplot(submesh(p)[:,1],x;args...)
        UnicodePlots.scatterplot!(P,p::ChainBundle,x;args...) = UnicodePlots.scatterplot!(P,submesh(p)[:,1],x;args...)
        UnicodePlots.scatterplot(p::Vector{<:Chain},x;args...) = UnicodePlots.scatterplot(submesh(p)[:,1],x;args...)
        UnicodePlots.scatterplot!(P,p::Vector{<:Chain},x;args...) = UnicodePlots.scatterplot!(P,submesh(p)[:,1],x;args...)
        UnicodePlots.scatterplot(p::ChainBundle;args...) = !ispoints(Manifold(p)) ? (s=submesh(p); UnicodePlots.scatterplot(s[:,1],s[:,2];args...)) : UnicodePlots.scatterplot(means(p);args...)
        UnicodePlots.scatterplot!(P,p::ChainBundle;args...) = !ispoints(Manifold(p)) ? (s=submesh(p); UnicodePlots.scatterplot!(P,s[:,1],s[:,2];args...)) : UnicodePlots.scatterplot!(P,means(p);args...)
        UnicodePlots.scatterplot(p::Vector{<:Chain};args...) = !ispoints(Manifold(p)) ? (s=submesh(p); UnicodePlots.scatterplot(s[:,1],s[:,2];args...)) : UnicodePlots.scatterplot(means(p);args...)
        UnicodePlots.scatterplot!(P,p::Vector{<:Chain};args...) = !ispoints(Manifold(p)) ? (s=submesh(p); UnicodePlots.scatterplot!(P,s[:,1],s[:,2];args...)) : UnicodePlots.scatterplot!(P,means(p);args...)
        UnicodePlots.densityplot(p::ChainBundle;args...) = !ispoints(Manifold(p)) ? (s=submesh(p); UnicodePlots.densityplot(s[:,1],s[:,2];args...)) : UnicodePlots.densityplot(means(p);args...)
        UnicodePlots.densityplot!(P,p::ChainBundle;args...) = !ispoints(Manifold(p)) ? (s=submesh(p); UnicodePlots.densityplot!(P,s[:,1],s[:,2];args...)) : UnicodePlots.densityplot!(P,means(p);args...)
        UnicodePlots.densityplot(p::Vector{<:Chain};args...) = !ispoints(Manifold(p)) ? (s=submesh(p); UnicodePlots.densityplot(s[:,1],s[:,2];args...)) : UnicodePlots.densityplot(means(p);args...)
        UnicodePlots.densityplot!(P,p::Vector{<:Chain};args...) = !ispoints(Manifold(p)) ? (s=submesh(p); UnicodePlots.densityplot!(P,s[:,1],s[:,2];args...)) : UnicodePlots.densityplot!(P,means(p);args...)
        UnicodePlots.lineplot(p::ChainBundle;args...) = UnicodePlots.lineplot(value(p);args...)
        UnicodePlots.lineplot!(P,p::ChainBundle;args...) = UnicodePlots.lineplot!(P,value(p);args...)
        UnicodePlots.lineplot(p::Vector{<:TensorAlgebra};args...) = (s=submesh(p); UnicodePlots.lineplot(s[:,1],s[:,2];args...))
        UnicodePlots.lineplot!(P,p::Vector{<:TensorAlgebra};args...) = (s=submesh(p); UnicodePlots.lineplot!(P,s[:,1],s[:,2];args...))
        UnicodePlots.spy(p::ChainBundle) = UnicodePlots.spy(antiadjacency(p))
        UnicodePlots.spy(p::Vector{<:Chain}) = UnicodePlots.spy(antiadjacency(p))
        vandermonde(x::Chain,y,V,grid) = vandermonde(value(x),y,V,grid)
        function vandermonde(x,y,V,grid) # grid=384
            coef,xp,yp = vandermondeinterp(x,y,V,grid)
            p = UnicodePlots.scatterplot(x,value(y)) # overlay points
            display(UnicodePlots.lineplot!(p,xp,yp)) # plot polynomial
            println("||ϵ||: ",norm(approx.(x,Ref(value(coef))).-value(y)))
            return coef # polynomial coefficients
        end
    end
    @require Delaunay="07eb4e4e-0c6d-46ef-bc4e-83d5e5d860a9" begin
        Delaunay.delaunay(p::ChainBundle) = Delaunay.delaunay(value(p))
        Delaunay.delaunay(p::Vector{<:Chain}) = initmesh(Delaunay.delaunay(submesh(p)))
        initmesh(t::Delaunay.Triangulation) = initmeshdata(t.points',t.convex_hull',t.simplices')
    end
    @require QHull="a8468747-bd6f-53ef-9e5c-744dbc5c59e7" begin
        QHull.chull(p::Vector{<:Chain},n=1:length(p)) = QHull.chull(ChainBundle(p),n)
        function QHull.chull(p::ChainBundle,n=1:length(p)); l = list(1,mdims(p))
            T = QHull.chull(submesh(length(n)==length(p) ? p : p[n])); V = p(list(2,mdims(p)))
            [Chain{V,1}(getindex.(Ref(n),k)) for k ∈ T.simplices]
        end
        initmesh(t::Chull) = (p=ChainBundle(initpoints(t.points')); Chain{p(list(2,mdims(p))),1}.(t.simplices))
    end
    @require MiniQhull="978d7f02-9e05-4691-894f-ae31a51d76ca" begin
        MiniQhull.delaunay(p::Vector{<:Chain},n=1:length(p)) = MiniQhull.delaunay(ChainBundle(p),n)
        function MiniQhull.delaunay(p::ChainBundle,n=1:length(p)); l = list(1,mdims(p))
            T = MiniQhull.delaunay(Matrix(submesh(length(n)==length(p) ? p : p[n])'))
            [Chain{p,1,Int}(getindex.(Ref(n),Int.(T[l,k]))) for k ∈ 1:size(T,2)]
        end
    end
    @require Triangulate = "f7e6ffb2-c36d-4f8f-a77e-16e897189344" begin
        const triangle_cache = (Array{T,2} where T)[]
        function triangle(p::Array{T,2} where T,B)
            for k ∈ length(triangle_cache):B
                push!(triangle_cache,Array{Any,2}(undef,0,0))
            end
            triangle_cache[B] = p
        end
        function triangle(p::ChainBundle{V,G,T,B} where {V,G,T}) where B
            if length(triangle_cache)<B || isempty(triangle_cache[B])
                ap = array(p)'
                triangle(islocal(p) ? Cint.(ap) : ap[2:end,:],B)
            else
                return triangle_cache[B]
            end
        end
        function triangle(p::Vector{<:Chain{V,1,T} where V}) where T
            ap = array(p)'
            T<:Int ? Cint.(ap) : ap[2:end,:]
        end
        function Triangulate.TriangulateIO(e::Vector{<:Chain},h=nothing)
            triin=Triangulate.TriangulateIO()
            triin.pointlist=triangle(points(e))
            triin.segmentlist=triangle(e)
            !isnothing(h) && (triin.holelist=triangle(h))
            return triin
        end
        function Triangulate.triangulate(i,e::Vector{<:Chain};holes=nothing)
            initmesh(Triangulate.triangulate(i,Triangulate.TriangulateIO(e,holes))[1])
        end
        initmesh(t::Triangulate.TriangulateIO) = initmeshdata(t.pointlist,t.segmentlist,t.trianglelist,Val(2))
        #aran(area=0.001,angle=20) = "pa$(Printf.@sprintf("%.15f",area))q$(Printf.@sprintf("%.15f",angle))Q"
    end
    @require TetGen="c5d3f3f7-f850-59f6-8a2e-ffc6dc1317ea" begin
        function TetGen.JLTetGenIO(mesh::ChainBundle;
                marker = :markers, holes = TetGen.Point{3,Float64}[])
            TetGen.JLTetGenIO(value(mesh); marker=marker, holes=holes)
        end
        function TetGen.JLTetGenIO(mesh::Vector{<:Chain};
                marker = :markers, holes = TetGen.Point{3, Float64}[])
            f = TetGen.TriangleFace{Cint}.(value.(mesh))
            kw_args = Any[:facets => TetGen.metafree(f),:holes => holes]
            if hasproperty(f, marker)
                push!(kw_args, :facetmarkers => getproperty(f, marker))
            end
            pm = points(mesh); V = Manifold(pm)
            TetGen.JLTetGenIO(TetGen.Point.(↓(V).(value(pm))); kw_args...)
        end
        function initmesh(tio::TetGen.JLTetGenIO, command = "Qp")
            r = TetGen.tetrahedralize(tio, command); V = Submanifold(ℝ^4)
            p = ChainBundle([Chain{V,1}(Values{4,Float64}(1.0,k...)) for k ∈ r.points])
            t = Chain{p,1}.(Values{4,Int}.(r.tetrahedra))
            e = Chain{p(2,3,4),1}.(Values{3,Int}.(r.trifaces))
            # Chain{p(2,3),1}.(Values{2,Int}.(r.edges)
            return p,ChainBundle(e),ChainBundle(t)
        end
        function TetGen.tetrahedralize(mesh::ChainBundle, command = "Qp";
                marker = :markers, holes = TetGen.Point{3,Float64}[])
            TetGen.tetrahedralize(value(mesh), command; marker=marker, holes=holes)
        end
        function TetGen.tetrahedralize(mesh::Vector{<:Chain}, command = "Qp";
                marker = :markers, holes = TetGen.Point{3, Float64}[])
            initmesh(TetGen.JLTetGenIO(mesh;marker=marker,holes=holes),command)
        end
    end
    @require MATLAB="10e44e05-a98a-55b3-a45b-ba969058deb6" begin
        const matlab_cache = (Array{T,2} where T)[]
        function matlab(p::Array{T,2} where T,B)
            for k ∈ length(matlab_cache):B
                push!(matlab_cache,Array{Any,2}(undef,0,0))
            end
            matlab_cache[B] = p
        end
        function matlab(p::ChainBundle{V,G,T,B} where {V,G,T}) where B
            if length(matlab_cache)<B || isempty(matlab_cache[B])
                ap = array(p)'
                matlab(islocal(p) ? vcat(ap,ones(length(p))') : ap[2:end,:],B)
            else
                return matlab_cache[B]
            end
        end
        initmesh(g,args...) = initmeshall(g,args...)[1:3]
        initmeshall(g::Matrix{Int},args...) = initmeshall(Matrix{Float64}(g),args...)
        function initmeshall(g,args...)
            P,E,T = MATLAB.mxcall(:initmesh,3,g,args...)
            p,e,t = initmeshdata(P,E,T,Val(2))
            return (p,e,t,T,E,P)
        end
        function initmeshes(g,args...)
            p,e,t,T = initmeshall(g,args...)
            p,e,t,[Int(T[end,k]) for k ∈ 1:size(T,2)]
        end
        export initmeshes
        function refinemesh(g,args...)
            p,e,t,T,E,P = initmeshall(g,args...)
            matlab(P,bundle(p)); matlab(E,bundle(e)); matlab(T,bundle(t))
            return (g,p,e,t)
        end
        refinemesh3(g,p::ChainBundle,e,t,s...) = MATLAB.mxcall(:refinemesh,3,g,matlab(p),matlab(e),matlab(t),s...)
        refinemesh4(g,p::ChainBundle,e,t,s...) = MATLAB.mxcall(:refinemesh,4,g,matlab(p),matlab(e),matlab(t),s...)
        refinemesh(g,p::ChainBundle,e,t) = refinemesh3(g,p,e,t)
        refinemesh(g,p::ChainBundle,e,t,s::String) = refinemesh3(g,p,e,t,s)
        refinemesh(g,p::ChainBundle,e,t,η::Vector{Int}) = refinemesh3(g,p,e,t,float.(η))
        refinemesh(g,p::ChainBundle,e,t,η::Vector{Int},s::String) = refinemesh3(g,p,e,t,float.(η),s)
        refinemes(g,p::ChainBundle,e,t,u) = refinemesh4(g,p,e,t,u)
        refinemesh(g,p::ChainBundle,e,t,u,s::String) = refinemesh4(g,p,e,t,u,s)
        refinemesh(g,p::ChainBundle,e,t,u,η) = refinemesh4(g,p,e,t,u,float.(η))
        refinemesh(g,p::ChainBundle,e,t,u,η,s) = refinemesh4(g,p,e,t,u,float.(η),s)
        refinemesh!(g::Matrix{Int},p::ChainBundle,args...) = refinemesh!(Matrix{Float64}(g),p,args...)
        function refinemesh!(g,p::ChainBundle{V},e,t,s...) where V
            P,E,T = refinemesh(g,p,e,t,s...); l = size(P,1)+1
            matlab(P,bundle(p)); matlab(E,bundle(e)); matlab(T,bundle(t))
            submesh!(p); array!(t); el,tl = list(1,l-1),list(1,l)
            bundle_cache[bundle(p)] = [Chain{V,1,Float64}(vcat(1,P[:,k])) for k ∈ 1:size(P,2)]
            bundle_cache[bundle(e)] = [Chain{↓(p),1,Int}(Int.(E[el,k])) for k ∈ 1:size(E,2)]
            bundle_cache[bundle(t)] = [Chain{p,1,Int}(Int.(T[tl,k])) for k ∈ 1:size(T,2)]
            return (p,e,t)
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

end # module
