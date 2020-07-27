module Grassmann

#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

using StaticArrays, SparseArrays, ComputedFieldTypes
using DirectSum, AbstractTensors, Requires

export ⊕, ℝ, @V_str, @S_str, @D_str, Manifold, SubManifold, Signature, DiagonalForm, value
export @basis, @basis_str, @dualbasis, @dualbasis_str, @mixedbasis, @mixedbasis_str, Λ

import Base: @pure, print, show, getindex, setindex!, promote_rule, ==, convert, ndims
import DirectSum: hasinf, hasorigin, dyadmode, dual, value, V0, ⊕, pre, vsn
import DirectSum: generate, basis, dual, getalgebra, getbasis, metric
import DirectSum: Bits, bit2int, doc2m, indexbits, indices, diffvars, diffmask, symmetricmask, indexstring, indexsymbol, combo

## cache

import DirectSum: algebra_limit, sparse_limit, cache_limit, fill_limit
import DirectSum: binomial, binomial_set, binomsum, binomsum_set, lowerbits, expandbits
import DirectSum: bladeindex, basisindex, indexbasis, indexbasis_set, loworder, intlog
import DirectSum: promote_type, mvec, svec, intlog, insert_expr

#=import Multivectors: TensorTerm, TensorGraded, Basis, MultiVector, SparseChain, MultiGrade, Fields, parval, parsym, Simplex, Chain, terms, valuetype, value_diff, basis, grade, order, bits, χ, gdims, rank, null, betti, isapprox, scalar, vector, volume, isscalar, isvector, subvert, mixed, choicevec, subindex, TensorMixed
import LinearAlgebra
import LinearAlgebra: I, UniformScaling
export UniformScaling, I=#

include("multivectors.jl")
include("parity.jl")
include("algebra.jl")
include("products.jl")
include("composite.jl")
include("forms.jl")

## fundamentals

export hyperplanes, points, TensorAlgebra

@pure hyperplanes(V::Manifold{N}) where N = map(n->UniformScaling{Bool}(false)*getbasis(V,1<<n),0:N-1-diffvars(V))

for M ∈ (:Signature,:DiagonalForm)
    @eval (::$M)(::S) where S<:SubAlgebra{V} where V = MultiVector{V,Int}(ones(Int,1<<ndims(V)))
end

points(f::F,r=-2π:0.0001:2π) where F<:Function = vector.(f.(r))

using Leibniz
import Leibniz: ∇, Δ, d # ∂
export ∇, Δ, ∂, d, δ, ↑, ↓

generate_products(:(Leibniz.Operator),:svec)
for T ∈ (:(Simplex{V}),:(Chain{V}),:(MultiVector{V}))
    @eval begin
        *(a::Derivation,b::$T) where V = V(a)*b
        *(a::$T,b::Derivation) where V = a*V(b)
    end
end
⊘(x::T,y::Derivation) where T<:TensorAlgebra{V} where V = x⊘V(y)
⊘(x::Derivation,y::T) where T<:TensorAlgebra{V} where V = V(x)⊘y

@pure function (V::Signature{N})(d::Leibniz.Derivation{T,O}) where {N,T,O}
    (O<1||diffvars(V)==0) && (return Chain{V,1,Int}(ones(Int,N)))
    G,D,C = grade(V),diffvars(V)==1,isdyadic(V)
    G2 = (C ? Int(G/2) : G)-1
    ∇ = sum([getbasis(V,1<<(D ? G : k+G))*getbasis(V,1<<k) for k ∈ 0:G2])
    isone(O) && (return ∇)
    x = (∇⋅∇)^div(isodd(O) ? O-1 : O,2)
    isodd(O) ? sum([(x*getbasis(V,1<<(k+G)))*getbasis(V,1<<k) for k ∈ 0:G2]) : x
end

@pure function (M::SubManifold{W,N})(d::Leibniz.Derivation{T,O}) where {W,N,T,O}
    V = isbasis(M) ? W : M
    (O<1||diffvars(V)==0) && (return Chain{V,1,Int}(ones(Int,N)))
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
    return SVector(out)
end

function boundary_null(t)
    d = gdims(t)
    r = boundary_rank(t,d)
    l = length(d)
    out = zeros(MVector{l,Int})
    for k ∈ 1:l-1
        @inbounds out[k] = d[k+1] - r[k]
    end
    return SVector(out)
end

"""
    betti(::TensorAlgebra)

Compute the Betti numbers.
"""
function betti(t::T) where T<:TensorAlgebra
    d = gdims(t)
    r = boundary_rank(t,d)
    l = length(d)-1
    out = zeros(MVector{l,Int})
    for k ∈ 1:l
        @inbounds out[k] = d[k+1] - r[k] - r[k+1]
    end
    return SVector(out)
end

@generated function ↑(ω::T) where T<:TensorAlgebra
    V = Manifold(ω)
    T<:SubManifold && !isbasis(ω) && (return DirectSum.supermanifold(V))
    !(hasinf(V)||hasorigin(V)) && (return :ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        :((($G.v∞*(one(valuetype(ω))/2))*ω^2+$G.v∅)+ω)
    else
        quote
            ω2 = ω^2
            iω2 = inv(ω2+1)
            (hasinf($V) ? $G.v∞ : $G.v∅)*(ω2-1)*iω2 + 2*iω2*ω
        end
    end
end
↑(ω::ChainBundle) = ω
function ↑(ω,b)
    ω2 = ω^2
    iω2 = inv(ω2+1)
    2*iω2*ω + (ω2-1)*iω2*b
end
function ↑(ω,p,m)
    ω2 = ω^2
    iω2 = inv(ω2+1)
    2*iω2*ω + (ω2-1)*iω2*p + (ω2+1)*iω2*m
end

@generated function ↓(ω::T) where T<:TensorAlgebra
    V,M = Manifold(ω),T<:SubManifold && !isbasis(ω)
    !(hasinf(V)||hasorigin(V)) && (return M ? V(2:ndims(V)) : :ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        M && (return ω(3:ndims(V)))
        :(inv(one(valuetype(ω))*$G.v∞∅)*($G.v∞∅∧ω)/(-ω⋅$G.v∞))
    else
        M && (return V(2:ndims(V)))
        quote
            b = hasinf($V) ? $G.v∞ : $G.v∅
            ((ω∧b)*b)/(1-b⋅ω)
        end
    end
end
↓(ω::ChainBundle) = ω(list(2,ndims(ω)))
↓(ω,b) = ((b∧ω)*b)/(1-ω⋅b)
↓(ω,∞,∅) = (m=∞∧∅;inv(m)*(m∧ω)/(-ω⋅∞))

## skeleton / subcomplex

export skeleton, 𝒫, collapse, subcomplex, chain, path

absym(t) = abs(t)
absym(t::SubManifold) = t
absym(t::T) where T<:TensorTerm{V,G} where {V,G} = Simplex{V,G}(absym(value(t)),basis(t))
absym(t::Chain{V,G,T}) where {V,G,T} = Chain{V,G}(absym.(value(t)))
absym(t::MultiVector{V,T}) where {V,T} = MultiVector{V}(absym.(value(t)))

collapse(a,b) = a⋅absym(∂(b))

function chain(t::S,::Val{T}=Val{true}()) where S<:TensorTerm{V} where {V,T}
    N,B,v = ndims(V),bits(basis(t)),value(t)
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
    B = bits(basis(x))
    count_ones(symmetricmask(V,B,B)[1])>0 ? absym(x)+skeleton(absym(∂(x)),v) : (T ? g_zero(V) : absym(x))
end
function skeleton(x::Chain{V},v::Val{T}=Val{true}()) where {V,T}
    N,G,g = ndims(V),rank(x),0
    ib = indexbasis(N,G)
    for k ∈ 1:binomial(N,G)
        if !iszero(x.v[k]) && (!T || count_ones(symmetricmask(V,ib[k],ib[k])[1])>0)
            g += skeleton(Simplex{V,G}(x.v[k],getbasis(V,ib[k])),v)
        end
    end
    return g
end
function skeleton(x::MultiVector{V},v::Val{T}=Val{true}()) where {V,T}
    N,g = ndims(V),0
    for i ∈ 0:N
        R = binomsum(N,i)
        ib = indexbasis(N,i)
        for k ∈ 1:binomial(N,i)
            if !iszero(x.v[k+R]) && (!T || count_ones(symmetricmask(V,ib[k],ib[k])[1])>0)
                g += skeleton(Simplex{V,i}(x.v[k+R],getbasis(V,ib[k])),v)
            end
        end
    end
    return g
end

# mesh

function initpoints(P)
    V = SubManifold(ℝ^(size(P,1)+1))
    p = [Chain{V,1,Float64}(vcat(1.0,P[:,k])) for k ∈ 1:size(P,2)]
end

function initpointsdata(P,E)
    p = ChainBundle(initpoints(P)); el = list(1,size(P,1))
    e = [Chain{↓(p),1,Int}(Int.(E[el,k])) for k ∈ 1:size(E,2)]
    return p,e
end

function initmeshdata(P,E,T)
    p,e = initpointsdata(P,E); tl = list(1,size(P,1)+1)
    t = [Chain{p,1,Int}(Int.(T[tl,k])) for k ∈ 1:size(T,2)]
    return p,ChainBundle(e),ChainBundle(t)
end

export pointset, edges, facets, adjacency, column, columns

column(t,i=1) = getindex.(value(t),i)
columns(t,i=1,j=ndims(Manifold(t))) = column.(Ref(value(t)),list(i,j))

function pointset(e)
    ndims(Manifold(e)) == 1 && (return column(e))
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
    np,N = length(points(t)),ndims(Manifold(t))
    A = spzeros(Int,np,np)
    for c ∈ combo(N,2)
        A += sparse(cols[c[1]],cols[c[2]],1,np,np)
    end
    return A
end

edges(t,cols::SVector) = edges(t,adjacency(t,cols))
function edges(t,adj=adjacency(t))
    ndims(t) == 2 && (return t)
    N = ndims(Manifold(t)); M = points(t)(list(N-1,N)...)
    f = findall(x->!iszero(x),LinearAlgebra.triu(adj))
    [Chain{M,1}(SVector{2,Int}(f[n].I)) for n ∈ 1:length(f)]
end

function facetsinterior(t::Vector{<:Chain{V}}) where V
    N = ndims(Manifold(t))-1
    W = V(list(2,N+1))
    N == 0 && (return [Chain{W,1}(list(2,1))],Int[])
    out = Chain{W,1,Int,N}[]
    bnd = Int[]
    for i ∈ t
        for w ∈ Chain{W,1}.(DirectSum.combinations(sort(value(i)),N))
            j = findfirst(isequal(w),out)
            isnothing(j) ? push!(out,w) : push!(bnd,j)
        end
    end
    return out,bnd
end
facets(t) = faces(t,Val(ndims(Manifold(t))-1))
facets(t,h) = faces(t,h,Val(ndims(Manifold(t))-1))
faces(t,v::Val) = faces(value(t),v)
faces(t,h,v,g=identity) = faces(value(t),h,v,g)
faces(t::Tuple,v,g=identity) = faces(t[1],t[2],v,g)
function faces(t::Vector{<:Chain{V}},::Val{N}) where {V,N}
    N == ndims(V) && (return t)
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
    vec = zeros(MVector{ndims(V),Int})
    val = N+1==ndims(V) ? ∂(Manifold(points(t))(list(1,N+1))(I)) : ones(SVector{binomial(ndims(V),N)})
    for i ∈ 1:length(t)
        vec[:] = value(t[i])
        par = DirectSum.indexparity!(vec)
        w = Chain{W,1}.(DirectSum.combinations(par[2],N))
        for k ∈ 1:binomial(ndims(V),N)
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
∂(t::SVector{N,<:Tuple}) where N = ∂.(t)
∂(t::SVector{N,<:Vector}) where N = ∂.(t)
∂(t::Tuple{Vector{<:Chain},Vector{Int}}) = ∂(t[1],t[2])
∂(t::Vector{<:Chain},u::Vector{Int}) = (f=facets(t,u); f[1][findall(x->!iszero(x),f[2])])
∂(t::Vector{<:Chain}) = ndims(t)≠3 ? (f=facetsinterior(t); f[1][setdiff(1:length(f[1]),f[2])]) : edges(t,adjacency(t).%2)
#∂(t::Vector{<:Chain}) = (f=facets(t,ones(Int,length(t))); f[1][findall(x->!iszero(x),f[2])])

skeleton(t::ChainBundle,v) = skeleton(value(t),v)
@inline (::Leibniz.Derivation)(x::Vector{<:Chain},v=Val{true}()) = skeleton(x,v)
@generated skeleton(t::Vector{<:Chain{V}},v) where V = :(faces.(Ref(t),Ref(ones(Int,length(t))),$(Val.(list(1,ndims(V)))),abs))
#@generated skeleton(t::Vector{<:Chain{V}},v) where V = :(faces.(Ref(t),$(Val.(list(1,ndims(V))))))

generate_products()
generate_products(Complex)
generate_products(Rational{BigInt},:svec)
for Big ∈ (BigFloat,BigInt)
    generate_products(Big,:svec)
    generate_products(Complex{Big},:svec)
end
generate_products(SymField,:svec,:($Sym.:∏),:($Sym.:∑),:($Sym.:-),:($Sym.conj))
function generate_derivation(m,t,d,c)
    @eval derive(n::$(:($m.$t)),b) = $m.$d(n,$m.$c(indexsymbol(Manifold(b),bits(b))))
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
        *(a::Reduce.RExpr,b::SubManifold{V}) where V = Simplex{V}(a,b)
        *(a::SubManifold{V},b::Reduce.RExpr) where V = Simplex{V}(b,a)
        *(a::Reduce.RExpr,b::MultiVector{V,T}) where {V,T} = MultiVector{V}(broadcast(Reduce.Algebra.:*,Ref(a),b.v))
        *(a::MultiVector{V,T},b::Reduce.RExpr) where {V,T} = MultiVector{V}(broadcast(Reduce.Algebra.:*,a.v,Ref(b)))
        *(a::Reduce.RExpr,b::MultiGrade{V}) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,Ref(a),b.v))
        *(a::MultiGrade{V},b::Reduce.RExpr) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,a.v,Ref(b)))
        ∧(a::Reduce.RExpr,b::Reduce.RExpr) = Reduce.Algebra.:*(a,b)
        ∧(a::Reduce.RExpr,b::B) where B<:TensorTerm{V,G} where {V,G} = Simplex{V,G}(a,b)
        ∧(a::A,b::Reduce.RExpr) where A<:TensorTerm{V,G} where {V,G} = Simplex{V,G}(b,a)
        DirectSum.extend_field(Reduce.RExpr)
        parsym = (parsym...,Reduce.RExpr)
        for T ∈ (:RExpr,:Symbol,:Expr)
            generate_inverses(:(Reduce.Algebra),T)
            generate_derivation(:(Reduce.Algebra),T,:df,:RExpr)
        end
    end
    @require SymPy="24249f21-da20-56a4-8eb1-6a02cf4ae2e6" begin
        generate_algebra(:SymPy,:Sym,:diff,:symbols)
        generate_symbolic_methods(:SymPy,:Sym, (:expand,:factor,:together,:apart,:cancel), (:N,:subs))
        for T ∈ (   Chain{V,G,SymPy.Sym} where {V,G},
                    MultiVector{V,SymPy.Sym} where V,
                    Simplex{V,G,SymPy.Sym} where {V,G} )
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
            N,G = ndims(V),rank(x)
            ib = indexbasis(N,G)
            for k ∈ 1:binomial(N,G)
                if !iszero(x.v[k])
                    B = symmetricmask(V,ib[k],ib[k])[1]
                    count_ones(B) ≠1 && LightGraphs.SimpleDiGraph(x.v[k]*getbasis(V,B),g)
                end
            end
            return g
        end
        function LightGraphs.SimpleDiGraph(x::MultiVector{V},g=LightGraphs.SimpleDiGraph(rank(V))) where V
           N = ndims(V)
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
    #@require GraphPlot="a2cc645c-3eea-5389-862e-a155d0052231"
    @require Compose="a81c6b42-2e10-5240-aca2-a61377ecd94b" begin
        import LightGraphs, GraphPlot, Cairo
        viewer = Base.Process(`$(haskey(ENV,"VIEWER") ? ENV["VIEWER"] : "xdg-open") simplex.pdf`,Ptr{Nothing}())
        function Compose.draw(img,x::T,l=layout=GraphPlot.circular_layout) where T<:TensorAlgebra
            Compose.draw(img,GraphPlot.gplot(LightGraphs.SimpleDiGraph(x),layout=l,nodelabel=collect(1:rank(Manifold(x)))))
        end
        function graph(x,n="simplex.pdf",l=GraphPlot.circular_layout)
            cmd = `$(haskey(ENV,"VIEWER") ? ENV["VIEWER"] : "xdg-open") $n`
            global viewer
            viewer.cmd == cmd && kill(viewer)
            Compose.draw(Compose.PDF(n,16Compose.cm,16Compose.cm),x,l)
            viewer = run(cmd,(devnull,stdout,stderr),wait=false)
        end
    end
    @require GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326" begin
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorTerm{V} where V = GeometryBasics.Point(value(Chain{V,valuetype(t)}(vector(t))))
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorTerm{V,0} where V = GeometryBasics.Point(zeros(valuetype(t),ndims(V))...)
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorAlgebra = GeometryBasics.Point(value(vector(t)))
        Base.convert(::Type{GeometryBasics.Point},t::Chain{V,G,T}) where {V,G,T} = G == 1 ? GeometryBasics.Point(value(vector(t))) : GeometryBasics.Point(zeros(T,ndims(V))...)
        GeometryBasics.Point(t::T) where T<:TensorAlgebra = convert(GeometryBasics.Point,t)
        pointpair(p,V) = Pair(GeometryBasics.Point.(V.(value(p)))...)
        function initmesh(m::GeometryBasics.Mesh)
            c,f = GeometryBasics.coordinates(m),GeometryBasics.faces(m)
            s = size(eltype(c))[1]+1; V = SubManifold(ℝ^s)
            n = size(eltype(f))[1]
            p = ChainBundle([Chain{V,1}(SVector{s,Float64}(1.0,k...)) for k ∈ c])
            M = s ≠ n ? p(list(s-n+1,s)) : p
            t = ChainBundle([Chain{M,1}(SVector{n,Int}(k)) for k ∈ f])
            return (p,ChainBundle(∂(t)),t)
        end
        @pure ptype(::GeometryBasics.Point{N,T} where N) where T = T
        export vectorfield, chainfield
        vectorfield(t,V=Manifold(t),W=V) = p->GeometryBasics.Point(V(vector(↓(↑((V∪Manifold(t))(Chain{W,1,ptype(p)}(p.data)))⊘t))))
        function chainfield(t,ϕ)
            M = Manifold(t)
            V = Manifold(M)
            z = ndims(V) ≠ 4 ? GeometryBasics(0.0,0.0) : GeometryBasics.Point(0.0,0.0,0.0)
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
    @require AbstractPlotting="537997a7-5e4e-5d89-9595-2241ea00577e" begin
        AbstractPlotting.arrows(p::ChainBundle{V},v;args...) where V = AbstractPlotting.arrows(value(p),v;args...)
        AbstractPlotting.arrows!(p::ChainBundle{V},v;args...) where V = AbstractPlotting.arrows!(value(p),v;args...)
        AbstractPlotting.arrows(p::Vector{<:Chain{V}},v;args...) where V = AbstractPlotting.arrows(GeometryBasics.Point.(↓(V).(p)),GeometryBasics.Point.(value(v));args...)
        AbstractPlotting.arrows!(p::Vector{<:Chain{V}},v;args...) where V = AbstractPlotting.arrows!(GeometryBasics.Point.(↓(V).(p)),GeometryBasics.Point.(value(v));args...)
        AbstractPlotting.scatter(p::ChainBundle,x;args...) = AbstractPlotting.scatter(submesh(p)[:,1],x;args...)
        AbstractPlotting.scatter!(p::ChainBundle,x;args...) = AbstractPlotting.scatter!(submesh(p)[:,1],x;args...)
        AbstractPlotting.scatter(p::Vector{<:Chain},x;args...) = AbstractPlotting.scatter(submesh(p)[:,1],x;args...)
        AbstractPlotting.scatter!(p::Vector{<:Chain},x;args...) = AbstractPlotting.scatter!(submesh(p)[:,1],x;args...)
        AbstractPlotting.scatter(p::ChainBundle;args...) = AbstractPlotting.scatter(submesh(p);args...)
        AbstractPlotting.scatter!(p::ChainBundle;args...) = AbstractPlotting.scatter!(submesh(p);args...)
        AbstractPlotting.scatter(p::Vector{<:Chain};args...) = AbstractPlotting.scatter(submesh(p);args...)
        AbstractPlotting.scatter!(p::Vector{<:Chain};args...) = AbstractPlotting.scatter!(submesh(p);args...)
        AbstractPlotting.lines(p::ChainBundle;args...) = AbstractPlotting.lines(value(p);args...)
        AbstractPlotting.lines!(p::ChainBundle;args...) = AbstractPlotting.lines!(value(p);args...)
        AbstractPlotting.lines(p::Vector{<:TensorAlgebra};args...) = AbstractPlotting.lines(GeometryBasics.Point.(p);args...)
        AbstractPlotting.lines!(p::Vector{<:TensorAlgebra};args...) = AbstractPlotting.lines!(GeometryBasics.Point.(p);args...)
        AbstractPlotting.linesegments(e::ChainBundle;args...) = AbstractPlotting.linesegments(value(e);args...)
        AbstractPlotting.linesegments!(e::ChainBundle;args...) = AbstractPlotting.linesegments!(value(e);args...)
        AbstractPlotting.linesegments(e::Vector{<:Chain};args...) = (p=points(e); AbstractPlotting.linesegments(pointpair.(p[e],↓(Manifold(p)));args...))
        AbstractPlotting.linesegments!(e::Vector{<:Chain};args...) = (p=points(e); AbstractPlotting.linesegments!(pointpair.(p[e],↓(Manifold(p)));args...))
        AbstractPlotting.wireframe(t::ChainBundle;args...) = AbstractPlotting.linesegments(edges(t);args...)
        AbstractPlotting.wireframe!(t::ChainBundle;args...) = AbstractPlotting.linesegments!(edges(t);args...)
        AbstractPlotting.wireframe(t::Vector{<:Chain};args...) = AbstractPlotting.linesegments(edges(t);args...)
        AbstractPlotting.wireframe!(t::Vector{<:Chain};args...) = AbstractPlotting.linesegments!(edges(t);args...)
        AbstractPlotting.mesh(t::ChainBundle;args...) = AbstractPlotting.mesh(points(t),t;args...)
        AbstractPlotting.mesh!(t::ChainBundle;args...) = AbstractPlotting.mesh!(points(t),t;args...)
        AbstractPlotting.mesh(t::Vector{<:Chain};args...) = AbstractPlotting.mesh(points(t),t;args...)
        AbstractPlotting.mesh!(t::Vector{<:Chain};args...) = AbstractPlotting.mesh!(points(t),t;args...)
        function AbstractPlotting.mesh(p::ChainBundle,t;args...)
            if ndims(p) == 2
                AbstractPlotting.plot(submesh(p)[:,1],args[:color])
            else
                AbstractPlotting.mesh(submesh(p),array(t);args...)
            end
        end
        function AbstractPlotting.mesh!(p::ChainBundle,t;args...)
            if ndims(p) == 2
                AbstractPlotting.plot!(submesh(p)[:,1],args[:color])
            else
                AbstractPlotting.mesh!(submesh(p),array(t);args...)
            end
        end
    end
    #@require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" nothing
    @require Delaunay="07eb4e4e-0c6d-46ef-bc4e-83d5e5d860a9" begin
        Delaunay.delaunay(p::ChainBundle) = Delaunay.delaunay(value(p))
        Delaunay.delaunay(p::Vector{<:Chain}) = initmesh(Delaunay.delaunay(Matrix(submesh(p))))
        initmesh(t::Delaunay.Triangulation) = initmeshdata(t.points',t.convex_hull',t.simplices)
    end
    @require MiniQhull="978d7f02-9e05-4691-894f-ae31a51d76ca" begin
        MiniQhull.delaunay(p::Vector{<:Chain},n=1:length(p)) = MiniQhull.delaunay(ChainBundle(p),n)
        function MiniQhull.delaunay(p::ChainBundle,n=1:length(p)); l = list(1,ndims(p))
            T = MiniQhull.delaunay(Matrix(submesh(length(n)==length(p) ? p : p[n])'))
            [Chain{p,1,Int}(getindex.(Ref(n),Int.(T[l,k]))) for k ∈ 1:size(T,2)]
        end
        #initmesh(p::ChainBundle) = (t=delaunay(p); (p,ChainBundle(∂(t)),ChainBundle(t)))
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
        initmesh(t::Triangulate.TriangulateIO) = initmeshdata(t.pointlist,t.segmentlist,t.trianglelist)
        #aran(area=0.001,angle=20) = "pa$(Printf.@sprintf("%.15f",area))q$(Printf.@sprintf("%.15f",angle))Q"
    end
    @require TetGen="c5d3f3f7-f850-59f6-8a2e-ffc6dc1317ea" begin
        function TetGen.TetgenIO(mesh::ChainBundle;
                marker = :markers, holes = TetGen.Point{3,Float64}[])
            TetGen.TetgenIO(value(mesh); marker=marker, holes=holes)
        end
        function TetGen.TetgenIO(mesh::Vector{<:Chain};
                marker = :markers, holes = TetGen.Point{3, Float64}[])
            f = TetGen.TriangleFace{Cint}.(value.(mesh))
            kw_args = Any[:facets => TetGen.metafree(f),:holes => holes]
            if hasproperty(f, marker)
                push!(kw_args, :facetmarkers => getproperty(f, marker))
            end
            pm = points(mesh); V = Manifold(pm)
            TetGen.TetgenIO(TetGen.Point.(↓(V).(value(pm))); kw_args...)
        end
        function initmesh(tio::TetGen.TetgenIO, command = "Qp")
            r = TetGen.tetrahedralize(tio, command); V = SubManifold(ℝ^4)
            p = ChainBundle([Chain{V,1}(SVector{4,Float64}(1.0,k...)) for k ∈ r.points])
            t = Chain{p,1}.(SVector{4,Int}.(r.tetrahedra))
            e = Chain{p(2,3,4),1}.(SVector{3,Int}.(r.trifaces))
            # Chain{p(2,3),1}.(SVector{2,Int}.(r.edges)
            return p,ChainBundle(e),ChainBundle(t)
        end
        function TetGen.tetrahedralize(mesh::ChainBundle, command = "Qp";
                marker = :markers, holes = TetGen.Point{3,Float64}[])
            TetGen.tetrahedralize(value(mesh), command; marker=marker, holes=holes)
        end
        function TetGen.tetrahedralize(mesh::Vector{<:Chain}, command = "Qp";
                marker = :markers, holes = TetGen.Point{3, Float64}[])
            initmesh(TetGen.TetgenIO(mesh;marker=marker,holes=holes),command)
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
            p,e,t = initmeshdata(P,E,T)
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

end # module
