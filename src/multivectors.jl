
#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export TensorTerm, TensorGraded, TensorMixed, SubManifold, Simplex, MultiVector, SparseChain, MultiGrade, ChainBundle

import DirectSum: TensorGraded, TensorTerm, grade
abstract type TensorMixed{V} <: TensorAlgebra{V} end

# symbolic print types

import DirectSum: Fields, parval, mixed
parsym = (Symbol,parval...)

## pseudoscalar

import LinearAlgebra
import LinearAlgebra: I, UniformScaling
export UniformScaling, I

## Chain{V,G,𝕂}

@computed struct Chain{V,G,T} <: TensorGraded{V,G}
    v::SVector{binomial(ndims(V),G),T}
end

"""
    Chain{V,G,𝕂} <: TensorGraded{V,G}

Chain type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, scalar field `𝕂::Type`.
"""
Chain{V,G}(val::S) where {V,G,S<:AbstractVector{T}} where T = Chain{V,G,T}(val)
function Chain(val::T,v::SubManifold{V,G}) where {V,G,T}
    N = ndims(V)
    Chain{V,G}(setblade!(zeros(mvec(N,G,T)),val,bits(v),Val{N}()))
end
Chain(v::SubManifold{V,G}) where {V,G} = Chain(one(Int),v)
for var ∈ ((:V,:G,:T),(:V,:T),(:T,))
    @eval Chain{$(var...)}(v::SubManifold{V,G}) where {V,G,T} = Chain(one(T),v)
end
for var ∈ ((:V,:G,:T),(:V,:T),(:T,),())
    @eval begin
        Chain{$(var...)}(v::Simplex{V,G,B,T}) where {V,G,B,T} = Chain(v.v,basis(v))
        Chain{$(var...)}(v::Chain{V,G,T}) where {V,G,T} = Chain{V,G}(SVector{binomial(ndims(V),G),T}(v.v))
    end
end

export Chain
getindex(m::Chain,i::Int) = m.v[i]
getindex(m::Chain,i::UnitRange{Int}) = m.v[i]
setindex!(m::Chain{V,G,T} where {V,G},k::T,i::Int) where T = (m.v[i] = k)
Base.firstindex(m::Chain) = 1
@pure Base.lastindex(m::Chain{V,G}) where {V,G} = binomial(ndims(V),G)
@pure Base.length(m::Chain{V,G}) where {V,G} = binomial(ndims(V),G)

function (m::Chain{V,G,T})(i::Integer) where {V,G,T}
    Simplex{V,G,SubManifold{V}(indexbasis(ndims(V),G)[i]),T}(m[i])
end

function show(io::IO, m::Chain{V,G,T}) where {V,G,T}
    ib = indexbasis(ndims(V),G)
    @inbounds tmv = typeof(m.v[1])
    if |(broadcast(<:,tmv,parsym)...)
        par = (!(tmv<:TensorTerm)) && |(broadcast(<:,tmv,parval)...)
        @inbounds par ? print(io,"(",m.v[1],")") : print(io,m.v[1])
    else
        @inbounds print(io,m.v[1])
    end
    @inbounds DirectSum.printindices(io,V,ib[1])
    for k ∈ 2:length(ib)
        @inbounds mvs = m.v[k]
        tmv = typeof(mvs)
        if |(broadcast(<:,tmv,parsym)...)
            par = (!(tmv<:TensorTerm)) && |(broadcast(<:,tmv,parval)...)
            par ? print(io," + (",mvs,")") : print(io," + ",mvs)
        else
            sbm = signbit(mvs)
            print(io,sbm ? " - " : " + ",sbm ? abs(mvs) : mvs)
        end
        @inbounds DirectSum.printindices(io,V,ib[k])
    end
end

function ==(a::Chain{V,G},b::T) where T<:TensorTerm{V,G} where {V,G}
    i = bladeindex(ndims(V),bits(basis(b)))
    @inbounds a[i] == value(b) && (isempty(a[1:i-1]) ? true : (prod(a[1:i-1].==0) && prod(a[i+1:end].==0)))
end
==(a::T,b::Chain{V}) where T<:TensorTerm{V} where V = b==a
==(a::Chain{V},b::T) where T<:TensorTerm{V} where V = prod(0==value(b).==value(a))
for T ∈ Fields
    @eval begin
        ==(a::T,b::Chain{V,G} where V) where {T<:$T,G} = G==0 ? a==value(b)[1] : prod(0==a.==value(b))
        ==(a::Chain{V,G} where V,b::T) where {T<:$T,G} = G==0 ? value(a)[1]==b : prod(0==b.==value(a))
    end
end
==(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T,S} = prod(a.v .== b.v)
==(a::Chain{V},b::Chain{V}) where V = prod(0 .==value(a)) && prod(0 .== value(b))

"""
    ChainBundle{V,G,P} <: Manifold{V} <: TensorAlgebra{V}

Subsets of a bundle cross-section over a `Manifold` topology.
"""
struct ChainBundle{V,G,T,Points} <: Manifold{V}
    @pure ChainBundle{V,G,T,P}() where {V,G,T,P} = new{V,G,T,P}()
end

const bundle_cache = (Vector{Chain{V,G,T,X}} where {V,G,T,X})[]
function ChainBundle(c::Vector{Chain{V,G,T,X}} where X) where {V,G,T}
    push!(bundle_cache,c)
    ChainBundle{V,G,T,length(bundle_cache)}()
end
function clearbundlecache!()
    for P ∈ 1:length(bundle_cache)
        deletebundle!(P)
    end
end
@pure bundle(::ChainBundle{V,G,T,P} where {V,G,T}) where P = P
@pure deletebundle!(V) = deletebundle!(bundle(V))
@pure deletebundle!(P::Int) = (bundle_cache[P] = [Chain{ℝ^0,0,Int}(SVector(0))])
@pure isbundle(::ChainBundle) = true
@pure isbundle(t) = false
@pure ispoints(t) = isbundle(t) && rank(t) == 1 && !isbundle(Manifold(t))
@pure islocal(t) = isbundle(t) && rank(t)==1 && valuetype(t)==Int && ispoints(Manifold(t))
@pure iscell(t) = isbundle(t) && islocal(Manifold(t))

@pure Manifold(::ChainBundle{V}) where V = V
@pure LinearAlgebra.rank(M::ChainBundle{V,G} where V) where G = G
@pure grade(::ChainBundle{V}) where V = grade(V)
@pure Base.ndims(::ChainBundle{V}) where V = ndims(V)
@pure Base.ndims(::Vector{Chain{V,G,T,X}} where {G,T,X}) where V = ndims(V)
@pure Base.parent(::ChainBundle{V}) where V = isbundle(V) ? parent(V) : V
@pure DirectSum.supermanifold(m::ChainBundle{V}) where V = V
@pure points(t::ChainBundle{p}) where p = isbundle(p) ? p : DirectSum.supermanifold(p)

value(c::Vector{Chain{V,G,T,X}} where {V,G,T,X}) = c
value(::ChainBundle{V,G,T,P}) where {V,G,T,P} = bundle_cache[P]::(Vector{Chain{V,G,T,binomial(ndims(V),G)}})
AbstractTensors.valuetype(::ChainBundle{V,G,T} where {V,G}) where T = T

getindex(m::ChainBundle,i::I) where I<:Integer = getindex(value(m),i)
getindex(m::ChainBundle,i) = getindex(value(m),i)
setindex!(m::ChainBundle,k,i) = setindex!(value(m),k,i)
Base.firstindex(m::ChainBundle) = 1
Base.lastindex(m::ChainBundle) = length(value(m))
Base.length(m::ChainBundle) = length(value(m))
Base.resize!(m::ChainBundle,n::Int) = resize!(value(m),n)

Base.display(m::ChainBundle) = (print(showbundle(m));display(value(m)))
Base.show(io::IO,m::ChainBundle) = print(io,showbundle(m),length(m))
@pure showbundle(m::ChainBundle{V,G}) where {V,G} = "$(iscell(m) ? 'C' : islocal(m) ? 'I' : 'Λ')$(DirectSum.sups[G])$V×"

## MultiVector{V,𝕂}

@computed struct MultiVector{V,T} <: TensorMixed{V}
    v::SVector{1<<ndims(V),T}
end

"""
    MultiVector{V,𝕂} <: TensorMixed{V} <: TensorAlgebra{V}

Chain type with pseudoscalar `V::Manifold` and scalar field `𝕂::Type`.
"""
MultiVector{V}(v::S) where {V,S<:AbstractVector{T}} where T = MultiVector{V,T}(v)
function MultiVector(val::T,v::SubManifold{V,G}) where {V,T,G}
    N = ndims(V)
    MultiVector{V}(setmulti!(zeros(mvec(N,T)),val,bits(v),Val{N}()))
end
MultiVector(v::SubManifold{V,G}) where {V,G} = MultiVector(one(Int),v)
for var ∈ ((:V,:T),(:T,))
    @eval function MultiVector{$(var...)}(v::SubManifold{V,G}) where {V,T,G}
        return MultiVector(one(T),v)
    end
end
for var ∈ ((:V,:T),(:T,),())
    @eval begin
        function MultiVector{$(var...)}(v::Simplex{V,G,B,T}) where {V,G,B,T}
            return MultiVector(v.v,basis(v))
        end
        function MultiVector{$(var...)}(v::Chain{V,G,T}) where {V,G,T}
            N = ndims(V)
            out = zeros(mvec(N,T))
            r = binomsum(N,G)
            @inbounds out[r+1:r+binomial(N,G)] = v.v
            return MultiVector{V}(out)
        end
    end
end

function getindex(m::MultiVector{V,T},i::Int) where {V,T}
    N = ndims(V)
    0 <= i <= N || throw(BoundsError(m, i))
    r = binomsum(N,i)
    return @view m.v[r+1:r+binomial(N,i)]
end
getindex(m::MultiVector,i::Int,j::Int) = m[i][j]
setindex!(m::MultiVector{V,T} where V,k::T,i::Int,j::Int) where T = (m[i][j] = k)
Base.firstindex(m::MultiVector) = 0
Base.lastindex(m::MultiVector{V,T} where T) where V = ndims(V)

(m::MultiVector{V,T})(g::Int) where {T,V,B} = m(Val{g}())
function (m::MultiVector{V,T})(::Val{g}) where {V,T,g,B}
    Chain{V,g,T}(m[g])
end
function (m::MultiVector{V,T})(g::Int,i::Int) where {V,T,B}
    Simplex{V,g,Basis{V}(indexbasis(ndims(V),g)[i]),T}(m[g][i])
end

function show(io::IO, m::MultiVector{V,T}) where {V,T}
    N = ndims(V)
    basis_count = true
    print(io,m[0][1])
    bs = binomsum_set(N)
    for i ∈ 2:N+1
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            @inbounds s = k+bs[i]
            @inbounds mvs = m.v[s]
            @inbounds if mvs ≠ 0
                tmv = typeof(mvs)
                if |(broadcast(<:,tmv,parsym)...)
                    par = (!(tmv<:TensorTerm)) && |(broadcast(<:,tmv,parval)...)
                    par ? print(io," + (",mvs,")") : print(io," + ",mvs)
                else
                    sba = signbit(mvs)
                    print(io,sba ? " - " : " + ",sba ? abs(mvs) : mvs)
                end
                @inbounds DirectSum.printindices(io,V,ib[k])
                basis_count = false
            end
        end
    end
    basis_count && print(io,pre[1]*'⃖')
end

==(a::MultiVector{V,T},b::MultiVector{V,S}) where {V,T,S} = prod(a.v .== b.v)
function ==(a::MultiVector{V,T},b::Chain{V,G,S}) where {V,T,G,S}
    N = ndims(V)
    r,R = binomsum(N,G), N≠G ? binomsum(N,G+1) : 2^N+1
    @inbounds prod(a[G] .== b.v) && prod(a.v[1:r] .== 0) && prod(a.v[R+1:end] .== 0)
end
==(a::Chain{V,G,T},b::MultiVector{V,S}) where {V,S,G,T} = b == a
function ==(a::MultiVector{V,S} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
    i = basisindex(ndims(V),bits(basis(b)))
    @inbounds a.v[i] == value(b) && prod(a.v[1:i-1] .== 0) && prod(a.v[i+1:end] .== 0)
end
==(a::T,b::MultiVector{V,S} where S) where T<:TensorTerm{V} where V = b==a
for T ∈ Fields
    @eval begin
        ==(a::T,b::MultiVector{V,S,G} where {V,S}) where {T<:$T,G} = (v=value(b);(a==v[1])*prod(0 .== v[2:end]))
        ==(a::MultiVector{V,S,G} where {V,S},b::T) where {T<:$T,G} = b == a
    end
end

## SparseChain{V,G}

"""
    SparseChain{V,G} <: TensorGraded{V,G}

Sparse chain type with pseudoscalar `V::Manifold` and grade/rank `G::Int`.
"""
struct SparseChain{V,G,T} <: TensorGraded{V,G}
    v::SparseVector{T,Int}
end

SparseChain{V,G}(v::SparseVector{T,Int}) where {V,G,T} = SparseChain{V,G,T}(v)
SparseChain{V}(v::Vector{<:TensorTerm{V,G}}) where {V,G} = SparseChain{V,G}(sparsevec(bladeindex.(ndims(V),bits.(v)),value.(v),ndims(V)))
SparseChain(v::T) where T <: TensorTerm = v

for Vec ∈ (:(SVector{L,T}),:(SubArray{T,1,SVector{L,T}}))
    @eval function chainvalues(V::Manifold{N},m::$Vec,::Val{G}) where {N,G,L,T}
        bng = binomial(N,G)
        G∉(0,N) && sum(m .== 0)/bng < fill_limit && (return Chain{V,G,T}(m))
        out = spzeros(T,bng)
        for i ∈ 1:bng
            @inbounds m[i] ≠ 0 && (out[i] = m[i])
        end
        length(out.nzval)≠1 ? SparseChain{V,G}(out) : Simplex{V,G,getbasis(V,@inbounds indexbasis(N,G)[out.nzind[1]]),T}(@inbounds m[out.nzind[1]])
    end
end

SparseChain{V,G}(m::Chain{V,G,T}) where {V,G,T} = chainvalues(V,value(m),Val{G}())
SparseChain{V}(m::Chain{V,G,T}) where {V,G,T} = SparseChain{V,G}(m)
SparseChain(m::Chain{V,G,T}) where {V,G,T} = SparseChain{V,G}(m)

function show(io::IO, m::SparseChain{V,G,T}) where {V,G,T}
    ib = indexbasis(ndims(V),G)
    o = m.v.nzind[1]
    @inbounds if T == Any && typeof(m.v[o]) ∈ parsym
        @inbounds tmv = typeof(m.v[o])
        par = (!(tmv<:TensorTerm)) && |(broadcast(<:,tmv,parval)...)
        @inbounds par ? print(io,m.v[o]) : print(io,"(",m.v[o],")")
    else
        @inbounds print(io,m.v[o])
    end
    @inbounds DirectSum.printindices(io,V,ib[o])
    length(m.v.nzind)>1 && for k ∈ m.v.nzind[2:end]
        @inbounds mvs = m.v[k]
        tmv = typeof(mvs)
        if |(broadcast(<:,tmv,parsym)...)
            par = (!(tmv<:TensorTerm)) && |(broadcast(<:,tmv,parval)...)
            par ? print(io," + (",mvs,")") : print(io," + ",mvs)
        else
            sbm = signbit(mvs)
            print(io,sbm ? " - " : " + ",sbm ? abs(mvs) : mvs)
        end
        @inbounds DirectSum.printindices(io,V,ib[k])
    end
end

==(a::SparseChain{V,G},b::SparseChain{V,G}) where {V,G} = prod(terms(a) .== terms(b))
==(a::SparseChain{V},b::SparseChain{V}) where V = iszero(a) && iszero(b)
==(a::SparseChain{V},b::T) where T<:TensorTerm{V} where V = false
==(a::T,b::SparseChain{V}) where T<:TensorTerm{V} where V = false

## MultiGrade{V,G}

@computed struct MultiGrade{V,G} <: TensorMixed{V}
    v::SVector{count_ones(G),TensorGraded{V}}
end

@doc """
    MultiGrade{V,G} <: TensorMixed{V} <: TensorAlgebra{V}

Sparse multivector type with pseudoscalar `V::Manifold` and grade encoding `G::UInt64`.
""" MultiGrade

terms(v::MultiGrade) = v.v
value(v::MultiGrade) = reduce(vcat,value.(terms(v)))

MultiGrade{V}(v::Vector{T}) where T<:TensorGraded{V} where V = MultiGrade{V,|(UInt(1).<<rank.(v)...)}(SVector(v...))
MultiGrade(v::Vector{T}) where T<:TensorGraded{V} where V = MultiGrade{V}(v)
MultiGrade(m::T) where T<:TensorAlgebra = m
MultiGrade(m::Chain{T,V,G}) where {T,V,G} = chainvalues(V,value(m),Val{G}())

function MultiGrade(m::MultiVector{V,T}) where {V,T}
    N = ndims(V)
    sum(m.v .== 0)/(1<<N) < fill_limit && (return m)
    out = zeros(SizedArray{Tuple{N+1},TensorGraded{V},1,1})
    G = zero(UInt)
    for i ∈ 0:N
        @inbounds !prod(m[i].==0) && (G|=UInt(1)<<i;out[i+1]=chainvalues(V,m[i],Val{i}()))
    end
    cG = count_ones(G)
    return cG≠1 ? MultiGrade{V,G}(SVector{cG,T}(out[indices(G,N+1)]...)) : out[1]
end

function show(io::IO, m::MultiGrade{V,G}) where {V,G}
    t = terms(m)
    isempty(t) && print(io,zero(V))
    for k ∈ 1:count_ones(G)
        k ≠ 1 && print(io," + ")
        print(io,t[k])
    end
end

#=function MultiVector{V,T}(v::MultiGrade{V}) where {V,T}
    N = ndims(V)
    sigcheck(v.s,V)
    g = rank.(v.v)
    out = zeros(mvec(N,T))
    for k ∈ 1:length(v.v)
        @inbounds (val,b) = typeof(v.v[k]) <: Basis ? (one(T),v.v[k]) : (v.v[k].v,basis(v.v[k]))
        setmulti!(out,convert(T,val),bits(b),Val{N}())
    end
    return MultiVector{V}(out)
end

MultiVector{V}(v::MultiGrade{V}) where V = MultiVector{V}(v)
MultiVector(v::MultiGrade{V}) where V = MultiVector{V,promote_type(typeval.(v.v)...)}(v)=#

==(a::MultiGrade{V,G},b::MultiGrade{V,G}) where {V,G} = prod(terms(a) .== terms(b))

## Generic

import Base: isinf, isapprox
import DirectSum: bits, basis, grade, order
import AbstractTensors: value, valuetype, scalar, isscalar, involute, unit, even, odd
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume, ⋆
import LinearAlgebra: rank, norm
export basis, grade, hasinf, hasorigin, isorigin, scalar, norm, gdims, betti, χ
export valuetype, scalar, isscalar, vector, isvector, indices

const VBV = Union{Simplex,Chain,MultiVector}

valuetype(t::MultiGrade) = promote_type(valuetype.(terms(t))...)
@pure valuetype(t::SparseChain{V,G,T} where {V,G}) where T = T
@pure valuetype(::MultiVector{V,T} where V) where T = T
@pure valuetype(::Chain{V,G,T} where {V,G}) where T = T
@inline value(m::MultiGrade,T) = m
for T ∈ (:Chain,:MultiVector)
    @eval @inline value(m::$T,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
end
@inline value(m::SparseChain,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(SparseVector{T,Int},m.v) : m.v
@inline value_diff(m::Chain{V,0} where V) = (v=value(m)[1];istensor(v) ? v : m)
@inline value_diff(m::Chain) = m

Base.isapprox(a::S,b::T) where {S<:MultiVector,T<:MultiVector} = Manifold(a)==Manifold(b) && DirectSum.:≈(value(a),value(b))

@inline scalar(t::Chain{V,0,T}) where {V,T} = @inbounds Simplex{V}(t.v[1])
@inline scalar(t::SparseChain{V,0}) where V = @inbounds Simplex{V}(t.v[1])
@inline scalar(t::MultiVector{V}) where V = @inbounds Simplex{V}(t.v[1])
@inline scalar(t::MultiGrade{V,G}) where {V,G} = @inbounds 1 ∈ indices(G) ? terms(t)[1] : zero(V)
@inline vector(t::MultiVector{V,T}) where {V,T} = @inbounds Chain{V,1,T}(t[1])
@inline vector(t::MultiGrade{V,G}) where {V,G} = @inbounds (i=indices(G);2∈i ? terms(t)[findfirst(x->x==2,i)] : zero(V))
@inline volume(t::MultiVector{V}) where V = @inbounds Simplex{V}(t.v[end])
@inline volume(t::MultiGrade{V,G}) where {V,G} = @inbounds ndims(V)+1∈indices(G) ? terms(t)[end] : zero(V)
@inline isscalar(t::MultiVector) = norm(t.v[2:end]) ≈ 0
@inline isscalar(t::MultiGrade) = norm(t) ≈ scalar(t)
@inline isvector(t::MultiVector) = norm(t) ≈ norm(vector(t))
@inline isvector(t::MultiGrade) = norm(t) ≈ norm(vector(t))

function DirectSum.gdims(t::MultiVector{V}) where V
    N = ndims(V)
    out = zeros(MVector{N+1,Int})
    bs = binomsum_set(N)
    for G ∈ 0:N
        ib = indexbasis(N,G)
        for k ∈ 1:length(ib)
            @inbounds t.v[k+bs[G+1]] ≠ 0 && (out[count_ones(symmetricmask(V,ib[k],ib[k])[1])+1] += 1)
        end
    end
    return out
end

## Adjoint

import Base: adjoint # conj

adjoint(b::SparseChain{V,G}) where {V,G} = SparseChain{dual(V),G}(adjoint.(terms(b)))
adjoint(b::MultiGrade{V,G}) where {V,G} = MultiGrade{dual(V),G}(adjoint.(terms(b)))

# Euclidean norm (unsplitter)

unsplitstart(g) = 1|((UInt(1)<<(g-1)-1)<<2)
unsplitend(g) = (UInt(1)<<g-1)<<2

const unsplitter_cache = SparseMatrixCSC{Float64,Int64}[]
@pure unsplitter_calc(n) = (n2=Int(n/2);sparse(1:n2,1:n2,1,n,n)+sparse(1:n2,(n2+1):n,-1/2,n,n)+sparse((n2+1):n,(n2+1):n,1/2,n,n)+sparse((n2+1):n,1:n2,1,n,n))
@pure function unsplitter(n::Int)
    n2 = Int(n/2)
    for k ∈ length(unsplitter_cache)+1:n2
        push!(unsplitter_cache,unsplitter_calc(2k))
    end
    @inbounds unsplitter_cache[n2]
end
@pure unsplitter(n,g) = unsplitter(bladeindex(n,unsplitend(g))-bladeindex(n,unsplitstart(g)))

for implex ∈ (Simplex,SubManifold)
    @eval begin
        #norm(t::$implex) = norm(unsplitval(t))
        function unsplitvalue(a::$implex{V,G}) where {V,G}
            !(hasinf(V) && hasorigin(V)) && (return value(a))
            #T = valuetype(a)
            #$(insert_expr((:N,:t,:out),:mvec,:T,:(typeof((one(T)/(2one(T))))))...)
            #out = copy(value(a,t))
            return unsplitvalue(Chain(a))
        end
    end
end

@eval begin
    #norm(t::$Chain) = norm(unsplitval(t))
    function unsplitvalue(a::$Chain{V,G,T}) where {V,G,T}
        !(hasinf(V) && hasorigin(V)) && (return value(a))
        $(insert_expr((:N,:t,:out),:mvec,:T,:(typeof((one(T)/(2one(T))))))...)
        out = copy(value(a,mvec(N,G,t)))
        bi = bladeindex(N,unsplitstart(G)):bladeindex(N,unsplitend(G))-1
        out[bi] = unsplitter(N,G)*out[bi]
        return out
    end
    #norm(t::MultiVector) = norm(unsplitval(t))
    function unsplitvalue(a::MultiVector{V,T}) where {V,T}
        !(hasinf(V) && hasorigin(V)) && (return value(a))
        $(insert_expr((:N,:t,:out),:mvec,:T,:(typeof((one(T)/(2one(T))))))...)
        out = copy(value(a,mvec(N,t)))
        for G ∈ 1:N-1
            bi = basisindex(N,unsplitstart(G)):basisindex(N,unsplitend(G))-1
            out[bi] = unsplitter(N,G)*out[bi]
        end
        return out
    end
end

# genfun
