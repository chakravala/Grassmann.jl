module Grassmann

#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

using Combinatorics, StaticArrays, Requires
using ComputedFieldTypes, AbstractLattices
using DirectSum, AbstractTensors

export VectorSpace, vectorspace, ⊕, ℝ, @V_str
import DirectSum: hasdual, hasorigin, dualtype, dual, value, vectorspace, V0, ⊕

include("utilities.jl")
include("multivectors.jl")
include("algebra.jl")
include("forms.jl")
include("generators.jl")

## fundamentals

export hyperplanes

@pure hyperplanes(V::VectorSpace{N}) where N = map(n->I*getbasis(V,1<<n),0:N-1)

abstract type SubAlgebra{V} <: TensorAlgebra{V} end

@pure adjoint(G::A) where A<:SubAlgebra{V} where V = Λ(dual(V))
@pure dual(G::A) where A<: SubAlgebra = G'
Base.firstindex(a::T) where T<:SubAlgebra = 1
Base.lastindex(a::T) where T<:SubAlgebra{V} where V = 1<<ndims(V)
Base.length(a::T) where T<:SubAlgebra{V} where V = 1<<ndims(V)

==(::SubAlgebra{V},::SubAlgebra{W}) where {V,W} = V == W

⊕(::SubAlgebra{V},::SubAlgebra{W}) where {V,W} = getalgebra(V⊕W)
+(::SubAlgebra{V},::SubAlgebra{W}) where {V,W} = getalgebra(V⊕W)

## Algebra{N}

@computed struct Algebra{V} <: SubAlgebra{V}
    b::SVector{1<<ndims(V),Basis{V}}
    g::Dict{Symbol,Int}
end

@pure getindex(a::Algebra,i::Int) = getfield(a,:b)[i]
@pure getindex(a::Algebra,i::Colon) = getfield(a,:b)
@pure getindex(a::Algebra,i::UnitRange{Int}) = [getindex(a,j) for j ∈ i]

@pure function Base.getproperty(a::Algebra{V},v::Symbol) where V
    return if v ∈ (:b,:g)
        getfield(a,v)
    elseif haskey(a.g,v)
        a[getfield(a,:g)[v]]
    else
        lookup_basis(V,v)
    end
end

@pure function Base.collect(s::VectorSpace)
    sym = labels(s)
    @inbounds Algebra{s}(generate(s),Dict{Symbol,Int}([sym[i]=>i for i ∈ 1:1<<ndims(s)]))
end

@pure Algebra(s::VectorSpace) = getalgebra(s)
@pure Algebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getalgebra(n,d,o,s)
Algebra(s::String) = getalgebra(VectorSpace(s))
Algebra(s::String,v::Symbol) = getbasis(VectorSpace(s),v)

function show(io::IO,a::Algebra{V}) where V
    N = ndims(V)
    print(io,"Grassmann.Algebra{$V,$(1<<N)}(")
    for i ∈ 1:1<<N-1
        print(io,a[i],", ")
    end
    print(io,a[end],")")
end

export Λ, @Λ_str, getalgebra, getbasis, TensorAlgebra, SubAlgebra

const Λ = Algebra

macro Λ_str(str)
    Algebra(str)
end

@pure getalgebra(n::Int,d::Int,o::Int,s,c::Int=0) = getalgebra(n,doc2m(d,o,c),s)
@pure getalgebra(n::Int,m::Int,s) = getalgebra(n,m,Bits(s))
@pure function getalgebra(V::VectorSpace)
    N,C = ndims(V),dualtype(V)
    C<0 && N>2algebra_limit && (return getextended(V))
    getalgebra(N,doc2m(Int(hasdual(V)),Int(hasorigin(V)),C),value(V))
end

@pure function Base.getproperty(λ::typeof(Λ),v::Symbol)
    v ∈ (:body,:var) && (return getfield(λ,v))
    V = string(v)
    N = parse(Int,V[2])
    C = V[1]∉('D','C') ? 0 : 1
    length(V) < 5 && (V *= join(zeros(Int,5-length(V))))
    S = Bits(parse(Int,V[5:end]))
    getalgebra(N,doc2m(parse(Int,V[3]),parse(Int,V[4]),C),C>0 ? DirectSum.flip_sig(N,S) : S)
end

# Allocating thread-safe $(2^n)×Basis{VectorSpace}
const Λ0 = Λ{V0}(SVector{1,Basis{V0}}(Basis{V0,0,zero(Bits)}()),Dict(:e=>1))
const algebra_cache = Vector{Dict{Bits,Λ}}[]
@pure function getalgebra(n::Int,m::Int,s::Bits)
    n==0 && (return Λ0)
    n > sparse_limit && (return getextended(n,m,s))
    n > algebra_limit && (return getsparse(n,m,s))
    for N ∈ length(algebra_cache)+1:n
        push!(algebra_cache,[Dict{Int,Λ}() for k∈1:12])
    end
    @inbounds if !haskey(algebra_cache[n][m+1],s)
        @inbounds push!(algebra_cache[n][m+1],s=>collect(VectorSpace{n,m,s}()))
    end
    @inbounds algebra_cache[n][m+1][s]
end

@pure getbasis(V::VectorSpace,v::Symbol) = getproperty(getalgebra(V),v)
@pure function getbasis(V::VectorSpace{N},b) where N
    B = Bits(b)
    if N ≤ algebra_limit
        @inbounds getalgebra(V).b[basisindex(ndims(V),B)]
    else
        Basis{V,count_ones(B),B}()
    end
end

## SparseAlgebra{V}

struct SparseAlgebra{V} <: SubAlgebra{V}
    b::Vector{Symbol}
    g::Dict{Symbol,Int}
end

@pure function SparseAlgebra(s::VectorSpace)
    sym = labels(s)
    SparseAlgebra{s}(sym,Dict{Symbol,Int}([sym[i]=>i for i ∈ 1:1<<ndims(s)]))
end

@pure function getindex(a::SparseAlgebra{V},i::Int) where V
    N = ndims(V)
    if N ≤ algebra_limit
        getalgebra(V).b[i]
    else
        F = findfirst(x->1+binomsum(N,x)-i>0,0:N)
        G = F ≠ nothing ? F-2 : N
        @inbounds B = indexbasis(N,G)[i-binomsum(N,G)]
        Basis{V,count_ones(B),B}()
    end
end

@pure function Base.getproperty(a::SparseAlgebra{V},v::Symbol) where V
    return if v ∈ (:b,:g)
        getfield(a,v)
    elseif haskey(a.g,v)
        @inbounds a[getfield(a,:g)[v]]
    else
        lookup_basis(V,v)
    end
end

@pure SparseAlgebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getsparse(n,d,o,s)
SparseAlgebra(s::String) = getsparse(VectorSpace(s))
SparseAlgebra(s::String,v::Symbol) = getbasis(VectorSpace(s),v)

function show(io::IO,a::SparseAlgebra{V}) where V
    print(io,"Grassmann.SparseAlgebra{$V,$(1<<ndims(V))}($(a[1]), ..., $(a[end]))")
end

# Declaring thread-safe $(1<<n)×Basis{VectorSpace}
const sparse_cache = Vector{Dict{Bits,SparseAlgebra}}[]
@pure getsparse(n::Int,d::Int,o::Int,s,c::Int=0) = getsparse(n,doc2m(d,o,c),s)
@pure getsparse(n::Int,m::Int,s) = getsparse(n,m,Bits(s))
@pure getsparse(V::VectorSpace) = getsparse(ndims(V),do2m(Int(hasdual(V)),Int(hasorigin(V)),dualtype(V)),value(V))
@pure function getsparse(n::Int,m::Int,s::Bits)
    n==0 && (return SparseAlgebra(V0))
    for N ∈ length(sparse_cache)+1:n
        push!(sparse_cache,[Dict{Int,SparseAlgebra}() for k∈1:12])
    end
    @inbounds if !haskey(sparse_cache[n][m+1],s)
        @inbounds push!(sparse_cache[n][m+1],s=>SparseAlgebra(VectorSpace{n,m,s}()))
    end
    @inbounds sparse_cache[n][m+1][s]
end

## ExtendedAlgebra{V}

struct ExtendedAlgebra{V} <: SubAlgebra{V} end

@pure ExtendedAlgebra(s::VectorSpace) = ExtendedAlgebra{s}()

@pure function Base.getproperty(a::ExtendedAlgebra{V},v::Symbol) where V
    if v ∈ (:b,:g)
        throw(error("ExtendedAlgebra does not have field $v"))
    else
        return lookup_basis(V,v)
    end
end

@pure ExtendedAlgebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getextended(n,d,o,s)
ExtendedAlgebra(s::String) = getextended(VectorSpace(s))
ExtendedAlgebra(s::String,v::Symbol) = getbasis(VectorSpace(s),v)

function show(io::IO,a::ExtendedAlgebra{V}) where V
    N = 1<<ndims(V)
    print(io,"Grassmann.ExtendedAlgebra{$V,$N}($(getbasis(V,0)), ..., $(getbasis(V,N-1)))")
end

# Extending thread-safe $(2^n)×Basis{VectorSpace}
const extended_cache = Vector{Dict{Bits,ExtendedAlgebra}}[]
@pure getextended(n::Int,d::Int,o::Int,s,c::Int=0) = getextended(n,doc2m(d,o,c),s)
@pure getextended(n::Int,m::Int,s) = getextended(n,m,Bits(s))
@pure getextended(V::VectorSpace) = getextended(ndims(V),doc2m(Int(hasdual(V)),Int(hasorigin(V)),dualtype(V)),value(V))
@pure function getextended(n::Int,m::Int,s::Bits)
    n==0 && (return ExtendedAlgebra(V0))
    for N ∈ length(extended_cache)+1:n
        push!(extended_cache,[Dict{Bits,ExtendedAlgebra}() for k∈1:12])
    end
    @inbounds if !haskey(extended_cache[n][m+1],s)
        @inbounds push!(extended_cache[n][m+1],s=>ExtendedAlgebra(VectorSpace{n,m,s}()))
    end
    @inbounds extended_cache[n][m+1][s]
end

# ParaAlgebra

function __init__()
    @require Reduce="93e0c654-6965-5f22-aba9-9c1ae6b3c259" include("symbolic.jl")
    @require SymPy="24249f21-da20-56a4-8eb1-6a02cf4ae2e6" generate_product_algebra(:(SymPy.Sym),:(SymPy.:*),:(SymPy.:+),:(SymPy.:-),:svec,:(SymPy.conj))
end

end # module
