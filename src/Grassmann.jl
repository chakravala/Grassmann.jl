module Grassmann

#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

using Combinatorics, StaticArrays, Requires
using ComputedFieldTypes, AbstractLattices

abstract type TensorAlgebra{V} end
abstract type SubAlgebra{V} <: TensorAlgebra{V} end

include("utilities.jl")
include("direct_sum.jl")
include("multivectors.jl")
include("algebra.jl")
include("forms.jl")

## Algebra{N}

@computed struct Algebra{V} <: SubAlgebra{V}
    b::SVector{1<<ndims(V),Basis{V}}
    g::Dict{Symbol,Int}
end

@pure getindex(a::Algebra,i::Int) = getfield(a,:b)[i]
Base.firstindex(a::T) where T<:TensorAlgebra = 1
Base.lastindex(a::T) where T<:TensorAlgebra{V} where V = 1<<ndims(V)
Base.length(a::T) where T<:TensorAlgebra{V} where V = 1<<ndims(V)

@noinline function lookup_basis(V::VectorSpace,v::Symbol)::Union{SValue,Basis}
    vs = string(v)
    vt = vs[1:1]≠pre[1]
    Z=match(Regex("([$(pre[1])]([0-9a-vx-zA-VX-Z]+))?([$(pre[2])]([0-9a-zA-Z]+))?"),vs)
    ef = String[]
    for k ∈ (2,4)
        Z[k] ≠ nothing && push!(ef,Z[k])
    end
    length(ef) == 0 && (return zero(V))
    let W = V,fs=false
        C = dualtype(V)
        X = C≥0 && ndims(V)<4sizeof(Bits)+1
        X && (W = C>0 ? V'⊕V : V⊕V')
        V2 = (vt ⊻ (vt ? C≠0 : C>0)) ? V' : V
        L = length(ef) > 1
        M = X ? Int(ndims(W)/2) : ndims(W)
        m = ((!L) && vt && (C<0)) ? M : 0
        chars = (L || (Z[2] ≠ nothing)) ? alphanumv : alphanumw
        (es,e,et) = indexjoin(Int[],[findfirst(x->x==ef[1][k],chars) for k∈1:length(ef[1])].+m,C<0 ? V : V2)
        et && (return zero(V))
        d = if L
            (fs,f,ft) = indexjoin(Int[],[findfirst(x->x==ef[2][k],alphanumw) for k∈1:length(ef[2])].+M,W)
            ft && (return zero(V))
            out = [e;f]
            Basis{W}(bit2int(basisbits(ndims(W),out)))
        else
            Basis{V2}(e)
        end
        return (es⊻fs) ? SValue(-1,d) : d
    end
end

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
    Algebra{s}(generate(s),Dict{Symbol,Int}([sym[i]=>i for i ∈ 1:1<<ndims(s)]))
end

Algebra(s::VectorSpace) = getalgebra(s)
Algebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getalgebra(n,d,o,s)
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

adjoint(G::A) where A<:SubAlgebra{V} where V = Λ(dual(V))
dual(G::A) where A<: SubAlgebra = G'

export Λ, @Λ_str, getalgebra, getbasis, TensorAlgebra, SubAlgebra

const Λ = Algebra

macro Λ_str(str)
    Algebra(str)
end

@pure do2m(d,o,c) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))
@pure getalgebra(n::Int,d::Int,o::Int,s,c::Int=0) = getalgebra(n,do2m(d,o,c),s)
@pure getalgebra(n::Int,m::Int,s) = getalgebra(n,m,Bits(s))
@pure function getalgebra(V::VectorSpace)
    N,C = ndims(V),dualtype(V)
    C<0 && N>2algebra_limit && (return getextended(V))
    getalgebra(N,do2m(Int(hasdual(V)),Int(hasorigin(V)),C),value(V))
end

@pure function Base.getproperty(λ::typeof(Λ),v::Symbol)
    v ∈ (:body,:var) && (return getfield(λ,v))
    V = string(v)
    N = parse(Int,V[2])
    C = V[1]∉('D','C') ? 0 : 1
    length(V) < 5 && (V *= join(zeros(Int,5-length(V))))
    S = Bits(parse(Int,V[5:end]))
    getalgebra(N,do2m(parse(Int,V[3]),parse(Int,V[4]),C),C>0 ? flip_sig(N,S) : S)
end

const V0 = VectorSpace(0)
const Λ0 = Λ{V0}(SVector{1,Basis{V0}}(Basis{V0,0,zero(Bits)}()),Dict(:e=>1))
const algebra_cache = Vector{Dict{Bits,Λ}}[]
@pure function getalgebra(n::Int,m::Int,s::Bits)
    n==0 && (return Λ0)
    n > sparse_limit && (return getextended(n,m,s))
    n > algebra_limit && (return getsparse(n,m,s))
    for N ∈ length(algebra_cache)+1:n
        push!(algebra_cache,[Dict{Int,Λ}() for k∈1:12])
    end
    if !haskey(algebra_cache[n][m+1],s)
        D = Int(m ∈ (1,3,5,7,9,11))
        O = Int(m ∈ (2,3,6,7,10,11))
        C = m ∈ 8:11 ? -1 : Int(m ∈ (4,5,6,7))
        c = C>0 ? "'" : C<0 ? "*" : ""
        @info("Allocating thread-safe $(2^n)×Basis{VectorSpace{$n,$D,$O,$(Int(s))}$c,...}")
        push!(algebra_cache[n][m+1],s=>collect(VectorSpace{n,D,O,s,C}()))
    end
    algebra_cache[n][m+1][s]
end

@pure getbasis(V::VectorSpace,v::Symbol) = getproperty(getalgebra(V),v)
@pure function getbasis(V::VectorSpace{N},b) where N
    B = Bits(b)
    G = getalgebra(V)
    if N ≤ algebra_limit
        G.b[basisindex(ndims(V),B)]
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
        B = indexbasis(N,G)[i-binomsum(N,G)]
        Basis{V,count_ones(B),B}()
    end
end

@pure function Base.getproperty(a::SparseAlgebra{V},v::Symbol) where V
    return if v ∈ (:b,:g)
        getfield(a,v)
    elseif haskey(a.g,v)
        a[getfield(a,:g)[v]]
    else
        lookup_basis(V,v)
    end
end

SparseAlgebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getsparse(n,d,o,s)
SparseAlgebra(s::String) = getsparse(VectorSpace(s))
SparseAlgebra(s::String,v::Symbol) = getbasis(VectorSpace(s),v)

function show(io::IO,a::SparseAlgebra{V}) where V
    print(io,"Grassmann.SparseAlgebra{$V,$(1<<ndims(V))}($(a[1]), ..., $(a[end]))")
end

const sparse_cache = Vector{Dict{Bits,SparseAlgebra}}[]
@pure getsparse(n::Int,d::Int,o::Int,s,c::Int=0) = getsparse(n,do2m(d,o,c),s)
@pure getsparse(n::Int,m::Int,s) = getsparse(n,m,Bits(s))
@pure getsparse(V::VectorSpace) = getsparse(ndims(V),do2m(Int(hasdual(V)),Int(hasorigin(V)),dualtype(V)),value(V))
@pure function getsparse(n::Int,m::Int,s::Bits)
    n==0 && (return SparseAlgebra(V0))
    for N ∈ length(sparse_cache)+1:n
        push!(sparse_cache,[Dict{Int,SparseAlgebra}() for k∈1:12])
    end
    if !haskey(sparse_cache[n][m+1],s)
        D = Int(m ∈ (1,3,5,7,9,11))
        O = Int(m ∈ (2,3,6,7,10,11))
        C = m ∈ 8:11 ? -1 : Int(m ∈ (4,5,6,7))
        c = C>0 ? "'" : C<0 ? "*" : ""
        @info("Declaring thread-safe $(1<<n)×Basis{VectorSpace{$n,$D,$O,$(Int(s))}$c,...}")
        push!(sparse_cache[n][m+1],s=>SparseAlgebra(VectorSpace{n,D,O,s,C}()))
    end
    sparse_cache[n][m+1][s]
end

## ExtendexAlgebra{V}

struct ExtendedAlgebra{V} <: SubAlgebra{V} end

@pure ExtendedAlgebra(s::VectorSpace) = ExtendedAlgebra{s}()

@pure function Base.getproperty(a::ExtendedAlgebra{V},v::Symbol) where V
    if v ∈ (:b,:g)
        throw(error("ExtendedAlgebra does not have field $v"))
    else
        return lookup_basis(V,v)
    end
end

ExtendedAlgebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getextended(n,d,o,s)
ExtendedAlgebra(s::String) = getextended(VectorSpace(s))
ExtendedAlgebra(s::String,v::Symbol) = getbasis(VectorSpace(s),v)

function show(io::IO,a::ExtendedAlgebra{V}) where V
    N = 1<<ndims(V)
    print(io,"Grassmann.ExtendedAlgebra{$V,$N}($(getbasis(V,0)), ..., $(getbasis(V,N-1)))")
end

const extended_cache = Vector{Dict{Bits,ExtendedAlgebra}}[]
@pure getextended(n::Int,d::Int,o::Int,s,c::Int=0) = getextended(n,do2m(d,o,c),s)
@pure getextended(n::Int,m::Int,s) = getextended(n,m,Bits(s))
@pure getextended(V::VectorSpace) = getextended(ndims(V),do2m(Int(hasdual(V)),Int(hasorigin(V)),dualtype(V)),value(V))
@pure function getextended(n::Int,m::Int,s::Bits)
    n==0 && (return ExtendedAlgebra(V0))
    for N ∈ length(extended_cache)+1:n
        push!(extended_cache,[Dict{Int,ExtendedAlgebra}() for k∈1:12])
    end
    if !haskey(extended_cache[n][m+1],s)
        D = Int(m ∈ (1,3,5,7,9,11))
        O = Int(m ∈ (2,3,6,7,10,11))
        C = m ∈ 8:11 ? -1 : Int(m ∈ (4,5,6,7))
        c = C>0 ? "'" : C<0 ? "*" : ""
        @info("Extending thread-safe $(2^n)×Basis{VectorSpace{$n,$D,$O,$(Int(s))}$c,...}")
        push!(extended_cache[n][m+1],s=>ExtendedAlgebra(VectorSpace{n,D,O,s,C}()))
    end
    extended_cache[n][m+1][s]
end

function __init__()
    @require Reduce="93e0c654-6965-5f22-aba9-9c1ae6b3c259" include("symbolic.jl")
end

end # module
