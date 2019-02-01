module Grassmann

#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

using Combinatorics, StaticArrays, Requires
using ComputedFieldTypes, AbstractLattices

abstract type TensorAlgebra{V} <: Number end
abstract type SubAlgebra{V} <: TensorAlgebra{V} end

include("utilities.jl")
include("direct_sum.jl")
include("multivectors.jl")
include("algebra.jl")
include("forms.jl")

## Algebra{N}

@computed struct Algebra{V} <: SubAlgebra{V}
    b::SVector{2^ndims(V),Basis{V}}
    g::Dict{Symbol,Int}
end

@pure getindex(a::Algebra,i::Int) = getfield(a,:b)[i]
Base.firstindex(a::T) where T<:TensorAlgebra = 1
Base.lastindex(a::T) where T<:TensorAlgebra{V} where V = 2^ndims(V)
Base.length(a::T) where T<:TensorAlgebra{V} where V = 2^ndims(V)

@noinline function lookup_basis(V::VectorSpace,v::Symbol)::Union{SValue,Basis}
    vs = string(v)
    vt = vs[1]≠'e'
    ef = split(vs,r"(e|f)")
    let W = V,fs=false
        C = dualtype(V)
        C≥0 && (W = C>0 ? V'⊕V : V⊕V')
        V2 = (vt ⊻ (vt ? C≠0 : C>0)) ? V' : V
        L = length(ef) > 2
        M = Int(ndims(W)/2)
        m = ((!L) && vt && (C<0)) ? M : 0
        (es,e,et) = indexjoin(Int[],[parse(Int,ef[2][k]) for k∈1:length(ef[2])].+m,C<0 ? V : V2)
        et && (return SValue{V}(0,getbasis(V,0)))
        d = if L
            (fs,f,ft) = indexjoin(Int[],[parse(Int,ef[3][k]) for k∈1:length(ef[3])].+M,W)
            ft && (return SValue{V}(0,getbasis(V,0)))
            ef = [e;f]
            Basis{W}(bit2int(basisbits(ndims(W),ef)))
        else
            Basis{V2}(e)
        end
        return (es⊻fs) ? SValue(-1,d) : d
    end
end

@pure function Base.getproperty(a::Algebra{V},v::Symbol) where V
    if v ∈ (:b,:g)
        return getfield(a,v)
    elseif haskey(a.g,v)
        return a[getfield(a,:g)[v]]
    else
        return lookup_basis(V,v)
    end
end

@pure function Base.collect(s::VectorSpace)
    g = Dict{Symbol,Int}()
    basis,sym = generate(s,:e),labels(s,:e)
    for i ∈ 1:2^ndims(s)
        push!(g,sym[i]=>i)
    end
    return Algebra{s}(basis,g)
end

Algebra(s::VectorSpace) = getalgebra(s)
Algebra(n::Int,d::Int=0,o::Int=0,s=0x0000) = getalgebra(n,d,o,s)
Algebra(s::String) = getalgebra(VectorSpace(s))
Algebra(s::String,v::Symbol) = getbasis(VectorSpace(s),v)

function show(io::IO,a::Algebra{V}) where V
    N = ndims(V)
    print(io,"Grassmann.Algebra{$V,$(2^N)}(")
    for i ∈ 1:2^N-1
        print(io,a[i],", ")
    end
    print(io,a[end],")")
end

export Λ, @Λ_str, getalgebra, getbasis, TensorAlgebra, SubAlgebra

const Λ = Algebra

macro Λ_str(str)
    Algebra(str)
end

@pure do2m(d,o,c) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))
@pure getalgebra(n::Int,d::Int,o::Int,s,c::Int=0) = getalgebra(n,do2m(d,o,c),s)
@pure getalgebra(n::Int,m::Int,s) = getalgebra(n,m,UInt16(s))
@pure getalgebra(V::VectorSpace) = getalgebra(ndims(V),do2m(Int(hasdual(V)),Int(hasorigin(V)),dualtype(V)),value(V))

@pure function Base.getproperty(λ::typeof(Λ),v::Symbol)
    v ∈ (:body,:var) && (return getfield(λ,v))
    V = string(v)
    N = parse(Int,V[2])
    C = V[1]∉('D','C') ? 0 : 1
    length(V) < 5 && (V *= join(zeros(Int,5-length(V))))
    S = UInt16(parse(Int,V[5:end]))
    getalgebra(N,do2m(parse(Int,V[3]),parse(Int,V[4]),C),C>0 ? flip_sig(N,S) : S)
end

const V0 = VectorSpace(0)
const Λ0 = Λ{V0}(SVector{1,Basis{V0}}(Basis{V0,0,0x0000}()),Dict(:e=>1))
const algebra_cache = Vector{Dict{UInt16,Λ}}[]
@pure function getalgebra(n::Int,m::Int,s::UInt16)
    n==0 && (return Λ0)
    n > 8 && (return getsparse(n::Int,m::Int,s::UInt16))
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
    B = UInt16(b)
    if N ≤ 8
        getalgebra(V).b[basisindex(ndims(V),B)]
    else
        Basis{V,count_zeros(B),B}()
    end
end

## SparseAlgebra{V}

@computed struct SparseAlgebra{V} <: SubAlgebra{V}
    b::Vector{Symbol}
    g::Dict{Symbol,Int}
end

@pure function SparseAlgebra(s::VectorSpace)
    g = Dict{Symbol,Int}()
    sym = labels(s,:e)
    for i ∈ 1:2^ndims(s)
        push!(g,sym[i]=>i)
    end
    return SparseAlgebra{s}(sym,g)
end

@pure function getindex(a::SparseAlgebra{V},i::Int) where V
    N = ndims(V)
    if N ≤ 8
        getalgebra(V).b[i]
    else
        F = findfirst(x->1+binomsum(N,x)-i>0,0:N)
        G = F ≠ nothing ? F-2 : N
        B = indexbasis(N,G)[i-binomsum(N,G)]
        Basis{V,count_zeros(B),B}()
    end
end

@pure function Base.getproperty(a::SparseAlgebra{V},v::Symbol) where V
    if v ∈ (:b,:g)
        return getfield(a,v)
    elseif haskey(a.g,v)
        return a[getfield(a,:g)[v]]
    else
        return lookup_basis(V,v)
    end
end

SparseAlgebra(n::Int,d::Int=0,o::Int=0,s=0x0000) = getsparse(n,d,o,s)
SparseAlgebra(s::String) = getsparse(VectorSpace(s))
SparseAlgebra(s::String,v::Symbol) = getbasis(VectorSpace(s),v)

function show(io::IO,a::SparseAlgebra{V}) where V
    N = ndims(V)
    print(io,"Grassmann.SparseAlgebra{$V,$(2^N)}(e, ..., e")
    print(io,[subscripts[i] for i ∈ shiftbasis(V,collect(1:N))]...,")")
end

const sparse_cache = Vector{Dict{UInt16,SparseAlgebra}}[]
@pure function getsparse(n::Int,m::Int,s::UInt16)
    n==0 && (return SparseAlgebra(V0))
    for N ∈ length(sparse_cache)+1:n
        push!(sparse_cache,[Dict{Int,SparseAlgebra}() for k∈1:12])
    end
    if !haskey(sparse_cache[n][m+1],s)
        D = Int(m ∈ (1,3,5,7,9,11))
        O = Int(m ∈ (2,3,6,7,10,11))
        C = m ∈ 8:11 ? -1 : Int(m ∈ (4,5,6,7))
        c = C>0 ? "'" : C<0 ? "*" : ""
        @info("Declaring thread-safe $(2^n)×Basis{VectorSpace{$n,$D,$O,$(Int(s))}$c,...}")
        push!(sparse_cache[n][m+1],s=>SparseAlgebra(VectorSpace{n,D,O,s,C}()))
    end
    sparse_cache[n][m+1][s]
end



function __init__()
    @require Reduce="93e0c654-6965-5f22-aba9-9c1ae6b3c259" include("symbolic.jl")
end

end # module
