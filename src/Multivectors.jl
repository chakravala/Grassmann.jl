module Multivectors

#   This file is part of Multivectors.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

using Combinatorics, StaticArrays
using ComputedFieldTypes

import Base: show, getindex, promote_rule, ==
export MultiBasis, MultiValue, MultiBlade, MultiVector, MultiGrade, Signature, @S_str

subscripts = Dict(
    1 => '₁',
    2 => '₂',
    3 => '₃',
    4 => '₄',
    5 => '₅',
    6 => '₆',
    7 => '₇',
    8 => '₈',
    9 => '₉',
    0 => '₀'
)

binomsum = ( () -> begin
        Y = Array{Int,1}[Int[1]]
        return (n::Int,i::Int) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    push!(Y,cumsum([binomial(n,k) for k ∈ 0:n]))
                end
                i ≠ 0 ? Y[n][i] : 0
            end)
    end)()

struct Signature{N}
    b::BitArray{1}
end

Signature(s::String) = Signature{length(s)}(BitArray([k ≠ '+' ? 0 : 1 for k ∈ s]))

show(io::IO,s::Signature) = print(io,[k ? '+' : '-' for k ∈ s.b]...)

macro S_str(str)
    Signature(str)
end

## MultiBasis{N}

struct MultiBasis{N,G}
    s::Signature{N}
    n::BitArray{1}
end
MultiBasis{N}(s::Signature,n::BitArray) where N = MultiBasis{N,sum(Int.(n))}(s,n)

VTI = Union{Vector{<:Integer},Tuple,NTuple}
MultiBasis(s::Signature,b::VTI) = MultiBasis{length(s.b)}(s,b)
MultiBasis(s::Signature,b::Integer...) = MultiBasis{length(s.b)}(s,b)
MultiBasis{N}(s::Signature,b::VTI) where N = MultiBasis{N}(s,basisbits(N,b))
MultiBasis{N}(s::Signature,b::Integer...) where N = MultiBasis{N}(s,basisbits(N,b))
MultiBasis{N,G}(s::Signature,b::VTI) where {N,G} = MultiBasis{N,G}(s,basisbits(N,b))
MultiBasis{N,G}(s::Signature,b::Integer...) where {N,G} = MultiBasis{N,G}(s,basisbits(N,b))

function ==(a::MultiBasis{N,G},b::MultiBasis{N,G}) where {N,G}
    return (a.s == b.s) && (a.n == b.n)
end

basisindices(b::MultiBasis) = findall(b.n)
function basisbits(d::Integer,b::VTI)
    out = falses(d)
    for k ∈ b
        out[k] = true
    end
    return out
end

show(io::IO, e::MultiBasis) = printbasis(io,basisindices(e))

printbasis(io::IO,b::VTI,e::String="e") = print(io,e,[subscripts[i] for i ∈ b]...)

export @multibasis

macro multibasis(label,sig,str)
    s = Signature(str)
    N = length(str)
    ind = collect(1:N)
    lab = string(label)
    io = IOBuffer()
    els = Symbol[sig,label]
    exp = Expr[Expr(:(=),esc(sig),s),
        Expr(:(=),esc(label),MultiBasis(s))]
    for i ∈ 1:N
        set = combinations(ind,i) |> collect
        for k ∈ 1:length(set)
            printbasis(io,set[k],lab)
            sym = Symbol(String(take!(io)))
            push!(els,sym)
            push!(exp,Expr(:(=),esc(sym),MultiBasis(s,set[k])))
        end
    end
    return Expr(:block,exp...,Expr(:tuple,esc.(els)...))
end

## MultiValue{N}

struct MultiValue{N,G,T}
    v::T
    b::MultiBasis{N,G}
end

MultiValue{N}(v::T,b::MultiBasis{N,G}) where {N,G,T} = MultiValue{N,G,T}(v,b)
MultiValue{N,G}(v::T,b::MultiBasis{N,G}) where {N,G,T} = MultiValue{N,G,T}(v,b)
MultiValue{N}(v::T) where {N,T} = MultiValue{N,0,T}(v,MultiBasis{N}())

#Base.+(v::MultiValue{N}...) = 

mutable struct MultiGrade{N}
    v::Vector{MultiValue{N}}
end

#mutable struct 

## MultiBlades{T,N}

@computed mutable struct MultiBlade{T,N,G}
    s::Signature
    v::MVector{binomial(N,G),T}
end

## MultiVector{T,N}

@computed mutable struct MultiVector{T,N}
    s::Signature
    v::MVector{2^N,T}
end

MultiVector{T}(s::Signature,v::MVector{M,T}) where {T,M} = MultiVector{T,intlog(M)}(s,v)
MultiVector{T}(s::Signature,v::Vector{T}) where T = MultiVector{T,intlog(length(v))}(s,v)
MultiVector(s::Signature,v::MVector{M,T}) where {T,M} = MultiVector{T,intlog(M)}(s,v)
MultiVector(s::Signature,v::Vector{T}) where T = MultiVector{T,intlog(length(v))}(s,v)
MultiVector(s::Signature,v::T...) where T = MultiVector{T,intlog(length(v))}(s,v)

function intlog(M::Integer)
    lM = log2(M)
    try; Int(lM)
    catch; lM end
end

function Base.getindex(m::MultiVector{T,N},i::Int) where {T,N}
    0 <= i <= N || throw(BoundsError(m, i))
    r = binomsum(N,i)
    return @view m.v[r+1:r+binomial(N,i)]
end
getindex(m::MultiVector,i::Int,j::Int) = m[i][j]

function Base.setindex!(m::MultiVector{T},k::T,i::Int,j::Int) where T
    m[i][j] = k
end

Base.firstindex(m::MultiVector) = 0
Base.lastindex(m::MultiVector{T,N}) where {T,N} = N

function show(io::IO, m::MultiVector{T,N}) where {T,N}
    print(io,m[0][1])
    ind = collect(1:N)
    for i ∈ 1:N
        b = m[i]
        set = combinations(ind,i) |> collect
        for k ∈ 1:length(set)
            print(io," + ",b[k])
            printbasis(io,set[k])
        end
    end
end

include("algebra.jl")

end # module
