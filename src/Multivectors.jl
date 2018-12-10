module Multivectors

#   This file is part of Multivectors.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

using Combinatorics, StaticArrays
using ComputedFieldTypes

import Base: print, show, getindex, promote_rule, ==
export AbstractTerm, MultiBasis, MultiValue, MultiBlade, MultiVector, MultiGrade, Signature, @S_str

abstract type AbstractTerm{N,G} end

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

const binomsum = ( () -> begin
        Y = Array{Int,1}[Int[1]]
        return (n::Int,i::Int) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    push!(Y,cumsum([binomial(k,q) for q ∈ 0:k]))
                end
                i ≠ 0 ? Y[n][i] : 0
            end)
    end)()

const combo = ( () -> begin
        Y = Array{Array{Array{Int,1},1},1}[]
        return (n::Int,g::Int) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    z = 1:k
                    push!(Y,[collect(combinations(z,q)) for q ∈ z])
                end
                g ≠ 0 ? Y[n][g] : [Int[]]
            end)
    end)()

const basisindex = ( () -> begin
        Y = Array{Dict{Array{Int,1},Int},1}[]
        return (n::Int,i::Array{Int,1}) -> (begin
                j = length(Y)
                g = length(i)
                for k ∈ j+1:n
                    push!(Y,[Dict{Array{Int,1},Int}() for k ∈ 1:k])
                end
                g>0 && !haskey(Y[n][g],i) && 
                    push!(Y[n][g],i=>findall(x->x==i,combo(n,g))[1])
                g>0 ? Y[n][g][i] : 1
            end)
    end)()


struct Signature{N}
    b::BitArray{1}
end

Signature(s::String) = Signature{length(s)}(BitArray([k ≠ '+' ? 0 : 1 for k ∈ s]))

show(io::IO,s::Signature) = print(io,s)
print(io::IO,s::Signature) = print(io,[k ? '+' : '-' for k ∈ s.b]...)

macro S_str(str)
    Signature(str)
end

## MultiBasis{N}

struct MultiBasis{N,G} <: AbstractTerm{N,G}
    s::Signature{N}
    b::BitArray{1}
end
MultiBasis{N}(s::Signature,b::BitArray) where N = MultiBasis{N,sum(b)}(s,b)

VTI = Union{Vector{<:Integer},Tuple,NTuple}
MultiBasis(s::Signature,b::VTI) = MultiBasis{length(s.b)}(s,b)
MultiBasis(s::Signature,b::Integer...) = MultiBasis{length(s.b)}(s,b)
MultiBasis{N}(s::Signature,b::VTI) where N = MultiBasis{N}(s,basisbits(N,b))
MultiBasis{N}(s::Signature,b::Integer...) where N = MultiBasis{N}(s,basisbits(N,b))
MultiBasis{N,G}(s::Signature,b::VTI) where {N,G} = MultiBasis{N,G}(s,basisbits(N,b))
MultiBasis{N,G}(s::Signature,b::Integer...) where {N,G} = MultiBasis{N,G}(s,basisbits(N,b))

function ==(a::MultiBasis{N,G},b::MultiBasis{N,G}) where {N,G}
    return (a.s == b.s) && (a.b == b.b)
end

basisindices(b::MultiBasis) = findall(b.b)
function basisbits(d::Integer,b::VTI)
    out = falses(d)
    for k ∈ b
        out[k] = true
    end
    return out
end

show(io::IO, e::MultiBasis) = print(io,e)
print(io::IO, e::MultiBasis) = printbasis(io,basisindices(e))

printbasis(io::IO,b::VTI,e::String="e") = print(io,e,[subscripts[i] for i ∈ b]...)

export @multibasis

macro multibasis(label,sig,str)
    s = Signature(str)
    N = length(str)
    lab = string(label)
    io = IOBuffer()
    els = Symbol[sig,label]
    exp = Expr[Expr(:(=),esc(sig),s),
        Expr(:(=),esc(label),MultiBasis(s))]
    for i ∈ 1:N
        set = combo(N,i) |> collect
        for k ∈ 1:length(set)
            print(io,lab,set[k]...)
            sym = Symbol(String(take!(io)))
            push!(els,sym)
            push!(exp,Expr(:(=),esc(sym),MultiBasis(s,set[k])))
        end
    end
    return Expr(:block,exp...,Expr(:tuple,esc.(els)...))
end

## MultiValue{N}

struct MultiValue{N,G,T} <: AbstractTerm{N,G}
    v::T
    b::MultiBasis{N,G}
end

MultiValue(b::MultiBasis{N,G}) where {N,G} = MultiValue{N,G,Int}(1,b)
MultiValue{N}(b::MultiBasis{N,G}) where {N,G} = MultiValue{N,G,Int}(1,b)
MultiValue{N}(v,b::MultiValue{N,G}) where {N,G} = MultiValue{N,G}(v*b.v,b.b)
MultiValue{N}(v::T,b::MultiBasis{N,G}) where {N,G,T} = MultiValue{N,G,T}(v,b)
MultiValue{N,G}(v::T,b::MultiBasis{N,G}) where {N,G,T} = MultiValue{N,G,T}(v,b)
MultiValue{N}(v::T) where {N,T} = MultiValue{N,0,T}(v,MultiBasis{N}())
MultiValue(v,b::AbstractTerm{N,G}) where {N,G} = MultiValue{N,G}(v,b)

show(io::IO,m::MultiValue) = print(io,m.v,m.b)

## MultiBlades{T,N}

@computed struct MultiBlade{T,N,G}
    s::Signature
    v::MVector{binomial(N,G),T}
end

function Base.getindex(m::MultiBlade,i::Int)
    #0 <= i <= N || throw(BoundsError(m, i))
    return m.v[i]
end

function Base.setindex!(m::MultiBlade{T},k::T,i::Int) where T
    m.v[i] = k
end

Base.firstindex(m::MultiBlade) = 1
Base.lastindex(m::MultiBlade{T,N,G}) where {T,N,G} = length(m.v)

(m::MultiBlade{T,N,G})(i) where {T,N,G} = MultiValue{N,G,T}(m[i],MultiBasis(m.s,combo(N,G)[i]))

function MultiBlade(v::MultiBasis{N,G}) where {N,G}
    out = MultiBlade{Int,N,G}(v.s,zeros(Int,binomial(N,G)))
    out[basisindex(N,findall(v.b))] = one(Int)
    return out
end
for var ∈ [[:T,:N,:G],[:T,:N],[:T]]
    @eval begin
        function MultiBlade{$(var...)}(v::MultiBasis{N,G}) where {T,N,G}
            out = MultiBlade{T,N,G}(v.s,zeros(T,binomial(N,G)))
            out[basisindex(N,findall(v.b))] = one(T)
            return out
        end
    end
end
for var ∈ [[:T,:N,:G],[:T,:N],[:T],[]]
    @eval begin
        function MultiBlade{$(var...)}(v::MultiValue{N,G,T}) where {T,N,G}
            out = MultiBlade{T,N,G}(v.b.s,zeros(T,binomial(N,G)))
            out[basisindex(N,findall(v.b.b))] = v.v
            return out
        end
    end
end

function show(io::IO, m::MultiBlade{T,N,G}) where {T,N,G}
    set = combo(N,G)
    print(io,m.v[1])
    printbasis(io,set[1])
    for k ∈ 2:length(set)
        print(io,signbit(m.v[k]) ? " - " : " + ",abs(m.v[k]))
        printbasis(io,set[k])
    end
end

## MultiVector{T,N}

@computed struct MultiVector{T,N}
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

(m::MultiVector{T,N})(g::Int) where {T,N} = MultiBlade{T,N,g}(m.s,m[g])
(m::MultiVector{T,N})(g::Int,i::Int) where {T,N} = MultiValue{N,g,T}(m[g][i],MultiBasis{N,g}(m.s,combo(N,g)[i]))

function MultiVector(v::MultiBasis{N,G}) where {N,G}
    out = MultiVector{Int,N}(v.s,zeros(Int,2^N))
    out.v[binomsum(N,G-1)+basisindex(N,findall(v.b))] = one(Int)
    return out
end
for var ∈ [[:T,:N,:G],[:T,:N],[:T]]
    @eval begin
        function MultiVector{$(var...)}(v::MultiBasis{N,G}) where {T,N,G}
            out = MultiVector{T,N}(v.s,zeros(T,2^N))
            out.v[binomsum(N,G-1)+basisindex(N,findall(v.b))] = one(T)
            return out
        end
    end
end
for var ∈ [[:T,:N,:G],[:T,:N],[:T],[]]
    @eval begin
        function MultiVector{$(var...)}(v::MultiValue{N,G,T}) where {T,N,G}
            out = MultiVector{T,N}(v.b.s,zeros(T,2^N))
            out.v[binomsum(N,G-1)+basisindex(N,findall(v.b.b))] = v.v
            return out
        end
        function MultiVector{$(var...)}(v::MultiBlade{T,N,G}) where {T,N,G}
            out = MultiVector{T,N}(v.s,zeros(T,2^N))
            r = binomsum(N,G)
            out.v[r+1:r+binomial(N,G)] = v.v
            return out
        end
    end
end

function show(io::IO, m::MultiVector{T,N}) where {T,N}
    print(io,m[0][1])
    for i ∈ 1:N
        b = m[i]
        set = combo(N,i)
        for k ∈ 1:length(set)
            if b[k] ≠ 0
                print(io,signbit(b[k]) ? " - " : " + ",abs(b[k]))
                printbasis(io,set[k])
            end
        end
    end
end

## MultiGrade{N}

struct MultiGrade{N}
    v::Vector{<:AbstractTerm{N}}
end

MultiGrade{N}(v::T...) where T <: (AbstractTerm{N,G} where G) where N = MultiGrade{N}(v)

function show(io::IO,m::MultiGrade)
    for k ∈ 1:length(m.v)
        x = m.v[k]
        if typeof(x) <: MultiValue && signbit(x.v)
            print(io," - ")
            ax = abs(x.v)
            ax ≠ 1 && print(io,ax)
        else
            k ≠ 1 && print(io," + ")
            typeof(x) <: MultiValue && x.v ≠ 1 && print(io,x.v)
        end
        show(io,typeof(x) <: MultiValue ? x.b : x)
    end
end

include("algebra.jl")

end # module
