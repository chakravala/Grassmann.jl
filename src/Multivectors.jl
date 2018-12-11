module Multivectors

#   This file is part of Multivectors.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

using Combinatorics, StaticArrays
using ComputedFieldTypes

import Base: print, show, getindex, setindex!, promote_rule, ==, convert
export AbstractTerm, MultiBasis, MultiValue, MultiBlade, MultiVector, MultiGrade, Signature, @S_str

abstract type AbstractTerm{N,G} end

include("utilities.jl")

## Signature{N}

struct Signature{N}
    b::BitArray{1}
end

Signature(s::String) = Signature{length(s)}(BitArray([k ≠ '+' ? 0 : 1 for k ∈ s]))

print(io::IO,s::Signature) = print(io,[k ? '+' : '-' for k ∈ s.b]...)
show(io::IO,s::Signature) = print(io,s)

macro S_str(str)
    Signature(str)
end

## MultiBasis{N}

struct MultiBasis{N,G} <: AbstractTerm{N,G}
    s::Signature{N}
    b::BitArray{1}
end

VTI = Union{Vector{<:Integer},Tuple,NTuple}

basisindices(b::MultiBasis) = findall(b.b)

function basisbits(d::Integer,b::VTI)
    out = falses(d)
    for k ∈ b
        out[k] = true
    end
    return out
end

MultiBasis{N}(s::Signature,b::BitArray) where N = MultiBasis{N,sum(b)}(s,b)
MultiBasis(s::Signature,b::VTI) = MultiBasis{length(s.b)}(s,b)
MultiBasis(s::Signature,b::Integer...) = MultiBasis{length(s.b)}(s,b)

for t ∈ [[:N],[:N,:G]]
    @eval begin
        function MultiBasis{$(t...)}(s::Signature,b::VTI) where {$(t...)}
            MultiBasis{$(t...)}(s,basisbits(N,b))
        end
        function MultiBasis{$(t...)}(s::Signature,b::Integer...) where {$(t...)}
            MultiBasis{$(t...)}(s,basisbits(N,b))
        end
    end
end

function ==(a::MultiBasis{N,G},b::MultiBasis{N,G}) where {N,G}
    return (a.s == b.s) && (a.b == b.b)
end

printbasis(io::IO,b::VTI,e::String="e") = print(io,e,[subscripts[i] for i ∈ b]...)
print(io::IO, e::MultiBasis) = printbasis(io,basisindices(e))
show(io::IO, e::MultiBasis) = print(io,e)

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

function MultiBlade{T,N,G}(val::T,v::MultiBasis{N,G}) where {T,N,G}
    out = MultiBlade{T,N}(v.s,zeros(T,binomial(N,G)))
    out.v[basisindex(N,findall(v.b))] = val
    return out
end

MultiBlade(v::MultiBasis{N,G}) where {N,G} = MultiBlade{Int,N,G}(one(Int),v)

for var ∈ [[:T,:N,:G],[:T,:N],[:T]]
    @eval begin
        function MultiBlade{$(var...)}(v::MultiBasis{N,G}) where {T,N,G}
            return MultiBlade{T,N,G}(one(T),v)
        end
    end
end
for var ∈ [[:T,:N,:G],[:T,:N],[:T],[]]
    @eval begin
        function MultiBlade{$(var...)}(v::MultiValue{N,G,T}) where {T,N,G}
            return MultiBlade{T,N,G}(v.v,v.b)
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

function getindex(m::MultiVector{T,N},i::Int) where {T,N}
    0 <= i <= N || throw(BoundsError(m, i))
    r = binomsum(N,i)
    return @view m.v[r+1:r+binomial(N,i)]
end
getindex(m::MultiVector,i::Int,j::Int) = m[i][j]

setindex!(m::MultiVector{T},k::T,i::Int,j::Int) where T = (m[i][j] = k)

Base.firstindex(m::MultiVector) = 0
Base.lastindex(m::MultiVector{T,N}) where {T,N} = N

function (m::MultiVector{T,N})(g::Int) where {T,N}
    MultiBlade{T,N,g}(m.s,m[g])
end
function (m::MultiVector{T,N})(g::Int,i::Int) where {T,N}
    MultiValue{N,g,T}(m[g][i],MultiBasis{N,g}(m.s,combo(N,g)[i]))
end

MultiVector(s::Signature,v::T...) where T = MultiVector{T,intlog(length(v))}(s,v)

for var ∈ [[:T],[]]
    @eval begin
        MultiVector{$(var...)}(s::Signature,v::MVector{M,T}) where {T,M} = MultiVector{T,intlog(M)}(s,v)
        MultiVector{$(var...)}(s::Signature,v::Vector{T}) where T = MultiVector{T,intlog(length(v))}(s,v)
    end
end

function MultiVector{T,N,G}(val::T,v::MultiBasis{N,G}) where {T,N,G}
    out = MultiVector{T,N}(v.s,zeros(T,2^N))
    out.v[binomsum(N,G)+basisindex(N,findall(v.b))] = val
    return out
end

MultiVector(v::MultiBasis{N,G}) where {N,G} = MultiVector{Int,N,G}(one(Int),v)

for var ∈ [[:T,:N,:G],[:T,:N],[:T]]
    @eval begin
        function MultiVector{$(var...)}(v::MultiBasis{N,G}) where {T,N,G}
            return MultiVector{T,N,G}(one(T),v)
        end
    end
end
for var ∈ [[:T,:N,:G],[:T,:N],[:T],[]]
    @eval begin
        function MultiVector{$(var...)}(v::MultiValue{N,G,T}) where {T,N,G}
            return MultiVector{T,N,G}(v.v,v.b)
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

## Generic

typeval(m::MultiBasis) = Int
typeval(m::MultiValue{N,G,T}) where {N,G,T} = T
grade(m::AbstractTerm{N,G}) where {N,G} = G


## MultiGrade{N}

struct MultiGrade{N}
    v::Vector{<:AbstractTerm{N}}
end

convert(::Type{Vector{<:AbstractTerm{N}}},m::Tuple) where N = [m...]

MultiGrade{N}(v::T...) where T <: (AbstractTerm{N,G} where G) where N = MultiGrade{N}(v)
MultiGrade(v::T...) where T <: (AbstractTerm{N,G} where G) where N = MultiGrade{N}(v)

function bladevalues(s::Signature,m,N::Int,G::Int,T::Type)
    com = combo(N,G)
    out = MultiValue{N,G,T}[]
    for i ∈ 1:binomial(N,G)
        m[i] ≠ 0 && push!(out,MultiValue{N,G,T}(m[i],MultiBasis{N,G}(s,com[i])))
    end
    return out
end

function MultiGrade{N}(v::MultiVector{T,N}) where {T,N}
    MultiGrade{N}(vcat([bladevalues(v.s,v[g],N,g,T) for g ∈ 1:N]...))
end

function MultiGrade{N}(v::MultiBlade{T,N,G}) where {T,N,G}
    MultiGrade{N}(bladevalues(v.s,v,N,G,T))
end

MultiGrade(v::MultiVector{T,N}) where {T,N} = MultiGrade{N}(v)
MultiGrade(v::MultiBlade{T,N,G}) where {T,N,G} = MultiGrade{N}(v)

#=function MultiGrade{N}(v::(MultiBlade{T,N} where T <: Number)...) where N
    t = typeof.(v)
    MultiGrade{N}([bladevalues(v[i].s,v[i],N,t[i].parameters[3],t[i].parameters[1]) for i ∈ 1:length(v)])
end

MultiGrade(v::(MultiBlade{T,N} where T <: Number)...) where N = MultiGrade{N}(v)=#

function MultiVector{N}(v::MultiGrade{N}) where N
    T = promote_type(typeval.(v.v)...)
    g = grade.(v.v)
    s = typeof(v.v[1]) <: MultiBasis ? v.v[1].s : v.v[1].b.s
    out = MultiVector{T,N}(s,zeros(T,2^N))
    for k ∈ 1:length(v.v)
        (val,b) = typeof(v.v[k]) <: MultiBasis ? (one(T),v.v[k]) : (v.v[k].v,v.v[k].b)
        out.v[binomsum(N,g[k])+basisindex(N,basisindices(b))] = val
    end
    return out
end

MultiVector(v::MultiGrade{N}) where N = MultiVector{N}(v)

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
