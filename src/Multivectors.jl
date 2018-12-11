module Multivectors

#   This file is part of Multivectors.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

using Combinatorics, StaticArrays
using ComputedFieldTypes

import Base: print, show, getindex, setindex!, promote_rule, ==, convert
export AbstractTerm, Basis, MultiVector, MultiGrade, Signature, @S_str

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

struct Basis{N,G} <: AbstractTerm{N,G}
    s::Signature{N}
    b::BitArray{1}
end

VTI = Union{Vector{<:Integer},Tuple,NTuple}

basisindices(b::Basis) = findall(b.b)

function basisbits(d::Integer,b::VTI)
    out = falses(d)
    for k ∈ b
        out[k] = true
    end
    return out
end

Basis{N}(s::Signature,b::BitArray) where N = Basis{N,sum(b)}(s,b)
Basis(s::Signature,b::VTI) = Basis{length(s.b)}(s,b)
Basis(s::Signature,b::Integer...) = Basis{length(s.b)}(s,b)

for t ∈ [[:N],[:N,:G]]
    @eval begin
        function Basis{$(t...)}(s::Signature,b::VTI) where {$(t...)}
            Basis{$(t...)}(s,basisbits(N,b))
        end
        function Basis{$(t...)}(s::Signature,b::Integer...) where {$(t...)}
            Basis{$(t...)}(s,basisbits(N,b))
        end
    end
end

function ==(a::Basis{N,G},b::Basis{N,G}) where {N,G}
    return (a.s == b.s) && (a.b == b.b)
end

printbasis(io::IO,b::VTI,e::String="e") = print(io,e,[subscripts[i] for i ∈ b]...)
print(io::IO, e::Basis) = printbasis(io,basisindices(e))
show(io::IO, e::Basis) = print(io,e)

export @basis

macro basis(label,sig,str)
    s = Signature(str)
    N = length(str)
    lab = string(label)
    io = IOBuffer()
    els = Symbol[sig,label]
    exp = Expr[Expr(:(=),esc(sig),s),
        Expr(:(=),esc(label),Basis(s))]
    for i ∈ 1:N
        set = combo(N,i) |> collect
        for k ∈ 1:length(set)
            print(io,lab,set[k]...)
            sym = Symbol(String(take!(io)))
            push!(els,sym)
            push!(exp,Expr(:(=),esc(sym),Basis(s,set[k])))
        end
    end
    return Expr(:block,exp...,Expr(:tuple,esc.(els)...))
end

## S/MValue{N}

MSV = [:MValue,:SValue]

for Value ∈ MSV
    eval(Expr(:struct,Value ≠ :SValue,:($Value{N,G,T} <: AbstractTerm{N,G}),quote
        v::T
        b::$Basis{N,G}
    end))
end
for Value ∈ MSV
    @eval begin
        export $Value
        $Value(b::Basis{N,G}) where {N,G} = $Value{N,G,Int}(1,b)
        $Value{N}(b::Basis{N,G}) where {N,G} = $Value{N,G,Int}(1,b)
        $Value{N}(v,b::SValue{N,G}) where {N,G} = $Value{N,G}(v*b.v,b.b)
        $Value{N}(v,b::MValue{N,G}) where {N,G} = $Value{N,G}(v*b.v,b.b)
        $Value{N}(v::T,b::Basis{N,G}) where {N,G,T} = $Value{N,G,T}(v,b)
        $Value{N,G}(v::T,b::Basis{N,G}) where {N,G,T} = $Value{N,G,T}(v,b)
        $Value{N}(v::T) where {N,T} = $Value{N,0,T}(v,Basis{N}())
        $Value(v,b::AbstractTerm{N,G}) where {N,G} = $Value{N,G}(v,b)
        show(io::IO,m::$Value) = print(io,m.v,m.b)
    end
end

## S/MBlade{T,N}

MSB = [:MBlade,:SBlade]

for (Blade,vector,Value) ∈ [(MSB[1],:MVector,MSV[1]),(MSB[2],:SVector,MSV[2])]
    @eval begin
        @computed struct $Blade{T,N,G}
            s::Signature
            v::$vector{binomial(N,G),T}
        end

        export $Blade
        getindex(m::$Blade,i::Int) = m.v[i]
        setindex!(m::$Blade{T},k::T,i::Int) where T = (m.v[i] = k)
        Base.firstindex(m::$Blade) = 1
        Base.lastindex(m::$Blade{T,N,G}) where {T,N,G} = length(m.v)

        function (m::$Blade{T,N,G})(i::Integer,B::Type=SValue) where {T,N,G}
            if B ≠ SValue
                MValue{N,G,T}(m[i],Basis(m.s,combo(N,G)[i]))
            else
                MValue{N,G,T}(m[i],Basis(m.s,combo(N,G)[i]))
            end
        end

        function $Blade{T,N,G}(val::T,v::Basis{N,G}) where {T,N,G}
            out = $Blade{T,N}(v.s,zeros(T,binomial(N,G)))
            out.v[basisindex(N,findall(v.b))] = val
            return out
        end

        $Blade(v::Basis{N,G}) where {N,G} = $Blade{Int,N,G}(one(Int),v)

        function show(io::IO, m::$Blade{T,N,G}) where {T,N,G}
            set = combo(N,G)
            print(io,m.v[1])
            printbasis(io,set[1])
            for k ∈ 2:length(set)
                print(io,signbit(m.v[k]) ? " - " : " + ",abs(m.v[k]))
                printtbasis(io,set[k])
            end
        end
    end
    for var ∈ [[:T,:N,:G],[:T,:N],[:T]]
        @eval begin
            $Blade{$(var...)}(v::Basis{N,G}) where {T,N,G} = $Blade{T,N,G}(one(T),v)
        end
    end
    for var ∈ [[:T,:N,:G],[:T,:N],[:T],[]]
        @eval begin
            $Blade{$(var...)}(v::SValue{N,G,T}) where {T,N,G} = $Blade{T,N,G}(v.v,v.b)
            $Blade{$(var...)}(v::MValue{N,G,T}) where {T,N,G} = $Blade{T,N,G}(v.v,v.b)
        end
    end
end

## MultiVector{T,N}

@computed struct MultiVector{T,N}
    s::Signature
    v::StaticArray{Tuple{2^N},T,1}
end

MultiVector{T,N}(s::Signature,v::T...) where {T,N} = MultiVector{T,N}(s,SVector(v))
MultiVector{T,N}(s::Signature,v::Vector{T}) where {T,N} = MultiVector{T,N}(s,MVector(v))

function getindex(m::MultiVector{T,N},i::Int) where {T,N}
    0 <= i <= N || throw(BoundsError(m, i))
    r = binomsum(N,i)
    return @view m.v[r+1:r+binomial(N,i)]
end
getindex(m::MultiVector,i::Int,j::Int) = m[i][j]

setindex!(m::MultiVector{T},k::T,i::Int,j::Int) where T = (m[i][j] = k)

Base.firstindex(m::MultiVector) = 0
Base.lastindex(m::MultiVector{T,N}) where {T,N} = N

function (m::MultiVector{T,N})(g::Int,B::Type=SBlade) where {T,N}
    B ≠ SBlade ? MBlade{T,N,g}(m.s,m[g]) : SBlade{T,N,g}(m.s,m[g])
end
function (m::MultiVector{T,N})(g::Int,i::Int,B::Type=SValue) where {T,N}
    if B ≠ SValue
        MValue{N,g,T}(m[g][i],Basis{N,g}(m.s,combo(N,g)[i]))
    else
        SValue{N,g,T}(m[g][i],Basis{N,g}(m.s,combo(N,g)[i]))
    end
end

for var ∈ [[:T],[]]
    @eval begin
        MultiVector{$(var...)}(s::Signature,v::StaticArray{Tuple{M},T,1}) where {T,M} = MultiVector{T,intlog(M)}(s,v)
        MultiVector{$(var...)}(s::Signature,v::Vector{T}) where T = MultiVector{T,intlog(length(v))}(s,MVector(v))
        MultiVector{$(var...)}(s::Signature,v::T...) where T = MultiVector{T,intlog(length(v))}(s,SVector(v))
    end
end

function MultiVector{T,N}(val::T,v::Basis{N,G}) where {T,N,G}
    out = MultiVector{T,N}(v.s,zeros(T,2^N))
    out.v[binomsum(N,G)+basisindex(N,findall(v.b))] = val
    return out
end

MultiVector(v::Basis{N,G}) where {N,G} = MultiVector{Int,N}(one(Int),v)

for var ∈ [[:T,:N],[:T]]
    @eval begin
        function MultiVector{$(var...)}(v::Basis{N,G}) where {T,N,G}
            return MultiVector{T,N}(one(T),v)
        end
    end
end
for var ∈ [[:T,:N],[:T],[]]
    for (Value,Blade) ∈ [(MSV[1],MSB[1]),(MSV[2],MSB[2])]
        @eval begin
            function MultiVector{$(var...)}(v::$Value{N,G,T}) where {N,G,T}
                return MultiVector{T,N}(v.v,v.b)
            end
            function MultiVector{$(var...)}(v::$Blade{T,N,G}) where {T,N,G}
                out = MultiVector{T,N}(v.s,zeros(T,2^N))
                r = binomsum(N,G)
                out.v[r+1:r+binomial(N,G)] = v.v
                return out
            end
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

typeval(m::Basis) = Int
typeval(m::Union{MValue{N,G,T},SValue{N,G,T}}) where {N,G,T} = T
typeval(m::Union{MBlade{T},SBlade{T}}) where T = T
typeval(m::MultiVector{T}) where T = T
grade(m::AbstractTerm{N,G}) where {N,G} = G
grade(m::Union{MBlade{T,N,G},SBlade{T,N,G}}) where {T,N,G} = G
grade(m::Number) = 0

## MultiGrade{N}

struct MultiGrade{N}
    v::Vector{<:AbstractTerm{N}}
end

convert(::Type{Vector{<:AbstractTerm{N}}},m::Tuple) where N = [m...]

MultiGrade{N}(v::T...) where T <: (AbstractTerm{N,G} where G) where N = MultiGrade{N}(v)
MultiGrade(v::T...) where T <: (AbstractTerm{N,G} where G) where N = MultiGrade{N}(v)

function bladevalues(s::Signature,m,N::Int,G::Int,T::Type)
    com = combo(N,G)
    out = SValue{N,G,T}[]
    for i ∈ 1:binomial(N,G)
        m[i] ≠ 0 && push!(out,SValue{N,G,T}(m[i],Basis{N,G}(s,com[i])))
    end
    return out
end

function MultiGrade{N}(v::MultiVector{T,N}) where {T,N}
    MultiGrade{N}(vcat([bladevalues(v.s,v[g],N,g,T) for g ∈ 1:N]...))
end

MultiGrade(v::MultiVector{T,N}) where {T,N} = MultiGrade{N}(v)

for Blade ∈ MSB
    @eval begin
        function MultiGrade{N}(v::$Blade{T,N,G}) where {T,N,G}
            MultiGrade{N}(bladevalues(v.s,v,N,G,T))
        end
        MultiGrade(v::$Blade{T,N,G}) where {T,N,G} = MultiGrade{N}(v)
    end
end

#=function MultiGrade{N}(v::(MultiBlade{T,N} where T <: Number)...) where N
    t = typeof.(v)
    MultiGrade{N}([bladevalues(v[i].s,v[i],N,t[i].parameters[3],t[i].parameters[1]) for i ∈ 1:length(v)])
end

MultiGrade(v::(MultiBlade{T,N} where T <: Number)...) where N = MultiGrade{N}(v)=#

function MultiVector{T,N}(v::MultiGrade{N}) where {T,N}
    g = grade.(v.v)
    s = typeof(v.v[1]) <: Basis ? v.v[1].s : v.v[1].b.s
    out = MultiVector{T,N}(s,zeros(T,2^N))
    for k ∈ 1:length(v.v)
        (val,b) = typeof(v.v[k]) <: Basis ? (one(T),v.v[k]) : (v.v[k].v,v.v[k].b)
        out.v[binomsum(N,g[k])+basisindex(N,basisindices(b))] = convert(T,val)
    end
    return out
end

MultiVector{T}(v::MultiGrade{N}) where {T,N} = MultiVector{T,N}(v)
MultiVector(v::MultiGrade{N}) where N = MultiVector{promote_type(typeval.(v.v)...),N}(v)

function show(io::IO,m::MultiGrade)
    for k ∈ 1:length(m.v)
        x = m.v[k]
        t = (typeof(x) <: MValue) | (typeof(x) <: SValue)
        if t && signbit(x.v)
            print(io," - ")
            ax = abs(x.v)
            ax ≠ 1 && print(io,ax)
        else
            k ≠ 1 && print(io," + ")
            t && x.v ≠ 1 && print(io,x.v)
        end
        show(io,t ? x.b : x)
    end
end

include("algebra.jl")

end # module
