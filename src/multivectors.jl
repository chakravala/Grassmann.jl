
#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed


import Base: print, show, getindex, setindex!, promote_rule, ==, convert
export AbstractTerm, Basis, MultiVector, MultiGrade, Signature, VectorSpace, @S_str, @V_str

abstract type AbstractTerm{N,G} end

## Signature{N}

struct Signature{N}
    b::BitArray{1}
    i::UInt16
    Signature{N}(b::BitArray{1}) where N = new{N}(b,bit2int(b[1:N]))
    Signature{N}(b::Array{Bool,1}) where N = Signature{N}(convert(BitArray{1},b))
end

getindex(s::Signature,i::Union{Int,UnitRange{Int}}) = s.b[i]
getindex(s::Signature{N},i::Colon) where N = s.b[1:N]
setindex!(s::Signature,k::Bool,i::Int) = (s.b[i] = k)
Base.firstindex(m::Signature) = 1
Base.lastindex(m::Signature{N}) where N = N+2
Base.length(s::Signature{N}) where N = N

Signature{N}(s::String) where N = Signature{N}(push!([k=='-' for k∈s],s[1]=='ϵ',s[(s[1]=='ϵ')+1]=='o'))
Signature{N}(s::String,b::Bool...) where N = Signature{N}(push!([k=='-' for k∈s],b...))
@inline Signature(s::String) = Signature{length(s)}(s)

@inline sig(s::Bool) = s ? '-' : '+'

@inline function print(io::IO,s::Signature{N}) where N
    s[end-1] && print(io,'ϵ')
    s[end] && print(io,'o')
    print(io,sig.(s[s[end-1]+s[end]+1:N])...)
end

show(io::IO,s::Signature{N}) where N = print(io,s)

macro S_str(str)
    Signature(str)
end

# VectorSpace

struct VectorSpace{N,D,O}
    s::Signature{N}
end
getindex(vs::VectorSpace{N,D,O},i::Union{Int,UnitRange{Int}}) where {N,D,O} = i > N ? Bool(i≠N+1 ? O : D) : vs.s.b[i]
getindex(vs::VectorSpace{N},i::Colon) where N = vs.s[:]
setindex!(s::VectorSpace,k::Bool,i::Int) = (s.s.b[i] = k)
Base.firstindex(m::VectorSpace) = 1
Base.lastindex(m::VectorSpace{N}) where N = N+2
Base.length(s::VectorSpace{N}) where N = N

hasdual(vs::VectorSpace{N,D}) where {N,D} = Bool(D)
hasorigin(vs::VectorSpace{N,D,O}) where {N,D,O} = Bool(O)

VectorSpace{N}(s::Signature{N}) where N = VectorSpace{N,Int(s[end-1]),Int(s[end])}(s)
VectorSpace(s::Signature{N}) where N = VectorSpace{N}(s)

VectorSpace{N,D,O}(str::String) where {N,D,O} = VectorSpace{N,D,O}(Signature{N}(replace(replace(str,'ϵ'=>'+'),'o'=>'+'),Bool(D),Bool(O)))
VectorSpace(n::Int,d::Int=0,o::Int=0) = VectorSpace(int2vs(n,d,o))
VectorSpace{N}(n::Int,d::Int=0,o::Int=0) where N = VectorSpace{N}(int2vs(n,d,o))
VectorSpace(str::String) = VectorSpace{length(str)}(str)
function VectorSpace{N}(str::String) where N
    try
        VectorSpace(parse(Int,str))
    catch
        VectorSpace{N,Int('ϵ'∈str),Int('o'∈str)}(str)
    end
end

function int2vs(n::Int,d::Int=0,o::Int=0)
    str = join(['+' for s ∈ 1:n-d-o])
    o>0 && (str = 'o'*str)
    d>0 && (str = 'ϵ'*str)
    return str
end

print(io::IO,vs::VectorSpace{N}) where N = print(io,vs.s)
show(io::IO,vs::VectorSpace{N}) where N = print(io,vs)

macro V_str(str)
    VectorSpace(str)
end

## MultiBasis{N}

struct Basis{N,G} <: AbstractTerm{N,G}
    s::Signature{N}
    i::UInt16
end

getindex(b::Basis,i::Int) = digits(b.i,base=2)[i] == 1
Base.firstindex(m::Basis) = 1
Base.lastindex(m::Basis{N}) where N = N
Base.length(b::Basis{N}) where N = N

function Base.iterate(r::Basis, i::Int=1)
    Base.@_inline_meta
    length(r) < i && return nothing
    Base.unsafe_getindex(r, i), i + 1
end

const VTI = Union{Vector{Int},Tuple,NTuple}

@inline basisindices(s::Signature{N},G::Int,i::Int) where N = indexbasis(N,G)[i]
@inline basisindices(b::Basis) = b.i
@inline shiftbasis(b::Basis{N}) where N = shiftbasis(b.s,findall(digits(b.i,base=2).==1))

function shiftbasis(s::Signature{N},set::Vector{Int}) where N
    if !isempty(set)
        k = 1
        s[end-1] && set[1] == 1 && (set[1] = -1; k += 1)
        shift = sum(s[end-1:end])
        s[end] && length(set)>=k && set[k]==shift && (set[k]=0;k+=1)
        shift > 0 && (set[k:end] .-= shift)
    end
    return set
end

@inline function basisbits(d::Integer,b::VTI)
    out = falses(d)
    for k ∈ b
        out[k] = true
    end
    return out
end

Basis{N,G}(s::Signature{N},b::BitArray{1}) where {N,G} = Basis{N,G}(s,bit2int(b))
Basis{N}(s::Signature{N},i::UInt16) where N = new{N,basisgrade(i)}(s,i)
Basis{N}(s::Signature{N},b::BitArray) where N = Basis{N,sum(b)}(s,b)
Basis(s::Signature{N},b::VTI) where N = Basis{N}(s,b)
Basis(s::Signature{N},b::Int...) where N = Basis{N}(s,b)

for t ∈ [[:N],[:N,:G]]
    @eval begin
        function Basis{$(t...)}(s::Signature{N},b::VTI) where {$(t...)}
            Basis{$(t...)}(s,basisbits(N,b))
        end
        function Basis{$(t...)}(s::Signature{N},b::Int...) where {$(t...)}
            Basis{$(t...)}(s,basisbits(N,b))
        end
    end
end

function ==(a::Basis{N,G},b::Basis{N,G}) where {N,G}
    return (a.i == b.i) && (a.s == b.s)
end

@inline printbasis(io::IO,b::VTI,e::String="e") = print(io,e,[subscripts[i] for i ∈ b]...)
@inline print(io::IO, e::Basis) = printbasis(io,shiftbasis(e))
show(io::IO, e::Basis) = print(io,e)

function generate(s::VectorSpace{N},label::Symbol) where N
    lab = string(label)
    io = IOBuffer()
    els = Symbol[label]
    exp = Basis{N}[Basis{N}(s.s)]
    for i ∈ 1:N
        set = combo(N,i)
        for k ∈ 1:length(set)
            sk = shiftbasis(s.s,deepcopy(set[k]))
            print(io,lab,[j≠0 ? (j > 0 ? j : 'ϵ') : 'o' for j∈sk]...)
            push!(els,Symbol(String(take!(io))))
            push!(exp,Basis(s.s,set[k]))
        end
    end
    return exp,els
end

export @basis

macro basis(label,sig,str)
    N = length(str)
    s = VectorSpace{N}(str)
    basis,sym = generate(s,label)
    exp = Expr[Expr(:(=),esc(sig),s),
        Expr(:(=),esc(label),basis[1])]
    for i ∈ 2:2^N
        push!(exp,Expr(:(=),esc(sym[i]),basis[i]))
    end
    return Expr(:block,exp...,Expr(:tuple,esc(sig),esc.(sym)...))
end

## S/MValue{N}

const MSV = [:MValue,:SValue]

for Value ∈ MSV
    eval(Expr(:struct,Value ≠ :SValue,:($Value{N,G,T} <: AbstractTerm{N,G}),quote
        v::T
        b::Basis{N,G}
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
        $Value{N}(s,v::T) where {N,T} = $Value{N,0,T}(v,Basis{N,0}(s))
        $Value(v,b::AbstractTerm{N,G}) where {N,G} = $Value{N,G}(v,b)
        show(io::IO,m::$Value) = print(io,m.v,m.b)
    end
end

## Grade{G}

struct Grade{G} end

## Dimension{N}

struct Dimension{N} end

## S/MBlade{T,N}

const MSB = [:MBlade,:SBlade]

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
        Base.length(s::$Blade{T,N,G}) where {T,N,G} = length(m.v)

        function (m::$Blade{T,N,G})(i::Integer,B::Type=SValue) where {T,N,G}
            if B ≠ SValue
                MValue{N,G,T}(m[i],Basis(m.s,UInt16(indexbasis(N,G)[i])))
            else
                SValue{N,G,T}(m[i],Basis(m.s,UInt16(indexbasis(N,G)[i])))
            end
        end

        function $Blade{T,N,G}(val::T,v::Basis{N,G}) where {T,N,G}
            SBlade{T,N}(v.s,setblade!(zeros(T,binomial(N,G)),val,basisindices(v),Dimension{N}()))
        end

        $Blade(v::Basis{N,G}) where {N,G} = $Blade{Int,N,G}(one(Int),v)

        function show(io::IO, m::$Blade{T,N,G}) where {T,N,G}
            set = combo(N,G)
            print(io,m.v[1])
            printbasis(io,set[1])
            for k ∈ 2:length(set)
                print(io,signbit(m.v[k]) ? " - " : " + ",abs(m.v[k]))
                printbasis(io,shiftbasis(m.s,deepcopy(set[k])))
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
    v::Union{MArray{Tuple{2^N},T,1,2^N},SArray{Tuple{2^N},T,1,2^N}}
end

MultiVector{T,N}(s::Signature,v::T...) where {T,N} = MultiVector{T,N}(s,SVector{2^N,T}(v))
MultiVector{T,N}(s::Signature,v::Vector{T}) where {T,N} = MultiVector{T,N}(s,SVector{2^N,T}(v))

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
        MValue{N,g,T}(m[g][i],Basis{N,g}(m.s,UInt16(indexbasis(N,g)[i])))
    else
        SValue{N,g,T}(m[g][i],Basis{N,g}(m.s,UInt16(indexbasis(N,g)[i])))
    end
end

for var ∈ [[:T],[]]
    @eval begin
        MultiVector{$(var...)}(s::Signature,v::StaticArray{Tuple{M},T,1}) where {T,M} = MultiVector{T,intlog(M)}(s,v)
        MultiVector{$(var...)}(s::Signature,v::Vector{T}) where T = MultiVector{T,intlog(length(v))}(s,SVector(v))
        MultiVector{$(var...)}(s::Signature,v::T...) where T = MultiVector{T,intlog(length(v))}(s,SVector(v))
    end
end

function MultiVector{T,N}(val::T,v::Basis{N,G}) where {T,N,G}
    MultiVector{T,N}(v.s,setmulti!(zeros(T,2^N),val,basisindices(v),Dimension{N}()))
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
                out = zeros(T,2^N)
                r = binomsum(N,G)
                out.v[r+1:r+binomial(N,G)] = v.v
                return MultiVector{T,N}(v.s,out)
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
                printbasis(io,shiftbasis(m.s,deepcopy(set[k])))
            end
        end
    end
end

## Generic

const VBV = Union{MValue,SValue,MBlade,SBlade,MultiVector}

valuetype(m::Basis) = Int
valuetype(m::Union{MValue{N,G,T},SValue{N,G,T}}) where {N,G,T} = T
valuetype(m::Union{MBlade{T},SBlade{T}}) where T = T
valuetype(m::MultiVector{T}) where T = T
value(m::Basis,T=Int) = one(T)
value(m::VBV,T::DataType=Nothing) = T≠Nothing ? m.v : convert(T,m.v)
sig(m::Basis) = m.s
sig(m::Union{MValue,SValue}) = m.b.s
sig(m::Union{MBlade,SBlade,MultiVector}) = m.s
basis(m::Basis) = m
basis(m::Union{MValue,SValue}) = m.b
grade(m::AbstractTerm{N,G}) where {N,G} = G
grade(m::Union{MBlade{T,N,G},SBlade{T,N,G}}) where {T,N,G} = G
grade(m::Number) = 0

isdual(e::Basis) = hasdual(e) && sum(digits(e.i,base=2)) == 1
hasdual(e::Basis) = e.s[end-1] && isodd(e.i)

isorigin(e::Basis) = e.s[end] && (d=digits(e.i,base=2); d[e.s[end-1]+1]≠0 && sum(d) == 1)
hasorigin(e::Basis) = e.s[end] && (e.s[end-1] ? e[2] : isodd(e.i))
hasorigin(s::Signature) = s[end]
hasorigin(t::Union{MValue,SValue}) = hasorigin(t.b)
hasorigin(m::VBV) = hasorigin(sig(m))

## MultiGrade{N}

struct MultiGrade{N}
    v::Vector{<:AbstractTerm{N}}
end

convert(::Type{Vector{<:AbstractTerm{N}}},m::Tuple) where N = [m...]

MultiGrade{N}(v::T...) where T <: (AbstractTerm{N,G} where G) where N = MultiGrade{N}(v)
MultiGrade(v::T...) where T <: (AbstractTerm{N,G} where G) where N = MultiGrade{N}(v)

function bladevalues(s::Signature,m,N::Int,G::Int,T::Type)
    com = indexbasis(N,G)
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
    out = zeros(T,2^N)
    for k ∈ 1:length(v.v)
        (val,b) = typeof(v.v[k]) <: Basis ? (one(T),v.v[k]) : (v.v[k].v,v.v[k].b)
        setmulti!(out,convert(T,val),basisindices(b),Dimension{N}())
    end
    return MultiVector{T,N}(s,out)
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

