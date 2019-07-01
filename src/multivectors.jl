
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export TensorTerm, TensorMixed, Basis, MultiVector, MultiGrade

abstract type TensorTerm{V,G} <: TensorAlgebra{V} end
abstract type TensorMixed{T,V} <: TensorAlgebra{V} end

# symbolic print types

parany = (Expr,Any)
parsym = (Expr,Symbol)
parval = (Expr,)

## pseudoscalar

using LinearAlgebra
import LinearAlgebra: I
export UniformScaling, I

## MultiBasis{N}

struct Basis{V,G,B} <: TensorTerm{V,G}
    @pure Basis{V,G,B}() where {V,G,B} = new{V,G,B}()
end

@pure bits(b::Basis{V,G,B} where {V,G}) where B = B
Base.one(b::Type{Basis{V}}) where V = getbasis(V,bits(b))
Base.zero(V::VectorBundle) = 0*one(V)
Base.one(V::VectorBundle) = Basis{V}()

function getindex(b::Basis,i::Int)
    d = one(Bits) << (i-1)
    return (d & bits(b)) == d
end

getindex(b::Basis,i::UnitRange{Int}) = [getindex(b,j) for j ∈ i]
getindex(b::Basis{V},i::Colon) where V = [getindex(b,j) for j ∈ 1:ndims(V)]
Base.firstindex(m::Basis) = 1
Base.lastindex(m::Basis{V}) where V = ndims(V)
Base.length(b::Basis{V}) where V = ndims(V)

function Base.iterate(r::Basis, i::Int=1)
    Base.@_inline_meta
    length(r) < i && return nothing
    Base.unsafe_getindex(r, i), i + 1
end

@inline indices(b::Basis{V}) where V = indices(bits(b),ndims(V))

@pure Basis{V}() where V = getbasis(V,0)
@pure Basis{V}(i::Bits) where V = getbasis(V,i)
Basis{V}(b::BitArray{1}) where V = getbasis(V,bit2int(b))

for t ∈ ((:V,),(:V,:G))
    @eval begin
        function Basis{$(t...)}(b::DirectSum.VTI) where {$(t...)}
            Basis{V}(indexbits(ndims(V),b))
        end
        function Basis{$(t...)}(b::Int...) where {$(t...)}
            Basis{V}(indexbits(ndims(V),b))
        end
    end
end

==(a::Basis{V,G},b::Basis{V,G}) where {V,G} = bits(a) == bits(b)
==(a::Basis{V,G} where V,b::Basis{W,L} where W) where {G,L} = false
==(a::Basis{V,G},b::Basis{W,G}) where {V,W,G} = interop(==,a,b)

==(a::Number,b::TensorTerm{V,G} where V) where G = G==0 ? a==value(b) : 0==a==value(b)
==(a::TensorTerm{V,G} where V,b::Number) where G = G==0 ? value(a)==b : 0==value(a)==b

@inline show(io::IO, e::Basis) = DirectSum.printindices(io,vectorspace(e),bits(e))

## S/MBlade{N}

const MSB = (:MBlade,:SBlade)

for Blade ∈ MSB
    eval(Expr(:struct,Blade ≠ :SBlade,:($Blade{V,G,B,T} <: TensorTerm{V,G}),quote
        v::T
    end))
end
for Blade ∈ MSB
    @eval begin
        export $Blade
        @pure $Blade(b::Basis{V,G}) where {V,G} = $Blade{V}(b)
        @pure $Blade{V}(b::Basis{V,G}) where {V,G} = $Blade{V,G,b,Int}(1)
        $Blade(v,b::TensorTerm{V}) where V = $Blade{V}(v,b)
        $Blade{V}(v,b::SBlade{V,G}) where {V,G} = $Blade{V,G,basis(b)}(v*b.v)
        $Blade{V}(v,b::MBlade{V,G}) where {V,G} = $Blade{V,G,basis(b)}(v*b.v)
        $Blade{V}(v::T,b::Basis{V,G}) where {V,G,T} = $Blade{V,G,b,T}(v)
        $Blade{V,G}(v::T,b::Basis{V,G}) where {V,G,T} = $Blade{V,G,b,T}(v)
        $Blade{V,G}(v,b::SBlade{V,G}) where {V,G} = $Blade{V,G,basis(b)}(v*b.v)
        $Blade{V,G}(v,b::MBlade{V,G}) where {V,G} = $Blade{V,G,basis(b)}(v*b.v)
        $Blade{V,G,B}(v::T) where {V,G,B,T} = $Blade{V,G,B,T}(v)
        $Blade{V}(v::T) where {V,T} = $Blade{V,0,Basis{V}(),T}(v)
        show(io::IO,m::$Blade) = print(io,(valuetype(m)∉parany ? [m.v] : ['(',m.v,')'])...,basis(m))
    end
end

==(a::TensorTerm{V,G},b::TensorTerm{V,G}) where {V,G} = basis(a) == basis(b) ? value(a) == value(b) : 0 == value(a) == value(b)
==(a::TensorTerm,b::TensorTerm) = 0 == value(a) == value(b)

## S/MChain{T,N}

const MSC = (:MChain,:SChain)

for (Chain,vector,Blade) ∈ ((MSC[1],:MVector,MSB[1]),(MSC[2],:SVector,MSB[2]))
    @eval begin
        @computed struct $Chain{T,V,G} <: TensorMixed{T,V}
            v::$vector{binomial(ndims(V),G),T}
        end

        export $Chain
        getindex(m::$Chain,i::Int) = m.v[i]
        getindex(m::$Chain,i::UnitRange{Int}) = m.v[i]
        setindex!(m::$Chain{T},k::T,i::Int) where T = (m.v[i] = k)
        Base.firstindex(m::$Chain) = 1
        @pure Base.lastindex(m::$Chain{T,V,G}) where {T,V,G} = binomial(ndims(V),G)
        @pure Base.length(m::$Chain{T,V,G}) where {T,V,G} = binomial(ndims(V),G)
    end
    @eval begin
        function (m::$Chain{T,V,G})(i::Integer,B::Type=SBlade) where {T,V,G}
            if B ≠ SBlade
                MBlade{V,G,Basis{V}(indexbasis(ndims(V),G)[i]),T}(m[i])
            else
                SBlade{V,G,Basis{V}(indexbasis(ndims(V),G)[i]),T}(m[i])
            end
        end

        function $Chain{T,V,G}(val::T,v::Basis{V,G}) where {T,V,G}
            N = ndims(V)
            $Chain{T,V,G}(setblade!(zeros(mvec(N,G,T)),val,bits(v),Dimension{N}()))
        end

        $Chain(v::Basis{V,G}) where {V,G} = $Chain{Int,V,G}(one(Int),v)

        function show(io::IO, m::$Chain{T,V,G}) where {T,V,G}
            ib = indexbasis(ndims(V),G)
            @inbounds if T == Any && typeof(m.v[1]) ∈ parsym
                @inbounds typeof(m.v[1])∉parval ? print(io,m.v[1]) : print(io,"(",m.v[1],")")
            else
                @inbounds print(io,m.v[1])
            end
            @inbounds DirectSum.printindices(io,V,ib[1])
            for k ∈ 2:length(ib)
                if T == Any && typeof(m.v[k]) ∈ parsym
                    @inbounds typeof(m.v[k])∉parval ? print(io," + ",m.v[k]) : print(io," + (",m.v[k],")")
                else
                    @inbounds print(io,signbit(m.v[k]) ? " - " : " + ",abs(m.v[k]))
                end
                @inbounds DirectSum.printindices(io,V,ib[k])
            end
        end
        function ==(a::$Chain{S,V,G} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
            i = bladeindex(ndims(V),bits(basis(b)))
            @inbounds a[i] == value(b) && prod(a[1:i-1] .== 0) && prod(a[i+1:end] .== 0)
        end
        ==(a::T,b::$Chain{S,V} where S) where T<:TensorTerm{V} where V = b==a
        ==(a::$Chain{S,V} where S,b::T) where T<:TensorTerm{V} where V = prod(0==value(b).==value(a))
        ==(a::Number,b::$Chain{S,V,G} where {S,V}) where G = G==0 ? a==value(b)[1] : prod(0==a.==value(b))
        ==(a::$Chain{S,V,G} where {S,V},b::Number) where G = G==0 ? value(a)[1]==b : prod(0==b.==value(a))
    end
    for var ∈ ((:T,:V,:G),(:T,:V),(:T,))
        @eval begin
            $Chain{$(var...)}(v::Basis{V,G}) where {T,V,G} = $Chain{T,V,G}(one(T),v)
        end
    end
    for var ∈ [[:T,:V,:G],[:T,:V],[:T],[]]
        @eval begin
            $Chain{$(var...)}(v::SBlade{V,G,B,T}) where {T,V,G,B} = $Chain{T,V,G}(v.v,basis(v))
            $Chain{$(var...)}(v::MBlade{V,G,B,T}) where {T,V,G,B} = $Chain{T,V,G}(v.v,basis(v))
        end
    end
end
for (Chain,Other,Vec) ∈ ((MSC...,:MVector),(reverse(MSC)...,:SVector))
    for var ∈ ((:T,:V,:G),(:T,:V),(:T,),())
        @eval begin
            $Chain{$(var...)}(v::$Other{T,V,G}) where {T,V,G} = $Chain{T,V,G}($Vec{binomial(ndims(V),G),T}(v.v))
        end
    end
end
for (Chain,Vec1,Vec2) ∈ ((MSC[1],:SVector,:MVector),(MSC[2],:MVector,:SVector))
    @eval begin
        $Chain{T,V,G}(v::$Vec1{M,T} where M) where {T,V,G} = $Chain{T,V,G}($Vec2{binomial(ndims(V),G),T}(v))
    end
end
for Chain ∈ MSC, Other ∈ MSC
    @eval begin
        ==(a::$Chain{T,V,G},b::$Other{S,V,G}) where {T,V,G,S} = prod(a.v .== b.v)
        ==(a::$Chain{T,V} where T,b::$Other{S,V} where S) where V = prod(0 .==value(a)) && prod(0 .== value(b))
    end
end

## MultiVector{T,N}

struct MultiVector{T,V,E} <: TensorMixed{T,V}
    v::Union{MArray{Tuple{E},T,1,E},SArray{Tuple{E},T,1,E}}
end
MultiVector{T,V}(v::MArray{Tuple{E},T,1,E}) where {T,V,E} = MultiVector{T,V,E}(v)
MultiVector{T,V}(v::SArray{Tuple{E},T,1,E}) where {T,V,E} = MultiVector{T,V,E}(v)

function getindex(m::MultiVector{T,V},i::Int) where {T,V}
    N = ndims(V)
    0 <= i <= N || throw(BoundsError(m, i))
    r = binomsum(N,i)
    return @view m.v[r+1:r+binomial(N,i)]
end
getindex(m::MultiVector,i::Int,j::Int) = m[i][j]
setindex!(m::MultiVector{T},k::T,i::Int,j::Int) where T = (m[i][j] = k)
Base.firstindex(m::MultiVector) = 0
Base.lastindex(m::MultiVector{T,V} where T) where V = ndims(V)

(m::MultiVector{T,V})(g::Int,b::Type{B}=SChain) where {T,V,B} = m(Dim{g}(),b)
function (m::MultiVector{T,V})(::Dim{g},::Type{B}=SChain) where {T,V,g,B}
    B ≠ SChain ? MChain{T,V,g}(m[g]) : SChain{T,V,g}(m[g])
end
function (m::MultiVector{T,V})(g::Int,i::Int,::Type{B}=SBlade) where {T,V,B}
    if B ≠ SBlade
        MBlade{V,g,Basis{V}(indexbasis(ndims(V),g)[i]),T}(m[g][i])
    else
        SBlade{V,g,Basis{V}(indexbasis(ndims(V),g)[i]),T}(m[g][i])
    end
end

MultiVector{V}(v::StaticArray{Tuple{M},T,1}) where {V,T,M} = MultiVector{T,V}(v)
for var ∈ ((:T,:V),(:V,))
    @eval begin
        MultiVector{$(var...)}(v::SizedArray) where {T,V} = MultiVector{T,V}(SVector{1<<ndims(V),T}(v))
        MultiVector{$(var...)}(v::Vector{T}) where {T,V} = MultiVector{T,V}(SVector{1<<ndims(V),T}(v))
        MultiVector{$(var...)}(v::T...) where {T,V} = MultiVector{T,V}(SVector{1<<ndims(V),T}(v))
        function MultiVector{$(var...)}(val::T,v::Basis{V,G}) where {T,V,G}
            N = ndims(V)
            MultiVector{T,V}(setmulti!(zeros(mvec(N,T)),val,bits(v),Dimension{N}()))
        end
    end
end
function MultiVector(val::T,v::Basis{V,G}) where {T,V,G}
    N = ndims(V)
    MultiVector{T,V}(setmulti!(zeros(mvec(N,T)),val,bits(v),Dimension{N}()))
end

MultiVector(v::Basis{V,G}) where {V,G} = MultiVector{Int,V}(one(Int),v)

for var ∈ ((:T,:V),(:T,))
    @eval begin
        function MultiVector{$(var...)}(v::Basis{V,G}) where {T,V,G}
            return MultiVector{T,V}(one(T),v)
        end
    end
end
for var ∈ ((:T,:V),(:T,),())
    for (Blade,Chain) ∈ ((MSB[1],MSC[1]),(MSB[2],MSC[2]))
        @eval begin
            function MultiVector{$(var...)}(v::$Blade{V,G,B,T}) where {V,G,B,T}
                return MultiVector{T,V}(v.v,basis(v))
            end
            function MultiVector{$(var...)}(v::$Chain{T,V,G}) where {T,V,G}
                N = ndims(V)
                out = zeros(mvec(N,T))
                r = binomsum(N,G)
                @inbounds out[r+1:r+binomial(N,G)] = v.v
                return MultiVector{T,V}(out)
            end
        end
    end
end

function show(io::IO, m::MultiVector{T,V}) where {T,V}
    N = ndims(V)
    basis_count = true
    print(io,m[0][1])
    bs = binomsum_set(N)
    for i ∈ 2:N+1
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            @inbounds s = k+bs[i]
            @inbounds if m.v[s] ≠ 0
                @inbounds if T == Any && typeof(m.v[s]) ∈ parsym
                    @inbounds typeof(m.v[s])∉parval ? print(io," + ",m.v[s]) : print(io," + (",m.v[s],")")
                else
                    @inbounds print(io,signbit(m.v[s]) ? " - " : " + ",abs(m.v[s]))
                end
                @inbounds DirectSum.printindices(io,V,ib[k])
                basis_count = false
            end
        end
    end
    basis_count && print(io,pre[1]*'⃖')
end

==(a::MultiVector{T,V},b::MultiVector{S,V}) where {T,V,S} = prod(a.v .== b.v)

for Chain ∈ MSC
    @eval begin
        function ==(a::MultiVector{T,V},b::$Chain{S,V,G}) where {T,V,S,G}
            N = ndims(V)
            r,R = binomsum(N,G), N≠G ? binomsum(N,G+1) : 2^N+1
            prod(a[G] .== b.v) && prod(a.v[1:r] .== 0) && prod(a.v[R+1:end] .== 0)
        end
        ==(a::$Chain{T,V,G},b::MultiVector{S,V}) where {T,V,S,G} = b == a
    end
end

function ==(a::MultiVector{S,V} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
    i = basisindex(ndims(V),bits(basis(b)))
    @inbounds a.v[i] == value(b) && prod(a.v[1:i-1] .== 0) && prod(a.v[i+1:end] .== 0)
end
==(a::T,b::MultiVector{S,V} where S) where T<:TensorTerm{V} where V = b==a
==(a::Number,b::MultiVector{S,V,G} where {S,V}) where G = (v=value(b);(a==v[1])*prod(0 .== v[2:end]))
==(a::MultiVector{S,V,G} where {S,V},b::Number) where G = b == a

## Generic

import Base: isinf, isapprox
import AbstractTensors: scalar, involute, unit, even, odd
export basis, grade, hasinf, hasorigin, isorigin, scalar, norm

const VBV = Union{MBlade,SBlade,MChain,SChain,MultiVector}

@pure valuetype(::Basis) = Int
@pure valuetype(::Union{MBlade{V,G,B,T},SBlade{V,G,B,T}} where {V,G,B}) where T = T
@pure valuetype(::TensorMixed{T}) where T = T
@inline value(::Basis,T=Int) = one(T)
@inline value(m::VBV,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@pure basis(m::Basis) = m
@pure basis(m::Union{MBlade{V,G,B},SBlade{V,G,B}}) where {V,G,B} = B
@pure grade(m::TensorTerm{V,G} where V) where G = G
@pure grade(m::Union{MChain{T,V,G},SChain{T,V,G}} where {T,V}) where G = G
@pure grade(m::Number) = 0

@pure isinf(e::Basis{V}) where V = hasinf(e) && count_ones(bits(e)) == 1
@pure hasinf(e::Basis{V}) where V = hasinf(V) && isodd(bits(e))

@pure isorigin(e::Basis{V}) where V = hasorigin(V) && count_ones(bits(e))==1 && e[hasinf(V)+1]
@pure hasorigin(e::Basis{V}) where V = hasorigin(V) && (hasinf(V) ? e[2] : isodd(bits(e)))
@pure hasorigin(t::Union{MBlade,SBlade}) = hasorigin(basis(t))
@pure hasorigin(m::TensorAlgebra) = hasorigin(vectorspace(m))

for A ∈ (:TensorTerm,MSC...), B ∈ (:TensorTerm,MSC...)
    @eval isapprox(a::S,b::T) where {S<:$A,T<:$B} = vectorspace(a)==vectorspace(b) && (grade(a)==grade(b) ? norm(a)≈norm(b) : (iszero(a) && iszero(b)))
end
function isapprox(a::S,b::T) where {S<:TensorAlgebra,T<:TensorAlgebra}
    rtol = Base.rtoldefault(valuetype(a), valuetype(b), 0)
    return norm(a-b) <= rtol * max(norm(a), norm(b))
end
isapprox(a::S,b::T) where {S<:MultiVector,T<:MultiVector} = vectorspace(a)==vectorspace(b) && value(a) ≈ value(b)
isapprox(a::S,b::T) where {S<:TensorAlgebra{V},T<:Number} where V =isapprox(a,SBlade{V}(b))
isapprox(a::S,b::T) where {S<:Number,T<:TensorAlgebra} = isapprox(b,a)

"""
    scalar(multivector)
    
Return the scalar (grade 0) part of any multivector.
"""
@inline scalar(t::T) where T<:TensorTerm{V,0} where V = t
@inline scalar(t::T) where T<:TensorTerm{V} where V = zero(V)
@inline scalar(t::MultiVector{T,V}) where {T,V} = SBlade{V}(t.v[1])
for Chain ∈ MSC
    @eval begin
        @inline scalar(t::$Chain{T,V,0}) where {T,V} = SBlade{V}(t.v[1])
        @inline scalar(t::$Chain{T,V} where T) where V = zero(V)
    end
end

@inline vector(t::T) where T<:TensorTerm{V,1} where V = t
@inline vector(t::T) where T<:TensorTerm{V} where V = zero(V)
@inline vector(t::MultiVector{T,V}) where {T,V} = SChain{T,V,1}(t[1])
for Chain ∈ MSC
    @eval begin
        @inline vector(t::$Chain{T,V,1} where {T,V}) = t
        @inline vector(t::$Chain{T,V} where T) where V = zero(V)
    end
end

@inline volume(t::T) where T<:TensorTerm{V,G} where {V,G} = G == ndims(V) ? t : zero(V)
@inline volume(t::MultiVector{T,V}) where {T,V} = SBlade{V}(t.v[end])
for Chain ∈ MSC
    @eval begin
        @inline volume(t::$Chain{T,V,G} where T) where {V,G} = G == ndims(V) ? t : zero(V)
    end
end

@inline isscalar(t::T) where T<:TensorTerm = grade(t) == 0 || iszero(t)
@inline isscalar(t::T) where T<:SChain = grade(t) == 0 || iszero(t)
@inline isscalar(t::T) where T<:MChain = grade(t) == 0 || iszero(t)
@inline isscalar(t::MultiVector) = norm(t) ≈ scalar(t)

## MultiGrade{N}

struct MultiGrade{V} <: TensorAlgebra{V}
    v::Vector{<:TensorTerm{V}}
end

#convert(::Type{Vector{<:TensorTerm{V}}},m::Tuple) where V = [m...]

MultiGrade{V}(v::T...) where T <: (TensorTerm{V,G} where G) where V = MultiGrade{V}(v)
MultiGrade(v::T...) where T <: (TensorTerm{V,G} where G) where V = MultiGrade{V}(v)

function bladevalues(V::VectorBundle{N},m,G::Int,T::Type) where N
    com = indexbasis(N,G)
    out = (SBlade{V,G,B,T} where B)[]
    for i ∈ 1:binomial(N,G)
        m[i] ≠ 0 && push!(out,SBlade{V,G,Basis{V,G}(com[i]),T}(m[i]))
    end
    return out
end

MultiGrade(v::MultiVector{T,V}) where {T,V} = MultiGrade{V}(vcat([bladevalues(V,v[g],g,T) for g ∈ 1:ndims(V)]...))

for Chain ∈ MSC
    @eval begin
        MultiGrade(v::$Chain{T,V,G}) where {T,V,G} = MultiGrade{V}(bladevalues(V,v,G,T))
    end
end

#=function MultiGrade{V}(v::(MultiChain{T,V} where T <: Number)...) where V
    sigcheck(v.s,V)
    t = typeof.(v)
    MultiGrade{V}([bladevalues(V,v[i],N,t[i].parameters[3],t[i].parameters[1]) for i ∈ 1:length(v)])
end

MultiGrade(v::(MultiChain{T,N} where T <: Number)...) where N = MultiGrade{N}(v)=#

function MultiVector{T,V}(v::MultiGrade{V}) where {T,V}
    N = ndims(V)
    sigcheck(v.s,V)
    g = grade.(v.v)
    out = zeros(mvec(N,T))
    for k ∈ 1:length(v.v)
        @inbounds (val,b) = typeof(v.v[k]) <: Basis ? (one(T),v.v[k]) : (v.v[k].v,basis(v.v[k]))
        setmulti!(out,convert(T,val),bits(b),Dimension{N}())
    end
    return MultiVector{T,V}(out)
end

MultiVector{T}(v::MultiGrade{V}) where {T,V} = MultiVector{T,V}(v)
MultiVector(v::MultiGrade{V}) where V = MultiVector{promote_type(typeval.(v.v)...),V}(v)

function show(io::IO,m::MultiGrade)
    for k ∈ 1:length(m.v)
        x = m.v[k]
        t = (typeof(x) <: MBlade) | (typeof(x) <: SBlade)
        if t && signbit(x.v)
            print(io," - ")
            ax = abs(x.v)
            ax ≠ 1 && print(io,ax)
        else
            k ≠ 1 && print(io," + ")
            t && x.v ≠ 1 && print(io,x.v)
        end
        show(io,t ? basis(x) : x)
    end
end

## Adjoint

import Base: adjoint # conj

function adjoint(b::Basis{V,G,B}) where {V,G,B}
    Basis{dual(V)}(mixedmode(V)<0 ? dual(V,B) : B)
end

## conversions

@inline (V::Signature)(s::UniformScaling{T}) where T = SBlade{V}(T<:Bool ? (s.λ ? one(Int) : -(one(Int))) : s.λ,getbasis(V,(one(T)<<(ndims(V)-diffmode(V)))-1))

@pure function (W::Signature)(b::Basis{V}) where V
    V==W && (return b)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    if WC<0 && VC≥0
        N = ndims(V)
        dm = diffmode(V)
        if dm≠0
            D = DirectSum.diffmask(V)
            m = (~D)&bits(b)
            d = (D&bits(b))<<(N-dm+(VC>0 ? dm : 0))
            return getbasis(W,(VC>0 ? m<<(N-dm) : m)⊻d)
        else
            return getbasis(W,VC>0 ? bits(b)<<(N-dm) : bits(b))
        end
    else
        throw(error("arbitrary VectorBundle intersection not yet implemented."))
    end
end
(W::Signature)(b::SBlade) = SBlade{W}(value(b),W(basis(b)))
(W::Signature)(b::MBlade) = MBlade{W}(value(b),W(basis(b)))

for Chain ∈ MSC
    @eval begin
        function (W::Signature)(b::$Chain{T,V,G}) where {T,V,G}
            V==W && (return b)
            !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
            WC,VC = mixedmode(W),mixedmode(V)
            #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
            #    return V0
            if WC<0 && VC≥0
                N,M = ndims(V),ndims(W)
                out = zeros(mvec(M,G,T))
                ib = indexbasis(N,G)
                for k ∈ 1:length(ib)
                    @inbounds if b[k] ≠ 0
                        @inbounds setblade!(out,b[k],VC>0 ? ib[k]<<N : ib[k],Dimension{M}())
                    end
                end
                return $Chain{T,W,G}(out)
            else
                throw(error("arbitrary VectorBundle intersection not yet implemented."))
            end
        end

    end
end

function (W::Signature)(m::MultiVector{T,V}) where {T,V}
    V==W && (return m)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    if WC<0 && VC≥0
        N,M = ndims(V),ndims(W)
        out = zeros(mvec(M,T))
        bs = binomsum_set(N)
        for i ∈ 1:N+1
            ib = indexbasis(N,i-1)
            for k ∈ 1:length(ib)
                @inbounds s = k+bs[i]
                @inbounds if m.v[s] ≠ 0
                    @inbounds setmulti!(out,m.v[s],VC>0 ? ib[k]<<N : ib[k],Dimension{M}())
                end
            end
        end
        return MultiVector{T,W}(out)
    else
        throw(error("arbitrary VectorBundle intersection not yet implemented."))
    end
end

