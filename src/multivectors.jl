
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export TensorTerm, TensorGraded, TensorMixed, Basis, MultiVector, SparseChain, MultiGrade

abstract type TensorGraded{V,G} <: TensorAlgebra{V} end
abstract type TensorTerm{V,G} <: TensorGraded{V,G} end
abstract type TensorMixed{V} <: TensorAlgebra{V} end

function Base.show_unquoted(io::IO, z::T, ::Int, prec::Int) where T<:TensorAlgebra
    if !(T<:TensorTerm) && Base.operator_precedence(:+) <= prec
        print(io, "(")
        show(io, z)
        print(io, ")")
    else
        show(io, z)
    end
end

# number fields

Fields = (Real,Complex)

# symbolic print types

parval = (Expr,Complex,Rational,TensorAlgebra)
parsym = (Symbol,parval...)

## pseudoscalar

import LinearAlgebra
import LinearAlgebra: I, UniformScaling
export UniformScaling, I

## Basis{V,G,B}

"""
    Basis{V,G,B} <: TensorTerm{V,G} <: TensorGraded{V,G} <: TensorAlgebra{V}

Basis type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, bits `B::UInt64`.
"""
struct Basis{V,G,B} <: TensorTerm{V,G}
    @pure Basis{V,G,B}() where {V,G,B} = new{V,G,B}()
end

@pure bits(b::Basis{V,G,B} where {V,G}) where B = B
Base.one(b::Type{Basis{V}}) where V = getbasis(V,bits(b))
Base.zero(V::Manifold) = 0*one(V)
Base.one(V::Manifold) = Basis{V}()

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

for T ∈ Fields
    @eval begin
        ==(a::T,b::TensorTerm{V,G} where V) where {T<:$T,G} = G==0 ? a==value(b) : 0==a==value(b)
        ==(a::TensorTerm{V,G} where V,b::T) where {T<:$T,G} = G==0 ? value(a)==b : 0==value(a)==b
    end
end

@inline show(io::IO, e::Basis) = DirectSum.printindices(io,vectorspace(e),bits(e))

## Simplex{V,G,B,𝕂}

"""
    Simplex{V,G,B,𝕂} <: TensorTerm{V,G} <: TensorGraded{V,G} <: TensorAlgebra{V}

Simplex type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, basis `B::Basis{V,G}`, scalar field `𝕂::Type`.
"""
struct Simplex{V,G,B,T} <: TensorTerm{V,G}
    v::T
    Simplex{A,B,C,D}(t::E) where E<:D where {A,B,C,D} = new{A,B,C,D}(t)
    Simplex{A,B,C,D}(t::E) where E<:TensorAlgebra{A} where {A,B,C,D} = new{A,B,C,D}(t)
end

export Simplex
@pure Simplex(b::Basis{V,G}) where {V,G} = Simplex{V}(b)
@pure Simplex{V}(b::Basis{V,G}) where {V,G} = Simplex{V,G,b,Int}(1)
Simplex{V}(v::T) where {V,T} = Simplex{V,0,Basis{V}(),T}(v)
Simplex{V}(v::S) where S<:TensorTerm where V = v
Simplex{V,G,B}(v::T) where {V,G,B,T} = Simplex{V,G,B,T}(v)
Simplex(v,b::S) where S<:TensorTerm{V} where V = Simplex{V}(v,b)
Simplex{V}(v,b::S) where S<:TensorAlgebra where V = v*b
Simplex{V}(v,b::Basis{V,G}) where {V,G} = Simplex{V,G}(v,b)
Simplex{V}(v,b::Basis{W,G}) where {V,W,G} = Simplex{V,G}(v,b)
function Simplex{V,G}(v::T,b::Basis{V,G}) where {V,G,T}
    order(v)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,b,T}(v)
end
function Simplex{V,G}(v::T,b::Basis{W,G}) where {V,W,G,T}
    order(v)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,V(b),T}(v)
end
function Simplex{V,G}(v::T,b::Basis{V,G}) where T<:TensorTerm where {V,G}
    order(v)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,b,Any}(v)
end
function Simplex{V,G,B}(b::T) where T<:TensorTerm{V} where {V,G,B}
    order(B)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,B,Any}(b)
end
function show(io::IO,m::Simplex)
    T = typeof(value(m))
    par = !(T <: TensorTerm) && |(broadcast(<:,T,parval)...)
    print(io,(par ? ['(',m.v,')'] : [m.v])...,basis(m))
end
for VG ∈ ((:V,),(:V,:G))
    @eval function Simplex{$(VG...)}(v,b::Simplex{V,G}) where {V,G}
        order(v)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,basis(b)}(DirectSum.∏(v,b.v))
    end
end

==(a::TensorTerm{V,G},b::TensorTerm{V,G}) where {V,G} = basis(a) == basis(b) ? value(a) == value(b) : 0 == value(a) == value(b)
==(a::TensorTerm,b::TensorTerm) = 0 == value(a) == value(b)

## Chain{V,G,𝕂}

@computed struct Chain{V,G,T} <: TensorGraded{V,G}
    v::SVector{binomial(ndims(V),G),T}
end

@doc """
    Chain{V,G,𝕂} <: TensorGraded{V,G} <: TensorAlgebra{V}

Chain type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, scalar field `𝕂::Type`.
""" Chain

export Chain
getindex(m::Chain,i::Int) = m.v[i]
getindex(m::Chain,i::UnitRange{Int}) = m.v[i]
setindex!(m::Chain{V,G,T} where {V,G},k::T,i::Int) where T = (m.v[i] = k)
Base.firstindex(m::Chain) = 1
@pure Base.lastindex(m::Chain{V,G}) where {V,G} = binomial(ndims(V),G)
@pure Base.length(m::Chain{V,G}) where {V,G} = binomial(ndims(V),G)

function (m::Chain{V,G,T})(i::Integer) where {V,G,T}
    Simplex{V,G,Basis{V}(indexbasis(ndims(V),G)[i]),T}(m[i])
end

function Chain{V,G,T}(val::T,v::Basis{V,G}) where {V,G,T}
    N = ndims(V)
    Chain{V,G,T}(setblade!(zeros(mvec(N,G,T)),val,bits(v),Val{N}()))
end

Chain(v::Basis{V,G}) where {V,G} = Chain{V,G,Int}(one(Int),v)

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
for var ∈ ((:V,:G,:T),(:V,:T),(:T,))
    @eval Chain{$(var...)}(v::Basis{V,G}) where {V,G,T} = Chain{V,G,T}(one(T),v)
end
for var ∈ ((:V,:G,:T),(:V,:T),(:T,),())
    @eval begin
        Chain{$(var...)}(v::Simplex{V,G,B,T}) where {V,G,B,T} = Chain{V,G,T}(v.v,basis(v))
        Chain{$(var...)}(v::Chain{V,G,T}) where {V,G,T} = Chain{V,G,T}(SVector{binomial(ndims(V),G),T}(v.v))
    end
end
==(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T,S} = prod(a.v .== b.v)
==(a::Chain{V},b::Chain{V}) where V = prod(0 .==value(a)) && prod(0 .== value(b))

## MultiVector{V,𝕂}

"""
    MultiVector{V,𝕂} <: TensorMixed{V} <: TensorAlgebra{V}

Chain type with pseudoscalar `V::Manifold` and scalar field `𝕂::Type`.
"""
struct MultiVector{V,T,E} <: TensorMixed{V}
    v::SArray{Tuple{E},T,1,E}
end
MultiVector{V,T}(v::MArray{Tuple{E},S,1,E}) where S<:T where {V,T,E} = MultiVector{V,S,E}(SVector(v))
MultiVector{V,T}(v::SArray{Tuple{E},S,1,E}) where S<:T where {V,T,E} = MultiVector{V,S,E}(v)

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

MultiVector{V}(v::StaticArray{Tuple{M},T,1}) where {V,T,M} = MultiVector{V,T}(v)
for var ∈ ((:V,:T),(:V,))
    @eval begin
        MultiVector{$(var...)}(v::SizedArray) where {V,T} = MultiVector{V,T}(SVector{1<<ndims(V),T}(v))
        MultiVector{$(var...)}(v::Vector{T}) where {T,V} = MultiVector{V,T}(SVector{1<<ndims(V),T}(v))
        MultiVector{$(var...)}(v::T...) where {T,V} = MultiVector{V,T}(SVector{1<<ndims(V),T}(v))
        function MultiVector{$(var...)}(val::T,v::Basis{V,G}) where {T,V,G}
            N = ndims(V)
            MultiVector{V,T}(setmulti!(zeros(mvec(N,T)),val,bits(v),Val{N}()))
        end
    end
end
function MultiVector(val::T,v::Basis{V,G}) where {V,T,G}
    N = ndims(V)
    MultiVector{V,T}(setmulti!(zeros(mvec(N,T)),val,bits(v),Val{N}()))
end

MultiVector(v::Basis{V,G}) where {V,G} = MultiVector{V,Int}(one(Int),v)

for var ∈ ((:V,:T),(:T,))
    @eval function MultiVector{$(var...)}(v::Basis{V,G}) where {V,T,G}
        return MultiVector{V,T}(one(T),v)
    end
end
for var ∈ ((:V,:T),(:T,),())
    @eval begin
        function MultiVector{$(var...)}(v::Simplex{V,G,B,T}) where {V,G,B,T}
            return MultiVector{V,T}(v.v,basis(v))
        end
        function MultiVector{$(var...)}(v::Chain{V,G,T}) where {V,G,T}
            N = ndims(V)
            out = zeros(mvec(N,T))
            r = binomsum(N,G)
            @inbounds out[r+1:r+binomial(N,G)] = v.v
            return MultiVector{V,T}(out)
        end
    end
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
    SparseChain{V,G} <: TensorGraded{V,G} <: TensorAlgebra{V}

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

## ParaVector{V,G}

## ComplexTensor{V,G}


#struct ComplexTensor{V,G} <:

## MultiGrade{V,G}

@computed struct MultiGrade{V,G} <: TensorMixed{V}
    v::SVector{count_ones(G),TensorGraded{V}}
end

@doc """
    MultiGrade{V,G} <: TensorMixed{V,G} <: TensorAlgebra{V}

Sparse multivector type with pseudoscalar `V::Manifold` and grade encoding `G::UInt64`.
""" MultiGrade

terms(v::MultiGrade) = v.v
value(v::MultiGrade) = reduce(vcat,value.(terms(v)))

MultiGrade{V}(v::Vector{T}) where T<:TensorGraded{V} where V = MultiGrade{V,|(UInt(1).<<grade.(v)...)}(SVector(v...))
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
    return count_ones(G)≠1 ? MultiGrade{V,G}(SVector(out[indices(G,N+1)]...)) : out[1]
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
    g = grade.(v.v)
    out = zeros(mvec(N,T))
    for k ∈ 1:length(v.v)
        @inbounds (val,b) = typeof(v.v[k]) <: Basis ? (one(T),v.v[k]) : (v.v[k].v,basis(v.v[k]))
        setmulti!(out,convert(T,val),bits(b),Val{N}())
    end
    return MultiVector{V,T}(out)
end

MultiVector{V}(v::MultiGrade{V}) where V = MultiVector{V}(v)
MultiVector(v::MultiGrade{V}) where V = MultiVector{V,promote_type(typeval.(v.v)...)}(v)=#

==(a::MultiGrade{V,G},b::MultiGrade{V,G}) where {V,G} = prod(terms(a) .== terms(b))

## Generic

import Base: isinf, isapprox
import DirectSum: grade
import AbstractTensors: scalar, involute, unit, even, odd
import LinearAlgebra: rank, norm
export basis, grade, hasinf, hasorigin, isorigin, scalar, norm, gdims, betti, χ
export valuetype, scalar, isscalar, vector, isvector, indices

const VBV = Union{Simplex,Chain,MultiVector}

valuetype(t::MultiGrade) = promote_type(valuetype.(terms(t))...)
@pure valuetype(t::SparseChain{V,G,T} where {V,G}) where T = T
@pure valuetype(::Basis) = Int
@pure valuetype(::Simplex{V,G,B,T} where {V,G,B}) where T = T
@pure valuetype(::MultiVector{V,T} where V) where T = T
@pure valuetype(::Chain{V,G,T} where {V,G}) where T = T
@inline value(::Basis,T=Int) = T==Any ? 1 : one(T)
@inline value(m::MultiGrade,T) = m
for T ∈ (:Simplex,:Chain,:MultiVector)
    @eval @inline value(m::$T,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
end
@inline value(m::SparseChain,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(SparseVector{T,Int},m.v) : m.v
@inline value_diff(m::T) where T<:TensorTerm = (v=value(m);typeof(v)<:TensorAlgebra ? v : m)
@inline value_diff(m::Chain{V,0} where V) = (v=value(m)[1];typeof(v)<:TensorAlgebra ? v : m)
@inline value_diff(m::Chain) = m
@pure basis(m::Basis) = m
@pure basis(m::Simplex{V,G,B}) where {V,G,B} = B
@pure grade(m::TensorGraded{V,G} where V) where G = G
@pure grade(m::Real) = 0
@pure order(m::Basis{V,G,B} where G) where {V,B} = count_ones(symmetricmask(V,B,B)[4])
@pure order(m::Simplex) = order(basis(m))+order(value(m))
@pure order(m) = 0
@pure UInt(m::T) where T<:TensorTerm = bits(basis(m))
@pure bits(m::T) where T<:TensorTerm = bits(basis(m))
@pure bits(::Type{Basis{V,G,B}}) where {V,G,B} = B

@pure isinf(e::Basis{V}) where V = hasinf(e) && count_ones(bits(e)) == 1
@pure hasinf(e::Basis{V}) where V = hasinf(V) && isodd(bits(e))

@pure isorigin(e::Basis{V}) where V = hasorigin(V) && count_ones(bits(e))==1 && e[hasinf(V)+1]
@pure hasorigin(e::Basis{V}) where V = hasorigin(V) && (hasinf(V) ? e[2] : isodd(bits(e)))
@pure hasorigin(t::Simplex) = hasorigin(basis(t))
@pure hasorigin(m::TensorAlgebra) = hasorigin(vectorspace(m))

"""
    χ(::TensorAlgebra)

Compute the Euler characteristic χ = ∑ₚ(-1)ᵖbₚ.
"""
χ(t::T) where T<:TensorAlgebra{V} where V = (B=gdims(t);sum([B[t]*(-1)^t for t ∈ 1:length(B)]))
χ(t::T) where T<:TensorTerm{V,G} where {V,G} = χ(V,bits(basis(t)),t)
@inline χ(V,b::UInt,t) = iszero(t) ? 0 : isodd(count_ones(symmetricmask(V,b,b)[1])) ? 1 : -1

function gdims(t::T) where T<:TensorTerm{V} where V
    B,N = bits(basis(t)),ndims(V)
    g = count_ones(symmetricmask(V,B,B)[1])
    MVector{N+1,Int}([g==G ? abs(χ(t)) : 0 for G ∈ 0:N])
end
function gdims(t::T) where T<:TensorAlgebra{V} where V
    N = ndims(V)
    out = zeros(MVector{N+1,Int})
    ib = indexbasis(N,grade(t))
    for k ∈ 1:length(ib)
        @inbounds t.v[k] ≠ 0 && (out[count_ones(symmetricmask(V,ib[k],ib[k])[1])+1] += 1)
    end
    return out
end
function gdims(t::MultiVector{V,T} where T) where V
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

function rank(t::T,d=gdims(t)) where T<:TensorAlgebra{V} where V
    out = gdims(∂(t))
    out[1] = 0
    for k ∈ 2:ndims(V)
        @inbounds out[k] = min(out[k],d[k+1])
    end
    return SVector(out)
end

function null(t::T) where T<:TensorAlgebra{V} where V
    d = gdims(t)
    r = rank(t,d)
    out = zeros(MVector{ndims(V)+1,Int})
    for k ∈ 1:ndims(V)
        @inbounds out[k] = d[k+1] - r[k]
    end
    return SVector(out)
end

"""
    betti(::TensorAlgebra)

Compute the Betti numbers.
"""
function betti(t::T) where T<:TensorAlgebra{V} where V
    d = gdims(t)
    r = rank(t,d)
    out = zeros(MVector{ndims(V),Int})
    for k ∈ 1:ndims(V)
        @inbounds out[k] = d[k+1] - r[k] - r[k+1]
    end
    return SVector(out)
end

function isapprox(a::S,b::T) where {S<:TensorGraded,T<:TensorGraded}
    vectorspace(a)==vectorspace(b) && (grade(a)==grade(b) ? DirectSum.:≈(norm(a),norm(b)) : (isnull(a) && isnull(b)))
end
function isapprox(a::S,b::T) where {S<:TensorAlgebra,T<:TensorAlgebra}
    rtol = Base.rtoldefault(valuetype(a), valuetype(b), 0)
    return norm(a-b) <= rtol * max(norm(a), norm(b))
end
isapprox(a::S,b::T) where {S<:MultiVector,T<:MultiVector} = vectorspace(a)==vectorspace(b) && DirectSum.:≈(value(a),value(b))
for T ∈ (Fields...,Symbol,Expr)
    @eval begin
        isapprox(a::S,b::T) where {S<:TensorAlgebra{V},T<:$T} where V = isapprox(a,Simplex{V}(b))
        isapprox(a::S,b::T) where {S<:$T,T<:TensorAlgebra} = isapprox(b,a)
    end
end

"""
    scalar(::TensorAlgebra)
    
Return the scalar (grade 0) part of any multivector.
"""
@pure scalar(t::Basis{V,0} where V) = t
@pure scalar(t::Basis{V}) where V = zero(V)
@inline scalar(t::T) where T<:TensorGraded{V,0} where V = t
@inline scalar(t::T) where T<:TensorGraded{V} where V = zero(V)
@inline scalar(t::Chain{V,0,T}) where {V,T} = @inbounds Simplex{V}(t.v[1])
@inline scalar(t::SparseChain{V,0}) where V = @inbounds Simplex{V}(t.v[1])
@inline scalar(t::MultiVector{V}) where V = @inbounds Simplex{V}(t.v[1])
@inline scalar(t::MultiGrade{V,G}) where {V,G} = @inbounds 1 ∈ indices(G) ? terms(t)[1] : zero(V)

"""
    vector(::TensorAlgebra)

Return the vector (grade 1) part of any multivector.
"""
@pure vector(t::Basis{V,0} where V) = t
@pure vector(t::Basis{V}) where V = zero(V)
@inline vector(t::T) where T<:TensorGraded{V,1} where V = t
@inline vector(t::T) where T<:TensorGraded{V} where V = zero(V)
@inline vector(t::MultiVector{V,T}) where {V,T} = @inbounds Chain{V,1,T}(t[1])
@inline vector(t::MultiGrade{V,G}) where {V,G} = @inbounds (i=indices(G);2∈i ? terms(t)[findfirst(x->x==2,i)] : zero(V))

@pure volume(t::T) where T<:Basis{V,G} where {V,G} = G == ndims(V) ? t : zero(V)
@inline volume(t::T) where T<:TensorGraded{V,G} where {V,G} = G == ndims(V) ? t : zero(V)
@inline volume(t::MultiVector{V}) where V = @inbounds Simplex{V}(t.v[end])
@inline volume(t::MultiGrade{V,G}) where {V,G} = @inbounds ndims(V)+1∈indices(G) ? terms(t)[end] : zero(V)

@pure isscalar(t::Basis) = grade(t) == 0
@inline isscalar(t::T) where T<:TensorGraded = grade(t) == 0 || iszero(t)
@inline isscalar(t::MultiVector) = norm(t.v[2:end]) ≈ 0
@inline isscalar(t::MultiGrade) = norm(t) ≈ scalar(t)

@pure isvector(t::Basis) = grade(t) == 1
@inline isvector(t::T) where T<:TensorGraded = grade(t) == 1 || iszero(t)
@inline isvector(t::MultiVector) = norm(t) ≈ norm(vector(t))
@inline isvector(t::MultiGrade) = norm(t) ≈ norm(vector(t))

for T ∈ (Expr,Symbol)
    @eval @inline Base.iszero(t::Simplex{V,G,B,$T} where {V,G,B}) = false
end

## Adjoint

import Base: adjoint # conj

adjoint(b::Basis{V,G,B}) where {V,G,B} = Basis{dual(V)}(mixedmode(V)<0 ? dual(V,B) : B)
adjoint(b::SparseChain{V,G}) where {V,G} = SparseChain{dual(V),G}(adjoint.(terms(b)))
adjoint(b::MultiGrade{V,G}) where {V,G} = MultiGrade{dual(V),G}(adjoint.(terms(b)))

## conversions

for M ∈ (:Signature,:DiagonalForm,:SubManifold)
    @eval begin
        @inline (V::$M)(s::UniformScaling{T}) where T = Simplex{V}(T<:Bool ? (s.λ ? one(Int) : -(one(Int))) : s.λ,getbasis(V,(one(T)<<(ndims(V)-diffvars(V)))-1))
        (W::$M)(b::Simplex) = Simplex{W}(value(b),W(basis(b)))
    end
end

#@pure supblade(N,S,B) = bladeindex(N,expandbits(N,S,B))
#@pure supmulti(N,S,B) = basisindex(N,expandbits(N,S,B))

@pure subvert(::SubManifold{M,V,S} where {M,V}) where S = S

@pure function mixed(V::M,ibk::UInt) where M<:Manifold
    N,D,VC = ndims(V),diffvars(V),mixedmode(V)
    return if D≠0
        A,B = ibk&(UInt(1)<<(N-D)-1),ibk&DirectSum.diffmask(V)
        VC>0 ? (A<<(N-D))|(B<<N) : A|(B<<(N-D))
    else
        VC>0 ? ibk<<N : ibk
    end
end

@pure function (W::Signature)(b::Basis{V}) where V
    V==W && (return b)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    B = typeof(V)<:SubManifold ? expandbits(ndims(W),subvert(V),bits(b)) : bits(b)
    if WC<0 && VC≥0
        getbasis(W,mixed(V,B))
    elseif WC≥0 && VC≥0
        getbasis(W,B)
    else
        throw(error("arbitrary Manifold intersection not yet implemented."))
    end
end
@pure function (W::SubManifold{M,V,S})(b::Basis{V,G,B}) where {M,V,S,G,B}
    count_ones(B&S)==G ? getbasis(W,lowerbits(ndims(V),S,B)) : g_zero(W)
end

@pure choicevec(M,G,T) = T ∈ (Any,BigFloat,BigInt,Complex{BigFloat},Rational{BigInt},Complex{BigInt}) ? svec(M,G,T) : mvec(M,G,T)
@pure choicevec(M,T) = T ∈ (Any,BigFloat,BigInt,Complex{BigFloat},Rational{BigInt},Complex{BigInt}) ? svec(M,T) : mvec(M,T)

@pure subindex(::SubManifold{M,V,S} where {M,V}) where S = S::UInt

function (W::Signature)(b::Chain{V,G,T}) where {V,G,T}
    V==W && (return b)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    N,M = ndims(V),ndims(W)
    out = zeros(choicevec(M,G,valuetype(b)))
    ib = indexbasis(N,G)
    for k ∈ 1:length(ib)
        @inbounds if b[k] ≠ 0
            @inbounds B = typeof(V)<:SubManifold ? expandbits(M,subvert(V),ib[k]) : ib[k]
            if WC<0 && VC≥0
                @inbounds setblade!(out,b[k],mixed(V,B),Val{M}())
            elseif WC≥0 && VC≥0
                @inbounds setblade!(out,b[k],B,Val{M}())
            else
                throw(error("arbitrary Manifold intersection not yet implemented."))
            end
        end
    end
    return Chain{W,G,T}(out)
end
function (W::SubManifold{M,V,S})(b::Chain{V,1,T}) where {M,V,S,T}
    Chain{W,1,T}(b.v[indices(subindex(W),ndims(V))])
end
function (W::SubManifold{M,V,S})(b::Chain{V,G,T}) where {M,V,S,T,G}
    out,N = zeros(choicevec(M,G,valuetype(b))),ndims(V)
    ib = indexbasis(N,G)
    for k ∈ 1:length(ib)
        @inbounds if b[k] ≠ 0
            @inbounds if count_ones(ib[k]&S) == G
                @inbounds setblade!(out,b[k],lowerbits(M,S,ib[k]),Val{M}())
            end
        end
    end
    return Chain{W,G,T}(out)
end

function (W::Signature)(m::MultiVector{V,T}) where {V,T}
    V==W && (return m)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    N,M = ndims(V),ndims(W)
    out = zeros(choicevec(M,valuetype(m)))
    bs = binomsum_set(N)
    for i ∈ 1:N+1
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            @inbounds s = k+bs[i]
            @inbounds if m.v[s] ≠ 0
                @inbounds B = typeof(V)<:SubManifold ? expandbits(M,subvert(V),ib[k]) : ib[k]
                if WC<0 && VC≥0
                    @inbounds setmulti!(out,m.v[s],mixed(V,B),Val{M}())
                elseif WC≥0 && VC≥0
                    @inbounds setmulti!(out,m.v[s],B,Val{M}())
                else
                    throw(error("arbitrary Manifold intersection not yet implemented."))
                end
            end
        end
    end
    return MultiVector{W,T}(out)
end

function (W::SubManifold{M,V,S})(m::MultiVector{V,T}) where {M,V,S,T}
    out,N = zeros(choicevec(M,valuetype(m))),ndims(V)
    bs = binomsum_set(N)
    for i ∈ 1:N+1
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            @inbounds s = k+bs[i]
            @inbounds if m.v[s] ≠ 0
                @inbounds if count_ones(ib[k]&S) == i-1
                    @inbounds setmulti!(out,m.v[s],lowerbits(N,S,ib[k]),Val{M}())
                end
            end
        end
    end
    return MultiVector{W,T}(out)
end

# QR compatibility

convert(a::Type{Simplex{V,G,B,A}},b::Simplex{V,G,B,T}) where {V,G,A,B,T} = Simplex{V,G,B,A}(convert(A,value(b)))
convert(::Type{Simplex{V,G,B,X}},t::Y) where {V,G,B,X,Y} = Simplex{V,G,B,X}(convert(X,t))

Base.copysign(x::Simplex{V,G,B,T},y::Simplex{V,G,B,T}) where {V,G,B,T} = Simplex{V,G,B,T}(copysign(value(x),value(y)))

@inline function LinearAlgebra.reflectorApply!(x::AbstractVector, τ::TensorAlgebra, A::StridedMatrix)
    @assert !LinearAlgebra.has_offset_axes(x)
    m, n = size(A)
    if length(x) != m
        throw(DimensionMismatch("reflector has length $(length(x)), which must match the dimension of matrix A, $m"))
    end
    @inbounds begin
        for j = 1:n
            #dot
            vAj = A[1,j]
            for i = 2:m
                vAj += conj(x[i])*A[i,j]
            end

            vAj = conj(τ)*vAj

            #ger
            A[1, j] -= vAj
            for i = 2:m
                A[i,j] -= x[i]*vAj
            end
        end
    end
    return A
end

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

for implex ∈ (Simplex,Basis)
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
