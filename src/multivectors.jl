
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export TensorTerm, TensorMixed, Basis, MultiVector, SparseChain, MultiGrade

abstract type GradedAlgebra{V,G} <: TensorAlgebra{V} end
abstract type TensorTerm{V,G} <: GradedAlgebra{V,G} end
abstract type TensorMixed{T,V} <: TensorAlgebra{V} end

function Base.show_unquoted(io::IO, z::T, ::Int, prec::Int) where T<:TensorAlgebra
    if T<:TensorMixed && Base.operator_precedence(:+) <= prec
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

parany = (Expr,Complex,Rational,TensorMixed)
parsym = (Expr,Complex,Rational,TensorAlgebra,Symbol)
parval = (Expr,Complex,Rational,TensorMixed)

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

## S/MSimplex{N}

const MSB = (:MSimplex,:Simplex)

for implex ∈ MSB
    eval(Expr(:struct,implex ≠ :Simplex,:($implex{V,G,B,T} <: TensorTerm{V,G}),quote
        v::T
        $implex{A,B,C,D}(t::E) where E<:D where {A,B,C,D} = new{A,B,C,D}(t)
        $implex{A,B,C,D}(t::E) where E<:TensorAlgebra{A} where {A,B,C,D} = new{A,B,C,D}(t)
    end))
end
for implex ∈ MSB
    @eval begin
        export $implex
        @pure $implex(b::Basis{V,G}) where {V,G} = $implex{V}(b)
        @pure $implex{V}(b::Basis{V,G}) where {V,G} = $implex{V,G,b,Int}(1)
        $implex{V}(v::T) where {V,T} = $implex{V,0,Basis{V}(),T}(v)
        $implex{V}(v::S) where S<:TensorTerm where V = v
        $implex{V,G,B}(v::T) where {V,G,B,T} = $implex{V,G,B,T}(v)
        $implex(v,b::S) where S<:TensorTerm{V} where V = $implex{V}(v,b)
        $implex{V}(v,b::S) where S<:TensorAlgebra where V = v*b
        $implex{V}(v,b::Basis{V,G}) where {V,G} = $implex{V,G}(v,b)
        $implex{V}(v,b::Basis{W,G}) where {V,W,G} = $implex{V,G}(v,b)
        function $implex{V,G}(v::T,b::Basis{V,G}) where {V,G,T}
            order(v)+order(b)>diffmode(V) ? zero(V) : $implex{V,G,b,T}(v)
        end
        function $implex{V,G}(v::T,b::Basis{W,G}) where {V,W,G,T}
            order(v)+order(b)>diffmode(V) ? zero(V) : $implex{V,G,V(b),T}(v)
        end
        function $implex{V,G}(v::T,b::Basis{V,G}) where T<:TensorTerm where {V,G}
            order(v)+order(b)>diffmode(V) ? zero(V) : $implex{V,G,b,Any}(v)
        end
        function $implex{V,G,B}(b::T) where T<:TensorTerm{V} where {V,G,B}
            order(B)+order(b)>diffmode(V) ? zero(V) : $implex{V,G,B,Any}(b)
        end
        function show(io::IO,m::$implex)
            T = typeof(value(m))
            par = !(T <: TensorTerm) && |(broadcast(<:,T,parany)...)
            print(io,(par ? ['(',m.v,')'] : [m.v])...,basis(m))
        end
    end
    for Other ∈ MSB, VG ∈ ((:V,),(:V,:G))
        @eval function $implex{$(VG...)}(v,b::$Other{V,G}) where {V,G}
            order(v)+order(b)>diffmode(V) ? zero(V) : $implex{V,G,basis(b)}(v*b.v)
        end
    end
end

==(a::TensorTerm{V,G},b::TensorTerm{V,G}) where {V,G} = basis(a) == basis(b) ? value(a) == value(b) : 0 == value(a) == value(b)
==(a::TensorTerm,b::TensorTerm) = 0 == value(a) == value(b)

## S/MChain{T,N}

const MSC = (:MChain,:SChain)

for (Chain,vector,implex) ∈ ((MSC[1],:MVector,MSB[1]),(MSC[2],:SVector,MSB[2]))
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
        function (m::$Chain{T,V,G})(i::Integer,B::Type=Simplex) where {T,V,G}
            if B ≠ Simplex
                MSimplex{V,G,Basis{V}(indexbasis(ndims(V),G)[i]),T}(m[i])
            else
                Simplex{V,G,Basis{V}(indexbasis(ndims(V),G)[i]),T}(m[i])
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
        function ==(a::$Chain{S,V,G} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
            i = bladeindex(ndims(V),bits(basis(b)))
            @inbounds a[i] == value(b) && prod(a[1:i-1] .== 0) && prod(a[i+1:end] .== 0)
        end
        ==(a::T,b::$Chain{S,V} where S) where T<:TensorTerm{V} where V = b==a
        ==(a::$Chain{S,V} where S,b::T) where T<:TensorTerm{V} where V = prod(0==value(b).==value(a))
    end
    for T ∈ Fields
        @eval begin
            ==(a::T,b::$Chain{S,V,G} where {S,V}) where {T<:$T,G} = G==0 ? a==value(b)[1] : prod(0==a.==value(b))
            ==(a::$Chain{S,V,G} where {S,V},b::T) where {T<:$T,G} = G==0 ? value(a)[1]==b : prod(0==b.==value(a))
        end
    end
    for var ∈ ((:T,:V,:G),(:T,:V),(:T,))
        @eval begin
            $Chain{$(var...)}(v::Basis{V,G}) where {T,V,G} = $Chain{T,V,G}(one(T),v)
        end
    end
    for var ∈ [[:T,:V,:G],[:T,:V],[:T],[]]
        @eval begin
            $Chain{$(var...)}(v::Simplex{V,G,B,T}) where {T,V,G,B} = $Chain{T,V,G}(v.v,basis(v))
            $Chain{$(var...)}(v::MSimplex{V,G,B,T}) where {T,V,G,B} = $Chain{T,V,G}(v.v,basis(v))
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
MultiVector{T,V}(v::MArray{Tuple{E},S,1,E}) where S<:T where {T,V,E} = MultiVector{S,V,E}(v)
MultiVector{T,V}(v::SArray{Tuple{E},S,1,E}) where S<:T where {T,V,E} = MultiVector{S,V,E}(v)


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
function (m::MultiVector{T,V})(g::Int,i::Int,::Type{B}=Simplex) where {T,V,B}
    if B ≠ Simplex
        MSimplex{V,g,Basis{V}(indexbasis(ndims(V),g)[i]),T}(m[g][i])
    else
        Simplex{V,g,Basis{V}(indexbasis(ndims(V),g)[i]),T}(m[g][i])
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
    for (implex,Chain) ∈ ((MSB[1],MSC[1]),(MSB[2],MSC[2]))
        @eval begin
            function MultiVector{$(var...)}(v::$implex{V,G,B,T}) where {V,G,B,T}
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

==(a::MultiVector{T,V},b::MultiVector{S,V}) where {T,V,S} = prod(a.v .== b.v)

for Chain ∈ MSC
    @eval begin
        function ==(a::MultiVector{T,V},b::$Chain{S,V,G}) where {T,V,S,G}
            N = ndims(V)
            r,R = binomsum(N,G), N≠G ? binomsum(N,G+1) : 2^N+1
            @inbounds prod(a[G] .== b.v) && prod(a.v[1:r] .== 0) && prod(a.v[R+1:end] .== 0)
        end
        ==(a::$Chain{T,V,G},b::MultiVector{S,V}) where {T,V,S,G} = b == a
    end
end

function ==(a::MultiVector{S,V} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
    i = basisindex(ndims(V),bits(basis(b)))
    @inbounds a.v[i] == value(b) && prod(a.v[1:i-1] .== 0) && prod(a.v[i+1:end] .== 0)
end
==(a::T,b::MultiVector{S,V} where S) where T<:TensorTerm{V} where V = b==a
for T ∈ Fields
    @eval begin
        ==(a::T,b::MultiVector{S,V,G} where {S,V}) where {T<:$T,G} = (v=value(b);(a==v[1])*prod(0 .== v[2:end]))
        ==(a::MultiVector{S,V,G} where {S,V},b::T) where {T<:$T,G} = b == a
    end
end

## SparseChain{V,G}

struct SparseChain{V,G} <: TensorAlgebra{V}
    v::Vector{<:TensorTerm{V,G}}
end

terms(v::SparseChain) = v.v
value(v::SparseChain) = value.(terms(v))

#SparseChain{V,G}(v::Vector{<:TensorTerm{V,G}}) where {V,G} = SparseChain{V,G}(v)
SparseChain{V}(v::Vector{<:TensorTerm{V,G}}) where {V,G} = SparseChain{V,G}(v)
SparseChain{V}(v::T) where T <: (TensorTerm{V,G}) where {V,G} = v
SparseChain(v::T) where T <: (TensorTerm{V,G}) where {V,G} = v

for (Chain,vec) ∈ ((MSC[1],:MVector),(MSC[2],:SVector))
    for Vec ∈ (:($vec{L,T}),:(SubArray{T,1,$vec{L,T}}))
        @eval function chainvalues(V::Manifold{N},m::$Vec,::Dim{G}) where {N,G,L,T}
            bng = binomial(N,G)
            G∉(0,N) && sum(m .== 0)/bng < fill_limit && (return $Chain{T,V,G}(m))
            com = indexbasis(N,G)
            out = (Simplex{V,G,B,T} where B)[]
            for i ∈ 1:bng
                @inbounds m[i] ≠ 0 && push!(out,Simplex{V,G,getbasis(V,com[i]),T}(m[i]))
            end
            length(out)≠1 ? SparseChain{V,G}(out) : out[1]::Simplex{V,G,B,T} where B
        end
    end
    @eval begin
        SparseChain{V,G}(m::$Chain{T,V,G}) where {T,V,G} = chainvalues(V,value(m),Dim{G}())
        SparseChain{V}(m::$Chain{T,V,G}) where {T,V,G} = SparseChain{V,G}(m)
        SparseChain(m::$Chain{T,V,G}) where {T,V,G} = SparseChain{V,G}(m)
    end
end

function show(io::IO, m::SparseChain{V}) where V
    t = terms(m)
    isempty(t) && print(io,zero(V))
    for k ∈ 1:length(t)
        k ≠ 1 && print(io," + ")
        print(io,t[k])
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

struct MultiGrade{V,G} <: TensorAlgebra{V}
    v::Vector{<:TensorAlgebra{V}}
end

terms(v::MultiGrade) = v.v
value(v::MultiGrade) = collect(Base.Iterators.flatten(value.(terms(v))))

MultiGrade{V}(v::Vector{<:TensorAlgebra{V}}) where V = MultiGrade{V}(v)
MultiGrade{V}(m::T) where {T<:TensorTerm{V,G}} where {V,G} = m
MultiGrade(m::T) where {T<:TensorTerm{V,G}} where {V,G} = m
MultiGrade{V}(v::SparseChain{V,G}) where {V,G} = v
MultiGrade(v::SparseChain{V,G}) where {V,G} = v

for Chain ∈ MSC
    @eval begin
        MultiGrade{V}(m::$Chain{T,V,G}) where {T,V,G} = chainvalues(V,value(m),Dim{G}())
        MultiGrade(m::$Chain{T,V,G}) where {T,V,G} = chainvalues(V,value(m),Dim{G}())
    end
end

MultiGrade(v::MultiVector{T,V}) where {T,V} = MultiGrade{V}(v)
function MultiGrade{V}(m::MultiVector{T,V}) where {T,V}
    N = ndims(V)
    sum(m.v .== 0)/(1<<N) < fill_limit && (return m)
    out = TensorAlgebra{V}[]
    G = zero(UInt)
    for i ∈ 0:N
        @inbounds !prod(m[i].==0) && (G|=UInt(1)<<i;push!(out,chainvalues(V,m[i],Dim{i}())))
    end
    return length(out)≠1 ? MultiGrade{V,G}(out) : out[1]
end

function show(io::IO, m::MultiGrade{V}) where V
    t = terms(m)
    isempty(t) && print(io,zero(V))
    for k ∈ 1:length(t)
        k ≠ 1 && print(io," + ")
        print(io,t[k])
    end
end

#=function MultiVector{T,V}(v::MultiGrade{V}) where {T,V}
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
MultiVector(v::MultiGrade{V}) where V = MultiVector{promote_type(typeval.(v.v)...),V}(v)=#

==(a::MultiGrade{V,G},b::MultiGrade{V,G}) where {V,G} = prod(terms(a) .== terms(b))

## Generic

import Base: isinf, isapprox
import DirectSum: grade
import AbstractTensors: scalar, involute, unit, even, odd
import LinearAlgebra: rank
export basis, grade, hasinf, hasorigin, isorigin, scalar, norm, gdims, betti, χ

const VBV = Union{MSimplex,Simplex,MChain,SChain,MultiVector}

valuetype(t::SparseChain) = promote_type(valuetype.(terms(t))...)
valuetype(t::MultiGrade) = promote_type(valuetype.(terms(t))...)
@pure valuetype(::Basis) = Int
@pure valuetype(::Union{MSimplex{V,G,B,T},Simplex{V,G,B,T}} where {V,G,B}) where T = T
@pure valuetype(::TensorMixed{T}) where T = T
@inline value(::Basis,T=Int) = T==Any ? 1 : one(T)
@inline value(m::VBV,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value_diff(m::T) where T<:TensorTerm = (v=value(m);typeof(v)<:TensorAlgebra ? v : m)
@pure basis(m::Basis) = m
@pure basis(m::Union{MSimplex{V,G,B},Simplex{V,G,B}}) where {V,G,B} = B
@pure grade(m::TensorTerm{V,G} where V) where G = G
@pure grade(m::Union{MChain{T,V,G},SChain{T,V,G}} where {T,V}) where G = G
@pure grade(m::SparseChain{V,G} where V) where G = G
@pure grade(m::Real) = 0
@pure order(m::Basis{V,G,B} where G) where {V,B} = count_ones(symmetricmask(V,B,B)[4])
@pure order(m::Union{MSimplex,Simplex}) = order(basis(m))+order(value(m))
@pure order(m) = 0
@pure bits(m::T) where T<:TensorTerm = bits(basis(m))

@pure isinf(e::Basis{V}) where V = hasinf(e) && count_ones(bits(e)) == 1
@pure hasinf(e::Basis{V}) where V = hasinf(V) && isodd(bits(e))

@pure isorigin(e::Basis{V}) where V = hasorigin(V) && count_ones(bits(e))==1 && e[hasinf(V)+1]
@pure hasorigin(e::Basis{V}) where V = hasorigin(V) && (hasinf(V) ? e[2] : isodd(bits(e)))
@pure hasorigin(t::Union{MSimplex,Simplex}) = hasorigin(basis(t))
@pure hasorigin(m::TensorAlgebra) = hasorigin(vectorspace(m))

@inline χ(V,b::UInt,t) = iszero(t) ? 0 : isodd(count_ones(symmetricmask(V,b,b)[1])) ? 1 : -1
χ(t::T) where T<:TensorTerm{V,G} where {V,G} = χ(V,bits(basis(t)),t)
χ(t::T) where T<:TensorAlgebra{V} where V = (B=gdims(t);sum([B[t]*(-1)^t for t ∈ 1:length(B)]))

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
function gdims(t::MultiVector{T,V} where T) where V
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
    return out
end

function null(t::T) where T<:TensorAlgebra{V} where V
    d = gdims(t)
    r = rank(t,d)
    out = zeros(MVector{ndims(V)+1,Int})
    for k ∈ 1:ndims(V)
        @inbounds out[k] = d[k+1] - r[k]
    end
    return out
end

function betti(t::T) where T<:TensorAlgebra{V} where V
    d = gdims(t)
    r = rank(t,d)
    out = zeros(MVector{ndims(V),Int})
    for k ∈ 1:ndims(V)
        @inbounds out[k] = d[k+1] - r[k] - r[k+1]
    end
    return out
end

for A ∈ (:TensorTerm,MSC...), B ∈ (:TensorTerm,MSC...)
    @eval isapprox(a::S,b::T) where {S<:$A,T<:$B} = vectorspace(a)==vectorspace(b) && (grade(a)==grade(b) ? norm(a)≈norm(b) : (iszero(a) && iszero(b)))
end
function isapprox(a::S,b::T) where {S<:TensorAlgebra,T<:TensorAlgebra}
    rtol = Base.rtoldefault(valuetype(a), valuetype(b), 0)
    return norm(a-b) <= rtol * max(norm(a), norm(b))
end
isapprox(a::S,b::T) where {S<:MultiVector,T<:MultiVector} = vectorspace(a)==vectorspace(b) && value(a) ≈ value(b)
for T ∈ Fields
    @eval begin
        isapprox(a::S,b::T) where {S<:TensorAlgebra{V},T<:$T} where V =isapprox(a,Simplex{V}(b))
        isapprox(a::S,b::T) where {S<:$T,T<:TensorAlgebra} = isapprox(b,a)
    end
end

"""
    scalar(multivector)
    
Return the scalar (grade 0) part of any multivector.
"""
@inline scalar(t::T) where T<:TensorTerm{V,0} where V = t
@inline scalar(t::T) where T<:TensorTerm{V} where V = zero(V)
@inline scalar(t::SparseChain{V,0}) where V = t
@inline scalar(t::SparseChain{V}) where V = zero(V)
@inline scalar(t::MultiVector{T,V}) where {T,V} = @inbounds Simplex{V}(t.v[1])
@inline scalar(t::MultiGrade{V,G}) where {V,G} = @inbounds 1 ∈ indices(G) ? terms(t)[1] : zero(V)
for Chain ∈ MSC
    @eval begin
        @inline scalar(t::$Chain{T,V,0}) where {T,V} = @inbounds Simplex{V}(t.v[1])
        @inline scalar(t::$Chain{T,V} where T) where V = zero(V)
    end
end

@inline vector(t::T) where T<:TensorTerm{V,1} where V = t
@inline vector(t::T) where T<:TensorTerm{V} where V = zero(V)
@inline vector(t::SparseChain{V,1}) where V = t
@inline vector(t::SparseChain{V}) where V = zero(V)
@inline vector(t::MultiVector{T,V}) where {T,V} = @inbounds SChain{T,V,1}(t[1])
@inline vector(t::MultiGrade{V,G}) where {V,G} = @inbounds (i=indices(G);2∈i ? terms(t)[findfirst(x->x==2,i)] : zero(V))
for Chain ∈ MSC
    @eval begin
        @inline vector(t::$Chain{T,V,1} where {T,V}) = t
        @inline vector(t::$Chain{T,V} where T) where V = zero(V)
    end
end

@inline volume(t::T) where T<:TensorTerm{V,G} where {V,G} = G == ndims(V) ? t : zero(V)
@inline volume(t::SparseChain{V,G}) where {V,G} = G == ndims(V) ? t : zero(V)
@inline volume(t::MultiVector{T,V}) where {T,V} = @inbounds Simplex{V}(t.v[end])
@inline volume(t::MultiGrade{V,G}) where {V,G} = @inbounds ndims(V)+1∈indices(G) ? terms(t)[end] : zero(V)
for Chain ∈ MSC
    @eval begin
        @inline volume(t::$Chain{T,V,G} where T) where {V,G} = G == ndims(V) ? t : zero(V)
    end
end

@inline isscalar(t::T) where T<:TensorTerm = grade(t) == 0 || iszero(t)
@inline isscalar(t::T) where T<:SChain = grade(t) == 0 || iszero(t)
@inline isscalar(t::T) where T<:MChain = grade(t) == 0 || iszero(t)
@inline isscalar(t::SparseChain) = grade(t) == 0 || iszero(t)
@inline isscalar(t::MultiVector) = norm(t) ≈ scalar(t)
@inline isscalar(t::MultiGrade) = norm(t) ≈ scalar(t)

for implex ∈ MSB
    for T ∈ (Expr,Symbol)
        @eval @inline Base.iszero(t::$implex{V,G,B,$T} where {V,G,B}) = false
    end
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
        (W::$M)(b::MSimplex) = MSimplex{W}(value(b),W(basis(b)))
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

for Chain ∈ MSC
    @eval begin
        function (W::Signature)(b::$Chain{T,V,G}) where {T,V,G}
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
                        @inbounds setblade!(out,b[k],mixed(V,B),Dimension{M}())
                    elseif WC≥0 && VC≥0
                        @inbounds setblade!(out,b[k],B,Dimension{M}())
                    else
                        throw(error("arbitrary Manifold intersection not yet implemented."))
                    end
                end
            end
            return $Chain{T,W,G}(out)
        end
        function (W::SubManifold{M,V,S})(b::$Chain{T,V,1}) where {M,V,S,T}
            $Chain{T,W,1}(b.v[indices(subindex(W),ndims(V))])
        end
        function (W::SubManifold{M,V,S})(b::$Chain{T,V,G}) where {M,V,S,T,G}
            out,N = zeros(choicevec(M,G,valuetype(b))),ndims(V)
            ib = indexbasis(N,G)
            for k ∈ 1:length(ib)
                @inbounds if b[k] ≠ 0
                    @inbounds if count_ones(ib[k]&S) == G
                        @inbounds setblade!(out,b[k],lowerbits(M,S,ib[k]),Dimension{M}())
                    end
                end
            end
            return $Chain{T,W,G}(out)
        end
    end
end

function (W::Signature)(m::MultiVector{T,V}) where {T,V}
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
                    @inbounds setmulti!(out,m.v[s],mixed(V,B),Dimension{M}())
                elseif WC≥0 && VC≥0
                    @inbounds setmulti!(out,m.v[s],B,Dimension{M}())
                else
                    throw(error("arbitrary Manifold intersection not yet implemented."))
                end
            end
        end
    end
    return MultiVector{T,W}(out)
end

function (W::SubManifold{M,V,S})(m::MultiVector{T,V}) where {M,V,S,T}
    out,N = zeros(choicevec(M,valuetype(m))),ndims(V)
    bs = binomsum_set(N)
    for i ∈ 1:N+1
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            @inbounds s = k+bs[i]
            @inbounds if m.v[s] ≠ 0
                @inbounds if count_ones(ib[k]&S) == i-1
                    @inbounds setmulti!(out,m.v[s],lowerbits(N,S,ib[k]),Dimension{M}())
                end
            end
        end
    end
    return MultiVector{T,W}(out)
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

for implex ∈ (MSB...,Basis)
    @eval begin
        #norm(t::$implex) = norm(unsplitval(t))
        function unsplitvalue(a::$implex{V,G}) where {V,G}
            !(hasinf(V) && hasorigin(V)) && (return value(a))
            #T = valuetype(a)
            #$(insert_expr((:N,:t,:out),:mvec,:T,:(typeof((one(T)/(2one(T))))))...)
            #out = copy(value(a,t))
            return unsplitvalue(MChain(a))
        end
    end
end

for Chain ∈ MSC
    @eval begin
        #norm(t::$Chain) = norm(unsplitval(t))
        function unsplitvalue(a::$Chain{T,V,G}) where {T,V,G}
            !(hasinf(V) && hasorigin(V)) && (return value(a))
            $(insert_expr((:N,:t,:out),:mvec,:T,:(typeof((one(T)/(2one(T))))))...)
            out = copy(value(a,mvec(N,G,t)))
            bi = bladeindex(N,unsplitstart(G)):bladeindex(N,unsplitend(G))-1
            out[bi] = unsplitter(N,G)*out[bi]
            return out
        end
    end
end

@eval begin
    #norm(t::MultiVector) = norm(unsplitval(t))
    function unsplitvalue(a::MultiVector{T,V}) where {T,V}
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
