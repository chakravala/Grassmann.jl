
#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export TensorTerm, TensorGraded, TensorMixed, SubManifold, Simplex, MultiVector, SparseChain, MultiGrade, ChainBundle

import AbstractTensors: TensorTerm, TensorGraded, TensorMixed
import Leibniz: grade

export TensorNested
abstract type TensorNested{V} <: Manifold{V} end

for op ∈ (:(Base.:+),:(Base.:-))
    @eval begin
        $op(a::A,b::B) where {A<:TensorNested,B<:TensorAlgebra} = $op(DyadicChain(a),b)
        $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorNested} = $op(a,DyadicChain(b))
    end
end

# symbolic print types

import Leibniz: Fields, parval, mixed
parsym = (Symbol,parval...)

## pseudoscalar

import LinearAlgebra
import LinearAlgebra: I, UniformScaling
export UniformScaling, I, points

## Chain{V,G,𝕂}

@computed struct Chain{V,G,𝕂} <: TensorGraded{V,G}
    v::Values{binomial(mdims(V),G),𝕂}
    Chain{V,G,𝕂}(v) where {V,G,𝕂} = new{DirectSum.submanifold(V),G,𝕂}(v)
end

"""
    Chain{V,G,𝕂} <: TensorGraded{V,G}

Chain type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, scalar field `𝕂::Type`.
"""
Chain{V,G}(val::S) where {V,G,S<:AbstractVector{𝕂}} where 𝕂 = Chain{V,G,𝕂}(val)
Chain{V}(val::S) where {V,S<:TupleVector{N,𝕂}} where {N,𝕂} = Chain{V,1,𝕂}(val)
Chain(val::S) where S<:TupleVector{N,𝕂} where {N,𝕂} = Chain{SubManifold(N),1,𝕂}(val)
#Chain{V,G}(args::𝕂...) where {V,G,𝕂} = Chain{V,G}(Values{binomial(mdims(V),G)}(args...))
@generated function Chain{V,G}(args::𝕂...) where {V,G,𝕂}
    bg = binomial(mdims(V),G)
    ref = Values{bg}([:(args[$i]) for i ∈ 1:bg])
    :(Chain{V,G}($(Expr(:call,:(Values{$bg,𝕂}),ref...))))
end

@generated function Chain{V}(args::𝕂...) where {V,𝕂}
    bg = mdims(V); ref = Values{bg}([:(args[$i]) for i ∈ 1:bg])
    :(Chain{V,1}($(Expr(:call,:(Values{$bg,𝕂}),ref...))))
end

@generated function Chain(args::𝕂...) where 𝕂
    N = length(args)
    V = SubManifold(N)
    ref = Values{N}([:(args[$i]) for i ∈ 1:N])
    :(Chain{$V,1}($(Expr(:call,:(Values{$N,𝕂}),ref...))))
end

Chain(v::Chain{V,G,𝕂}) where {V,G,𝕂} = Chain{V,G}(Values{binomial(mdims(V),G),𝕂}(v.v))
Chain{𝕂}(v::Chain{V,G}) where {V,G,𝕂} = Chain{V,G}(Values{binomial(mdims(V),G),𝕂}(v.v))

DyadicProduct{V,W,G,T,N} = Chain{V,G,Chain{W,G,T,N},N}
DyadicChain{V,G,T,N} = DyadicProduct{V,V,G,T,N}

export Chain, DyadicProduct, DyadicChain
getindex(m::Chain,i::Int) = m.v[i]
getindex(m::Chain,i::UnitRange{Int}) = m.v[i]
getindex(m::Chain,i::T) where T<:AbstractVector = m.v[i]
getindex(m::Chain{V,G,<:Chain} where {V,G},i::Int,j::Int) = m[j][i]
setindex!(m::Chain{V,G,T} where {V,G},k::T,i::Int) where T = (m.v[i] = k)
Base.firstindex(m::Chain) = 1
@pure Base.lastindex(m::Chain{V,G}) where {V,G} = binomial(mdims(V),G)
@pure Base.length(m::Chain{V,G}) where {V,G} = binomial(mdims(V),G)
Base.zero(::Type{<:Chain{V,G,T}}) where {V,G,T} = Chain{V,G}(zeros(svec(mdims(V),G,T)))
Base.zero(::Chain{V,G,T}) where {V,G,T} = Chain{V,G}(zeros(svec(mdims(V),G,T)))

transpose_row(t::Values{N,<:Chain{V}},i,W=V) where {N,V} = Chain{W,1}(getindex.(t,i))
transpose_row(t::FixedVector{N,<:Chain{V}},i,W=V) where {N,V} = Chain{W,1}(getindex.(t,i))
transpose_row(t::Chain{V,1,<:Chain},i) where V = transpose_row(value(t),i,V)
@generated _transpose(t::Values{N,<:Chain{V,1}},W=V) where {N,V} = :(Chain{V,1}(transpose_row.(Ref(t),$(list(1,mdims(V))),W)))
@generated _transpose(t::FixedVector{N,<:Chain{V,1}},W=V) where {N,V} = :(Chain{V,1}(transpose_row.(Ref(t),$(list(1,mdims(V))),W)))
Base.transpose(t::Chain{V,1,<:Chain{V,1}}) where V = _transpose(value(t))
Base.transpose(t::Chain{V,1,<:Chain{W,1}}) where {V,W} = _transpose(value(t),V)

showparens(tmv) = (!(tmv<:TensorTerm||tmv<:Projector)) && |(broadcast(<:,tmv,parval)...)

function show(io::IO, m::Chain{V,G,T}) where {V,G,T}
    ib = indexbasis(mdims(V),G)
    @inbounds tmv = typeof(m.v[1])
    if |(broadcast(<:,tmv,parsym)...)
        par = showparens(tmv)
        @inbounds par ? print(io,"(",m.v[1],")") : print(io,m.v[1])
    else
        @inbounds print(io,m.v[1])
    end
    @inbounds Leibniz.printindices(io,V,ib[1])
    for k ∈ 2:length(ib)
        @inbounds mvs = m.v[k]
        tmv = typeof(mvs)
        if |(broadcast(<:,tmv,parsym)...)
            par = showparens(tmv)
            par ? print(io," + (",mvs,")") : print(io," + ",mvs)
        else
            sbm = signbit(mvs)
            print(io,sbm ? " - " : " + ",sbm ? abs(mvs) : mvs)
        end
        @inbounds Leibniz.printindices(io,V,ib[k])
    end
end

for T ∈ Fields
    @eval begin
        ==(a::T,b::Chain{V,G} where V) where {T<:$T,G} = G==0 ? a==value(b)[1] : prod(0==a.==value(b))
        ==(a::Chain{V,G} where V,b::T) where {T<:$T,G} = G==0 ? value(a)[1]==b : prod(0==b.==value(a))
        isapprox(a::T,b::Chain{V,G} where V) where {T<:$T,G} = G==0 ? a≈value(b)[1] : prod(0≈a.≈value(b))
        isapprox(a::Chain{V,G} where V,b::T) where {T<:$T,G} = G==0 ? value(a)[1]≈b : prod(0≈b.≈value(a))
    end
end
==(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T,S} = prod(a.v .== b.v)
==(a::Chain{V},b::Chain{V}) where V = prod(0 .==value(a)) && prod(0 .== value(b))
isapprox(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T,S} = prod(a.v .≈ b.v)
isapprox(a::Chain{V},b::Chain{V}) where V = prod(0 .≈value(a)) && prod(0 .≈ value(b))

function Chain(val::𝕂,v::SubManifold{V,G}) where {V,G,𝕂}
    N = mdims(V)
    Chain{V,G}(setblade!(zeros(mvec(N,G,𝕂)),val,UInt(v),Val(N)))
end
Chain(v::SubManifold) = Chain(one(Int),v)
Chain(v::Simplex) = Chain(v.v,basis(v))
Chain{𝕂}(v::SubManifold{V,G}) where {V,G,𝕂} = Chain(one(𝕂),v)
Chain{𝕂}(v::Simplex{V,G,B}) where {V,G,B,𝕂} = Chain{𝕂}(v.v,basis(v))
Chain{V,G,T,X}(x::Simplex{V,0}) where {V,G,T,X} = Chain{V,G}(zeros(mvec(mdims(V),G,T)))
function Chain{V,0,T,X}(x::Simplex{V,0,v}) where {V,T,X,v}
    N = mdims(V)
    Chain{V,0}(setblade!(zeros(mvec(N,0,T)),value(x),UInt(v),Val(N)))
end

getindex(m::Chain,i::T) where T<:AbstractVector{<:SubManifold} = getindex.(m,i)
getindex(m::Chain{V,G},i::SubManifold{V,G}) where {V,G} = m[bladeindex(mdims(V),UInt(i))]
getindex(m::Chain{V,G,T},i::SubManifold{V}) where {V,G,T} = zero(T)

function (m::Chain{V,G,T})(i::Integer) where {V,G,T}
    Simplex{V,G,SubManifold{V}(indexbasis(mdims(V),G)[i]),T}(m[i])
end

function ==(a::Chain{V,G},b::T) where T<:TensorTerm{V,G} where {V,G}
    i = bladeindex(mdims(V),UInt(basis(b)))
    @inbounds a[i] == value(b) && (prod(a[1:i-1].==0) && prod(a[i+1:end].==0))
end
==(a::T,b::Chain{V}) where T<:TensorTerm{V} where V = b==a
==(a::Chain{V},b::T) where T<:TensorTerm{V} where V = prod(0==value(b).==value(a))

function isapprox(a::Chain{V,G},b::T) where T<:TensorTerm{V,G} where {V,G}
    i = bladeindex(mdims(V),UInt(basis(b)))
    @inbounds a[i] ≈ value(b) && (prod(a[1:i-1].==0) && prod(a[i+1:end].==0))
end
isapprox(a::T,b::Chain{V}) where T<:TensorTerm{V} where V = b==a
isapprox(a::Chain{V},b::T) where T<:TensorTerm{V} where V = prod(0==value(b).==value(a))


Base.ones(::Type{Chain{V,G,T,X}}) where {V,G,T,X} = Chain{V,G,T}(ones(Values{X,T}))
Base.ones(::Type{Chain{V,G,T,X}}) where {V,G,T<:Chain,X} = Chain{V,G,T}(ones.(ntuple(n->T,mdims(V))))
⊗(a::Type{<:Chain{V}},b::Type{<:Chain{W}}) where {V,W} = Chain{V,1,Chain{W,1,Float64,mdims(W)},mdims(V)}
⊗(a::Type{<:Chain{V,1}},b::Type{<:Chain{W,1}}) where {V,W} = Chain{V,1,Chain{W,1,Float64,mdims(W)},mdims(V)}
⊗(a::Type{<:Chain{V,1}},b::Type{<:Chain{W,1,T}}) where {V,W,T} = Chain{V,1,Chain{W,1,T,mdims(W)},mdims(V)}

"""
    ChainBundle{V,G,P} <: Manifold{V} <: TensorAlgebra{V}

Subsets of a bundle cross-section over a `Manifold` topology.
"""
struct ChainBundle{V,G,𝕂,Points} <: Manifold{V}
    @pure ChainBundle{V,G,𝕂,P}() where {V,G,𝕂,P} = new{DirectSum.submanifold(V),G,𝕂,P}()
end

const bundle_cache = (Vector{Chain{V,G,T,X}} where {V,G,T,X})[]
function ChainBundle(c::Vector{Chain{V,G,T,X}} where X) where {V,G,T}
    push!(bundle_cache,c)
    ChainBundle{V,G,T,length(bundle_cache)}()
end
function clearbundlecache!()
    for P ∈ 1:length(bundle_cache)
        deletebundle!(P)
    end
end
@pure bundle(::ChainBundle{V,G,T,P} where {V,G,T}) where P = P
@pure deletebundle!(V) = deletebundle!(bundle(V))
@pure function deletebundle!(P::Int)
    bundle_cache[P] = [Chain{ℝ^0,0,Int}(Values(0))]
end
@pure isbundle(::ChainBundle) = true
@pure isbundle(t) = false
@pure ispoints(t::SubManifold{V}) where V = isbundle(V) && rank(V) == 1 && !isbundle(Manifold(V))
@pure ispoints(t) = isbundle(t) && rank(t) == 1 && !isbundle(Manifold(t))
@pure islocal(t) = isbundle(t) && rank(t)==1 && valuetype(t)==Int && ispoints(Manifold(t))
@pure iscell(t) = isbundle(t) && islocal(Manifold(t))

@pure Manifold(::ChainBundle{V}) where V = V
@pure Manifold(::Type{<:ChainBundle{V}}) where V = V
@pure Manifold(::Vector{<:Chain{V}}) where V = V
@pure LinearAlgebra.rank(M::ChainBundle{V,G} where V) where G = G
@pure grade(::ChainBundle{V}) where V = grade(V)
@pure AbstractTensors.mdims(::ChainBundle{V}) where V = mdims(V)
@pure AbstractTensors.mdims(::Type{T}) where T<:ChainBundle{V} where V = mdims(V)
@pure AbstractTensors.mdims(::Vector{<:Chain{V}}) where V = mdims(V)
@pure Base.parent(::ChainBundle{V}) where V = isbundle(V) ? parent(V) : V
@pure Base.parent(::Vector{<:Chain{V}}) where V = isbundle(V) ? parent(V) : V
@pure DirectSum.supermanifold(m::ChainBundle{V}) where V = V
@pure DirectSum.supermanifold(m::Vector{<:Chain{V}}) where V = V
@pure DirectSum.submanifold(m::ChainBundle) = m
@pure points(t::ChainBundle{p}) where p = isbundle(p) ? p : DirectSum.supermanifold(p)
@pure points(t::Vector{<:Chain{p}}) where p = isbundle(p) ? p : DirectSum.supermanifold(p)
@pure points(t::Chain{p}) where p = isbundle(p) ? p : DirectSum.supermanifold(p)

value(c::Vector{<:Chain}) = c
value(::ChainBundle{V,G,T,P}) where {V,G,T,P} = bundle_cache[P]::(Vector{Chain{V,G,T,binomial(mdims(V),G)}})
AbstractTensors.valuetype(::ChainBundle{V,G,T} where {V,G}) where T = T

getindex(m::ChainBundle,i::I) where I<:Integer = getindex(value(m),i)
getindex(m::ChainBundle,i) = getindex(value(m),i)
getindex(m::ChainBundle,i::Chain{V,1}) where V = Chain{Manifold(V),1}(m[value(i)])
getindex(m::ChainBundle{V},i::ChainBundle) where V = m[value(i)]
getindex(m::ChainBundle{V},i::T) where {V,T<:AbstractVector{<:Chain}} = getindex.(Ref(m),i)
setindex!(m::ChainBundle,k,i) = setindex!(value(m),k,i)
Base.firstindex(m::ChainBundle) = 1
Base.lastindex(m::ChainBundle) = length(value(m))
Base.length(m::ChainBundle) = length(value(m))
Base.resize!(m::ChainBundle,n::Int) = resize!(value(m),n)

Base.display(m::ChainBundle) = (print(showbundle(m));display(value(m)))
Base.show(io::IO,m::ChainBundle) = print(io,showbundle(m),length(m))
@pure showbundle(m::ChainBundle{V,G}) where {V,G} = "$(iscell(m) ? 'C' : islocal(m) ? 'I' : 'Λ')$(DirectSum.sups[G])$V×"

## MultiVector{V,𝕂}

@computed struct MultiVector{V,𝕂} <: TensorMixed{V}
    v::Values{1<<mdims(V),𝕂}
    MultiVector{V,𝕂}(v) where {V,𝕂} = new{DirectSum.submanifold(V),𝕂}(v)
end

"""
    MultiVector{V,𝕂} <: TensorMixed{V} <: TensorAlgebra{V}

Chain type with pseudoscalar `V::Manifold` and scalar field `𝕂::Type`.
"""
MultiVector{V}(v::S) where {V,S<:AbstractVector{T}} where T = MultiVector{V,T}(v)
for var ∈ ((:V,:T),(:T,),())
    @eval function MultiVector{$(var...)}(v::Chain{V,G,T}) where {V,G,T}
        N = mdims(V)
        out = zeros(mvec(N,T))
        r = binomsum(N,G)
        @inbounds out[r+1:r+binomial(N,G)] = v.v
        return MultiVector{V}(out)
    end
end

@generated function MultiVector{V}(args::𝕂...) where {V,𝕂}
    bg = 1<<mdims(V); ref = Values{bg}([:(args[$i]) for i ∈ 1:bg])
    :(MultiVector{V}($(Expr(:call,:(Values{$bg,𝕂}),ref...))))
end

@generated function MultiVector(args::𝕂...) where 𝕂
    N = length(args)
    V = SubManifold(try
        Int(log2(N))
    catch
        throw("Constructor for MultiVector got $N inputs, which is invalid.")
    end)
    ref = Values{N}([:(args[$i]) for i ∈ 1:N])
    :(MultiVector{$V}($(Expr(:call,:(Values{$N,𝕂}),ref...))))
end

function getindex(m::MultiVector{V,T},i::Int) where {V,T}
    N = mdims(V)
    0 <= i <= N || throw(BoundsError(m, i))
    r = binomsum(N,i)
    return @view m.v[r+1:r+binomial(N,i)]
end
getindex(m::MultiVector,i::Int,j::Int) = m[i][j]
getindex(m::MultiVector,i::UnitRange{Int}) = m.v[i]
getindex(m::MultiVector,i::T) where T<:AbstractVector = m.v[i]
setindex!(m::MultiVector{V,T} where V,k::T,i::Int,j::Int) where T = (m[i][j] = k)
Base.firstindex(m::MultiVector) = 0
Base.lastindex(m::MultiVector{V,T} where T) where V = mdims(V)

(m::MultiVector{V,T})(g::Int) where {T,V,B} = m(Val(g))
function (m::MultiVector{V,T})(::Val{g}) where {V,T,g,B}
    Chain{V,g,T}(m[g])
end
function (m::MultiVector{V,T})(g::Int,i::Int) where {V,T,B}
    Simplex{V,g,Basis{V}(indexbasis(mdims(V),g)[i]),T}(m[g][i])
end

function show(io::IO, m::MultiVector{V,T}) where {V,T}
    N = mdims(V)
    basis_count = true
    print(io,m[0][1])
    bs = binomsum_set(N)
    for i ∈ 2:N+1
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            @inbounds s = k+bs[i]
            @inbounds mvs = m.v[s]
            @inbounds if !isnull(mvs)
                tmv = typeof(mvs)
                if |(broadcast(<:,tmv,parsym)...)
                    par = showparens(tmv)
                    par ? print(io," + (",mvs,")") : print(io," + ",mvs)
                else
                    sba = signbit(mvs)
                    print(io,sba ? " - " : " + ",sba ? abs(mvs) : mvs)
                end
                @inbounds Leibniz.printindices(io,V,ib[k])
                basis_count = false
            end
        end
    end
    basis_count && print(io,pre[1]*'⃖')
end

==(a::MultiVector{V,T},b::MultiVector{V,S}) where {V,T,S} = prod(a.v .== b.v)
function ==(a::MultiVector{V,T},b::Chain{V,G,S}) where {V,T,G,S}
    N = mdims(V)
    r,R = binomsum(N,G), N≠G ? binomsum(N,G+1) : 2^N+1
    @inbounds prod(a[G] .== b.v) && prod(a.v[1:r] .== 0) && prod(a.v[R+1:end] .== 0)
end
==(a::Chain{V,G,T},b::MultiVector{V,S}) where {V,S,G,T} = b == a
function ==(a::MultiVector{V,S} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
    i = basisindex(mdims(V),UInt(basis(b)))
    @inbounds a.v[i] == value(b) && prod(a.v[1:i-1] .== 0) && prod(a.v[i+1:end] .== 0)
end
==(a::T,b::MultiVector{V,S} where S) where T<:TensorTerm{V} where V = b==a
for T ∈ Fields
    @eval begin
        ==(a::T,b::MultiVector{V,S,G} where {V,S}) where {T<:$T,G} = (v=value(b);(a==v[1])*prod(0 .== v[2:end]))
        ==(a::MultiVector{V,S,G} where {V,S},b::T) where {T<:$T,G} = b == a
    end
end

function MultiVector(val::T,v::SubManifold{V,G}) where {V,T,G}
    N = mdims(V)
    MultiVector{V}(setmulti!(zeros(mvec(N,T)),val,UInt(v),Val{N}()))
end
MultiVector(v::SubManifold{V,G}) where {V,G} = MultiVector(one(Int),v)
for var ∈ ((:V,:T),(:T,))
    @eval function MultiVector{$(var...)}(v::SubManifold{V,G}) where {V,T,G}
        return MultiVector(one(T),v)
    end
end
for var ∈ ((:V,:T),(:T,),())
    @eval begin
        function MultiVector{$(var...)}(v::Simplex{V,G,B,T}) where {V,G,B,T}
            return MultiVector(v.v,basis(v))
        end
    end
end

getindex(m::MultiVector,i::T) where T<:AbstractVector{<:SubManifold} = getindex.(m,i)
getindex(m::MultiVector{V},i::SubManifold{V}) where V = m[basisindex(mdims(V),UInt(i))]

# Dyadic

export Projector, Dyadic, Proj

struct Projector{V,T,Λ} <: TensorNested{V}
    v::T
    λ::Λ
    Projector{V,T,Λ}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
    Projector{V,T}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
    Projector{V}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
end

const Proj = Projector

Proj(v::T,λ=1) where T<:TensorGraded{V} where V = Proj{V}(v/abs(v),λ)
Proj(v::Chain{W,1,<:Chain{V}},λ=1) where {V,W} = Proj{V}(Chain(value(v)./abs.(value(v))),λ)
#Proj(v::Chain{V,1,<:TensorNested},λ=1) where V = Proj{V}(v,λ)

(P::Projector)(x) = contraction(P,x)

getindex(P::Proj,i::Int,j::Int) = P.v[i]*P.v[j]
getindex(P::Proj{V,<:Chain{W,1,<:Chain}} where {V,W},i::Int,j::Int) = sum(column(P.v,i).*column(P.v,j))
#getindex(P::Proj{V,<:Chain{V,1,<:TensorNested}} where V,i::Int,j::Int) = sum(getindex.(value(P.v),i,j))

show(io::IO,P::Proj{V,T,Λ}) where {V,T,Λ<:Real} = print(io,isone(P.λ) ? "" : P.λ,"Proj(",P.v,")")
show(io::IO,P::Proj{V,T,Λ}) where {V,T,Λ} = print(io,"(",P.λ,")Proj(",P.v,")")

DyadicChain{V,1,T}(P::Proj{V,T}) where {V,T} = outer(P.v*P.λ,P.v)
DyadicChain{V,1,T}(P::Proj{V,T}) where {V,T<:Chain{V,1,<:Chain}} = sum(outer.(value(P.v).*value(P.λ),P.v))
#DyadicChain{V,T}(P::Proj{V,T}) where {V,T<:Chain{V,1,<:TensorNested}} = sum(DyadicChain.(value(P.v)))
DyadicChain{V}(P::Proj{V,T}) where {V,T} = DyadicChain{V,1,T}(P)
DyadicChain(P::Proj{V,T}) where {V,T} = DyadicChain{V,1,T}(P)

struct Dyadic{V,X,Y} <: TensorNested{V}
    x::X
    y::Y
    Dyadic{V,X,Y}(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = new{DirectSum.submanifold(V),X,Y}(x,y)
    Dyadic{V}(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = new{DirectSum.submanifold(V),X,Y}(x,y)
end

Dyadic(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = Dyadic{V}(x,y)
Dyadic(P::Projector) = Dyadic(P.v,P.v)
Dyadic(D::Dyadic) = D

(P::Dyadic)(x) = contraction(P,x)

getindex(P::Dyadic,i::Int,j::Int) = P.x[i]*P.y[j]

show(io::IO,P::Dyadic) = print(io,"(",P.x,")⊗(",P.y,")")

DyadicChain(P::Dyadic{V}) where V = DyadicProduct{V}(P)
DyadicChain{V}(P::Dyadic{V}) where V = DyadicProduct{V}(p)
DyadicProduct(P::Dyadic{V}) where V = DyadicProduct{V}(P)
DyadicProduct{V}(P::Dyadic{V}) where V = outer(P.x,P.y)

## Generic

import Base: isinf, isapprox
import Leibniz: basis, grade, order
import AbstractTensors: value, valuetype, scalar, isscalar, involute, unit, even, odd
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume, ⋆
import LinearAlgebra: rank, norm
export basis, grade, hasinf, hasorigin, scalar, norm, gdims, betti, χ
export valuetype, scalar, isscalar, vector, isvector, indices

#const VBV = Union{Simplex,Chain,MultiVector}

@pure valuetype(::Chain{V,G,T} where {V,G}) where T = T
@pure valuetype(::MultiVector{V,T} where V) where T = T
@pure valuetype(::Type{<:Chain{V,G,T} where {V,G}}) where T = T
@pure valuetype(::Type{<:MultiVector{V,T} where V}) where T = T

@inline value(m::Chain,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value(m::MultiVector,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value_diff(m::Chain{V,0} where V) = (v=value(m)[1];istensor(v) ? v : m)
@inline value_diff(m::Chain) = m

Base.isapprox(a::S,b::T) where {S<:MultiVector,T<:MultiVector} = Manifold(a)==Manifold(b) && DirectSum.:≈(value(a),value(b))

@inline scalar(t::Chain{V,0,T}) where {V,T} = @inbounds Simplex{V}(t.v[1])
@inline scalar(t::MultiVector{V}) where V = @inbounds Simplex{V}(t.v[1])
@inline vector(t::MultiVector{V,T}) where {V,T} = @inbounds Chain{V,1,T}(t[1])
@inline volume(t::MultiVector{V}) where V = @inbounds Simplex{V}(t.v[end])
@inline isscalar(t::MultiVector) = norm(t.v[2:end]) ≈ 0
@inline isvector(t::MultiVector) = norm(t) ≈ norm(vector(t))

Leibniz.gdims(t::Tuple{Vector{<:Chain},Vector{Int}}) = gdims(t[1][findall(x->!iszero(x),t[2])])
function Leibniz.gdims(t::Vector{<:Chain})
    out = zeros(Variables{mdims(Manifold(points(t)))+1,Int})
    @inbounds out[mdims(Manifold(t))+1] = length(t)
    return out
end
function Leibniz.gdims(t::Values{N,<:Vector}) where N
    out = zeros(Variables{mdims(points(t[1]))+1,Int})
    for i ∈ list(1,N)
        @inbounds out[mdims(Manifold(t[i]))+1] = length(t[i])
    end
    return out
end
function Leibniz.gdims(t::Values{N,<:Tuple}) where N
    out = zeros(Variables{mdims(points(t[1][1]))+1,Int})
    for i ∈ list(1,N)
        @inbounds out[mdims(Manifold(t[i][1]))+1] = length(t[i][1])
    end
    return out
end
function Leibniz.gdims(t::MultiVector{V}) where V
    N = mdims(V)
    out = zeros(Variables{N+1,Int})
    bs = binomsum_set(N)
    for G ∈ list(0,N)
        ib = indexbasis(N,G)
        for k ∈ 1:length(ib)
            @inbounds t.v[k+bs[G+1]] ≠ 0 && (out[count_ones(symmetricmask(V,ib[k],ib[k])[1])+1] += 1)
        end
    end
    return out
end

Leibniz.χ(t::Values{N,<:Vector}) where N = (B=gdims(t);sum([B[t]*(-1)^t for t ∈ 1:length(B)]))
Leibniz.χ(t::Values{N,<:Tuple}) where N = (B=gdims(t);sum([B[t]*(-1)^t for t ∈ 1:length(B)]))

## Adjoint

import Base: adjoint # conj

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

for implex ∈ (Simplex,SubManifold)
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
        @inbounds out[bi] = unsplitter(N,G)*out[bi]
        return out
    end
    #norm(t::MultiVector) = norm(unsplitval(t))
    function unsplitvalue(a::MultiVector{V,T}) where {V,T}
        !(hasinf(V) && hasorigin(V)) && (return value(a))
        $(insert_expr((:N,:t,:out),:mvec,:T,:(typeof((one(T)/(2one(T))))))...)
        out = copy(value(a,mvec(N,t)))
        for G ∈ 1:N-1
            bi = basisindex(N,unsplitstart(G)):basisindex(N,unsplitend(G))-1
            @inbounds out[bi] = unsplitter(N,G)*out[bi]
        end
        return out
    end
end

# genfun
