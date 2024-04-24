
#   This file is part of Grassmann.jl
#   It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

export TensorTerm, TensorGraded, TensorMixed, Scalar, GradedVector, Bivector, Trivector
export Submanifold, Single, Multivector, Spinor, SparseChain, MultiGrade, ChainBundle
export Zero, One, Quaternion, GaussianInteger, PointCloud, ElementMesh, AbstractSpinor
export AbstractReal, AbstractComplex, AbstractRational, ScalarFloat, ScalarIrrational
export AbstractInteger, AbstractBool, AbstractSigned, AbstractUnsigned, AntiSpinor

import AbstractTensors: Scalar, GradedVector, Bivector, Trivector
import AbstractTensors: TensorTerm, TensorGraded, TensorMixed, equal
import Leibniz: grade, antigrade, showvalue, basis, order

export TensorNested
abstract type TensorNested{V,T} <: Manifold{V,T} end

for op ∈ (:(Base.:+),:(Base.:-))
    @eval begin
        $op(a::A,b::B) where {A<:TensorNested,B<:TensorAlgebra} = $op(DyadicChain(a),b)
        $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorNested} = $op(a,DyadicChain(b))
    end
end

# symbolic print types

import Leibniz: Fields, parval, mixed, mvecs, svecs, spinsum, spinsum_set
parsym = (Symbol,parval...)

function showterm(io::IO,V,B::UInt,i::T,compact=get(io,:compact,false)) where T
    if !(|(broadcast(<:,T,parsym)...)) && signbit(i) && !isnan(i)
        print(io, compact ? "-" : " - ")
        if isa(i,Signed) && !isa(i,BigInt) && i == typemin(typeof(i))
            showvalue(io, V, B, -widen(i))
        else
            showvalue(io, V, B, -i)
        end
    else
        print(io, compact ? "+" : " + ")
        showvalue(io, V, B, i)
    end
end

## pseudoscalar

import LinearAlgebra
import LinearAlgebra: I, UniformScaling
export UniformScaling, I, points

## Chain{V,G,𝕂}

@computed struct Chain{V,G,T} <: TensorGraded{V,G,T}
    v::Values{binomial(mdims(V),G),T}
    Chain{V,G,T}(v) where {V,G,T} = new{DirectSum.submanifold(V),G,T}(v)
end

"""
    Chain{V,G,T} <: TensorGraded{V,G,T} <: TensorAlgebra{V,T}

Chain type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, scalar field `T::Type`.
"""
Chain{V,G}(val::S) where {V,G,S<:AbstractVector{𝕂}} where 𝕂 = Chain{V,G,𝕂}(val)
Chain{V,G}(val::NTuple{N,T}) where {V,G,N,T} = Chain{V,G}(Values{N,T}(val))
Chain{V,G}(val::NTuple{N,Any}) where {V,G,N} = Chain{V,G}(Values{N}(val))
Chain{V}(val::S) where {V,S<:TupleVector{N,𝕂}} where {N,𝕂} = Chain{V,1,𝕂}(val)
Chain{V}(val::NTuple{N,T}) where {V,N,T} = Chain{V}(Values{N,T}(val))
Chain{V}(val::NTuple{N,Any}) where {V,N} = Chain{V}(Values{N}(val))
Chain(val::S) where S<:TupleVector{N,𝕂} where {N,𝕂} = Chain{Submanifold(N),1,𝕂}(val)
Chain(val::NTuple{N,T}) where {N,T} = Chain(Values{N,T}(val))
Chain(val::NTuple{N,Any}) where N = Chain(Values{N}(val))
Chain(v::Chain{V,G,𝕂}) where {V,G,𝕂} = v
#Chain{𝕂}(v::Chain{V,G}) where {V,G,𝕂} = Chain{V,G}(Values{binomial(mdims(V),G),𝕂}(v.v))
@inline (::Type{T})(x...) where {T<:Chain} = T(x)

DyadicProduct{V,W,G,T,N} = Chain{V,G,Chain{W,G,T,N},N}
DyadicChain{V,G,T,N} = DyadicProduct{V,V,G,T,N}

Base.Matrix(m::Chain{V,G,<:TensorGraded{W,G}}) where {V,W,G} = hcat(value.(Chain.(value(m)))...)
Base.Matrix(m::Chain{V,G,<:Chain{W,G}}) where {V,W,G} = hcat(value.(value(m))...)
DyadicChain(m::Matrix) = Chain{Submanifold(size(m)[1]),1}(m)
function Chain{V,G}(m::Matrix) where {V,G}
    N = size(m)[2]
    Chain{V,G,Chain{N≠mdims(V) ? Submanifold(N) : V,G}}(m)
end
Chain{V,G,Chain{W,G}}(m::Matrix) where {V,W,G} = Chain{V,G}(Chain{W,G}.(getindex.(Ref(m),:,list(1,size(m)[2]))))

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
Base.one(::Type{<:Chain{V,G,T}} where G) where {V,T} = Chain{V,0}(ones(svec(mdims(V),0,T)))
Base.one(::Chain{V,G,T} where G) where {V,T} = Chain{V,0}(ones(svec(mdims(V),0,T)))

transpose_row(t::Values{N,<:Chain{V}},i,W=V) where {N,V} = Chain{W,1}(getindex.(t,i))
transpose_row(t::FixedVector{N,<:Chain{V}},i,W=V) where {N,V} = Chain{W,1}(getindex.(t,i))
transpose_row(t::Chain{V,1,<:Chain},i) where V = transpose_row(value(t),i,V)
@generated _transpose(t::Values{N,<:Chain{V,1}},W=V) where {N,V} = :(Chain{V,1}(transpose_row.(Ref(t),$(list(1,mdims(V))),W)))
@generated _transpose(t::FixedVector{N,<:Chain{V,1}},W=V) where {N,V} = :(Chain{V,1}(transpose_row.(Ref(t),$(list(1,mdims(V))),W)))
Base.transpose(t::Chain{V,1,<:Chain{V,1}}) where V = _transpose(value(t))
Base.transpose(t::Chain{V,1,<:Chain{W,1}}) where {V,W} = _transpose(value(t),V)

function show(io::IO, m::Chain{V,G,T}) where {V,G,T}
    ib,compact = indexbasis(mdims(V),G),get(io,:compact,false)
    @inbounds Leibniz.showvalue(io,V,ib[1],m.v[1])
    for k ∈ list(2,binomial(mdims(V),G))
        @inbounds showterm(io,V,ib[k],m.v[k],compact)
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
equal(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T,S} = prod(a.v .== b.v)
equal(a::Chain{V},b::Chain{V}) where V = prod(0 .==value(a)) && prod(0 .== value(b))
isapprox(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T,S} = prod(a.v .≈ b.v)
isapprox(a::Chain{V},b::Chain{V}) where V = prod(0 .≈value(a)) && prod(0 .≈ value(b))

@generated function Chain{V,G,𝕂}(val,v::Submanifold{V,G}) where {V,G,𝕂}
    N = mdims(V)
    b = bladeindex(N,UInt(basis(v)))
    if N<cache_limit
        :(Chain{V,G,𝕂}(Values($([i==b ? :val : zero(𝕂) for i ∈ 1:binomial(N,G)]...))))
    else
        :(Chain{V,G}(setblade!(zeros(mvec($N,G,𝕂)),val,UInt(v),$(Val(N)))))
    end
end
Chain(val::𝕂,v::Submanifold{V,G}) where {V,G,𝕂} = Chain{V,G,𝕂}(val,v)
Chain(v::Submanifold) = Chain(one(Int),v)
Chain(v::Single) = Chain(v.v,basis(v))
Chain{V,G,𝕂}(v::Submanifold{V,G}) where {V,G,𝕂} = Chain(one(𝕂),v)
Chain{V,G,𝕂}(v::Single{V}) where {V,G,𝕂} = Chain{V,G,𝕂}(value(v),basis(v))
Chain{V,G,T,X}(x::Single{V,0}) where {V,G,T,X} = Chain{V,G}(zeros(mvec(mdims(V),G,T)))
Chain{V,0,T,X}(x::Single{V,0,v}) where {V,T,X,v} = Chain{V,0,T}(value(x),basis(x))

Single(m::Chain{V,0,T,1} where {V,T}) = scalar(m)
Single(m::Chain{V,G,T,1} where {V,G,T}) = volume(m)
Single{V}(m::Chain{V,0,T,1} where T) where V = scalar(m)
Single{V}(m::Chain{V,G,T,1} where {G,T}) where V = volume(m)
(::Type{T})(m::Chain{V,G,<:Real,1} where {V,G}) where T<:Real = T(value(m)[1])
(::Type{Complex})(m::Chain{V,0,T,1} where V) where T<:Real = Complex(value(m)[1],zero(T))
(::Type{Complex{T}})(m::Chain{V,0,<:Real,1} where V) where T<:Real = Complex{T}(value(m)[1],zero(T))
(::Type{Complex})(m::Chain{V,G,T,1} where {V,G}) where T<:Real = Complex(zero(T),value(m)[1])
(::Type{Complex{T}})(m::Chain{V,G,<:Real,1} where {V,G}) where T<:Real = Complex{T}(zero(T),value(m)[1])
(::Type{Complex})(m::Chain{V,G,<:Complex,1} where {V,G}) = value(m)[1]
(::Type{Complex{T}})(m::Chain{V,G,<:Complex,1} where {V,G}) where T<:Real = Complex{T}(value(m)[1])

getindex(m::Chain,i::T) where T<:AbstractVector{<:Submanifold} = getindex.(m,i)
getindex(m::Chain{V,G},i::Submanifold{V,G}) where {V,G} = m[bladeindex(mdims(V),UInt(i))]
getindex(m::Chain{V,G,T},i::Submanifold{V}) where {V,G,T} = zero(T)

function (m::Chain{V,G,T})(i::Integer) where {V,G,T}
    Single{V,G,DirectSum.getbasis(V,indexbasis(mdims(V),G)[i]),T}(m[i])
end

@pure function Base.getproperty(a::Chain{V,G,T},v::Symbol) where {V,G,T}
    return if v == :v
        getfield(a,:v)
    else
        B = getproperty(Λ(V),v)
        G == grade(B) ? a.v[bladeindex(mdims(V),UInt(B))]*B : zero(T)*B
    end
end

function equal(a::Chain{V,G},b::T) where T<:TensorTerm{V,G} where {V,G}
    i = bladeindex(mdims(V),UInt(basis(b)))
    @inbounds a[i] == value(b) && (prod(a[1:i-1].==0) && prod(a[i+1:end].==0))
end
equal(a::T,b::Chain{V}) where T<:TensorTerm{V} where V = b==a
equal(a::Chain{V},b::T) where T<:TensorTerm{V} where V = prod(0==value(b).==value(a))

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
    ChainBundle{V,G,T,P} <: Manifold{V,T} <: TensorAlgebra{V,T}

Subsets of a bundle cross-section over a `Manifold` topology.
"""
struct ChainBundle{V,G,T,Points} <: Manifold{V,T}
    @pure ChainBundle{V,G,T,P}() where {V,G,T,P} = new{DirectSum.submanifold(V),G,T,P}()
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
@pure ispoints(t::Submanifold{V}) where V = isbundle(V) && rank(V) == 1 && !isbundle(Manifold(V))
@pure ispoints(t) = isbundle(t) && rank(t) == 1 && !isbundle(Manifold(t))
@pure islocal(t) = isbundle(t) && rank(t)==1 && valuetype(t)==Int && ispoints(Manifold(t))
@pure iscell(t) = isbundle(t) && islocal(Manifold(t))

@pure Manifold(::ChainBundle{V}) where V = V
@pure Manifold(::Type{<:ChainBundle{V}}) where V = V
@pure Manifold(::Vector{<:Chain{V}}) where V = V
@pure LinearAlgebra.rank(M::ChainBundle{V,G} where V) where G = G
@pure grade(::ChainBundle{V}) where V = grade(V)
@pure antigrade(::ChainBundle{V}) where V = antigrade(V)
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

getindex(m::ChainBundle,i::I) where I<:Integer = getindex(value(m),i)
getindex(m::ChainBundle,i) = getindex(value(m),i)
getindex(m::ChainBundle,i::Chain{V,1}) where V = Chain{Manifold(V),1}(m[value(i)])
getindex(m::AbstractVector,i::Chain{V,1}) where V = Chain{Manifold(V),1}(m[value(i)])
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

## Multivector{V,T}

@computed struct Multivector{V,T} <: TensorMixed{V,T}
    v::Values{1<<mdims(V),T}
    Multivector{V,T}(v) where {V,T} = new{DirectSum.submanifold(V),T}(v)
end

"""
    Multivector{V,T} <: TensorMixed{V,T} <: TensorAlgebra{V,T}

Chain type with pseudoscalar `V::Manifold` and scalar field `T::Type`.
"""
Multivector{V}(v::S) where {V,S<:AbstractVector{T}} where T = Multivector{V,T}(v)
for var ∈ ((:V,:T),(:T,),())
    @eval @generated function Multivector{$(var...)}(t::Chain{V,G,T}) where {V,G,T}
        N = mdims(V)
        chain_src(N,G,T,list(0,N))
    end
end

function chain_src(N,G,T,grades,type=:Multivector)
    if N<cache_limit
        :($type{V,$T}($(Expr(:call,:Values,vcat([G==g ? [:(@inbounds t.v[$i]) for i ∈ list(1,binomial(N,g))] : zeros(T,binomial(N,g)) for g ∈ grades]...)...))))
    else
        quote
            out = zeros(mvec($N,T))
            @inbounds out[list($(r+1),$(r+b))] = v.v
            return Multivector{V,T}(out)
        end
    end
end

@pure function log2sub(N)
    Submanifold(try
        Int(log2(N))
    catch
        throw("Constructor for Multivector got $N inputs, which is invalid.")
    end)
end

Multivector(val::TensorAlgebra{V}) where V = Multivector{V}(val)
Multivector{V}(val::NTuple{N,T}) where {V,N,T} = Multivector{V}(Values{N,T}(val))
Multivector{V}(val::NTuple{N,Any}) where {V,N} = Multivector{V}(Values{N}(val))
Multivector(val::NTuple{N,T}) where {N,T} = Multivector{log2sub(N)}(Values{N,T}(val))
Multivector(val::NTuple{N,Any}) where N = Multivector{log2sub(N)}(Values{N}(val))
@inline (::Type{T})(x...) where {T<:Multivector} = T(x)

DyadicMultivector{V,T,N} = Multivector{V,Multivector{V,T,N},N}

Base.Matrix(m::Multivector{V,<:TensorAlgebra{W}}) where {V,W} = hcat(value.(Multivector.(value(m)))...)
Base.Matrix(m::Multivector{V,<:Multivector{W}}) where {V,W} = hcat(value.(value(m))...)
DyadicMultivector(m::Matrix) = Multivector{log2sub(size(m)[1]),1}(m)
function Multivector{V}(m::Matrix) where V
    N = size(m)[2]
    Multivector{V,Multivector{Int(log2(N))≠mdims(V) ? log2sub(N) : V}}(m)
end
Multivector{V,Chain{W}}(m::Matrix) where {V,W} = Multivector{V}(Multivector{W}.(getindex.(Ref(m),:,list(1,size(m)[2]))))

function grade_src_chain(N,G,r=binomsum(N,G),is=isempty,T=Int)
    :(Chain{V,$G,T}($(grade_src(N,G,r,is,T))))
end
function grade_src(N,G,r=binomsum(N,G),is=isempty,T=Int)
    b = binomial(N,G)
    return if is(G)
        zeros(Values{b,T})
    elseif N<cache_limit
        :(Values($([:(@inbounds t.v[$(i+r)]) for i ∈ 1:b]...)))
    else
        :(@view t.v[list($(r+1),$(r+b))])
    end
end
for fun ∈ (:grade_src,:grade_src_chain)
    nex = Symbol(fun,:_next)
    @eval function $nex(N,G,r=binomsum,is=isempty,T=Int)
        Expr(:elseif,:(G==$(N-G)),($fun(N,N-G,r(N,N-G),is,T),G-1≥0 ? $nex(N,G-1,r,is,T) : nothing)...)
    end
end

@generated function (t::Multivector{V,T})(G::Int) where {V,T}
    N = mdims(V)
    Expr(:block,:(0 <= G <= $N || throw(BoundsError(t, G))),
        Expr(:if,:(G==0),grade_src_chain(N,0),grade_src_chain_next(N,N-1)))
end
@generated function getindex(t::Multivector{V,T},G::Int) where {V,T}
    N = mdims(V)
    Expr(:block,:(0 <= G <= $N || throw(BoundsError(t, G))),
        Expr(:if,:(G==0),grade_src(N,0),grade_src_next(N,N-1)))
end
@generated function getindex(t::Multivector{V},::Val{G}) where {V,G}
    N = mdims(V)
    0 <= G <= N || throw(BoundsError(t, G))
    return grade_src(N,G)
end
getindex(m::Multivector,i::Int,j::Int) = m[i][j]
getindex(m::Multivector,i::UnitRange{Int}) = m.v[i]
getindex(m::Multivector,i::T) where T<:AbstractVector = m.v[i]
setindex!(m::Multivector{V,T} where V,k::T,i::Int,j::Int) where T = (m[i][j] = k)
Base.firstindex(m::Multivector) = 0
Base.lastindex(m::Multivector{V,T} where T) where V = mdims(V)

(m::Multivector{V,T})(g::Val{G}) where {V,T,G} = Chain{V,G,T}(m[g])
(m::Multivector{V,T})(g::Int,i) where {V,T} = m(Val(g),i)
function (m::Multivector{V,T})(g::Val{G},i::Int) where {V,T,G}
    Single{V,G,DirectSum.getbasis(V,indexbasis(mdims(V),G)[i]),T}(m[g][i])
end

@pure function Base.getproperty(a::Multivector{V},v::Symbol) where V
    return if v == :v
        getfield(a,:v)
    else
        B = getproperty(Λ(V),v)
        a.v[bladeindex(mdims(V),UInt(B))]*B
    end
end

function show(io::IO, m::Multivector{V,T}) where {V,T}
    N,compact,bases = mdims(V),get(io,:compact,false),true
    bs = binomsum_set(N)
    @inbounds print(io,m.v[1])
    for i ∈ list(2,N+1)
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            @inbounds mvs = m.v[k+bs[i]]
            if !isnull(mvs)
                @inbounds showterm(io,V,ib[k],mvs,compact)
                bases = false
            end
        end
    end
    bases && (@inbounds Leibniz.showstar(io,m.v[1]); @inbounds print(io,pre[1]*'⃖'))
end

equal(a::Multivector{V,T},b::Multivector{V,S}) where {V,T,S} = prod(a.v .== b.v)
function equal(a::Multivector{V,T},b::Chain{V,G,S}) where {V,T,G,S}
    N = mdims(V)
    r,R = binomsum(N,G), N≠G ? binomsum(N,G+1) : 1<<N+1
    @inbounds prod(a[Val(G)] .== b.v) && prod(a.v[list(1,r)] .== 0) && prod(a.v[list(R+1,1<<N)] .== 0)
end
equal(a::Chain{V,G,T},b::Multivector{V,S}) where {V,S,G,T} = b == a
function equal(a::Multivector{V,S} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
    i = basisindex(mdims(V),UInt(basis(b)))
    @inbounds a.v[i] == value(b) && prod(a.v[list(1,i-1)] .== 0) && prod(a.v[list(i+1,2<<mdims(V))] .== 0)
end
equal(a::T,b::Multivector{V,S} where S) where T<:TensorTerm{V} where V = b==a
for T ∈ Fields
    @eval begin
        ==(a::T,b::Multivector{V,S,G} where {V,S}) where {T<:$T,G} = (v=value(b);@inbounds (a==v[1])*prod(0 .== v[list(2,1<<mdims(V))]))
        ==(a::Multivector{V,S,G} where {V,S},b::T) where {T<:$T,G} = b == a
    end
end

Base.zero(::Multivector{V,T,X}) where {V,T,X} = Multivector{V,T}(zeros(Values{X,T}))
Base.one(t::Multivector{V}) where V = zero(t)+one(V)
Base.zero(::Type{Multivector{V,T,X}}) where {V,T,X} = Multivector{V,T}(zeros(Values{X,T}))
Base.one(t::Type{Multivector{V,T,X}}) where {V,T,X} = zero(t)+one(V)

Single(v,b::Multivector) = v*b

@generated function Multivector(val::T,v::Submanifold{V,G}) where {V,T,G}
    N = mdims(V)
    b = basisindex(N,UInt(basis(v)))
    if N<cache_limit
        :(Multivector{V}(Values($([i==b ? :val : zero(T) for i ∈ 1:1<<N]...))))
    else
        :(Multivector{V}(setmulti!(zeros(mvec(N,T)),val,UInt(v),Val{N}())))
    end
end
Multivector(v::Submanifold{V,G}) where {V,G} = Multivector(one(Int),v)
for var ∈ ((:V,:T),(:T,))
    @eval function Multivector{$(var...)}(v::Submanifold{V,G}) where {V,T,G}
        return Multivector(one(T),v)
    end
end
for var ∈ ((:V,:T),(:T,),())
    @eval begin
        function Multivector{$(var...)}(v::Single{V,G,B,T}) where {V,G,B,T}
            return Multivector(v.v,basis(v))
        end
    end
end

getindex(m::Multivector,i::T) where T<:AbstractVector{<:Submanifold} = getindex.(m,i)
getindex(m::Multivector{V},i::Submanifold{V}) where V = m[basisindex(mdims(V),UInt(i))]

## AbstractSpinor{V}

"""
    AbstractSpinor{V,T} <: TensorMixed{V,T} <: TensorAlgebra{V,T}

Elements of `TensorAlgebra` having non-homogenous grade being a spinor in the abstract.
"""
abstract type AbstractSpinor{V,T} <: TensorMixed{V,T} end

## Spinor{V}, PsuedoSpinor

@pure log2sub2(N) = log2sub(2N)

for pinor ∈ (:Spinor,:AntiSpinor)
    dpinor = Symbol(:Dyadic,pinor)
    @eval begin
        @computed struct $pinor{V,T} <: AbstractSpinor{V,T}
            v::Values{1<<(mdims(V)-1),T}
            $pinor{V,T}(v::Values{N,T}) where {N,V,T} = new{DirectSum.submanifold(V),T}(v)
        end
        $pinor{V,T}(v::AbstractVector{T}) where {V,T} = $pinor{V,T}(Values{1<<(mdims(V)-1),T}(v))
        $pinor{V}(v::AbstractVector{T}) where {V,T} = $pinor{V,T}(v)
        $pinor{V,𝕂}(val::Single{V,G,B,𝕂}) where {V,G,B,𝕂} = $pinor{V,𝕂}(val.v,B)
        $pinor{V}(val::Single{V,G,B,𝕂}) where {V,G,B,𝕂} = $pinor{V,𝕂}(val)
        $pinor(val::TensorAlgebra{V}) where V = $pinor{V}(val)
        $pinor(t::Chain{V}) where V = $pinor{V}(t)
        $pinor{V}(val::NTuple{N,T}) where {V,N,T} = $pinor{V}(Values{N,T}(val))
        $pinor{V}(val::NTuple{N,Any}) where {V,N} = $pinor{V}(Values{N}(val))
        $pinor(val::NTuple{N,T}) where {N,T} = $pinor{log2sub2(N)}(Values{N,T}(val))
        $pinor(val::NTuple{N,Any}) where N = $pinor{log2sub2(N)}(Values{N}(val))
        @inline (::Type{T})(x...) where {T<:$pinor} = T(x)
        getindex(m::$pinor,i::Int,j::Int) = m[i][j]
        #getindex(m::$pinor,i::UnitRange{Int}) = m.v[i]
        #getindex(m::$pinor,i::T) where T<:AbstractVector = m.v[i]
        #setindex!(m::$pinor{V,T} where V,k::T,i::Int,j::Int) where T = (m[i][j] = k)
        Base.firstindex(m::$pinor) = 0
        Base.lastindex(m::$pinor{V,T} where T) where V = mdims(V)
        grade(m::$pinor,g::Val) = m(g)
        (m::$pinor{V,T})(g::Int,i::Int) where {T,V} = m(Val(g),i)
        Multivector(t::$pinor{V}) where V = Multivector{V}(t)
        Base.zero(::$pinor{V,T}) where {V,T} = $pinor{V,T}(zeros(Values{1<<(mdims(V)-1)}))
        Base.zero(::Type{$pinor{V,T}}) where {V,T} = $pinor{V,T}(zeros(Values{1<<(mdims(V)-1)}))
        equal(a::$pinor{V,T},b::$pinor{V,S}) where {V,T,S} = prod(a.v .== b.v)
        equal(a::$pinor{V,T},b::Multivector{V,S}) where {V,T,S} = equal(Multivector(a),b)
        equal(a::Multivector{V,T},b::$pinor{V,S}) where {V,T,S} = equal(a,Multivector(b))
        equal(a::Chain{V,G,T},b::$pinor{V,S}) where {V,S,G,T} = b == a
        equal(a::T,b::$pinor{V,S} where S) where T<:TensorTerm{V} where V = b==a
        $dpinor{V,T,N} = $pinor{V,$pinor{V,T,N},N}
        Base.Matrix(m::$pinor{V,<:TensorAlgebra{W}}) where {V,W} = hcat(value.($pinor.(value(m)))...)
        Base.Matrix(m::$pinor{V,<:$pinor{W}}) where {V,W} = hcat(value.(value(m))...)
        $pinor(m::Matrix) = $pinor{log2sub(size(m)[1]),1}(m)
        function $pinor{V}(m::Matrix) where V
            N = size(m)[2]
            $pinor{V,$pinor{Int(log2(N))≠mdims(V) ? log2sub(N) : V}}(m)
        end
        $pinor{V,$pinor{W}}(m::Matrix) where {V,W} = $pinor{V}($pinor{W}.(getindex.(Ref(m),:,list(1,size(m)[2]))))
    end
end

"""
    Spinor{V,T} <: AbstractSpinor{V,T} <: TensorAlgebra{V,T}

Spinor (`even` grade) type with pseudoscalar `V::Manifold` and scalar field `T::Type`.
"""
Spinor{V}(val::Submanifold{V}) where V = Spinor{V,Int}(1,val)
Spinor{V,𝕂}(v::Submanifold{V,G}) where {V,G,𝕂} = Spinor{V,𝕜}(1,v)
@generated function Spinor{V,𝕂}(val,v::Submanifold{V,G}) where {V,G,𝕂}
    isodd(G) && error("$v is not expressible as a Spinor")
    N = mdims(V)
    b = spinindex(N,UInt(basis(v)))
    if N<cache_limit
        :(Spinor{V}(Values($([i==b ? :val : zero(𝕂) for i ∈ 1:1<<(N-1)]...))))
    else
        :(Spinor{V,𝕂}(setspin!(zeros(mvecs($N,𝕂)),val,UInt(v),$(Val(N)))))
    end
end

"""
    AntiSpinor{V,T} <: AbstractSpinor{V,T} <: TensorAlgebra{V,T}

PsuedoSpinor (`odd` grade) type with pseudoscalar `V::Manifold` and scalar `T::Type`.
"""
AntiSpinor{V}(val::Submanifold{V}) where V = AntiSpinor{V,Int}(1,val)
AntiSpinor{V,𝕂}(v::Submanifold{V,G}) where {V,G,𝕂} = AntiSpinor{V,𝕜}(1,v)
@generated function AntiSpinor{V,𝕂}(val,v::Submanifold{V,G}) where {V,G,𝕂}
    iseven(G) && error("$v is not expressible as an AntiSpinor")
    N = mdims(V)
    b = antiindex(N,UInt(basis(v)))
    if N<cache_limit
        :(AntiSpinor{V}(Values($([i==b ? :val : zero(𝕂) for i ∈ 1:1<<(N-1)]...))))
    else
        :(AntiSpinor{V,𝕂}(setanti!(zeros(mvecs($N,𝕂)),val,UInt(v),$(Val(N)))))
    end
end

@generated function Spinor{V}(t::Chain{V,G,T}) where {V,G,T}
    isodd(G) && error("$t is not expressible as a Spinor")
    N = mdims(V)
    chain_src(N,G,T,evens(0,N),:Spinor)
end
@generated function AntiSpinor{V}(t::Chain{V,G,T}) where {V,G,T}
    iseven(G) && error("$t is not expressible as a AntiSpinor")
    N = mdims(V)
    chain_src(N,G,T,evens(1,N),:AntiSpinor)
end

@generated function (t::Spinor{V,T})(G::Int) where {V,T}
    N = mdims(V)
    Expr(:block,:(0 <= G <= $N || throw(BoundsError(t, G))),
        #:(isodd(G) && return Zero(V)),
        Expr(:if,:(G==0),grade_src_chain(N,0,0),grade_src_chain_next(N,N-1,spinsum,isodd,T)))
end
@generated function (t::AntiSpinor{V,T})(G::Int) where {V,T}
    N = mdims(V)
    Expr(:block,:(0 <= G <= $N || throw(BoundsError(t, G))),
        #:(iseven(G) && return Zero(V)),
        Expr(:if,:(G==0),grade_src_chain(N,0,0,iseven),grade_src_chain_next(N,N-1,antisum,iseven,T)))
end
@generated function getindex(t::Spinor{V,T},G::Int) where {V,T}
    N = mdims(V)
    Expr(:block,:(0 <= G <= $N || throw(BoundsError(t, G))),
        Expr(:if,:(G==0),grade_src(N,0,0),grade_src_next(N,N-1,spinsum,isodd,T)))
end
@generated function getindex(t::AntiSpinor{V,T},G::Int) where {V,T}
    N = mdims(V)
    Expr(:block,:(0 <= G <= $N || throw(BoundsError(t, G))),
        Expr(:if,:(G==0),grade_src(N,0,0,iseven),grade_src_next(N,N-1,antisum,iseven,T)))
end
@generated function getindex(t::Spinor{V,T},::Val{G}) where {V,T,G}
    N = mdims(V)
    0 <= G <= N || throw(BoundsError(t, G))
    isodd(G) && return zeros(svec(N,G,T))
    return grade_src(N,G,spinsum(N,G))
end
@generated function getindex(t::AntiSpinor{V,T},::Val{G}) where {V,T,G}
    N = mdims(V)
    0 <= G <= N || throw(BoundsError(t, G))
    iseven(G) && return zeros(svec(N,G,T))
    return grade_src(N,G,antisum(N,G))
end

(m::Spinor{V,T})(g::Val{G}) where {V,T,G} = isodd(G) ? Zero(V) : Chain{V,G,T}(m[g])
(m::AntiSpinor{V,T})(g::Val{G}) where {V,T,G} = iseven(G) ? Zero(V) : Chain{V,G,T}(m[g])
function (m::Spinor{V,T})(g::Val{G},i::Int) where {V,T,G}
    if isodd(G)
        return Zero(V)
    else
        Single{V,G,DirectSum.getbasis(V,indexbasis(mdims(V),G)[i]),T}(m[g][i])
    end
end
function (m::AntiSpinor{V,T})(g::Val{G},i::Int) where {V,T,G}
    if iseven(G)
        return Zero(V)
    else
        Single{V,G,DirectSum.getbasis(V,indexbasis(mdims(V),G)[i]),T}(m[g][i])
    end
end

@generated function Multivector{V}(t::Spinor{V,T}) where {V,T}
    N = mdims(V)
    bs = spinsum_set(N)
    :(Multivector{V,T}($(Expr(:call,:Values,vcat([iseven(G) ? [:(@inbounds t.v[$(i+bs[G+1])]) for i ∈ list(1,binomial(N,G))] : zeros(T,binomial(N,G)) for G ∈ list(0,N)]...)...))))
end
@generated function Multivector{V}(t::AntiSpinor{V,T}) where {V,T}
    N = mdims(V)
    bs = antisum_set(N)
    :(Multivector{V,T}($(Expr(:call,:Values,vcat([isodd(G) ? [:(@inbounds t.v[$(i+bs[G+1])]) for i ∈ list(1,binomial(N,G))] : zeros(T,binomial(N,G)) for G ∈ list(0,N)]...)...))))
end

@pure function Base.getproperty(a::Spinor{V,T},v::Symbol) where {V,T}
    return if v == :v
        getfield(a,:v)
    else
        B = getproperty(Λ(V),v)
        iseven(grade(B)) ? a.v[bladeindex(mdims(V),UInt(B))]*B : zero(T)*B
    end
end
@pure function Base.getproperty(a::AntiSpinor{V,T},v::Symbol) where {V,T}
    return if v == :v
        getfield(a,:v)
    else
        B = getproperty(Λ(V),v)
        isodd(grade(B)) ? a.v[bladeindex(mdims(V),UInt(B))]*B : zero(T)*B
    end
end

function Base.show(io::IO, m::Spinor{V,T}) where {V,T}
    N,compact = mdims(V),get(io,:compact,false)
    bs = spinsum_set(N)
    @inbounds print(io,m.v[1])
    for i ∈ evens(2,N)
        ib = indexbasis(N,i)
        for k ∈ 1:length(ib)
            @inbounds showterm(io,V,ib[k],m.v[k+bs[i+1]],compact)
        end
    end
end
function Base.show(io::IO, m::AntiSpinor{V,T}) where {V,T}
    N,compact = mdims(V),get(io,:compact,false)
    bs = antisum_set(N)
    for i ∈ evens(1,N)
        ib = indexbasis(N,i)
        for k ∈ 1:length(ib)
            if (i==1) & (k==1)
                @inbounds showvalue(io,V,ib[k],m.v[k+bs[i+1]])
            else
                @inbounds showterm(io,V,ib[k],m.v[k+bs[i+1]],compact)
            end
        end
    end
end

Base.one(::Spinor{V,T}) where {V,T} = Spinor{V,T}(one(T),Submanifold{V}())
Base.one(::Type{Spinor{V,T}}) where {V,T} = Spinor{V,T}(one(T),Submanifold{V}())

function equal(a::Spinor{V,T},b::Chain{V,G,S}) where {V,T,G,S}
    isodd(G) && (return iszero(b) && iszero(a))
    N = mdims(V)
    r,R = spinsum(N,G), N≠G ? spinsum(N,G+1) : 1<<(N-1)+1
    @inbounds prod(a[Val(G)] .== b.v) && prod(a.v[list(1,r)] .== 0) && prod(a.v[list(R+1,1<<(N-1))] .== 0)
end
function equal(a::Spinor{V,S} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
    i = spinindex(mdims(V),UInt(basis(b)))
    @inbounds a.v[i] == value(b) && prod(a.v[list(1,i-1)] .== 0) && prod(a.v[list(i+1,1<<(mdims(V)-1))] .== 0)
end
function equal(a::AntiSpinor{V,T},b::Chain{V,G,S}) where {V,T,G,S}
    iseven(G) && (return iszero(b) && iszero(a))
    N = mdims(V)
    r,R = antisum(N,G), N≠G ? antisum(N,G+1) : 1<<(N-1)+1
    @inbounds prod(a[Val(G)] .== b.v) && prod(a.v[list(1,r)] .== 0) && prod(a.v[list(R+1,1<<(N-1))] .== 0)
end
function equal(a::AntiSpinor{V,S} where S,b::T) where T<:TensorTerm{V,G} where {V,G}
    i = antiindex(mdims(V),UInt(basis(b)))
    @inbounds a.v[i] == value(b) && prod(a.v[list(1,i-1)] .== 0) && prod(a.v[list(i+1,1<<(mdims(V)-1))] .== 0)
end
for T ∈ Fields
    @eval begin
        ==(a::T,b::Spinor{V,S,G} where {V,S}) where {T<:$T,G} = (v=value(b);@inbounds (a==v[1])*prod(0 .== v[list(2,1<<(mdims(V)-1))]))
        ==(a::Spinor{V,S,G} where {V,S},b::T) where {T<:$T,G} = b == a
        ==(a::T,b::AntiSpinor{V,S,G} where {V,S}) where {T<:$T,G} = iszero(b)
        ==(a::AntiSpinor{V,S,G} where {V,S},b::T) where {T<:$T,G} = iszero(a)
    end
end

## Couple{V,B,T}, PseudoCouple{V,B,T}

"""
    Couple{V,B,T} <: AbstractSpinor{V,T} <: TensorAlgebra{V,T}

`Complex{T}` wrapper with `V::Manifold`, basis `B::Submanifold`, scalar field of `T::Type`.
"""
struct Couple{V,B,T} <: AbstractSpinor{V,T}
    v::Complex{T}
    Couple{V,B}(a::T,b::T) where {V,B,T} = new{DirectSum.submanifold(V),B,T}(Complex{T}(a,b))
    Couple{V,B}(v::Complex{T}) where {V,B,T} = new{DirectSum.submanifold(V),B,T}(v)
end

Couple(a,b) = (V=Submanifold(2); Couple{V,Submanifold(V)}(a,b))
Base.abs2(z::Couple{V,B}) where {V,B} = abs2(z.v.re) + abs2(z.v.im)*abs2_inv(B)
grade(z::Couple{V,B},::Val{G}) where {V,G,B} = grade(B)==G ? z.v.im : G==0 ? z.v.re : Zero(V)

"""
    PseudoCouple{V,B,T} <: AbstractSpinor{V,T} <: TensorAlgebra{V,T}

`Complex{T}` wrapper with `V::Manifold`, basis `B::Submanifold`, pseudoscalar of `T::Type`.
"""
struct PseudoCouple{V,B,T} <: AbstractSpinor{V,T}
    v::Complex{T}
    PseudoCouple{V,B}(a::T,b::T) where {V,B,T} = new{DirectSum.submanifold(V),B,T}(Complex{T}(a,b))
    PseudoCouple{V,B}(v::Complex{T}) where {V,B,T} = new{DirectSum.submanifold(V),B,T}(v)
end

function Base.abs2(z::PseudoCouple{V,B}) where {V,B}
    out = abs2(z.v.re)*abs2_inv(B) + abs2(z.v.im)*abs2_inv(V)
    if (~B)*basis(V) ≠ (~basis(V))*B
        return out
    else
        out + (2complementrighthodge(B))*(z.v.re*z.v.im)
    end
end
grade(z::PseudoCouple{V,B},::Val{G}) where {V,G,B} = grade(B)==G ? z.v.re : G==mdims(V) ? z.v.im : Zero(V)

Couple(m::TensorTerm{V}) where V = Couple{V,basis(m)}(Complex(m))
Couple(m::TensorTerm{V,0}) where V = Couple{V,Submanifold(V)}(Complex(m))
PseudoCouple(m::TensorTerm{V,G}) where {V,G} = G==grade(V) ? PseudoCouple{V,One(V)}(value(m)*im) : PseudoCouple{V,basis(m)}(Complex(value(m)))

@generated Multivector{V}(a::Single{V,L},b::Single{V,G}) where {V,L,G} = addermulti(a,b,:+)
Multivector{V,T}(z::Couple{V,B,T}) where {V,B,T} = Multivector{V}(scalar(z), imaginary(z))
Multivector{V,T}(z::PseudoCouple{V,B,T}) where {V,B,T} = Multivector{V}(imaginary(z), volume(z))

@generated Spinor{V}(a::Single{V,L},b::Single{V,G}) where {V,L,G} = adderspin(a,b,:+)
Spinor{V,𝕂}(z::Couple{V,B,𝕂}) where {V,B,𝕂} = Spinor{V}(scalar(z), imaginary(z))
Spinor{V,𝕂}(z::PseudoCouple{V,B,𝕂}) where {V,B,𝕂} = Spinor{V}(imaginary(z), volume(z))

@generated AntiSpinor{V}(a::Single{V,L},b::Single{V,G}) where {V,L,G} = adderanti(a,b,:+)
AntiSpinor{V}(val::PseudoCouple{V,B,𝕂}) where {V,B,𝕂} = AntiSpinor{V,𝕂}(val)
AntiSpinor{V,𝕂}(z::PseudoCouple{V,B,𝕂}) where {V,B,𝕂} = AntiSpinor{V}(imaginary(z),volume(z))

(t::Couple{V,B})(G::Int) where {V,B} = grade(B) == G ? imaginary(t) : iszero(G) ? scalar(t) : Zero(V)
(t::PseudoCouple{V,B})(G::Int) where {V,B} = grade(B) == G ? imaginary(t) : G==mdims(V) ? volume(t) : Zero(V)
(t::Couple{V,B})(::Val{G}) where {V,B,G} = grade(B) == G ? imaginary(t) : iszero(G) ? scalar(t) : Zero(V)
(t::PseudoCouple{V,B})(::Val{G}) where {V,B,G} = grade(B) == G ? imaginary(t) : G==mdims(V) ? volume(t) : Zero(V)

@pure function Base.getproperty(a::Couple{V,B,T},v::Symbol) where {V,B,T}
    return if v == :v
        getfield(a,:v)
    else
        b = getproperty(Λ(V),v)
        if basis(b) == B
            a.v.im*b
        elseif grade(b) == 0
            a.v.re*b
        else
            zero(T)*b
        end
    end
end
@pure function Base.getproperty(a::PseudoCouple{V,B,T},v::Symbol) where {V,B,T}
    return if v == :v
        getfield(a,:v)
    else
        b = getproperty(Λ(V),v)
        if basis(b) == B
            a.v.re*b
        elseif grade(b) == mdims(V)
            a.v.im*b
        else
            zero(T)*b
        end
    end
end

function Base.show(io::IO,z::Couple{V,B}) where {V,B}
    r, i = reim(z)
    show(io, r)
    showterm(io, V, UInt(B), i)
end
function Base.show(io::IO,z::PseudoCouple{V,B}) where {V,B}
    r, i = reim(z)
    showvalue(io, V, UInt(B), r)
    showterm(io, V, UInt(V), i)
end

Base.one(t::Couple{V,B,T}) where {V,B,T} = Couple{V,B}(one(Complex{T}))
Base.one(t::Type{Couple{V,B,T}}) where {V,B,T} = Couple{V,B}(one(Complex{T}))

equal(a::Couple{V},b::Couple{V}) where V = a.v.re==b.v.re && a.v.im==b.v.im==0
isapprox(a::Couple{V},b::Couple{V}) where V = a.v.re≈b.v.re && a.v.im≈b.v.im≈0
equal(a::PseudoCouple{V},b::PseudoCouple{V}) where V = a.v.im==b.v.im && a.v.re==b.v.re==0
isapprox(a::PseudoCouple{V},b::PseudoCouple{V}) where V = a.v.im≈b.v.im && a.v.re≈b.v.re≈0

for T ∈ Fields
    @eval begin
        ==(a::T,b::Couple) where T<:$T = isscalar(b) && a == b.v.re
        ==(a::Couple,b::T) where T<:$T = b == a
        ==(a::T,b::PseudoCouple) where T<:$T = isscalar(b) && a == b.v.re
        ==(a::PseudoCouple,b::T) where T<:$T = b == a
    end
end

for (eq,qe) ∈ ((:(Base.:(==)),:equal), (:(Base.isapprox),:(Base.isapprox)))
    @eval begin
        $qe(a::Couple{V,B},b::Couple{V,B}) where {V,B} = $eq(a.v,b.v)
        $qe(a::Couple{V},b::TensorTerm{V,0}) where V = isscalar(a) && $eq(a.v.re, value(b))
        $qe(a::TensorTerm{V,0},b::Couple{V}) where V = isscalar(b) && $eq(b.v.re,value(a))
        $qe(a::Couple{V,B},b::TensorTerm{V}) where {V,B} = B == basis(b) && iszero(a.v.re) && $eq(a.v.im,value(b))
        $qe(a::TensorTerm{V},b::Couple{V,B}) where {V,B} = B == basis(a) && iszero(b.v.re) && $eq(b.v.im,value(a))
        $qe(a::PseudoCouple{V,B},b::PseudoCouple{V,B}) where {V,B} = $eq(a.v,b.v)
        function $qe(a::PseudoCouple{V,B},b::TensorTerm{V}) where {V,B}
            if Submanifold(V) == basis(b)
                iszero(a.v.re) && $eq(b.v.im,value(b))
            else
                B == basis(b) && iszero(a.v.im) && $eq(a.v.re,value(b))
            end
        end
        function $qe(a::TensorTerm{V},b::PseudoCouple{V,B}) where {V,B}
            if Submanifold(V) == basis(b)
                iszero(b.v.re) && $eq(b.v.im,value(a))
            else
                B == basis(a) && iszero(b.v.im) && $eq(b.v.re,value(a))
            end
        end
    end
    for couple ∈ (:Couple,:PseudoCouple)
        @eval begin
            $qe(a::$couple{V},b::Chain{V}) where V = $eq(multispin(a),b)
            $qe(a::Chain{V},b::$couple{V}) where V = $eq(a,multispin(b))
            $qe(a::$couple{V},b::Multivector{V}) where V = $eq(multispin(a),b)
            $qe(a::Multivector{V},b::$couple{V}) where V = $eq(a,multispin(b))
            $qe(a::$couple{V,B},b::Spinor{V}) where {V,B} = $eq(multispin(a),b)
            $qe(a::Spinor{V},b::$couple{V,B}) where {V,B} = $eq(a,multispin(b))
            $qe(a::$couple{V,B},b::AntiSpinor{V}) where {V,B} = $eq(multispin(a),b)
            $qe(a::AntiSpinor{V},b::$couple{V,B}) where {V,B} = $eq(a,multispin(b))
        end
    end
end

for couple ∈ (:Couple,:PseudoCouple)
    @eval begin
        export $couple
        $couple{V,B}(a,b) where {V,B} = $couple{V,B}(Complex(a,b))
        DirectSum.basis(::$couple{V,B}) where {V,B} = B
        Base.reim(z::$couple) = reim(z.v)
        Base.widen(z::$couple{V,B}) where {V,B} = $couple{V,B}(widen(z.v))
        (::Type{Complex})(m::$couple) = value(m)
        (::Type{Complex{T}})(m::$couple) where T<:Real = Complex{T}(value(m))
        Multivector{V}(z::$couple{V,B,T}) where {V,B,T} = Multivector{V,T}(z)
        Multivector(z::$couple{V,B,T}) where {V,B,T} = Multivector{V,T}(z)
        Spinor{V}(val::$couple{V,B,𝕂}) where {V,B,𝕂} = Spinor{V,𝕂}(val)
        Base.zero(::$couple{V,B,T}) where {V,B,T} = $couple{V,B}(zero(Complex{T}))
        Base.zero(::Type{$couple{V,B,T}}) where {V,B,T} = $couple{V,B}(zero(Complex{T}))
    end
end

## Phasor{V,B}

export Phasor, ∠, radius

"""
    Phasor{V,B,T} <: AbstractSpinor{V,T} <: TensorAlgebra{V,T}

`Complex{T}` wrapper with `V::Manifold`, basis `B::Submanifold`, scalar field `T::Type`.
"""
struct Phasor{V,B,T} <: AbstractSpinor{V,T}
    v::Complex{T}
    Phasor{V,B}(r::T,iθ::T) where {V,B,T} = new{DirectSum.submanifold(V),B,T}(Complex{T}(r,iθ))
    Phasor{V,B}(v::Complex{T}) where {V,B,T} = new{DirectSum.submanifold(V),B,T}(v)
end

Phasor{V,B}(a,b) where {V,B} = Couple{V,B}(Complex(a,b))
Phasor(a,b) = (V=Submanifold(2); Couple{V,Submanifold(V)}(a,b))

const ∠ = Phasor
@pure DirectSum.basis(::Phasor{V,B}) where {V,B} = B
Base.reim(z::Phasor) = reim(z.v)
Base.widen(z::Phasor{V,B}) where {V,B} = Phasor{V,B}(widen(z.v))
function Base.abs2(z::Phasor{V,B}) where {V,B}
    if B*B == -1
        Single{V}(radius(z)*radius(z))
    else
        abs2(Couple(z))
    end
end

Base.angle(z::Phasor) = z.v.im*basis(z)
radius(z::Phasor) = z.v.re
radius(z::Couple{V,B}) where {V,B} = sqrt(z.v.re^2 - z.v.im^2*value(B*B))

Base.promote_rule(a::Type{<:Couple},b::Type{<:Phasor}) = a

grade(z::Phasor{V,B},::Val{G}) where {V,G,B} = grade(B)==G ? angle(z) : G==0 ? radius(z) : Zero(V)

(::Type{Complex})(m::Phasor) = Complex(Couple(m))
(::Type{Complex{T}})(m::Phasor) where T<:Real = Complex{T}(Couple(m))
Phasor(m::TensorTerm{V}) where V = Phasor{V,basis(m)}(Complex(1,value(m)))
Phasor(m::TensorTerm{V,0}) where V = Phasor{V,Submanifold(V)}(Complex(m))
Phasor(m::Couple) = Phasor(radius(m),angle(m))
Phasor(r::Real,iθ::Single{V,G,B}) where {V,G,B} = Phasor{V,B}(Complex(r,value(iθ)))
Phasor(r::Real,iθ::Submanifold{V}) where V = Phasor{V,iθ}(Complex(r,value(iθ)))
Phasor(r::TensorTerm{V,0} where V,iθ) = Phasor(value(r),iθ)
Couple(z::Phasor{V,B}) where {V,B} = radius(z)*exp(angle(z))
Couple{V,B,T}(z::Phasor{V,B}) where {V,B,T} = Couple(z)

(z::Phasor{V,B})(ωt) where {V,B} = Phasor{V,B}(radius(z),ωt+z.v.im)

Multivector{V,T}(z::Phasor{V,B,T}) where {V,B,T} = Multivector{V,T}(Couple(z))
Multivector{V}(z::Phasor{V,B,T}) where {V,B,T} = Multivector{V}(Couple(z))
Multivector(z::Phasor) = Multivector(Couple(z))

Spinor{V}(val::Phasor{V}) where V = Couple(val)

function Base.show(io::IO,z::Phasor)
    show(io, radius(z))
    print(io, " ∠ ", angle(z))
end

Base.zero(::Phasor{V,B,T}) where {V,B,T} = Phasor{V,B}(zero(Complex{T}))
Base.one(t::Phasor{V,B,T}) where {V,B,T} = Phasor{V,B}(one(Complex{T}))
Base.zero(::Type{Phasor{V,B,T}}) where {V,B,T} = Phasor{V,B}(zero(Complex{T}))
Base.one(t::Type{Phasor{V,B,T}}) where {V,B,T} = Phasor{V,B}(one(Complex{T}))

equal(a::Phasor{V},b::Phasor{V}) where V = a.v.re==b.v.re && a.v.im==b.v.im==0
isapprox(a::Phasor{V},b::Phasor{V}) where V = a.v.re≈b.v.re && a.v.im≈b.v.im≈0

for T ∈ Fields
    @eval begin
        ==(a::T,b::Phasor) where T<:$T = isscalar(b) && a == b.v.re
        ==(a::Phasor,b::T) where T<:$T = b == a
    end
end

for (eq,qe) ∈ ((:(Base.:(==)),:equal), (:(Base.isapprox),:(Base.isapprox)))
    @eval begin
        $qe(a::Phasor{V,B},b::Phasor{V,B}) where {V,B} = $eq(a.v,b.v)
        $qe(a::Phasor{V,B},b::Couple{V,B}) where {V,B} = $eq(Couple(a),b)
        $qe(a::Couple{V,B},b::Phasor{V,B}) where {V,B} = $eq(a,Couple(b))
    end
end

# Dyadic

export Projector, Dyadic, Proj

struct Projector{V,T,Λ} <: TensorNested{V,T}
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

Leibniz.extend_parnot(Projector)

show(io::IO,P::Proj{V,T,Λ}) where {V,T,Λ<:Real} = print(io,isone(P.λ) ? "" : P.λ,"Proj(",P.v,")")
show(io::IO,P::Proj{V,T,Λ}) where {V,T,Λ} = print(io,"(",P.λ,")Proj(",P.v,")")

DyadicChain{V,1,T}(P::Proj{V,T}) where {V,T} = outer(P.v*P.λ,P.v)
DyadicChain{V,1,T}(P::Proj{V,T}) where {V,T<:Chain{V,1,<:Chain}} = sum(outer.(value(P.v).*value(P.λ),P.v))
#DyadicChain{V,T}(P::Proj{V,T}) where {V,T<:Chain{V,1,<:TensorNested}} = sum(DyadicChain.(value(P.v)))
DyadicChain{V}(P::Proj{V,T}) where {V,T} = DyadicChain{V,1,T}(P)
DyadicChain(P::Proj{V,T}) where {V,T} = DyadicChain{V,1,T}(P)

struct Dyadic{V,X,Y} <: TensorNested{V,X}
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
import AbstractTensors: antiabs, antiabs2, geomabs, unit, unitize, unitnorm
import AbstractTensors: value, valuetype, scalar, isscalar, involute, even, odd
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume, ⋆
import LinearAlgebra: rank, norm
export gdims, betti, χ, unit
export basis, grade, pseudograde, antigrade, hasinf, hasorigin, scalar, norm, unitnorm
export valuetype, scalar, isscalar, vector, isvector, indices, imaginary, unitize, geomabs
export bivector, isbivector, trivector, istrivector, volume, isvolume, antiabs, antiabs2

const Imaginary{V,T} = Spinor{V,T,2}
const Quaternion{V,T} = Spinor{V,T,4}
const AntiQuaternion{V,T} = AntiSpinor{V,T,4}
const LipschitzInteger{V,T<:Integer} = Quaternion{V,T}
const GaussianInteger{V,B,T<:Integer} = Couple{V,B,T}
const PointCloud{T<:Chain{V,1} where V} = AbstractVector{T}
const ElementMesh{T<:Chain{V,1,<:Integer} where V} = AbstractVector{T}

const AbstractReal = Union{Real,Single{V,G,B,<:Real} where {V,G,B},Chain{V,G,<:Real,1} where {V,G}}
const AbstractComplex{T<:Real} = Union{Complex{T},Phasor{V,B,T} where {V,B},Couple{V,B,T} where {V,B},Single{V,G,B,Complex{T}} where {V,G,B},Chain{V,G,Complex{T},1} where {V,G}}
const AbstractBool = Union{Bool,Single{V,G,B,Bool} where {V,G,B},Chain{V,G,Bool,1} where {V,G}}
const AbstractInteger = Union{Integer,Single{V,G,B,<:Integer} where {V,G,B},Chain{V,G,<:Integer,1} where {V,G}}
const AbstractSigned = Union{Signed,Single{V,G,B,<:Signed} where {V,G,B},Chain{V,G,<:Signed,1} where {V,G}}
const AbstractUnsigned = Union{Unsigned,Single{V,G,B,<:Unsigned} where {V,G,B},Chain{V,G,<:Unsigned,1} where {V,G}}
const AbstractRational{T<:Integer} = Union{Rational{T},Single{V,G,B,Rational{T}} where {V,G,B},Chain{V,G,Rational{T},1} where {V,G}}
const ScalarFloat = Union{AbstractFloat,Single{V,G,B,<:AbstractFloat} where {V,G,B},Chain{V,G,<:AbstractFloat,1} where {V,G}}
const ScalarIrrational = Union{AbstractIrrational,Single{V,G,B,<:AbstractIrrational} where {V,G,B},Chain{V,G,<:AbstractIrrational,1} where {V,G}}

#const VBV = Union{Single,Chain,Multivector}

(::Type{Complex})(m::Imaginary) = Complex(value(m)...)
(::Type{Complex{T}})(m::Imaginary) where T<:Real = Complex{T}(value(m)...)
Couple(m::Imaginary{V}) where V = Couple{V,Submanifold(V)}(Complex(m))

Spinor{V}(t::Spinor{V}) where V = t
AntiSpinor{V}(t::AntiSpinor{V}) where V = t
Multivector{V}(t::Multivector{V}) where V = t

multispin(t::Multivector) = t
multispin(t::Spinor) = t
multispin(t::AntiSpinor) = t
multispin(t::TensorGraded) = iseven(grade(t)) ? Spinor(t) : AntiSpinor(t)
function multispin(t::Couple{V,B}) where {V,B}
    iseven(grade(B)) ? Spinor(t) : Multivector(t)
end
function multispin(t::PseudoCouple{V,B}) where {V,B}
    if iseven(grade(V)) && iseven(grade(B))
        Spinor(t)
    elseif isodd(grade(V)) && isodd(grade(B))
        AntiSpinor(t)
    else
        Multivector(t)
    end
end

export quaternion, quatvalue, quatvalues
quaternion(sijk::NTuple{4}) = quaternion(Submanifold(3),sijk...)
quaternion(s,ijk::NTuple{3}) = quaternion(Submanifold(3),s,ijk...)
quaternion(sijk::Values{4}) = quaternion(Submanifold(3),sijk...)
quaternion(s,ijk::Values{3}) = quaternion(Submanifold(3),s,ijk...)
quaternion(s::T=0,i=zero(T),j=zero(T),k=zero(T)) where T = quaternion(Submanifold(3),s,i,j,k)
quaternion(V::Submanifold,s::T,i=zero(T),j=zero(T),k=zero(T)) where T = Spinor{V}(Values(s,i,-j,k))
quaternion(V::Submanifold,sijk::Values{4}) = quaternion(V,sijk...)
quatvalue(q::TensorAlgebra) = quatvalues(Spinor(even(q)))
quatvalue(q::Quaternion{V,T}) where {V,T} = Values{4,T}(q.v[1],q.v[2],-q.v[3],q.v[4])
quatvalue(q::AntiQuaternion{V,T}) where {V,T} = Values{4,T}(q.v[4],q.v[3],q.v[2],q.v[1])
const quatvalues = quatvalue

@inline value(m::Chain,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value(m::Multivector,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value(m::Spinor,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value(m::AntiSpinor,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value(m::Couple,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(Complex{T},m.v) : m.v
@inline value(m::PseudoCouple,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(Complex{T},m.v) : m.v
#@inline value(m::Phasor,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(Complex{T},m.v) : m.v
@inline value_diff(m::Chain{V,0} where V) = (v=value(m)[1];istensor(v) ? v : m)
@inline value_diff(m::Chain) = m

Base.isapprox(a::S,b::T) where {S<:Multivector,T<:Multivector} = Manifold(a)==Manifold(b) && DirectSum.:≈(value(a),value(b))
Base.isapprox(a::S,b::T) where {S<:Spinor,T<:Spinor} = Manifold(a)==Manifold(b) && DirectSum.:≈(value(a),value(b))
Base.isapprox(a::S,b::T) where {S<:AntiSpinor,T<:AntiSpinor} = Manifold(a)==Manifold(b) && DirectSum.:≈(value(a),value(b))

@inline scalar(z::Couple{V}) where V = Single{V}(z.v.re)
@inline scalar(z::PseudoCouple{V,B}) where {V,B} = grade(B)==0 ? Single{V}(z.v.re) : Zero(V)
@inline scalar(z::Phasor) = scalar(Couple(z))
@inline scalar(t::Chain{V,0,T}) where {V,T} = @inbounds Single{V}(t.v[1])
@inline scalar(t::Multivector{V}) where V = @inbounds Single{V}(t.v[1])
@inline scalar(t::Spinor{V}) where V = @inbounds Single{V}(t.v[1])
@inline scalar(t::AntiSpinor{V}) where V = Zero(V)
@inline vector(z::Couple{V,B}) where {V,B} = grade(B)==1 ? Single{V}(z.v.im) : Zero(V)
@inline vector(t::Multivector) = t(Val(1))
@inline vector(t::Spinor{V}) where V = Zero(V)
@inline vector(t::AntiSpinor) = t(Val(1))
@inline bivector(z::Couple{V,B}) where {V,B} = grade(B)==2 ? Single{V,2,B}(z.v.im) : Zero(V)
@inline bivector(z::PseudoCouple{V,B}) where {V,B} = grade(B)==2 ? Single{V,2,B}(z.v.re) : Zero(V)
@inline bivector(t::Multivector) = t(Val(2))
@inline bivector(t::Spinor) = t(Val(2))
@inline bivector(t::AntiSpinor{V}) where V = Zero(V)
@inline trivector(z::Couple{V,B}) where {V,B} = grade(B)==3 ? Single{V,3,B}(z.v.im) : Zero(V)
@inline trivector(z::PseudoCouple{V,B}) where {V,B} = grade(B)==3 ? Single{V,3,B}(z.v.re) : Zero(V)
@inline trivector(t::Multivector) = t(Val(3))
@inline trivector(t::Spinor{V}) where V = Zero(V)
@inline trivector(t::AntiSpinor) = t(Val(3))
#@inline bivector(t::Quaternion{V}) where V = @inbounds Chain{V,2}(t.v[2],t.v[3],t.v[4])
@inline volume(t::Chain{V,G,T,1}) where {V,G,T} = @inbounds Single{V,G,basis(V)}(t.v[1])
@inline volume(z::Couple{V,B}) where {V,B} = grade(B)==grade(V) ? Single{V,grade(B),B}(z.v.im) : Zero(V)
@inline volume(z::PseudoCouple{V}) where V = Single{V,mdims(V),Submanifold(V)}(z.v.im)
@inline volume(t::Multivector{V}) where V = @inbounds Single{V,mdims(V),Submanifold(V)}(t.v[end])
@inline volume(t::Spinor{V}) where V = iseven(grade(V)) ? (@inbounds Single{V,mdims(V),Submanifold(V)}(t.v[end])) : Zero(V)
@inline volume(t::AntiSpinor{V}) where V = isodd(grade(V)) ? (@inbounds Single{V,mdims(V),Submanifold(V)}(t.v[end])) : Zero(V)
@inline imaginary(z::Couple{V,B}) where {V,B} = Single{V,grade(B),B}(z.v.im)
@inline imaginary(z::PseudoCouple{V,B}) where {V,B} = Single{V,grade(B),B}(z.v.re)
@inline imaginary(z::Quaternion) = bivector(z)
@inline imaginary(z::AntiQuaternion) = vector(z)
@inline isscalar(t) = norm(t) ≈ norm(scalar(t))
@inline isvector(t) = norm(t) ≈ norm(vector(t))
@inline isbivector(t) = norm(t) ≈ norm(bivector(t))
@inline istrivector(t) = norm(t) ≈ norm(trivector(t))
@inline isvolume(t) = norm(t) ≈ norm(volume(t))

function isscalar(z::Phasor)
    if basis(z)*basis(z) == -1
        (z.v.im%π) ≈ 0
    else
        isscalar(Couple(z))
    end
end

@pure maxgrade(t::TensorGraded) = grade(t)
@pure maxgrade(t::Couple{V,B}) where {V,B} = grade(B)
@pure maxgrade(t::PseudoCouple{V}) where V = grade(V)
@pure maxgrade(t::Spinor{V}) where V = isodd(mdims(V)) ? mdims(V)-1 : mdims(V)
@pure maxgrade(t::AntiSpinor{V}) where V = isodd(mdims(V)) ? mdims(V) : mdims(V)-1
@pure maxgrade(t::Multivector{V}) where V = mdims(V)
@pure mingrade(t::TensorGraded) = grade(t)
@pure mingrade(t::Couple) = 0
@pure mingrade(t::PseudoCouple{V,B}) where {V,B} = grade(B)
@pure mingrade(t::Spinor) = 0
@pure mingrade(t::AntiSpinor) = 1
@pure mingrade(t::Multivector) = 0

@pure maxgrade(t::Type{<:TensorGraded}) = grade(t)
@pure maxgrade(t::Type{<:Couple{V,B}}) where {V,B} = grade(B)
@pure maxgrade(t::Type{<:PseudoCouple{V}}) where V = grade(V)
@pure maxgrade(t::Type{<:Spinor{V}}) where V = isodd(mdims(V)) ? mdims(V)-1 : mdims(V)
@pure maxgrade(t::Type{<:AntiSpinor{V}}) where V = isodd(mdims(V)) ? mdims(V) : mdims(V)-1
@pure maxgrade(t::Type{<:Multivector{V}}) where V = mdims(V)
@pure mingrade(t::Type{<:TensorGraded}) = grade(t)
@pure mingrade(t::Type{<:Couple}) = 0
@pure mingrade(t::Type{<:PseudoCouple{V,B}}) where {V,B} = grade(B)
@pure mingrade(t::Type{<:Spinor}) = 0
@pure mingrade(t::Type{<:AntiSpinor}) = 1
@pure mingrade(t::Type{<:Multivector}) = 0

@pure nextgrade(t::Spinor) = 2
@pure nextgrade(t::AntiSpinor) = 2
@pure nextgrade(t::Multivector) = 1
@pure nextgrade(t::Type{<:Spinor}) = 2
@pure nextgrade(t::Type{<:AntiSpinor}) = 2
@pure nextgrade(t::Type{<:Multivector}) = 1
@pure nextmingrade(t) = mingrade(t)+nextgrade(t)
@pure nextmaxgrade(t) = maxgrade(t)-nextgrade(t)

@pure maxpseudograde(t::Multivector) = maxgrade(t)
@pure maxpseudograde(t::Spinor{V}) where V = mdims(V)
@pure maxpseudograde(t::AntiSpinor{V}) where V = mdims(V)-1
@pure maxpseudograde(t::Type{<:Multivector}) = maxgrade(t)
@pure maxpseudograde(t::Type{<:Spinor{V}}) where V = mdims(V)
@pure maxpseudograde(t::Type{<:AntiSpinor{V}}) where V = mdims(V)-1
@pure nextmaxpseudograde(t) = maxpseudograde(t)-nextgrade(t)

Leibniz.gdims(t::Tuple{Vector{<:Chain},Vector{Int}}) = gdims(t[1][findall(x->!iszero(x),t[2])])
function Leibniz.gdims(t::Vector{<:Chain})
    out = @inbounds zeros(Variables{mdims(Manifold(points(t)))+1,Int})
    @inbounds out[mdims(Manifold(t))+1] = length(t)
    return out
end
function Leibniz.gdims(t::Values{N,<:Vector}) where N
    out = @inbounds zeros(Variables{mdims(points(t[1]))+1,Int})
    for i ∈ list(1,N)
        @inbounds out[mdims(Manifold(t[i]))+1] = length(t[i])
    end
    return out
end
function Leibniz.gdims(t::Values{N,<:Tuple}) where N
    out = @inbounds zeros(Variables{mdims(points(t[1][1]))+1,Int})
    for i ∈ list(1,N)
        @inbounds out[mdims(Manifold(t[i][1]))+1] = length(t[i][1])
    end
    return out
end
function Leibniz.gdims(t::Multivector{V}) where V
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

Leibniz.χ(t::Values{N,<:Vector}) where N = (B=gdims(t);sum([B[t]*(-1)^t for t ∈ list(1,length(B))]))
Leibniz.χ(t::Values{N,<:Tuple}) where N = (B=gdims(t);sum([B[t]*(-1)^t for t ∈ list(1,length(B))]))

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

for implex ∈ (Single,Submanifold)
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
    #norm(t::Multivector) = norm(unsplitval(t))
    function unsplitvalue(a::Multivector{V,T}) where {V,T}
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
