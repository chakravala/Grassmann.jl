
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export TensorTerm, TensorMixed, Basis, MultiVector, MultiGrade

abstract type TensorTerm{V,G} <: TensorAlgebra{V} end
abstract type TensorMixed{T,V} <: TensorAlgebra{V} end

# print tools

import DirectSum: indexbits, indices, shift_indices, printindex, printindices, VTI

## MultiBasis{N}

struct Basis{V,G,B} <: TensorTerm{V,G}
    @pure Basis{V,G,B}() where {V,G,B} = new{V,G,B}()
end

@pure bits(b::Basis{V,G,B} where {V,G}) where B = B
@pure Base.one(b::Type{Basis{V}}) where V = getbasis(V,bits(b))
@pure Base.zero(V::VectorSpace) = 0*one(V)
@pure Base.one(V::VectorSpace) = Basis{V}()

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

@inline indices(b::Basis) = indices(bits(b))
@inline shift_indices(b::Basis{V}) where V = shift_indices(V,indices(b))

Basis{V}(i::Bits) where V = getbasis(V,i)
Basis{V}(b::BitArray{1}) where V = getbasis(V,bit2int(b))

for t ∈ ((:V,),(:V,:G))
    @eval begin
        function Basis{$(t...)}(b::VTI) where {$(t...)}
            Basis{V}(indexbits(ndims(V),b))
        end
        function Basis{$(t...)}(b::Int...) where {$(t...)}
            Basis{V}(indexbits(ndims(V),b))
        end
    end
end

==(a::Basis{V,G},b::Basis{V,G}) where {V,G} = bits(a) == bits(b)
==(a::Basis{V,G} where V,b::Basis{W,L} where W) where {G,L} = false
==(a::Basis{V,G},b::Basis{W,G}) where {V,W,G} = throw(error("not implemented yet"))

@inline show(io::IO, e::Basis{V}) where V = printindices(io,V,bits(e))

@pure function labels(V::VectorSpace{N},label::Symbol=Symbol(pre[1]),dual::Symbol=Symbol(pre[2])) where N
    lab = string(label)
    io = IOBuffer()
    els = Array{Symbol,1}(undef,1<<N)
    els[1] = label
    icr = 1
    C = dualtype(V)
    C < 0 && (M = Int(N/2))
    for i ∈ 1:N
        set = combo(N,i)
        for k ∈ 1:length(set)
            sk = copy(set[k])
            if C < 0
                a = Int[]
                b = Int[]
                for j ∈ sk
                    push!(j ≤ M ? a : b, j)
                end
                b .-= M
                e = shift_indices(V,a)
                f = shift_indices(V,b)
                F = !isempty(f)
                if !(F && isempty(e))
                    print(io,lab,a[1:min(9,end)]...)
                    for j ∈ 10:length(a)
                        print(io,subs[j])
                    end
                end
                if F
                    print(io,string(dual),b[1:min(9,end)]...)
                    for j ∈ 10:length(b)
                        print(io,sups[j])
                    end
                end
            else
                print(io,C>0 ? string(dual) : lab)
                for j ∈ shift_indices(V,sk)
                    print(io,j≠0 ? (j>0 ? (j>9 ? (C>0 ? sups[j] : subs[j]) : j) : 'ϵ') : 'o')
                end
            end
            icr += 1
            els[icr] = Symbol(String(take!(io)))
        end
    end
    return els
end

@pure function generate(V::VectorSpace{N}) where N
    exp = Basis{V}[Basis{V,0,zero(Bits)}()]
    for i ∈ 1:N
        set = combo(N,i)
        for k ∈ 1:length(set)
            push!(exp,Basis{V,i,bit2int(indexbits(N,set[k]))}())
        end
    end
    return exp
end

export @basis, @basis_str, @dualbasis, @dualbasis_str, @mixedbasis, @mixedbasis_str

function basis(V::VectorSpace,sig::Symbol=vsn[1],label::Symbol=Symbol(pre[1]),dual::Symbol=Symbol(pre[2]))
    N = ndims(V)
    if N > algebra_limit
        Λ(V)
        basis = generate(V)
        sym = labels(V,label,dual)
    else
        basis = Λ(V).b
        sym = labels(V,label,dual)
    end
    exp = Expr[Expr(:(=),esc(sig),V),
        Expr(:(=),esc(label),basis[1])]
    for i ∈ 2:1<<N
        push!(exp,Expr(:(=),esc(sym[i]),basis[i]))
    end
    return Expr(:block,exp...,Expr(:tuple,esc(sig),esc.(sym)...))
end

macro basis(q,sig=vsn[1],label=Symbol(pre[1]),dual=Symbol(pre[2]))
    basis(typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q),sig,label,dual)
end

macro basis_str(str)
    basis(VectorSpace(str))
end

const indexbasis_cache = Vector{Vector{Bits}}[]
@pure function indexbasis(n::Int,g::Int)
    n>sparse_limit && (return [bit2int(indexbits(n,combo(n,g)[q])) for q ∈ 1:binomial(n,g)])
    for k ∈ length(indexbasis_cache)+1:n
        push!(indexbasis_cache,[[bit2int(indexbits(k,combo(k,G)[q])) for q ∈ 1:binomial(k,G)] for G ∈ 1:k])
    end
    g>0 ? indexbasis_cache[n][g] : [zero(Bits)]
end

indexbasis(Int((sparse_limit+cache_limit)/2),1)

@pure indexbasis_set(N) = SVector(Vector{Bits}[indexbasis(N,g) for g ∈ 0:N]...)

macro dualbasis(q,sig=vsn[2],label=Symbol(pre[1]),dual=Symbol(pre[2]))
    basis((typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q))',sig,label,dual)
end

macro dualbasis_str(str)
    basis(VectorSpace(str)',vsn[2])
end

macro mixedbasis(q,sig=vsn[3],label=Symbol(pre[1]),dual=Symbol(pre[2]))
    V = typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q)
    bases = basis(V⊕V',sig,label,dual)
    Expr(:block,bases,basis(V',vsn[2]),basis(V),bases.args[end])
end

macro mixedbasis_str(str)
    V = VectorSpace(str)
    bases = basis(V⊕V',vsn[3])
    Expr(:block,bases,basis(V',vsn[2]),basis(V),bases.args[end])
end

## S/MValue{N}

const MSV = (:MValue,:SValue)

for Value ∈ MSV
    eval(Expr(:struct,Value ≠ :SValue,:($Value{V,G,B,T} <: TensorTerm{V,G}),quote
        v::T
    end))
end
for Value ∈ MSV
    @eval begin
        export $Value
        $Value(b::Basis{V,G}) where {V,G} = $Value{V}(b)
        $Value(v,b::TensorTerm{V}) where V = $Value{V}(v,b)
        $Value{V}(b::Basis{V,G}) where {V,G} = $Value{V,G,b,Int}(1)
        $Value{V}(v,b::SValue{V,G}) where {V,G} = $Value{V,G,basis(b)}(v*b.v)
        $Value{V}(v,b::MValue{V,G}) where {V,G} = $Value{V,G,basis(b)}(v*b.v)
        $Value{V}(v::T,b::Basis{V,G}) where {V,G,T} = $Value{V,G,b,T}(v)
        $Value{V,G}(v::T,b::Basis{V,G}) where {V,G,T} = $Value{V,G,b,T}(v)
        $Value{V,G,B}(v::T) where {V,G,B,T} = $Value{V,G,B,T}(v)
        $Value{V}(v::T) where {V,T} = $Value{V,0,Basis{V}(),T}(v)
        show(io::IO,m::$Value) = print(io,(valuetype(m)∉(Expr,Any) ? [m.v] : ['(',m.v,')'])...,basis(m))
    end
end

==(a::TensorTerm{V,G},b::TensorTerm{V,G}) where {V,G} = basis(a) == basis(b) && value(a) == value(b)
==(a::TensorTerm,b::TensorTerm) = false

## Grade{G}

struct Grade{G}
    @pure Grade{G}() where G = new{G}()
end

## Dimension{N}

struct Dimension{N}
    @pure Dimension{N}() where N = new{N}()
end

## S/MBlade{T,N}

const MSB = (:MBlade,:SBlade)

for (Blade,vector,Value) ∈ ((MSB[1],:MVector,MSV[1]),(MSB[2],:SVector,MSV[2]))
    @eval begin
        @computed struct $Blade{T,V,G} <: TensorMixed{T,V}
            v::$vector{binomial(ndims(V),G),T}
        end

        export $Blade
        getindex(m::$Blade,i::Int) = m.v[i]
        setindex!(m::$Blade{T},k::T,i::Int) where T = (m.v[i] = k)
        Base.firstindex(m::$Blade) = 1
        Base.lastindex(m::$Blade{T,N,G}) where {T,N,G} = length(m.v)
        Base.length(s::$Blade{T,N,G}) where {T,N,G} = length(m.v)
    end
    @eval begin
        function (m::$Blade{T,V,G})(i::Integer,B::Type=SValue) where {T,V,G}
            if B ≠ SValue
                MValue{V,G,Basis{V}(indexbasis(ndims(V),G)[i]),T}(m[i])
            else
                SValue{V,G,Basis{V}(indexbasis(ndims(V),G)[i]),T}(m[i])
            end
        end

        function $Blade{T,V,G}(val::T,v::Basis{V,G}) where {T,V,G}
            SBlade{T,V}(setblade!(zeros(mvec(ndims(V),G,T)),val,bits(v),Dimension{N}()))
        end

        $Blade(v::Basis{V,G}) where {V,G} = $Blade{Int,V,G}(one(Int),v)

        function show(io::IO, m::$Blade{T,V,G}) where {T,V,G}
            ib = indexbasis(ndims(V),G)
            if T == Any && typeof(m.v[1]) ∈ (Expr,Symbol)
                typeof(m.v[1])≠Expr ? print(io,m.v[1]) : print(io,"(",m.v[1],")")
            else
                print(io,m.v[1])
            end
            printindices(io,V,ib[1])
            for k ∈ 2:length(ib)
                if T == Any && typeof(m.v[k]) ∈ (Expr,Symbol)
                    typeof(m.v[k])≠Expr ? print(io," + ",m.v[k]) : print(io," + (",m.v[k],")")
                else
                    print(io,signbit(m.v[k]) ? " - " : " + ",abs(m.v[k]))
                end
                printindices(io,V,ib[k])
            end
        end
    end
    for var ∈ ((:T,:V,:G),(:T,:V),(:T,))
        @eval begin
            $Blade{$(var...)}(v::Basis{V,G}) where {T,V,G} = $Blade{T,V,G}(one(T),v)
        end
    end
    for var ∈ [[:T,:V,:G],[:T,:V],[:T],[]]
        @eval begin
            $Blade{$(var...)}(v::SValue{V,G,B,T}) where {T,V,G,B} = $Blade{T,V,G}(v.v,basis(v))
            $Blade{$(var...)}(v::MValue{V,G,B,T}) where {T,V,G,B} = $Blade{T,V,G}(v.v,basis(v))
        end
    end
end
for (Blade,Other,Vec) ∈ ((MSB...,:MVector),(reverse(MSB)...,:SVector))
    for var ∈ ((:T,:V,:G),(:T,:V),(:T,),())
        @eval begin
            $Blade{$(var...)}(v::$Other{T,V,G}) where {T,V,G} = $Blade{T,V,G}($Vec{binomial(ndims(V),G),T}(v.v))
        end
    end
end
for (Blade,Vec1,Vec2) ∈ ((MSB[1],:SVector,:MVector),(MSB[2],:MVector,:SVector))
    @eval begin
        $Blade{T,V,G}(v::$Vec1{M,T} where M) where {T,V,G} = $Blade{T,V,G}($Vec2{binomial(ndims(V),G),T}(v))
    end
end
for Blade ∈ MSB, Other ∈ MSB
    @eval begin
        ==(a::$Blade{T,V,G},b::$Other{T,V,G}) where {T,V,G} = a.v == b.v
        ==(a::$Blade{T,V} where T,b::$Other{S,V} where S) where V = false
        ==(a::$Blade{T,V} where T,b::$Other{S,W} where S) where {V,W,G} = throw(error("not implemented yet"))
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

function (m::MultiVector{T,V})(g::Int,::Type{B}=SBlade) where {T,V,B}
    B ≠ SBlade ? MBlade{T,V,g}(m[g]) : SBlade{T,V,g}(m[g])
end
function (m::MultiVector{T,V})(g::Int,i::Int,::Type{B}=SValue) where {T,V,B}
    if B ≠ SValue
        MValue{V,g,Basis{V}(indexbasis(ndims(V),g)[i]),T}(m[g][i])
    else
        SValue{V,g,Basis{V}(indexbasis(ndims(V),g)[i]),T}(m[g][i])
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
    for (Value,Blade) ∈ ((MSV[1],MSB[1]),(MSV[2],MSB[2]))
        @eval begin
            function MultiVector{$(var...)}(v::$Value{V,G,B,T}) where {V,G,B,T}
                return MultiVector{T,V}(v.v,basis(v))
            end
            function MultiVector{$(var...)}(v::$Blade{T,V,G}) where {T,V,G}
                N = ndims(V)
                out = zeros(mvec(N,T))
                r = binomsum(N,G)
                out.v[r+1:r+binomial(N,G)] = v.v
                return MultiVector{T,V}(out)
            end
        end
    end
end

function show(io::IO, m::MultiVector{T,V}) where {T,V}
    N = ndims(V)
    print(io,m[0][1])
    bs = binomsum_set(N)
    for i ∈ 2:N+1
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            s = k+bs[i]
            if m.v[s] ≠ 0
                if T == Any && typeof(m.v[s]) ∈ (Expr,Symbol)
                    typeof(m.v[s])≠Expr ? print(io," + ",m.v[s]) : print(io," + (",m.v[s],")")
                else
                    print(io,signbit(m.v[s]) ? " - " : " + ",abs(m.v[s]))
                end
                printindices(io,V,ib[k])
            end
        end
    end
end

==(a::MultiVector{T,V},b::MultiVector{T,V}) where {T,V} = a.v == b.v
==(a::MultiVector{T,V} where T,b::MultiVector{S,V} where S) where V = false
==(a::MultiVector{T,V} where T,b::MultiVector{S,W} where S) where {V,W,G} = throw(error("not implemented yet"))

## Generic

export basis, grade, hasdual, hasorigin, isdual, isorigin

const VBV = Union{MValue,SValue,MBlade,SBlade,MultiVector}

@pure ndims(::Basis{V}) where V = ndims(V)
@pure valuetype(::Basis) = Int
@pure valuetype(::Union{MValue{V,G,B,T},SValue{V,G,B,T}} where {V,G,B}) where T = T
@pure valuetype(::TensorMixed{T}) where T = T
@inline value(::Basis,T=Int) = one(T)
@inline value(m::VBV,T::DataType=valuetype(m)) = T≠valuetype(m) ? convert(T,m.v) : m.v
@inline sig(::TensorAlgebra{V}) where V = V
@pure basis(m::Basis) = m
@pure basis(m::Union{MValue{V,G,B},SValue{V,G,B}}) where {V,G,B} = B
@inline grade(m::TensorTerm{V,G} where V) where G = G
@inline grade(m::Union{MBlade{T,V,G},SBlade{T,V,G}} where {T,V}) where G = G
@inline grade(m::Number) = 0

isdual(e::Basis{V}) where V = hasdual(e) && count_ones(bits(e)) == 1
hasdual(e::Basis{V}) where V = hasdual(V) && isodd(bits(e))

isorigin(e::Basis{V}) where V = hasorigin(V) && count_ones(bits(e))==1 && e[hasdual(V)+1]
hasorigin(e::Basis{V}) where V = hasorigin(V) && (hasdual(V) ? e[2] : isodd(bits(e)))
hasorigin(t::Union{MValue,SValue}) = hasorigin(basis(t))
hasorigin(m::TensorAlgebra) = hasorigin(sig(m))

## MultiGrade{N}

struct MultiGrade{V} <: TensorAlgebra{V}
    v::Vector{<:TensorTerm{V}}
end

#convert(::Type{Vector{<:TensorTerm{V}}},m::Tuple) where V = [m...]

MultiGrade{V}(v::T...) where T <: (TensorTerm{V,G} where G) where V = MultiGrade{V}(v)
MultiGrade(v::T...) where T <: (TensorTerm{V,G} where G) where V = MultiGrade{V}(v)

function bladevalues(V::VectorSpace{N},m,G::Int,T::Type) where N
    com = indexbasis(N,G)
    out = (SValue{V,G,B,T} where B)[]
    for i ∈ 1:binomial(N,G)
        m[i] ≠ 0 && push!(out,SValue{V,G,Basis{V,G}(com[i]),T}(m[i]))
    end
    return out
end

MultiGrade(v::MultiVector{T,V}) where {T,V} = MultiGrade{V}(vcat([bladevalues(V,v[g],g,T) for g ∈ 1:ndims(V)]...))

for Blade ∈ MSB
    @eval begin
        MultiGrade(v::$Blade{T,V,G}) where {T,V,G} = MultiGrade{V}(bladevalues(V,v,G,T))
    end
end

#=function MultiGrade{V}(v::(MultiBlade{T,V} where T <: Number)...) where V
    sigcheck(v.s,V)
    t = typeof.(v)
    MultiGrade{V}([bladevalues(V,v[i],N,t[i].parameters[3],t[i].parameters[1]) for i ∈ 1:length(v)])
end

MultiGrade(v::(MultiBlade{T,N} where T <: Number)...) where N = MultiGrade{N}(v)=#

function MultiVector{T,V}(v::MultiGrade{V}) where {T,V}
    N = ndims(V)
    sigcheck(v.s,V)
    g = grade.(v.v)
    out = zeros(mvec(N,T))
    for k ∈ 1:length(v.v)
        (val,b) = typeof(v.v[k]) <: Basis ? (one(T),v.v[k]) : (v.v[k].v,basis(v.v[k]))
        setmulti!(out,convert(T,val),bits(b),Dimension{N}())
    end
    return MultiVector{T,V}(out)
end

MultiVector{T}(v::MultiGrade{V}) where {T,V} = MultiVector{T,V}(v)
MultiVector(v::MultiGrade{V}) where V = MultiVector{promote_type(typeval.(v.v)...),V}(v)

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
        show(io,t ? basis(x) : x)
    end
end

## Adjoint

import Base: adjoint # conj

function adjoint(b::Basis{V,G,B}) where {V,G,B}
    Basis{dual(V)}(dualtype(V)<0 ? dual(V,B) : B)
end

## conversions

@pure function (W::VectorSpace)(b::Basis{V}) where V
    V==W && (return B)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = dualtype(W),dualtype(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    if WC<0 && VC≥0
        return getbasis(W,VC>0 ? bits(b)<<ndims(V) : bits(b))
    else
        throw(error("arbitrary VectorSpace intersection not yet implemented."))
    end
end
@pure (W::VectorSpace)(b::SValue) = SValue{W}(value(b),W(basis(b)))
(W::VectorSpace)(b::MValue) = MValue{W}(value(b),W(basis(b)))

for Blade ∈ MSB
    @eval begin
        function (W::VectorSpace)(b::$Blade{T,V,G}) where {T,V,G}
            V==W && (return b)
            !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
            WC,VC = dualtype(W),dualtype(V)
            #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
            #    return V0
            if WC<0 && VC≥0
                N,M = ndims(V),ndims(W)
                out = zeros(mvec(M,G,T))
                ib = indexbasis(N,G)
                for k ∈ 1:length(ib)
                    if b[k] ≠ 0
                        setblade!(out,b[k],VC>0 ? ib[k]<<N : ib[k],Dimension{M}())
                    end
                end
                return $Blade{T,W,G}(out)
            else
                throw(error("arbitrary VectorSpace intersection not yet implemented."))
            end
        end

    end
end

function (W::VectorSpace)(m::MultiVector{T,V}) where {T,V}
    V==W && (return m)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = dualtype(W),dualtype(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    if WC<0 && VC≥0
        N,M = ndims(V),ndims(W)
        out = zeros(mvec(M,T))
        bs = binomsum_set(N)
        for i ∈ 1:N+1
            ib = indexbasis(N,i-1)
            for k ∈ 1:length(ib[i])
                s = k+bs[i]
                if m.v[s] ≠ 0
                    setmulti!(out,m.v[s],VC>0 ? ib[k]<<N : ib[k],Dimension{M}())
                end
            end
        end
        return MultiVector{T,W}(out)
    else
        throw(error("arbitrary VectorSpace intersection not yet implemented."))
    end
end

