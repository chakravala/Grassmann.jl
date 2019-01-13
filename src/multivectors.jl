
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export AbstractTerm, Basis, MultiVector, MultiGrade, @S_str, @V_str

abstract type AbstractTerm{V,G} end

## MultiBasis{N}

struct Basis{V,G,B} <: AbstractTerm{V,G}
    @pure Basis{V,G,B}() where {V,G,B} = new{V,G,B}()
end

@pure UInt16(b::Basis{V,G,B} where {V,G}) where B = B

function getindex(b::Basis,i::Int)
    d = 0x0001 << (i-1)
    return (d & UInt16(b)) == d
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

const VTI = Union{Vector{Int},Tuple,NTuple}

@inline basisindices(b::UInt16) = findall(digits(UInt16(b),base=2).==1)
@inline basisindices(b::Basis) = basisindices(UInt16(b))
@inline shiftbasis(b::Basis{V}) where V = shiftbasis(V,basisindices(b))
@inline shiftbasis(V::VectorSpace,b::UInt16) = shiftbasis(V,basisindices(b))

function shiftbasis(s::VectorSpace{N,D,O} where N,set::Vector{Int}) where {D,O}
    if !isempty(set)
        k = 1
        hasdual(s) && set[1] == 1 && (set[1] = -1; k += 1)
        shift = D + O
        hasorigin(s) && length(set)>=k && set[k]==shift && (set[k]=0;k+=1)
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

Basis{V}(i::UInt16) where V = getbasis(V,i)
Basis{V}(b::BitArray{1}) where V = getbasis(V,bit2int(b))

for t ∈ [[:V],[:V,:G]]
    @eval begin
        function Basis{$(t...)}(b::VTI) where {$(t...)}
            Basis{V}(basisbits(ndims(V),b))
        end
        function Basis{$(t...)}(b::Int...) where {$(t...)}
            Basis{V}(basisbits(ndims(V),b))
        end
    end
end

==(a::Basis{V,G},b::Basis{V,G}) where {V,G} = UInt16(a) == UInt16(b)
==(a::Basis{V,G} where V,b::Basis{W,L} where W) where {G,L} = false
==(a::Basis{V,G},b::Basis{W,G}) where {V,W,G} = throw(error("not implemented yet"))

@inline printbasis(io::IO,b::VTI,e::String="e") = print(io,e,[e≠"e" ? super[i] : subscripts[i] for i ∈ b]...)
@inline function printbasis(io::IO,a::VTI,b::VTI,e::String="e",f::String="f")
    F = !isempty(b)
    !(F && isempty(a)) && printbasis(io,a,e)
    F && printbasis(io,b,f)
end
@inline function printbasis(io::IO,V::VectorSpace,e::UInt16)
    C = dualtype(V)
    if C < 0
        N = Int(ndims(V)/2)
        printbasis(io,shiftbasis(V,e & UInt16(2^N-1)),shiftbasis(V,e>>N))
    else
        printbasis(io,shiftbasis(V,e),C>0 ? "f" : "e")
    end
end
@inline show(io::IO, e::Basis{V}) where V = printbasis(io,V,UInt16(e))

@pure function generate(V::VectorSpace{N},label::Symbol,dual::Symbol=:f) where N
    lab = string(label)
    io = IOBuffer()
    els = Symbol[label]
    exp = Basis{V}[Basis{V,0,0x0000}()]
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
                e = shiftbasis(V,a)
                f = shiftbasis(V,b)
                F = !isempty(f)
                !(F && isempty(e)) && print(io,lab,a...)
                F && print(io,string(dual),b...)
            else
                print(io,C>0 ? string(dual) : lab,[j≠0 ? (j > 0 ? j : 'ϵ') : 'o' for j∈shiftbasis(V,sk)]...)
            end
            push!(els,Symbol(String(take!(io))))
            push!(exp,Basis{V,length(set[k]),bit2int(basisbits(N,set[k]))}())
        end
    end
    return exp,els
end

export @basis, @basis_str, @dualbasis, @dualbasis_str, @mixedbasis, @mixedbasis_str

function basis(V::VectorSpace,sig::Symbol=:V,label::Symbol=:e,dual::Symbol=:f)
    N = ndims(V)
    basis,sym = generate(V,label,dual)
    exp = Expr[Expr(:(=),esc(sig),V),
        Expr(:(=),esc(label),basis[1])]
    for i ∈ 2:2^N
        push!(exp,Expr(:(=),esc(sym[i]),basis[i]))
    end
    return Expr(:block,exp...,Expr(:tuple,esc(sig),esc.(sym)...))
end

macro basis(q,sig=:V,label=:e,dual=:f)
    basis(typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q),sig,label,dual)
end

macro basis_str(str)
    basis(VectorSpace(str))
end

const indexbasis_cache = ( () -> begin
        Y = Vector{Vector{UInt16}}[]
        return (n::Int,g::Int) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    push!(Y,[[bit2int(basisbits(k,combo(k,G)[q])) for q ∈ 1:binomial(k,G)] for G ∈ 1:k])
                end
                g>0 ? Y[n][g] : [0x0000]
            end)
    end)()

@pure indexbasis(n::Int,g::Int) = indexbasis_cache(n,g)
@pure indexbasis_set(N) = SVector(Vector{UInt16}[indexbasis(N,g) for g ∈ 0:N]...)

macro dualbasis(q,sig=:VV,label=:e,dual=:f)
    basis((typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q))',sig,label,dual)
end

macro dualbasis_str(str)
    basis(VectorSpace(str)',:VV)
end

macro mixedbasis(q,sig=:W,label=:e,dual=:f)
    V = typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q)
    basis(V⊕V',sig,label,dual)
end

macro mixedbasis_str(str)
    V = VectorSpace(str)
    basis(V⊕V',:W)
end

## S/MValue{N}

const MSV = [:MValue,:SValue]

for Value ∈ MSV
    eval(Expr(:struct,Value ≠ :SValue,:($Value{V,G,B,T} <: AbstractTerm{V,G}),quote
        v::T
    end))
end
for Value ∈ MSV
    @eval begin
        export $Value
        $Value(b::Basis{V,G}) where {V,G} = $Value{V,G,b,Int}(1)
        $Value{V}(b::Basis{V,G}) where {V,G} = $Value{V,G,b,Int}(1)
        $Value{V}(v,b::SValue{V,G}) where {V,G} = $Value{V,G,basis(b)}(v*b.v)
        $Value{V}(v,b::MValue{V,G}) where {V,G} = $Value{V,G,basis(b)}(v*b.v)
        $Value{V}(v::T,b::Basis{V,G}) where {V,G,T} = $Value{V,G,b,T}(v)
        $Value{V,G}(v::T,b::Basis{V,G}) where {V,G,T} = $Value{V,G,b,T}(v)
        $Value{V,G,B}(v::T) where {V,G,B,T} = $Value{V,G,B,T}(v)
        $Value{V}(v::T) where {V,T} = $Value{V,0,Basis{V}(),T}(v)
        $Value(v,b::AbstractTerm{V,G}) where {V,G} = $Value{V,G,b}(v)
        show(io::IO,m::$Value) = print(io,(valuetype(m)≠Expr ? [m.v] : ['(',m.v,')'])...,basis(m))
    end
end

==(a::AbstractTerm{V,G},b::AbstractTerm{V,G}) where {V,G} = basis(a) == basis(b) && value(a) == value(b)
==(a::AbstractTerm,b::AbstractTerm) = false

## Grade{G}

struct Grade{G}
    @pure Grade{G}() where G = new{G}()
end

## Dimension{N}

struct Dimension{N}
    @pure Dimension{N}() where N = new{N}()
end

## S/MBlade{T,N}

const MSB = [:MBlade,:SBlade]

for (Blade,vector,Value) ∈ [(MSB[1],:MVector,MSV[1]),(MSB[2],:SVector,MSV[2])]
    @eval begin
        @computed struct $Blade{T,V,G}
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
                MValue{V,G,indexbasis(ndims(V),G)[i],T}(m[i])
            else
                SValue{V,G,indexbasis(ndims(V),G)[i],T}(m[i])
            end
        end

        function $Blade{T,V,G}(val::T,v::Basis{V,G}) where {T,V,G}
            SBlade{T,V}(setblade!(zeros(mvec(ndims(V),G,T)),val,UInt16(v),Dimension{N}()))
        end

        $Blade(v::Basis{V,G}) where {V,G} = $Blade{Int,V,G}(one(Int),v)

        function show(io::IO, m::$Blade{T,V,G}) where {T,V,G}
            ib = indexbasis(ndims(V),G)
            if T == Any && typeof(m.v[1]) ∈ (Expr,Symbol)
                typeof(m.v[1])≠Expr ? print(io,m.v[1]) : print(io,"(",m.v[1],")")
            else
                print(io,m.v[1])
            end
            printbasis(io,V,ib[1])
            for k ∈ 2:length(ib)
                if T == Any && typeof(m.v[k]) ∈ (Expr,Symbol)
                    typeof(m.v[k])≠Expr ? print(io," + ",m.v[k]) : print(io," + (",m.v[k],")")
                else
                    print(io,signbit(m.v[k]) ? " - " : " + ",abs(m.v[k]))
                end
                printbasis(io,V,ib[k])
            end
        end
    end
    for var ∈ [[:T,:V,:G],[:T,:V],[:T]]
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
for (Blade,Other,Vec) ∈ [(MSB...,:MVector),(reverse(MSB)...,:SVector)]
    for var ∈ [[:T,:V,:G],[:T,:V],[:T],[]]
        @eval begin
            $Blade{$(var...)}(v::$Other{T,V,G}) where {T,V,G} = $Blade{T,V,G}($Vec{binomial(ndims(V),G),T}(v.v))
        end
    end
end
for (Blade,Vec1,Vec2) ∈ [(MSB[1],:SVector,:MVector),(MSB[2],:MVector,:SVector)]
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

struct MultiVector{T,V,E}
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

function (m::MultiVector{T,V})(g::Int,B::Type=SBlade) where {T,V}
    B ≠ SBlade ? MBlade{T,V,g}(m[g]) : SBlade{T,V,g}(m[g])
end
function (m::MultiVector{T,V})(g::Int,i::Int,B::Type=SValue) where {T,V}
    if B ≠ SValue
        MValue{V,g,indexbasis(ndims(V),g)[i],T}(m[g][i])
    else
        SValue{V,g,indexbasis(ndims(V),g)[i],T}(m[g][i])
    end
end

MultiVector{V}(v::StaticArray{Tuple{M},T,1}) where {V,T,M} = MultiVector{T,V}(v)
for var ∈ [[:T,:V],[:V]]
    @eval begin
        MultiVector{$(var...)}(v::SizedArray) where {T,V} = MultiVector{T,V}(SVector{2^ndims(V),T}(v))
        MultiVector{$(var...)}(v::Vector{T}) where {T,V} = MultiVector{T,V}(SVector{2^ndims(V),T}(v))
        MultiVector{$(var...)}(v::T...) where {T,V} = MultiVector{T,V}(SVector{2^ndims(V),T}(v))
        function MultiVector{$(var...)}(val::T,v::Basis{V,G}) where {T,V,G}
            N = ndims(V)
            MultiVector{T,V}(setmulti!(zeros(mvec(N,T)),val,UInt16(v),Dimension{N}()))
        end
    end
end
function MultiVector(val::T,v::Basis{V,G}) where {T,V,G}
    N = ndims(V)
    MultiVector{T,V}(setmulti!(zeros(mvec(N,T)),val,UInt16(v),Dimension{N}()))
end

MultiVector(v::Basis{V,G}) where {V,G} = MultiVector{Int,V}(one(Int),v)

for var ∈ [[:T,:V],[:T]]
    @eval begin
        function MultiVector{$(var...)}(v::Basis{V,G}) where {T,V,G}
            return MultiVector{T,V}(one(T),v)
        end
    end
end
for var ∈ [[:T,:V],[:T],[]]
    for (Value,Blade) ∈ [(MSV[1],MSB[1]),(MSV[2],MSB[2])]
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
    for i ∈ 1:N
        b = m[i]
        ib = indexbasis(N,i)
        for k ∈ 1:length(ib)
            if b[k] ≠ 0
                if T == Any && typeof(b[k]) ∈ (Expr,Symbol)
                    typeof(b[k])≠Expr ? print(io," + ",b[k]) : print(io," + (",b[k],")")
                else
                    print(io,signbit(b[k]) ? " - " : " + ",abs(b[k]))
                end
                printbasis(io,V,ib[k])
            end
        end
    end
end

==(a::MultiVector{T,V},b::MultiVector{T,V}) where {T,V} = a.v == b.v
==(a::MultiVector{T,V} where T,b::MultiVector{S,V} where S) where V = false
==(a::MultiVector{T,V} where T,b::MultiVector{S,W} where S) where {V,W,G} = throw(error("not implemented yet"))

## Generic

const VBV = Union{MValue,SValue,MBlade,SBlade,MultiVector}

@pure ndims(::VectorSpace{N}) where N = N
@pure ndims(::Basis{V}) where V = ndims(V)
@pure valuetype(::Basis) = Int
@pure valuetype(::Union{MValue{V,G,B,T},SValue{V,G,B,T}} where {V,G,B}) where T = T
@pure valuetype(::Union{MBlade{T},SBlade{T}}) where T = T
@pure valuetype(::MultiVector{T}) where T = T
@inline value(::Basis,T=Int) = one(T)
@inline value(m::VBV,T::DataType=valuetype(m)) = T≠valuetype(m) ? convert(T,m.v) : m.v
@inline value(::VectorSpace{N,D,O,S}) where {N,D,O,S} = S
@inline sig(::Basis{V}) where V = V
@inline sig(::Union{MValue{V},SValue{V}}) where V = V
@inline sig(m::Union{MBlade{T,V},SBlade{T,V},MultiVector{T,V}} where T) where V = V
@pure basis(m::Basis) = m
@pure basis(m::Union{MValue{V,G,B},SValue{V,G,B}}) where {V,G,B} = B
@inline grade(m::AbstractTerm{V,G} where V) where G = G
@inline grade(m::Union{MBlade{T,V,G},SBlade{T,V,G}} where {T,V}) where G = G
@inline grade(m::Number) = 0

hasdual(::VectorSpace{N,D} where N) where D = Bool(D)
hasorigin(::VectorSpace{N,D,O} where {N,D}) where O = Bool(O)

isdual(e::Basis{V}) where V = hasdual(e) && count_ones(UInt16(e)) == 1
hasdual(e::Basis{V}) where V = hasdual(V) && isodd(UInt16(e))

isorigin(e::Basis{V}) where V = hasorigin(V) && count_ones(UInt16(e))==1 && e[hasdual(V)+1]
hasorigin(e::Basis{V}) where V = hasorigin(V) && (hasdual(V) ? e[2] : isodd(UInt16(e)))
hasorigin(t::Union{MValue,SValue}) = hasorigin(basis(t))
hasorigin(m::VBV) = hasorigin(sig(m))

## MultiGrade{N}

struct MultiGrade{V}
    v::Vector{<:AbstractTerm{V}}
end

#convert(::Type{Vector{<:AbstractTerm{V}}},m::Tuple) where V = [m...]

MultiGrade{V}(v::T...) where T <: (AbstractTerm{V,G} where G) where V = MultiGrade{V}(v)
MultiGrade(v::T...) where T <: (AbstractTerm{V,G} where G) where V = MultiGrade{V}(v)

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
        setmulti!(out,convert(T,val),UInt16(b),Dimension{N}())
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

