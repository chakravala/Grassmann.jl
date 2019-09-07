module Grassmann

#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

using Combinatorics, StaticArrays, Requires
using ComputedFieldTypes, AbstractLattices
using DirectSum, AbstractTensors

export vectorspace, ⊕, ℝ, @V_str, @S_str, @D_str, Signature,DiagonalForm,SubManifold, value
import DirectSum: hasinf, hasorigin, mixedmode, dual, value, vectorspace, V0, ⊕, pre, vsn

include("utilities.jl")
include("multivectors.jl")
include("parity.jl")
include("algebra.jl")
include("composite.jl")
include("forms.jl")

## generators

function labels(V::T,vec::String=pre[1],cov::String=pre[2],duo::String=pre[3],dif::String=pre[4]) where T<:Manifold
    N,io,icr = ndims(V),IOBuffer(),1
    els = Array{Symbol,1}(undef,1<<N)
    els[1] = Symbol(vec)
    for i ∈ 1:N
        set = combo(N,i)
        for k ∈ 1:length(set)
            @inbounds DirectSum.printlabel(io,V,bit2int(indexbits(N,set[k])),true,vec,cov,duo,dif)
            icr += 1
            @inbounds els[icr] = Symbol(String(take!(io)))
        end
    end
    return els
end

#@pure labels(V::T) where T<:Manifold = labels(V,pre[1],pre[2],pre[3],pre[4])

@pure function generate(V::Manifold{N}) where N
    exp = Basis{V}[Basis{V,0,g_zero(Bits)}()]
    for i ∈ 1:N
        set = combo(N,i)
        for k ∈ 1:length(set)
            @inbounds push!(exp,Basis{V,i,bit2int(indexbits(N,set[k]))}())
        end
    end
    return exp
end

export @basis, @basis_str, @dualbasis, @dualbasis_str, @mixedbasis, @mixedbasis_str

function basis(V::Manifold,sig=vsn[1],vec=pre[1],cov=pre[2],duo=pre[3],dif=pre[4])
    N = ndims(V)
    if N > algebra_limit
        Λ(V) # fill cache
        basis = generate(V)
        sym = labels(V,string.([vec,cov,duo,dif])...)
    else
        basis = Λ(V).b
        sym = labels(V,string.([vec,cov,duo,dif])...)
    end
    @inbounds exp = Expr[Expr(:(=),esc(sig),V),
        Expr(:(=),esc(Symbol(vec)),basis[1])]
    for i ∈ 2:1<<N
        @inbounds push!(exp,Expr(:(=),esc(Symbol("$(basis[i])")),basis[i]))
        @inbounds push!(exp,Expr(:(=),esc(sym[i]),basis[i]))
    end
    push!(exp,Expr(:(=),esc(Symbol(vec,'⃖')) ,MultiVector(basis[1])))
    return Expr(:block,exp...,Expr(:tuple,esc(sig),esc.(sym)...))
end

macro basis(q,sig=vsn[1],vec=pre[1],cov=pre[2],duo=pre[3],dif=pre[4])
    basis(typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : vectorspace(q),sig,string.([vec,cov,duo,dif])...)
end

macro basis_str(str)
    basis(vectorspace(str))
end

macro dualbasis(q,sig=vsn[2],vec=pre[1],cov=pre[2],duo=pre[3],dif=pre[4])
    basis((typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : vectorspace(q))',sig,string.([vec,cov,duo,dif])...)
end

macro dualbasis_str(str)
    basis(vectorspace(str)',vsn[2])
end

macro mixedbasis(q,sig=vsn[3],vec=pre[1],cov=pre[2],duo=pre[3],dif=pre[4])
    V = typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : vectorspace(q)
    bases = basis(V⊕V',sig,string.([vec,cov,duo,dif])...)
    Expr(:block,bases,basis(V',vsn[2]),basis(V),bases.args[end])
end

macro mixedbasis_str(str)
    V = vectorspace(str)
    bases = basis(V⊕V',vsn[3])
    Expr(:block,bases,basis(V',vsn[2]),basis(V),bases.args[end])
end

@inline function lookup_basis(V::Manifold,v::Symbol)::Union{SBlade,Basis}
    p,b,w,z = DirectSum.indexparity(V,v)
    z && return g_zero(V)
    d = Basis{w}(b)
    return p ? SBlade(-1,d) : d
end

## fundamentals

export hyperplanes

@pure hyperplanes(V::Manifold{N}) where N = map(n->UniformScaling{Bool}(false)*getbasis(V,1<<n),0:N-1-diffvars(V))

abstract type SubAlgebra{V} <: TensorAlgebra{V} end

adjoint(G::A) where A<:SubAlgebra{V} where V = Λ(dual(V))
@pure dual(G::A) where A<: SubAlgebra = G'
Base.firstindex(a::T) where T<:SubAlgebra = 1
Base.lastindex(a::T) where T<:SubAlgebra{V} where V = 1<<ndims(V)
Base.length(a::T) where T<:SubAlgebra{V} where V = 1<<ndims(V)

==(::SubAlgebra{V},::SubAlgebra{W}) where {V,W} = V == W

⊕(::SubAlgebra{V},::SubAlgebra{W}) where {V,W} = getalgebra(V⊕W)
+(::SubAlgebra{V},::SubAlgebra{W}) where {V,W} = getalgebra(V⊕W)

for M ∈ (:Signature,:DiagonalForm)
    @eval (::$M)(::S) where S<:SubAlgebra{V} where V = MultiVector{Int,V}(ones(Int,1<<ndims(V)))
end

## Algebra{N}

@computed struct Algebra{V} <: SubAlgebra{V}
    b::SVector{1<<ndims(V),Basis{V}}
    g::Dict{Symbol,Int}
end

getindex(a::Algebra,i::Int) = getfield(a,:b)[i]
getindex(a::Algebra,i::Colon) = getfield(a,:b)
getindex(a::Algebra,i::UnitRange{Int}) = [getindex(a,j) for j ∈ i]

@pure function Base.getproperty(a::Algebra{V},v::Symbol) where V
    return if v ∈ (:b,:g)
        getfield(a,v)
    elseif haskey(a.g,v)
        a[getfield(a,:g)[v]]
    else
        lookup_basis(V,v)
    end
end

function Base.collect(s::Manifold)
    sym = labels(s)
    @inbounds Algebra{s}(generate(s),Dict{Symbol,Int}([sym[i]=>i for i ∈ 1:1<<ndims(s)]))
end

@pure Algebra(s::Manifold) = getalgebra(s)
@pure Algebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getalgebra(n,d,o,s)
Algebra(s::String) = getalgebra(vectorspace(s))
Algebra(s::String,v::Symbol) = getbasis(vectorspace(s),v)

function show(io::IO,a::Algebra{V}) where V
    N = ndims(V)
    print(io,"Grassmann.Algebra{$V,$(1<<N)}(")
    for i ∈ 1:1<<N-1
        print(io,a[i],", ")
    end
    print(io,a[end],")")
end

export Λ, @Λ_str, getalgebra, getbasis, TensorAlgebra, SubAlgebra

const Λ = Algebra

macro Λ_str(str)
    Algebra(str)
end

@pure function Base.getproperty(λ::typeof(Λ),v::Symbol)
    v ∈ (:body,:var) && (return getfield(λ,v))
    V = string(v)
    N = parse(Int,V[2])
    C = V[1]∉('D','C') ? 0 : 1
    length(V) < 5 && (V *= join(zeros(Int,5-length(V))))
    S = Bits(parse(Int,V[5:end]))
    getalgebra(N,doc2m(parse(Int,V[3]),parse(Int,V[4]),C),C>0 ? DirectSum.flip_sig(N,S) : S)
end

# Allocating thread-safe $(2^n)×Basis{VectorBundle}
const Λ0 = Λ{V0}(SVector{1,Basis{V0}}(Basis{V0,0,zero(Bits)}()),Dict(:e=>1))

for (vs,dat) ∈ ((:Signature,Bits),(:DiagonalForm,Int))
    algebra_cache = Symbol(:algebra_cache_,vs)
    getalg = Symbol(:getalgebra_,vs)
    @eval begin
        const $algebra_cache = Vector{Vector{Vector{Dict{$dat,Λ}}}}[]
        @pure function $getalg(n::Int,m::Int,s::$dat,f::Int=0,d::Int=0)
            n==0 && (return Λ0)
            n > sparse_limit && (return $(Symbol(:getextended_,vs))(n,m,s,f,d))
            n > algebra_limit && (return $(Symbol(:getsparse_,vs))(n,m,s,f,d))
            f1,d1,m1 = f+1,d+1,m+1
            for F ∈ length($algebra_cache)+1:f1
                push!($algebra_cache,Vector{Vector{Dict{$dat,Λ}}}[])
            end
            for D ∈ length($algebra_cache[f1])+1:d1
                push!($algebra_cache[f1],Vector{Dict{$dat,Λ}}[])
            end
            @inbounds for N ∈ length($algebra_cache[f1][d1])+1:n
                @inbounds push!($algebra_cache[f1][d1],[Dict{$dat,Λ}() for k∈1:12])
            end
            @inbounds if !haskey($algebra_cache[f1][d1][n][m1],s)
                @inbounds push!($algebra_cache[f1][d1][n][m1],s=>collect($vs{n,m,s,f,d}()))
            end
            @inbounds $algebra_cache[f1][d1][n][m1][s]
        end
        @pure function getalgebra(V::$vs{N,M,S,F,D}) where {N,M,S,F,D}
            mixedmode(V)<0 && N>2algebra_limit && (return getextended(V))
            $getalg(N,M,S,F,D)
        end
    end
end
for (vs,dat) ∈ ((:SubManifold,Bits),)
    algebra_cache = Symbol(:algebra_cache_,vs)
    getalg = Symbol(:getalgebra_,vs)
    for V ∈ (:Signature,:DiagonalForm)
        @eval const $(Symbol(algebra_cache,:_,V)) = Vector{Vector{Dict{$dat,Vector{Dict{$dat,Λ}}}}}[]
    end
    @eval begin
        @pure function $getalg(n::Int,m::Int,s::$dat,S::$dat,vs,f::Int=0,d::Int=0)
            n==0 && (return Λ0)
            n > sparse_limit && (return $(Symbol(:getextended_,vs))(n,m,s,f,d))
            n > algebra_limit && (return $(Symbol(:getsparse_,vs))(n,m,s,f,d))
            f1,d1,m1 = f+1,d+1,m+1
            alc = if vs <: Signature
                $(Symbol(algebra_cache,:_Signature))
            elseif vs <: DiagonalForm
                $(Symbol(algebra_cache,:_DiagonalForm))
            end
            for F ∈ length(alc)+1:f1
                push!(alc,Vector{Dict{$dat,Vector{Dict{$dat,Λ}}}}[])
            end
            for D ∈ length(alc[f1])+1:d1
                push!(alc[f1],Dict{$dat,Vector{Dict{$dat,Λ}}}[])
            end
            for D ∈ length(alc[f1][d1])+1:n
                push!(alc[f1][d1],Dict{$dat,Vector{Dict{$dat,Λ}}}())
            end
            @inbounds if !haskey(alc[f1][d1][n],S)
                @inbounds push!(alc[f1][d1][n],S=>[Dict{$dat,Λ}() for k∈1:12])
            end
            @inbounds if !haskey(alc[f1][d1][n][S][m1],s)
                @inbounds push!(alc[f1][d1][n][S][m1],s=>collect($vs{count_ones(S),vs(),S}()))
            end
            @inbounds alc[f1][d1][n][S][m1][s]
        end
        @pure function getalgebra(V::$vs{N,M,S}) where {N,M,S}
            mixedmode(V)<0 && N>2algebra_limit && (return getextended(V))
            $getalg(ndims(M),DirectSum.options(M),value(M),S,typeof(M),diffvars(M),DirectSum.diffmode(M))
        end
    end
end

@pure getalgebra(n::Int,d::Int,o::Int,s,c::Int=0) = getalgebra_Signature(n,doc2m(d,o,c),s)
@pure getalgebra(n::Int,m::Int,s) = getalgebra_Signature(n,m,Bits(s))

@pure getbasis(V::Manifold,v::Symbol) = getproperty(getalgebra(V),v)
@pure function getbasis(V::Manifold{N},b) where N
    B = Bits(b)
    if N ≤ algebra_limit
        @inbounds getalgebra(V).b[basisindex(ndims(V),B)]
    else
        Basis{V,count_ones(B),B}()
    end
end

## SparseAlgebra{V}

struct SparseAlgebra{V} <: SubAlgebra{V}
    b::Vector{Symbol}
    g::Dict{Symbol,Int}
end

@pure function SparseAlgebra(s::Manifold)
    sym = labels(s)
    SparseAlgebra{s}(sym,Dict{Symbol,Int}([sym[i]=>i for i ∈ 1:1<<ndims(s)]))
end

@pure function getindex(a::SparseAlgebra{V},i::Int) where V
    N = ndims(V)
    if N ≤ algebra_limit
        getalgebra(V).b[i]
    else
        F = findfirst(x->1+binomsum(N,x)-i>0,0:N)
        G = F ≠ nothing ? F-2 : N
        @inbounds B = indexbasis(N,G)[i-binomsum(N,G)]
        Basis{V,count_ones(B),B}()
    end
end

@pure function Base.getproperty(a::SparseAlgebra{V},v::Symbol) where V
    return if v ∈ (:b,:g)
        getfield(a,v)
    elseif haskey(a.g,v)
        @inbounds a[getfield(a,:g)[v]]
    else
        lookup_basis(V,v)
    end
end

@pure SparseAlgebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getsparse(n,d,o,s)
SparseAlgebra(s::String) = getsparse(vectorspace(s))
SparseAlgebra(s::String,v::Symbol) = getbasis(vectorspace(s),v)

function show(io::IO,a::SparseAlgebra{V}) where V
    print(io,"Grassmann.SparseAlgebra{$V,$(1<<ndims(V))}($(a[1]), ..., $(a[end]))")
end

## ExtendedAlgebra{V}

struct ExtendedAlgebra{V} <: SubAlgebra{V} end

@pure ExtendedAlgebra(s::Manifold) = ExtendedAlgebra{s}()

@pure function Base.getproperty(a::ExtendedAlgebra{V},v::Symbol) where V
    if v ∈ (:b,:g)
        throw(error("ExtendedAlgebra does not have field $v"))
    else
        return lookup_basis(V,v)
    end
end

@pure ExtendedAlgebra(n::Int,d::Int=0,o::Int=0,s=zero(Bits)) = getextended(n,d,o,s)
ExtendedAlgebra(s::String) = getextended(vectorspace(s))
ExtendedAlgebra(s::String,v::Symbol) = getbasis(vectorspace(s),v)

function show(io::IO,a::ExtendedAlgebra{V}) where V
    N = 1<<ndims(V)
    print(io,"Grassmann.ExtendedAlgebra{$V,$N}($(getbasis(V,0)), ..., $(getbasis(V,N-1)))")
end

# Extending (2^n)×Basis{Manifold}

for (ExtraAlgebra,extra) ∈ ((SparseAlgebra,:sparse),(ExtendedAlgebra,:extended))
    getextra = Symbol(:get,extra)
    gets = Symbol(getextra,:_Signature)
    for (vs,dat) ∈ ((:Signature,Bits),(:DiagonalForm,Int))
        extra_cache = Symbol(extra,:_cache_,vs)
        getalg = Symbol(:get,extra,:_,vs)
        @eval begin
            const $extra_cache = Vector{Vector{Vector{Dict{$dat,$ExtraAlgebra}}}}[]
            @pure function $getalg(n::Int,m::Int,s::$dat,f::Int=0,d::Int=0)
                n==0 && (return $ExtraAlgebra(V0))
                d1,f1,m1 = d+1,f+1,m+1
                for F ∈ length($extra_cache)+1:f1
                    push!($extra_cache,Vector{Vector{Dict{$dat,$ExtraAlgebra}}}[])
                end
                for D ∈ length($extra_cache[f1])+1:d1
                    push!($extra_cache[f1],Vector{Dict{$dat,$ExtraAlgebra}}[])
                end
                @inbounds for N ∈ length($extra_cache[f1][d1])+1:n
                    @inbounds push!($extra_cache[f1][d1],[Dict{$dat,$ExtraAlgebra}() for k∈1:12])
                end
                @inbounds if !haskey($extra_cache[f1][d1][n][m1],s)
                    @inbounds push!($extra_cache[f1][d1][n][m1],s=>$ExtraAlgebra($vs{n,m,s,f,d}()))
                end
                @inbounds $extra_cache[f1][d1][n][m1][s]
            end
            @pure $getextra(V::$vs{N,M,S,F,D}) where {N,M,S,F,D} = $getalg(N,M,S,F,D)
        end
    end
    vs,dat =  (:SubManifold,Bits)
    extra_cache = Symbol(extra,:_cache_,vs)
    getalg = Symbol(:get,extra,:_,vs)
    for V ∈ (:Signature,:DiagonalForm)
        @eval const $(Symbol(extra_cache,:_,V)) = Vector{Vector{Dict{$dat,Vector{Dict{$dat,$ExtraAlgebra}}}}}[]
    end
    @eval begin
        @pure function $getalg(n::Int,m::Int,s::$dat,S::$dat,vs,f::Int=0,d::Int=0)
            n==0 && (return $ExtraAlgebra(V0))
            d1,f1,m1 = d+1,f+1,m+1
            exc = if vs <: Signature
                $(Symbol(extra_cache,:_Signature))
            elseif vs <: DiagonalForm
                $(Symbol(extra_cache,:_DiagonalForm))
            end
            for F ∈ length(exc)+1:f1
                push!(exc,Vector{Dict{$dat,Vector{Dict{$dat,$ExtraAlgebra}}}}[])
            end
            for D ∈ length(exc[f1])+1:d1
                push!(exc[f1],Dict{$dat,Vector{Dict{$dat,$ExtraAlgebra}}}[])
            end
            for D ∈ length(exc[f1][d1])+1:n
                push!(exc[f1][d1],Dict{$dat,Vector{Dict{$dat,$ExtraAlgebra}}}())
            end
            @inbounds if !haskey(exc[f1][d1][n],S)
                @inbounds push!(exc[f1][d1][n],S=>[Dict{$dat,$ExtraAlgebra}() for k∈1:12])
            end
            @inbounds if !haskey(exc[f1][d1][n][S][m1],s)
                @inbounds push!(exc[f1][d1][n][S][m1],s=>$ExtraAlgebra($vs{count_ones(S),vs(),S}()))
            end
            @inbounds exc[f1][d1][n][S][m1][s]
        end
        @pure $getextra(V::$vs{N,M,S} where N) where {M,S} = $getalg(ndims(M),DirectSum.options(M),value(M),S,typeof(M),diffvars(M),DirectSum.diffmode(M))
    end
    @eval begin
        @pure $getextra(n::Int,d::Int,o::Int,s,c::Int=0) = $gets(n,doc2m(d,o,c),s)
        @pure $getextra(n::Int,m::Int,s) = $gets(n,m,Bits(s))
    end
end

# ParaAlgebra

using Leibniz
import Leibniz: ∂, d

generate_product_algebra(:(Leibniz.Operator),:svec)

function (V::Signature{N})(d::Leibniz.Derivation{T,O}) where {N,T,O}
    x=sum([(∂(V,k)^2) for k ∈ 1:N])^div(isodd(O) ? O-1 : O,2)
    isodd(O) ? sum([(x*∂(V,k))*getbasis(V,1<<(k-1)) for k ∈ 1:N]) : x*getbasis(V,0)
end

∂(ω::T) where T<:TensorAlgebra{V} where V = ω⋅V(∇)
d(ω::T) where T<:TensorAlgebra{V} where V = V(∇)∧ω

function __init__()
    @require Reduce="93e0c654-6965-5f22-aba9-9c1ae6b3c259" begin
        import Reduce: RExpr
        *(a::RExpr,b::Basis{V}) where V = SBlade{V}(a,b)
        *(a::Basis{V},b::RExpr) where V = SBlade{V}(b,a)
        *(a::RExpr,b::MultiVector{T,V}) where {T,V} = MultiVector{promote_type(T,F),V}(broadcast(Reduce.Algebra.:*,a,b.v))
        *(a::MultiVector{T,V},b::RExpr) where {T,V} = MultiVector{promote_type(T,F),V}(broadcast(Reduce.Algebra.:*,a.v,b))
        *(a::RExpr,b::MultiGrade{V}) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,a,b.v))
        *(a::MultiGrade{V},b::RExpr) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,a.v,b))
        ∧(a::RExpr,b::RExpr) = Reduce.Algebra.:*(a,b)
        ∧(a::RExpr,b::B) where B<:TensorTerm{V,G} where {V,G} = SBlade{V,G}(a,b)
        ∧(a::A,b::RExpr) where A<:TensorTerm{V,G} where {V,G} = SBlade{V,G}(b,a)
        parany = (parany...,RExpr)
        parval = (parval...,RExpr)
        parsym = (parsym...,RExpr)
    end
    @require SymPy="24249f21-da20-56a4-8eb1-6a02cf4ae2e6" generate_product_algebra(:(SymPy.Sym),:svec,:(SymPy.:*),:(SymPy.:+),:(SymPy.:-),:(SymPy.conj))
end

end # module
