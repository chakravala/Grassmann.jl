
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: @pure, print, show, getindex, setindex!, promote_rule, ==, convert, ndims
import DirectSum: Bits, bit2int, doc2m, indexbits, indices, diffmode, Dim

bcast(op,arg) = op ∈ (:(DirectSum.:∑),:(DirectSum.:-)) ? Expr(:.,op,arg) : Expr(:call,op,arg.args...)

@pure binomial_set(N) = SVector(Int[binomial(N,g) for g ∈ 0:N]...)
@pure binomial(N,G) = Base.binomial(N,G)
@pure mvec(N,G,t) = MVector{binomial(N,G),t}
@pure mvec(N,t) = MVector{2^N,t}
@pure svec(N,G,t) = SizedArray{Tuple{binomial(N,G)},t,1,1}
@pure svec(N,t) = SizedArray{Tuple{1<<N},t,1,1}

const algebra_limit = 8
const sparse_limit = 22
const cache_limit = 12
const fill_limit = 0.5

const binomsum_cache = [[0],[0,1]]
const binomsum_extra = Vector{Int}[]
@pure function binomsum(n::Int, i::Int)::Int
    if n>sparse_limit
        N=n-sparse_limit
        for k ∈ length(binomsum_extra)+1:N
            push!(binomsum_extra,Int[])
        end
        @inbounds isempty(binomsum_extra[N]) && (binomsum_extra[N]=[0;cumsum([binomial(n,q) for q=0:n])])
        @inbounds binomsum_extra[N][i+1]
    else
        for k=length(binomsum_cache):n+1
            push!(binomsum_cache, [0;cumsum([binomial(k,q) for q=0:k])])
        end
        @inbounds binomsum_cache[n+1][i+1]
    end
end
@pure function binomsum_set(n::Int)::Vector{Int}
    if n>sparse_limit
        N=n-sparse_limit
        for k ∈ length(binomsum_extra)+1:N
            push!(binomsum_extra,Int[])
        end
        @inbounds isempty(binomsum_extra[N]) && (binomsum_extra[N]=[0;cumsum([binomial(n,q) for q=0:n])])
        @inbounds binomsum_extra[N]
    else
        for k=length(binomsum_cache):n+1
            push!(binomsum_cache, [0;cumsum([binomial(k,q) for q=0:k])])
        end
        @inbounds binomsum_cache[n+1]
    end
end

const combo_cache = Vector{Vector{Vector{Int}}}[]
const combo_extra = Vector{Vector{Vector{Int}}}[]
function combo(n::Int,g::Int)::Vector{Vector{Int}}
    if g == 0
        [Int[]]
    elseif n>sparse_limit
        N=n-sparse_limit
        for k ∈ length(combo_extra)+1:N
            push!(combo_extra,Vector{Vector{Int}}[])
        end
        @inbounds for k ∈ length(combo_extra[N])+1:g
            @inbounds push!(combo_extra[N],Vector{Int}[])
        end
        @inbounds isempty(combo_extra[N][g]) && (combo_extra[N][g]=collect(combinations(1:n,g)))
        @inbounds combo_extra[N][g]
    else
        for k ∈ length(combo_cache)+1:min(n,sparse_limit)
            z = 1:k
            push!(combo_cache,[collect(combinations(z,q)) for q ∈ z])
        end
        @inbounds combo_cache[n][g]
    end
end

#=const basisindex_cache = Array{Dict{String,Int},1}[]
function basisindex(n::Int,i::Array{Int,1})
    g = length(i)
    s = string(i...)
    for k ∈ length(basisindex_cache)+1:n
        push!(basisindex_cache,[Dict{String,Int}() for k ∈ 1:k])
    end
    g>0 && !haskey(basisindex_cache[n][g],s) &&
        push!(basisindex_cache[n][g],s=>findall(x->x==i,combo(n,g))[1])
    g>0 ? basisindex_cache[n][g][s] : 1
end=#

@pure function bladeindex_calc(d,k)
    H = indices(UInt(d),k)
    findall(x->x==H,combo(k,length(H)))[1]
end
const bladeindex_cache = Vector{Int}[]
const bladeindex_extra = Vector{Int}[]
@pure function bladeindex(n::Int,s::Bits)::Int
    if s == 0
        1
    elseif n>(DirectSum.index_limit)
        bladeindex_calc(s,n)
    elseif n>cache_limit
        N = n-cache_limit
        for k ∈ length(bladeindex_extra)+1:N
            push!(bladeindex_extra,Int[])
        end
        @inbounds isempty(bladeindex_extra[N]) && (bladeindex_extra[N]=-ones(Int,1<<n-1))
        @inbounds signbit(bladeindex_extra[N][s]) && (bladeindex_extra[N][s]=bladeindex_calc(s,n))
        @inbounds bladeindex_extra[N][s]
    else
        j = length(bladeindex_cache)+1
        for k ∈ j:min(n,cache_limit)
            push!(bladeindex_cache,[bladeindex_calc(d,k) for d ∈ 1:1<<k-1])
            GC.gc()
        end
        @inbounds bladeindex_cache[n][s]
    end
end

@inline basisindex_calc(d,k) = binomsum(k,count_ones(UInt(d)))+bladeindex(k,UInt(d))
const basisindex_cache = Vector{Int}[]
const basisindex_extra = Vector{Int}[]
@pure function basisindex(n::Int,s::Bits)::Int
    if s == 0
        1
    elseif n>(DirectSum.index_limit)
        basisindex_calc(s,n)
    elseif n>cache_limit
        N = n-cache_limit
        for k ∈ length(basisindex_extra)+1:N
            push!(basisindex_extra,Int[])
        end
        @inbounds isempty(basisindex_extra[N]) && (basisindex_extra[N]=-ones(Int,1<<n-1))
        @inbounds signbit(basisindex_extra[N][s]) && (basisindex_extra[N][s]=basisindex_calc(s,n))
        @inbounds basisindex_extra[N][s]
    else
        j = length(basisindex_cache)+1
        for k ∈ j:min(n,cache_limit)
            push!(basisindex_cache,[basisindex_calc(d,k) for d ∈ 1:1<<k-1])
            GC.gc()
        end
        @inbounds basisindex_cache[n][s]
    end
end

bladeindex(cache_limit,one(Bits))
basisindex(cache_limit,one(Bits))

intlog(M::Integer) = Int(log2(M))

Base.@pure promote_type(t...) = Base.promote_type(t...)

## constructor

@inline assign_expr!(e,x::Vector{Any},v::Symbol,expr) = v ∈ e && push!(x,Expr(:(=),v,expr))

@pure function insert_expr(e,vec=:mvec,T=:(valuetype(a)),S=:(valuetype(b)),L=:(2^N))
    x = Any[] # Any[:(sigcheck(sig(a),sig(b)))]
    assign_expr!(e,x,:N,:(ndims(V)))
    assign_expr!(e,x,:M,:(Int(N/2)))
    assign_expr!(e,x,:t,vec≠:mvec ? :Any : :(promote_type($T,$S)))
    assign_expr!(e,x,:out,:(zeros($vec(N,t))))
    assign_expr!(e,x,:mv,:(MBlade{V,0,getbasis(V,0),t}(0)))
    assign_expr!(e,x,:r,:(binomsum(N,G)))
    assign_expr!(e,x,:bng,:(binomial(N,G)))
    assign_expr!(e,x,:bnl,:(binomial(N,L)))
    assign_expr!(e,x,:ib,:(indexbasis(N,G)))
    assign_expr!(e,x,:bs,:(binomsum_set(N)))
    assign_expr!(e,x,:bn,:(binomial_set(N)))
    assign_expr!(e,x,:df,:(dualform(V)))
    assign_expr!(e,x,:di,:(dualindex(V)))
    assign_expr!(e,x,:D,:(diffmode(V)))
    return x
end

@eval begin
    const indexbasis_cache = Vector{Vector{$Bits}}[]
    const indexbasis_extra = Vector{Vector{$Bits}}[]
    @pure function indexbasis(n::Int,g::Int)::Vector{$Bits}
        if n>sparse_limit
            N = n-sparse_limit
            for k ∈ length(indexbasis_extra)+1:N
                push!(indexbasis_extra,Vector{$Bits}[])
            end
            @inbounds for k ∈ length(indexbasis_extra[N])+1:g
                @inbounds push!(indexbasis_extra[N],$Bits[])
            end
            @inbounds if isempty(indexbasis_extra[N][g])
                @inbounds indexbasis_extra[N][g] = [bit2int(indexbits(n,combo(n,g)[q])) for q ∈ 1:binomial(n,g)]
            end
            @inbounds indexbasis_extra[N][g]
        else
            for k ∈ length(indexbasis_cache)+1:n
                push!(indexbasis_cache,[[bit2int(indexbits(k,@inbounds(combo(k,G)[q]))) for q ∈ 1:binomial(k,G)] for G ∈ 1:k])
            end
            @inbounds g>0 ? indexbasis_cache[n][g] : [zero($Bits)]
        end
    end
end

indexbasis(Int((sparse_limit+cache_limit)/2),1)

@pure indexbasis_set(N) = SVector(((N≠0 && N<sparse_limit) ? @inbounds(indexbasis_cache[N]) : Vector{Bits}[indexbasis(N,g) for g ∈ 0:N])...)

@pure indexbasis(N) = vcat(indexbasis(N,0),indexbasis_set(N)...)

## Grade{G}

struct Grade{G}
    @pure Grade{G}() where G = new{G}()
end

## Dimension{N}

struct Dimension{N}
    @pure Dimension{N}() where N = new{N}()
end
