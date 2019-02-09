
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: @pure, print, show, getindex, setindex!, promote_rule, ==, convert, ndims
#import DirectSum: bit2int, doc2m, Bits, pre, alphanumv, alphanumw, vsn, subs, sups

@pure binomial_set(N) = SVector(Int[binomial(N,g) for g ∈ 0:N]...)
@pure binomial(N,G) = Base.binomial(N,G)
@pure mvec(N,G,t) = MVector{binomial(N,G),t}
@pure mvec(N,t) = MVector{2^N,t}
@pure svec(N,G,t) = SizedArray{Tuple{binomial(N,G)},t,1,1}
@pure svec(N,t) = SizedArray{Tuple{1<<N},t,1,1}

const algebra_limit = 8
const sparse_limit = 22
const cache_limit = 12

const binomsum_cache = [[1]]
@pure function binomsum(n::Int, i::Int)
    n>sparse_limit && (return cumsum([binomial(n,q) for q=0:i])[end])
    for k=length(binomsum_cache)+1:n
        push!(binomsum_cache, cumsum([binomial(k,q) for q=0:k]))
    end
    i ≠ 0 ? binomsum_cache[n][i] : 0
end
@pure function binomsum_set(n::Int)
    n>sparse_limit && (return cumsum([binomial(n,q) for q=0:n]))
    for k=length(binomsum_cache)+1:n
        push!(binomsum_cache, cumsum([binomial(k,q) for q=0:k]))
    end
    [0;binomsum_cache[n]]
end

const combo_cache = Array{Array{Array{Int,1},1},1}[]
function combo(n::Int,g::Int)
    n>sparse_limit && (return collect(combinations(1:n,g)))
    for k ∈ length(combo_cache)+1:min(n,sparse_limit)
        z = 1:k
        push!(combo_cache,[collect(combinations(z,q)) for q ∈ z])
    end
    g≠0 ? combo_cache[n][g] : [Int[]]
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

@inline function bladeindex_calc(d,k)
    H = findall(digits(d,base=2).==1)
    findall(x->x==H,combo(k,length(H)))[1]
end
const bladeindex_cache = Vector{Int}[]
@pure function bladeindex(n::Int,s::Bits)
    n>cache_limit && (return bladeindex_calc(s,n))
    j = length(bladeindex_cache)+1
    for k ∈ j:min(n,cache_limit)
        y = Array{Int,1}(undef,1<<k-1)
        for d ∈ 1:1<<k-1
            y[d] = bladeindex_calc(d,k)
        end
        push!(bladeindex_cache,y)
        GC.gc()
    end
    s>0 ? bladeindex_cache[n][s] : 1
end

@inline function basisindex_calc(d,k)
    H = findall(digits(d,base=2).==1)
    lh = length(H)
    binomsum(k,lh)+findall(x->x==H,combo(k,lh))[1]
end
const basisindex_cache = Vector{Int}[]
@pure function basisindex(n::Int,s::Bits)
    n>cache_limit && (return basisindex_calc(s,n))
    j = length(basisindex_cache)+1
    for k ∈ j:min(n,cache_limit)
        y = Array{Int,1}(undef,1<<k-1)
        for d ∈ 1:1<<k-1
            y[d] = basisindex_calc(d,k)
        end
        push!(basisindex_cache,y)
        GC.gc()
    end
    s>0 ? basisindex_cache[n][s] : 1
end

bladeindex(cache_limit,one(Bits))
basisindex(cache_limit,one(Bits))

intlog(M::Integer) = Int(log2(M))

Base.@pure promote_type(t...) = Base.promote_type(t...)
