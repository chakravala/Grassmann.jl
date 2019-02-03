
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: @pure, print, show, getindex, setindex!, promote_rule, ==, convert, ndims

@pure binomial_set(N) = SVector(Int[binomial(N,g) for g ∈ 0:N]...)
@pure binomial(N,G) = Base.binomial(N,G)
@pure mvec(N,G,t) = MVector{binomial(N,G),t}
@pure mvec(N,t) = MVector{2^N,t}
@pure svec(N,G,t) = SizedArray{Tuple{binomial(N,G)},t,1,1}
@pure svec(N,t) = SizedArray{Tuple{1<<N},t,1,1}

const pre = ("v","w")
const vsn = (:V,:VV,:W)
const alphanumv = "123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
const alphanumw = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

const subscripts = Dict{Int,Char}(
   -1 => 'ϵ',
    0 => 'o',
    1 => '₁',
    2 => '₂',
    3 => '₃',
    4 => '₄',
    5 => '₅',
    6 => '₆',
    7 => '₇',
    8 => '₈',
    9 => '₉',
    [j=>alphanumv[j] for j ∈ 10:35]...
)

const super = Dict{Int,Char}(
   -1 => 'ϵ',
    0 => 'o',
    1 => '¹',
    2 => '²',
    3 => '³',
    4 => '⁴',
    5 => '⁵',
    6 => '⁶',
    7 => '⁷',
    8 => '⁸',
    9 => '⁹',
    [j=>alphanumw[j] for j ∈ 10:35]...
)

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

const Bits = UInt

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

bit2int(b::BitArray{1}) = parse(Bits,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

Base.@pure promote_type(t...) = Base.promote_type(t...)
