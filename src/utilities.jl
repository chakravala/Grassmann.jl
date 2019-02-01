
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: @pure, print, show, getindex, setindex!, promote_rule, ==, convert, ndims

@pure binomial_set(N) = SVector(Int[binomial(N,g) for g ∈ 0:N]...)
@pure binomial(N,G) = Base.binomial(N,G)
@pure mvec(N,G,t) = MVector{binomial(N,G),t}
@pure mvec(N,t) = MVector{2^N,t}
@pure svec(N,G,t) = SizedArray{Tuple{binomial(N,G)},t,1,1}
@pure svec(N,t) = SizedArray{Tuple{2^N},t,1,1}

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
    10 => 'a',
    11 => 'b',
    12 => 'c',
    13 => 'd',
    14 => 'e',
    15 => 'f',
    16 => 'g',
    17 => 'h',
    18 => 'i',
    19 => 'j',
    20 => 'k',
    21 => 'l',
    22 => 'm',
    23 => 'n',
    24 => 'o',
    25 => 'p',
    26 => 'q',
    27 => 'r',
    28 => 's',
    29 => 't',
    30 => 'u',
    31 => 'v',
    32 => 'w',
    33 => 'x',
    34 => 'y',
    35 => 'z'
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
    10 => 'A',
    11 => 'B',
    12 => 'C',
    13 => 'D',
    14 => 'E',
    15 => 'F',
    16 => 'G',
    17 => 'H',
    18 => 'I',
    19 => 'J',
    20 => 'K',
    21 => 'L',
    22 => 'M',
    23 => 'N',
    24 => 'O',
    25 => 'P',
    26 => 'Q',
    27 => 'R',
    28 => 'S',
    29 => 'T',
    30 => 'U',
    31 => 'V',
    32 => 'W',
    33 => 'X',
    34 => 'Y',
    35 => 'Z'
)

const binomsum_cache = [[1]]
@pure function binomsum(n::Int, i::Int)
    for k=length(binomsum_cache)+1:n
        push!(binomsum_cache, cumsum([binomial(k,q) for q=0:k]))
    end
    i ≠ 0 ? binomsum_cache[n][i] : 0
end

const combo_limit = 22
const combo_cache = Array{Array{Array{Int,1},1},1}[]
function combo(n::Int,g::Int)
    for k ∈ length(combo_cache)+1:min(n,combo_limit)
        z = 1:k
        push!(combo_cache,[collect(combinations(z,q)) for q ∈ z])
    end
    g≠0 ? (n>combo_limit ? collect(combinations(n,g)) : combo_cache[n][g]) : [Int[]]
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

const cache_limit = 12

@inline function basisindexb_calc(d,k)
    H = findall(digits(d,base=2).==1)
    findall(x->x==H,combo(k,length(H)))[1]
end
const basisindexb_cache = Vector{Int}[]
@pure function basisindexb(n::Int,s::UInt16)
    j = length(basisindexb_cache)+1
    for k ∈ j:min(n,cache_limit)
        y = Array{Int,1}(undef,2^k-1)
        for d ∈ 1:2^k-1
            y[d] = basisindexb_calc(d,k)
        end
        push!(basisindexb_cache,y)
        GC.gc()
    end
    s>0 ? (n>cache_limit ? basisindexb_calc(s,n) : basisindexb_cache[n][s]) : 1
end

@inline function basisindex_calc(d,k)
    H = findall(digits(d,base=2).==1)
    lh = length(H)
    binomsum(k,lh)+findall(x->x==H,combo(k,lh))[1]
end
const basisindex_cache = Vector{Int}[]
@pure function basisindex(n::Int,s::UInt16)
    j = length(basisindex_cache)+1
    for k ∈ j:min(n,cache_limit)
        y = Array{Int,1}(undef,2^k-1)
        for d ∈ 1:2^k-1
            y[d] = basisindex_calc(d,k)
        end
        push!(basisindex_cache,y)
        GC.gc()
    end
    s>0 ? (n>cache_limit ? basisindex_calc(s,n) : basisindex_cache[n][s]) : 1
end

basisindexb(cache_limit,0x0001)
basisindex(cache_limit,0x0001)

intlog(M::Integer) = Int(log2(M))

bit2int(b::BitArray{1}) = parse(UInt16,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

Base.@pure promote_type(t...) = Base.promote_type(t...)
