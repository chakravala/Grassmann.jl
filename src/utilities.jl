
#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

import Base: @pure

@pure binomial_set(N) = SVector(Int[binomial(N,g) for g ∈ 0:N]...)
@pure binomial(N,G) = Base.binomial(N,G)
@pure mvec(N,G,t) = MVector{binomial(N,G),t}
@pure mvec(N,t) = MVector{2^N,t}

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
    9 => '₉'
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
    9 => '⁹'
)

const binomsum_cache = ( () -> begin
        Y = Array{Int,1}[Int[1]]
        return (n::Int,i::Int) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    push!(Y,cumsum([binomial(k,q) for q ∈ 0:k]))
                end
                i ≠ 0 ? Y[n][i] : 0
            end)
    end)()

Base.@pure binomsum(n::Int,i::Int) = binomsum_cache(n,i)

const combo = ( () -> begin
        Y = Array{Array{Array{Int,1},1},1}[]
        return (n::Int,g::Int) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    z = 1:k
                    push!(Y,[collect(combinations(z,q)) for q ∈ z])
                end
                g ≠ 0 ? Y[n][g] : [Int[]]
            end)
    end)()

#=const basisindex = ( () -> begin
        Y = Array{Dict{String,Int},1}[]
        return (n::Int,i::Array{Int,1}) -> (begin
                j = length(Y)
                g = length(i)
                s = string(i...)
                for k ∈ j+1:n
                    push!(Y,[Dict{String,Int}() for k ∈ 1:k])
                end
                g>0 && !haskey(Y[n][g],s) && 
                    push!(Y[n][g],s=>findall(x->x==i,combo(n,g))[1])
                g>0 ? Y[n][g][s] : 1
            end)
end)()=#

const basisindexb = ( () -> begin
        Y = Vector{Int}[]
        return (n::Int,s::UInt16) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    y = Array{Int,1}(undef,2^k-1)
                    for d ∈ 1:2^k-1
                        H = findall(digits(d,base=2).==1)
                        y[d] = findall(x->x==H,combo(k,length(H)))[1]
                    end
                    push!(Y,y)
                end
                s>0 ? Y[n][s] : 1
            end)
    end)()

const basisindex = ( () -> begin
        Y = Vector{Int}[]
        return (n::Int,s::UInt16) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    y = Array{Int,1}(undef,2^k-1)
                    for d ∈ 1:2^k-1
                        H = findall(digits(d,base=2).==1)
                        lh = length(H)
                        y[d] = binomsum(k,lh)+findall(x->x==H,combo(k,lh))[1]
                    end
                    push!(Y,y)
                end
                s>0 ? Y[n][s] : 1
            end)
    end)()

function intlog(M::Integer)
    lM = log2(M)
    try; Int(lM)
    catch; lM end
end

bit2int(b::BitArray{1}) = parse(UInt16,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

Base.@pure promote_type(t...) = Base.promote_type(t...)
