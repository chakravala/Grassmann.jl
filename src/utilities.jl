
#   This file is part of Multivectors.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

subscripts = Dict(
    1 => '₁',
    2 => '₂',
    3 => '₃',
    4 => '₄',
    5 => '₅',
    6 => '₆',
    7 => '₇',
    8 => '₈',
    9 => '₉',
    0 => '₀'
)

const binomsum = ( () -> begin
        Y = Array{Int,1}[Int[1]]
        return (n::Int,i::Int) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    push!(Y,cumsum([binomial(k,q) for q ∈ 0:k]))
                end
                i ≠ 0 ? Y[n][i] : 0
            end)
    end)()

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

const basisindex = ( () -> begin
        Y = Array{Dict{Array{Int,1},Int},1}[]
        return (n::Int,i::Array{Int,1}) -> (begin
                j = length(Y)
                g = length(i)
                for k ∈ j+1:n
                    push!(Y,[Dict{Array{Int,1},Int}() for k ∈ 1:k])
                end
                g>0 && !haskey(Y[n][g],i) && 
                    push!(Y[n][g],i=>findall(x->x==i,combo(n,g))[1])
                g>0 ? Y[n][g][i] : 1
            end)
    end)()

function intlog(M::Integer)
    lM = log2(M)
    try; Int(lM)
    catch; lM end
end

