module Multivectors

# package code goes here

using Combinatorics, StaticArrays
using ComputedFieldTypes

import Base: show, getindex
export MultiBasis, MultiVector

struct MultiBasis
    n::BitArray
end

subscripts = Dict(
    '1' => '₁',
    '2' => '₂',
    '3' => '₃',
    '4' => '₄',
    '5' => '₅',
    '6' => '₆',
    '7' => '₇',
    '8' => '₈',
    '9' => '₉',
    '0' => '₀'
)

binomsum = ( () -> begin
        Y = Array{Int,1}[Int[1]]
        return (n::Int,i::Int) -> (begin
                j = length(Y)
                for k ∈ j+1:n
                    push!(Y,cumsum([binomial(n,k) for k ∈ 0:n]))
                end
                i ≠ 0 ? Y[n][i] : 0
            end)
    end)()

function show(io::IO, e::MultiBasis)
    print(io,"e")
    se = string(e.n)
    for i ∈ 1:length(se)
        print(io,subscripts[i])
        e.n > 3 && print(io,",")
    end
end

@computed mutable struct MultiVector{T,N}
    v::MVector{2^N,T}
end

function Base.getindex(m::MultiVector{T,N},i::Int) where {T,N}
    0 <= i <= N || throw(BoundsError(m, i))
    r = binomsum(N,i)
    return @view m.v[r+1:r+binomial(N,i)]
end
getindex(m::MultiVector,i::Int,j::Int) = m[i][j]

function Base.setindex!(m::MultiVector{T},k::T,i::Int,j::Int) where T
    m[i][j] = k
end

Base.firstindex(m::MultiVector) = 0
Base.lastindex(m::MultiVector{T,N}) where {T,N} = N

function show(io::IO, m::MultiVector{T,N}) where {T,N}
    print(io,m[0][1])
    ind = collect(1:N)
    for i ∈ 1:N
        b = m[i]
        set = combinations(ind,i) |> collect
        for k ∈ 1:length(set)
            print(io," + ",b[k],"e",[subscripts[s[1]] for s ∈ string.(set[k])]...)
        end
    end
end

end # module
