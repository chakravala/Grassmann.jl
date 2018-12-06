module Multivectors

# package code goes here

using Combinatorics, StaticArrays

import Base: show, getindex
export MultiBasis, MultiVector

struct MultiBasis
    n::UInt8
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

function show(io::IO, e::MultiBasis)
    print(io,"e",[subscripts[i] for i in string(e.n)]...,' ')
end

mutable struct MultiVector{T}
    n::UInt8
    v::Vector{T}
end

function Base.getindex(m::MultiVector,i::Int) 
    0 <= i <= m.n || throw(BoundsError(m, i))
    r = sum([binomial(Int(m.n),k) for k ∈ 0:i-1])
    return @view m.v[r+1:r+binomial(Int(m.n),i)]
end
getindex(m::MultiVector,i::Int,j::Int) = m[i][j]

function Base.setindex!(m::MultiVector{T},k::T,i::Int,j::Int) where T
    m[i][j] = k
end

Base.firstindex(m::MultiVector) = 0
Base.lastindex(m::MultiVector) = Int(m.n)

function show(io::IO, m::MultiVector{T}) where T
    print(io,m[0][1])
    ind = collect(1:Int(m.n))
    for i ∈ 1:m.n
        b = m[i]
        set = combinations(ind,i) |> collect
        for k ∈ 1:length(set)
            print(io," + ",b[k],"e",[subscripts[s[1]] for s ∈ string.(set[k])]...)
        end
    end
end

end # module
