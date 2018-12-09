module Multivectors

# package code goes here

using Combinatorics, StaticArrays
using ComputedFieldTypes

import Base: show, getindex, promote_rule
export MultiBasis, MultiValue, MultiBlade, MultiVector

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

## MultiBasis{N}

struct MultiBasis{N,G}
    n::BitArray
end
MultiBasis{N}(n::BitArray) where N = MultiBasis{N,sum(Int.(n))}(n)

VTI = Union{Vector{<:Integer},Tuple,NTuple}
MultiBasis{N}(b::VTI) where N = MultiBasis{N}(basisbits(N,b))
MultiBasis{N}(b::Integer...) where N = MultiBasis{N}(basisbits(N,b))
MultiBasis{N,G}(b::VTI) where {N,G} = MultiBasis{N,G}(basisbits(N,b))
MultiBasis{N,G}(b::Integer...) where {N,G} = MultiBasis{N,G}(basisbits(N,b))

basisindices(b::MultiBasis) = findall(b.n)
function basisbits(d::Integer,b::VTI)
    out = falses(d)
    for k ∈ b
        out[k] = true
    end
    return out
end

show(io::IO, e::MultiBasis) = printbasis(io,basisindices(e))

function printbasis(io::IO,b::VTI)
    print(io,"e")
    for i ∈ b
        print(io,subscripts[string(i)[1]])
        #(N > 9) && (i ≠ b[end]) && print(io,",")
    end
end

## MultiValue{N}

struct MultiValue{N,G,T}
    v::T
    b::MultiBasis{N,G}
end

MultiValue{N}(v::T,b::MultiBasis{N,G}) where {N,T,G} = MultiValue{N,T,G}(v,b)
MultiValue{N,G}(v::T,b::MultiBasis{N,G}) where {N,T,G} = MultiValue{N,T,G}(v,b)
MultiValue{N}(v::T) where {N,T} = MultiValue{N,T,0}(v,MultiBasis{N}())


#Base.+(v::MultiValue{N}...) = 

## MultiBlades{T,N}

@computed mutable struct MultiBlade{T,N,G}
    v::MVector{binomial(N,G),T}
end

## MultiVector{T,N}

@computed mutable struct MultiVector{T,N}
    v::MVector{2^N,T}
end

MultiVector{T}(v::MVector{M,T}) where {T,M} = MultiVector{T,intlog(M)}(v)
MultiVector{T}(v::Vector{T}) where T = MultiVector{T,intlog(length(v))}(v)
MultiVector(v::MVector{M,T}) where {T,M} = MultiVector{T,intlog(M)}(v)
MultiVector(v::Vector{T}) where T = MultiVector{T,intlog(length(v))}(v)
MultiVector(v::T...) where T = MultiVector{T,intlog(length(v))}(v)

function intlog(M::Integer)
    lM = log2(M)
    try; Int(lM)
    catch; lM end
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
            print(io," + ",b[k])
            printbasis(io,set[k])
        end
    end
end




end # module
