module Grassmann

#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

using Combinatorics, StaticArrays #, Requires
using ComputedFieldTypes

include("utilities.jl")
include("multivectors.jl")
include("algebra.jl")

## Algebra{N}

@computed struct Algebra{V}
    b::SVector{2^ndims(V),Basis{V}}
    g::Dict{Symbol,Int}
end

@pure getindex(a::Algebra,i::Int) = a.b[i]
Base.firstindex(a::Algebra) = 1
Base.lastindex(a::Algebra{V}) where V = 2^ndims(V)
Base.length(a::Algebra{V}) where V = 2^ndims(V)
@pure Base.getproperty(a::Algebra,v::Symbol) = v ∈ (:b,:g) ? getfield(a,v) : a[a.g[v]]

@pure function Base.collect(s::VectorSpace)
    g = Dict{Symbol,Int}()
    basis,sym = generate(s,:e)
    for i ∈ 1:2^ndims(s)
        push!(g,sym[i]=>i)
    end
    return Algebra{s}(basis,g)
end

Algebra(s::VectorSpace) = getalgebra(s)
Algebra(n::Int,d::Int=0,o::Int=0,s=0x0000) = getalgebra(n,d,o,s)
Algebra(s::String) = getalgebra(VectorSpace(s))
Algebra(s::String,v::Symbol) = getbasis(VectorSpace(s),v)

function show(io::IO,a::Algebra{V}) where V
    N = ndims(V)
    print(io,"Grassmann.Algebra{$V,$(2^N)}(")
    for i ∈ 1:2^N-1
        print(io,a[i],", ")
    end
    print(io,a[end],")")
end

export Λ, @Λ_str, getalgebra, getbasis

Λ = Algebra

macro Λ_str(str)
    Algebra(str)
end

@pure do2m(d,o) = (1<<(d-1))+(1<<(2*o-1))
@pure getalgebra(n::Int,d::Int,o::Int,s) = getalgebra(n,do2m(d,o),s)
@pure getalgebra(n::Int,m::Int,s) = algebra_cache(n,m,UInt16(s))
@pure getalgebra(V::VectorSpace) = algebra_cache(ndims(V),do2m(Int(hasdual(V)),Int(hasorigin(V))),value(V))

@pure function Base.getproperty(λ::typeof(Λ),v::Symbol)
    v ∈ (:body,:var) && (return getfield(λ,v))
    V = string(v)
    length(V) < 5 && (V *= join(zeros(Int,5-length(V))))
    getalgebra(parse(Int,V[2]),do2m(parse(Int,V[3]),parse(Int,V[4])),parse(Int,V[5:end]))
end

const algebra_cache = ( () -> begin
        Y = Vector{Dict{UInt16,Λ}}[]
        return (n::Int,m::Int,s::UInt16) -> (begin
                for N ∈ length(Y)+1:n
                    push!(Y,[Dict{Int,Λ}() for k∈1:4])
                end
                if !haskey(Y[n][m+1],s)
                    D = Int(m ∈ (1,3))
                    O = Int(m ∈ (2,3))
                    @info("Precomputing $(2^n)×Basis{VectorSpace{$n,$D,$O,$(Int(s))},...}")
                    push!(Y[n][m+1],s=>collect(VectorSpace{n,D,O,s}()))
                end
                Y[n][m+1][s]
            end)
    end)()

@pure getbasis(V::VectorSpace,b) = getalgebra(V).b[basisindex(ndims(V),UInt16(b))]
@pure getbasis(V::VectorSpace,v::Symbol) = getproperty(getalgebra(V),v)

#=function __init__()
    @require Reduce="93e0c654-6965-5f22-aba9-9c1ae6b3c259" include("symbolic.jl")
end=#

end # module
