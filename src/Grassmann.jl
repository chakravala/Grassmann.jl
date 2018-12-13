module Grassmann

#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

using Combinatorics, StaticArrays
using ComputedFieldTypes

include("utilities.jl")
include("multivectors.jl")
include("algebra.jl")

## Algebra{N}

@computed struct Algebra{N}
    s::Signature{N}
    b::SVector{2^N,Basis{N}}
    g::Dict{Symbol,Int}
end

getindex(a::Algebra,i::Int) = a.b[i]
Base.firstindex(a::Algebra) = 1
Base.lastindex(a::Algebra{N}) where N = 2^N
Base.length(a::Algebra{N}) where N = 2^N
Base.getproperty(a::Algebra,v::Symbol) = v ∈ [:s,:b,:g] ? getfield(a,v) : a[a.g[v]]

function Algebra{N}(s::Signature{N}) where N
    g = Dict{Symbol,Int}()
    basis,sym = generate(s,:e)
    for i ∈ 1:2^N
        push!(g,sym[i]=>i)
    end
    return Algebra{N}(s,basis,g)
end

Algebra(s::Signature{N}) where N = Algebra{N}(s)
Algebra(s::String) = Algebra{length(s)}(Signature{length(s)}(s))
Algebra{N}(s::String) where N = Algebra{N}(Signature{N}(s))

function show(io::IO,a::Algebra{N}) where N
    print(io,"Grassmann.Algebra{$N,$(2^N)}(",a.s,", ")
    for i ∈ 1:2^N-1
        print(io,a[i],", ")
    end
    print(io,a[end],")")
end

end # module
