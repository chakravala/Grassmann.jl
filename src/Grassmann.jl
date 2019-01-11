module Grassmann

#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

using Combinatorics, StaticArrays, Requires
using ComputedFieldTypes

include("utilities.jl")
include("direct_sum.jl")
include("multivectors.jl")
include("algebra.jl")
include("forms.jl")

## Algebra{N}

@computed struct Algebra{V}
    b::SVector{2^ndims(V),Basis{V}}
    g::Dict{Symbol,Int}
end

@pure getindex(a::Algebra,i::Int) = getfield(a,:b)[i]
Base.firstindex(a::Algebra) = 1
Base.lastindex(a::Algebra{V}) where V = 2^ndims(V)
Base.length(a::Algebra{V}) where V = 2^ndims(V)

@noinline function lookup_basis(V::VectorSpace,v::Symbol)::Union{SValue,Basis}
    vs = string(v)
    vt = vs[1]≠'e'
    ef = split(vs,r"(e|f)")
    let W = V,fs=false
        C = dualtype(V)
        C≥0 && (W = C>0 ? V'⊕V : V⊕V')
        V2 = (vt ⊻ (vt ? C≠0 : C>0)) ? V' : V
        L = length(ef) > 2
        M = Int(ndims(W)/2)
        m = ((!L) && vt && (C<0)) ? M : 0
        (es,e,et) = indexjoin(Int[],[parse(Int,ef[2][k]) for k∈1:length(ef[2])].+m,C<0 ? V : V2)
        et && (return SValue{V}(0,getbasis(V,0)))
        d = if L
            (fs,f,ft) = indexjoin(Int[],[parse(Int,ef[3][k]) for k∈1:length(ef[3])].+M,W)
            ft && (return SValue{V}(0,getbasis(V,0)))
            ef = [e;f]
            Basis{W,length(ef),bit2int(basisbits(ndims(W),ef))}()
        else
            Basis{V2}(e)
        end
        return (es⊻fs) ? SValue(-1,d) : d
    end
end

@pure function Base.getproperty(a::Algebra{V},v::Symbol) where V
    if v ∈ (:b,:g)
        return getfield(a,v)
    elseif haskey(a.g,v)
        return a[getfield(a,:g)[v]]
    else
        return lookup_basis(V,v)
    end
end

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

@pure do2m(d,o,c) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))
@pure getalgebra(n::Int,d::Int,o::Int,s,c::Int=0) = getalgebra(n,do2m(d,o,c),s)
@pure getalgebra(n::Int,m::Int,s) = algebra_cache(n,m,UInt16(s))
@pure getalgebra(V::VectorSpace) = algebra_cache(ndims(V),do2m(Int(hasdual(V)),Int(hasorigin(V)),dualtype(V)),value(V))

@pure function Base.getproperty(λ::typeof(Λ),v::Symbol)
    v ∈ (:body,:var) && (return getfield(λ,v))
    V = string(v)
    N = parse(Int,V[2])
    C = V[1]∉('D','C') ? 0 : 1
    length(V) < 5 && (V *= join(zeros(Int,5-length(V))))
    getalgebra(N,do2m(parse(Int,V[3]),parse(Int,V[4]),C),flip_sig(N,UInt16(parse(Int,V[5:end]))))
end

const algebra_cache = ( () -> begin
        Y = Vector{Dict{UInt16,Λ}}[]
        V=VectorSpace(0)
        Λ0=Λ{V}(SVector{1,Basis{V}}(Basis{V,0,0x0000}()),Dict{Symbol,Int}(:e=>1))
        return (n::Int,m::Int,s::UInt16) -> (begin
                n==0 && (return Λ0)
                for N ∈ length(Y)+1:n
                    push!(Y,[Dict{Int,Λ}() for k∈1:12])
                end
                if !haskey(Y[n][m+1],s)
                    D = Int(m ∈ (1,3,5,7,9,11))
                    O = Int(m ∈ (2,3,6,7,10,11))
                    C = m ∈ 8:11 ? -1 : Int(m ∈ (4,5,6,7))
                    c = C>0 ? "'" : C<0 ? "*" : ""
                    @info("Precomputing $(2^n)×Basis{VectorSpace{$n,$D,$O,$(Int(s))}$c,...}")
                    push!(Y[n][m+1],s=>collect(VectorSpace{n,D,O,s,C}()))
                end
                Y[n][m+1][s]
            end)
    end)()

@pure getbasis(V::VectorSpace,b) = getalgebra(V).b[basisindex(ndims(V),UInt16(b))]
@pure getbasis(V::VectorSpace,v::Symbol) = getproperty(getalgebra(V),v)

function __init__()
    @require Reduce="93e0c654-6965-5f22-aba9-9c1ae6b3c259" include("symbolic.jl")
end

end # module
