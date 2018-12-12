
#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

import Base: +, *

## geometric product

function *(a::Basis{N},b::Basis{N}) where N
    a.s ≠ b.s && throw(error("$(a.s) ≠ $(b.s)"))
    (s,c,t) = indexjoin(basisindices(a),basisindices(b),a.s)
    t && (return SValue{N}(0,Basis{N,0}(a.s)))
    d = Basis{N}(a.s,c)
    return s ? SValue{N}(-1,d) : d
end

function indexjoin(a::Vector{Int},b::Vector{Int},s::Signature)
    ind = [a;b]
    k = 1
    t = false
    while k < length(ind)
        if ind[k] == ind[k+1]
            ind[k] == 1 && s[end-1] && (return t, ind, true)
            s[ind[k]] && (t = !t)
            deleteat!(ind,[k,k+1])
        elseif ind[k] > ind[k+1]
            ind[k:k+1] = ind[k+1:-1:k]
            t = !t
            k ≠ 1 && (k -= 1)
        else
            k += 1
        end
    end
    return t, ind, false
end

*(a::Number,b::Basis{N}) where N = SValue{N}(a,b)
*(a::Basis{N},b::Number) where N = SValue{N}(b,a)

for Value ∈ MSV
    @eval begin
        *(a::Number,b::$Value{N}) where N = SValue{N}(a*b.v,b.b)
        *(a::$Value{N},b::Number) where N = SValue{N}(a.v*b,a.b)
        *(a::$Value{N},b::Basis{N}) where N = SValue{N}(a.v,a.b*b)
        *(a::Basis{N},b::$Value{N}) where N = SValue{N}(b.v,a*b.b)
    end
end
for (A,B) ∈ [(A,B) for A ∈ MSV, B ∈ MSV]
    @eval *(a::$A{N},b::$B{N}) where N = SValue{N}(a.v*b.v,a.b*b.b)
end

## term addition

function +(a::AbstractTerm{N,A},b::AbstractTerm{N,B}) where {N,A,B}
    if sig(a) ≠ sig(b)
        throw(error("$(sig(a)) ≠ $(sig(b))"))
    elseif basis(a) == basis(b)
        return SValue{N,A}(value(a)+value(b),a)
    elseif A == B
        out = MBlade{Int,N,A}(sig(a),zeros(Int,binomial(N,A)))
        out.v[basisindex(N,findall(basis(a).b))] = value(a)
        out.v[basisindex(N,findall(basis(b).b))] = value(b)
        return out
    else
        return MultiGrade{N}(a,b)
    end
end

for Blade ∈ MSB
    @eval begin
        function +(a::$Blade{T,N,G},b::$Blade{S,N,G}) where {T,N,G,S}
            sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
            return $Blade{promote_type(T,S),N,G}(a.v+b.v)
        end
        function +(a::$Blade{T,N,G},b::AbstractTerm{N,G}) where {T,N,G}
            sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
            return MBlade{promote_type(T,valuetype(b)),N,G}(a.v+b.v)
        end
    end
end



