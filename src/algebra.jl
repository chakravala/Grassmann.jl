
#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

import Base: +, *

## geometric product

function *(a::Basis{N},b::Basis{N}) where N
    a.s ≠ b.s && throw(error("$(a.s) ≠ $(b.s)"))
    (s,c) = indexjoin(basisindices(a),basisindices(b),a.s)
    d = Basis{N}(a.s,c)
    return s ? SValue{N}(-1,d) : d
end

function indexjoin(a::Vector{Int},b::Vector{Int},s::Signature)
    ind = [a;b]
    k = 1
    t = false
    while k < length(ind)
        if ind[k] == ind[k+1]
            !s.b[ind[k]] && (t = !t)
            deleteat!(ind,[k,k+1])
        elseif ind[k] > ind[k+1]
            ind[k:k+1] = ind[k+1:-1:k]
            t = !t
            k ≠ 1 && (k -= 1)
        else
            k += 1
        end
    end
    return t, ind
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

function +(a::Basis{N,A},b::Basis{N,B}) where {N,A,B}
    if a.s ≠ b.s
        throw(error("$(a.s) ≠ $(b.s)"))
    elseif a == b
        return SValue{N,A}(2,a)
    elseif A == B
        return nothing
    else
        return MultiGrade{N}(a,b)
    end
end



