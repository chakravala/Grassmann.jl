
#   This file is part of Multivectors.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

import Base: +, *

function *(a::MultiBasis{N},b::MultiBasis{N}) where N
    a.s ≠ b.s && throw(error("$(a.s) ≠ $(b.s)"))
    (s,c) = indexjoin(basisindices(a),basisindices(b),a.s)
    d = MultiBasis{N}(a.s,c)
    return s ? d : MultiValue{N}(-1,d)
end

function indexjoin(a::Vector{Int},b::Vector{Int},s::Signature)
    ind = [a;b]
    k = 1
    t = true
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

*(a::Number,b::MultiBasis{N}) where N = MultiValue{N}(a,b)
*(a::MultiBasis{N},b::Number) where N = MultiValue{N}(b,a)
*(a::Number,b::MultiValue{N}) where N = MultiValue{N}(a*b.v,b)
*(a::MultiValue{N},b::Number) where N = MultiValue{N}(a.v*b,a)
*(a::MultiValue{N},b::MultiBasis{N}) where N = MultiValue{N}(a.v,a.b*b)
*(a::MultiBasis{N},b::MultiValue{N}) where N = MultiValue{N}(b.v,a*b.b)

function +(a::MultiBasis{N,A},b::MultiBasis{N,B}) where {N,A,B}
    if a.s ≠ b.s
        throw(error("$(a.s) ≠ $(b.s)"))
    elseif a == b
        return MultiValue{N,A}(2,a)
    elseif A == B
        return nothing
    else
        return MultiGrade{N}([a,b])
    end
end
