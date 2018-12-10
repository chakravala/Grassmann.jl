
#   This file is part of Multivectors.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

function Base.:*(a::MultiBasis{N},b::MultiBasis{N}) where N
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
