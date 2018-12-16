
#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

import Base: +, -, *

## geometric product

function *(a::Basis{N},b::Basis{N}) where N
    a.s ≠ b.s && throw(error("$(a.s) ≠ $(b.s)"))
    (s,c,t) = indexjoin(basisindices(a),basisindices(b),a.s)
    t && (return SValue{N}(0,Basis{N,0}(a.s)))
    d = Basis{N}(a.s,c)
    return s ? SValue{N}(-1,d) : d
    #d = Basis{N}(a.s,indexjoin(a.b,b.b,a.s))
    #return parity(a,b) ? SValue{N}(-1,d) : d
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

@inline function indexjoin(a::BitArray{1},b::BitArray{1},s::Signature{N}) where N
    (s[end-1] && a[1] && b[1]) ? (return BitArray{1}(falses(N))) : a .⊻ b
end

@inline function parity(a::Basis, b::Basis)
    B = [0,Int.(b.b)...]
    c = a .& b
    isodd(sum([Int.(a.b)...,0] .* cumsum!(B,B))+sum((c .⊻ a.s[1:N]) .& c))
end

*(a::Number,b::Basis{N}) where N = SValue{N}(a,b)
*(a::Basis{N},b::Number) where N = SValue{N}(b,a)

for Value ∈ MSV
    @eval begin
        *(a::Number,b::$Value{N,G}) where {N,G} = SValue{N,G}(a*b.v,b.b)
        *(a::$Value{N,G},b::Number) where {N,G} = SValue{N,G}(a.v*b,a.b)
        *(a::$Value{N},b::Basis{N}) where N = SValue{N}(a.v,a.b*b)
        *(a::Basis{N},b::$Value{N}) where N = SValue{N}(b.v,a*b.b)
    end
end
for (A,B) ∈ [(A,B) for A ∈ MSV, B ∈ MSV]
    @eval *(a::$A{N},b::$B{N}) where N = SValue{N}(a.v*b.v,a.b*b.b)
end
for Blade ∈ MSB
    @eval begin
        *(a::Number,b::$Blade{T,N,G}) where {T,N,G} = SBlade{T,N,G}(a.*b.v,b.b)
        *(a::$Blade{T,N,G},b::Number) where {T,N,G} = SBlade{T,N,G}(a.v.*b,a.b)
        function *(a::$Blade{T,N,G},b::Basis{N}) where {T,N,G}
            a.s ≠ b.s && throw(error("$(a.s) ≠ $(b.s)"))
            (s,c,t) = indexjoin(basisindices(s,G,i),basisindices(b),a.s)
            t && (return SValue{N}(0,Basis{N,0}(a.s)))
            d = Basis{N}(a.s,c)
            return s ? SValue{N}(-1,d) : d
        end
        #*(a::Basis{N},b::$Blade{T,N}) where {T,N} = SBlade{T,N}(b.v,a*b.b)
    end
end
for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
    #@eval *(a::$A{T,N},b::$B{T,N}) where N = SBlade{T,N}(a.v*b.v,a.b*b.b)
end
*(a::Number,b::MultiVector{T,N,G}) where {T,N,G} = MultiVector{T,N,G}(a.*b.v,b.b)
*(a::MultiVector{T,N,G},b::Number) where {T,N,G} = MultiVector{T,N,G}(a.v.*b,a.b)
#*(a::MultiVector{T,N},b::Basis{N}) where {T,N} = MultiVector{T,N}(a.v,a.b*b)
#*(a::Basis{N},b::MultiVector{T,N}) where {T,N} = MultiVector{T,N}(b.v,a*b.b)
#*(a::MultiVector{T,N},b::MultiVector{T,N}) where {T,N} = MultiVector{T,N}(a.v*b.v,a.b*b.b)
*(a::Number,b::MultiGrade{N}) where N = MultiGrade{N}(a.s,a.*b.v)
*(a::MultiGrade{N},b::Number) where N = MultiGrade{N}(a.s,a.v.*b)
#*(a::MultiGrade{N},b::Basis{N}) where N = MultiGrade{N}(a.v,a.b*b)
#*(a::Basis{N},b::MultiGrade{N}) where N = MultiGrade{N}(b.v,a*b.b)
#*(a::MultiGrade{N},b::MultiGrade{N}) where N = MultiGrade{N}(a.v*b.v,a.b*b.b)

## term addition

for (op,eop) ∈ [(:+,:(+=)),(:-,:(-=))]
    @eval begin
        function $op(a::AbstractTerm{N,A},b::AbstractTerm{N,B}) where {N,A,B}
            if sig(a) ≠ sig(b)
                throw(error("$(sig(a)) ≠ $(sig(b))"))
            elseif basis(a) == basis(b)
                return SValue{N,A}($op(value(a),value(b)),a)
            elseif A == B
                T = promote_type(valuetype(a),valuetype(b))
                out = MBlade{T,N,A}(sig(a),zeros(T,binomial(N,A)))
                out.v[basisindex(N,findall(basis(a).b))] = value(a,T)
                out.v[basisindex(N,findall(basis(b).b))] = $op(value(b,T))
                return out
            else
                warn("sparse MultiGrade{N} objects not properly handled yet")
                return MultiGrade{N}(a,b)
            end
        end

        function $op(a::A,b::MultiVector{T,N}) where A<:AbstractTerm{N,G} where {T,N,G}
            sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
            t = promote_type(T,valuetype(a))
            out = MultiVector{t,N}(sig(b),$op(value(b,Vector{t})))
            out.v[binomsum(N,G)+basisindex(N,findall(basis(a).b))] += value(b,Vector{t})
            return out
        end
        function $op(a::MultiVector{T,N},b::B) where B<:AbstractTerm{N,G} where {T,N,G}
            sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
            t = promote_type(T,valuetype(b))
            out = MultiVector{t,N,G}(sig(a),value(a,Vector{t}))
            $(Expr(eop,:(out.v[binomsum(N,G)+basisindex(N,findall(basis(b).b))]),:(value(b,t))))
            return out
        end
    end

    for Blade ∈ MSB
        @eval begin
            function $op(a::$Blade{T,N,G},b::$Blade{S,N,G}) where {T,N,G,S}
                sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
                return $Blade{promote_type(T,S),N,G}($op(a.v,b.v))
            end
            function $op(a::$Blade{T,N,G},b::B) where B<:AbstractTerm{N,G} where {T,N,G}
                sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
                t = promote_type(T,valuetype(b))
                out = MBlade{t,N,G}(sig(a),value(a,Vector{t}))
                $(Expr(eop,:(out.v[basisindex(N,findall(basis(b).b))]),:(value(b,t))))
                return out
            end
            function $op(a::A,b::$Blade{T,N,G}) where A<:AbstractTerm{N,G} where {T,N,G}
                sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
                t = promote_type(T,valuetype(a))
                out = MBlade{t,N,G}(sig(b),$op(value(b,Vector{t})))
                out.v[basisindex(N,findall(basis(a).b))] += value(a,t)
                return out
            end
            function $op(a::$Blade{T,N,G},b::B) where B<:AbstractTerm{N,L} where {T,N,G,L}
                sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
                t = promote_type(T,valuetype(b))
                r = binomsum(N,G)
                out = MultiVector{t,N}(sig(a),zeros(t,2^N))
                out.v[r+1:r+binomial(N,G)] = value(a,Vector{t})
                out.v[binomsum(N,L)+basisindex(N,findall(basis(b).b))] = $op(value(b,t))
                return out
            end
            function $op(a::A,b::$Blade{T,N,G}) where A<:AbstractTerm{N,L} where {T,N,G,L}
                sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
                t = promote_type(T,valuetype(a))
                r = binomsum(N,G)
                out = MultiVector{t,N}(sig(b),zeros(t,2^N))
                out.v[r+1:r+binomial(N,G)] = $op(value(b,Vector{t}))
                out.v[binomsum(N,L)+basisindex(N,findall(basis(a).b))] = value(a,t)
                return out
            end
            function $op(a::$Blade{T,N,G},b::MultiVector{S,N}) where {T,N,G,S}
                sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
                t = promote_type(T,S)
                r = binomsum(N,G)
                out = MultiVector{t,N}(sig(b),$op(value(b,Vector{t})))
                out.v[r+1:r+binomial(N,G)] += value(b,Vector{t})
                return out
            end
            function $op(a::MultiVector{T,N},b::$Blade{S,N,G}) where {T,N,G,S}
                sig(a) ≠ sig(b) && throw(error("$(sig(a)) ≠ $(sig(b))"))
                t = promote_type(T,S)
                r = binomsum(N,G)
                out = MultiVector{t,N,G}(sig(a),value(a,Vector{t}))
                $(Expr(eop,:(out.v[r+1:r+binomial(N,G)]),:(value(b,Vector{t}))))
                return out
            end
        end
    end
end

## outer product

## inner product

## regressive product
