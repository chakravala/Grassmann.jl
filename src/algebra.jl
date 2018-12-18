
#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

import Base: +, -, *

## signature compatibility

sigcheck(a::Signature,b::Signature) = a ≠ b && throw(error("$(a.s) ≠ $(b.s)"))
sigcheck(a,b) = sigcheck(sig(a),sig(b))

## mutating operations

function add!(out::MultiVector{T,N},val::T,A::Vector{Int},B::Vector{Int}) where {T,N}
    (s,c,t) = indexjoin(A,B,out.s)
    !t && (out[length(c)][basisindex(N,c)] += s ? -(val) : val)
    return out
end

@inline add!(out::MultiVector{T,N},val::T,a::Int,b::Int) where {T,N} = add!(out,val,UInt16(a),UInt16(b))
@inline function add!(out::MultiVector{T,N},val::T,A::UInt16,B::UInt16) where {T,N}
    !(out.s[end-1] && isodd(A) && isodd(B)) && (out.v[basisindex(N,A .⊻ B)] += parity(A,B,out.s) ? -(val) : val)
     return out
end

## geometric product

function *(a::Basis{N},b::Basis{N}) where N
    sigcheck(a.s,b.s)
    #(s,c,t) = indexjoin(basisindices(a),basisindices(b),a.s)
    #t && (return SValue{N}(0,Basis{N,0}(a.s)))
    #d = Basis{N}(a.s,c)
    #return s ? SValue{N}(-1,d) : d
    (a.s[end-1] && a[1] && b[1]) && (return SValue{N}(0,Basis{N,0}(a.s)))
    c = a.i ⊻ b.i
    d = Basis{N,basisgrade(N,c)}(a.s,c)
    return parity(a,b) ? SValue{N}(-1,d) : d
end

function indexjoin(a::Vector{Int},b::Vector{Int},s::Signature{N}) where N
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

@inline function parity(a::UInt16, b::UInt16,s::Signature{N}) where N
    B = digits(b<<1,base=2,pad=N+1)
    isodd(sum(digits(a,base=2,pad=N+1) .* cumsum!(B,B))+sum(digits((a .& b) .& s.i,base=2)))
end

@inline parity(a::Basis{N}, b::Basis{N}) where N = parity(a.i,b.i,a.s)

*(a::Number,b::Basis{N}) where N = SValue{N}(a,b)
*(a::Basis{N},b::Number) where N = SValue{N}(b,a)

function *(a::MultiVector{T,N},b::Basis{N,G}) where {T,N,G}
    sigcheck(a.s,b.s)
    t = promote_type(T,valuetype(b))
    out = MultiVector{t,N}(a.s,zeros(t,2^N))
    for g ∈ 0:N
        r = binomsum(N,g)
        for i ∈ 1:binomial(N,g)
            add!(out,a.v[r+i],basisindices(a.s,g,i),basisindices(b))
        end
    end
    return out
end
function *(a::Basis{N,G},b::MultiVector{T,N}) where {N,G,T}
    sigcheck(a.s,b.s)
    t = promote_type(T,valuetype(a))
    out = MultiVector{t,N}(a.s,zeros(t,2^N))
    for g ∈ 0:N
        r = binomsum(N,g)
        for i ∈ 1:binomial(N,g)
            add!(out,b.v[r+i],basisindices(a),basisindices(a.s,g,i))
        end
    end
    return out
end

for Value ∈ MSV
    @eval begin
        *(a::Number,b::$Value{N,G}) where {N,G} = SValue{N,G}(a*b.v,b.b)
        *(a::$Value{N,G},b::Number) where {N,G} = SValue{N,G}(a.v*b,a.b)
        *(a::$Value{N},b::Basis{N}) where N = SValue{N}(a.v,a.b*b)
        *(a::Basis{N},b::$Value{N}) where N = SValue{N}(b.v,a*b.b)
        function *(a::MultiVector{T,N},b::$Value{N,G,S}) where {T,N,G,S}
            sigcheck(a.s,b.b.s)
            t = promote_type(T,S)
            out = MultiVector{t,N}(a.s,zeros(t,2^N))
            for g ∈ 0:N
                r = binomsum(N,g)
                for i ∈ 1:binomial(N,g)
                    add!(out,a.v[r+i]*b.v,basisindices(a.s,g,i),basisindices(b.b))
                end
            end
            return out
        end
        function *(a::$Value{N,G,T},b::MultiVector{S,N}) where {N,G,T,S}
            sigcheck(a.b.s,b.s)
            t = promote_type(T,S)
            out = MultiVector{t,N}(a.s,zeros(t,2^N))
            for g ∈ 0:N
                r = binomsum(N,g)
                for i ∈ 1:binomial(N,g)
                    add!(out,a.v*b.v[r+i],basisindices(a.b),basisindices(a.s,g,i))
                end
            end
            return out
        end
    end
end
for (A,B) ∈ [(A,B) for A ∈ MSV, B ∈ MSV]
    @eval *(a::$A{N},b::$B{N}) where N = SValue{N}(a.v*b.v,a.b*b.b)
end
for Blade ∈ MSB
    @eval begin
        *(a::Number,b::$Blade{T,N,G}) where {T,N,G} = SBlade{T,N,G}(b.s,a.*b.v)
        *(a::$Blade{T,N,G},b::Number) where {T,N,G} = SBlade{T,N,G}(a.s,a.v.*b)
        function *(a::$Blade{T,N,G},b::Basis{N}) where {T,N,G}
            sigcheck(a.s,b.s)
            t = promote_type(T,valuetype(a))
            out = MultiVector{t,N}(a.s,zeros(t,2^N))
            for i ∈ 1:binomial(N,G)
                add!(out,a[i],basisindices(a.s,G,i),basisindices(b))
            end
            return out
        end
        function *(a::Basis{N},b::$Blade{T,N,G}) where {T,N,G}
            sigcheck(a.s,b.s)
            t = promote_type(T,valuetype(a))
            out = MultiVector{t,N}(b.s,zeros(t,2^N))
            for i ∈ 1:binomial(N,G)
                add!(out,b[i],basisindices(a),basisindices(b.s,G,i))
            end
            return out
        end
        function *(a::MultiVector{T,N},b::$Blade{S,N,G}) where {T,N,S,G}
            sigcheck(a.s,b.s)
            t = promote_type(T,S)
            bng = binomial(N,G)
            out = MultiVector{t,N}(a.s,zeros(t,2^N))
            B = [basisindices(a.s,G,i) for i ∈ 1:bng]
            for g ∈ 0:N
                r = binomsum(N,g)
                for i ∈ 1:binomial(N,g)
                    A = basisindices(a.s,g,i)
                    for j ∈ 1:bng
                        add!(out,a.v[r+i]*b[j],A,B[j])
                    end
                end
            end
            return out
        end
        function *(a::$Blade{T,N,G},b::MultiVector{S,N}) where {N,G,S,T}
            sigcheck(a.s,b.s)
            t = promote_type(T,S)
            bng = binomial(N,G)
            out = MultiVector{t,N}(a.s,zeros(t,2^N))
            A = [basisindices(a.s,G,i) for i ∈ 1:bng]
            for g ∈ 0:N
                r = binomsum(N,g)
                for i ∈ 1:binomial(N,g)
                    B = basisindices(a.s,g,i)
                    for j ∈ 1:bng
                        add!(out,a[j]*b.v[r+i],A[j],B)
                    end
                end
            end
            return out
        end
    end
    for Value ∈ MSV
        @eval begin
            function *(a::$Blade{T,N,G},b::$Value{N,L,S}) where {T,N,G,L,S}
                sigcheck(a.s,b.b.s)
                t = promote_type(T,S)
                out = MultiVector{t,N}(a.s,zeros(t,2^N))
                for i ∈ 1:binomial(N,G)
                    add!(out,a[i]*b.v,basisindices(a.s,G,i),basisindices(b.b))
                end
                return out
            end
            function *(a::$Value{N,L,S},b::$Blade{T,N,G}) where {T,N,G,L,S}
                sigcheck(a.b.s,b.s)
                t = promote_type(T,S)
                out = MultiVector{t,N}(b.s,zeros(t,2^N))
                for i ∈ 1:binomial(N,G)
                    add!(out,a.v*b[i],basisindices(a.b),basisindices(b.s,G,i))
                end
                return out
            end
        end
    end
end
for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
    @eval begin
        function *(a::$A{T,N,G},b::$B{T,N,L}) where {T,N,G,L}
            sigcheck(a.s,b.s)
            bnl = binomial(N,L)
            out = MultiVector{T,N}(a.s,zeros(T,2^N))
            B = [basisindices(b.s,L,i) for i ∈ 1:bnl]
            for i ∈ 1:binomial(N,G)
                A = basisindices(a.s,G,i)
                for j ∈ 1:bnl
                    add!(out,a[i]*b[j],A,B[j])
                end
            end
            return out
        end
    end
end
*(a::Number,b::MultiVector{T,N}) where {T,N} = MultiVector{T,N}(b.s,a.*b.v)
*(a::MultiVector{T,N},b::Number) where {T,N} = MultiVector{T,N}(a.s,a.v.*b)
function *(a::MultiVector{T,N},b::MultiVector{S,N}) where {N,T,S}
    sigcheck(a.s,b.s)
    t = promote_type(T,S)
    bng = [binomial(N,g) for g ∈ 0:N]
    out = MultiVector{t,N}(a.s,zeros(t,2^N))
    A = [[basisindices(a.s,g,i) for i ∈ 1:bng[g+1]] for g ∈ 0:N]
    for g ∈ 0:N
        r = binomsum(N,g)
        for i ∈ 1:binomial(N,g)
            B = basisindices(a.s,g,i)
            for G ∈ 0:N
                for j ∈ 1:bng[g+1]
                    add!(out,a[G][j]*b.v[r+i],A[G][j],B)
                end
            end
        end
    end
    return out
end

*(a::Number,b::MultiGrade{N}) where N = MultiGrade{N}(a.s,a.*b.v)
*(a::MultiGrade{N},b::Number) where N = MultiGrade{N}(a.s,a.v.*b)
#*(a::MultiGrade{N},b::Basis{N}) where N = MultiGrade{N}(a.v,a.b*b)
#*(a::Basis{N},b::MultiGrade{N}) where N = MultiGrade{N}(b.v,a*b.b)
#*(a::MultiGrade{N},b::MultiGrade{N}) where N = MultiGrade{N}(a.v*b.v,a.b*b.b)

## term addition

for (op,eop) ∈ [(:+,:(+=)),(:-,:(-=))]
    @eval begin
        function $op(a::AbstractTerm{N,A},b::AbstractTerm{N,B}) where {N,A,B}
            sigcheck(sig(a),sig(b))
            if basis(a) == basis(b)
                return SValue{N,A}($op(value(a),value(b)),a)
            elseif A == B
                T = promote_type(valuetype(a),valuetype(b))
                out = MBlade{T,N,A}(sig(a),zeros(T,binomial(N,A)))
                out.v[basisindexb(N,basisindices(basis(a)))] = value(a,T)
                out.v[basisindexb(N,basisindices(basis(b)))] = $op(value(b,T))
                return out
            else
                warn("sparse MultiGrade{N} objects not properly handled yet")
                return MultiGrade{N}(a,b)
            end
        end

        function $op(a::A,b::MultiVector{T,N}) where A<:AbstractTerm{N,G} where {T,N,G}
            sigcheck(sig(a),sig(b))
            t = promote_type(T,valuetype(a))
            out = MultiVector{t,N}(sig(b),$op(value(b,Vector{t})))
            out.v[basisindex(N,basisindices(basis(a)))] += value(b,Vector{t})
            return out
        end
        function $op(a::MultiVector{T,N},b::B) where B<:AbstractTerm{N,G} where {T,N,G}
            sigcheck(sig(a),sig(b))
            t = promote_type(T,valuetype(b))
            out = MultiVector{t,N}(sig(a),value(a,Vector{t}))
            $(Expr(eop,:(out.v[basisindex(N,basisindices(basis(b)))]),:(value(b,t))))
            return out
        end
    end

    for Blade ∈ MSB
        @eval begin
            function $op(a::$Blade{T,N,G},b::$Blade{S,N,G}) where {T,N,G,S}
                sigcheck(sig(a),sig(b))
                return $Blade{promote_type(T,S),N,G}($op(a.v,b.v))
            end
            function $op(a::$Blade{T,N,G},b::B) where B<:AbstractTerm{N,G} where {T,N,G}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,valuetype(b))
                out = MBlade{t,N,G}(sig(a),value(a,Vector{t}))
                $(Expr(eop,:(out.v[basisindexb(N,basisindices(basis(b)))]),:(value(b,t))))
                return out
            end
            function $op(a::A,b::$Blade{T,N,G}) where A<:AbstractTerm{N,G} where {T,N,G}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,valuetype(a))
                out = MBlade{t,N,G}(sig(b),$op(value(b,Vector{t})))
                out.v[basisindexb(N,basisindices(basis(a)))] += value(a,t)
                return out
            end
            function $op(a::$Blade{T,N,G},b::B) where B<:AbstractTerm{N,L} where {T,N,G,L}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,valuetype(b))
                r = binomsum(N,G)
                out = MultiVector{t,N}(sig(a),zeros(t,2^N))
                out.v[r+1:r+binomial(N,G)] = value(a,Vector{t})
                out.v[basisindex(N,basisindices(basis(b)))] = $op(value(b,t))
                return out
            end
            function $op(a::A,b::$Blade{T,N,G}) where A<:AbstractTerm{N,L} where {T,N,G,L}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,valuetype(a))
                r = binomsum(N,G)
                out = MultiVector{t,N}(sig(b),zeros(t,2^N))
                out.v[r+1:r+binomial(N,G)] = $op(value(b,Vector{t}))
                out.v[basisindex(N,basisindices(basis(a)))] = value(a,t)
                return out
            end
            function $op(a::$Blade{T,N,G},b::MultiVector{S,N}) where {T,N,G,S}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,S)
                r = binomsum(N,G)
                out = MultiVector{t,N}(sig(b),$op(value(b,Vector{t})))
                out.v[r+1:r+binomial(N,G)] += value(b,Vector{t})
                return out
            end
            function $op(a::MultiVector{T,N},b::$Blade{S,N,G}) where {T,N,G,S}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,S)
                r = binomsum(N,G)
                out = MultiVector{t,N}(sig(a),value(a,Vector{t}))
                $(Expr(eop,:(out.v[r+1:r+binomial(N,G)]),:(value(b,Vector{t}))))
                return out
            end
        end
    end
end

## outer product

## inner product

## regressive product
