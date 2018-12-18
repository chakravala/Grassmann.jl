
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

for (op,set) ∈ [(:add,:(+=)),(:set,:(=))]
    sm = Symbol(op,:multi!)
    sb = Symbol(op,:blade!)
    @eval begin
        @inline function $sm(out::MArray{Tuple{M},T,1,M},val::T,i::UInt16) where {M,T}
            $(Expr(set,:(out[basisindex(intlog(M),i)]),:val))
            return out
        end
        @inline function $sm(out::Union{MArray{Tuple{M},T,1,M},Array{T,1}},val::T,i::UInt16,::Dimension{N}) where {M,T,N}
            $(Expr(set,:(out[basisindex(N,i)]),:val))
            return out
        end
        @inline function $sm(s::Signature{N},m::Union{MArray{Tuple{M},T,1,M},Array{T,1}},v::T,A::UInt16,B::UInt16) where {N,T,M}
            !(s[end-1] && isodd(A) && isodd(B)) && $sm(m,parity(A,B,s) ? -(v) : v,A .⊻ B,Dimension{N}())
            return m
        end
        @inline function $sb(out::MArray{Tuple{M},T,1,M},val::T,i::UInt16) where {M,T}
            $(Expr(set,:(out[basisindexb(intlog(M),i)]),:val))
            return out
        end
        @inline function $sb(out::Union{MArray{Tuple{M},T,1,M},Array{T,1}},val::T,i::UInt16,::Dimension{N}) where {M,T,N}
            $(Expr(set,:(out[basisindexb(N,i)]),:val))
            return out
        end

    end
end

@inline add!(out::MultiVector{T,N},val::T,a::Int,b::Int) where {T,N} = add!(out,val,UInt16(a),UInt16(b))
@inline function add!(m::MultiVector{T,N},v::T,A::UInt16,B::UInt16) where {T,N}
    !(m.s[end-1] && isodd(A) && isodd(B)) && addmulti!(m.v,parity(A,B,m.s) ? -(v) : v,A.⊻B)
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
    out = zeros(t,2^N)
    for g ∈ 0:N
        r = binomsum(N,g)
        ib = indexbasis(N,g)
        for i ∈ 1:binomial(N,g)
            addmulti!(a.s,out,a.v[r+i],ib[i],b.i)
        end
    end
    return MultiVector{t,N}(a.s,zeros(t,2^N))
end
function *(a::Basis{N,G},b::MultiVector{T,N}) where {N,G,T}
    sigcheck(a.s,b.s)
    t = promote_type(T,valuetype(a))
    out = zeros(t,2^N)
    for g ∈ 0:N
        r = binomsum(N,g)
        ib = indexbasis(N,g)
        for i ∈ 1:binomial(N,g)
            addmulti!(a.s,out,b.v[r+i],a.i,ib[i])
        end
    end
    return MultiVector{t,N}(a.s,out)
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
            out = zeros(t,2^N)
            for g ∈ 0:N
                r = binomsum(N,g)
                ib = indexbasis(N,g)
                for i ∈ 1:binomial(N,g)
                    addmulti!(a.s,out,a.v[r+i]*b.v,ib[i],b.b.i)
                end
            end
            return MultiVector{t,N}(a.s,out)
        end
        function *(a::$Value{N,G,T},b::MultiVector{S,N}) where {N,G,T,S}
            sigcheck(a.b.s,b.s)
            t = promote_type(T,S)
            out = zeros(t,2^N)
            for g ∈ 0:N
                r = binomsum(N,g)
                ib = indexbasis(N,g)
                for i ∈ 1:binomial(N,g)
                    addmulti!(a.s,out,a.v*b.v[r+i],a.b.i,ib[i])
                end
            end
            return MultiVector{t,N}(a.s,out)
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
            out = zeros(t,2^N)
            ib = indexbasis(N,G)
            for i ∈ 1:binomial(N,G)
                addmulti!(a.s,out,a[i],ib[i],b.i)
            end
            return MultiVector{t,N}(a.s,out)
        end
        function *(a::Basis{N},b::$Blade{T,N,G}) where {T,N,G}
            sigcheck(a.s,b.s)
            t = promote_type(T,valuetype(a))
            out = zeros(t,2^N)
            ib = indexbasis(N,G)
            for i ∈ 1:binomial(N,G)
                addmulti!(a.s,out,b[i],a.i,ib[i])
            end
            return MultiVector{t,N}(b.s,out)
        end
        function *(a::MultiVector{T,N},b::$Blade{S,N,G}) where {T,N,S,G}
            sigcheck(a.s,b.s)
            t = promote_type(T,S)
            out = zeros(t,2^N)
            bng = binomial(N,G)
            B = indexbasis(N,G)
            for g ∈ 0:N
                r = binomsum(N,g)
                A = indexbasis(N,g)
                for i ∈ 1:binomial(N,g)
                    for j ∈ 1:bng
                        addmulti!(a.s,out,a.v[r+i]*b[j],A[i],B[j])
                    end
                end
            end
            return MultiVector{t,N}(a.s,out)
        end
        function *(a::$Blade{T,N,G},b::MultiVector{S,N}) where {N,G,S,T}
            sigcheck(a.s,b.s)
            t = promote_type(T,S)
            out = zeros(t,2^N)
            bng = binomial(N,G)
            A = indexbasis(N,G)
            for g ∈ 0:N
                r = binomsum(N,g)
                B = indexbasis(N,g)
                for i ∈ 1:binomial(N,g)
                    for j ∈ 1:bng
                        addmulti!(a.s,out,a[j]*b.v[r+i],A[j],B[i])
                    end
                end
            end
            return MultiVector{t,N}(a.s,out)
        end
    end
    for Value ∈ MSV
        @eval begin
            function *(a::$Blade{T,N,G},b::$Value{N,L,S}) where {T,N,G,L,S}
                sigcheck(a.s,b.b.s)
                t = promote_type(T,S)
                out = zeros(t,2^N)
                ib = indexbasis(N,G)
                for i ∈ 1:binomial(N,G)
                    addmulti!(a.s,out,a[i]*b.v,ib[i],b.b.i)
                end
                return MultiVector{t,N}(a.s,out)
            end
            function *(a::$Value{N,L,S},b::$Blade{T,N,G}) where {T,N,G,L,S}
                sigcheck(a.b.s,b.s)
                t = promote_type(T,S)
                out = zeros(t,2^N)
                ib = indexbasis(N,G)
                for i ∈ 1:binomial(N,G)
                    addmulti!(o.s,ut,a.v*b[i],a.b.i,ib[i])
                end
                return MultiVector{t,N}(b.s,out)
            end
        end
    end
end
for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
    @eval begin
        function *(a::$A{T,N,G},b::$B{T,N,L}) where {T,N,G,L}
            sigcheck(a.s,b.s)
            bnl = binomial(N,L)
            out = zeros(T,2^N)
            B = indexbasis(N,L)
            A = indexbasis(N,G)
            for i ∈ 1:binomial(N,G)
                for j ∈ 1:bnl
                    addmulti!(a.s,out,a[i]*b[j],A[i],B[j])
                end
            end
            return MultiVector{T,N}(a.s,out)
        end
    end
end
*(a::Number,b::MultiVector{T,N}) where {T,N} = MultiVector{T,N}(b.s,a.*b.v)
*(a::MultiVector{T,N},b::Number) where {T,N} = MultiVector{T,N}(a.s,a.v.*b)
function *(a::MultiVector{T,N},b::MultiVector{S,N}) where {N,T,S}
    sigcheck(a.s,b.s)
    t = promote_type(T,S)
    out = zeros(t,2^N)
    bng = [binomial(N,g) for g ∈ 0:N]
    A = [indexbasis(N,g) for g ∈ 0:N]
    for g ∈ 0:N
        r = binomsum(N,g)
        B = indexbasis(N,g)
        for i ∈ 1:binomial(N,g)
            for G ∈ 0:N
                for j ∈ 1:bng[g+1]
                    addmulti!(a.s,out,a[G][j]*b.v[r+i],A[G][j],B[i])
                end
            end
        end
    end
    return MultiVector{t,N}(a.s,out)
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
                out = zeros(T,binomial(N,A))
                setblade!(out,value(a,T),basis(a).i,Dimension{N}())
                setblade!(out,$op(value(b,T)),basis(b).i,Dimension{N}())
                return MBlade{T,N,A}(sig(a),out)
            else
                warn("sparse MultiGrade{N} objects not properly handled yet")
                return MultiGrade{N}(a,b)
            end
        end

        function $op(a::A,b::MultiVector{T,N}) where A<:AbstractTerm{N,G} where {T,N,G}
            sigcheck(sig(a),sig(b))
            t = promote_type(T,valuetype(a))
            out = $op(value(b,Vector{t}))
            addmulti!(out,value(b,Vector{t}),basis(a).i,Dimension{N}())
            return MultiVector{t,N}(sig(b),out)
        end
        function $op(a::MultiVector{T,N},b::B) where B<:AbstractTerm{N,G} where {T,N,G}
            sigcheck(sig(a),sig(b))
            t = promote_type(T,valuetype(b))
            out = copy(value(a,Vector{t}))
            addmulti!(out,$op(value(b,t)),basis(b).i,Dimension{N}())
            return MultiVector{t,N}(sig(a),out)
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
                out = copy(value(a,Vector{t}))
                addblade!(out,$op(value(b,t)),basis(b).i,Dimension{N}())
                return MBlade{t,N,G}(sig(a),out)
            end
            function $op(a::A,b::$Blade{T,N,G}) where A<:AbstractTerm{N,G} where {T,N,G}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,valuetype(a))
                out = $op(value(b,Vector{t}))
                addblade!(out,value(a,t),basis(a).i,Dimension{N}())
                return MBlade{t,N,G}(sig(b),out)
            end
            function $op(a::$Blade{T,N,G},b::B) where B<:AbstractTerm{N,L} where {T,N,G,L}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,valuetype(b))
                r = binomsum(N,G)
                out = zeros(t,2^N)
                out[r+1:r+binomial(N,G)] = value(a,Vector{t})
                addmulti!(out,$op(value(b,t)),basis(b).i,Dimension{N}())
                return MultiVector{t,N}(sig(a),out)
            end
            function $op(a::A,b::$Blade{T,N,G}) where A<:AbstractTerm{N,L} where {T,N,G,L}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,valuetype(a))
                r = binomsum(N,G)
                out = zeros(t,2^N)
                out[r+1:r+binomial(N,G)] = $op(value(b,Vector{t}))
                addmulti!(out,value(a,t),basis(a).i,Dimension{N}())
                return MultiVector{t,N}(sig(b),out)
            end
            function $op(a::$Blade{T,N,G},b::MultiVector{S,N}) where {T,N,G,S}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,S)
                r = binomsum(N,G)
                out = $op(value(b,Vector{t}))
                out[r+1:r+binomial(N,G)] += value(b,Vector{t})
                return MultiVector{t,N}(sig(b),out)
            end
            function $op(a::MultiVector{T,N},b::$Blade{S,N,G}) where {T,N,G,S}
                sigcheck(sig(a),sig(b))
                t = promote_type(T,S)
                r = binomsum(N,G)
                out = copy(value(a,Vector{t}))
                $(Expr(eop,:(out[r+1:r+binomial(N,G)]),:(value(b,Vector{t}))))
                return MultiVector{t,N}(sig(a),out)
            end
        end
    end
end

## outer product

## inner product

## regressive product
