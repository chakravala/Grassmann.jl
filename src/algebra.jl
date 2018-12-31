
#   This file is part of Grassmann.jl. It is licensed under the MIT license
#   Copyright (C) 2018 Michael Reed

import Base: +, -, *

Field = Number

## signature compatibility

sigcheck(a::VectorSpace,b::VectorSpace) = a ≠ b && throw(error("$(a.s) ≠ $(b.s)"))
@inline sigcheck(a,b) = sigcheck(sig(a),sig(b))

## mutating operations

function add!(out::MultiVector{T,V},val::T,A::Vector{Int},B::Vector{Int}) where {T<:Field,V}
    (s,c,t) = indexjoin(A,B,V)
    !t && (out[length(c)][basisindex(N,c)] += s ? -(val) : val)
    return out
end

for (op,set) ∈ [(:add,:(+=)),(:set,:(=))]
    sm = Symbol(op,:multi!)
    sb = Symbol(op,:blade!)
    @eval begin
        @inline function $sm(out::MArray{Tuple{M},T,1,M},val::T,i::UInt16) where {M,T<:Field}
            $(Expr(set,:(out[basisindex(intlog(M),i)]),:val))
            return out
        end
        @inline function $sm(out::Q,val::T,i::UInt16,::Dimension{N}) where Q<:MArray{Tuple{M},T,1,M} where {M,T<:Field,N}
            $(Expr(set,:(out[basisindex(N,i)]),:val))
            return out
        end
        @inline function $sm(s::VectorSpace{N,D},m::MArray{Tuple{M},T,1,M},v::T,A::UInt16,B::UInt16) where {N,D,T<:Field,M}
            !(Bool(D) && isodd(A) && isodd(B)) && $sm(m,parity(A,B,s) ? -(v) : v,A .⊻ B,Dimension{N}())
            return m
        end
        @inline function $sb(out::MArray{Tuple{M},T,1,M},val::T,i::UInt16) where {M,T<:Field}
            $(Expr(set,:(out[basisindexb(intlog(M),i)]),:val))
            return out
        end
        @inline function $sb(out::Q,val::T,i::UInt16,::Dimension{N}) where Q<:MArray{Tuple{M},T,1,M} where {M,T<:Field,N}
            $(Expr(set,:(out[basisindexb(N,i)]),:val))
            return out
        end
    end
end

@inline add!(out::MultiVector{T,V},val::T,a::Int,b::Int) where {T,V} = add!(out,val,UInt16(a),UInt16(b))
@inline function add!(m::MultiVector{T,V},v::T,A::UInt16,B::UInt16) where {T<:Field,V}
    !(hasdual(V) && isodd(A) && isodd(B)) && addmulti!(m.v,parity(A,B,V) ? -(v) : v,A.⊻B)
    return out
end

## constructor


function fun()
    x = Any[:(sigcheck(sig(a),sig(b)))]
    :t => :(t = promote_type($T,$S))
    :N => :(ndims(V))
    :r => :(binomsum(N,G))
    :out => :(zeros(t,2^N))
    :bng => :(binomial(N,G))
    :bnl => :(binomial(N,L))
    :ib => :(indexbasis(N,G))
    return x
end

## geometric product

function *(a::Basis{V},b::Basis{V}) where V
    #(s,c,t) = indexjoin(basisindices(a),basisindices(b),V)
    #t && (return SValue{V}(0,Basis{V,0}()))
    #d = Basis{V}(c)
    #return s ? SValue{V}(-1,d) : d
    hasdual(V) && hasdual(a) && hasdual(b) && (return SValue{V}(0,Basis{V,0}()))
    c = UInt16(a) ⊻ UInt16(b)
    d = Basis{V,count_ones(c)}(c)
    return parity(a,b) ? SValue{V}(-1,d) : d
end

function indexjoin(a::Vector{Int},b::Vector{Int},s::VectorSpace{N,D,O} where {N,O}) where D
    ind = [a;b]
    k = 1
    t = false
    while k < length(ind)
        if ind[k] == ind[k+1]
            ind[k] == 1 && Bool(D) && (return t, ind, true)
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

function parity_calc(N,S,a,b)
    B = digits(b<<1,base=2,pad=N+1)
    isodd(sum(digits(a,base=2,pad=N+1) .* cumsum!(B,B))+count_ones((a .& b) .& S))
end

const parity_cache = ( () -> begin
        Y = Vector{Vector{Vector{Bool}}}[]
        return (n,s,a,b) -> (begin
                s1,a1,b1 = s+1,a+1,b+1
                N = length(Y)
                for k ∈ N+1:n
                    push!(Y,Vector{Vector{Bool}}[])
                end
                S = length(Y[n])
                for k ∈ S+1:s1
                    push!(Y[n],Vector{Bool}[])
                end
                A = length(Y[n][s1])
                for k ∈ A+1:a1
                    push!(Y[n][s1],Bool[])
                end
                B = length(Y[n][s1][a1])
                for k ∈ B+1:b1
                    push!(Y[n][s1][a1],parity_calc(n,s,a,k-1))
                end
                Y[n][s1][a1][b1]
            end)
    end)()

@inline parity(a,b,s::VectorSpace{N,D,O,S} where {D,O}) where {N,S} = parity_cache(N,S,a,b)
@inline parity(a::Basis{V},b::Basis{V}) where V = parity(UInt16(a),UInt16(b),V)

*(a::Field,b::Basis{V}) where V = SValue{V}(a,b)
*(a::Basis{V},b::Field) where V = SValue{V}(b,a)

function *(a::MultiVector{T,V},b::Basis{V,G}) where {T<:Field,V,G}
    N = ndims(V)
    t = promote_type(T,valuetype(b))
    out = @MVector zeros(t,2^N)
    for g ∈ 0:N
        r = binomsum(N,g)
        ib = indexbasis(N,g)
        for i ∈ 1:binomial(N,g)
            addmulti!(V,out,a.v[r+i],ib[i],UInt16(b))
        end
    end
    return MultiVector{t,V}(out)
end
function *(a::Basis{V,G},b::MultiVector{T,V}) where {V,G,T<:Field}
    N = ndims(V)
    t = promote_type(T,valuetype(a))
    out = @MVector zeros(t,2^N)
    for g ∈ 0:N
        r = binomsum(N,g)
        ib = indexbasis(N,g)
        for i ∈ 1:binomial(N,g)
            addmulti!(V,out,b.v[r+i],UInt16(a),ib[i])
        end
    end
    return MultiVector{t,V}(out)
end

for Value ∈ MSV
    @eval begin
        *(a::Field,b::$Value{V,G}) where {V,G} = SValue{V,G}(a*b.v,b.b)
        *(a::$Value{V,G},b::Field) where {V,G} = SValue{V,G}(a.v*b,a.b)
        *(a::$Value{V},b::Basis{V}) where V = SValue{V}(a.v,a.b*b)
        *(a::Basis{V},b::$Value{V}) where V = SValue{V}(b.v,a*b.b)
        function *(a::MultiVector{T,V},b::$Value{V,G,S}) where {T<:Field,V,G,S<:Field}
            N = ndims(V)
            t = promote_type(T,S)
            out = @MVector zeros(t,2^N)
            for g ∈ 0:N
                r = binomsum(N,g)
                ib = indexbasis(N,g)
                for i ∈ 1:binomial(N,g)
                    addmulti!(V,out,a.v[r+i]*b.v,ib[i],UInt16(b.b))
                end
            end
            return MultiVector{t,V}(out)
        end
        function *(a::$Value{V,G,T},b::MultiVector{S,V}) where {V,G,T<:Field,S<:Field}
            N = ndims(V)
            t = promote_type(T,S)
            out = @MVector zeros(t,2^N)
            for g ∈ 0:N
                r = binomsum(N,g)
                ib = indexbasis(N,g)
                for i ∈ 1:binomial(N,g)
                    addmulti!(V,out,a.v*b.v[r+i],UInt16(a.b),ib[i])
                end
            end
            return MultiVector{t,V}(out)
        end
    end
end
for (A,B) ∈ [(A,B) for A ∈ MSV, B ∈ MSV]
    @eval *(a::$A{V},b::$B{V}) where V = SValue{V}(a.v*b.v,a.b*b.b)
end
for Blade ∈ MSB
    @eval begin
        *(a::Field,b::$Blade{T,V,G}) where {T<:Field,V,G} = SBlade{T,V,G}(a.*b.v)
        *(a::$Blade{T,V,G},b::Field) where {T<:Field,V,G} = SBlade{T,V,G}(a.v.*b)
        function *(a::$Blade{T,V,G},b::Basis{V}) where {T<:Field,V,G}
            N = ndims(V)
            t = promote_type(T,valuetype(a))
            out = @MVector zeros(t,2^N)
            ib = indexbasis(N,G)
            for i ∈ 1:binomial(N,G)
                addmulti!(V,out,a[i],ib[i],UInt16(b))
            end
            return MultiVector{t,V}(out)
        end
        function *(a::Basis{V},b::$Blade{T,V,G}) where {V,T<:Field,G}
            N = ndims(V)
            t = promote_type(T,valuetype(a))
            out = @MVector zeros(t,2^N)
            ib = indexbasis(N,G)
            for i ∈ 1:binomial(N,G)
                addmulti!(V,out,b[i],UInt16(a),ib[i])
            end
            return MultiVector{t,V}(out)
        end
        function *(a::MultiVector{T,V},b::$Blade{S,V,G}) where {T<:Field,V,S<:Field,G}
            N = ndims(V)
            t = promote_type(T,S)
            out = @MVector zeros(t,2^N)
            bng = binomial(N,G)
            B = indexbasis(N,G)
            for g ∈ 0:N
                r = binomsum(N,g)
                A = indexbasis(N,g)
                for i ∈ 1:binomial(N,g)
                    for j ∈ 1:bng
                        addmulti!(V,out,a.v[r+i]*b[j],A[i],B[j])
                    end
                end
            end
            return MultiVector{t,V}(out)
        end
        function *(a::$Blade{T,V,G},b::MultiVector{S,V}) where {V,G,S<:Field,T<:Field}
            N = ndims(V)
            t = promote_type(T,S)
            out = @MVector zeros(t,2^N)
            bng = binomial(N,G)
            A = indexbasis(N,G)
            for g ∈ 0:N
                r = binomsum(N,g)
                B = indexbasis(N,g)
                for i ∈ 1:binomial(N,g)
                    for j ∈ 1:bng
                        addmulti!(V,out,a[j]*b.v[r+i],A[j],B[i])
                    end
                end
            end
            return MultiVector{t,V}(out)
        end
    end
    for Value ∈ MSV
        @eval begin
            function *(a::$Blade{T,V,G},b::$Value{V,L,S}) where {T<:Field,V,G,L,S<:Field}
                N = ndims(V)
                t = promote_type(T,S)
                out = @MVector zeros(t,2^N)
                ib = indexbasis(N,G)
                for i ∈ 1:binomial(N,G)
                    addmulti!(V,out,a[i]*b.v,ib[i],UInt16(b.b))
                end
                return MultiVector{t,V}(out)
            end
            function *(a::$Value{V,L,S},b::$Blade{T,V,G}) where {T<:Field,V,G,L,S<:Field}
                N = ndims(V)
                t = promote_type(T,S)
                out = @MVector zeros(t,2^N)
                ib = indexbasis(N,G)
                for i ∈ 1:binomial(N,G)
                    addmulti!(V,ut,a.v*b[i],UInt16(a.b),ib[i])
                end
                return MultiVector{t,V}(out)
            end
        end
    end
end
for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
    @eval begin
        function *(a::$A{T,V,G},b::$B{T,V,L}) where {T<:Field,V,G,L}
            N = ndims(V)
            bnl = binomial(N,L)
            out = @MVector zeros(T,2^N)
            B = indexbasis(N,L)
            A = indexbasis(N,G)
            for i ∈ 1:binomial(N,G)
                for j ∈ 1:bnl
                    addmulti!(V,out,a[i]*b[j],A[i],B[j])
                end
            end
            return MultiVector{T,V}(out)
        end
    end
end
*(a::Field,b::MultiVector{T,V}) where {T<:Field,V} = MultiVector{T,V}(a.*b.v)
*(a::MultiVector{T,V},b::Field) where {T<:Field,V} = MultiVector{T,V}(a.v.*b)
function *(a::MultiVector{T,V},b::MultiVector{S,V}) where {V,T<:Field,S<:Field}
    N = ndims(V)
    t = promote_type(T,S)
    out = @MVector zeros(t,2^N)
    bng = [binomial(N,g) for g ∈ 0:N]
    A = [indexbasis(N,g) for g ∈ 0:N]
    for g ∈ 0:N
        r = binomsum(N,g)
        B = indexbasis(N,g)
        for i ∈ 1:binomial(N,g)
            for G ∈ 0:N
                R = binomsum(N,G)
                for j ∈ 1:bng[G+1]
                    addmulti!(V,out,a.v[R+j]*b.v[r+i],A[G+1][j],B[i])
                end
            end
        end
    end
    return MultiVector{t,V}(out)
end

*(a::Field,b::MultiGrade{V}) where V = MultiGrade{V}(a.*b.v)
*(a::MultiGrade{V},b::Field) where V = MultiGrade{V}(a.v.*b)
#*(a::MultiGrade{V},b::Basis{V}) where V = MultiGrade{V}(a.v,a.b*b)
#*(a::Basis{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*b.b)
#*(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,a.b*b.b)

## term addition

for (op,eop) ∈ [(:+,:(+=)),(:-,:(-=))]
    @eval begin
        function $op(a::AbstractTerm{V,A},b::AbstractTerm{V,B}) where {V,A,B}
            if basis(a) == basis(b)
                return SValue{V,A}($op(value(a),value(b)),a)
            elseif A == B
                N = ndims(V)
                T = promote_type(valuetype(a),valuetype(b))
                out = @MVector zeros(T,binomial(N,A))
                setblade!(out,value(a,T),UInt16(basis(a)),Dimension{N}())
                setblade!(out,$op(value(b,T)),UInt16(basis(b)),Dimension{N}())
                return MBlade{T,V,A}(out)
            else
                @warn("sparse MultiGrade{V} objects not properly handled yet")
                return MultiGrade{V}(a,b)
            end
        end

        function $op(a::A,b::MultiVector{T,V}) where A<:AbstractTerm{V,G} where {T<:Field,V,G}
            N = ndims(V)
            t = promote_type(T,valuetype(a))
            out = $op(value(b,MVector{2^N,t}))
            addmulti!(out,value(b,t),UInt16(basis(a)),Dimension{N}())
            return MultiVector{t,V}(out)
        end
        function $op(a::MultiVector{T,V},b::B) where B<:AbstractTerm{V,G} where {T<:Field,V,G}
            N = ndims(V)
            t = promote_type(T,valuetype(b))
            out = copy(value(a,MVector{2^N,t}))
            addmulti!(out,$op(value(b,t)),UInt16(basis(b)),Dimension{N}())
            return MultiVector{t,V}(out)
        end
    end

    for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
        @eval begin
            function $op(a::$A{T,V,G},b::$B{S,V,L}) where {T<:Field,V,G,S<:Field,L}
                N = ndims(V)
                t = promote_type(T,S)
                out = @MVector zeros(t,2^N)
                ra = binomsum(N,G)
                Ra = binomial(N,G)
                out[ra+1:ra+Ra] = value(a,MVector{Ra,t})
                rb = binomsum(N,L)
                Rb = binomial(N,L)
                out[rb+1:rb+Rb] = $op(value(b,MVector{Rb,t}))
                return MultiVector{promote_type(T,S),V}(out)
            end
        end
    end
    for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
        C = (A == MSB[1] && B == MSB[1]) ? MSB[1] : MSB[2]
        @eval begin
            function $op(a::$A{T,V,G},b::$B{S,V,G}) where {T<:Field,V,G,S<:Field}
                return $C{promote_type(T,S),V,G}($op(a.v,b.v))
            end
        end
    end
    for Blade ∈ MSB
        @eval begin
            function $op(a::$Blade{T,V,G},b::B) where B<:AbstractTerm{V,G} where {T<:Field,V,G}
                N = ndims(V)
                t = promote_type(T,valuetype(b))
                out = copy(value(a,MVector{binomial(N,G),t}))
                addblade!(out,$op(value(b,t)),UInt16(basis(b)),Dimension{N}())
                return MBlade{t,V,G}(out)
            end
            function $op(a::A,b::$Blade{T,V,G}) where A<:AbstractTerm{V,G} where {T<:Field,V,G}
                N = ndims(V)
                t = promote_type(T,valuetype(a))
                out = $op(value(b,MVector{binomial(N,G),t}))
                addblade!(out,value(a,t),UInt16(basis(a)),Dimension{N}())
                return MBlade{t,V,G}(out)
            end
            function $op(a::$Blade{T,V,G},b::B) where B<:AbstractTerm{V,L} where {T<:Field,V,G,L}
                N = ndims(V)
                t = promote_type(T,valuetype(b))
                r = binomsum(N,G)
                R = binomial(N,G)
                out = @MVector zeros(t,2^N)
                out[r+1:r+R] = value(a,MVector{R,t})
                addmulti!(out,$op(value(b,t)),UInt16(basis(b)),Dimension{N}())
                return MultiVector{t,V}(out)
            end
            function $op(a::A,b::$Blade{T,V,G}) where A<:AbstractTerm{V,L} where {T<:Field,V,G,L}
                N = ndims(V)
                t = promote_type(T,valuetype(a))
                r = binomsum(N,G)
                R = binomial(N,G)
                out = @MVector zeros(t,2^N)
                out[r+1:r+R] = $op(value(b,MVector{R,t}))
                addmulti!(out,value(a,t),UInt16(basis(a)),Dimension{N}())
                return MultiVector{t,V}(out)
            end
            function $op(a::$Blade{T,V,G},b::MultiVector{S,V}) where {T<:Field,V,G,S}
                N = ndims(V)
                t = promote_type(T,S)
                r = binomsum(N,G)
                R = binomial(N,G)
                out = $op(value(b,MVector{2^N,t}))
                out[r+1:r+R] += value(b,MVector{R,t})
                return MultiVector{t,V}(out)
            end
            function $op(a::MultiVector{T,V},b::$Blade{S,V,G}) where {T<:Field,V,G,S}
                N = ndims(V)
                t = promote_type(T,S)
                r = binomsum(N,G)
                R = binomial(N,G)
                out = copy(value(a,MVector{2^N,t}))
                $(Expr(eop,:(out[r+1:r+R]),:(value(b,MVector{R,t}))))
                return MultiVector{t,V}(out)
            end
        end
    end
end

## outer product

## inner product

## regressive product
