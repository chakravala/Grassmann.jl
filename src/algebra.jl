
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: +, -, *, ^, /, inv
import AbstractLattices: ∧, ∨, dist

Field = Number

## signature compatibility

@inline interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = op(a,b)
@inline function interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{W}} where {V,W}
    VW = V∪W
    return op(VW(a),VW(b))
end

## mutating operations

add_val(set,expr,val,OP) = Expr(OP∉(:-,:+) ? :.= : set,expr,OP∉(:-,:+) ? Expr(:.,OP,Expr(:tuple,expr,val)) : val)

function add!(out::MultiVector{T,V},val::T,A::Vector{Int},B::Vector{Int}) where {T<:Field,V}
    (s,c,t) = indexjoin([A;B],V)
    !t && (out[length(c)][basisindex(N,c)] += s ? -(val) : val)
    return out
end

for (op,set) ∈ [(:add,:(+=)),(:set,:(=))]
    sm = Symbol(op,:multi!)
    sb = Symbol(op,:blade!)
    @eval begin
        @inline function $sm(out::MArray{Tuple{M},T,1,M},val::T,i::Bits) where {M,T<:Field}
            $(Expr(set,:(out[basisindex(intlog(M),i)]),:val))
            return out
        end
        @inline function $sm(out::Q,val::T,i::Bits,::Dimension{N}) where Q<:MArray{Tuple{M},T,1,M} where {M,T<:Field,N}
            $(Expr(set,:(out[basisindex(N,i)]),:val))
            return out
        end
        @inline function $(Symbol(:join,sm))(V::VectorSpace{N,D},m::MArray{Tuple{M},T,1,M},v::T,A::Bits,B::Bits) where {N,D,T<:Field,M}
            !(hasdual(V) && isodd(A) && isodd(B)) && $sm(m,parity(A,B,V) ? -(v) : v,A .⊻ B,Dimension{N}())
            return m
        end
        @inline function $(Symbol(:join,sm))(m::MArray{Tuple{M},T,1,M},v::T,A::Basis{V},B::Basis{V}) where {V,T<:Field,M}
            !(hasdual(V) && hasdual(A) && hasdual(B)) && $sm(m,parity(A,B) ? -(v) : v,bits(A) .⊻ bits(B),Dimension{ndims(V)}())
            return m
        end
        @inline function $sb(out::MArray{Tuple{M},T,1,M},val::T,i::Basis) where {M,T<:Field}
            $(Expr(set,:(out[bladeindex(intlog(M),bits(i))]),:val))
            return out
        end
        @inline function $sb(out::Q,val::T,i::Basis,::Dimension{N}) where Q<:MArray{Tuple{M},T,1,M} where {M,T<:Field,N}
            $(Expr(set,:(out[bladeindex(N,bits(i))]),:val))
            return out
        end
        @inline function $sb(out::MArray{Tuple{M},T,1,M},val::T,i::UInt16) where {M,T<:Field}
            $(Expr(set,:(out[bladeindex(intlog(M),i)]),:val))
            return out
        end
        @inline function $sb(out::Q,val::T,i::Bits,::Dimension{N}) where Q<:MArray{Tuple{M},T,1,M} where {M,T<:Field,N}
            $(Expr(set,:(out[bladeindex(N,i)]),:val))
            return out
        end
    end
end

@inline function add!(out::MultiVector{T,V},val::T,a::Int,b::Int) where {T,V}
    ua = Bits(a)
    ub = Bits(b)
    add!(out,val,Basis{V,count_ones(ua),ua},Basis{V,count_ones(ub),ub})
end
@inline function add!(m::MultiVector{T,V},v::T,A::Basis{V},B::Basis{V}) where {T<:Field,V}
    !(hasdual(V) && isodd(A) && isodd(B)) && addmulti!(m.v,parity(A,B) ? -(v) : v,bits(A).⊻bits(B))
    return out
end

## constructor

@inline assign_expr!(e,x::Vector{Any},v::Symbol,expr) = v ∈ e && push!(x,Expr(:(=),v,expr))

@pure function insert_expr(e,vec=:mvec,T=:(valuetype(a)),S=:(valuetype(b)),L=:(2^N))
    x = Any[] # Any[:(sigcheck(sig(a),sig(b)))]
    assign_expr!(e,x,:N,:(ndims(V)))
    assign_expr!(e,x,:t,vec≠:mvec ? :Any : :(promote_type($T,$S)))
    assign_expr!(e,x,:out,:(zeros($vec(N,t))))
    assign_expr!(e,x,:r,:(binomsum(N,G)))
    assign_expr!(e,x,:bng,:(binomial(N,G)))
    assign_expr!(e,x,:bnl,:(binomial(N,L)))
    assign_expr!(e,x,:ib,:(indexbasis(N,G)))
    return x
end

## geometric product

function *(a::Basis{V},b::Basis{V}) where V
    hasdual(V) && hasdual(a) && hasdual(b) && (return zero(V))
    c = bits(a) ⊻ bits(b)
    d = Basis{V}(c)
    return parity(a,b) ? SValue{V}(-1,d) : d
end

*(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop(*,a,b)

function indexjoin(ind::Vector{Int},s::VectorSpace{N,M} where N) where M
    k = 1
    t = false
    while k < length(ind)
        if ind[k] == ind[k+1]
            ind[k] == 1 && hasdual(s) && (return t, ind, true)
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

@pure function parity_calc(N,S,a,b)
    B = digits(b<<1,base=2,pad=N+1)
    isodd(sum(digits(a,base=2,pad=N+1) .* cumsum!(B,B))+count_ones((a .& b) .& S))
end

const parity_cache = Vector{Vector{Vector{Bool}}}[]
@pure function parity(n,s,a,b)
    s1,a1,b1 = s+1,a+1,b+1
    N = length(parity_cache)
    for k ∈ N+1:n
        push!(parity_cache,Vector{Vector{Bool}}[])
    end
    S = length(parity_cache[n])
    for k ∈ S+1:s1
        push!(parity_cache[n],Vector{Bool}[])
    end
    A = length(parity_cache[n][s1])
    for k ∈ A+1:a1
        push!(parity_cache[n][s1],Bool[])
    end
    B = length(parity_cache[n][s1][a1])
    for k ∈ B+1:b1
        push!(parity_cache[n][s1][a1],parity_calc(n,s,a,k-1))
    end
    parity_cache[n][s1][a1][b1]
end

Base.@pure parity(a::Bits,b::Bits,v::VectorSpace) = parity(ndims(v),value(v),a,b)
Base.@pure parity(a::Basis{V,G,B},b::Basis{V,L,C}) where {V,G,B,L,C} = parity(ndims(V),value(V),bits(a),bits(b))

for Value ∈ MSV
    @eval begin
        *(a::$Value{V},b::Basis{V}) where V = SValue{V}(a.v,basis(a)*b)
        *(a::Basis{V},b::$Value{V}) where V = SValue{V}(b.v,a*basis(b))
    end
end

#*(a::MultiGrade{V},b::Basis{V}) where V = MultiGrade{V}(a.v,basis(a)*b)
#*(a::Basis{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*basis(b))
#*(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,basis(a)*basis(b))

@inline geometric_product!(V::VectorSpace,out,α,β,γ) = γ≠0 && joinaddmulti!(V,out,γ,α,β)

## exterior product

export ∧

function ∧(a::Basis{V},b::Basis{V}) where V
    A = bits(a)
    B = bits(b)
    (count_ones(A&B)>0 || A+B==0) && (return zero(V))
    d = Basis{V}(A⊻B)
    return parity(a,b) ? SValue{V}(-1,d) : d
end

function ∧(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    x = basis(a)
    y = basis(b)
    A = bits(x)
    B = bits(y)
    (count_ones(A&B)>0 || A+B==0) && (return zero(V))
    v = value(a)*value(b)
    return SValue{V}(parity(x,y) ? -v : v,Basis{V}(A⊻B))
end

∧(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∧,a,b)

@inline function exterior_product!(V::VectorSpace,out,α,β,γ)
    (γ≠0) && (count_ones(α&β)==0) && (α+β≠0) && joinaddmulti!(V,out,γ,α,β)
end

#∧(a::MultiGrade{V},b::Basis{V}) where V = MultiGrade{V}(a.v,basis(a)*b)
#∧(a::Basis{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*basis(b))
#∧(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,basis(a)*basis(b))

## inner product

export ⋅

⋅(a::A,b::B) where {A<:TensorTerm{V,1},B<:TensorTerm{V,1}} where V = (a*b+b*a)/(2*Basis{V}())
⋅(a::A,b::B) where {A<:TensorTerm{V,1},B<:TensorTerm{W,1}} where {V,W} = interop(⋅,a,b)
for Blade ∈ MSB
    for Other ∈ MSB
        @eval begin
            ⋅(a::$Blade{T,V,1},b::$Other{S,V,1}) where {T,V,S} = (a*b+b*a)/(2*Basis{V}())
        end
    end
    @eval begin
        ⋅(a::$Blade{T,V,1},b::TensorTerm{V,1}) where {T,V,S} = (a*b+b*a)/(2*Basis{V}())
        ⋅(a::TensorTerm{V,1},b::$Blade{T,V,1}) where {T,V} = (a*b+b*a)/(2*Basis{V}())
    end
end

## regressive product

### Product Algebra Constructor

function generate_product_algebra(Field=Field,MUL=:*,ADD=:+,SUB=:-,VEC=:mvec,CONJ=:conj)
    TF = Field ≠ Number ? :Any : :T
    for Value ∈ MSV
        @eval begin
            adjoint(b::$Value{V,G,B,T}) where {V,G,B,T<:$Field} = $Value{dual(V),G,B',$TF}($CONJ(value(b)))
        end
    end
    for Blade ∈ MSB
        @eval begin
            function adjoint(m::$Blade{T,V,G}) where {T<:$Field,V,G}
                if dualtype(V)<0
                    N = ndims(V)
                    M = Int(N/2)
                    ib = indexbasis(N,G)
                    out = zeros(mvec(N,G,$TF))
                    for i ∈ 1:binomial(N,G)
                        setblade!(out,$CONJ(m.v[i]),dual(V,ib[i],M),Dimension{N}())
                    end
                else
                    out = $CONJ.(value(m))
                end
                $Blade{$TF,dual(V),G}(out)
            end
        end
    end
    @eval begin
        function adjoint(m::MultiVector{T,V}) where {T<:$Field,V}
            if dualtype(V)<0
                N = ndims(V)
                M = Int(N/2)
                out = zeros(mvec(N,$TF))
                bng = binomial_set(N)
                bs = binomsum_set(N)
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    for i ∈ 1:bng[g]
                        setmulti!(out,$CONJ(m.v[bs[g]+i]),dual(V,ib[i],M))
                    end
                end
            else
                out = $CONJ.(value(m))
            end
            MultiVector{$TF,dual(V)}(out)
        end
    end
    for Value ∈ MSV
        @eval begin
            *(a::$Field,b::$Value{V,G,B,T} where B) where {V,G,T<:$Field} = SValue{V,G}($MUL(a,b.v),basis(b))
            *(a::$Value{V,G,B,T} where B,b::$Field) where {V,G,T<:$Field} = SValue{V,G}($MUL(a.v,b),basis(a))
        end
    end
    for (A,B) ∈ [(A,B) for A ∈ MSV, B ∈ MSV]
        @eval begin
            function *(a::$A{V,G,A,T} where {V,G,A},b::$B{W,L,B,S} where {W,L,B}) where {T<:$Field,S<:$Field}
                SValue($MUL(a.v,b.v),basis(a)*basis(b))
            end
        end
    end
    for Blade ∈ MSB
        @eval begin
            *(a::Field,b::$Blade{T,V,G}) where {T<:$Field,V,G} = SBlade{T,V,G}(broadcast($MUL,a,b.v))
            *(a::$Blade{T,V,G},b::Field) where {T<:$Field,V,G} = SBlade{T,V,G}(broadcast(a.v,b))
        end
    end
    @eval begin
        *(a::$Field,b::Basis{V}) where V = SValue{V}(a,b)
        *(a::Basis{V},b::$Field) where V = SValue{V}(b,a)
        *(a::$Field,b::MultiVector{T,V}) where {T<:$Field,V} = MultiVector{T,V}(broadcast($MUL,a,b.v))
        *(a::MultiVector{T,V},b::$Field) where {T<:$Field,V} = MultiVector{T,V}(broadcast($MUL,a.v,b))
        *(a::$Field,b::MultiGrade{V}) where V = MultiGrade{V}(broadcast($MUL,a,b.v))
        *(a::MultiGrade{V},b::$Field) where V = MultiGrade{V}(broadcast($MUL,a.v,b))
        ∧(::$Field,::$Field) = 0
        ∧(a,b::B) where B<:TensorTerm{V,G} where {V,G} = G≠0 ? SValue{V,G}(a,b) : zero(V)
        ∧(a::A,b) where A<:TensorTerm{V,G} where {V,G} = G≠0 ? SValue{V,G}(b,a) : zero(V)
        #=
        ∧(a::$Field,b::MultiVector{T,V}) where {T<:$Field,V} = MultiVector{T,V}(a.*b.v)
        ∧(a::MultiVector{T,V},b::$Field) where {T<:$Field,V} = MultiVector{T,V}(a.v.*b)
        ∧(a::$Field,b::MultiGrade{V}) where V = MultiGrade{V}(a.*b.v)
        ∧(a::MultiGrade{V},b::$Field) where V = MultiGrade{V}(a.v.*b)
        =#
    end
    #=for Blade ∈ MSB
        @eval begin
            ∧(a::$Field,b::$Blade{T,V,G}) where {T<:$Field,V,G} = SBlade{T,V,G}(a.*b.v)
            ∧(a::$Blade{T,V,G},b::$Field) where {T<:$Field,V,G} = SBlade{T,V,G}(a.v.*b)
        end
    end=#
    for (op,product!) ∈ [(:*,:geometric_product!),(:∧,:exterior_product!)]
        @eval begin
            function $op(a::MultiVector{T,V},b::Basis{V,G}) where {T<:$Field,V,G}
                $(insert_expr((:N,:t,:out),VEC)...)
                bs = binomsum_set(N)
                bn = binomial_set(N)
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    for i ∈ 1:bn[g]
                        $product!(V,out,ib[i],bits(b),a.v[bs[g]+i])
                    end
                end
                return MultiVector{t,V}(out)
            end
            function $op(a::Basis{V,G},b::MultiVector{T,V}) where {V,G,T<:$Field}
                $(insert_expr((:N,:t,:out),VEC)...)
                bs = binomsum_set(N)
                bn = binomial_set(N)
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    for i ∈ 1:bn[g]
                        $product!(V,out,bits(a),ib[i],b.v[bs[g]+i])
                    end
                end
                return MultiVector{t,V,2^N}(out)::MultiVector{t,V,2^N}
            end
        end
        for Value ∈ MSV
            @eval begin
                function $op(a::MultiVector{T,V},b::$Value{V,G,B,S}) where {T<:$Field,V,G,B,S<:$Field}
                    $(insert_expr((:N,:t,:out),VEC)...)
                    bs = binomsum_set(N)
                    bn = binomial_set(N)
                    ib = indexbasis_set(N)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        for i ∈ 1:bn[g]
                            $product!(V,out,ib[i],bits(basis(b)),$MUL(a.v[bs[g]+i],b.v))
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Value{V,G,B,T},b::MultiVector{S,V}) where {V,G,B,T<:$Field,S<:$Field}
                    $(insert_expr((:N,:t,:out),VEC)...)
                    bs = binomsum_set(N)
                    bn = binomial_set(N)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        for i ∈ 1:bn[g]
                            $product!(V,out,bits(basis(s)),ib[i],$MUL(a.v,b.v[bs[g]+i]))
                        end
                    end
                    return MultiVector{t,V,2^N}(out)
                end
            end
        end
        for Blade ∈ MSB
            @eval begin
                function $op(a::$Blade{T,V,G},b::Basis{V}) where {T<:$Field,V,G}
                    $(insert_expr((:N,:t,:out,:ib),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        $product!(V,out,ib[i],bits(b),a[i])
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::Basis{V},b::$Blade{T,V,G}) where {V,T<:$Field,G}
                    $(insert_expr((:N,:t,:out,:ib),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        $product!(V,out,bits(a),ib[i],b[i])
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::MultiVector{T,V},b::$Blade{S,V,G}) where {T<:$Field,V,S<:$Field,G}
                    $(insert_expr((:N,:t,:out,:bng,:ib),VEC)...)
                    bs = binomsum_set(N)
                    bn = binomial_set(N)
                    for g ∈ 1:N+1
                        A = indexbasis(N,g-1)
                        for i ∈ 1:bn[g]
                            r = bs[g]+i
                            for j ∈ 1:bng
                                $product!(V,out,A[i],ib[j],$MUL(a.v[r],b[j]))
                            end
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Blade{T,V,G},b::MultiVector{S,V}) where {V,G,S<:$Field,T<:$Field}
                    $(insert_expr((:N,:t,:out,:bng,:ib),VEC)...)
                    bs = binomsum_set(N)
                    bn = binomial_set(N)
                    for g ∈ 1:N+1
                        B = indexbasis(N,g-1)
                        for i ∈ 1:bn[g]
                            r = bs[g]+i
                            for j ∈ 1:bng
                                $product!(V,out,ib[j],B[i],$MUL(a[j],b.v[r]))
                            end
                        end
                    end
                    return MultiVector{t,V}(out)
                end
            end
            for Value ∈ MSV
                @eval begin
                    function $op(a::$Blade{T,V,G},b::$Value{V,L,B,S}) where {T<:$Field,V,G,L,B,S<:$Field}
                        $(insert_expr((:N,:t,:out,:ib),VEC)...)
                        for i ∈ 1:binomial(N,G)
                            $product!(V,out,ib[i],bits(basis(b)),$MUL(a[i],b.v))
                        end
                        return MultiVector{t,V}(out)
                    end
                    function $op(a::$Value{V,L,B,S},b::$Blade{T,V,G}) where {T<:$Field,V,G,L,B,S<:$Field}
                        $(insert_expr((:N,:t,:out,:ib),VEC)...)
                        for i ∈ 1:binomial(N,G)
                            $product!(V,out,bits(basis(a)),ib[i],$MUL(a.v,b[i]))
                        end
                        return MultiVector{t,V}(out)
                    end
                end
            end
        end
        for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
            @eval begin
                function $op(a::$A{T,V,G},b::$B{T,V,L}) where {T<:$Field,V,G,L}
                    $(insert_expr((:N,:bnl,:ib),VEC)...)
                    out = zeros($VEC(N,T))
                    B = indexbasis(N,L)
                    for i ∈ 1:binomial(N,G)
                        for j ∈ 1:bnl
                            $product!(V,out,ib[i],B[j],$MUL(a[i],b[j]))
                        end
                    end
                    return MultiVector{T,V}(out)
                end
            end
        end
        @eval begin
            function $op(a::MultiVector{T,V},b::MultiVector{S,V}) where {V,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t,:out),VEC)...)
                bng = binomial_set(N)
                bs = binomsum_set(N)
                for g ∈ 1:N+1
                    Y = indexbasis(N,g-1)
                    for i ∈ 1:bng[g]
                        r = bs[g]+i
                        for G ∈ 1:N+1
                            R = bs[G]
                            X = indexbasis(N,G-1)
                            for j ∈ 1:bng[G]
                                $product!(V,out,X[j],Y[i],$MUL(a.v[R+j],b.v[r]))
                            end
                        end
                    end
                end
                return MultiVector{t,V}(out)
            end
        end
    end

    ## term addition

    for (op,eop,bop) ∈ [(:+,:(+=),ADD),(:-,:(-=),SUB)]
        for (Value,Other) ∈ [(a,b) for a ∈ MSV, b ∈ MSV]
            @eval begin
                function $op(a::$Value{V,A,X,T},b::$Other{V,B,Y,S}) where {V,A,X,T<:$Field,B,Y,S<:$Field}
                    if X == Y
                        return SValue{V,A}($bop(value(a),value(b)),X)
                    elseif A == B
                        $(insert_expr((:N,:t),VEC)...)
                        out = zeros($VEC(N,A,t))
                        setblade!(out,value(a,t),bits(X),Dimension{N}())
                        setblade!(out,$bop(value(b,t)),bits(Y),Dimension{N}())
                        return MBlade{t,V,A}(out)
                    else
                        #@warn("sparse MultiGrade{V} objects not properly handled yet")
                        #return MultiGrade{V}(a,b)
                        $(insert_expr((:N,:t,:out),VEC)...)
                        setmulti!(out,value(a,t),bits(X),Dimension{N}())
                        setmulti!(out,$bop(value(b,t)),bits(Y),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                end
            end
        end
        for Value ∈ MSV
            @eval begin
                $op(a::$Value{V,G,B,T}) where {V,G,B,T<:$Field} = $Value{V,G,B,$TF}($bop(value(a)))
                function $op(a::$Value{V,A,X,T},b::Basis{V,B,Y}) where {V,A,X,T<:$Field,B,Y}
                    if X == b
                        return SValue{V,A}($bop(value(a),value(b)),b)
                    elseif A == B
                        $(insert_expr((:N,:t),VEC)...)
                        out = zeros($VEC(N,A,t))
                        setblade!(out,value(a,t),bits(X),Dimension{N}())
                        setblade!(out,$bop(value(b,t)),Y,Dimension{N}())
                        return MBlade{t,V,A}(out)
                    else
                        #@warn("sparse MultiGrade{V} objects not properly handled yet")
                        #return MultiGrade{V}(a,b)
                        $(insert_expr((:N,:t,:out),VEC)...)
                        setmulti!(out,value(a,t),bits(X),Dimension{N}())
                        setmulti!(out,$bop(value(b,t)),Y,Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                end
                function $op(a::Basis{V,A,X},b::$Value{V,B,Y,S}) where {V,A,X,B,Y,S<:$Field}
                    if a == Y
                        return SValue{V,A}($bop(value(a),value(b)),a)
                    elseif A == B
                        $(insert_expr((:N,:t),VEC)...)
                        out = zeros($VEC(N,A,t))
                        setblade!(out,value(a,t),X,Dimension{N}())
                        setblade!(out,$bop(value(b,t)),bits(Y),Dimension{N}())
                        return MBlade{t,V,A}(out)
                    else
                        #@warn("sparse MultiGrade{V} objects not properly handled yet")
                        #return MultiGrade{V}(a,b)
                        $(insert_expr((:N,:t,:out),VEC)...)
                        setmulti!(out,value(a,t),X,Dimension{N}())
                        setmulti!(out,$bop(value(b,t)),bits(Y),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                end
            end
        end
        for Value ∈ MSV
            @eval begin
                function $op(a::$Value{V,G,A,S} where A,b::MultiVector{T,V}) where {T<:$Field,V,G,S<:$Field}
                    $(insert_expr((:N,:t),VEC)...)
                    out = $bop(value(b,$VEC(N,t)))
                    addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
                function $op(a::MultiVector{T,V},b::$Value{V,G,B,S} where B) where {T<:$Field,V,G,S<:$Field}
                    $(insert_expr((:N,:t),VEC)...)
                    out = copy(value(a,$VEC(N,t)))
                    addmulti!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
            end
        end
        @eval begin
            $op(a::MultiVector{T,V}) where {T<:$Field,V} = MultiVector{$TF,V}($bop.(value(a)))
            function $op(a::Basis{V,G},b::MultiVector{T,V}) where {T<:$Field,V,G}
                $(insert_expr((:N,:t),VEC)...)
                out = $bop(value(b,$VEC(N,t)))
                addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                return MultiVector{t,V}(out)
            end
            function $op(a::MultiVector{T,V},b::Basis{V,G}) where {T<:$Field,V,G}
                $(insert_expr((:N,:t),VEC)...)
                out = copy(value(a,$VEC(N,t)))
                addmulti!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                return MultiVector{t,V}(out)
            end
            function $op(a::MultiVector{T,V},b::MultiVector{S,V}) where {T<:$Field,V,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = copy(value(a,$VEC(N,t)))
                $(add_val(eop,:out,:(value(b,mvec(N,t))),bop))
                return MultiVector{t,V}(out)
            end
        end

        for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
            @eval begin
                function $op(a::$A{T,V,G},b::$B{S,V,L}) where {T<:$Field,V,G,S<:$Field,L}
                    $(insert_expr((:N,:t,:out),VEC)...)
                    ra = binomsum(N,G)
                    Ra = binomial(N,G)
                    out[ra+1:ra+Ra] = value(a,MVector{Ra,t})
                    rb = binomsum(N,L)
                    Rb = binomial(N,L)
                    out[rb+1:rb+Rb] = $bop(value(b,MVector{Rb,t}))
                    return MultiVector{t,V}(out)
                end
            end
        end
        for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
            C = (A == MSB[1] && B == MSB[1]) ? MSB[1] : MSB[2]
            @eval begin
                function $op(a::$A{T,V,G},b::$B{S,V,G}) where {T<:$Field,V,G,S<:$Field}
                    return $C{promote_type(valuetype(a),valuetype(b)),V,G}($bop(a.v,b.v))
                end
            end
        end
        for Blade ∈ MSB
            for Value ∈ MSV
                @eval begin
                    function $op(a::$Blade{T,V,G},b::$Value{V,G,B,S} where B) where {T<:$Field,V,G,S<:$Field}
                        $(insert_expr((:N,:t),VEC)...)
                        out = copy(value(a,$VEC(N,G,t)))
                        addblade!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                        return MBlade{t,V,G}(out)
                    end
                    function $op(a::$Value{V,G,A,S} where A,b::$Blade{T,V,G}) where {T<:$Field,V,G,S<:$Field}
                        $(insert_expr((:N,:t),VEC)...)
                        out = $bop(value(b,$VEC(N,G,t)))
                        addblade!(out,value(a,t),basis(a),Dimension{N}())
                        return MBlade{t,V,G}(out)
                    end
                    function $op(a::$Blade{T,V,G},b::$Value{V,L,B,S} where B) where {T<:$Field,V,G,L,S<:$Field}
                        $(insert_expr((:N,:t,:out),VEC)...)
                        r = binomsum(N,G)
                        R = binomial(N,G)
                        out[r+1:r+R] = value(a,MVector{R,t})
                        addmulti!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                    function $op(a::$Value{V,L,A,S} where A,b::$Blade{T,V,G}) where {T<:$Field,V,G,L,S<:$Field}
                        $(insert_expr((:N,:t,:out),VEC)...)
                        r = binomsum(N,G)
                        R = binomial(N,G)
                        out[r+1:r+R] = $bop(value(b,MVector{R,t}))
                        addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                end
            end
            @eval begin
                $op(a::$Blade{T,V,G}) where {T<:$Field,V,G} = $Blade{$TF,V,G}($bop.(value(a)))
                function $op(a::$Blade{T,V,G},b::Basis{V,G}) where {T<:$Field,V,G}
                    $(insert_expr((:N,:t),VEC)...)
                    out = copy(value(a,$VEC(N,G,t)))
                    addblade!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                    return MBlade{t,V,G}(out)
                end
                function $op(a::Basis{V,G},b::$Blade{T,V,G}) where {T<:$Field,V,G}
                    $(insert_expr((:N,:t),VEC)...)
                    out = $bop(value(b,$VEC(N,G,t)))
                    addblade!(out,value(a,t),basis(a),Dimension{N}())
                    return MBlade{t,V,G}(out)
                end
                function $op(a::$Blade{T,V,G},b::Basis{V,L}) where {T<:$Field,V,G,L}
                    $(insert_expr((:N,:t,:out),VEC)...)
                    r = binomsum(N,G)
                    R = binomial(N,G)
                    out[r+1:r+R] = value(a,MVector{R,t})
                    addmulti!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
                function $op(a::Basis{V,L},b::$Blade{T,V,G}) where {T<:$Field,V,G,L}
                    $(insert_expr((:N,:t,:out),VEC)...)
                    r = binomsum(N,G)
                    R = binomial(N,G)
                    out[r+1:r+R] = $bop(value(b,MVector{R,t}))
                    addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Blade{T,V,G},b::MultiVector{S,V}) where {T<:$Field,V,G,S}
                    $(insert_expr((:N,:t),VEC)...)
                    r = binomsum(N,G)
                    R = binomial(N,G)
                    out = $bop(value(b,$VEC(N,t)))
                    out[r+1:r+R] += value(b,MVector{R,t})
                    return MultiVector{t,V}(out)
                end
                function $op(a::MultiVector{T,V},b::$Blade{S,V,G}) where {T<:$Field,V,G,S}
                    $(insert_expr((:N,:t),VEC)...)
                    r = binomsum(N,G)
                    R = binomial(N,G)
                    out = copy(value(a,$VEC(N,t)))
                    $(Expr(eop,:(out[r+1:r+R]),:(value(b,MVector{R,t}))))
                    return MultiVector{t,V}(out)
                end
            end
        end
    end
end

generate_product_algebra()

for (op,eop) ∈ [(:+,:(+=)),(:-,:(-=))]
    @eval begin
        $op(a::Basis{V,G,B} where G) where {V,B} = SValue($op(value(a)),a)
        function $op(a::Basis{V,A},b::Basis{V,B}) where {V,A,B}
            if a == b
                return SValue{V,A}($op(value(a),value(b)),basis(a))
            elseif A == B
                $(insert_expr((:N,:t))...)
                out = zeros(mvec(N,A,t))
                setblade!(out,value(a,t),bits(a),Dimension{N}())
                setblade!(out,$op(value(b,t)),bits(b),Dimension{N}())
                return MBlade{t,V,A}(out)
            else
                #@warn("sparse MultiGrade{V} objects not properly handled yet")
                #return MultiGrade{V}(a,b)
                $(insert_expr((:N,:t,:out))...)
                setmulti!(out,value(a,t),bits(a),Dimension{N}())
                setmulti!(out,$op(value(b,t)),bits(b),Dimension{N}())
                return MultiVector{t,V}(out)
            end
        end
    end
end

## exponentiation

function ^(v::TensorTerm,i::Integer)
    i == 0 && (return getbasis(sig(v),0))
    out = v
    for k ∈ 1:(i-1)%4
        out *= v
    end
    return out
end
for Term ∈ [MSB...,:MultiVector,:MultiGrade]
    @eval begin
        function ^(v::$Term,i::Integer)
            i == 0 && (return getbasis(sig(v),0))
            out = v
            for k ∈ 1:i-1
                out *= v
            end
            return out
        end
    end
end

## division

@pure inv_parity(G) = isodd(Int((G*(G-1))/2))
@pure inv(b::Basis) = inv_parity(grade(b)) ? -1*b : b
function inv(b::SValue{V,G,B,T}) where {V,G,B,T}
    SValue{V,G,B}((inv_parity(G) ? -one(T) : one(T))/value(b))
end
for Term ∈ [:TensorTerm,MSB...,:MultiVector,:MultiGrade]
    @eval begin
        @pure /(a::$Term,b::TensorTerm) = a*inv(b)
        @pure /(a::$Term,b::Number) = a*inv(b)
    end
end


