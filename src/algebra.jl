
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: +, -, *, ^, /, //, inv, <, >, <<, >>, >>>
import AbstractLattices: ∧, ∨, dist
import AbstractTensors: ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, rem, div, contraction
import DirectSum: dualcheck, tangent, hasinforigin, hasorigininf
export tangent

const Field = Number
const ExprField = Union{Expr,Symbol}

@pure g_one(b::Type{Basis{V}}) where V = getbasis(V,bits(b))
@pure g_zero(V::VectorSpace) = 0*one(V)
@pure g_one(V::VectorSpace) = Basis{V}()
@pure g_one(::Type{T}) where T = one(T)
@pure g_zero(::Type{T}) where T = zero(T)

## mutating operations

add_val(set,expr,val,OP) = Expr(OP∉(:-,:+) ? :.= : set,expr,OP∉(:-,:+) ? Expr(:.,OP,Expr(:tuple,expr,val)) : val)

const Sym = :(Reduce.Algebra)
const SymField = Any

set_val(set,expr,val) = Expr(:(=),expr,set≠:(=) ? Expr(:call,:($Sym.:+),expr,val) : val)

function declare_mutating_operations(M,F,set_val,SUB,MUL)
    for (op,set) ∈ ((:add,:(+=)),(:set,:(=)))
        sm = Symbol(op,:multi!)
        sb = Symbol(op,:blade!)
        for (s,index) ∈ ((sm,:basisindex),(sb,:bladeindex))
            for (i,B) ∈ ((:i,Bits),(:(bits(i)),Basis))
                @eval begin
                    @inline function $s(out::$M,val::S,i::$B) where {M,T<:$F,S<:$F}
                        @inbounds $(set_val(set,:(out[$index(intlog(M),$i)]),:val))
                        return out
                    end
                    @inline function $s(out::Q,val::S,i::$B,::Dimension{N}) where Q<:$M where {M,T<:$F,S<:$F,N}
                        @inbounds $(set_val(set,:(out[$index(N,$i)]),:val))
                        return out
                    end
                end
            end
        end
        for s ∈ (sm,sb)
            @eval begin
                @inline function $(Symbol(:join,s))(V::W,m::$M,A::Bits,B::Bits,v::S) where W<:VectorSpace{N,D} where {N,D,T<:$F,S<:$F,M}
                    if v ≠ 0 && !dualcheck(V,A,B)
                        val = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(A,B,V) ? $SUB(v) : v) : $MUL(parityinner(A,B,V),v)
                        $s(m,val,A⊻B,Dimension{N}())
                    end
                    return m
                end
                @inline function $(Symbol(:geom,s))(V::W,m::$M,A::Bits,B::Bits,v::S) where W<:VectorSpace{N,D} where {N,D,T<:$F,S<:$F,M}
                    if v ≠ 0 && !dualcheck(V,A,B)
                        pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(A,B,V) : (false,A⊻B,false)
                        val = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(A,B,V)⊻pcc ? $SUB(v) : v) : $MUL(parityinner(A,B,V),pcc ? $SUB(v) : v)
                        $s(m,val,bas,Dimension{N}())
                        cc && $s(m,hasinforigin(V,A,B) ? $SUB(val) : val,conformalmask(V)⊻bas,Dimension{N}())
                    end
                    return m
                end
            end
            for j ∈ (:join,:geom)
                @eval @inline function $(Symbol(j,s))(m::$M,v::S,A::Basis{V},B::Basis{V}) where {V,T<:$F,S<:$F,M}
                    $(Symbol(j,s))(V,m,bits(A),bits(B),v)
                end
            end
            for (prod,uct) ∈ ((:meet,:regressive),(:skew,:interior),(:cross,:crossprod))
                @eval begin
                    @inline function $(Symbol(prod,s))(V::W,m::$M,A::Bits,B::Bits,v::T) where W<:VectorSpace{N,D} where {N,D,T,M}
                        if v ≠ 0
                            g,C,t = $uct(A,B,V)
                            t && $s(m,typeof(V) <: Signature ? g ? $SUB(v) : v : $MUL(g,v),C,Dimension{N}())
                        end
                        return m
                    end
                    @inline function $(Symbol(prod,s))(m::$M,A::Basis{V},B::Basis{V},v::T) where {V,T,M}
                        $(Symbol(prod,s))(V,m,bits(A),bits(B),v)
                    end
                end
            end
        end
    end
end

@inline exteraddmulti!(V::W,out,α,β,γ) where W<:VectorSpace = (count_ones(α&β)==0) && joinaddmulti!(V,out,α,β,γ)

@inline outeraddblade!(V::W,out,α,β,γ) where W<:VectorSpace = (count_ones(α&β)==0) && joinaddblade!(V,out,α,β,γ)

@inline function add!(out::MultiVector{T,V},val::T,a::Int,b::Int) where {T,V}
    A,B = Bits(a), Bits(b)
    add!(out,val,Basis{V,count_ones(A),A},Basis{V,count_ones(B),B})
end
@inline function add!(m::MultiVector{T,V},v::T,a::Basis{V},b::Basis{V}) where {T<:Field,V}
    A,B = bits(a), bits(b)
    !dualcheck(V,A,B) && addmulti!(m.v,parity(A,B) ? -(v) : v,A.⊻B)
    return out
end

# Hodge star ★

const complementright = ⋆

## complement

export complementleft, complementright, ⋆

for side ∈ (:left,:right)
    c = Symbol(:complement,side)
    p = Symbol(:parity,side)
    @eval @pure function $c(b::Basis{V,G,B}) where {V,G,B}
        d = getbasis(V,complement(ndims(V),B,diffmode(V)))
        dualtype(V)<0 && throw(error("Complement for mixed tensors is undefined"))
        typeof(V)<:Signature ? ($p(b) ? SValue{V}(-value(d),d) : d) : SValue{V}($p(b)*value(d),d)
    end
    for Value ∈ MSV
        @eval $c(b::$Value) = value(b)≠0 ? value(b)*$c(basis(b)) : g_zero(vectorspace(b))
    end
end

## reverse

import Base: reverse, conj, ~
export involute

for r ∈ (:reverse,:involute,:conj)
    p = Symbol(:parity,r)
    @eval @pure $r(b::Basis{V,G,B}) where {V,G,B} =$p(G) ? SValue{V}(-value(b),b) : b
    for Value ∈ MSV
        @eval $r(b::$Value) = value(b) ≠ 0 ? value(b) * $r(basis(b)) : g_zero(vectorspace(b))
    end
end

reverse(a::UniformScaling{Bool}) = UniformScaling(!a.λ)
reverse(a::UniformScaling{T}) where T<:Field = UniformScaling(-a.λ)

@inline ~(b::TensorAlgebra) = reverse(b)
@inline ~(b::UniformScaling) = reverse(b)

## geometric product

@pure function *(a::Basis{V},b::Basis{V}) where V
    A,B = bits(a), bits(b)
    dualcheck(V,A,B) && (return g_zero(V))
    pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(A,B,V) : (false,A⊻B,false)
    d = Basis{V}(bas)
    out = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(a,b)⊻pcc ? SValue{V}(-1,d) : d) : SValue{V}((pcc ? -1 : 1)*parityinner(A,B,V),d)
    return cc ? (v=value(out);out+SValue{V}(hasinforigin(V,A,B) ? -(v) : v,Basis{V}(conformalmask(V)⊻bits(d)))) : out
end

for Value ∈ MSV
    @eval begin
        *(a::$Value{V},b::Basis{V}) where V = SValue{V}(a.v,basis(a)*b)
        *(a::Basis{V},b::$Value{V}) where V = SValue{V}(b.v,a*basis(b))
    end
end

#*(a::MultiGrade{V},b::Basis{V}) where V = MultiGrade{V}(a.v,basis(a)*b)
#*(a::Basis{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*basis(b))
#*(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,basis(a)*basis(b))

for Value ∈ MSV
    @eval begin
        *(a::UniformScaling,b::$Value{V}) where V = V(a)*b
        *(a::$Value{V},b::UniformScaling) where V = a*V(b)
    end
end
for Blade ∈ MSB
    @eval begin
        *(a::UniformScaling,b::$Blade{T,V} where T) where V = V(a)*b
        *(a::$Blade{T,V} where T,b::UniformScaling) where V = a*V(b)
    end
end

export ⊛

## exterior product

export ∧, ∨, ⊗

@pure function ∧(a::Basis{V},b::Basis{V}) where V
    A,B = bits(a), bits(b)
    (count_ones(A&B)>0) && (return g_zero(V))
    d = Basis{V}(A⊻B)
    return parity(a,b) ? SValue{V}(-1,d) : d
end

function ∧(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    x,y = basis(a), basis(b)
    A,B = bits(x), bits(y)
    (count_ones(A&B)>0) && (return g_zero(V))
    v = value(a)*value(b)
    return SValue{V}(parity(x,y) ? -v : v,Basis{V}(A⊻B))
end

#∧(a::MultiGrade{V},b::Basis{V}) where V = MultiGrade{V}(a.v,basis(a)*b)
#∧(a::Basis{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*basis(b))
#∧(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,basis(a)*basis(b))
@inline ∧(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∧,a,b)
@inline ∧(a::TensorAlgebra{V},b::UniformScaling{T}) where {V,T<:Field} = a∧V(b)
@inline ∧(a::UniformScaling{T},b::TensorAlgebra{V}) where {V,T<:Field} = V(a)∧b

⊗(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a∧b

## regressive product: (L = grade(a) + grade(b); (-1)^(L*(L-ndims(V)))*⋆(⋆(a)∧⋆(b)))

@pure function ∨(a::Basis{V},b::Basis{V}) where V
    p,C,t = regressive(a,b)
    !t && (return g_zero(V))
    d = Basis{V}(C)
    return p ? SValue{V}(-1,d) : d
end

function ∨(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    p,C,t = regressive(bits(basis(a)),bits(basis(b)),V)
    !t && (return g_zero(V))
    v = value(a)*value(b)
    return SValue{V}(p ? -v : v,Basis{V}(C))
end

@inline ∨(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∨,a,b)
@inline ∨(a::TensorAlgebra{V},b::UniformScaling{T}) where {V,T<:Field} = a∨V(b)
@inline ∨(a::UniformScaling{T},b::TensorAlgebra{V}) where {V,T<:Field} = V(a)∨b

Base.:&(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a∨b

## interior product: a ∨ ⋆(b)

import LinearAlgebra: dot, ⋅
export ⋅

@pure function contraction(a::Basis{V},b::Basis{V}) where V
    g,C,t = interior(a,b)
    !t && (return g_zero(V))
    d = Basis{V}(C)
    return typeof(V) <: Signature ? (g ? SValue{V}(-1,d) : d) : SValue{V}(g,d)
end

function contraction(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    g,C,t = interior(bits(basis(a)),bits(basis(b)),V)
    !t && (return g_zero(V))
    v = value(a)*value(b)
    return SValue{V}(typeof(V) <: Signature ? (g ? -v : v) : g*v,Basis{V}(C))
end

export ⨼, ⨽

for T ∈ (:TensorTerm,MSB...)
    @eval @inline Base.abs2(t::T) where T<:$T = contraction(t,t)
end

@inline dot(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = contraction(a,b)

#=for A ∈ (:TensorTerm,MSB...), B ∈ (:TensorTerm,MSB...)
    @eval contraction(a::A,b::B) where {A<:$A,B<:$B} where V = contraction(a,b)
end=#

## cross product

import LinearAlgebra: cross
export ×

cross(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = ⋆(a∧b)

@pure function cross(a::Basis{V},b::Basis{V}) where V
    p,C,t = crossprod(a,b)
    !t && (return zero(V))
    d = Basis{V}(C)
    return p ? SValue{V}(-1,d) : d
end

function cross(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    p,C,t = crossprod(bits(basis(a)),bits(basis(b)),V)
    !t && (return zero(V))
    v = value(a)*value(b)
    return SValue{V}(p ? -v : v,Basis{V}(C))
end

# symmetrization and anti-symmetrization

export ⊙, ⊠

⊙(x::TensorAlgebra...) = (K=length(x); sum([prod(x[k]) for k ∈ collect(permutations(1:K))])/factorial(K))

function ⊠(x::TensorAlgebra...)
    K,V,out = length(x),∪(vectorspace.(x)...),prod(x)
    P,F = collect(permutations(1:K)),factorial(K)
    for n ∈ 2:F
        p = prod(x[P[n]])
        DirectSum.indexparity!(P[n],V)[1] ? (out-=p) : (out+=p)
    end
    return out/F
end

<<(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a⊙b
>>(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a⊠b

## sandwich product

>>>(x::TensorAlgebra{V},y::TensorAlgebra{V}) where V = x * y * ~x

### Product Algebra Constructor

function generate_product_algebra(Field=Field,VEC=:mvec,MUL=:*,ADD=:+,SUB=:-,CONJ=:conj)
    if Field == Grassmann.Field
        declare_mutating_operations(:(MArray{Tuple{M},T,1,M}),Field,Expr,:-,:*)
    elseif Field ∈ (SymField,:(SymPy.Sym))
        declare_mutating_operations(:(SizedArray{Tuple{M},T,1,1}),Field,set_val,SUB,MUL)
    end
    Field == :(SymPy.Sym) && for par ∈ (:parany,:parval,:parsym)
        @eval $par = ($par...,$Field)
    end
    TF = Field ≠ Number ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    @eval begin
        @inline function inneraddvalue!(mv::MValue{V,0,B,T} where {W,B},α,β,γ::T) where {V,T<:$Field}
            if γ≠0
                g,C,f = interior(α,β,V)
                f && (mv.v = typeof(V)<:Signature ? (g ? $SUB(mv.v,γ) : $ADD(mv.v,γ)) : $ADD(mv.v,$MUL(g,γ)))
            end
            return mv
        end
        function adjoint(m::MultiVector{T,V}) where {T<:$Field,V}
            if dualtype(V)<0
                $(insert_expr((:N,:M,:bs,:bn),VEC)...)
                out = zeros($VEC(N,$TF))
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds setmulti!(out,$CONJ(m.v[bs[g]+i]),dual(V,ib[i],M))
                    end
                end
            else
                out = $CONJ.(value(m))
            end
            MultiVector{$TF,dual(V)}(out)
        end
        *(a::F,b::Basis{V}) where {F<:$EF,V} = SValue{V}(a,b)
        *(a::Basis{V},b::F) where {F<:$EF,V} = SValue{V}(b,a)
        *(a::F,b::MultiVector{T,V}) where {F<:$Field,T<:$Field,V} = MultiVector{promote_type(T,F),V}(broadcast($MUL,a,b.v))
        *(a::MultiVector{T,V},b::F) where {F<:$Field,T<:$Field,V} = MultiVector{promote_type(T,F),V}(broadcast($MUL,a.v,b))
        *(a::F,b::MultiGrade{V}) where {F<:$EF,V} = MultiGrade{V}(broadcast($MUL,a,b.v))
        *(a::MultiGrade{V},b::F) where {F<:$EF,V} = MultiGrade{V}(broadcast($MUL,a.v,b))
        ∧(a::$Field,b::$Field) = $MUL(a,b)
        ∧(a::F,b::B) where B<:TensorTerm{V,G} where {F<:$EF,V,G} = SValue{V,G}(a,b)
        ∧(a::A,b::F) where A<:TensorTerm{V,G} where {F<:$EF,V,G} = SValue{V,G}(b,a)
        #=∧(a::$Field,b::MultiVector{T,V}) where {T<:$Field,V} = MultiVector{T,V}(a.*b.v)
        ∧(a::MultiVector{T,V},b::$Field) where {T<:$Field,V} = MultiVector{T,V}(a.v.*b)
        ∧(a::$Field,b::MultiGrade{V}) where V = MultiGrade{V}(a.*b.v)
        ∧(a::MultiGrade{V},b::$Field) where V = MultiGrade{V}(a.v.*b)=#
    end
    for Value ∈ MSV
        @eval begin
            adjoint(b::$Value{V,G,B,T}) where {V,G,B,T<:$Field} = $Value{dual(V),G,B',$TF}($CONJ(value(b)))
            *(a::F,b::$Value{V,G,B,T} where B) where {F<:$Field,V,G,T<:$Field} = SValue{V,G}($MUL(a,b.v),basis(b))
            *(a::$Value{V,G,B,T} where B,b::F) where {F<:$Field,V,G,T<:$Field} = SValue{V,G}($MUL(a.v,b),basis(a))
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
            function adjoint(m::$Blade{T,V,G}) where {T<:$Field,V,G}
                if dualtype(V)<0
                    $(insert_expr((:N,:M,:ib),VEC)...)
                    out = zeros($VEC(N,G,$TF))
                    for i ∈ 1:binomial(N,G)
                        @inbounds setblade!(out,$CONJ(m.v[i]),dual(V,ib[i],M),Dimension{N}())
                    end
                else
                    out = $CONJ.(value(m))
                end
                $Blade{$TF,dual(V),G}(out)
            end
            *(a::F,b::$Blade{T,V,G}) where {F<:$Field,T<:$Field,V,G} = SBlade{promote_type(T,F),V,G}(broadcast($MUL,a,b.v))
            *(a::$Blade{T,V,G},b::F) where {F<:$Field,T<:$Field,V,G} = SBlade{promote_type(T,F),V,G}(broadcast($MUL,a.v,b))
            #∧(a::$Field,b::$Blade{T,V,G}) where {T<:$Field,V,G} = SBlade{T,V,G}(a.*b.v)
            #∧(a::$Blade{T,V,G},b::$Field) where {T<:$Field,V,G} = SBlade{T,V,G}(a.v.*b)
            function contraction(a::$Blade{T,V,G},b::Basis{V,G}) where {T<:$Field,V,G}
                $(insert_expr((:N,:t,:mv,:ib),VEC)...)
                for i ∈ 1:binomial(N,G)
                    @inbounds inneraddvalue!(mv,ib[i],bits(b),a[i])
                end
                return mv
            end
            function contraction(a::Basis{V,G},b::$Blade{T,V,G}) where {V,T<:$Field,G}
                $(insert_expr((:N,:t,:mv,:ib),VEC)...)
                for i ∈ 1:binomial(N,G)
                    @inbounds inneraddvalue!(mv,bits(a),ib[i],b[i])
                end
                return mv
            end
            function ∧(a::$Blade{T,w,1},b::Basis{W,1}) where {T<:$Field,w,W}
                V = w==W ? w : ((w==dual(W)) ? (dualtype(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                $(insert_expr((:N,:t),VEC)...)
                ib = indexbasis(ndims(w),1)
                out = zeros($VEC(N,2,t))
                C,y = dualtype(w)>0,dualtype(W)>0 ? dual(V,bits(b)) : bits(b)
                for i ∈ 1:length(a)
                    @inbounds outeraddblade!(V,out,C ? dual(V,ib[i]) : ib[i],y,a[i])
                end
                return MBlade{t,V,2}(out)
            end
            function ∧(a::Basis{w,1},b::$Blade{T,W,1}) where {w,W,T<:$Field}
                V = w==W ? w : ((w==dual(W)) ? (dualtype(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                $(insert_expr((:N,:t),VEC)...)
                ib = indexbasis(ndims(W),1)
                out = zeros($VEC(N,2,t))
                C,x = dualtype(W)>0,dualtype(w)>0 ? dual(V,bits(a)) : bits(a)
                for i ∈ 1:length(b)
                    @inbounds outeraddblade!(V,out,x,C ? dual(V,ib[i]) : ib[i],b[i])
                end
                return MBlade{t,V,2}(out)
            end
        end
        for Value ∈ MSV
            @eval begin
                function contraction(a::$Blade{T,V,G},b::$Value{V,G,B,S}) where {T<:$Field,V,G,B,S<:$Field}
                    $(insert_expr((:N,:t,:mv,:ib),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        @inbounds inneraddvalue!(mv,ib[i],bits(basis(b)),$MUL(a[i],b.v))
                    end
                    return mv
                end
                function contraction(a::$Value{V,G,B,S},b::$Blade{T,V,G}) where {T<:$Field,V,G,B,S<:$Field}
                    $(insert_expr((:N,:t,:mv,:ib),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        @inbounds inneraddvalue!(mv,bits(basis(a)),ib[i],$MUL(a.v,b[i]))
                    end
                    return mv
                end
                function ∧(a::$Blade{T,w,1},b::$Value{W,1,B,S}) where {T<:$Field,w,W,B,S<:$Field}
                    V = w==W ? w : ((w==dual(W)) ? (dualtype(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                    $(insert_expr((:N,:t),VEC)...)
                    ib = indexbasis(ndims(w),1)
                    out = zeros($VEC(N,2,t))
                    C,y = dualtype(w)>0,dualtype(W)>0 ? dual(V,bits(basis(b))) : bits(basis(b))
                    for i ∈ 1:length(a)
                        @inbounds outeraddblade!(V,out,C ? dual(V,ib[i]) : ib[i],y,$MUL(a[i],b.v))
                    end
                    return MBlade{t,V,2}(out)
                end
                function ∧(a::$Value{w,1,B,S},b::$Blade{T,W,1}) where {T<:$Field,w,W,B,S<:$Field}
                    V = w==W ? w : ((w==dual(W)) ? (dualtype(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                    $(insert_expr((:N,:t),VEC)...)
                    ib = indexbasis(ndims(W),1)
                    out = zeros($VEC(N,2,t))
                    C,x = dualtype(W)>0,dualtype(w)>0 ? dual(V,bits(basis(a))) : bits(basis(a))
                    for i ∈ 1:length(b)
                        @inbounds outeraddblade!(V,out,x,C ? dual(V,ib[i]) : ib[i],$MUL(a.v,b[i]))
                    end
                    return MBlade{t,V,2}(out)
                end
            end
        end
    end
    for Blade ∈ MSB, Other ∈ MSB
        @eval begin
            function contraction(a::$Blade{T,V,G},b::$Other{S,V,G}) where {T<:$Field,V,G,S<:$Field}
                $(insert_expr((:N,:t,:mv,:bng,:ib),VEC)...)
                for i ∈ 1:bng
                    @inbounds v,ibi = a[i],ib[i]
                    v≠0 && for j ∈ 1:bng
                        @inbounds inneraddvalue!(mv,ibi,ib[j],$MUL(v,b[j]))
                    end
                end
                return mv
            end
            function ∧(a::$Blade{T,w,1},b::$Other{S,W,1}) where {T<:$Field,w,S<:$Field,W}
                V = w==W ? w : ((w==dual(W)) ? (dualtype(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                $(insert_expr((:N,:t),VEC)...)
                ia = indexbasis(ndims(w),1)
                ib = indexbasis(ndims(W),1)
                out = zeros($VEC(N,2,t))
                CA,CB = dualtype(w)>0,dualtype(W)>0
                for i ∈ 1:length(a)
                    @inbounds v,iai = a[i],ia[i]
                    x = CA ? dual(V,iai) : iai
                    v≠0 && for j ∈ 1:length(b)
                        @inbounds outeraddblade!(V,out,x,CB ? dual(V,ib[j]) : ib[j],$MUL(v,b[j]))
                    end
                end
                return MBlade{t,V,2}(out)
            end
        end
    end
    for side ∈ (:left,:right)
        c = Symbol(:complement,side)
        p = Symbol(:parity,side)
        for Blade ∈ MSB
            @eval begin
                function $c(b::$Blade{T,V,G}) where {T<:$Field,V,G}
                    dualtype(V)<0 && throw(error("Complement for mixed tensors is undefined"))
                    $(insert_expr((:N,:ib,:D),VEC)...)
                    out = zeros($VEC(N,G,T))
                    D = diffmode(V)
                    for k ∈ 1:binomial(N,G)
                        @inbounds val = b.v[k]
                        if val≠0
                            p = $p(V,ib[k])
                            v = typeof(V)<:Signature ? (p ? $SUB(val) : val) : p*val
                            @inbounds setblade!(out,v,complement(N,ib[k],D),Dimension{N}())
                        end
                    end
                    return $Blade{T,V,N-G}(out)
                end
            end
        end
        @eval begin
            function $c(m::MultiVector{T,V}) where {T<:$Field,V}
                dualtype(V)<0 && throw(error("Complement for mixed tensors is undefined"))
                $(insert_expr((:N,:bs,:bn),VEC)...)
                out = zeros($VEC(N,T))
                D = diffmode(V)
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds val = m.v[bs[g]+i]
                        if val≠0
                            v = typeof(V)<:Signature ? ($p(V,ib[i]) ? $SUB(val) : val) : $p(V,ib[i])*val
                            @inbounds setmulti!(out,v,complement(N,ib[i],D),Dimension{N}())
                        end
                    end
                end
                return MultiVector{T,V}(out)
            end
        end
    end
    for reverse ∈ (:reverse,:involute,:conj)
        p = Symbol(:parity,reverse)
        for Blade ∈ MSB
            @eval begin
                function $reverse(b::$Blade{T,V,G}) where {T<:$Field,V,G}
                    !$p(G) && (return b)
                    $(insert_expr((:N,:ib),VEC)...)
                    out = zeros($VEC(N,G,T))
                    for k ∈ 1:binomial(N,G)
                        @inbounds val = b.v[k]
                        @inbounds val≠0 && setblade!(out,$SUB(val),ib[k],Dimension{N}())
                    end
                    return $Blade{T,V,G}(out)
                end
            end
        end
        @eval begin
            function $reverse(m::MultiVector{T,V}) where {T<:$Field,V}
                $(insert_expr((:N,:bs,:bn),VEC)...)
                out = zeros($VEC(N,T))
                for g ∈ 1:N+1
                    pg = $p(g-1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds val = m.v[bs[g]+i]
                        @inbounds val≠0 && setmulti!(out,pg ? $SUB(val) : val,ib[i],Dimension{N}())
                    end
                end
                return MultiVector{T,V}(out)
            end
        end
    end

    for (op,product!) ∈ ((:∧,:exteraddmulti!),(:*,:geomaddmulti!),
                         (:∨,:meetaddmulti!),(:contraction,:skewaddmulti!),
                         (:cross,:crossaddmulti!))
        @eval begin
            function $op(a::MultiVector{T,V},b::Basis{V,G}) where {T<:$Field,V,G}
                $(insert_expr((:N,:t,:out,:bs,:bn),VEC)...)
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds $product!(V,out,ib[i],bits(b),a.v[bs[g]+i])
                    end
                end
                return MultiVector{t,V}(out)
            end
            function $op(a::Basis{V,G},b::MultiVector{T,V}) where {V,G,T<:$Field}
                $(insert_expr((:N,:t,:out,:bs,:bn),VEC)...)
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds $product!(V,out,bits(a),ib[i],b.v[bs[g]+i])
                    end
                end
                return MultiVector{t,V,2^N}(out)::MultiVector{t,V,2^N}
            end
            function $op(a::MultiVector{T,V},b::MultiVector{S,V}) where {V,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t,:out,:bs,:bn),VEC)...)
                for g ∈ 1:N+1
                    Y = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds val = b.v[bs[g]+i]
                        val≠0 && for G ∈ 1:N+1
                            @inbounds R = bs[G]
                            X = indexbasis(N,G-1)
                            @inbounds for j ∈ 1:bn[G]
                                @inbounds $product!(V,out,X[j],Y[i],$MUL(a.v[R+j],val))
                            end
                        end
                    end
                end
                return MultiVector{t,V}(out)
            end
        end
        for Value ∈ MSV
            @eval begin
                function $op(a::MultiVector{T,V},b::$Value{V,G,B,S}) where {T<:$Field,V,G,B,S<:$Field}
                    $(insert_expr((:N,:t,:out,:bs,:bn),VEC)...)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds $product!(V,out,ib[i],bits(basis(b)),$MUL(a.v[bs[g]+i],b.v))
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Value{V,G,B,T},b::MultiVector{S,V}) where {V,G,B,T<:$Field,S<:$Field}
                    $(insert_expr((:N,:t,:out,:bs,:bn),VEC)...)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds $product!(V,out,bits(basis(a)),ib[i],$MUL(a.v,b.v[bs[g]+i]))
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
                        @inbounds $product!(V,out,ib[i],bits(b),a[i])
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::Basis{V},b::$Blade{T,V,G}) where {V,T<:$Field,G}
                    $(insert_expr((:N,:t,:out,:ib),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        @inbounds $product!(V,out,bits(a),ib[i],b[i])
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::MultiVector{T,V},b::$Blade{S,V,G}) where {T<:$Field,V,S<:$Field,G}
                    $(insert_expr((:N,:t,:out,:bng,:ib,:bs,:bn),VEC)...)
                    for g ∈ 1:N+1
                        A = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = a.v[bs[g]+i]
                            val≠0 && for j ∈ 1:bng
                                @inbounds $product!(V,out,A[i],ib[j],$MUL(val,b[j]))
                            end
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Blade{T,V,G},b::MultiVector{S,V}) where {V,G,S<:$Field,T<:$Field}
                    $(insert_expr((:N,:t,:out,:bng,:ib,:bs,:bn),VEC)...)
                    for g ∈ 1:N+1
                        B = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = b.v[bs[g]+i]
                            val≠0 && for j ∈ 1:bng
                                @inbounds $product!(V,out,ib[j],B[i],$MUL(a[j],val))
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
                            @inbounds $product!(V,out,ib[i],bits(basis(b)),$MUL(a[i],b.v))
                        end
                        return MultiVector{t,V}(out)
                    end
                    function $op(a::$Value{V,L,B,S},b::$Blade{T,V,G}) where {T<:$Field,V,G,L,B,S<:$Field}
                        $(insert_expr((:N,:t,:out,:ib),VEC)...)
                        for i ∈ 1:binomial(N,G)
                            @inbounds $product!(V,out,bits(basis(a)),ib[i],$MUL(a.v,b[i]))
                        end
                        return MultiVector{t,V}(out)
                    end
                end
            end
        end
        for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
            @eval begin
                function $op(a::$A{T,V,G},b::$B{S,V,L}) where {T<:$Field,V,G,S<:$Field,L}
                    $(insert_expr((:N,:t,:bnl,:ib),VEC)...)
                    out = zeros($VEC(N,t))
                    B = indexbasis(N,L)
                    for i ∈ 1:binomial(N,G)
                        @inbounds v,ibi = a[i],ib[i]
                        v≠0 && for j ∈ 1:bnl
                            @inbounds $product!(V,out,ibi,B[j],$MUL(v,b[j]))
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                #=function $op(a::$A{T,V,1},b::$B{S,W,1}) where {T<:$Field,V,S<:$Field,W}
                    !(V == dual(W) && V ≠ W) && throw(error())
                    $(insert_expr((:N,:t,:bnl,:ib),VEC)...)
                    out = zeros($VEC(N,2,t))
                    B = indexbasis(N,L)
                    for i ∈ 1:binomial(N,G)
                        for j ∈ 1:bnl
                            @inbounds $product!(V,out,ib[i],B[j],$MUL(a[i],b[j]))
                        end
                    end
                    return MultiVector{t,V}(out)
                end=#
            end
        end
    end

    ## term addition

    for (op,eop,bop) ∈ ((:+,:(+=),ADD),(:-,:(-=),SUB))
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
                function $op(a::$Value{V,G,A,S} where A,b::MultiVector{T,V}) where {T<:$Field,V,G,S<:$Field}
                    $(insert_expr((:N,:t),VEC)...)
                    out = $(bcast(bop,:(copy(value(b,$VEC(N,t))),)))
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
            $op(a::MultiVector{T,V}) where {T<:$Field,V} = MultiVector{$TF,V}($(bcast(bop,:(value(a),))))
            function $op(a::Basis{V,G},b::MultiVector{T,V}) where {T<:$Field,V,G}
                $(insert_expr((:N,:t),VEC)...)
                out = $(bcast(bop,:(copy(value(b,$VEC(N,t))),)))
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
                $(add_val(eop,:out,:(value(b,$VEC(N,t))),bop))
                return MultiVector{t,V}(out)
            end
        end
        for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
            C = (A == MSB[1] && B == MSB[1]) ? MSB[1] : MSB[2]
            @eval begin
                function $op(a::$A{T,V,G},b::$B{S,V,L}) where {T<:$Field,V,G,S<:$Field,L}
                    $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                    @inbounds out[r+1:r+bng] = value(a,MVector{bng,t})
                    rb = binomsum(N,L)
                    Rb = binomial(N,L)
                    @inbounds out[rb+1:rb+Rb] = $(bcast(bop,:(value(b,$VEC(N,L,t)),)))
                    return MultiVector{t,V}(out)
                end
                function $op(a::$A{T,V,G},b::$B{S,V,G}) where {T<:$Field,V,G,S<:$Field}
                    return $C{promote_type(valuetype(a),valuetype(b)),V,G}($(bcast(bop,:(a.v,b.v))))
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
                        out = $(bcast(bop,:(copy(value(b,$VEC(N,G,t))),)))
                        addblade!(out,value(a,t),basis(a),Dimension{N}())
                        return MBlade{t,V,G}(out)
                    end
                    function $op(a::$Blade{T,V,G},b::$Value{V,L,B,S} where B) where {T<:$Field,V,G,L,S<:$Field}
                        $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                        @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                        addmulti!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                    function $op(a::$Value{V,L,A,S} where A,b::$Blade{T,V,G}) where {T<:$Field,V,G,L,S<:$Field}
                        $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                        @inbounds out[r+1:r+bng] = $(bcast(bop,:(value(b,$VEC(N,G,t)),)))
                        addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                end
            end
            @eval begin
                $op(a::$Blade{T,V,G}) where {T<:$Field,V,G} = $Blade{$TF,V,G}($(bcast(bop,:(value(a),))))
                function $op(a::$Blade{T,V,G},b::Basis{V,G}) where {T<:$Field,V,G}
                    $(insert_expr((:N,:t),VEC)...)
                    out = copy(value(a,$VEC(N,G,t)))
                    addblade!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                    return MBlade{t,V,G}(out)
                end
                function $op(a::Basis{V,G},b::$Blade{T,V,G}) where {T<:$Field,V,G}
                    $(insert_expr((:N,:t),VEC)...)
                    out = $(bcast(bop,:(copy(value(b,$VEC(N,G,t))),)))
                    addblade!(out,value(a,t),basis(a),Dimension{N}())
                    return MBlade{t,V,G}(out)
                end
                function $op(a::$Blade{T,V,G},b::Basis{V,L}) where {T<:$Field,V,G,L}
                    $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                    @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                    addmulti!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
                function $op(a::Basis{V,L},b::$Blade{T,V,G}) where {T<:$Field,V,G,L}
                    $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                    @inbounds out[r+1:r+bng] = $(bcast(bop,:(value(b,$VEC(N,G,t)),)))
                    addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Blade{T,V,G},b::MultiVector{S,V}) where {T<:$Field,V,G,S}
                    $(insert_expr((:N,:t,:r,:bng),VEC)...)
                    out = $(bcast(bop,:(copy(value(b,$VEC(N,t))),)))
                    @inbounds out[r+1:r+bng] += value(a,$VEC(N,G,t))
                    return MultiVector{t,V}(out)
                end
                function $op(a::MultiVector{T,V},b::$Blade{S,V,G}) where {T<:$Field,V,G,S}
                    $(insert_expr((:N,:t,:r,:bng),VEC)...)
                    out = copy(value(a,$VEC(N,t)))
                    @inbounds $(Expr(eop,:(out[r+1:r+bng]),:(value(b,$VEC(N,G,t)))))
                    return MultiVector{t,V}(out)
                end
            end
        end
    end
end

generate_product_algebra()
generate_product_algebra(SymField,:svec,:($Sym.:*),:($Sym.:+),:($Sym.:-),:($Sym.conj))

const NSE = Union{Symbol,Expr,<:Number}

for (op,eop) ∈ ((:+,:(+=)),(:-,:(-=)))
    for Term ∈ (:TensorTerm,:TensorMixed)
        @eval begin
            $op(a::T,b::NSE) where T<:$Term = $op(a,b*one(vectorspace(a)))
            $op(a::NSE,b::T) where T<:$Term = $op(a*one(vectorspace(b)),b)
        end
    end
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

function ^(v::T,i::Integer) where T<:TensorTerm
    i == 0 && (return getbasis(vectorspace(v),0))
    out = basis(v)
    for k ∈ 1:(i-1)%4
        out *= basis(v)
    end
    return typeof(v)<:Basis ? out : out*value(v)^i
end
for Term ∈ (MSB...,:MultiVector,:MultiGrade)
    @eval begin
        function ^(v::$Term,i::Integer)
            i == 0 && (return getbasis(vectorspace(v),0))
            out = v
            for k ∈ 1:i-1
                out *= v
            end
            return out
        end
    end
end

## division

for (nv,d) ∈ ((:inv,:/),(:inv_rat,://))
    @eval begin
        @pure $nv(b::Basis{V,G}) where {V,G}=$d(parityreverse(G) ? -1 : 1,value(b⋅b))*b
        @pure $d(a,b::T) where T<:TensorAlgebra = a*$nv(b)
        function $nv(m::MultiVector{T,V}) where {T,V}
            rm = ~m
            d = rm*m
            fd = norm(d)
            for k ∈ 0:ndims(V)
                dk = d(k)
                norm(dk) ≈ fd && (return $d(rm,dk))
            end
            throw(error("inv($m) is undefined"))
        end
    end
    for Value ∈ MSV
        @eval $nv(b::$Value{V,G,B,T}) where {V,G,B,T} = $Value{V,G,B}($d(parityreverse(G) ? -one(T) : one(T),value(abs2(B))*value(b)))
    end
    for Blade ∈ MSB
        @eval $nv(a::$Blade) = $d(~a,value(abs2(a)))
    end
    for Term ∈ (:TensorTerm,MSB...,:MultiVector,:MultiGrade)
        @eval begin
            @pure $d(a::S,b::T) where {S<:$Term,T<:Number} = a*$d(1,b)
            @pure $d(a::S,b::UniformScaling) where S<:$Term = a*$nv(vectorspace(a)(b))
        end
    end
end

for op ∈ (:div,:rem,:mod,:mod1,:fld,:fld1,:cld,:ldexp)
    for Value ∈ MSV
        @eval Base.$op(b::$Value{V,G,B,T},m) where {V,G,B,T} = $Value{V,G,B}($op(value(b),m))
    end
    for Blade ∈ MSB
        @eval Base.$op(a::$Blade{T,V,G},m) where {T,V,G} = $Blade{T,V,G}($op.(value(a),m))
    end
    @eval begin
        Base.$op(a::Basis{V,G},m) where {V,G} = Basis{V,G}($op(value(a),m))
        Base.$op(a::MultiVector{T,V},m) where {T,V} = MultiVector{T,V}($op.(value(a),m))
    end
end
for op ∈ (:mod2pi,:rem2pi,:rad2deg,:deg2rad)
    for Value ∈ MSV
        @eval Base.$op(b::$Value{V,G,B,T}) where {V,G,B,T} = $Value{V,G,B}($op(value(b)))
    end
    for Blade ∈ MSB
        @eval Base.$op(a::$Blade{T,V,G}) where {T,V,G} = $Blade{T,V,G}($op.(value(a)))
    end
    @eval begin
        Base.$op(a::Basis{V,G}) where {V,G} = Basis{V,G}($op(value(a)))
        Base.$op(a::MultiVector{T,V}) where {T,V} = MultiVector{T,V}($op.(value(a)))
    end
end
for Value ∈ MSV
    @eval Base.rationalize(t::Type,b::$Value{V,G,B,T};tol::Real=eps(T)) where {V,G,B,T} = $Value{V,G,B}(rationalize(t,value(b),tol))
end
for Blade ∈ MSB
    @eval Base.rationalize(t::Type,a::$Blade{T,V,G};tol::Real=eps(T)) where {T,V,G} = $Blade{T,V,G}(rationalize.(t,value(a),tol))
end
Base.rationalize(t::Type,a::Basis{V,G},tol::Real=eps(T)) where {V,G} = Basis{V,G}(rationalize(t,value(a),tol))
Base.rationalize(t::Type,a::MultiVector{T,V};tol::Real=eps(T)) where {T,V} = MultiVector{T,V}(rationalize.(t,value(a),tol))
Base.rationalize(t::T;kvs...) where T<:TensorAlgebra = rationalize(Int,t;kvs...)
