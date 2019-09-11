
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: +, -, *, ^, /, //, inv, <, >, <<, >>, >>>
import AbstractLattices: ∧, ∨, dist
import AbstractTensors: ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, ∗, rem, div, contraction
import DirectSum: diffcheck, diffmode, tangent, hasinforigin, hasorigininf
export tangent

const Field = Real
const ExprField = Union{Expr,Symbol}

@pure g_one(b::Type{Basis{V}}) where V = getbasis(V,bits(b))
@pure g_zero(V::Manifold) = 0*one(V)
@pure g_one(V::Manifold) = Basis{V}()
@pure g_one(::Type{T}) where T = one(T)
@pure g_zero(::Type{T}) where T = zero(T)

## mutating operations

add_val(set,expr,val,OP) = Expr(OP∉(:-,:+) ? :.= : set,expr,OP∉(:-,:+) ? Expr(:.,OP,Expr(:tuple,expr,val)) : val)

const Sym = :DirectSum
const SymField = Any

set_val(set,expr,val) = Expr(:(=),expr,set≠:(=) ? Expr(:call,:($Sym.:∑),expr,val) : val)

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
                @inline function $(Symbol(:join,s))(V::W,m::$M,a::Bits,b::Bits,v::S) where W<:Manifold{N} where {N,T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        val = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(A,B,V) ? $SUB(v) : v) : $MUL(parityinner(A,B,V),v)
                        if diffvars(V)≠0
                            !iszero(Z) && (T≠Any ? (return true) : (val *= getbasis(V,Z)))
                            count_ones(Q)+order(val)>diffmode(V) && (return false)
                        end
                        $s(m,val,(A⊻B)|Q,Dimension{N}())
                    end
                    return false
                end
                @inline function $(Symbol(:geom,s))(V::W,m::$M,a::Bits,b::Bits,v::S) where W<:Manifold{N} where {N,T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(A,B,V) : (false,A⊻B,false)
                        val = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(A,B,V)⊻pcc ? $SUB(v) : v) : $MUL(parityinner(A,B,V),pcc ? $SUB(v) : v)
                        if diffvars(V)≠0
                            !iszero(Z) && (T≠Any ? (return true) : (val *= getbasis(V,Z)))
                            count_ones(Q)+order(val)>diffmode(V) && (return false)
                        end
                        $s(m,val,bas|Q,Dimension{N}())
                        cc && $s(m,hasinforigin(V,A,B) ? $SUB(val) : val,(conformalmask(V)⊻bas)|Q,Dimension{N}())
                    end
                    return false
                end
            end
            for j ∈ (:join,:geom)
                @eval @inline function $(Symbol(j,s))(m::$M,v::S,A::Basis{V},B::Basis{V}) where {V,T<:$F,S<:$F,M}
                    $(Symbol(j,s))(V,m,bits(A),bits(B),v)
                end
            end
            for (prod,uct) ∈ ((:meet,:regressive),(:skew,:interior),(:cross,:crossprod))
                @eval begin
                    @inline function $(Symbol(prod,s))(V::W,m::$M,A::Bits,B::Bits,val::T) where W<:Manifold{N} where {N,T,M}
                        if val ≠ 0
                            g,C,t,Z = $uct(A,B,V)
                            v = val
                            if diffvars(V)≠0
                                if !iszero(Z)
                                    T≠Any && (return true)
                                    _,_,Q,_ = symmetricmask(V,A,B)
                                    v *= getbasis(V,Z)
                                end
                                count_ones(Q)+order(v)>diffmode(V) && (return false)
                            end
                            t && $s(m,typeof(V) <: Signature ? g ? $SUB(v) : v : $MUL(g,v),C,Dimension{N}())
                        end
                        return false
                    end
                    @inline function $(Symbol(prod,s))(m::$M,A::Basis{V},B::Basis{V},v::T) where {V,T,M}
                        $(Symbol(prod,s))(V,m,bits(A),bits(B),v)
                    end
                end
            end
        end
    end
end

@inline exteraddmulti!(V::W,out,α,β,γ) where W<:Manifold = (count_ones(α&β)==0) && joinaddmulti!(V,out,α,β,γ)

@inline outeraddblade!(V::W,out,α,β,γ) where W<:Manifold = (count_ones(α&β)==0) && joinaddblade!(V,out,α,β,γ)

@inline function add!(out::MultiVector{T,V},val::T,a::Int,b::Int) where {T,V}
    A,B = Bits(a), Bits(b)
    add!(out,val,Basis{V,count_ones(A),A},Basis{V,count_ones(B),B})
end
@inline function add!(m::MultiVector{T,V},v::T,a::Basis{V},b::Basis{V}) where {T<:Field,V}
    A,B = bits(a), bits(b)
    !diffcheck(V,A,B) && addmulti!(m.v,parity(A,B) ? -(v) : v,A.⊻B)
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
        d = getbasis(V,complement(ndims(V),B,diffvars(V)))
        mixedmode(V)<0 && throw(error("Complement for mixed tensors is undefined"))
        typeof(V)<:Signature ? ($p(b) ? SBlade{V}(-value(d),d) : d) : SBlade{V}($p(b)*value(d),d)
    end
    for Blade ∈ MSB
        @eval $c(b::$Blade) = value(b)≠0 ? value(b)*$c(basis(b)) : g_zero(vectorspace(b))
    end
end

@doc """
    complementright(ω::TensorAlgebra)

Grassmann-Poincare-Hodge complement: ⋆ω = ω∗I
""" complementright

@doc """
    complementleft(ω::TensorAlgebra)

Grassmann-Poincare left complement: ⋆'ω = I∗'ω
""" complementleft

## reverse

import Base: reverse, conj, ~
export involute

@pure grade_basis(V,B) = B&(one(Bits)<<DirectSum.grade(V)-1)
@pure grade_basis(v,::Basis{V,G,B} where G) where {V,B} = grade_basis(V,B)
@pure grade(V,B) = count_ones(grade_basis(V,B))
@pure grade(v,::Basis{V,G,B} where G) where {V,B} = grade(V,B)

for r ∈ (:reverse,:involute,:conj)
    p = Symbol(:parity,r)
    @eval @pure function $r(b::Basis{V,G,B}) where {V,G,B}
        $p(grade(V,B)) ? SBlade{V}(-value(b),b) : b
    end
    for Blade ∈ MSB
        @eval $r(b::$Blade) = value(b) ≠ 0 ? value(b) * $r(basis(b)) : g_zero(vectorspace(b))
    end
end

"""
    ~(ω::TensorAlgebra)

Reverse of a `MultiVector` element: ~ω = (-1)^(grade(ω)*(grade(ω)-1)/2)*ω
"""
reverse(a::UniformScaling{Bool}) = UniformScaling(!a.λ)
reverse(a::UniformScaling{T}) where T<:Field = UniformScaling(-a.λ)

"""
    reverse(ω::TensorAlgebra)

Reverse of a `MultiVector` element: ~ω = (-1)^(grade(ω)*(grade(ω)-1)/2)*ω
"""
@inline ~(b::TensorAlgebra) = reverse(b)
@inline ~(b::UniformScaling) = reverse(b)

@doc """
    involute(ω::TensorAlgebra)

Involute of a `MultiVector` element: ~ω = (-1)^grade(ω)*ω
""" involute

@doc """
    conj(ω::TensorAlgebra)

Clifford conjugate of a `MultiVector` element: conj(ω) = involute(~ω)
""" conj

## geometric product

"""
    *(a::TensorAlgebra,b::TensorAlgebra)

Geometric algebraic product: a*b = (-1)ᵖdet(a∩b)⊗(Λ(a⊖b)∪L(a⊕b))
"""
@pure function *(a::Basis{V},b::Basis{V}) where V
    ba,bb = bits(a),bits(b)
    diffcheck(V,ba,bb) && (return g_zero(V))
    A,B,Q,Z = symmetricmask(V,bits(a),bits(b))
    pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(A,B,V) : (false,A⊻B,false)
    d = getbasis(V,bas|Q)
    out = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(a,b)⊻pcc ? SBlade{V}(-1,d) : d) : SBlade{V}((pcc ? -1 : 1)*parityinner(A,B,V),d)
    diffvars(V)≠0 && !iszero(Z) && (out = SBlade{V}(getbasis(V,Z),out))
    return cc ? (v=value(out);out+SBlade{V}(hasinforigin(V,A,B) ? -(v) : v,getbasis(V,conformalmask(V)⊻bits(d)))) : out
end

for Blade ∈ MSB
    @eval begin
        function *(a::$Blade{V},b::Basis{V}) where V
            bas = basis(a)*b
            order(a.v)+order(bas)>diffmode(V) ? zero(V) : SBlade{V}(a.v,bas)
        end
        function *(a::Basis{V},b::$Blade{V}) where V
            bas = a*basis(b)
            order(b.v)+order(bas)>diffmode(V) ? zero(V) : SBlade{V}(b.v,bas)
        end
    end
end

#*(a::MultiGrade{V},b::Basis{V}) where V = MultiGrade{V}(a.v,basis(a)*b)
#*(a::Basis{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*basis(b))
#*(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,basis(a)*basis(b))

for Blade ∈ MSB
    @eval begin
        *(a::UniformScaling,b::$Blade{V}) where V = V(a)*b
        *(a::$Blade{V},b::UniformScaling) where V = a*V(b)
    end
end
for Chain ∈ MSC
    @eval begin
        *(a::UniformScaling,b::$Chain{T,V} where T) where V = V(a)*b
        *(a::$Chain{T,V} where T,b::UniformScaling) where V = a*V(b)
    end
end

export ∗, ⊛, ⊖
const ⊖ = *

"""
    ∗(a::TensorAlgebra,b::TensorAlgebra)

Reversed geometric product: a∗b = (~a)*b
"""
@inline ∗(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = (~a)*b

## exterior product

export ∧, ∨, ⊗

@pure function ∧(a::Basis{V},b::Basis{V}) where V
    ba,bb = bits(a),bits(b)
    A,B,Q,Z = symmetricmask(V,ba,bb)
    ((count_ones(A&B)>0) || diffcheck(V,ba,bb)) && (return g_zero(V))
    d = getbasis(V,(A⊻B)|Q)
    diffvars(V)≠0 && !iszero(Z) && (d = SBlade{V}(getbasis(V,Z),d))
    return parity(a,b) ? SBlade{V}(-1,d) : d
end

function ∧(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    x,y = basis(a), basis(b)
    ba,bb = bits(x),bits(y)
    A,B,Q,Z = symmetricmask(V,ba,bb)
    ((count_ones(A&B)>0) || diffcheck(V,ba,bb)) && (return g_zero(V))
    v = value(a)*value(b)
    if diffvars(V)≠0 && !iszero(Z)
        v=typeof(v)<:TensorMixed ? SBlade{V}(getbasis(V,Z),v) : SBlade{V}(v,getbasis(V,Z))
        count_ones(Q)+order(v)>diffmode(V) && (return zero(V))
    end
    return SBlade{V}(parity(x,y) ? -v : v,getbasis(V,(A⊻B)|Q))
end

#∧(a::MultiGrade{V},b::Basis{V}) where V = MultiGrade{V}(a.v,basis(a)*b)
#∧(a::Basis{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*basis(b))
#∧(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,basis(a)*basis(b))

"""
    ∧(ω::TensorAlgebra,η::TensorAlgebra)

Exterior product as defined by the anti-symmetric quotient Λ≡⊗/~
"""
@inline ∧(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∧,a,b)
@inline ∧(a::TensorAlgebra{V},b::UniformScaling{T}) where {V,T<:Field} = a∧V(b)
@inline ∧(a::UniformScaling{T},b::TensorAlgebra{V}) where {V,T<:Field} = V(a)∧b

⊗(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a∧b

## regressive product: (L = grade(a) + grade(b); (-1)^(L*(L-ndims(V)))*⋆(⋆(a)∧⋆(b)))

@pure function ∨(a::Basis{V},b::Basis{V}) where V
    p,C,t,Z = regressive(a,b)
    !t && (return g_zero(V))
    d = getbasis(V,C)
    diffvars(V)≠0 && !iszero(Z) && (d = SBlade{V}(getbasis(V,Z),d))
    return p ? SBlade{V}(-1,d) : d
end

function ∨(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    p,C,t,Z = regressive(bits(basis(a)),bits(basis(b)),V)
    !t && (return g_zero(V))
    v = value(a)*value(b)
    if diffvars(V)≠0 && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,bits(basis(a)),bits(basis(b)))
        v=typeof(v)<:TensorMixed ? SBlade{V}(getbasis(V,Z),v) : SBlade{V}(v,getbasis(V,Z))
        count_ones(Q)+order(v)>diffmode(V) && (return zero(V))
    end
    return SBlade{V}(p ? -v : v,getbasis(V,C))
end

"""
    ∨(ω::TensorAlgebra,η::TensorAlgebra)

Regressive product as defined by the DeMorgan's law: ∨(ω...) = ⋆⁻¹(∧(⋆.(ω)...))
"""
@inline ∨(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∨,a,b)
@inline ∨(a::TensorAlgebra{V},b::UniformScaling{T}) where {V,T<:Field} = a∨V(b)
@inline ∨(a::UniformScaling{T},b::TensorAlgebra{V}) where {V,T<:Field} = V(a)∨b

"""
    ∨(ω::TensorAlgebra,η::TensorAlgebra)

Regressive product as defined by the DeMorgan's law: ∨(ω...) = ⋆⁻¹(∧(⋆.(ω)...))
"""
Base.:&(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a∨b

## interior product: a ∨ ⋆(b)

import LinearAlgebra: dot, ⋅
export ⋅

"""
    contraction(ω::TensorAlgebra,η::TensorAlgebra)

Interior (right) contraction product: ω⋅η = ω∨⋆η
"""
@pure function contraction(a::Basis{V},b::Basis{V}) where V
    g,C,t,Z = interior(a,b)
    !t && (return g_zero(V))
    d = getbasis(V,C)
    diffvars(V)≠0 && !iszero(Z) && (d = SBlade{V}(getbasis(V,Z),d))
    return typeof(V) <: Signature ? (g ? SBlade{V}(-1,d) : d) : SBlade{V}(g,d)
end

function contraction(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    g,C,t,Z = interior(bits(basis(a)),bits(basis(b)),V)
    !t && (return g_zero(V))
    v = value(a)*value(b)
    if diffvars(V)≠0 && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,bits(basis(a)),bits(basis(b)))
        v=typeof(v)<:TensorMixed ? SBlade{V}(getbasis(V,Z),v) : SBlade{V}(v,getbasis(V,Z))
        count_ones(Q)+order(v)>diffmode(V) && (return zero(V))
    end
    return SBlade{V}(typeof(V) <: Signature ? (g ? -v : v) : g*v,getbasis(V,C))
end

export ⨼, ⨽

for T ∈ (:TensorTerm,MSC...)
    @eval @inline Base.abs2(t::T) where T<:$T = contraction(t,t)
end

"""
    dot(ω::TensorAlgebra,η::TensorAlgebra)

Interior (right) contraction product: ω⋅η = ω∨⋆η
"""
@inline dot(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = contraction(a,b)

#=for A ∈ (:TensorTerm,MSC...), B ∈ (:TensorTerm,MSC...)
    @eval contraction(a::A,b::B) where {A<:$A,B<:$B} where V = contraction(a,b)
end=#

## cross product

import LinearAlgebra: cross
export ×

"""
    cross(ω::TensorAlgebra,η::TensorAlgebra)

Cross product: ω×η = ⋆(ω∧η)
"""
cross(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = ⋆(a∧b)

@pure function cross(a::Basis{V},b::Basis{V}) where V
    p,C,t,Z = crossprod(a,b)
    !t && (return zero(V))
    d = getbasis(V,C)
    diffvars(V)≠0 && !iszero(Z) && (d = SBlade{V}(getbasis(V,Z),d))
    return p ? SBlade{V}(-1,d) : d
end

function cross(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    p,C,t,Z = crossprod(bits(basis(a)),bits(basis(b)),V)
    !t && (return zero(V))
    v = value(a)*value(b)
    if diffvars(V)≠0 && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,bits(basis(a)),bits(basis(b)))
        v=typeof(v)<:TensorMixed ? SBlade{V}(getbasis(V,Z),v) : SBlade{V}(v,getbasis(V,Z))
        count_ones(Q)+order(v)>diffmode(V) && (return zero(V))
    end
    return SBlade{V}(p ? -v : v,getbasis(V,C))
end

# symmetrization and anti-symmetrization

export ⊙, ⊠

"""
    ⊙(ω::TensorAlgebra,η::TensorAlgebra)

Symmetrization projection: ⊙(ω...) = ∑(∏(σ.(ω)...))/factorial(length(ω))
"""
⊙(x::TensorAlgebra...) = (K=length(x); sum([prod(x[k]) for k ∈ permutations(1:K)])/factorial(K))

"""
    ⊠(ω::TensorAlgebra,η::TensorAlgebra)

Anti-symmetrization projection: ⊠(ω...) = ∑(∏(πσ.(ω)...))/factorial(length(ω))
"""
function ⊠(x::TensorAlgebra...)
    K,V,out = length(x),∪(vectorspace.(x)...),prod(x)
    P,F = collect(permutations(1:K)),factorial(K)
    for n ∈ 2:F
        p = prod(x[P[n]])
        DirectSum.indexparity!(P[n],V)[1] ? (out-=p) : (out+=p)
    end
    return out/F
end

"""
    ⊙(ω::TensorAlgebra,η::TensorAlgebra)

Symmetrization projection: ⊙(ω...) = ∑(∏(σ.(ω)...))/factorial(length(ω))
"""
<<(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a⊙b

"""
    ⊠(ω::TensorAlgebra,η::TensorAlgebra)

Anti-symmetrization projection: ⊠(ω...) = ∑(∏(πσ.(ω)...))/factorial(length(ω))
"""
>>(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a⊠b

## sandwich product

export ⊘

"""
    ⊘(ω::TensorAlgebra,η::TensorAlgebra)

Sandwich product: ω⊘η = (~ω)⊖η⊖ω
"""
⊘(x::TensorAlgebra{V},y::TensorAlgebra{V}) where V = (x ∗ y) * x

"""
    ⊘(ω::TensorAlgebra,η::TensorAlgebra)

Sandwich product: ω>>>η = ω⊖η⊖(~ω)
"""
>>>(x::TensorAlgebra{V},y::TensorAlgebra{V}) where V = x * y * ~x

### Product Algebra Constructor

function generate_product_algebra(Field=Field,VEC=:mvec,MUL=:*,ADD=:+,SUB=:-,CONJ=:conj)
    if Field == Grassmann.Field
        declare_mutating_operations(:(MArray{Tuple{M},T,1,M}),Number,Expr,:-,:*)
    elseif Field ∈ (SymField,:(SymPy.Sym))
        declare_mutating_operations(:(SizedArray{Tuple{M},T,1,1}),Field,set_val,SUB,MUL)
    end
    Field == :(SymPy.Sym) && for par ∈ (:parany,:parval,:parsym)
        @eval $par = ($par...,$Field)
    end
    TF = Field ∉ Fields ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    @eval begin
        @inline function inneraddvalue!(mv::MBlade{V,0,B,T} where {W,B},α,β,γ::T) where {V,T<:$Field}
            if γ≠0
                g,C,f,Z = interior(α,β,V)
                !iszero(C) && T≠Any && (return true)
                v = iszero(C) ? γ : γ*getbasis(V,C)
                if diffvars(V)≠0
                    if !iszero(Z)
                        _,_,Q,_ = symmetricmask(V,α,β)
                        v *= getbasis(V,Z)
                    end
                    order(v)>diffmode(V) && (return false)
                end
                f && (mv.v = typeof(V)<:Signature ? (g ? $SUB(mv.v,v) : $ADD(mv.v,v)) : $ADD(mv.v,$MUL(g,v)))
                return false
            end
            return false
        end
        function adjoint(m::MultiVector{T,V}) where {T<:$Field,V}
            if mixedmode(V)<0
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
        *(a::F,b::Basis{V}) where {F<:$EF,V} = SBlade{V}(a,b)
        *(a::Basis{V},b::F) where {F<:$EF,V} = SBlade{V}(b,a)
        *(a::F,b::MultiVector{T,V}) where {F<:$Field,T<:$Field,V} = MultiVector{promote_type(T,F),V}(broadcast($MUL,a,b.v))
        *(a::MultiVector{T,V},b::F) where {F<:$Field,T<:$Field,V} = MultiVector{promote_type(T,F),V}(broadcast($MUL,a.v,b))
        *(a::F,b::MultiGrade{V}) where {F<:$EF,V} = MultiGrade{V}(broadcast($MUL,a,b.v))
        *(a::MultiGrade{V},b::F) where {F<:$EF,V} = MultiGrade{V}(broadcast($MUL,a.v,b))
        ∧(a::$Field,b::$Field) = $MUL(a,b)
        ∧(a::F,b::B) where B<:TensorTerm{V,G} where {F<:$EF,V,G} = SBlade{V,G}(a,b)
        ∧(a::A,b::F) where A<:TensorTerm{V,G} where {F<:$EF,V,G} = SBlade{V,G}(b,a)
        #=∧(a::$Field,b::MultiVector{T,V}) where {T<:$Field,V} = MultiVector{T,V}(a.*b.v)
        ∧(a::MultiVector{T,V},b::$Field) where {T<:$Field,V} = MultiVector{T,V}(a.v.*b)
        ∧(a::$Field,b::MultiGrade{V}) where V = MultiGrade{V}(a.*b.v)
        ∧(a::MultiGrade{V},b::$Field) where V = MultiGrade{V}(a.v.*b)=#
    end
    for Blade ∈ MSB
        @eval begin
            adjoint(b::$Blade{V,G,B,T}) where {V,G,B,T<:$Field} = $Blade{dual(V),G,B',$TF}($CONJ(value(b)))
            *(a::F,b::$Blade{V,G,B,T} where B) where {F<:$Field,V,G,T<:$Field} = SBlade{V,G}($MUL(a,b.v),basis(b))
            *(a::$Blade{V,G,B,T} where B,b::F) where {F<:$Field,V,G,T<:$Field} = SBlade{V,G}($MUL(a.v,b),basis(a))
        end
    end
    for (A,B) ∈ [(A,B) for A ∈ MSB, B ∈ MSB]
        @eval function *(a::$A{V,G,A,T} where {V,G,A},b::$B{W,L,B,S} where {W,L,B}) where {T<:$Field,S<:$Field}
            SBlade($MUL(a.v,b.v),basis(a)*basis(b))
        end
    end
    for Chain ∈ MSC
        @eval begin
            function adjoint(m::$Chain{T,V,G}) where {T<:$Field,V,G}
                if mixedmode(V)<0
                    $(insert_expr((:N,:M,:ib),VEC)...)
                    out = zeros($VEC(N,G,$TF))
                    for i ∈ 1:binomial(N,G)
                        @inbounds setblade!(out,$CONJ(m.v[i]),dual(V,ib[i],M),Dimension{N}())
                    end
                else
                    out = $CONJ.(value(m))
                end
                $Chain{$TF,dual(V),G}(out)
            end
            *(a::F,b::$Chain{T,V,G}) where {F<:$Field,T<:$Field,V,G} = SChain{promote_type(T,F),V,G}(broadcast($MUL,a,b.v))
            *(a::$Chain{T,V,G},b::F) where {F<:$Field,T<:$Field,V,G} = SChain{promote_type(T,F),V,G}(broadcast($MUL,a.v,b))
            #∧(a::$Field,b::$Chain{T,V,G}) where {T<:$Field,V,G} = SChain{T,V,G}(a.*b.v)
            #∧(a::$Chain{T,V,G},b::$Field) where {T<:$Field,V,G} = SChain{T,V,G}(a.v.*b)
            function contraction(a::$Chain{T,V,G},b::Basis{V,G}) where {T<:$Field,V,G}
                $(insert_expr((:N,:t,:mv,:ib,:μ),VEC)...)
                for i ∈ 1:binomial(N,G)
                    if @inbounds inneraddvalue!(mv,ib[i],bits(b),a[i])&μ
                        $(insert_expr((:mv,);mv=:(value(mv)))...)
                        @inbounds inneraddvalue!(mv,ib[i],bits(b),a[i])
                    end
                end
                return value_diff(mv)
            end
            function contraction(a::Basis{V,G},b::$Chain{T,V,G}) where {V,T<:$Field,G}
                $(insert_expr((:N,:t,:mv,:ib,:μ),VEC)...)
                for i ∈ 1:binomial(N,G)
                    if @inbounds inneraddvalue!(mv,bits(a),ib[i],b[i])&μ
                        $(insert_expr((:mv,);mv=:(value(mv)))...)
                        @inbounds inneraddvalue!(mv,bits(a),ib[i],b[i])
                    end
                end
                return value_diff(mv)
            end
            function ∧(a::$Chain{T,w,1},b::Basis{W,1}) where {T<:$Field,w,W}
                V = w==W ? w : ((w==dual(W)) ? (mixedmode(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                $(insert_expr((:N,:t,:μ),VEC)...)
                ib = indexbasis(ndims(w),1)
                out = zeros($VEC(N,2,t))
                C,y = mixedmode(w)>0,mixedmode(W)>0 ? dual(V,bits(b)) : bits(b)
                for i ∈ 1:length(a)
                    if @inbounds outeraddblade!(V,out,C ? dual(V,ib[i]) : ib[i],y,a[i])&μ
                        out,t = zeros(svec(N,2,Any)) .+ out,Any
                        @inbounds outeraddblade!(V,out,C ? dual(V,ib[i]) : ib[i],y,a[i])
                    end
                end
                return MChain{t,V,2}(out)
            end
            function ∧(a::Basis{w,1},b::$Chain{T,W,1}) where {w,W,T<:$Field}
                V = w==W ? w : ((w==dual(W)) ? (mixedmode(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                $(insert_expr((:N,:t,:μ),VEC)...)
                ib = indexbasis(ndims(W),1)
                out = zeros($VEC(N,2,t))
                C,x = mixedmode(W)>0,mixedmode(w)>0 ? dual(V,bits(a)) : bits(a)
                for i ∈ 1:length(b)
                    if @inbounds outeraddblade!(V,out,x,C ? dual(V,ib[i]) : ib[i],b[i])&μ
                        out,t = zeros(svec(N,2,Any)) .+ out,Any
                        @inbounds outeraddblade!(V,out,x,C ? dual(V,ib[i]) : ib[i],b[i])
                    end
                end
                return MChain{t,V,2}(out)
            end
        end
        for Blade ∈ MSB
            @eval begin
                function contraction(a::$Chain{T,V,G},b::$Blade{V,G,B,S}) where {T<:$Field,V,G,B,S<:$Field}
                    $(insert_expr((:N,:t,:mv,:ib,:μ),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        if @inbounds inneraddvalue!(mv,ib[i],bits(basis(b)),$MUL(a[i],b.v))&μ
                            $(insert_expr((:mv,);mv=:(value(mv)))...)
                            @inbounds inneraddvalue!(mv,ib[i],bits(basis(b)),$MUL(a[i],b.v))
                        end

                    end
                    return value_diff(mv)
                end
                function contraction(a::$Blade{V,G,B,S},b::$Chain{T,V,G}) where {T<:$Field,V,G,B,S<:$Field}
                    $(insert_expr((:N,:t,:mv,:ib,:μ),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        if @inbounds inneraddvalue!(mv,bits(basis(a)),ib[i],$MUL(a.v,b[i]))&μ
                            $(insert_expr((:mv,);mv=:(value(mv)))...)
                            @inbounds inneraddvalue!(mv,bits(basis(a)),ib[i],$MUL(a.v,b[i]))
                        end
                    end
                    return value_diff(mv)
                end
                function ∧(a::$Chain{T,w,1},b::$Blade{W,1,B,S}) where {T<:$Field,w,W,B,S<:$Field}
                    V = w==W ? w : ((w==dual(W)) ? (mixedmode(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                    $(insert_expr((:N,:t,:μ),VEC)...)
                    ib = indexbasis(ndims(w),1)
                    out = zeros($VEC(N,2,t))
                    C,y = mixedmode(w)>0,mixedmode(W)>0 ? dual(V,bits(basis(b))) : bits(basis(b))
                    for i ∈ 1:length(a)
                        if @inbounds outeraddblade!(V,out,C ? dual(V,ib[i]) : ib[i],y,$MUL(a[i],b.v))&μ
                            out,t = zeros(svec(N,2,Any)) .+ out,Any
                            @inbounds outeraddblade!(V,out,C ? dual(V,ib[i]) : ib[i],y,$MUL(a[i],b.v))
                        end
                    end
                    return MChain{t,V,2}(out)
                end
                function ∧(a::$Blade{w,1,B,S},b::$Chain{T,W,1}) where {T<:$Field,w,W,B,S<:$Field}
                    V = w==W ? w : ((w==dual(W)) ? (mixedmode(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                    $(insert_expr((:N,:t,:μ),VEC)...)
                    ib = indexbasis(ndims(W),1)
                    out = zeros($VEC(N,2,t))
                    C,x = mixedmode(W)>0,mixedmode(w)>0 ? dual(V,bits(basis(a))) : bits(basis(a))
                    for i ∈ 1:length(b)
                        if @inbounds outeraddblade!(V,out,x,C ? dual(V,ib[i]) : ib[i],$MUL(a.v,b[i]))&μ
                            out,t = zeros(svec(N,2,Any)) .+ out,Any
                            @inbounds outeraddblade!(V,out,x,C ? dual(V,ib[i]) : ib[i],$MUL(a.v,b[i]))
                        end
                    end
                    return MChain{t,V,2}(out)
                end
            end
        end
    end
    for Chain ∈ MSC, Other ∈ MSC
        @eval begin
            function contraction(a::$Chain{T,V,G},b::$Other{S,V,G}) where {T<:$Field,V,G,S<:$Field}
                $(insert_expr((:N,:t,:mv,:bng,:ib,:μ),VEC)...)
                for i ∈ 1:bng
                    @inbounds v,ibi = a[i],ib[i]
                    v≠0 && for j ∈ 1:bng
                        if @inbounds inneraddvalue!(mv,ibi,ib[j],$MUL(v,b[j]))&μ
                            $(insert_expr((:mv,);mv=:(value(mv)))...)
                            @inbounds inneraddvalue!(mv,ibi,ib[j],$MUL(v,b[j]))
                        end
                    end
                end
                return value_diff(mv)
            end
            function ∧(a::$Chain{T,w,1},b::$Other{S,W,1}) where {T<:$Field,w,S<:$Field,W}
                V = w==W ? w : ((w==dual(W)) ? (mixedmode(w)≠0 ? W+w : w+W) : (return interop(∧,a,b)))
                $(insert_expr((:N,:t,:μ),VEC)...)
                ia = indexbasis(ndims(w),1)
                ib = indexbasis(ndims(W),1)
                out = zeros($VEC(N,2,t))
                CA,CB = mixedmode(w)>0,mixedmode(W)>0
                for i ∈ 1:length(a)
                    @inbounds v,iai = a[i],ia[i]
                    x = CA ? dual(V,iai) : iai
                    v≠0 && for j ∈ 1:length(b)
                        if @inbounds outeraddblade!(V,out,x,CB ? dual(V,ib[j]) : ib[j],$MUL(v,b[j]))&μ
                            out,t = zeros(svec(N,2,promote_type,Any)) .+ out,Any
                            @inbounds outeraddblade!(V,out,x,CB ? dual(V,ib[j]) : ib[j],$MUL(v,b[j]))
                        end
                    end
                end
                return MChain{t,V,2}(out)
            end
        end
    end
    for side ∈ (:left,:right)
        c = Symbol(:complement,side)
        p = Symbol(:parity,side)
        for Chain ∈ MSC
            @eval begin
                function $c(b::$Chain{T,V,G}) where {T<:$Field,V,G}
                    mixedmode(V)<0 && throw(error("Complement for mixed tensors is undefined"))
                    $(insert_expr((:N,:ib,:D),VEC)...)
                    out = zeros($VEC(N,G,T))
                    D = diffvars(V)
                    for k ∈ 1:binomial(N,G)
                        @inbounds val = b.v[k]
                        if val≠0
                            @inbounds p = $p(V,ib[k])
                            v = typeof(V)<:Signature ? (p ? $SUB(val) : val) : p*val
                            @inbounds setblade!(out,v,complement(N,ib[k],D),Dimension{N}())
                        end
                    end
                    return $Chain{T,V,N-G}(out)
                end
            end
        end
        @eval begin
            function $c(m::MultiVector{T,V}) where {T<:$Field,V}
                mixedmode(V)<0 && throw(error("Complement for mixed tensors is undefined"))
                $(insert_expr((:N,:bs,:bn),VEC)...)
                out = zeros($VEC(N,T))
                D = diffvars(V)
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
        for Chain ∈ MSC
            @eval begin
                function $reverse(b::$Chain{T,V,G}) where {T<:$Field,V,G}
                    D = diffvars(V)
                    D==0 && !$p(G) && (return b)
                    $(insert_expr((:N,:ib),VEC)...)
                    out = zeros($VEC(N,G,T))
                    for k ∈ 1:binomial(N,G)
                        @inbounds v = b.v[k]
                        v≠0 && if D==0
                            @inbounds setblade!(out,$SUB(v),ib[k],Dimension{N}())
                        else
                            @inbounds B = ib[k]
                            setblade!(out,$p(grade(V,B)) ? $SUB(v) : v,B,Dimension{N}())
                        end
                    end
                    return $Chain{T,V,G}(out)
                end
            end
        end
        @eval begin
            function $reverse(m::MultiVector{T,V}) where {T<:$Field,V}
                $(insert_expr((:N,:bs,:bn,:D),VEC)...)
                out = zeros($VEC(N,T))
                for g ∈ 1:N+1
                    pg = $p(g-1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds v = m.v[bs[g]+i]
                        v≠0 && if D==0
                            @inbounds setmulti!(out,pg ? $SUB(v) : v,ib[i],Dimension{N}())
                        else
                            @inbounds B = ib[i]
                            setmulti!(out,$p(grade(V,B)) ? $SUB(v) : v,B,Dimension{N}())
                        end
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
                $(insert_expr((:N,:t,:out,:bs,:bn,:μ),VEC)...)
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        if @inbounds $product!(V,out,ib[i],bits(b),a.v[bs[g]+i])&μ
                            $(insert_expr((:out,);mv=:out)...)
                            @inbounds $product!(V,out,ib[i],bits(b),a.v[bs[g]+i])
                        end
                    end
                end
                return MultiVector{t,V}(out)
            end
            function $op(a::Basis{V,G},b::MultiVector{T,V}) where {V,G,T<:$Field}
                $(insert_expr((:N,:t,:out,:bs,:bn,:μ),VEC)...)
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        if @inbounds $product!(V,out,bits(a),ib[i],b.v[bs[g]+i])&μ
                            $(insert_expr((:out,);mv=:out)...)
                            @inbounds $product!(V,out,bits(a),ib[i],b.v[bs[g]+i])
                        end
                    end
                end
                return MultiVector{t,V,2^N}(out)::MultiVector{t,V,2^N}
            end
            function $op(a::MultiVector{T,V},b::MultiVector{S,V}) where {V,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t,:out,:bs,:bn,:μ),VEC)...)
                for g ∈ 1:N+1
                    Y = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds val = b.v[bs[g]+i]
                        val≠0 && for G ∈ 1:N+1
                            @inbounds R = bs[G]
                            X = indexbasis(N,G-1)
                            @inbounds for j ∈ 1:bn[G]
                                if @inbounds $product!(V,out,X[j],Y[i],$MUL(a.v[R+j],val))&μ
                                    $(insert_expr((:out,);mv=:out)...)
                                    @inbounds $product!(V,out,X[j],Y[i],$MUL(a.v[R+j],val))
                                end
                            end
                        end
                    end
                end
                return MultiVector{t,V}(out)
            end
        end
        for Blade ∈ MSB
            @eval begin
                function $op(a::MultiVector{T,V},b::$Blade{V,G,B,S}) where {T<:$Field,V,G,B,S<:$Field}
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),VEC)...)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            if @inbounds $product!(V,out,ib[i],bits(basis(b)),$MUL(a.v[bs[g]+i],b.v))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $product!(V,out,ib[i],bits(basis(b)),$MUL(a.v[bs[g]+i],b.v))
                            end
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Blade{V,G,B,T},b::MultiVector{S,V}) where {V,G,B,T<:$Field,S<:$Field}
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),VEC)...)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            if @inbounds $product!(V,out,bits(basis(a)),ib[i],$MUL(a.v,b.v[bs[g]+i]))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $product!(V,out,bits(basis(a)),ib[i],$MUL(a.v,b.v[bs[g]+i]))
                            end
                        end
                    end
                    return MultiVector{t,V,2^N}(out)
                end
            end
        end
        for Chain ∈ MSC
            @eval begin
                function $op(a::$Chain{T,V,G},b::Basis{V}) where {T<:$Field,V,G}
                    $(insert_expr((:N,:t,:out,:ib,:μ),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        if @inbounds $product!(V,out,ib[i],bits(b),a[i])&μ
                            $(insert_expr((:out,);mv=:out)...)
                            @inbounds $product!(V,out,ib[i],bits(b),a[i])
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::Basis{V},b::$Chain{T,V,G}) where {V,T<:$Field,G}
                    $(insert_expr((:N,:t,:out,:ib,:μ),VEC)...)
                    for i ∈ 1:binomial(N,G)
                        if @inbounds $product!(V,out,bits(a),ib[i],b[i])&μ
                            $(insert_expr((:out,);mv=:out)...)
                            @inbounds $product!(V,out,bits(a),ib[i],b[i])
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::MultiVector{T,V},b::$Chain{S,V,G}) where {T<:$Field,V,S<:$Field,G}
                    $(insert_expr((:N,:t,:out,:bng,:ib,:bs,:bn,:μ),VEC)...)
                    for g ∈ 1:N+1
                        A = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = a.v[bs[g]+i]
                            val≠0 && for j ∈ 1:bng
                                if @inbounds $product!(V,out,A[i],ib[j],$MUL(val,b[j]))&μ
                                    $(insert_expr((:out,);mv=:out)...)
                                    @inbounds $product!(V,out,A[i],ib[j],$MUL(val,b[j]))
                                end
                            end
                        end
                    end
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Chain{T,V,G},b::MultiVector{S,V}) where {V,G,S<:$Field,T<:$Field}
                    $(insert_expr((:N,:t,:out,:bng,:ib,:bs,:bn,:μ),VEC)...)
                    for g ∈ 1:N+1
                        B = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = b.v[bs[g]+i]
                            val≠0 && for j ∈ 1:bng
                                if @inbounds $product!(V,out,ib[j],B[i],$MUL(a[j],val))&μ
                                    $(insert_expr((:out,);mv=:out)...)
                                    @inbounds $product!(V,out,ib[j],B[i],$MUL(a[j],val))
                                end
                            end
                        end
                    end
                    return MultiVector{t,V}(out)
                end
            end
            for Blade ∈ MSB
                @eval begin
                    function $op(a::$Chain{T,V,G},b::$Blade{V,L,B,S}) where {T<:$Field,V,G,L,B,S<:$Field}
                        $(insert_expr((:N,:t,:out,:ib,:μ),VEC)...)
                        for i ∈ 1:binomial(N,G)
                            if @inbounds $product!(V,out,ib[i],bits(basis(b)),$MUL(a[i],b.v))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $product!(V,out,ib[i],bits(basis(b)),$MUL(a[i],b.v))
                            end
                        end
                        return MultiVector{t,V}(out)
                    end
                    function $op(a::$Blade{V,L,B,S},b::$Chain{T,V,G}) where {T<:$Field,V,G,L,B,S<:$Field}
                        $(insert_expr((:N,:t,:out,:ib,:μ),VEC)...)
                        for i ∈ 1:binomial(N,G)
                            if @inbounds $product!(V,out,bits(basis(a)),ib[i],$MUL(a.v,b[i]))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $product!(V,out,bits(basis(a)),ib[i],$MUL(a.v,b[i]))
                            end
                        end
                        return MultiVector{t,V}(out)
                    end
                end
            end
        end
        for (A,B) ∈ [(A,B) for A ∈ MSC, B ∈ MSC]
            @eval begin
                function $op(a::$A{T,V,G},b::$B{S,V,L}) where {T<:$Field,V,G,S<:$Field,L}
                    $(insert_expr((:N,:t,:bnl,:ib,:μ),VEC)...)
                    out = zeros($VEC(N,t))
                    B = indexbasis(N,L)
                    for i ∈ 1:binomial(N,G)
                        @inbounds v,ibi = a[i],ib[i]
                        v≠0 && for j ∈ 1:bnl
                            if @inbounds $product!(V,out,ibi,B[j],$MUL(v,b[j]))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $product!(V,out,ibi,B[j],$MUL(v,b[j]))
                            end
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
        for (Blade,Other) ∈ [(a,b) for a ∈ MSB, b ∈ MSB]
            @eval begin
                function $op(a::$Blade{V,A,X,T},b::$Other{V,B,Y,S}) where {V,A,X,T<:$Field,B,Y,S<:$Field}
                    if X == Y
                        return SBlade{V,A}($bop(value(a),value(b)),X)
                    elseif A == B
                        $(insert_expr((:N,:t),VEC)...)
                        out = zeros($VEC(N,A,t))
                        setblade!(out,value(a,t),bits(X),Dimension{N}())
                        setblade!(out,$bop(value(b,t)),bits(Y),Dimension{N}())
                        return MChain{t,V,A}(out)
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
        for Blade ∈ MSB
            @eval begin
                $op(a::$Blade{V,G,B,T}) where {V,G,B,T<:$Field} = $Blade{V,G,B,$TF}($bop(value(a)))
                function $op(a::$Blade{V,A,X,T},b::Basis{V,B,Y}) where {V,A,X,T<:$Field,B,Y}
                    if X == b
                        return SBlade{V,A}($bop(value(a),value(b)),b)
                    elseif A == B
                        $(insert_expr((:N,:t),VEC)...)
                        out = zeros($VEC(N,A,t))
                        setblade!(out,value(a,t),bits(X),Dimension{N}())
                        setblade!(out,$bop(value(b,t)),Y,Dimension{N}())
                        return MChain{t,V,A}(out)
                    else
                        #@warn("sparse MultiGrade{V} objects not properly handled yet")
                        #return MultiGrade{V}(a,b)
                        $(insert_expr((:N,:t,:out),VEC)...)
                        setmulti!(out,value(a,t),bits(X),Dimension{N}())
                        setmulti!(out,$bop(value(b,t)),Y,Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                end
                function $op(a::Basis{V,A,X},b::$Blade{V,B,Y,S}) where {V,A,X,B,Y,S<:$Field}
                    if a == Y
                        return SBlade{V,A}($bop(value(a),value(b)),a)
                    elseif A == B
                        $(insert_expr((:N,:t),VEC)...)
                        out = zeros($VEC(N,A,t))
                        setblade!(out,value(a,t),X,Dimension{N}())
                        setblade!(out,$bop(value(b,t)),bits(Y),Dimension{N}())
                        return MChain{t,V,A}(out)
                    else
                        #@warn("sparse MultiGrade{V} objects not properly handled yet")
                        #return MultiGrade{V}(a,b)
                        $(insert_expr((:N,:t,:out),VEC)...)
                        setmulti!(out,value(a,t),X,Dimension{N}())
                        setmulti!(out,$bop(value(b,t)),bits(Y),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                end
                function $op(a::$Blade{V,G,A,S} where A,b::MultiVector{T,V}) where {T<:$Field,V,G,S<:$Field}
                    $(insert_expr((:N,:t),VEC)...)
                    out = $(bcast(bop,:(copy(value(b,$VEC(N,t))),)))
                    addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
                function $op(a::MultiVector{T,V},b::$Blade{V,G,B,S} where B) where {T<:$Field,V,G,S<:$Field}
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
            function $op(a::SparseChain{V,G},b::MultiVector{T,V}) where {T<:$Field,V,G}
                $(insert_expr((:N,),VEC)...)
                at = terms(a)
                t = promote_type(T,valuetype.(at)...)
                out = $(bcast(bop,:(copy(value(b,$VEC(N,t))),)))
                for A ∈ at
                    addmulti!(out,value(A,t),bits(A),Dimension{N}())
                end
                return MultiVector{t,V}(out)
            end
            function $op(a::MultiVector{T,V},b::SparseChain{V,G}) where {T<:$Field,V,G}
                $(insert_expr((:N,),VEC)...)
                bt = terms(b)
                t = promote_type(T,valuetype.(bt)...)
                out = copy(value(a,$VEC(N,t)))
                for B ∈ bt
                    addmulti!(out,$bop(value(B,t)),bits(B),Dimension{N}())
                end
                return MultiVector{t,V}(out)
            end
            function $op(a::MultiGrade{V,G},b::MultiVector{T,V}) where {T<:$Field,V,G}
                $(insert_expr((:N,),VEC)...)
                at = terms(a)
                t = promote_type(T,valuetype.(at)...)
                out = $(bcast(bop,:(copy(value(b,$VEC(N,t))),)))
                for A ∈ at
                    TA = typeof(A)
                    if TA <: TensorTerm
                        addmulti!(out,value(A,t),bits(A),Dimension{N}())
                    elseif TA <: SparseChain
                        for α ∈ terms(A)
                            addmulti!(out,value(α,t),bits(α),Dimension{N}())
                        end
                    elseif TA <: TensorMixed
                        g = grade(A)
                        r = binomsum(N,g)
                        @inbounds out[r+1:r+binomial(N,g)] += value(A,$VEC(N,g,t))
                    end
                end
                return MultiVector{t,V}(out)
            end
            function $op(a::MultiVector{T,V},b::MultiGrade{V,G}) where {T<:$Field,V,G}
                $(insert_expr((:N,),VEC)...)
                bt = terms(b)
                t = promote_type(T,valuetype.(bt)...)
                out = copy(value(a,$VEC(N,t)))
                for B ∈ bt
                    TB = typeof(B)
                    if TB <: TensorTerm
                        addmulti!(out,$bop(value(B,t)),bits(B),Dimension{N}())
                    elseif TB <: SparseChain
                        for β ∈ terms(B)
                            addmulti!(out,$bop(value(β,t)),bits(β),Dimension{N}())
                        end
                    elseif TB <: TensorMixed
                        g = grade(B)
                        r = binomsum(N,g)
                        @inbounds $(Expr(eop,:(out[r+1:r+binomial(N,g)]),:(value(B,$VEC(N,g,t)))))
                    end
                end
                return MultiVector{t,V}(out)
            end
        end
        for (A,B) ∈ [(A,B) for A ∈ MSC, B ∈ MSC]
            C = (A == MSC[1] && B == MSC[1]) ? MSC[1] : MSC[2]
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
        for Chain ∈ MSC
            for Blade ∈ MSB
                @eval begin
                    function $op(a::$Chain{T,V,G},b::$Blade{V,G,B,S} where B) where {T<:$Field,V,G,S<:$Field}
                        $(insert_expr((:N,:t),VEC)...)
                        out = copy(value(a,$VEC(N,G,t)))
                        addblade!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                        return MChain{t,V,G}(out)
                    end
                    function $op(a::$Blade{V,G,A,S} where A,b::$Chain{T,V,G}) where {T<:$Field,V,G,S<:$Field}
                        $(insert_expr((:N,:t),VEC)...)
                        out = $(bcast(bop,:(copy(value(b,$VEC(N,G,t))),)))
                        addblade!(out,value(a,t),basis(a),Dimension{N}())
                        return MChain{t,V,G}(out)
                    end
                    function $op(a::$Chain{T,V,G},b::$Blade{V,L,B,S} where B) where {T<:$Field,V,G,L,S<:$Field}
                        $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                        @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                        addmulti!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                    function $op(a::$Blade{V,L,A,S} where A,b::$Chain{T,V,G}) where {T<:$Field,V,G,L,S<:$Field}
                        $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                        @inbounds out[r+1:r+bng] = $(bcast(bop,:(value(b,$VEC(N,G,t)),)))
                        addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                        return MultiVector{t,V}(out)
                    end
                end
            end
            @eval begin
                $op(a::$Chain{T,V,G}) where {T<:$Field,V,G} = $Chain{$TF,V,G}($(bcast(bop,:(value(a),))))
                function $op(a::$Chain{T,V,G},b::Basis{V,G}) where {T<:$Field,V,G}
                    $(insert_expr((:N,:t),VEC)...)
                    out = copy(value(a,$VEC(N,G,t)))
                    addblade!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                    return MChain{t,V,G}(out)
                end
                function $op(a::Basis{V,G},b::$Chain{T,V,G}) where {T<:$Field,V,G}
                    $(insert_expr((:N,:t),VEC)...)
                    out = $(bcast(bop,:(copy(value(b,$VEC(N,G,t))),)))
                    addblade!(out,value(a,t),basis(a),Dimension{N}())
                    return MChain{t,V,G}(out)
                end
                function $op(a::$Chain{T,V,G},b::Basis{V,L}) where {T<:$Field,V,G,L}
                    $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                    @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                    addmulti!(out,$bop(value(b,t)),bits(basis(b)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
                function $op(a::Basis{V,L},b::$Chain{T,V,G}) where {T<:$Field,V,G,L}
                    $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                    @inbounds out[r+1:r+bng] = $(bcast(bop,:(value(b,$VEC(N,G,t)),)))
                    addmulti!(out,value(a,t),bits(basis(a)),Dimension{N}())
                    return MultiVector{t,V}(out)
                end
                function $op(a::$Chain{T,V,G},b::SparseChain{V,G}) where {T<:$Field,V,G}
                    $(insert_expr((:N,),VEC)...)
                    bt = terms(b)
                    t = promote_type(T,valuetype.(bt)...)
                    out = copy(value(a,$VEC(N,G,t)))
                    for B ∈ bt
                        addblade!(out,$bop(value(B,t)),bits(B),Dimension{N}())
                    end
                    return MChain{t,V,G}(out)
                end
                function $op(a::SparseChain{V,G},b::$Chain{T,V,G}) where {T<:$Field,V,G}
                    $(insert_expr((:N,),VEC)...)
                    at = terms(a)
                    t = promote_type(T,valuetype.(at)...)
                    out = $(bcast(bop,:(copy(value(b,$VEC(N,G,t))),)))
                    for A ∈ at
                        addblade!(out,value(A,t),basis(A),Dimension{N}())
                    end
                    return MChain{t,V,G}(out)
                end
                function $op(a::$Chain{T,V,G},b::MultiVector{S,V}) where {T<:$Field,V,G,S}
                    $(insert_expr((:N,:t,:r,:bng),VEC)...)
                    out = $(bcast(bop,:(copy(value(b,$VEC(N,t))),)))
                    @inbounds out[r+1:r+bng] += value(a,$VEC(N,G,t))
                    return MultiVector{t,V}(out)
                end
                function $op(a::MultiVector{T,V},b::$Chain{S,V,G}) where {T<:$Field,V,G,S}
                    $(insert_expr((:N,:t,:r,:bng),VEC)...)
                    out = copy(value(a,$VEC(N,t)))
                    @inbounds $(Expr(eop,:(out[r+1:r+bng]),:(value(b,$VEC(N,G,t)))))
                    return MultiVector{t,V}(out)
                end
            end
        end
    end
end

for F ∈ Fields
    @eval begin
        *(a::F,b::MultiVector{T,V}) where {F<:$F,T<:Number,V} = MultiVector{promote_type(T,F),V}(broadcast(*,a,b.v))
        *(a::MultiVector{T,V},b::F) where {F<:$F,T<:Number,V} = MultiVector{promote_type(T,F),V}(broadcast(*,a.v,b))
    end
    for Blade ∈ MSB
        @eval begin
            *(a::F,b::$Blade{V,G,B,T} where B) where {F<:$F,V,G,T<:Number} = SBlade{V,G}(*(a,b.v),basis(b))
            *(a::$Blade{V,G,B,T} where B,b::F) where {F<:$F,V,G,T<:Number} = SBlade{V,G}(*(a.v,b),basis(a))
        end
    end
    for Chain ∈ MSC
        @eval begin
            *(a::F,b::$Chain{T,V,G}) where {F<:$F,T<:Number,V,G} = SChain{promote_type(T,F),V,G}(broadcast(*,a,b.v))
            *(a::$Chain{T,V,G},b::F) where {F<:$F,T<:Number,V,G} = SChain{promote_type(T,F),V,G}(broadcast(*,a.v,b))
        end
    end
end

generate_product_algebra()
generate_product_algebra(Complex)
generate_product_algebra(SymField,:svec,:($Sym.:∏),:($Sym.:∑),:($Sym.:-),:($Sym.conj))

const NSE = Union{Symbol,Expr,<:Real,<:Complex}

for (op,eop) ∈ ((:+,:(+=)),(:-,:(-=)))
    for Term ∈ (:TensorTerm,:TensorMixed)
        @eval begin
            $op(a::T,b::NSE) where T<:$Term = iszero(b) ? a : $op(a,b*one(vectorspace(a)))
            $op(a::NSE,b::T) where T<:$Term = iszero(a) ? $op(b) : $op(a*one(vectorspace(b)),b)
        end
    end
    @eval begin
        $op(a::Basis{V,G,B} where G) where {V,B} = SBlade($op(value(a)),a)
        function $op(a::Basis{V,A},b::Basis{V,B}) where {V,A,B}
            if a == b
                return SBlade{V,A}($op(value(a),value(b)),basis(a))
            elseif A == B
                $(insert_expr((:N,:t))...)
                out = zeros(mvec(N,A,t))
                setblade!(out,value(a,t),bits(a),Dimension{N}())
                setblade!(out,$op(value(b,t)),bits(b),Dimension{N}())
                return MChain{t,V,A}(out)
            else
                #@warn("sparse MultiGrade{V} objects not properly handled yet")
                #return MultiGrade{V}(a,b)
                $(insert_expr((:N,:t,:out))...)
                setmulti!(out,value(a,t),bits(a),Dimension{N}())
                setmulti!(out,$op(value(b,t)),bits(b),Dimension{N}())
                return MultiVector{t,V}(out)
            end
        end
        function $op(a::SparseChain{V,G},b::SparseChain{V,G}) where {V,G}
            at,bt = terms(a),terms(b)
            isempty(at) && (return b)
            isempty(bt) && (return a)
            bl = length(bt)
            out = copy(at)
            N = ndims(V)
            i,k,bk = 0,1,basisindex(N,bits(out[1]))
            while i < bl
                k += 1
                i += 1
                bas = basisindex(N,bits(bt[i]))
                if bas == bk
                    $(Expr(eop,:(out[k-1]),:(bt[i])))
                    k < length(out) ? (bk = basisindex(N,bits(out[k]))) : (k -= 1)
                elseif bas<bk
                    insert!(out,k-1,bt[i])
                elseif k ≤ length(out)
                    bk = basisindex(N,bits(out[k]))
                    i -= 1
                else
                    insert!(out,k,bt[i])
                end
            end
            SparseChain{V,G}(out)
        end
        function $op(a::SparseChain{V,G},b::T) where T<:TensorTerm{V,G} where {V,G}
            N = ndims(V)
            bas = basisindex(N,bits(b))
            out = copy(terms(a))
            i,k,bk = 0,1,basisindex(N,bits(out[1]))
            while i < length(out)
                k += 1
                i += 1
                if bk == bas
                    $(Expr(eop,:(out[k-1]),:b))
                    break
                elseif bas<bk
                    insert!(out,k-1,b)
                    break
                elseif k ≤ length(out)
                    bk = basisindex(N,bits(out[k]))
                else
                    insert!(out,k,b)
                    break
                end
            end
            SparseChain{V,G}(out)
        end
        function $op(a::MultiGrade{V,A},b::MultiGrade{V,B}) where {V,A,B}
            at,bt = terms(a),terms(b)
            isempty(at) && (return b)
            isempty(bt) && (return a)
            bl = length(bt)
            out = copy(at)
            N = ndims(V)
            i,k,bk = 0,1,grade(out[1])
            while i < bl
                k += 1
                i += 1
                bas = grade(bt[i])
                if bas == bk
                    $(Expr(eop,:(out[k-1]),:(bt[i])))
                    k < length(out) ? (bk = grade(out[k])) : (k -= 1)
                elseif bas<bk
                    insert!(out,k-1,bt[i])
                elseif k ≤ length(out)
                    bk = grade(out[k])
                    i -= 1
                else
                    insert!(out,k,bt[i])
                end
            end
            MultiGrade{V,A|B}(out)
        end
    end
    for Tens ∈ (:(TensorTerm{V,B}),[:($(MSC[k]){T,V,B} where T) for k ∈ 1:2]...,:(SparseChain{V,B}))
        @eval begin
            function $op(a::MultiGrade{V,A},b::T) where {T<:$Tens} where {V,A,B}
                N = ndims(V)
                out = copy(terms(a))
                i,k,bk,bl = 0,1,grade(out[1]),length(out)
                while i < bl
                    k += 1
                    i += 1
                    if bk == B
                        $(Expr(eop,:(out[k-1]),:b))
                        break
                    elseif B<bk
                        insert!(out,k-1,b)
                        break
                    elseif k ≤ length(out)
                        bk = grade(out[k])
                    else
                        insert!(out,k,b)
                        break
                    end
                end
                MultiGrade{V,A|(UInt(1)<<B)}(out)
            end
            $op(a::SparseChain{V,A},b::T) where {T<:$Tens} where {V,A,B} = MultiGrade{V,(UInt(1)<<A)|(UInt(1)<<B)}(A<B ? [a,b] : [b,a])
        end
        Tens≠:(SparseChain{V,B}) && (@eval $op(a::T,b::SparseChain{V,A}) where {T<:$Tens} where {V,A,B} = b+a)
    end
end

for un ∈ (:complementleft,:complementright)
    @eval begin
        $un(t::SparseChain{V,G}) where {V,G} = SparseChain{V,ndims(V)-G}($un.(terms(t)))
        $un(t::MultiGrade{V,G}) where {V,G} = SparseChain{V,G⊻(UInt(1)<<ndims(V)-1)}(reverse($un.(terms(t))))
    end
end
for un ∈ (:reverse,:involute,:conj,:+,:-)
    @eval begin
        $un(t::SparseChain{V,G}) where {V,G} = SparseChain{V,G}($un.(terms(t)))
        $un(t::MultiGrade{V,G}) where {V,G} = SparseChain{V,G}($un.(terms(t)))
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

function Base.:^(v::T,i::S) where {T<:TensorAlgebra{V},S<:Integer} where V
    out = one(V)
    if i < 8 # optimal choice ?
        for k ∈ 1:i
            out *= v
        end
    else
        ind = indices(UInt(i))
        K = length(ind)>0 ? ind[end] : 0
        b = falses(K)
        for k ∈ ind
            b[k] = true
        end
        p = v
        for k ∈ 1:K
            b[k] && (out *= p)
            k ≠ K && (p *= p)
        end
    end
    return out
end

## division

@pure abs2_inv(::Basis{V,G,B} where G) where {V,B} = abs2(getbasis(V,grade_basis(V,B)))

for (nv,d) ∈ ((:inv,:/),(:inv_rat,://))
    @eval begin
        @pure function $nv(b::Basis{V,G,B}) where {V,G,B}
            $d(parityreverse(grade(V,B)) ? -1 : 1,value(abs2_inv(b)))*b
        end
        @pure $d(a,b::T) where T<:TensorAlgebra = a*$nv(b)
        @pure $d(a::N,b::T) where {N<:Number,T<:TensorAlgebra} = a*$nv(b)
        function $nv(m::MultiVector{T,V}) where {T,V}
            rm = ~m
            d = rm*m
            fd = norm(d)
            sd = scalar(d)
            value(sd) ≈ fd && (return $d(rm,sd))
            for k ∈ 1:ndims(V)
                @inbounds norm(d[k]) ≈ fd && (return $d(rm,d(k)))
            end
            throw(error("inv($m) is undefined"))
        end
    end
    for Blade ∈ MSB
        @eval begin
            function $nv(b::$Blade{V,G,B,T}) where {V,G,B,T}
                $Blade{V,G,B}($d(parityreverse(grade(V,B)) ? -one(T) : one(T),value(abs2_inv(B))*value(b)))
            end
            function $nv(b::$Blade{V,G,B,Any}) where {V,G,B}
                $Blade{V,G,B}($d(parityreverse(grade(V,B)) ? -1 : 1,value(abs2_inv(B))*value(b)))
            end
        end
    end
    for Chain ∈ MSC
        @eval function $nv(a::$Chain)
            r,v,q = ~a,abs2(a),diffvars(vectorspace(a))≠0
            q&&typeof(v)<:TensorMixed ? Expr(:call,$(QuoteNode(d)),r,v) : $d(r,value(v))
        end
    end
    for Term ∈ (:TensorTerm,MSC...,:MultiVector,:MultiGrade)
        @eval begin
            @pure $d(a::S,b::T) where {S<:$Term,T<:Real} = a*$d(1,b)
            @pure $d(a::S,b::T) where {S<:$Term,T<:Complex} = a*$d(1,b)
            @pure $d(a::S,b::UniformScaling) where S<:$Term = a*$nv(vectorspace(a)(b))
        end
    end
end

for op ∈ (:div,:rem,:mod,:mod1,:fld,:fld1,:cld,:ldexp)
    for Blade ∈ MSB
        @eval Base.$op(b::$Blade{V,G,B,T},m) where {V,G,B,T} = $Blade{V,G,B}($op(value(b),m))
    end
    for Chain ∈ MSC
        @eval Base.$op(a::$Chain{T,V,G},m::S) where {T,V,G,S} = $Chain{promote_type(T,S),V,G}($op.(value(a),m))
    end
    @eval begin
        Base.$op(a::Basis{V,G},m) where {V,G} = Basis{V,G}($op(value(a),m))
        Base.$op(a::MultiVector{T,V},m::S) where {T,V,S} = MultiVector{promote_type(T,S),V}($op.(value(a),m))
    end
end
for op ∈ (:mod2pi,:rem2pi,:rad2deg,:deg2rad,:round)
    for Blade ∈ MSB
        @eval Base.$op(b::$Blade{V,G,B,T}) where {V,G,B,T} = $Blade{V,G,B}($op(value(b)))
    end
    for Chain ∈ MSC
        @eval Base.$op(a::$Chain{T,V,G}) where {T,V,G} = $Chain{promote_type(T,Float64),V,G}($op.(value(a)))
    end
    @eval begin
        Base.$op(a::Basis{V,G}) where {V,G} = Basis{V,G}($op(value(a)))
        Base.$op(a::MultiVector{T,V}) where {T,V} = MultiVector{promote_type(T,Float64),V}($op.(value(a)))
    end
end
for Blade ∈ MSB
    @eval Base.rationalize(t::Type,b::$Blade{V,G,B,T};tol::Real=eps(T)) where {V,G,B,T} = $Blade{V,G,B}(rationalize(t,value(b),tol))
end
for Chain ∈ MSC
    @eval Base.rationalize(t::Type,a::$Chain{T,V,G};tol::Real=eps(T)) where {T,V,G} = $Chain{T,V,G}(rationalize.(t,value(a),tol))
end
Base.rationalize(t::Type,a::Basis{V,G},tol::Real=eps(T)) where {V,G} = Basis{V,G}(rationalize(t,value(a),tol))
Base.rationalize(t::Type,a::MultiVector{T,V};tol::Real=eps(T)) where {T,V} = MultiVector{T,V}(rationalize.(t,value(a),tol))
Base.rationalize(t::T;kvs...) where T<:TensorAlgebra = rationalize(Int,t;kvs...)
