
#   This file is part of Grassmann.jl
#   It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

import Base: +, -, *, ^, /, //, inv, <, >, <<, >>, >>>
import AbstractTensors: ∧, ∨, ⟑, ⊖, ⊘, ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, ∗, rem, div, TAG, SUB
import AbstractTensors: plus, minus, times, contraction, equal, wedgedot, veedot
import AbstractTensors: pseudosandwich, antisandwich, cosandwich, antidot, codot
import Leibniz: diffcheck, diffmode, symmetricsplit
import Leibniz: loworder, isnull, Field, ExprField
const Sym,SymField = :AbstractTensors,Any

export ∗, ⊛, ⊖, ∧, ∨, ⟑, wedgedot, veedot, ⊗, ⨼, ⨽, ⊙, ⊠, ⟂, ∥
export ⊘, sandwich, pseudosandwich, antisandwich, cosandwich

if VERSION >= v"1.10.0"; @eval begin
    import AbstractTensors.$(Symbol("⟇"))
    export $(Symbol("⟇"))
end end

## geometric product

"""
    *(ω::TensorAlgebra,η::TensorAlgebra)

Geometric algebraic product: ω⊖η = (-1)ᵖdet(ω∩η)⊗(Λ(ω⊖η)∪L(ω⊕η))
"""
@pure ⟑(a::Submanifold{V},b::Submanifold{V}) where V = mul(a,b)
@pure wedgedot_metric(a::Submanifold{V},b::Submanifold{V},g) where V = mul_metric(a,b,g)
⟑(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = ⟑(a⟑b,c...)
*(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = *(a*b,c...)

@pure function mul(a::Submanifold{V},b::Submanifold{V},der=derive_mul(V,UInt(a),UInt(b),1,true)) where V
    if isdiag(V)
        ba,bb = UInt(a),UInt(b)
        istangent(V) && (diffcheck(V,ba,bb) || iszero(der)) && (return Zero(V))
        A,B,Q,Z = symmetricmask(V,ba,bb)
        d = getbasis(V,(A⊻B)|Q)
        out = if typeof(V)<:Signature || count_ones(A&B)==0
            (parity(a,b) ? Single{V}(-1,d) : d)
        else
            Single{V}(signbool(parityinner(V,A,B)),d)
        end
        diffvars(V)≠0 && !iszero(Z) && (out = Single{V}(getbasis(loworder(V),Z),out))
        return out
    else
        out = paritygeometric(V,UInt(a),UInt(b))
        isempty(out) ? Zero(V) : +(Single{V}.(out)...)
    end
end

mul_term(t) = :(Single{V}(($(t[1]),$(t[2]))))

@pure @generated function mul_metric(a::Submanifold{V},b::Submanifold{V},g,der=derive_mul(V,UInt(a),UInt(b),1,true)) where V
    isinduced(g) && (return :(mul(a,b,der)))
    out = paritygeometric(V,UInt(a),UInt(b),Val(true))
    if isempty(out)
        Zero(V)
    else
        Expr(:call,:+,mul_term.(out)...)
    end
end

function ⟑(a::Single{V},b::Submanifold{V}) where V
    v = derive_mul(V,UInt(basis(a)),UInt(b),a.v,true)
    bas = mul(basis(a),b,v)
    order(a.v)+order(bas)>diffmode(V) ? Zero(V) : v*bas
end
function ⟑(a::Submanifold{V},b::Single{V}) where V
    v = derive_mul(V,UInt(a),UInt(basis(b)),b.v,false)
    bas = mul(a,basis(b),v)
    order(b.v)+order(bas)>diffmode(V) ? Zero(V) : v*bas
end
function wedgedot_metric(a::Single{V},b::Submanifold{V},g) where V
    v = derive_mul(V,UInt(basis(a)),UInt(b),a.v,true)
    bas = mul_metric(basis(a),b,g,v)
    order(a.v)+order(bas)>diffmode(V) ? Zero(V) : v*bas
end
function wedgedot_metric(a::Submanifold{V},b::Single{V},g) where V
    v = derive_mul(V,UInt(a),UInt(basis(b)),b.v,false)
    bas = mul_metric(a,basis(b),g,v)
    order(b.v)+order(bas)>diffmode(V) ? Zero(V) : v*bas
end

@doc """
    ∗(ω::TensorAlgebra,η::TensorAlgebra)

Reversed geometric product: ω∗η = (~ω)*η
""" Grassmann.:∗

## exterior product

"""
    ∧(ω::TensorAlgebra,η::TensorAlgebra)

Exterior product as defined by the anti-symmetric quotient Λ≡⊗/~
"""
@inline ∧(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∧,a,b)
@inline ∧(a::TensorAlgebra{V},b::UniformScaling{T}) where {V,T<:Field} = a∧V(b)
@inline ∧(a::UniformScaling{T},b::TensorAlgebra{V}) where {V,T<:Field} = V(a)∧b
@generated ∧(t::T) where T<:Values{N} where N = wedges([:(t[$i]) for i ∈ list(1,N)])
@generated ∧(t::T) where T<:FixedVector{N} where N = wedges([:(@inbounds t[$i]) for i ∈ list(1,N)])
∧(::Values{0,<:Chain{V}}) where V = One(V) # ∧() = 1
∧(::FixedVector{0,<:Chain{V}}) where V = One(V)
function ∧(t::Chain{V,1,<:Chain{W}}) where {V,W}
    if mdims(V)>mdims(W)
        map(Real,compound(t,Val(min(mdims(V),mdims(W)))))
    else
        ∧(value(t))
    end
end
∧(t::Chain{V,1,<:Single} where V) = ∧(value(t))
∧(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = ∧(a∧b,c...)

wedges(x,i=length(x)-1) = i ≠ 0 ? Expr(:call,:∧,wedges(x,i-1),x[1+i]) : x[1+i]

@pure function ∧(a::Submanifold{V},b::Submanifold{V}) where V
    ba,bb = UInt(a),UInt(b)
    A,B,Q,Z = symmetricmask(V,ba,bb)
    ((count_ones(A&B)>0) || diffcheck(V,ba,bb) || iszero(derive_mul(V,ba,bb,1,true))) && (return Zero(V))
    d = getbasis(V,(A⊻B)|Q)
    diffvars(V)≠0 && !iszero(Z) && (d = Single{V}(getbasis(loworder(V),Z),d))
    return parity(a,b) ? Single{V}(-1,d) : d
end

function ∧(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    x,y = basis(a), basis(b)
    ba,bb = UInt(x),UInt(y)
    A,B,Q,Z = symmetricmask(V,ba,bb)
    ((count_ones(A&B)>0) || diffcheck(V,ba,bb)) && (return Zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.∏)
    if istangent(V) && !iszero(Z)
        v = !(typeof(v)<:TensorTerm) ? Single{V}(v,getbasis(V,Z)) : Single{V}(v,getbasis(loworder(V),Z))
        count_ones(Q)+order(v)>diffmode(V) && (return Zero(V))
    end
    return Single{V}(parity(x,y) ? -v : v,getbasis(V,(A⊻B)|Q))
end

#⊗(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = a∧b
⊗(a::A,b::B) where {A<:TensorGraded,B<:TensorGraded} = Dyadic(a,b)
⊗(a::A,b::B) where {A<:TensorGraded,B<:TensorGraded{V,0} where V} = a*b
⊗(a::A,b::B) where {A<:TensorGraded{V,0} where V,B<:TensorGraded} = a*b

## regressive product: (L = grade(a) + grade(b); (-1)^(L*(L-mdims(V)))*⋆(⋆(a)∧⋆(b)))

@pure function ∨(a::Submanifold{V},b::Submanifold{V}) where V
    p,C,t,Z = regressive(a,b)
    (!t || iszero(derive_mul(V,UInt(a),UInt(b),1,true))) && (return Zero(V))
    d = getbasis(V,C)
    istangent(V) && !iszero(Z) && (d = Single{V}(getbasis(loworder(V),Z),d))
    return isone(p) ? d : Single{V}(p,d)
end

function ∨(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    ba,bb = UInt(basis(a)),UInt(basis(b))
    p,C,t,Z = regressive(V,ba,bb)
    !t  && (return Zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.∏)
    if istangent(V) && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,ba,bb)
        v = !(typeof(v)<:TensorTerm) ? Single{V}(v,getbasis(V,Z)) : Single{V}(v,getbasis(loworder(V),Z))
        count_ones(Q)+order(v)>diffmode(V) && (return Zero(V))
    end
    return Single{V}(isone(p) ? v : p*v,getbasis(V,C))
end

"""
    ∨(ω::TensorAlgebra,η::TensorAlgebra)

Regressive product as defined by the DeMorgan's law: ∨(ω...) = ⋆⁻¹(∧(⋆.(ω)...))
"""
@inline ∨(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∨,a,b)
@inline ∨(a::TensorAlgebra{V},b::UniformScaling{T}) where {V,T<:Field} = a∨V(b)
@inline ∨(a::UniformScaling{T},b::TensorAlgebra{V}) where {V,T<:Field} = V(a)∨b
@generated ∨(t::T) where T<:Values = Expr(:call,:∨,[:(t[$k]) for k ∈ list(1,length(t))]...)
@generated ∨(t::T) where T<:FixedVector = Expr(:call,:∨,[:(t[$k]) for k ∈ list(1,length(t))]...)
∨(::Values{0,<:Chain{V}}) where V = Submanifold(V) # ∨() = I
∨(::FixedVector{0,<:Chain{V}}) where V = Submanifold(V)
∨(t::Chain{V,1,<:Chain} where V) = ∧(value(t))
∨(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = ∨(a∨b,c...)

for X ∈ TAG, Y ∈ TAG
    @eval Base.:&(a::$X{V},b::$Y{V}) where V = a∨b
end

@doc """
    ∨(ω::TensorAlgebra,η::TensorAlgebra)

Regressive product as defined by the DeMorgan's law: ∨(ω...) = ⋆⁻¹(∧(⋆.(ω)...))
""" Grassmann.:&

## interior product: a ∨ ⋆(b)

"""
    contraction(ω::TensorAlgebra,η::TensorAlgebra)

Interior (right) contraction product: ω⋅η = ω∨⋆η
"""
@pure function contraction(a::Submanifold{V},b::Submanifold{V}) where V
    iszero(derive_mul(V,UInt(a),UInt(b),1,true)) && (return Zero(V))
    if isdiag(V) || hasconformal(V)
        g,C,t,Z = interior(a,b)
        (!t) && (return Zero(V))
        d = getbasis(V,C)
        istangent(V) && !iszero(Z) && (d = Single{V}(getbasis(loworder(V),Z),d))
        return isone(g) ? d : Single{V}(g,d)
    else
        Cg,Z = parityinterior(V,UInt(a),UInt(b),Val(true))
        #d = getbasis(V,C)
        #istangent(V) && !iszero(Z) && (d = Single{V}(getbasis(loworder(V),Z),d))
        return isempty(Cg) ? Zero(V) : +(Single{V}.(Cg)...)
    end
end

@pure @generated function contraction_metric(a::Submanifold{V},b::Submanifold{V},g) where V
    isinduced(g) && (return :(contraction(a,b)))
    iszero(derive_mul(V,UInt(a),UInt(b),1,true)) && (return Zero(V))
    Cg,Z = parityinterior(V,UInt(a),UInt(b),Val(true),Val(true))
    #d = getbasis(V,C)
    #istangent(V) && !iszero(Z) && (d = Single{V}(getbasis(loworder(V),Z),d))
    if isempty(Cg)
        Zero(V)
    else
        Expr(:call,:+,mul_term.(Cg)...)
    end
end

function contraction(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    ba,bb = UInt(basis(a)),UInt(basis(b))
    if isdiag(V) || hasconformal(V)
        g,C,t,Z = interior(V,ba,bb)
        !t && (return Zero(V))
        v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.dot)
        if istangent(V) && !iszero(Z)
            _,_,Q,_ = symmetricmask(V,ba,bb)
            v = !(typeof(v)<:TensorTerm) ? Single{V}(v,getbasis(V,Z)) : Single{V}(v,getbasis(loworder(V),Z))
            count_ones(Q)+order(v)>diffmode(V) && (return Zero(V))
        end
        return Single{V}(g*v,getbasis(V,C))
    else
        Cg,Z = parityinterior(V,ba,bb,Val(true))
        v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.dot)
        if istangent(V) && !iszero(Z)
            _,_,Q,_ = symmetricmask(V,ba,bb)
            v = !(typeof(v)<:TensorTerm) ? Single{V}(v,getbasis(V,Z)) : Single{V}(v,getbasis(loworder(V),Z))
            count_ones(Q)+order(v)>diffmode(V) && (return Zero(V))
        end
        return isempty(Cg) ? Zero(V) : (+(Single{V}.(Cg)...))*v
    end
end

@generated function contraction_metric(a::X,b::Y,g) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    isinduced(g) && (return :(contraction(a,b)))
    ba,bb = UInt(basis(a)),UInt(basis(b))
    Cg,Z = parityinterior(V,ba,bb,Val(true),Val(true))
    if isempty(Cg)
        Zero(V)
    else
        Expr(:call,:*,Expr(:call,:+,mul_term.(Cg)...),:(derive_mul(V,$ba,$bb,value(a),value(b),AbstractTensors.dot)))
    end
end

@doc """
    dot(ω::TensorAlgebra,η::TensorAlgebra)

Interior (right) contraction product: ω⋅η = ω∨⋆η
""" dot

## cross product

@doc """
    cross(ω::TensorAlgebra,η::TensorAlgebra)

Cross product: ω×η = ⋆(ω∧η)
""" LinearAlgebra.cross

# symmetrization and anti-symmetrization

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
    K,V,out = length(x),∪(Manifold.(x)...),prod(x)
    P,F = collect(permutations(1:K)),factorial(K)
    for n ∈ 2:F
        p = prod(x[P[n]])
        DirectSum.indexparity!(P[n],V)[1] ? (out-=p) : (out+=p)
    end
    return out/F
end

## sandwich product

⊘(x::TensorTerm{V},y::TensorTerm{V}) where V = reverse(y)*x*involute(y)
⊘(x::TensorAlgebra{V},y::TensorAlgebra{V}) where V = reverse(y)*x*involute(y)
⊘(x::Couple{V},y::TensorAlgebra{V}) where V = (scalar(x)⊘y)+(imaginary(x)⊘y)
⊘(x::PseudoCouple{V},y::TensorAlgebra{V}) where V = (imaginary(x)⊘y)+(volume(x)⊘y)
@generated ⊘(a::TensorGraded{V,G},b::Spinor{V}) where {V,G} = product_sandwich(a,b)
@generated ⊘(a::TensorGraded{V,G},b::CoSpinor{V}) where {V,G} = product_sandwich(a,b)
@generated ⊘(a::TensorGraded{V,G},b::Couple{V}) where {V,G} = product_sandwich(a,b)
@generated ⊘(a::TensorGraded{V,G},b::PseudoCouple{V}) where {V,G} = product_sandwich(a,b)
@generated ⊘(a::TensorGraded{V,G},b::TensorGraded{V,L}) where {V,G,L} = product_sandwich(a,b)
@generated function ⊘(a::TensorGraded{V,G},b::Spinor{V},g) where {V,G}
    isinduced(g) && (return :(a⊘b))
    product_sandwich(a,b,false,true)
end
@generated function ⊘(a::TensorGraded{V,G},b::CoSpinor{V},g) where {V,G}
    isinduced(g) && (return :(a⊘b))
    product_sandwich(a,b,false,true)
end
@generated function ⊘(a::TensorGraded{V,G},b::Couple{V},g) where {V,G}
    isinduced(g) && (return :(a⊘b))
    product_sandwich(a,b,false,true)
end
@generated function ⊘(a::TensorGraded{V,G},b::PseudoCouple{V},g) where {V,G}
    isinduced(g) && (return :(a⊘b))
    product_sandwich(a,b,false,true)
end
@generated function ⊘(a::TensorGraded{V,G},b::TensorGraded{V,L},g) where {V,G,L}
    isinduced(g) && (return :(a⊘b))
    product_sandwich(a,b,false,true)
end

@doc """
    ⊘(ω::TensorAlgebra,η::TensorAlgebra)

General sandwich product: ω⊘η = reverse(η)⊖ω⊖involute(η)

For normalized even grade η it is ω⊘η = (~η)⊖ω⊖η
""" Grassmann.:⊘

>>>(y::TensorTerm{V},x::TensorTerm{V}) where V = y*x*clifford(y)
>>>(y::TensorAlgebra{V},x::TensorAlgebra{V}) where V = y*x*clifford(y)
>>>(y::TensorAlgebra{V},x::Couple{V}) where V = (y>>>scalar(x))+(y>>>imaginary(x))
>>>(y::TensorAlgebra{V},x::PseudoCouple{V}) where V = (y>>>imaginary(x))+(y>>>volume(x))
@generated >>>(b::Spinor{V},a::TensorGraded{V,G}) where {V,G} = product_sandwich(a,b,true)
@generated >>>(b::CoSpinor{V},a::TensorGraded{V,G}) where {V,G} = product_sandwich(a,b,true)
@generated >>>(b::Couple{V},a::TensorGraded{V,G}) where {V,G} = product_sandwich(a,b,true)
@generated >>>(b::PseudoCouple{V},a::TensorGraded{V,G}) where {V,G} = product_sandwich(a,b,true)
@generated >>>(b::TensorGraded{V,L},a::TensorGraded{V,G}) where {V,G,L} = product_sandwich(a,b,true)
@generated function >>>(b::Spinor{V},a::TensorGraded{V,G},g) where {V,G}
    isinduced(g) && (return :(b>>>a))
    product_sandwich(a,b,true,true)
end
@generated function >>>(b::CoSpinor{V},a::TensorGraded{V,G},g) where {V,G}
    isinduced(g) && (return :(b>>>a))
    product_sandwich(a,b,true,true)
end
@generated function >>>(b::Couple{V},a::TensorGraded{V,G},g) where {V,G}
    isinduced(g) && (return :(b>>>a))
    product_sandwich(a,b,true,true)
end
@generated function >>>(b::PseudoCouple{V},a::TensorGraded{V,G},g) where {V,G}
    isinduced(g) && (return :(b>>>a))
    product_sandwich(a,b,true,true)
end
@generated function >>>(b::TensorGraded{V,L},a::TensorGraded{V,G},g) where {V,G,L}
    isinduced(g) && (return :(b>>>a))
    product_sandwich(a,b,true,true)
end

@doc """
    >>>(ω::TensorAlgebra,η::TensorAlgebra)

Traditional sandwich product: ω>>>η = ω⊖η⊖clifford(ω)

For normalized even grade η it is ω>>>η = ω⊖η⊖(~ω)
""" Grassmann.:>>>

## veedot

veedot(a,b) = complementleft(complementright(a)*complementright(b))
veedot_metric(a,b,g) = complementleft(wedgedot_metric(complementright(a),complementright(b),g))

## antidot

antidot(a,b) = complementleft(contraction(complementright(a),complementright(b)))
antidot_metric(a,b) = complementleft(contraction_metric(complementright(a),complementright(b),g))

## linear algebra

∥(a,b) = iszero(a∧b)

## exponentiation

# Inline x^2 and x^3 for Val
# (The first argument prevents unexpected behavior if a function ^
# is defined that is not equal to Base.^)
@inline literal_pow(::typeof(^), x::TensorAlgebra, ::Val{0}) = one(x)
@inline literal_pow(::typeof(^), x::TensorAlgebra, ::Val{1}) = x
@inline literal_pow(::typeof(^), x::TensorAlgebra, ::Val{2}) = x*x
@inline literal_pow(::typeof(^), x::TensorAlgebra, ::Val{3}) = x*x*x
@inline literal_pow(::typeof(^), x::TensorAlgebra, ::Val{-1}) = inv(x)
@inline literal_pow(::typeof(^), x::TensorAlgebra, ::Val{-2}) = (i=inv(x); i*i)

# don't use the inv(x) transformation here since float^p is slightly more accurate
@inline literal_pow(::typeof(^), x::Scalar{V,<:AbstractFloat} where V, ::Val{p}) where {p} = x^p
@inline literal_pow(::typeof(^), x::Scalar{V,<:AbstractFloat} where V, ::Val{-1}) = inv(x)

for (op,field) ∈ ((:*,false),(:wedgedot_metric,true)); args = field ? (:g,) : ()
@eval function Base.:^(v::T,i::S,$(args...)) where {T<:TensorTerm,S<:Integer}
    i == 0 && (return getbasis(Manifold(v),0))
    i == 1 && (return v)
    j,bas = (i-1)%4,basis(v)
    out = if j == 0
        bas
    elseif j == 1
        $op(bas,bas,$(args...))
    elseif j == 2
        $op($op(bas,bas,$(args...)),bas,$(args...))
    elseif j == 3
        $op($op($op(bas,bas,$(args...)),bas,$(args...)),bas,$(args...))
    end
    return typeof(v)<:Submanifold ? out : out*AbstractTensors.:^(value(v),i)
end

@eval function Base.:^(v::T,i::S,$(args...)) where {T<:TensorAlgebra,S<:Integer}
    V = Manifold(v)
    isone(i) && (return v)
    if T<:Chain && mdims(V)≤3 && diffvars(v)==0
        sq,d = $(field ? :contraction2_metric : :contraction2)(~v,v),i÷2
        val = isone(d) ? sq : ^(sq,d,$(args...))
        return iszero(i%2) ? val : val*v
    elseif !$field && T<:Couple && value(basis(v)*basis(v))==-1
        return Couple{V,basis(v)}(v.v^i)
    end
    out = One(V)
    if i < 8 # optimal choice ?
        for k ∈ 1:i
            $(field ? :(out = $op(out,v,g)) : :(out *= v))
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
            b[k] && $(field ? :(out = $op(out,p,g)) : :(out *= p))
            k ≠ K && $(field ? :(p = $op(p,p,g)) : :(p *= p))
        end
    end
    return out
end

## division

@eval @pure abs2_inv(::Submanifold{V,G,B} where G,$(args...)) where {V,B} = abs2(getbasis(V,grade_basis(V,B)),$(args...))

for (nv,d) ∈ ((:inv,:/),(:inv_rat,://))
    @eval begin
        @pure $d(a,b::T,$(args...)) where T<:TensorAlgebra = $op(a,$nv(b,$(args...)),$(args...))
        @pure $d(a::N,b::T,$(args...)) where {N<:Number,T<:TensorAlgebra} = $op(a,$nv(b,$(args...)),$(args...))
        @pure $d(a::S,b::UniformScaling,$(args...)) where S<:TensorGraded = $op(a,$nv(Manifold(a)(b),$(args...)),$(args...))
        @pure $d(a::S,b::UniformScaling,$(args...)) where S<:TensorMixed = $op(a,$nv(Manifold(a)(b),$(args...)),$(args...))
        $nv(a::PseudoCouple,$(args...)) = /(~a,abs2(a,$(args...)),$(args...))
        function $nv(a::Chain,$(args...))
            r,v,q = ~a,abs2(a,$(args...)),diffvars(Manifold(a))≠0
            q&&!(typeof(v)<:TensorGraded && grade(v)==0) ? $d(r,v,$(args...)) : $d(r,value(scalar(v)))
        end
        function $nv(m::Multivector{V,T},$(args...)) where {V,T}
            rm = ~m
            d = $op(rm,m,$(args...))
            fd = norm(d)
            sd = scalar(d)
            norm(sd) ≈ fd && (return $d(rm,sd))
            for k ∈ list(1,mdims(V))
                @inbounds AbstractTensors.norm(d[k]) ≈ fd && (return $d(rm,d(k),$(args...)))
            end
            throw(error("inv($m) is undefined"))
        end
        function $nv(m::Multivector{V,Any},$(args...)) where V
            rm = ~m
            d = $op(rm,m,$(args...))
            fd = $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d)]...)
            sd = scalar(d)
            $Sym.:∏(value(sd),value(sd)) == fd && (return $d(rm,sd))
            for k ∈ list(1,mdims(V))
                @inbounds $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d[k])]...) == fd && (return $d(rm,d(k),$(args...)))
            end
            throw(error("inv($m) is undefined"))
        end
    end
    for pinor ∈ (:Spinor,:CoSpinor)
        @eval begin
            function $nv(m::$pinor{V,T},$(args...)) where {V,T}
                rm = ~m
                d = $op(rm,m,$(args...))
                fd = norm(d)
                sd = scalar(d)
                norm(sd) ≈ fd && (return $d(rm,sd))
                for k ∈ evens(2,mdims(V))
                    @inbounds AbstractTensors.norm(d[k]) ≈ fd && (return $d(rm,d(k),$(args...)))
                end
                throw(error("inv($m) is undefined"))
            end
            function $nv(m::$pinor{V,Any},$(args...)) where V
                rm = ~m
                d = $op(rm,m,$(args...))
                fd = $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d)]...)
                sd = scalar(d)
                $Sym.:∏(value(sd),value(sd)) == fd && (return $d(rm,sd))
                for k ∈ evens(2,mdims(V))
                    @inbounds $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d[k])]...) == fd && (return $d(rm,d(k),$(args...)))
                end
                throw(error("inv($m) is undefined"))
            end
        end
    end
    @eval begin
        @pure $nv(b::Submanifold{V,0} where V,$(args...)) = b
        @pure function $nv(b::Submanifold{V,G,B},$(args...)) where {V,G,B}
            $d(parityreverse(grade(V,B)) ? -1 : 1,value(abs2_inv(b,$(args...))))*b
        end
        $nv(b::Single{V,0,B},$(args...)) where {V,B} = Single{V,0,B}(AbstractTensors.inv(value(b)))
        function $nv(b::Single{V,G,B,T},$(args...)) where {V,G,B,T}
            Single{V,G,B}($d(parityreverse(grade(V,B)) ? -one(T) : one(T),value(abs2_inv(B,$(args...))*value(b))))
        end
        function $nv(b::Single{V,G,B,Any},$(args...)) where {V,G,B}
            Single{V,G,B}($Sym.$d(parityreverse(grade(V,B)) ? -1 : 1,value($Sym.:∏(abs2_inv(B,$(args...)),value(b)))))
        end
    end
end

@eval /(a::TensorTerm{V,0},b::Couple{V,B,S},$(args...)) where {V,B,S} = a*inv(b,$(args...))
@eval /(a::Couple{V,B},b::TensorTerm{V,0},$(args...)) where {V,B} = Couple{V,B}(realvalue(a)/value(b),imagvalue(a)/value(b))

@eval function /(a::Couple{V,B,T}, b::Couple{V,B,T},$(args...)) where {V,B,T<:Real}
    are,aim = reim(a); bre,bim = reim(b)
    B2 = value(abs2_inv(B,$(args...)))
    (rout,iout) = if abs(bre) <= abs(bim)
        if isinf(bre) && isinf(bim)
            r = sign(bre)/sign(bim)
        else
            r = bre / bim
        end
        den = bim*B2 + r*bre
        ((are*r + aim*B2)/den, (aim*r - are)/den)
    else
        if isinf(bre) && isinf(bim)
            r = sign(bim)/sign(bre)
        else
            r = bim / bre
        end
        den = bre + (r*bim)*B2
        ((are + (aim*r)*B2)/den, (aim - are*r)/den)
    end
    Couple{V,B}(rout,iout)
end

@eval inv(z::Couple{V,B,T},$(args...)) where {V,B,T<:Union{Float16,Float32}} =
    (w = inv(widen(z),$(args...)); Couple{V,B}(convert(T,realvalue(w)),convert(T,imagvalue(w))))

@eval /(z::Couple{V,B,T}, w::Couple{V,B,T},$(args...)) where {V,B,T<:Union{Float16,Float32}} =
    (w = $op(widen(z),inv(widen(w),$(args...)),$(args...)); Couple{V,B}(convert(T,realvalue(w)),convert(T,imagvalue(w))))

# robust complex division for double precision
# variables are scaled & unscaled to avoid over/underflow, if necessary
# based on arxiv.1210.4539
#             a + i*b
#  p + i*q = ---------
#             c + i*d
@eval function /(z::Couple{V,B,Float64}, w::Couple{V,B,Float64},$(args...)) where {V,B}
    a, b = reim(z); c, d = reim(w)
    absa = abs(a); absb = abs(b);  ab = absa >= absb ? absa : absb # equiv. to max(abs(a),abs(b)) but without NaN-handling (faster)
    absc = abs(c); absd = abs(d);  cd = absc >= absd ? absc : absd
    halfov = 0.5*floatmax(Float64)              # overflow threshold
    twounϵ = floatmin(Float64)*2.0/eps(Float64) # underflow threshold
    # actual division operations
    e = Val(float(value(abs2_inv(B,$(args...)))))
    if  ab>=halfov || ab<=twounϵ || cd>=halfov || cd<=twounϵ # over/underflow case
        p,q = scaling_cdiv(a,b,c,d,ab,cd,e) # scales a,b,c,d before division (unscales after)
    else
        p,q = cdiv(a,b,c,d,e)
    end
    return Couple{V,B,Float64}(p,q)
end

@eval function inv(z::Couple{V,B},$(args...)) where {V,B}
    c, d = reim(z)
    (isinf(c) | isinf(d)) && (return Couple{V,B}(copysign(zero(c), c), flipsign(-zero(d), d)))
    e = c*c + d*d*value(abs2_inv(B,$(args...)))
    Couple{V,B}(c/e, parityreverse(grade(B)) ? -d/e : d/e)
end
@eval inv(z::Couple{V,B,<:Integer},$(args...)) where {V,B} = inv(Couple{V,B}(float.(z.v)),$(args...))

@eval function inv(w::Couple{V,B,Float64},$(args...)) where {V,B}
    c, d = reim(w)
    (isinf(c) | isinf(d)) && return complex(copysign(0.0, c), flipsign(-0.0, d))
    absc, absd = abs(c), abs(d)
    cd = ifelse(absc>absd, absc, absd) # cheap `max`: don't need sign- and nan-checks here
    ϵ  = eps(Float64)
    bs = 2/(ϵ*ϵ)
    # scaling
    s = 1.0
    if cd >= floatmax(Float64)/2
        c *= 0.5; d *= 0.5; s = 0.5 # scale down c, d
    elseif cd <= 2floatmin(Float64)/ϵ
        c *= bs;  d *= bs;  s = bs  # scale up c, d
    end
    # inversion operations
    a = Val(float(value(abs2_inv(B,$(args...)))))
    if absd <= absc
        p, q = robust_cinv(c, d, a)
    else
        q, p = robust_cinv_rev(-d, -c, a)
    end
    return Couple{V,B,Float64}(p*s, q*s) # undo scaling
end
end

function robust_cinv(c::Float64, d::Float64, ::Val{a}) where a
    r = d/c
    p = inv(muladd(d, r*a, c))
    q = -r*p
    return p, q
end
function robust_cinv_rev(c::Float64, d::Float64, ::Val{a}) where a
    r = d/c
    p = inv(muladd(d, r, c*a))
    q = -r*p
    return p, q
end

# sub-functionality for /(z::ComplexF64, w::ComplexF64)
@inline function cdiv(a::Float64, b::Float64, c::Float64, d::Float64, e::Val)
    if abs(d)<=abs(c)
        p,q = Base.robust_cdiv1(a,b,c,d,e)
    else
        p,q = robust_cdiv1_rev(b,a,d,c,e)
        q = -q
    end
    return p,q
end
@noinline function scaling_cdiv(a::Float64, b::Float64, c::Float64, d::Float64, ab::Float64, cd::Float64, e::Val)
    # this over/underflow functionality is outlined for performance, cf. #29688
    a,b,c,d,s = Base.scaleargs_cdiv(a,b,c,d,ab,cd)
    p,q = cdiv(a,b,c,d,e)
    return p*s,q*s
end
@inline function Base.robust_cdiv1(a::Float64, b::Float64, c::Float64, d::Float64, e::Val{f}) where f
    r = d/c
    t = 1.0/(c+(d*r)*f)
    p = Base.robust_cdiv2(a,b,c,d,r,t,e)
    q = Base.robust_cdiv2(b,-a,c,d,r,t)
    return p,q
end
@inline function robust_cdiv1_rev(a::Float64, b::Float64, c::Float64, d::Float64, e::Val{f}) where f
    r = d/c
    t = 1.0/(c*f+d*r)
    p = robust_cdiv2_rev(a,b,c,d,r,t,e)
    q = Base.robust_cdiv2(b,-a,c,d,r,t)
    return p,q
end
function Base.robust_cdiv2(a::Float64, b::Float64, c::Float64, d::Float64, r::Float64, t::Float64, ::Val{e}) where e
    if r != 0
        br = b*r
        return (br != 0 ? (a+br*e)*t : a*t + ((b*t)*r)*e)
    else
        return (a + (d*(b/c))*e) * t
    end
end
function robust_cdiv2_rev(a::Float64, b::Float64, c::Float64, d::Float64, r::Float64, t::Float64, ::Val{e}) where e
    if r != 0
        br = b*r
        return (br != 0 ? (a*e+br)*t : (a*t)*e + (b*t)*r)
    else
        return (a*e + d*(b/c)) * t
    end
end

function generate_inverses(Mod,T)
    for (nv,d,ds) ∈ ((:inv,:/,:($Sym.:/)),(:inv_rat,://,:($Sym.://)))
        for Term ∈ (:TensorGraded,:TensorMixed)
            @eval $d(a::S,b::T) where {S<:$Term,T<:$Mod.$T} = a*$ds(1,b)
        end
        @eval function $nv(b::Single{V,G,B,$Mod.$T}) where {V,G,B}
            Single{V,G,B}($Mod.$d(parityreverse(grade(V,B)) ? -1 : 1,value($Sym.:∏(abs2_inv(B),value(b)))))
        end
    end
end

for T ∈ (:Real,:Complex)
    generate_inverses(Base,T)
end

### Algebra Constructors

addvec(a,b,s,o) = o ≠ :+ ? subvec(a,b,s) : addvec(a,b,s)
addvec(a,b,s) = isfixed(a,b) ? (:($Sym.:∑),:($Sym.:∑),:svec) : (:+,:+,:mvec)
subvec(a,b,s) = isfixed(a,b) ? (s ? (:($Sym.:-),:($Sym.:∑),:svec) : (:($Sym.:∑),:($Sym.:-),:svec)) : (s ? (:-,:+,:mvec) : (:+,:-,:mvec))

subvec(b) = isfixed(valuetype(b)) ? (:($Sym.:-),:svec,:($Sym.:∏)) : (:-,:mvec,:*)
subvecs(b) = isfixed(valuetype(b)) ? (:($Sym.:-),:svecs,:($Sym.:∏)) : (:-,:mvecs,:*)
conjvec(b) = isfixed(valuetype(b)) ? (:($Sym.conj),:svec) : (:conj,:mvec)

mulvec(a,b,c) = c≠:contraction ? mulvec(a,b) : isfixed(a,b) ? (:($Sym.dot),:svec) : (:dot,:mvec)
mulvec(a,b) = isfixed(a,b) ? (:($Sym.:∏),:svec) : (:*,:mvec)
mulvecs(a,b) = isfixed(a,b) ? (:($Sym.:∏),:svecs) : (:*,:mvecs)
mulvecs(a,b,c) = c≠:contraction ? mulvecs(a,b) : isfixed(a,b) ? (:($Sym.dot),:svecs) : (:dot,:mvecs)
isfixed(a,b) = isfixed(valuetype(a))||isfixed(valuetype(b))
isfixed(::Type{Rational{BigInt}}) = true
isfixed(::Type{BigFloat}) = true
isfixed(::Type{BigInt}) = true
isfixed(::Type{Complex{BigFloat}}) = true
isfixed(::Type{Complex{BigInt}}) = true
isfixed(::Type{<:Number}) = false
isfixed(::Type{<:Any}) = true

const NSE = Union{Symbol,Expr,<:Real,<:Complex}

@inline swapper(a,b,swap) = swap ? (b,a) : (a,b)

adder(a,b,op=:+) = adder(typeof(a),typeof(b),op)

@eval begin
    @noinline function adder(a::Type{<:TensorTerm{V,L}},b::Type{<:TensorTerm{V,G}},op) where {V,L,G}
        left,bop,VEC = addvec(a,b,false,op)
        if basis(a) == basis(b)
            :(Single{V,L}($bop(value(a),value(b)),basis(a)))
        elseif !istangent(V) && !hasconformal(V) && L == 0
            :(Couple{V,basis(b)}(value(a),$bop(value(b))))
        elseif !istangent(V) && !hasconformal(V) && G == 0
            :(Couple{V,basis(a)}($bop(value(b)),value(a)))
        elseif !istangent(V) && !hasconformal(V) && L == grade(V)
            :(PseudoCouple{V,basis(b)}($bop(value(b)),value(a)))
        elseif !istangent(V) && !hasconformal(V) && G == grade(V)
            :(PseudoCouple{V,basis(a)}(value(a),$bop(value(b))))
        elseif L == G
            if binomial(mdims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:ib,:t),:mvec)...)
                out = svec(N,G,Any)(zeros(svec(N,G,t<:Number ? t : Int)))
                setblade!_pre(out,:(value(a,$t)),UInt(basis(a)),Val{N}())
                setblade!_pre(out,:($bop(value(b,$t))),UInt(basis(b)),Val{N}())
                return :(Chain{V,L}($(Expr(:call,tvec(N,G,t<:Number ? t : Any),out...))))
            else return quote
                $(insert_expr((:N,:t))...)
                out = zeros($VEC(N,L,t))
                setblade!(out,value(a,t),UInt(basis(a)),Val{N}())
                setblade!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
                return Chain{V,L}(out)
            end end
        elseif iseven(L) && iseven(G)
            adderspin(a,b,op)
        elseif isodd(L) && isodd(G)
            adderanti(a,b,op)
        else
            addermulti(a,b,op)
        end
    end
    @noinline function adderspin(a::Type{<:TensorTerm{V,L}},b::Type{<:TensorTerm{V,G}},op) where {V,L,G}
        (isodd(L) || isodd(G)) && (return :(error("$(basis(a)) and $(basis(b)) are not expressible as Spinor")))
        left,bop,VEC = addvec(a,b,false,op)
        if mdims(V)-1<cache_limit
            $(insert_expr((:N,:t),:mvecs)...)
            out,ib = svecs(N,Any)(zeros(svecs(N,t<:Number ? t : Int))),indexbasis(N)
            setspin!_pre(out,:(value(a,$t)),UInt(basis(a)),Val{N}())
            setspin!_pre(out,:($bop(value(b,$t))),UInt(basis(b)),Val{N}())
            return :(Spinor{V}($(Expr(:call,tvecs(N,t<:Number ? t : Any),out...))))
        else quote
            $(insert_expr((:N,:t),VEC)...)
            out = zeros(mvecs(N,t))
            setspin!(out,value(a,t),UInt(basis(a)),Val{N}())
            setspin!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
            return Spinor{V}(out)
        end end
    end
    @noinline function adderanti(a::Type{<:TensorTerm{V,L}},b::Type{<:TensorTerm{V,G}},op) where {V,L,G}
        (iseven(L) || iseven(G)) && (return :(error("$(basis(a)) and $(basis(b)) are not expressible as CoSpinor")))
        left,bop,VEC = addvec(a,b,false,op)
        if mdims(V)-1<cache_limit
            $(insert_expr((:N,),:svecs)...)
            t = promote_type(valuetype(a),valuetype(b))
            out,ib = svecs(N,Any)(zeros(svecs(N,t<:Number ? t : Int))),indexbasis(N)
            setanti!_pre(out,:(value(a,$t)),UInt(basis(a)),Val{N}())
            setanti!_pre(out,:($bop(value(b,$t))),UInt(basis(b)),Val{N}())
            return :(CoSpinor{V}($(Expr(:call,tvecs(N,t<:Number ? t : Any),out...))))
        else quote
            $(insert_expr((:N,:t),VEC)...)
            out = zeros(mvecs(N,t))
            setanti!(out,value(a,t),UInt(basis(a)),Val{N}())
            setanti!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
            return CoSpinor{V}(out)
        end end
    end
    @noinline function addermulti(a::Type{<:TensorTerm{V,L}},b::Type{<:TensorTerm{V,G}},op) where {V,L,G}
        left,bop,VEC = addvec(a,b,false,op)
        if mdims(V)<cache_limit
            $(insert_expr((:N,:t),:mvec)...)
            out,ib = svec(N,Any)(zeros(svec(N,t<:Number ? t : Int))),indexbasis(N)
            setmulti!_pre(out,:(value(a,$t)),UInt(basis(a)),Val{N}())
            setmulti!_pre(out,:($bop(value(b,$t))),UInt(basis(b)),Val{N}())
            return :(Multivector{V}($(Expr(:call,tvec(N,t<:Number ? t : Any),out...))))
        else quote
            $(insert_expr((:N,:t,:out),VEC)...)
            setmulti!(out,value(a,t),UInt(basis(a)),Val{N}())
            setmulti!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
            return Multivector{V}(out)
        end end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:Chain{V,G,T}},op,swap=false) where {V,G,T}
        left,right,VEC = addvec(a,b,swap,op)
        if binomial(mdims(V),G)<(1<<cache_limit)
            $(insert_expr((:N,:ib,:t),:mvec)...)
            out = svec(N,G,Any)(zeros(svec(N,G,t<:Number ? t : Int)))
            X = UInt(basis(a))
            for k ∈ list(1,binomial(N,G))
                B = @inbounds ib[k]
                val = :(@inbounds $right(b.v[$k]))
                val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                setblade!_pre(out,val,B,Val{N}())
            end
            return :(Chain{V,G}($(Expr(:call,tvec(N,G,t<:Number ? t : Any),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t),VEC)...)
            out = convert($VEC(N,G,t),$(bcast(right,:(value(b,$VEC(N,G,t)),))))
            addblade!(out,value(a,t),basis(a),Val{N}())
            return Chain{V,G}(out)
        end; else quote
            $(insert_expr((:N,:t),VEC)...)
            out = value(b,$VEC(N,G,t))
            addblade!(out,$left(value(a,t)),UInt(basis(a)),Val{N}())
            return Chain{V,G}(out)
        end end end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,L}},b::Type{<:Chain{V,G,T}},op,swap=false) where {V,G,T,L}
        left,right,VEC = addvec(a,b,swap,op)
        if !istangent(V) && !hasconformal(V) && L == 0 && G == mdims(V)
            if swap
                :(Couple{V,basis(V)}($right(value(a)),b.v[1]))
            else
                :(Couple{V,basis(V)}(value(a),$right(b.v[1])))
            end
        elseif !istangent(V) && !hasconformal(V) && G == 0
            if swap
                :(Couple{V,basis(a)}(b.v[1],$right(value(a))))
            else
                :(Couple{V,basis(a)}($right(b.v[1]),value(a)))
            end
        elseif !istangent(V) && !hasconformal(V) && G == grade(V)
            if swap
                :(PseudoCouple{V,basis(a)}($right(value(a)),b.v[1]))
            else
                :(PseudoCouple{V,basis(a)}(value(a),$right(b.v[1])))
            end
        elseif iseven(L) && iseven(G)
            if mdims(V)-1<cache_limit
                $(insert_expr((:N,:ib,:bn,:t),:mvecs)...)
                out = svecs(N,Any)(zeros(svecs(N,t<:Number ? t : Int)))
                X = UInt(basis(a))
                for k ∈ list(1,binomial(N,G))
                    B = @inbounds ib[k]
                    val = :(@inbounds $right(b.v[$k]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    @inbounds setspin!_pre(out,val,B,Val(N))
                end
                val = :(@inbounds $left(value(a,$t)))
                setspin!_pre(out,val,X,Val(N))
                return :(Spinor{V}($(Expr(:call,tvecs(N,t<:Number ? t : Any),out...))))
            else return if !swap; quote
                $(insert_expr((:N,:t,:out,:rr,:bng),VECS)...)
                @inbounds out[rr+1:rr+bng] = $(bcast(right,:(value(b,$VEC(N,G,t)),)))
                addspin!(out,value(a,t),UInt(basis(a)),Val(N))
                return Spinor{V}(out)
            end; else quote
                $(insert_expr((:N,:t,:out,:rr,:bng),VECS)...)
                @inbounds out[rr+1:rr+bng] = value(a,$VEC(N,G,t))
                addspin!(out,$left(value(b,t)),UInt(basis(b)),Val(N))
                return Spinor{V}(out)
            end end end
        elseif isodd(L) && isodd(G)
            if mdims(V)-1<cache_limit
                $(insert_expr((:N,:ib,:bn,:t),:mvecs)...)
                out = svecs(N,Any)(zeros(svecs(N,t<:Number ? t : Int)))
                X = UInt(basis(a))
                for k ∈ list(1,binomial(N,G))
                    B = @inbounds ib[k]
                    val = :(@inbounds $right(b.v[$k]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    @inbounds setanti!_pre(out,val,B,Val(N))
                end
                val = :(@inbounds $left(value(a,$t)))
                setanti!_pre(out,val,X,Val(N))
                return :(CoSpinor{V}($(Expr(:call,tvecs(N,t<:Number ? t : Any),out...))))
            else return if !swap; quote
                $(insert_expr((:N,:t,:out,:rrr,:bng),VECS)...)
                @inbounds out[rrr+1:rrr+bng] = $(bcast(right,:(value(b,$VEC(N,G,t)),)))
                addpseudo(out,value(a,t),UInt(basis(a)),Val(N))
                return CoSpinor{V}(out)
            end; else quote
                $(insert_expr((:N,:t,:out,:rrr,:bng),VECS)...)
                @inbounds out[rrr+1:rrr+bng] = value(a,$VEC(N,G,t))
                addanti!(out,$left(value(b,t)),UInt(basis(b)),Val(N))
                return CoSpinor{V}(out)
            end end end
        else
            if mdims(V)<cache_limit
                $(insert_expr((:N,:ib,:bn,:t),:mvec)...)
                out = svec(N,Any)(zeros(svec(N,t<:Number ? t : Int)))
                X = UInt(basis(a))
                for k ∈ list(1,binomial(N,G))
                    B = @inbounds ib[k]
                    val = :(@inbounds $right(b.v[$k]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    @inbounds setmulti!_pre(out,val,B,Val(N))
                end
                val = :(@inbounds $left(value(a,$t)))
                setmulti!_pre(out,val,X,Val(N))
                return :(Multivector{V}($(Expr(:call,tvec(N,t<:Number ? t : Any),out...))))
            else return if !swap; quote
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[r+1:r+bng] = $(bcast(right,:(value(b,$VEC(N,G,t)),)))
                addmulti!(out,value(a,t),UInt(basis(a)),Val(N))
                return Multivector{V}(out)
            end; else quote
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                addmulti!(out,$left(value(b,t)),UInt(basis(b)),Val(N))
                return Multivector{V}(out)
            end end end
        end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:Multivector{V,T}},op,swap=false) where {V,G,T}
        left,right,VEC = addvec(a,b,swap,op)
        if mdims(V)<cache_limit
            $(insert_expr((:N,:bs,:bn,:t),:mvec)...)
            out = svec(N,Any)(zeros(svec(N,t<:Number ? t : Int)))
            X = UInt(basis(a))
            for g ∈ list(1,N+1)
                ib = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    B = @inbounds ib[i]
                    val = :(@inbounds $right(b.v[$(bs[g]+i)]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    setmulti!_pre(out,val,B,Val(N))
                end
            end
            return :(Multivector{V}($(Expr(:call,tvec(N,t<:Number ? t : Any),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t),VEC)...)
            out = convert($VEC(N,t),$(bcast(right,:(value(b,$VEC(N,t)),))))
            addmulti!(out,value(a,t),UInt(basis(a)),Val(N))
            return Multivector{V}(out)
        end; else quote
            $(insert_expr((:N,:t),VEC)...)
            out = value(a,$VEC(N,t))
            addmulti!(out,$left(value(b,t)),UInt(basis(b)),Val(N))
            return Multivector{V}(out)
        end end end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:Spinor{V,T}},op,swap=false) where {V,G,T}
        left,right,VEC = addvec(a,b,swap,op)
        VECS = Symbol(string(VEC)*"s")
        !iseven(G) && (return swap ? :($op(Multivector(b),a)) : :($op(a,Multivector(b))))
        if mdims(V)<cache_limit
            $(insert_expr((:N,:rs,:bn,:t),:mvecs)...)
            out = svecs(N,Any)(zeros(svecs(N,t<:Number ? t : Int)))
            X = UInt(basis(a))
            for g ∈ evens(1,N+1)
                ib = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    B = @inbounds ib[i]
                    val = :(@inbounds $right(b.v[$(rs[g]+i)]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    setspin!_pre(out,val,B,Val(N))
                end
            end
            return :(Spinor{V}($(Expr(:call,tvecs(N,t<:Number ? t : Any),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t),VEC)...)
            out = convert($VECS(N,t),$(bcast(right,:(value(b,$VECS(N,t)),))))
            addspin!(out,value(a,t),UInt(basis(a)),Val(N))
            return Spinor{V}(out)
        end; else quote
            $(insert_expr((:N,:t),VEC)...)
            out = value(a,$VECS(N,t))
            addspin!(out,$left(value(b,t)),UInt(basis(b)),Val(N))
            return Spinor{V}(out)
        end end end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:CoSpinor{V,T}},op,swap=false) where {V,G,T}
        left,right,VEC = addvec(a,b,swap,op)
        VECS = Symbol(string(VEC)*"s")
        !isodd(G) && (return swap ? :($op(Multivector(b),a)) : :($op(a,Multivector(b))))
        if mdims(V)<cache_limit
            $(insert_expr((:N,:ps,:bn,:t),:mvecs)...)
            out = svecs(N,Any)(zeros(svecs(N,t<:Number ? t : Int)))
            X = UInt(basis(a))
            for g ∈ evens(2,N+1)
                ib = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    B = @inbounds ib[i]
                    val = :(@inbounds $right(b.v[$(ps[g]+i)]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    setanti!_pre(out,val,B,Val(N))
                end
            end
            return :(CoSpinor{V}($(Expr(:call,tvecs(N,t<:Number ? t : Any),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t),VEC)...)
            out = convert($VECS(N,t),$(bcast(right,:(value(b,$VECS(N,t)),))))
            addanti!(out,value(a,t),UInt(basis(a)),Val(N))
            return CoSpinor{V}(out)
        end; else quote
            $(insert_expr((:N,:t),VEC)...)
            out = value(a,$VECS(N,t))
            addpseudo(out,$left(value(b,t)),UInt(basis(b)),Val(N))
            return CoSpinor{V}(out)
        end end end
    end
    #=@noinline function adder(a::Type{<:Chain{V,G,T}},b::Type{<:Chain{V,L,S}},op) where {V,G,T,L,S}
        (G == 0 || G == mdims(V)) && (return :($op(Single(a),b)))
        (L == 0 || L == mdims(V)) && (return :($op(a,Single(b))))
        ((isodd(G) && isodd(L))||(iseven(G) && iseven(L))) && (return :($op(multispin(a),multispin(b))))
        return :($op(Multivector{V}(a),Multivector{V}(b)))
        #=left,right,VEC = addvec(a,b,false,op)
        quote # improve this
            $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
            @inbounds out[list(r+1,r+bng)] = value(a,$VEC(N,G,t))
            rb = binomsum(N,L)
            Rb = binomial(N,L)
            @inbounds out[list(rb+1,rb+Rb)] = $(bcast(right,:(value(b,$VEC(N,L,t)),)))
            return Multivector{V}(out)
        end=#
    end=#
    @noinline function adder(a::Type{<:Chain{V,G,T}},b::Type{<:Chain{V,G,S}},op) where {V,G,T,S}
        left,right,VEC = addvec(a,b,false,op)
        :(return Chain{V,G}($(bcast(right,:(a.v,b.v)))))
    end
    @noinline function adder(a::Type{<:Multivector{V,T}},b::Type{<:Multivector{V,S}},op) where {V,T,S}
        left,right,VEC = addvec(a,b,false,op)
        :(return Multivector{V}($(bcast(right,:(a.v,b.v)))))
    end
    @noinline function adder(a::Type{<:Spinor{V,T}},b::Type{<:Spinor{V,S}},op) where {V,T,S}
        left,right,VEC = addvec(a,b,false,op)
        :(return Spinor{V}($(bcast(right,:(a.v,b.v)))))
    end
    @noinline function adder(a::Type{<:CoSpinor{V,T}},b::Type{<:CoSpinor{V,S}},op) where {V,T,S}
        left,right,VEC = addvec(a,b,false,op)
        :(return CoSpinor{V}($(bcast(right,:(a.v,b.v)))))
    end
    #=@noinline function adder(a::Type{<:Chain{V,G,T}},b::Type{<:Multivector{V,S}},op,swap=false) where {V,G,T,S}
        return :($op(Multivector{V}(a),b))
        #=left,right,VEC = addvec(a,b,false,op)
        quote
            $(insert_expr((:N,:t,:r,:bng),VEC)...)
            out = convert($VEC(N,t),$(bcast(right,:(value(b,$VEC(N,t)),))))
            @inbounds $(add_val(:(+=),:(out[list(r+1,r+bng)]),:(value(a,$VEC(N,G,t))),left))
            return Multivector{V}(out)
        end=#
    end
    @noinline function adder(a::Type{<:Multivector{V,T}},b::Type{<:Chain{V,G,S}},op) where {V,G,T,S}
        return :($op(a,Multivector{V}(b)))
        #=left,right,VEC = addvec(a,b,false,op)
        quote
            $(insert_expr((:N,:t,:r,:bng),VEC)...)
            out = value(a,$VEC(N,t))
            @inbounds $(add_val(op≠:+ ? :(-=) : :(+=),:(out[list(r+1,r+bng)]),:(value(b,$VEC(N,G,t))),right))
            return Multivector{V}(out)
        end=#
    end
    @noinline function adder(a::Type{<:Chain{V,G,T}},b::Type{<:Spinor{V,S}},op,swap=false) where {V,G,T,S}
        if iseven(G)
            return :($op(multispin(a),b))
            #=left,right,VEC = addvec(a,b,false,op)
            VECS = Symbol(string(VEC)*"s")
            quote
                $(insert_expr((:N,:t,:rr,:bng),VEC)...)
                out = convert($VECS(N,t),$(bcast(right,:(value(b,$VECS(N,t)),))))
                @inbounds $(add_val(:(+=),:(out[list(rr+1,rr+bng)]),:(value(a,$VEC(N,G,t))),left))
                return Spinor{V}(out)
            end=#
        else
            :(return $op(a,Multivector{V}(b)))
        end
    end
    @noinline function adder(a::Type{<:Spinor{V,T}},b::Type{<:Chain{V,G,S}},op) where {V,G,T,S}
        if iseven(G)
            return :($op(a,multispin(b)))
            #=left,right,VEC = addvec(a,b,false,op)
            VECS = Symbol(string(VEC)*"s")
            quote
                $(insert_expr((:N,:t,:rr,:bng),VEC)...)
                out = value(a,$VECS(N,t))
                @inbounds $(add_val(op≠:+ ? :(-=) : :(+=),:(out[list(rr+1,rr+bng)]),:(value(b,$VEC(N,G,t))),right))
                return Spinor{V}(out)
            end=#
        else
            :(return $op(Multivector{V}(a),b))
        end
    end
    @noinline function adder(a::Type{<:Chain{V,G,T}},b::Type{<:CoSpinor{V,S}},op,swap=false) where {V,G,T,S}
        if isodd(G)
            return :($op(multispin(a),b))
            #=left,right,VEC = addvec(a,b,false,op)
            VECS = Symbol(string(VEC)*"s")
            quote
                $(insert_expr((:N,:t,:rrr,:bng),VEC)...)
                out = convert($VECS(N,t),$(bcast(right,:(value(b,$VECS(N,t)),))))
                @inbounds $(add_val(:(+=),:(out[list(rrr+1,rrr+bng)]),:(value(a,$VEC(N,G,t))),left))
                return CoSpinor{V}(out)
            end=#
        else
            :(return $op(a,Multivector{V}(b)))
        end
    end
    @noinline function adder(a::Type{<:CoSpinor{V,T}},b::Type{<:Chain{V,G,S}},op) where {V,G,T,S}
        if isodd(G)
            return :($op(a,multispin(b)))
            #=left,right,VEC = addvec(a,b,false,op)
            VECS = Symbol(string(VEC)*"s")
            quote
                $(insert_expr((:N,:t,:rrr,:bng),VEC)...)
                out = value(a,$VECS(N,t))
                @inbounds $(add_val(op≠:+ ? :(-=) : :(+=),:(out[list(rrr+1,rrr+bng)]),:(value(b,$VEC(N,G,t))),right))
                return CoSpinor{V}(out)
            end=#
        else
            :(return $op(Multivector{V}(a),b))
        end
    end=#
    @noinline function product(a::Type{S},b::Type{<:Chain{V,G,T}},swap=false,field=false) where S<:TensorGraded{V,L} where {V,G,L,T}
        MUL,VEC = mulvecs(a,b)
        vfield = Val(field)
        anti = isodd(L) ≠ isodd(G)
        type = anti ? :CoSpinor : :Spinor
        args = field ? (:g,) : ()
        (S<:Zero || S<:Infinity) && (return :a)
        if G == 0
            return S<:Chain ? :(Chain{V,L}(broadcast($MUL,a.v,Ref(@inbounds b[1])))) : swap ? :(Single(b)⟑a) : :(a⟑Single(b))
        elseif S<:Chain && L == 0
            return :(Chain{V,G}(broadcast($MUL,Ref(@inbounds a[1]),b.v)))
        elseif (swap ? L : G) == mdims(V) && !istangent(V)
            return swap ? (S<:Single ? :(⋆(~b,$(args...))*value(a)) : S<:Chain ? :(@inbounds a[1]*⋆(~b,$(args...))) : :(⋆(~b,$(args...)))) : :(@inbounds ⋆(~a,$(args...))*b[1])
        elseif (swap ? G : L) == mdims(V) && !istangent(V)
            return swap ? :(b[1]*complementlefthodge(~a,$(args...))) : S<:Single ? :(value(a)*complementlefthodge(~b,$(args...))) : S<:Chain ? :(@inbounds a[1]*complementlefthodge(~b,$(args...))) : :(complementlefthodge(~b,$(args...)))
        elseif binomial(mdims(V),G)*(S<:Chain ? binomial(mdims(V),L) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:bng,:ib,:μ),:mvecs)...)
                out = svecs(N,Any)(zeros(svecs(N,t)))
                B = indexbasis(N,L)
                for i ∈ list(1,binomial(N,L))
                    @inbounds v,ibi = :(@inbounds a[$i]),B[i]
                    for j ∈ 1:bng
                        @inbounds (anti ? geomaddanti!_pre : geomaddspin!_pre)(V,out,ibi,ib[j],derive_pre(V,ibi,ib[j],v,:(@inbounds b[$j]),MUL),vfield)
                    end
                end
            else
                $(insert_expr((:N,:t,:ib,:μ),:mvecs)...)
                out = svecs(N,Any)(zeros(svecs(N,t)))
                U = UInt(basis(a))
                for i ∈ list(1,binomial(N,G))
                    A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                    if S<:Single
                        @inbounds (anti ? geomaddanti!_pre : geomaddspin!_pre)(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL),vfield)
                    else
                        @inbounds (anti ? geomaddanti!_pre : geomaddspin!_pre)(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false),vfield)
                    end
                end
            end
            return :($type{V}($(Expr(:call,tvecs(N,t),out...))))
        elseif S<:Chain; return quote
            $(insert_expr((:N,:t,:bng,:ib,:μ),VEC)...)
            out = zeros($VEC(N,t))
            B = indexbasis(N,L)
            for i ∈ 1:binomial(N,L)
                @inbounds v,ibi = a[i],B[i]
                v≠0 && for j ∈ 1:bng
                    if @inbounds $(anti ? :geomaddanti! : :geomaddspin!)(V,out,ibi,ib[j],derive_mul(V,ibi,ib[j],v,b[j],$MUL))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds $(anti ? :geomaddanti! : :geomaddspin!)(V,out,ibi,ib[j],derive_mul(V,ibi,ib[j],v,b[j],$MUL))
                    end
                end
            end
            return $type{V}(out)
        end else return quote
            $(insert_expr((:N,:t,:out,:ib,:μ),VEC)...)
            U = UInt(basis(a))
            for i ∈ 1:binomial(N,G)
                A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                $(if S<:Single
                    :(if @inbounds $(anti ? :geomaddanti! : :geomaddspin!)(V,out,A,B,derive_mul(V,A,B,a.v,b[i],$MUL))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds $(anti ? :geomaddanti! : :geomaddspin!)(V,out,A,B,derive_mul(V,A,B,a.v,b[i],$MUL))
                    end)
                else
                    :(if @inbounds $(anti ? :geomaddanti! : :geomaddspin!)(V,out,A,B,derive_mul(V,A,B,b[i],false))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds $(anti ? :geomaddanti! : :geomaddspin!)(V,out,A,B,derive_mul(V,A,B,b[i],false))
                    end)
                end)
            end
            return $type{V}(out)
        end end
    end
    @noinline function product_contraction(a::Type{S},b::Type{<:Chain{V,G,T}},swap=false,field=false,contr=:contraction) where S<:TensorGraded{V,L} where {V,G,T,L}
        MUL,VEC = mulvec(a,b,contr)
        vfield = Val(field); args = field ? (:g,) : ()
        (swap ? G<L : L<G) && (!istangent(V)) && (return Zero(V))
        (S<:Zero || S<:Infinity) && (return :a)
        if (G==0 || G==mdims(V)) && (!istangent(V))
            return swap ? :(contraction(Single(b),a,$(args...))) : :(contraction(a,Single(b),$(args...)))
        end
        GL = swap ? G-L : L-G
        if binomial(mdims(V),G)*(S<:Chain ? binomial(mdims(V),L) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:bng,:bnl),:mvec)...)
                μ = istangent(V)|hasconformal(V)
                ia = indexbasis(N,L)
                ib = indexbasis(N,G)
                out = (μ ? svec(N,Any) : svec(N,GL,Any))(zeros(μ ? svec(N,t) : svec(N,GL,t)))
                for i ∈ list(1,bnl)
                    @inbounds v,iai = :(@inbounds a[$i]),ia[i]
                    for j ∈ list(1,bng)
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(@inbounds b[$j]),MUL),vfield)
                        else
                            @inbounds skewaddblade!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(@inbounds b[$j]),MUL),vfield)
                        end
                    end
                end
            else
                $(insert_expr((:N,:t,:ib,:bng,:μ),:mvec)...)
                out = (μ ? svec(N,Any) : svec(N,GL,Any))(zeros(μ ? svec(N,t) : svec(N,GL,t)))
                U = UInt(basis(a))
                for i ∈ list(1,bng)
                    A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                    if S<:Single
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL),vfield)
                        else
                            @inbounds skewaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL),vfield)
                        end
                    else
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false),vfield)
                        else
                            @inbounds skewaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false),vfield)
                        end
                    end
                end
            end
            #return :(value_diff(Single{V,0,$(getbasis(V,0))}($(value(mv)))))
            return if μ
                :(Multivector{$V}($(Expr(:call,istangent(V) ? tvec(N) : tvec(N,t),out...))))
            else
                :(value_diff(Chain{$V,$GL}($(Expr(:call,tvec(N,GL,t),out...)))))
            end
        elseif S<:Chain; return quote
            $(insert_expr((:N,:t,:bng,:bnl,:μ),VEC)...)
            ia = indexbasis(N,L)
            ib = indexbasis(N,G)
            out = zeros(μ ? $VEC(N,t) : $VEC(N,$GL,t))
            for i ∈ 1:bnl
                @inbounds v,iai = a[i],ia[i]
                v≠0 && for j ∈ 1:bng
                    if μ
                        if @inbounds skewaddmulti!(V,out,iai,ib[j],derive_mul(V,iai,ib[j],v,b[j],$MUL))
                            out,t = zeros(svec(N,Any)) .+ out,Any
                            @inbounds skewaddmulti!(V,out,iai,ib[j],derive_mul(V,iai,ib[j],v,b[j],$MUL))
                        end
                    else
                        @inbounds skewaddblade!(V,out,iai,ib[j],derive_mul(V,iai,ib[j],v,b[j],$MUL))
                    end
                end
            end
            return μ ? Multivector{V}(out) : value_diff(Chain{V,G-L}(out))
        end else return quote
            $(insert_expr((:N,:t,:ib,:bng,:μ),VEC)...)
            out = zeros(μ ? $VEC(N,t) : $VEC(N,$GL,t))
            U = UInt(basis(a))
            for i ∈ 1:bng
                if μ
                    A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                    $(if S<:Single
                        :(if @inbounds skewaddmulti!(V,out,A,B,derive_mul(V,A,B,b[i],$MUL))
                            #$(insert_expr((:out,);mv=:(value(mv)))...)
                            out,t = zeros(svec(N,Any)) .+ out,Any
                            @inbounds skewaddmulti!(V,out,A,B,derive_mul(V,A,B,b[i],$MUL))
                        end)
                    else
                        :(if @inbounds skewaddmulti!(V,out,A,B,derive_mul(V,A,B,b[i],false))
                            #$(insert_expr((:out,);mv=:(value(mv)))...)
                            out,t = zeros(svec(N,Any)) .+ out,Any
                            @inbounds skewaddmulti!(V,out,A,B,derive_mul(V,A,B,b[i],false))
                        end)
                    end)
                else
                    @inbounds skewaddblade!(V,out,A,B,derive_mul(V,A,B,b[i],false))
                end
            end
            return μ ? Multivector{V}(out) : value_diff(Chain{V,$GL}(out))
        end end
    end
end

for (op,po,GL,grass) ∈ ((:∧,:>,:(G+L),:exter),(:∨,:<,:(G+L-mdims(V)),:meet))
    grassaddmulti! = Symbol(grass,:addmulti!)
    grassaddblade! = Symbol(grass,:addblade!)
    grassaddmulti!_pre = Symbol(grassaddmulti!,:_pre)
    grassaddblade!_pre = Symbol(grassaddblade!,:_pre)
    prop = Symbol(:product_,op)
    @eval @noinline function $prop(a::Type{S},b::Type{<:Chain{R,L,T}},swap=false) where S<:TensorGraded{Q,G} where {Q,R,T,G,L}
        MUL,VEC = mulvec(a,b)
        w,W = swap ? (R,Q) : (Q,R)
        V = w==W ? w : ((w==dual(W)) ? (dyadmode(w)≠0 ? W⊕w : w⊕W) : (return :(interop($$op,a,b))))
        $po(G+L,mdims(V)) && (!istangent(V)) && (return Zero(V))
        (S<:Zero || S<:Infinity) && (return :a)
        if (L==0 || L==mdims(V)) && (!istangent(V))
            return swap ? :($$op(Single(b),a)) : :($$op(a,Single(b)))
        end
        if binomial(mdims(W),L)*(S<:Chain ? binomial(mdims(w),G) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:μ),:mvec,:T,:S)...)
                t = promote_type(valuetype(a),valuetype(b))
                ia = indexbasis(mdims(w),G)
                ib = indexbasis(mdims(W),L)
                out = (μ ? svec(N,Any) : svec(N,$GL,Any))(zeros(μ ? svec(N,t) : svec(N,$GL,t)))
                CA,CB = isdual(w),isdual(W)
                for i ∈ list(1,binomial(mdims(w),G))
                    @inbounds v,iai = :(@inbounds a[$i]),ia[i]
                    x = CA ? dual(V,iai) : iai
                    for j ∈ list(1,binomial(mdims(W),L))
                        X = @inbounds CB ? dual(V,ib[j]) : ib[j]
                        if μ
                            $grassaddmulti!_pre(V,out,x,X,derive_pre(V,x,X,v,:(@inbounds b[$j]),MUL))
                        else
                            $grassaddblade!_pre(V,out,x,X,derive_pre(V,x,X,v,:(@inbounds b[$j]),MUL))
                        end
                    end
                end
            else
                $(insert_expr((:N,:μ),:mvec,Int,:T)...)
                t = promote_type(valuetype(a),valuetype(b))
                ib = indexbasis(mdims(R),L)
                out = (μ ? svec(N,Any) : svec(N,$GL,Any))(zeros(μ ? svec(N,t) : svec(N,$GL,t)))
                C,x = isdual(R),isdual(Q) ? dual(V,UInt(basis(a))) : UInt(basis(a))
                for i ∈ list(1,binomial(mdims(W),L))
                    X = @inbounds C ? dual(V,ib[i]) : ib[i]
                    A,B = swap ? (X,x) : (x,X)
                    if S<:Single
                        if μ
                            $grassaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL))
                        else
                            $grassaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL))
                        end
                    else
                        if μ
                            $grassaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false))
                        else
                            $grassaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false))
                        end
                    end
                end
            end
            return if μ
                :(Multivector{$V}($(Expr(:call,istangent(V) ? tvec(N) : tvec(N,t),out...))))
            else
                :(Chain{$V,$$GL}($(Expr(:call,tvec(N,$GL,t),out...))))
            end
        elseif S<:Chain; return quote
            V = $V
            $(insert_expr((:N,:t,:μ),VEC)...)
            ia = indexbasis(mdims(w),G)
            ib = indexbasis(mdims(W),L)
            out = zeros(μ ? $VEC(N,t) : $VEC(N,$$GL,t))
            CA,CB = isdual(w),isdual(W)
            for i ∈ 1:binomial(mdims(w),G)
                @inbounds v,iai = a[i],ia[i]
                x = CA ? dual(V,iai) : iai
                v≠0 && for j ∈ 1:binomial(mdims(W),L)
                    X = @inbounds CB ? dual(V,ib[j]) : ib[j]
                    if μ
                        if @inbounds $$grassaddmulti!(V,out,x,X,derive_mul(V,x,X,v,b[j],$MUL))
                            out,t = zeros(svec(N,promote_type,Any)) .+ out,Any
                            @inbounds $$grassaddmulti!(V,out,x,X,derive_mul(V,x,X,v,b[j],$MUL))
                        end
                    else
                        @inbounds $$grassaddblade!(V,out,x,X,derive_mul(V,x,X,v,b[j],$MUL))
                    end
                end
            end
            return μ ? Multivector{V}(out) : Chain{V,$$GL}(out)
        end else return quote
            V = $V
            $(insert_expr((:N,:t,:μ),VEC)...)
            ib = indexbasis(mdims(R),L)
            out = zeros(μ ? $VEC(N,t) : $VEC(N,$$GL,t))
            C,x = isdual(R),isdual(Q) ? dual(V,UInt(basis(a))) : UInt(basis(a))
            for i ∈ 1:binomial(mdims(W),L)
                X = @inbounds C ? dual(V,ib[i]) : ib[i]
                A,B = (X,x) : (x,X)
                $(if S<:Single
                    :(if μ
                        if @inbounds $$grassaddmulti!(V,out,A,B,derive_mul(V,A,B,a.v,b[i],false))
                            out,t = zeros(svec(N,Any)) .+ out,Any
                            @inbounds $$grassaddmulti!(V,out,A,B,derive_mul(V,A,B,a.v,b[i],false))
                        end
                    else
                        @inbounds $$grassaddblade!(V,out,A,B,derive_mul(V,A,B,a.v,b[i],false))
                    end)
                else
                    :(if μ
                        if @inbounds $$grassaddmulti!(V,out,A,B,derive_mul(V,A,B,b[i],$pro))
                            out,t = zeros(svec(N,Any)) .+ out,Any
                            @inbounds $$grassaddmulti!(V,out,A,B,derive_mul(V,A,B,b[i],$MUL))
                        end
                    else
                        @inbounds $$grassaddblade!(V,out,A,B,derive_mul(V,A,B,b[i],$MUL))
                    end)
                end)
            end
            return μ ? Multivector{V}(out) : Chain{V,$$GL}(out)
        end end
    end
end

for input ∈ (:Multivector,:Spinor,:CoSpinor)
    inspin,inanti = input==:Spinor,input==:CoSpinor
for (op,product) ∈ ((:∧,:exteradd),(:*,:geomadd),
                     (:∨,:meetadd),(:contraction,:skewadd))
    outspin = product ∈ (:exteradd,:geomadd,:skewadd)
    outmulti = input == :Multivector
    outype = outmulti ? :Multivector : outspin ? :($(inspin ? :isodd : :iseven)(G) ? CoSpinor : Spinor) : inspin ?  :(isodd(G)⊻isodd(N) ? CoSpinor : Spinor) : :(isodd(G)⊻isodd(N) ? Spinor : CoSpinor)
    product! = outmulti ? Symbol(product,:multi!) : outspin ? :($(inspin ? :isodd : :iseven)(G) ? $(Symbol(product,:anti!)) : $(Symbol(product,:spin!))) : :(isodd(G)⊻isodd(N) ? $(Symbol(product,outspin⊻inspin ? :anti! : :spin!)) : $(Symbol(product,outspin⊻inspin ? :spin! : :anti!)))
    preproduct! = outmulti ? Symbol(product,:multi!_pre) : outspin ? :($(inspin ? :isodd : :iseven)(G) ? $(Symbol(product,:anti!_pre)) : $(Symbol(product,:spin!_pre))) : :(isodd(G)⊻isodd(N) ? $(Symbol(product,outspin⊻inspin ? :anti!_pre : :spin!_pre)) : $(Symbol(product,outspin⊻inspin ? :spin!_pre : :anti!_pre)))
    prop = op≠:* ? Symbol(:product_,op) : :product
    outmulti && @eval $prop(a,b,swap=false) = $prop(typeof(a),typeof(b),swap)
    mgrade,nmgrade = op≠:∧ ? (:maxgrade,:nextmaxgrade) : (:mingrade,:nextmingrade)
    @eval @noinline function $prop(a::Type{S},b::Type{<:$input{V,T}},swap=false,field=false) where S<:TensorGraded{V,G} where {V,G,T}
        MUL,VEC = mulvec(a,b,$(QuoteNode(op)))
        VECS = isodd(G) ? VEC : string(VEC)*"s"
        args = field ? (:g,) : (); vfield = Val(field)
        (S<:Zero || S<:Infinity) && (return :a)
        $(if op ∈ (:∧,:∨); quote
            if $(op≠:∧ ? :(<(G+maxgrade(b),mdims(V))) : :(>(G+mingrade(b),mdims(V)))) && (!istangent(V))
                return Zero(V)
            elseif G+$mgrade(b)==mdims(V) && (!istangent(V))
                return swap ? :($$op(b(Val($$mgrade(b))),a)) : :($$op(a,b(Val($$mgrade(b)))))
            elseif G+$nmgrade(b)==mdims(V) && (!istangent(V))
                return swap ? :($$op(b(Val($$mgrade(b))),a)+$$op(b(Val($$nmgrade(b))),a)) : :($$op(a,b(Val($$mgrade(b))))+$$op(a,b(Val($$nmgrade(b)))))
            end
        end; elseif op == :contraction; quote
            if (swap ? maxgrade(b)<G : G<mingrade(b)) && (!istangent(V))
                return Zero(V)
            elseif (swap ? maxgrade(b)==G : G+maxpseudograde(b)==mdims(V)) && (!istangent(V))
                return swap ? :($$op(b(Val(maxgrade(b))),a,$(args...))) : :($$op(a,b(Val(mingrade(b))),$(args...)))
            elseif (swap ? nextmaxgrade(b)==G : G+nextmaxpseudograde(b)==mdims(V)) && (!istangent(V))
                return swap ? :($$op(b(Val(maxgrade(b))),a,$(args...))+$$op(b(Val(nextmaxgrade(b))),a,$(args...))) : :($$op(a,b(Val(mingrade(b))),$(args...))+$$op(a,b(Val(nextmingrade(b))),$(args...)))
            end
        end; elseif op == :*; quote
            if S<:Chain && G == 0
                return :($input{V,G}(broadcast($MUL,Ref(@inbounds a[1]),b.v)))
            elseif G == mdims(V) && !istangent(V)
                return if swap
                    S<:Single ? :(⋆(~b,$(args...))*value(a)) : S<:Chain ? :(@inbounds a[1]*⋆(~b,$(args...))) : :(⋆(~b,$(args...)))
                else
                    S<:Single ? :(value(a)*complementlefthodge(~b,$(args...))) : S<:Chain ? :(@inbounds a[1]*complementlefthodge(~b,$(args...))) : :(complementlefthodge(~b,$(args...)))
                end
            end
        end; else nothing end)
        if mdims(V)<cache_limit
            $(insert_expr((:N,:t,:ib,:bn,:μ))...)
            bs = $(inspin ? :spinsum_set : inanti ? :antisum_set : :binomsum_set)(N)
            out = $(outmulti ? :svec : :svecs)(N,Any)(zeros($(outmulti ? :svec : :svecs)(N,t)))
            for g ∈ $(inspin ? :(evens(1,N+1)) : inanti ? :(evens(2,N+1)) : :(list(1,N+1)))
                ia = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    if S<:Chain
                        @inbounds val = :(@inbounds b.v[$(bs[g]+i)])
                        @inbounds for j ∈ 1:bn[G+1]
                            @inbounds A,B = swapper(ib[j],ia[i],swap)
                            X,Y = swapper(:(@inbounds a[$j]),val,swap)
                            $preproduct!(V,out,A,B,derive_pre(V,A,B,X,Y,MUL),vfield)
                        end
                    else
                        @inbounds A,B = swapper(UInt(basis(a)),ia[i],swap)
                        if S<:Single
                            X,Y = swapper(:(a.v),:(@inbounds b.v[$(bs[g]+i)]),swap)
                            $preproduct!(V,out,A,B,derive_pre(V,A,B,X,Y,MUL),vfield)
                        else
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,:(@inbounds b.v[$(bs[g]+i)]),false),vfield)
                        end
                    end
                end
            end
            return :($$outype{V}($(Expr(:call,$(outmulti ? :tvec : :tvecs)(N,t),out...))))
        else return quote
            $(insert_expr((:N,:t,:ib,:bn,:μ),VECS)...)
            out = zeros($(outmulti ? :svec : :svecs)(N,t)) # VECS
            bs = $(inspin ? :spinsum_set : inanti ? :antisum_set : :binomsum_set)(N)
            for g ∈ $(inspin ? :(1:2:N+1) : inanti ? :(2:2:N+1) : :(1:N+1))
                ia = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    $(if S<:Chain; quote
                        @inbounds val = b.v[bs[g]+i]
                        val≠0 && for j ∈ 1:bn[G+1]
                            A,B = $(swap ? :(@inbounds (ia[i],ib[j])) : :(@inbounds (ib[j],ia[i])))
                            X,Y = $(swap ? :((val,@inbounds a[j])) : :((@inbounds a[j],val)))
                            dm = derive_mul(V,A,B,X,Y,$MUL)
                            if $$product!(V,out,A,B,dm)&μ
                                $(insert_expr((:out,);mv=:out)...)
                                $$product!(V,out,A,B,dm)
                            end
                        end end
                    else quote
                        A,B = $(swap ? :((@inbounds ia[i],$(UInt(basis(a))))) : :(($(UInt(basis(a))),@inbounds ia[i])))
                        $(if S<:Single; quote
                            X,Y=$(swap ? :((b.v[bs[g]+1],a.v)) : :((a.v,@inbounds b.v[rs[g]+1])))
                            dm = derive_mul(V,A,B,X,Y,$MUL)
                            if $$product!(V,out,A,B,dm)&μ
                                $(insert_expr((:out,);mv=:out)...)
                                $$product!(V,out,A,B,dm)
                            end end
                        else
                            :(if @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,b.v[rs[g]+i],false))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,b.v[rs[g]+i],false))
                            end)
                        end) end
                    end)
                end
            end
            return $$outype{V}(out)
        end end
    end
end
end

for input ∈ (:Spinor,:CoSpinor)
    inspin = input==:Spinor
    outype = :($(inspin ? :isodd : :iseven)(G) ? CoSpinor : Spinor)
    product! = :($(inspin ? :isodd : :iseven)(G) ? geomaddanti! : geomaddspin!)
    preproduct! = :($(inspin ? :isodd : :iseven)(G) ? geomaddanti!_pre : geomaddspin!_pre)
    product2! = :(isodd(G) ? geomaddanti! : geomaddspin!)
    preproduct2! = :(isodd(G) ? geomaddanti!_pre : geomaddspin!_pre)
    input≠:Spinor && @eval product_sandwich(a,b,swap=false) = product_sandwich(typeof(a),typeof(b),swap)
    @eval @noinline function product_sandwich(a::Type{S},b::Type{<:$input{V,T}},swap=false,field=false) where S<:TensorGraded{V,G} where {V,G,T}
        MUL,VEC = mulvec(a,b,:*)
        VECS = isodd(G) ? VEC : string(VEC)*"s"
        args = field ? (:g,) : (); vfield = Val(field)
        if mdims(V)<cache_limit
            $(insert_expr((:N,:t,:ib,:bn,:μ))...)
            bs = $(inspin ? :spinsum_set : :antisum_set)(N)
            out = svecs(N,Any)(zeros(svecs(N,t)))
            for g ∈ $(inspin ? :(evens(1,N+1)) : :(evens(2,N+1)))
                ia = indexbasis(N,g-1)
                par = swap ? false : parityclifford(g-1)
                @inbounds for i ∈ 1:bn[g]
                    @inbounds val = par ? :(@inbounds -b.v[$(bs[g]+i)]) : :(@inbounds b.v[$(bs[g]+i)])
                    if S<:Chain
                        @inbounds for j ∈ 1:bn[G+1]
                            @inbounds A,B = ia[i],ib[j]
                            $preproduct!(V,out,A,B,derive_pre(V,A,B,val,:(@inbounds a[$j]),MUL),vfield)
                        end
                    else
                        @inbounds A,B = ia[i],UInt(basis(a))
                        if S<:Single
                            $preproduct!(V,out,A,B,derive_pre(V,A,B,val,:(a.v),MUL),vfield)
                        else
                            $preproduct!(V,out,A,B,derive_pre(V,A,B,val,false),vfield)
                        end
                    end
                end
            end
            bs2 = ($(inspin ? :isodd : :iseven)(G) ? antisum_set : spinsum_set)(N)
            out2 = svecs(N,Any)(zeros(svecs(N,t)))
            for g ∈ ($(inspin ? :isodd : :iseven)(G) ? evens(2,N+1) : evens(1,N+1))
                ia = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    @inbounds val = out[bs2[g]+i]
                    !isnull(val) && for g2 ∈ $(inspin ? :(evens(1,N+1)) : :(evens(2,N+1)))
                        io = indexbasis(N,g2-1)
                        par = swap ? parityclifford(g2-1) : false
                        @inbounds for j ∈ 1:bn[g2]
                            val2 = :(b.v[$(bs[g2]+j)])
                            @inbounds A,B = ia[i],io[j]
                            Y = par ? :(@inbounds -$val2) : :(@inbounds $val2)
                            $preproduct2!(V,out2,A,B,derive_pre(V,A,B,val,Y,MUL),vfield)
                        end
                    end
                end
            end
            bs3 = (isodd(G) ? antisum : spinsum)(N,G)
            return :(Chain{V,G}($(Expr(:call,tvec(N,G,t),out2[bs3+1:bs3+binomial(N,G)]...))))
        #=else return quote
            $(insert_expr((:N,:t,:ib,:bn,:μ),VECS)...)
            out = zeros(svecs(N,t)) # VECS
            bs = $(inspin ? :spinsum_set : :antisum_set)(N)
            for g ∈ $(inspin ? :(1:2:N+1) : :(2:2:N+1))
                ia = indexbasis(N,g-1)
                par = parityclifford(g-1)
                @inbounds for i ∈ 1:bn[g]
                    $(if S<:Chain; quote
                        @inbounds val = par ? -b.v[bs[g]+i] : b.v[bs[g]+i]
                        val≠0 && for j ∈ 1:bn[G+1]
                            A,B = $(!swap ? :(@inbounds (ia[i],ib[j])) : :(@inbounds (ib[j],ia[i])))
                            X,Y = $(!swap ? :((val,@inbounds a[j])) : :((@inbounds a[j],val)))
                            dm = derive_mul(V,A,B,X,Y,$MUL)
                            if @inbounds $$product!(V,out,A,B,dm)&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,dm)
                            end
                        end end
                    else quote
                        A,B = $(swap ? :((@inbounds ia[i],$(UInt(basis(a))))) : :(($(UInt(basis(a))),@inbounds ia[i])))
                        $(if S<:Single; quote
                            X,Y=$(swap ? :((b.v[bs[g]+1],a.v)) : :((a.v,@inbounds b.v[rs[g]+1])))
                            dm = derive_mul(V,A,B,X,Y,$MUL)
                            if @inbounds $$product!(V,out,A,B,dm)&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,dm)
                            end end
                        else
                            :(if @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,b.v[rs[g]+i],false))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,b.v[rs[g]+i],false))
                            end)
                        end) end
                    end)
                end
            end
            return $$outype{V}(out)
        end=# end
    end
end

for input ∈ (:Chain,)
    product! = :((iseven(L) ? isodd : iseven)(G) ? geomaddanti! : geomaddspin!)
    preproduct! = :((iseven(L) ? isodd : iseven)(G) ? geomaddanti!_pre : geomaddspin!_pre)
    product2! = :(isodd(G) ? geomaddanti! : geomaddspin!)
    preproduct2! = :(isodd(G) ? geomaddanti!_pre : geomaddspin!_pre)
    @eval @noinline function product_sandwich(a::Type{S},b::Type{Q},swap=false,field=false) where {S<:TensorGraded{V,G},Q<:TensorGraded{V,L}} where {V,G,L}
        MUL,VEC = mulvec(a,b,:*)
        VECS = isodd(G) ? VEC : string(VEC)*"s"
        args = field ? (:g,) : (); vfield = Val(field)
        if mdims(V)<cache_limit
            $(insert_expr((:N,:t,:ib,:bn,:μ))...)
            il = indexbasis(N,L)
            bs = (iseven(L) ? spinsum_set : antisum_set)(N)
            out = svecs(N,Any)(zeros(svecs(N,t)))
            par = parityclifford(L)
            if Q <: Chain
                @inbounds for i ∈ 1:bn[L+1]
                    @inbounds val = (swap ? false : par) ? :(@inbounds -b.v[$i]) : :(@inbounds b.v[$i])
                    if S<:Chain
                        @inbounds for j ∈ 1:bn[G+1]
                            @inbounds A,B = il[i],ib[j]
                            $preproduct!(V,out,A,B,derive_pre(V,A,B,val,:(@inbounds a[$j]),MUL),vfield)
                        end
                    else
                        @inbounds A,B = il[i],UInt(basis(a))
                        if S<:Single
                            $preproduct!(V,out,A,B,derive_pre(V,A,B,val,:(a.v),MUL),vfield)
                        else
                            $preproduct!(V,out,A,B,derive_pre(V,A,B,val,false),vfield)
                        end
                    end
                end
            else
                A = UInt(basis(b))
                val = (swap ? false : par) ? :(-value(b)) : :(value(b))
                if S<:Chain
                    @inbounds for j ∈ 1:bn[G+1]
                        @inbounds B = ib[j]
                        $preproduct!(V,out,A,B,derive_pre(V,A,B,val,:(@inbounds a[$j]),MUL),vfield)
                    end
                else
                    B = UInt(basis(a))
                    if S<:Single
                        $preproduct!(V,out,A,B,derive_pre(V,A,B,val,:(a.v),MUL),vfield)
                    else
                        $preproduct!(V,out,A,B,derive_pre(V,A,B,val,false),vfield)
                    end
                end
            end
            bs2 = ((iseven(L) ? isodd : iseven)(G) ? antisum_set : spinsum_set)(N)
            out2 = svecs(N,Any)(zeros(svecs(N,t)))
            for g ∈ ((iseven(L) ? isodd : iseven)(G) ? evens(2,N+1) : evens(1,N+1))
                ia = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    @inbounds val = out[bs2[g]+i]
                    !isnull(val) && if Q<:Chain
                        @inbounds for j ∈ 1:bn[L+1]
                            @inbounds A,B = ia[i],il[j]
                            Y = (swap ? par : false) ? :(@inbounds -b[$j]) : :(@inbounds b[$j])
                            $preproduct2!(V,out2,A,B,derive_pre(V,A,B,val,Y,MUL),vfield)
                        end
                    else
                        @inbounds A,B = ia[i],UInt(basis(b))
                        if Q<:Single
                            Y = swapper((swap ? par : false) ? :(-value(b)) : :(value(b)),val,true)
                            $preproduct2!(V,out2,A,B,derive_pre(V,A,B,val,Y,MUL),vfield)
                        else
                            $preproduct2!(V,out2,A,B,derive_pre(V,A,B,val,false),vfield)
                        end
                    end
                end
            end
            bs3 = (isodd(G) ? antisum : spinsum)(N,G)
            return :(Chain{V,G}($(Expr(:call,tvec(N,G,t),out2[bs3+1:bs3+binomial(N,G)]...))))
        #=else return quote
        end=# end
    end
end

for input ∈ (:Couple,:PseudoCouple)
    product! = :((inspin ? isodd : iseven)(G) ? geomaddanti! : geomaddspin!)
    preproduct! = :((inspin ? isodd : iseven)(G) ? geomaddanti!_pre : geomaddspin!_pre)
    product2! = :(isodd(G) ? geomaddanti! : geomaddspin!)
    preproduct2! = :(isodd(G) ? geomaddanti!_pre : geomaddspin!_pre)
    calar = input == :Couple ? :scalar : :volume
    pg = input == :Couple ? 0 : :N
    @eval @noinline function product_sandwich(a::Type{S},b::Type{Q},swap=false,field=false) where {S<:TensorGraded{V,G},Q<:$input{V,BB}} where {V,G,BB}
        MUL,VEC = mulvec(a,b,:*)
        N = mdims(V)
        args = field ? (:g,) : (); vfield = Val(field)
        $(if input == :Couple
            :(isodd(grade(BB)) && return :($(swap ? :>>> : :⊘)(a,multispin(b),$(args...))))
        else
            :(isodd(grade(BB))≠isodd(N) && return :($(swap ? :>>> : :⊘)(a,multispin(b),$(args...))))
        end)
        inspin = iseven(grade(BB))
        VECS = isodd(G) ? VEC : string(VEC)*"s"
        if N<cache_limit
            $(insert_expr((:t,:ib,:bn,:μ))...)
            out = svecs(N,Any)(zeros(svecs(N,t)))
            for (A,val) ∈ ((UInt(BB),(swap ? false : parityclifford(grade(BB))) ? :(-value(imaginary(b))) : :(value(imaginary(b)))),(indexbasis(N,$pg)[1],(swap ? false : parityclifford($pg)) ? :(-value($$calar(b))) : :(value($$calar(b)))))
                if S<:Chain
                    @inbounds for j ∈ 1:bn[G+1]
                        @inbounds B = ib[j]
                        $preproduct!(V,out,A,B,derive_pre(V,A,B,val,:(@inbounds a[$j]),MUL),vfield)
                    end
                else
                    B = UInt(basis(a))
                    if S<:Single
                        $preproduct!(V,out,A,B,derive_pre(V,A,B,val,:(a.v),MUL),vfield)
                    else
                        $preproduct!(V,out,A,B,derive_pre(V,A,B,val,false),vfield)
                    end
                end
            end
            bs2 = ((inspin ? isodd : iseven)(G) ? antisum_set : spinsum_set)(N)
            out2 = svecs(N,Any)(zeros(svecs(N,t)))
            for g ∈ ((inspin ? isodd : iseven)(G) ? evens(2,N+1) : evens(1,N+1))
                ia = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    @inbounds val = out[bs2[g]+i]
                    !isnull(val) && for (B,val2) ∈ ((UInt(BB),(swap ? parityclifford(grade(BB)) : false) ? :(-value(imaginary(b))) : :(value(imaginary(b)))),(indexbasis(N,$pg)[1],(swap ? parityclifford($pg) : false) ? :(-value($$calar(b))) : :(value($$calar(b)))))
                        @inbounds A = ia[i]
                        $preproduct2!(V,out2,A,B,derive_pre(V,A,B,val,val2,MUL),vfield)
                    end
                end
            end
            bs3 = (isodd(G) ? antisum : spinsum)(N,G)
            return :(Chain{V,G}($(Expr(:call,tvec(N,G,t),out2[bs3+1:bs3+binomial(N,G)]...))))
        #=else return quote
        end=# end
    end
end

outsym(com) = com ∈ (:spinor,:anti) ? :tvecs : :tvec
function leftrightsym(com)
    ls = com ∈ (:spinor,:spin_multi,:s_m,:s_a)
    la = com ∈ (:anti,:anti_multi,:a_m,:a_s)
    rs = com ∈ (:spinor,:spin_multi,:m_s,:a_s)
    ra = com ∈ (:anti,:anti_multi,:m_a,:s_a)
    left = if ls
        (:rs,:(evens(1,N+1)))
    elseif la
        (:ps,:(evens(2,N+1)))
    else
        (:bs,:(list(1,N+1)))
    end
    right = if rs
        (:rs,:(evens(1,N+1)))
    elseif ra
        (:ps,:(evens(2,N+1)))
    else
        (:bs,:(list(1,N+1)))
    end
    br = if ls&&rs
        (:rs,)
    elseif la&&ra
        (:ps,)
    elseif (la&&rs)||(ls&&ra)
        (:rs,:ps)
    elseif (ls||rs)&!(la||ra)
        (:bs,:rs)
    elseif (la||ra)&!(ls||rs)
        (:bs,:ps)
    else
        (:bs,)
    end
    return (left...,right...,br)
end

function product_loop(V,type,loop,VEC)
    if mdims(V)<cache_limit/2
        return :($type{V}($(loop[2].args[2])))
    else return quote
        $(insert_expr(loop[1],VEC)...)
        $(loop[2])
        return $type{V,t}(out)
    end end
end

for com ∈ (:spinor,:s_m,:m_s,:anti,:a_m,:m_a,:multivector,:s_a,:a_s)
    outspin = com ∈ (:spinor,:anti,:s_a,:a_s)
    left,leftspin,right,rightspin,br = leftrightsym(com)
    VEC = outspin ? :svecs : :svec
    genloop = Symbol(:generate_loop_,com)
    @eval @noinline function $genloop(V,a,b,t,MUL,product!,preproduct!,field=false,d=nothing)
        $(insert_expr((:N,br...,:bn),:mvec)...)
        vfield = Val(field)
        if mdims(V)<cache_limit/2
            out = $VEC(N,Any)(zeros($VEC(N,t)))
            for g ∈ $leftspin
                X = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    Xi = @inbounds X[i]
                    @inbounds val = nothing≠d ? :(@inbounds $a[$($left[g]+i)]/$d) : :(@inbounds $a[$($left[g]+i)])
                    for G ∈ $rightspin
                        @inbounds R = $right[G]
                        Y = indexbasis(N,G-1)
                        @inbounds for j ∈ 1:bn[G]
                            Yj = @inbounds Y[j]
                            preproduct!(V,out,Xi,Yj,derive_pre(V,Xi,Yj,val,:(@inbounds $b[$(R+j)]),MUL),vfield)
                        end
                    end
                end
            end
            (:N,:t,:out), :(out = $(Expr(:call,$(outspin ? :tvecs : :tvec)(N,t),out...)))
        else
            (:N,:t,:out,$br...,:bn,:μ), quote
                $(nothing≠d ? :(out = zeros($$(outspin ? :mvecs : :mvec)(N,t))) : nothing)
                for g ∈ $$leftspin
                    X = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        Xi = @inbounds X[i]
                        @inbounds val = $(nothing≠d ? :(@inbounds $a[$$left[g]+i]/$d) : :(@inbounds $a[$$left[g]+i]))
                        val≠0 && for G ∈ $$rightspin
                            @inbounds R = $$right[G]
                            Y = indexbasis(N,G-1)
                            @inbounds for j ∈ 1:bn[G]
                                Yj = @inbounds Y[j]
                                dm = @inbounds derive_mul(V,Xi,Yj,val,$b[R+j],$MUL)
                                if $product!(V,out,Xi,Yj,dm)&μ
                                    $(insert_expr((:out,);mv=:out)...)
                                    $product!(V,out,Xi,Yj,dm)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
