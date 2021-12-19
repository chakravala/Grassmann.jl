
#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: +, -, *, ^, /, //, inv, <, >, <<, >>, >>>
import AbstractTensors: ∧, ∨, ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, ∗, rem, div, contraction, TAG, SUB
import Leibniz: diffcheck, diffmode, hasinforigin, hasorigininf, symmetricsplit
import Leibniz: loworder, isnull, g_one, g_zero, Field, ExprField
const Sym,SymField = :AbstractTensors,Any

## geometric product

"""
    *(ω::TensorAlgebra,η::TensorAlgebra)

Geometric algebraic product: ω⊖η = (-1)ᵖdet(ω∩η)⊗(Λ(ω⊖η)∪L(ω⊕η))
"""
@pure *(a::SubManifold{V},b::SubManifold{V}) where V = mul(a,b)
*(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = *(a*b,c...)

@pure function mul(a::SubManifold{V},b::SubManifold{V},der=derive_mul(V,UInt(a),UInt(b),1,true)) where V
    ba,bb = UInt(a),UInt(b)
    (diffcheck(V,ba,bb) || iszero(der)) && (return g_zero(V))
    A,B,Q,Z = symmetricmask(V,ba,bb)
    pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(V,A,B) : (false,A⊻B,false)
    d = getbasis(V,bas|Q)
    out = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(a,b)⊻pcc ? Simplex{V}(-1,d) : d) : Simplex{V}((pcc ? -1 : 1)*parityinner(V,A,B),d)
    diffvars(V)≠0 && !iszero(Z) && (out = Simplex{V}(getbasis(loworder(V),Z),out))
    return cc ? (v=value(out);out+Simplex{V}(hasinforigin(V,A,B) ? -(v) : v,getbasis(V,conformalmask(V)⊻UInt(d)))) : out
end

function *(a::Simplex{V},b::SubManifold{V}) where V
    v = derive_mul(V,UInt(basis(a)),UInt(b),a.v,true)
    bas = mul(basis(a),b,v)
    order(a.v)+order(bas)>diffmode(V) ? zero(V) : Simplex{V}(v,bas)
end
function *(a::SubManifold{V},b::Simplex{V}) where V
    v = derive_mul(V,UInt(a),UInt(basis(b)),b.v,false)
    bas = mul(a,basis(b),v)
    order(b.v)+order(bas)>diffmode(V) ? zero(V) : Simplex{V}(v,bas)
end

#*(a::MultiGrade{V},b::SubManifold{V}) where V = MultiGrade{V}(a.v,basis(a)*b)
#*(a::SubManifold{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*basis(b))
#*(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,basis(a)*basis(b))

*(a::UniformScaling,b::Simplex{V}) where V = V(a)*b
*(a::Simplex{V},b::UniformScaling) where V = a*V(b)
*(a::UniformScaling,b::Chain{V}) where V = V(a)*b
*(a::Chain{V},b::UniformScaling) where V = a*V(b)

export ∗, ⊛, ⊖
import AbstractTensors: ⊖, ⊘, ∗

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
@generated ∧(t::T) where T<:Values{N} where N = wedges([:(t[$i]) for i ∈ 1:N])
@generated ∧(t::T) where T<:FixedVector{N} where N = wedges([:(@inbounds t[$i]) for i ∈ 1:N])
∧(::Values{0,<:Chain{V}}) where V = one(V) # ∧() = 1
∧(::FixedVector{0,<:Chain{V}}) where V = one(V)
∧(t::Chain{V,1,<:Chain} where V) = ∧(value(t))
∧(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = ∧(a∧b,c...)

wedges(x,i=length(x)-1) = i ≠ 0 ? Expr(:call,:∧,wedges(x,i-1),x[1+i]) : x[1+i]

@pure function ∧(a::SubManifold{V},b::SubManifold{V}) where V
    ba,bb = UInt(a),UInt(b)
    A,B,Q,Z = symmetricmask(V,ba,bb)
    ((count_ones(A&B)>0) || diffcheck(V,ba,bb) || iszero(derive_mul(V,ba,bb,1,true))) && (return g_zero(V))
    d = getbasis(V,(A⊻B)|Q)
    diffvars(V)≠0 && !iszero(Z) && (d = Simplex{V}(getbasis(loworder(V),Z),d))
    return parity(a,b) ? Simplex{V}(-1,d) : d
end

function ∧(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    x,y = basis(a), basis(b)
    ba,bb = UInt(x),UInt(y)
    A,B,Q,Z = symmetricmask(V,ba,bb)
    ((count_ones(A&B)>0) || diffcheck(V,ba,bb)) && (return g_zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.∏)
    if istangent(V) && !iszero(Z)
        v = !(typeof(v)<:TensorTerm) ? Simplex{V}(v,getbasis(V,Z)) : Simplex{V}(v,getbasis(loworder(V),Z))
        count_ones(Q)+order(v)>diffmode(V) && (return zero(V))
    end
    return Simplex{V}(parity(x,y) ? -v : v,getbasis(V,(A⊻B)|Q))
end

#∧(a::MultiGrade{V},b::SubManifold{V}) where V = MultiGrade{V}(a.v,basis(a)*b)
#∧(a::SubManifold{V},b::MultiGrade{V}) where V = MultiGrade{V}(b.v,a*basis(b))
#∧(a::MultiGrade{V},b::MultiGrade{V}) where V = MultiGrade{V}(a.v*b.v,basis(a)*basis(b))

export ∧, ∨, ⊗

#⊗(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = a∧b
⊗(a::A,b::B) where {A<:TensorGraded,B<:TensorGraded} = Dyadic(a,b)
⊗(a::A,b::B) where {A<:TensorGraded,B<:TensorGraded{V,0} where V} = a*b
⊗(a::A,b::B) where {A<:TensorGraded{V,0} where V,B<:TensorGraded} = a*b

## regressive product: (L = grade(a) + grade(b); (-1)^(L*(L-mdims(V)))*⋆(⋆(a)∧⋆(b)))

@pure function ∨(a::SubManifold{V},b::SubManifold{V}) where V
    p,C,t,Z = regressive(a,b)
    (!t || iszero(derive_mul(V,UInt(a),UInt(b),1,true))) && (return g_zero(V))
    d = getbasis(V,C)
    istangent(V) && !iszero(Z) && (d = Simplex{V}(getbasis(loworder(V),Z),d))
    return isone(p) ? d : Simplex{V}(p,d)
end

function ∨(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    ba,bb = UInt(basis(a)),UInt(basis(b))
    p,C,t,Z = regressive(V,ba,bb)
    !t  && (return g_zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.∏)
    if istangent(V) && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,ba,bb)
        v = !(typeof(v)<:TensorTerm) ? Simplex{V}(v,getbasis(V,Z)) : Simplex{V}(v,getbasis(loworder(V),Z))
        count_ones(Q)+order(v)>diffmode(V) && (return zero(V))
    end
    return Simplex{V}(isone(p) ? v : p*v,getbasis(V,C))
end

"""
    ∨(ω::TensorAlgebra,η::TensorAlgebra)

Regressive product as defined by the DeMorgan's law: ∨(ω...) = ⋆⁻¹(∧(⋆.(ω)...))
"""
@inline ∨(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∨,a,b)
@inline ∨(a::TensorAlgebra{V},b::UniformScaling{T}) where {V,T<:Field} = a∨V(b)
@inline ∨(a::UniformScaling{T},b::TensorAlgebra{V}) where {V,T<:Field} = V(a)∨b
@generated ∨(t::T) where T<:Values = Expr(:call,:∨,[:(t[$k]) for k ∈ 1:length(t)]...)
@generated ∨(t::T) where T<:FixedVector = Expr(:call,:∨,[:(t[$k]) for k ∈ 1:length(t)]...)
∨(::Values{0,<:Chain{V}}) where V = SubManifold(V) # ∨() = I
∨(::FixedVector{0,<:Chain{V}}) where V = SubManifold(V)
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

import LinearAlgebra: dot, ⋅
export ⋅

"""
    contraction(ω::TensorAlgebra,η::TensorAlgebra)

Interior (right) contraction product: ω⋅η = ω∨⋆η
"""
@pure function contraction(a::SubManifold{V},b::SubManifold{V}) where V
    g,C,t,Z = interior(a,b)
    (!t || iszero(derive_mul(V,UInt(a),UInt(b),1,true))) && (return g_zero(V))
    d = getbasis(V,C)
    istangent(V) && !iszero(Z) && (d = Simplex{V}(getbasis(loworder(V),Z),d))
    return typeof(V) <: Signature ? (g ? Simplex{V}(-1,d) : d) : Simplex{V}(g,d)
end

function contraction(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    ba,bb = UInt(basis(a)),UInt(basis(b))
    g,C,t,Z = interior(V,ba,bb)
    !t && (return g_zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.dot)
    if istangent(V) && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,ba,bb)
        v = !(typeof(v)<:TensorTerm) ? Simplex{V}(v,getbasis(V,Z)) : Simplex{V}(v,getbasis(loworder(V),Z))
        count_ones(Q)+order(v)>diffmode(V) && (return zero(V))
    end
    return Simplex{V}(typeof(V) <: Signature ? (g ? -v : v) : g*v,getbasis(V,C))
end

export ⨼, ⨽

@doc """
    dot(ω::TensorAlgebra,η::TensorAlgebra)

Interior (right) contraction product: ω⋅η = ω∨⋆η
""" dot

# dyadic products

export outer

outer(a::Leibniz.Derivation,b::Chain{V,1}) where V= outer(V(a),b)
outer(a::Chain{W},b::Leibniz.Derivation{T,1}) where {W,T} = outer(a,W(b))
outer(a::Chain{W},b::Chain{V,1}) where {W,V} = Chain{V,1}(a.*value(b))

contraction(a::Proj,b::TensorGraded) = a.v⊗(a.λ*(a.v⋅b))
contraction(a::Dyadic,b::TensorGraded) = a.x⊗(a.y⋅b)
contraction(a::TensorGraded,b::Dyadic) = (a⋅b.x)⊗b.y
contraction(a::TensorGraded,b::Proj) = ((a⋅b.v)*b.λ)⊗b.v
contraction(a::Dyadic,b::Dyadic) = (a.x*(a.y⋅b.x))⊗b.y
contraction(a::Dyadic,b::Proj) = (a.x*((a.y⋅b.v)*b.λ))⊗b.v
contraction(a::Proj,b::Dyadic) = (a.v*(a.λ*(a.v⋅b.x)))⊗b.y
contraction(a::Proj,b::Proj) = (a.v*((a.λ*b.λ)*(a.v⋅b.v)))⊗b.v
contraction(a::Dyadic{V},b::TensorGraded{V,0}) where V = Dyadic{V}(a.x*b,a.y)
contraction(a::Proj{V},b::TensorTerm{V,0}) where V = Proj{V}(a.v,a.λ*value(b))
contraction(a::Proj{V},b::Chain{V,0}) where V = Proj{V}(a.v,a.λ*(@inbounds b[1]))
contraction(a::Proj{V,<:Chain{V,1,<:TensorNested}},b::TensorGraded{V,0}) where V = Proj(Chain{V,1}(contraction.(value(a.v),b)))
#contraction(a::Chain{W,1,<:Proj{V}},b::Chain{V,1}) where {W,V} = Chain{W,1}(value(a).⋅b)
contraction(a::Chain{W,1,<:Dyadic{V}},b::Chain{V,1}) where {W,V} = Chain{W,1}(value(a).⋅Ref(b))
contraction(a::Proj{W,<:Chain{W,1,<:TensorNested{V}}},b::Chain{V,1}) where {W,V} = a.v:b
contraction(a::Chain{W},b::Chain{V,G,<:Chain}) where {W,G,V} = Chain{V,G}(column(Ref(a).⋅value(b)))
contraction(a::Chain{W,L,<:Chain},b::Chain{V,G,<:Chain{W,L}}) where {W,L,G,V} = Chain{V,G}(column(Ref(a).⋅value(b)))
contraction(a::MultiVector{W,<:MultiVector},b::MultiVector{V,<:MultiVector{W}}) where {W,V} = MultiVector{V}(column(Ref(a).⋅value(b)))
Base.:(:)(a::Chain{V,1,<:Chain},b::Chain{V,1,<:Chain}) where V = sum(value(a).⋅value(b))
Base.:(:)(a::Chain{W,1,<:Dyadic{V}},b::Chain{V,1}) where {W,V} = sum(value(a).⋅Ref(b))
#Base.:(:)(a::Chain{W,1,<:Proj{V}},b::Chain{V,1}) where {W,V} = sum(broadcast(⋅,value(a),Ref(b)))

contraction(a::SubManifold{W},b::Chain{V,G,<:Chain}) where {W,G,V} = Chain{V,G}(column(Ref(a).⋅value(b)))
contraction(a::Simplex{W},b::Chain{V,G,<:Chain}) where {W,G,V} = Chain{V,G}(column(Ref(a).⋅value(b)))
contraction(x::Chain{V,G,<:Chain},y::Simplex{V,G}) where {V,G} = value(y)*x[bladeindex(mdims(V),UInt(basis(y)))]
contraction(x::Chain{V,G,<:Chain},y::SubManifold{V,G}) where {V,G} = x[bladeindex(mdims(V),UInt(y))]
contraction(a::Chain{V,L,<:Chain{V,G}},b::Chain{V,G,<:Chain{V}}) where {V,G,L} = Chain{V,G}(matmul(value(a),value(b)))
contraction(x::Chain{W,L,<:Chain{V,G},N},y::Chain{V,G,T,N}) where {W,L,N,V,G,T} = Chain{V,G}(matmul(value(x),value(y)))
contraction(x::Chain{W,L,<:MultiVector{V},N},y::Chain{V,G,T,N}) where {W,L,N,V,G,T} = MultiVector{V}(matmul(value(x),value(y)))
contraction(x::MultiVector{W,<:Chain{V,G},N},y::MultiVector{V,T,N}) where {W,N,V,G,T} = Chain{V,G}(matmul(value(x),value(y)))
contraction(x::MultiVector{W,<:MultiVector{V},N},y::MultiVector{V,T,N}) where {W,N,V,T} = MultiVector{V}(matmul(value(x),value(y)))
@inline @generated function matmul(x::Values{N,<:Simplex{V,G}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,:(@inbounds y[$i]*value(x[$i]))) for i ∈ 1:N]...)
end
@inline @generated function matmul(x::Values{N,<:Chain{V,G}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds y[$i]*x[$i][$j]) for i ∈ 1:N]...) for j ∈ 1:binomial(mdims(V),G)]...)
end
@inline @generated function matmul(x::Values{N,<:MultiVector{V}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds y[$i]*value(x[$i])[$j]) for i ∈ 1:N]...) for j ∈ 1:1<<mdims(V)]...)
end

contraction(a::Dyadic{V,<:Chain{V,1,<:Chain},<:Chain{V,1,<:Chain}} where V,b::TensorGraded) = sum(value(a.x).⊗(value(a.y).⋅b))
contraction(a::Dyadic{V,<:Chain{V,1,<:Chain}} where V,b::TensorGraded) = sum(value(a.x).⊗(a.y.⋅b))
contraction(a::Dyadic{V,T,<:Chain{V,1,<:Chain}} where {V,T},b::TensorGraded) = sum(a.x.⊗(value(a.y).⋅b))
contraction(a::Proj{V,<:Chain{W,1,<:Chain} where W} where V,b::TensorGraded) = sum(value(a.v).⊗(value(a.λ).*value(a.v).⋅b))
contraction(a::Proj{V,<:Chain{W,1,<:Chain{V,1}} where W},b::TensorGraded{V,1}) where V = sum(value(a.v).⊗(value(a.λ).*column(value(a.v).⋅b)))

+(a::Proj{V}...) where V = Proj{V}(Chain(Values(eigvec.(a)...)),Chain(Values(eigval.(a)...)))
+(a::Dyadic{V}...) where V = Proj(Chain(a...))
+(a::TensorNested{V}...) where V = Proj(Chain(Dyadic.(a)...))
+(a::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W,b::TensorNested{V}) where V = +(value(a.v)...,b)
+(a::TensorNested{V},b::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W) where V = +(a,value(b.v)...)
+(a::Proj{M,<:Chain{M,1,<:TensorNested{V}}} where M,b::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W) where V = +(value(a.v)...,value(b.v)...)
+(a::Proj{M,<:Chain{M,1,<:Chain{V}}} where M,b::Proj{W,<:Chain{W,1,<:Chain{V}}} where W) where V = Chain(Values(value(a.v)...,value(b.v)...))
#+(a::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W,b::TensorNested{V}) where V = +(b,Proj.(value(a.v),value(a.λ))...)
#+(a::TensorNested{V},b::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W) where V = +(a,value(b.v)...)

-(a::TensorNested) where V = -1a
-(a::TensorNested,b::TensorNested) where V = a+(-b)
*(a::Number,b::TensorNested{V}) where V = (a*one(V))*b
*(a::TensorNested{V},b::Number) where V = a*(b*one(V))
@inline *(a::TensorGraded{V,0},b::TensorNested{V}) where V = b⋅a
@inline *(a::TensorNested{V},b::TensorGraded{V,0}) where V = a⋅b
@inline *(a::TensorGraded{V,0},b::Proj{V,<:Chain{V,1,<:TensorNested}}) where V = Proj{V}(a*b.v)
@inline *(a::Proj{V,<:Chain{V,1,<:TensorNested}},b::TensorGraded{V,0}) where V = Proj{V}(a.v*b)

@inline *(a::DyadicChain,b::DyadicChain) where V = a⋅b
@inline *(a::DyadicChain,b::Chain) where V = a⋅b
@inline *(a::DyadicChain,b::TensorTerm) where V = a⋅b
@inline *(a::TensorGraded,b::DyadicChain) where V = a⋅b
@inline *(a::DyadicChain,b::TensorNested) where V = a⋅b
@inline *(a::TensorNested,b::DyadicChain) where V = a⋅b

# dyadic identity element

Base.:+(t::LinearAlgebra.UniformScaling,g::TensorNested) = t+DyadicChain(g)
Base.:+(g::TensorNested,t::LinearAlgebra.UniformScaling) = DyadicChain(g)+t
Base.:+(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling) where V = t+g
Base.:-(t::LinearAlgebra.UniformScaling,g::TensorNested) = t-DyadicChain(g)
Base.:-(g::TensorNested,t::LinearAlgebra.UniformScaling) = DyadicChain(g)-t
@generated Base.:+(t::LinearAlgebra.UniformScaling{Bool},g::Chain{V,1,<:Chain{V,1}}) where V = :(Chain{V,1}($(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)]).+value(g)))
@generated Base.:+(t::LinearAlgebra.UniformScaling,g::Chain{V,1,<:Chain{V,1}}) where V = :(Chain{V,1}(t.λ*$(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)]).+value(g)))
@generated Base.:-(t::LinearAlgebra.UniformScaling{Bool},g::Chain{V,1,<:Chain{V,1}}) where V = :(Chain{V,1}($(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)]).-value(g)))
@generated Base.:-(t::LinearAlgebra.UniformScaling,g::Chain{V,1,<:Chain{V,1}}) where V = :(Chain{V,1}(t.λ*$(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)]).-value(g)))
@generated Base.:-(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling{Bool}) where V = :(Chain{V,1}(value(g).-$(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)])))
@generated Base.:-(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling) where V = :(Chain{V,1}(value(g).-t.λ*$(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)])))

## cross product

import LinearAlgebra: cross, ×
export ×

@doc """
    cross(ω::TensorAlgebra,η::TensorAlgebra)

Cross product: ω×η = ⋆(ω∧η)
""" LinearAlgebra.cross

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
    K,V,out = length(x),∪(Manifold.(x)...),prod(x)
    P,F = collect(permutations(1:K)),factorial(K)
    for n ∈ 2:F
        p = prod(x[P[n]])
        DirectSum.indexparity!(P[n],V)[1] ? (out-=p) : (out+=p)
    end
    return out/F
end

for X ∈ TAG, Y ∈ TAG
    @eval <<(a::$X{V},b::$Y{V}) where V = a⊙b
end

@doc """
    ⊙(ω::TensorAlgebra,η::TensorAlgebra)

Symmetrization projection: ⊙(ω...) = ∑(∏(σ.(ω)...))/factorial(length(ω))
""" Grassmann.:<<

for X ∈ TAG, Y ∈ TAG
    @eval >>(a::$X{V},b::$Y{V}) where V = a⊠b
end

@doc """
    ⊠(ω::TensorAlgebra,η::TensorAlgebra)

Anti-symmetrization projection: ⊠(ω...) = ∑(∏(πσ.(ω)...))/factorial(length(ω))
""" Grassmann.:>>

## sandwich product

export ⊘

#=for X ∈ TAG, Y ∈ TAG
    @eval ⊘(x::X,y::Y) where {X<:$X{V},Y<:$Y{V}} where V = diffvars(V)≠0 ? conj(y)*x*y : y\x*involute(y)
end=#
for Y ∈ TAG
    @eval ⊘(x::TensorGraded{V,1},y::Y) where Y<:$Y{V} where V = diffvars(V)≠0 ? conj(y)*x*involute(y) : y\x*involute(y)
end
#=for Z ∈ TAG
    @eval ⊘(x::Chain{V,G},y::T) where {V,G,T<:$Z} = diffvars(V)≠0 ? conj(y)*x*y : ((~y)*x*involute(y))(Val(G))/abs2(y)
end=#

@doc """
    ⊘(ω::TensorAlgebra,η::TensorAlgebra)

General sandwich product: ω⊘η = involute(η)\\ω⊖η

For normalized even grade η it is ω⊘η = (~η)⊖ω⊖η
""" Grassmann.:⊘

for X ∈ TAG, Y ∈ TAG
    @eval >>>(x::$X{V},y::$Y{V}) where V = x * y * ~x
end

@doc """
    >>>(ω::TensorAlgebra,η::TensorAlgebra)

Sandwich product: ω>>>η = ω⊖η⊖(~ω)
""" Grassmann.:>>>

## linear algebra

export ⟂, ∥

∥(a,b) = iszero(a∧b)

## exponentiation

function Base.:^(v::T,i::S) where {T<:TensorTerm,S<:Integer}
    i == 0 && (return getbasis(Manifold(v),0))
    i == 1 && (return v)
    j,bas = (i-1)%4,basis(v)
    out = if j == 0
        bas
    elseif j == 1
        bas*bas
    elseif j == 2
        bas*bas*bas
    elseif j == 3
        bas*bas*bas*bas
    end
    return typeof(v)<:SubManifold ? out : out*AbstractTensors.:^(value(v),i)
end

function Base.:^(v::T,i::S) where {T<:TensorAlgebra,S<:Integer}
    V = Manifold(v)
    isone(i) && (return v)
    if T<:Chain && mdims(V)≤3 && diffvars(v)==0
        sq,d = contraction2(~v,v),i÷2
        val = isone(d) ? sq : sq^d
        return iszero(i%2) ? val : val*v
    end
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

@pure abs2_inv(::SubManifold{V,G,B} where G) where {V,B} = abs2(getbasis(V,grade_basis(V,B)))

for (nv,d) ∈ ((:inv,:/),(:inv_rat,://))
    @eval begin
        @pure $d(a,b::T) where T<:TensorAlgebra = a*$nv(b)
        @pure $d(a::N,b::T) where {N<:Number,T<:TensorAlgebra} = a*$nv(b)
        @pure $d(a::S,b::UniformScaling) where S<:TensorGraded = a*$nv(Manifold(a)(b))
        @pure $d(a::S,b::UniformScaling) where S<:TensorMixed = a*$nv(Manifold(a)(b))
        function $nv(a::Chain)
            r,v,q = ~a,abs2(a),diffvars(Manifold(a))≠0
            q&&!(typeof(v)<:TensorGraded && grade(v)==0) ? $d(r,v) : $d(r,value(scalar(v)))
        end
        function $nv(m::MultiVector{V,T}) where {V,T}
            rm = ~m
            d = rm*m
            fd = norm(d)
            sd = scalar(d)
            norm(sd) ≈ fd && (return $d(rm,sd))
            for k ∈ 1:mdims(V)
                @inbounds AbstractTensors.norm(d[k]) ≈ fd && (return $d(rm,d(k)))
            end
            throw(error("inv($m) is undefined"))
        end
        function $nv(m::MultiVector{V,Any}) where V
            rm = ~m
            d = rm*m
            fd = $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d)]...)
            sd = scalar(d)
            $Sym.:∏(value(sd),value(sd)) == fd && (return $d(rm,sd))
            for k ∈ 1:mdims(V)
                @inbounds $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d[k])]...) == fd && (return $d(rm,d(k)))
            end
            throw(error("inv($m) is undefined"))
        end
        @pure $nv(b::SubManifold{V,0} where V) = b
        @pure function $nv(b::SubManifold{V,G,B}) where {V,G,B}
            $d(parityreverse(grade(V,B)) ? -1 : 1,value(abs2_inv(b)))*b
        end
        $nv(b::Simplex{V,0,B}) where {V,B} = Simplex{V,0,B}(AbstractTensors.inv(value(b)))
        function $nv(b::Simplex{V,G,B,T}) where {V,G,B,T}
            Simplex{V,G,B}($d(parityreverse(grade(V,B)) ? -one(T) : one(T),value(abs2_inv(B)*value(b))))
        end
        function $nv(b::Simplex{V,G,B,Any}) where {V,G,B}
            Simplex{V,G,B}($Sym.$d(parityreverse(grade(V,B)) ? -1 : 1,value($Sym.:∏(abs2_inv(B),value(b)))))
        end
    end
end

function generate_inverses(Mod,T)
    for (nv,d,ds) ∈ ((:inv,:/,:($Sym.:/)),(:inv_rat,://,:($Sym.://)))
        for Term ∈ (:TensorGraded,:TensorMixed)
            @eval $d(a::S,b::T) where {S<:$Term,T<:$Mod.$T} = a*$ds(1,b)
        end
        @eval function $nv(b::Simplex{V,G,B,$Mod.$T}) where {V,G,B}
            Simplex{V,G,B}($Mod.$d(parityreverse(grade(V,B)) ? -1 : 1,value($Sym.:∏(abs2_inv(B),value(b)))))
        end
    end
end

for T ∈ (:Real,:Complex)
    generate_inverses(Base,T)
end

### Algebra Constructors

insert_t(x) = Expr(:block,:(t=promote_type(valuetype(a),valuetype(b))),x)

addvec(a,b,s,o) = o ≠ :+ ? subvec(a,b,s) : addvec(a,b,s)
addvec(a,b,s) = isfixed(a,b) ? (:($Sym.:∑),:($Sym.:∑),:svec) : (:+,:+,:mvec)
subvec(a,b,s) = isfixed(a,b) ? (s ? (:($Sym.:-),:($Sym.:∑),:svec) : (:($Sym.:∑),:($Sym.:-),:svec)) : (s ? (:-,:+,:mvec) : (:+,:-,:mvec))

subvec(b) = isfixed(valuetype(b)) ? (:($Sym.:-),:svec,:($Sym.:∏)) : (:-,:mvec,:*)
conjvec(b) = isfixed(valuetype(b)) ? (:($Sym.conj),:svec) : (:conj,:mvec)

mulvec(a,b,c) = c≠:contraction ? mulvec(a,b) : isfixed(a,b) ? (:($Sym.dot),:svec) : (:dot,:mvec)
mulvec(a,b) = isfixed(a,b) ? (:($Sym.:∏),:svec) : (:*,:mvec)
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
            :(Simplex{V,L}($bop(value(a),value(b)),basis(a)))
        elseif L == G
            if binomial(mdims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:ib),:svec)...)
                out = zeros(svec(N,G,Any))
                X,Y = UInt(basis(a)),UInt(basis(b))
                for k ∈ 1:binomial(N,G)
                    C = ib[k]
                    if C ∈ (X,Y)
                        val = C≠X ? :($bop(value(b,t))) : :(value(a,t))
                        @inbounds setblade!_pre(out,val,C,Val{N}())
                    end
                end
                return Expr(:block,insert_expr((:t,),VEC)...,
                    :(Chain{V,L}($(Expr(:call,tvec(N,G,:t),out...)))))
            else return quote
                $(insert_expr((:N,:t))...)
                out = zeros($VEC(N,L,t))
                setblade!(out,value(a,t),UInt(basis(a)),Val{N}())
                setblade!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
                return Chain{V,L}(out)
            end end
        else
            if mdims(V)<cache_limit
                $(insert_expr((:N,:ib),:svec)...)
                out,ib = zeros(svec(N,Any)),indexbasis(N)
                X,Y = UInt(basis(a)),UInt(basis(b))
                for k ∈ 1:1<<N
                    C = ib[k]
                    if C ∈ (X,Y)
                        val = C≠X ? :($bop(value(b,t))) : :(value(a,t))
                        @inbounds setmulti!_pre(out,val,C,Val{N}())
                    end
                end
                return Expr(:block,insert_expr((:t,),VEC)...,
                    :(MultiVector{V}($(Expr(:call,tvec(N,:t),out...)))))
            else quote
                #@warn("sparse MultiGrade{V} objects not properly handled yet")
                #return MultiGrade{V}(a,b)
                $(insert_expr((:N,:t,:out),VEC)...)
                setmulti!(out,value(a,t),UInt(basis(a)),Val{N}())
                setmulti!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
                return MultiVector{V}(out)
            end end
        end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:Chain{V,G,T}},op,swap=false) where {V,G,T}
        left,right,VEC = addvec(a,b,swap,op)
        if binomial(mdims(V),G)<(1<<cache_limit)
            $(insert_expr((:N,:ib),:svec)...)
            t = promote_type(valuetype(a),valuetype(b))
            out = zeros(svec(N,G,Any))
            X = UInt(basis(a))
            for k ∈ 1:binomial(N,G)
                B = @inbounds ib[k]
                val = :(@inbounds $right(b.v[$k]))
                val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                setblade!_pre(out,val,B,Val{N}())
            end
            return :(Chain{V,G}($(Expr(:call,tvec(N,G,t),out...))))
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
        if mdims(V)<cache_limit
            $(insert_expr((:N,:ib,:bn),:svec)...)
            t = promote_type(valuetype(a),valuetype(b))
            out = zeros(svec(N,Any))
            X = UInt(basis(a))
            for k ∈ 1:binomial(N,G)
                B = @inbounds ib[k]
                val = :(@inbounds $right(b.v[$k]))
                val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                @inbounds setmulti!_pre(out,val,B,Val(N))
            end
            for g ∈ 1:N+1
                g-1 == G && continue
                ib = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    B = @inbounds ib[i]
                    if B == X
                        val = :(@inbounds $left(value(a,$t)))
                        setmulti!_pre(out,val,B,Val(N))
                    end
                end
            end
            return :(MultiVector{V}($(Expr(:call,tvec(N,t),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
            @inbounds out[r+1:r+bng] = $(bcast(right,:(value(b,$VEC(N,G,t)),)))
            addmulti!(out,value(a,t),UInt(basis(a)),Val(N))
            return MultiVector{V}(out)
        end; else quote
            $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
            @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
            addmulti!(out,$left(value(b,t)),UInt(basis(b)),Val(N))
            return MultiVector{V}(out)
        end end end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:MultiVector{V,T}},op,swap=false) where {V,G,T}
        left,right,VEC = addvec(a,b,swap,op)
        if mdims(V)<cache_limit
            $(insert_expr((:N,:bs,:bn),:svec)...)
            t = promote_type(valuetype(a),valuetype(b))
            out = zeros(svec(N,Any))
            X = UInt(basis(a))
            for g ∈ 1:N+1
                ib = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    B = @inbounds ib[i]
                    val = :(@inbounds $right(b.v[$(bs[g]+i)]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    setmulti!_pre(out,val,B,Val(N))
                end
            end
            return :(MultiVector{V}($(Expr(:call,tvec(N,t),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t),VEC)...)
            out = convert($VEC(N,t),$(bcast(right,:(value(b,$VEC(N,t)),))))
            addmulti!(out,value(a,t),UInt(basis(a)),Val(N))
            return MultiVector{V}(out)
        end; else quote
            $(insert_expr((:N,:t),VEC)...)
            out = value(a,$VEC(N,t))
            addmulti!(out,$left(value(b,t)),UInt(basis(b)),Val(N))
            return MultiVector{V}(out)
        end end end
    end
    @noinline function product(a::Type{S},b::Type{<:Chain{V,G,T}},swap=false) where S<:TensorGraded{V,L} where {V,G,L,T}
        MUL,VEC = mulvec(a,b)
        if G == 0
            return S<:Chain ? :(Chain{V,L}(broadcast($MUL,a.v,Ref(@inbounds b[1])))) : swap ? :(@inbounds b[1]*a) : :(@inbounds a*b[1])
        elseif S<:Chain && L == 0
            return :(Chain{V,G}(broadcast($MUL,Ref(@inbounds a[1]),b.v)))
        elseif (swap ? L : G) == mdims(V) && !istangent(V)
            return swap ? (S<:Simplex ? :(⋆(~b)*value(a)) : :(⋆(~b))) : :(@inbounds ⋆(~a)*b[1])
        elseif (swap ? G : L) == mdims(V) && !istangent(V)
            return swap ? :(b[1]*complementlefthodge(~a)) : S<:Simplex ? :(value(a)*complementlefthodge(~b)) : S<:Chain ? :(@inbounds a[1]*complementlefthodge(~b)) : :(complementlefthodge(~b))
        elseif binomial(mdims(V),G)*(S<:Chain ? binomial(mdims(V),L) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:bng,:ib,:μ),:svec)...)
                out = zeros(svec(N,t))
                B = indexbasis(N,L)
                for i ∈ 1:binomial(N,L)
                    @inbounds v,ibi = :(@inbounds a[$i]),B[i]
                    for j ∈ 1:bng
                        @inbounds geomaddmulti!_pre(V,out,ibi,ib[j],derive_pre(V,ibi,ib[j],v,:(@inbounds b[$j]),MUL))
                    end
                end
            else
                $(insert_expr((:N,:t,:out,:ib,:μ),:svec)...)
                U = UInt(basis(a))
                for i ∈ 1:binomial(N,G)
                    A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                    if S<:Simplex
                        @inbounds geomaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL))
                    else
                        @inbounds geomaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false))
                    end
                end
            end
            return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
        elseif S<:Chain; return quote
            $(insert_expr((:N,:t,:bng,:ib,:μ),VEC)...)
            out = zeros($VEC(N,t))
            B = indexbasis(N,L)
            for i ∈ 1:binomial(N,L)
                @inbounds v,ibi = a[i],B[i]
                v≠0 && for j ∈ 1:bng
                    if @inbounds geomaddmulti!(V,out,ibi,ib[j],derive_mul(V,ibi,ib[j],v,b[j],$MUL))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds geomaddmulti!(V,out,ibi,ib[j],derive_mul(V,ibi,ib[j],v,b[j],$MUL))
                    end
                end
            end
            return MultiVector{V}(out)
        end else return quote
            $(insert_expr((:N,:t,:out,:ib,:μ),VEC)...)
            U = UInt(basis(a))
            for i ∈ 1:binomial(N,G)
                A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                $(if S<:Simplex
                    :(if @inbounds geomaddmulti!(V,out,A,B,derive_mul(V,A,B,a.v,b[i],$MUL))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds geomaddmulti!(V,out,A,B,derive_mul(V,A,B,a.v,b[i],$MUL))
                    end)
                else
                    :(if @inbounds geomaddmulti!(V,out,A,B,derive_mul(V,A,B,b[i],false))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds geomaddmulti!(V,out,A,B,derive_mul(V,A,B,b[i],false))
                    end)
                end)
            end
            return MultiVector{V}(out)
        end end
    end
    @noinline function product_contraction(a::Type{S},b::Type{<:Chain{V,G,T}},swap=false,contr=:contraction) where S<:TensorGraded{V,L} where {V,G,T,L}
        MUL,VEC = mulvec(a,b,contr)
        (swap ? G<L : L<G) && (!istangent(V)) && (return g_zero(V))
        GL = swap ? G-L : L-G
        if binomial(mdims(V),G)*(S<:Chain ? binomial(mdims(V),L) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:bng,:bnl),:svec)...)
                μ = istangent(V)|hasconformal(V)
                ia = indexbasis(N,L)
                ib = indexbasis(N,G)
                out = zeros(μ ? svec(N,Any) : svec(N,GL,Any))
                for i ∈ 1:bnl
                    @inbounds v,iai = :(@inbounds a[$i]),ia[i]
                    for j ∈ 1:bng
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(@inbounds b[$j]),MUL))
                        else
                            @inbounds skewaddblade!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(@inbounds b[$j]),MUL))
                        end
                    end
                end
            else
                $(insert_expr((:N,:t,:ib,:bng,:μ),:svec)...)
                out = zeros(μ ? svec(N,Any) : svec(N,GL,Any))
                U = UInt(basis(a))
                for i ∈ 1:bng
                    A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                    if S<:Simplex
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL))
                        else
                            @inbounds skewaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL))
                        end
                    else
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false))
                        else
                            @inbounds skewaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false))
                        end
                    end
                end
            end
            #return :(value_diff(Simplex{V,0,$(getbasis(V,0))}($(value(mv)))))
            return if μ
                insert_t(:(MultiVector{$V}($(Expr(:call,istangent(V) ? tvec(N) : tvec(N,:t),out...)))))
            else
                insert_t(:(value_diff(Chain{$V,$GL}($(Expr(:call,tvec(N,GL,:t),out...))))))
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
            return μ ? MultiVector{V}(out) : value_diff(Chain{V,G-L}(out))
        end else return quote
            $(insert_expr((:N,:t,:ib,:bng,:μ),VEC)...)
            out = zeros(μ ? $VEC(N,t) : $VEC(N,$GL,t))
            U = UInt(basis(a))
            for i ∈ 1:bng
                if μ
                    A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                    $(if S<:Simplex
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
            return μ ? MultiVector{V}(out) : value_diff(Chain{V,$GL}(out))
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
        $po(G+L,mdims(V)) && (!istangent(V)) && (return g_zero(V))
        if binomial(mdims(W),L)*(S<:Chain ? binomial(mdims(w),G) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:μ),:mvec,:T,:S)...)
                ia = indexbasis(mdims(w),G)
                ib = indexbasis(mdims(W),L)
                out = zeros(μ ? svec(N,Any) : svec(N,$GL,Any))
                CA,CB = isdual(w),isdual(W)
                for i ∈ 1:binomial(mdims(w),G)
                    @inbounds v,iai = :(@inbounds a[$i]),ia[i]
                    x = CA ? dual(V,iai) : iai
                    for j ∈ 1:binomial(mdims(W),L)
                        X = @inbounds CB ? dual(V,ib[j]) : ib[j]
                        if μ
                            $grassaddmulti!_pre(V,out,x,X,derive_pre(V,x,X,v,:(@inbounds b[$j]),MUL))
                        else
                            $grassaddblade!_pre(V,out,x,X,derive_pre(V,x,X,v,:(@inbounds b[$j]),MUL))
                        end
                    end
                end
            else
                $(insert_expr((:N,:t,:μ),:mvec,Int,:T)...)
                ib = indexbasis(mdims(R),L)
                out = zeros(μ ? svec(N,Any) : svec(N,$GL,Any))
                C,x = isdual(R),isdual(Q) ? dual(V,UInt(basis(a))) : UInt(basis(a))
                for i ∈ 1:binomial(mdims(W),L)
                    X = @inbounds C ? dual(V,ib[i]) : ib[i]
                    A,B = swap ? (X,x) : (x,X)
                    if S<:Simplex
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
                insert_t(:(MultiVector{$V}($(Expr(:call,istangent(V) ? tvec(N) : tvec(N,:t),out...)))))
            else
                insert_t(:(Chain{$V,$$GL}($(Expr(:call,tvec(N,$GL,:t),out...)))))
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
            return μ ? MultiVector{V}(out) : Chain{V,$$GL}(out)
        end else return quote
            V = $V
            $(insert_expr((:N,:t,:μ),VEC)...)
            ib = indexbasis(mdims(R),L)
            out = zeros(μ ? $VEC(N,t) : $VEC(N,$$GL,t))
            C,x = isdual(R),isdual(Q) ? dual(V,UInt(basis(a))) : UInt(basis(a))
            for i ∈ 1:binomial(mdims(W),L)
                X = @inbounds C ? dual(V,ib[i]) : ib[i]
                A,B = (X,x) : (x,X)
                $(if S<:Simplex
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
            return μ ? MultiVector{V}(out) : Chain{V,$$GL}(out)
        end end
    end
end

for (op,product!) ∈ ((:∧,:exteraddmulti!),(:*,:geomaddmulti!),
                     (:∨,:meetaddmulti!),(:contraction,:skewaddmulti!))
    preproduct! = Symbol(product!,:_pre)
    prop = op≠:* ? Symbol(:product_,op) : :product
    @eval $prop(a,b,swap=false) = $prop(typeof(a),typeof(b),swap)
    @eval @noinline function $prop(a::Type{S},b::Type{<:MultiVector{V,T}},swap=false) where S<:TensorGraded{V,G} where {V,G,T}
        MUL,VEC = mulvec(a,b,$(QuoteNode(op)))
        if mdims(V)<cache_limit
            $(insert_expr((:N,:t,:ib,:bs,:bn,:μ),:svec)...)
            out = zeros(svec(N,Any))
            t = promote_type(valuetype(a),valuetype(b))
            for g ∈ 1:N+1
                ia = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    if S<:Chain
                        @inbounds val = :(@inbounds b.v[$(bs[g]+i)])
                        for j ∈ 1:bn[G+1]
                            A,B = swapper(ib[j],ia[i],swap)
                            X,Y = swapper(:(@inbounds a[$j]),val,swap)
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,X,Y,MUL))
                        end
                    else
                        U = UInt(basis(a))
                        A,B = swapper(U,ia[i],swap)
                        if S<:Simplex
                            X,Y = swapper(:(a.v),:(@inbounds b.v[$(bs[g]+i)]),swap)
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,X,Y,MUL))
                        else
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,:(@inbounds b.v[$(bs[g]+i)]),false))
                        end
                    end
                end
            end
            return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
        else return quote
            $(insert_expr((:N,:t,:out,:ib,:bs,:bn,:μ),VEC)...)
            for g ∈ 1:N+1
                ia = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    $(if S<:Chain; quote
                        @inbounds val = b.v[bs[g]+i]
                        val≠0 && for j ∈ 1:bn[G+1]
                            A,B = $(swap ? :(@inbounds (ia[i],ib[j])) : :(@inbounds (ib[j],ia[i])))
                            X,Y = $(swap ? :((val,@inbounds a[j])) : :((@inbounds a[j],val)))
                            dm = derive_mul(V,A,B,X,Y,$MUL)
                            if @inbounds $$product!(V,out,A,B,dm)&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,dm)
                            end
                        end end
                    else quote
                        A,B = $(swap ? :((@inbounds ia[i],$(UInt(basis(a))))) : :(($(UInt(basis(a))),@inbounds ia[i])))
                        $(if S<:Simplex; quote
                            X,Y=$(swap ? :((b.v[bs[g]+1],a.v)) : :((a.v,@inbounds b.v[bs[g]+1])))
                            dm = derive_mul(V,A,B,X,Y,$MUL)
                            if @inbounds $$product!(V,out,A,B,dm)&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,dm)
                            end end
                        else
                            :(if @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,b.v[bs[g]+i],false))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,b.v[bs[g]+i],false))
                            end)
                        end) end
                    end)
                end
            end
            return MultiVector{V}(out)
        end end
    end
end

@eval @noinline function generate_loop_multivector(V,a,b,MUL,product!,preproduct!,d=nothing)
    if mdims(V)<cache_limit/2
        $(insert_expr((:N,:t,:out,:bs,:bn),:svec)...)
        for g ∈ 1:N+1
            X = indexbasis(N,g-1)
            @inbounds for i ∈ 1:bn[g]
                @inbounds val = nothing≠d ? :(@inbounds $a[$(bs[g]+i)]/$d) : :(@inbounds $a[$(bs[g]+i)])
                for G ∈ 1:N+1
                    @inbounds R = bs[G]
                    Y = indexbasis(N,G-1)
                    @inbounds for j ∈ 1:bn[G]
                        @inbounds preproduct!(V,out,X[i],Y[j],derive_pre(V,X[i],Y[j],val,:(@inbounds $b[$(R+j)]),MUL))
                    end
                end
            end
        end
        (:N,:t,:out), :(out .= $(Expr(:call,tvec(N,:t),out...)))
    else
        (:N,:t,:out,:bs,:bn,:μ), quote
            for g ∈ 1:N+1
                X = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    @inbounds val = $(nothing≠d ? :(@inbounds $a[bs[g]+i]/$d) : :(@inbounds $a[bs[g]+i]))
                    val≠0 && for G ∈ 1:N+1
                        @inbounds R = bs[G]
                        Y = indexbasis(N,G-1)
                        @inbounds for j ∈ 1:bn[G]
                            dm = derive_mul(V,X[i],Y[j],val,$b[R+j],$MUL)
                            if @inbounds $product!(V,out,X[i],Y[j],dm)&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $product!(V,out,X[i],Y[j],dm)
                            end
                        end
                    end
                end
            end
        end
    end
end
