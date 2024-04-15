
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
import AbstractTensors: ∧, ∨, ⟑, ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, ∗, rem, div, TAG, SUB
import AbstractTensors: plus, minus, times, contraction, equal, wedgedot, veedot, antidot
import Leibniz: diffcheck, diffmode, hasinforigin, hasorigininf, symmetricsplit
import Leibniz: loworder, isnull, Field, ExprField
const Sym,SymField = :AbstractTensors,Any

if VERSION >= v"1.10.0"; @eval begin
    import AbstractTensors.$(Symbol("⟇"))
    export $(Symbol("⟇"))
end end

## geometric product

"""
    *(ω::TensorAlgebra,η::TensorAlgebra)

Geometric algebraic product: ω⊖η = (-1)ᵖdet(ω∩η)⊗(Λ(ω⊖η)∪L(ω⊕η))
"""
@pure times(a::Submanifold{V},b::Submanifold{V}) where V = mul(a,b)
*(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = *(a*b,c...)

@pure function mul(a::Submanifold{V},b::Submanifold{V},der=derive_mul(V,UInt(a),UInt(b),1,true)) where V
    ba,bb = UInt(a),UInt(b)
    (diffcheck(V,ba,bb) || iszero(der)) && (return Zero(V))
    A,B,Q,Z = symmetricmask(V,ba,bb)
    pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(V,A,B) : (false,A⊻B,false)
    d = getbasis(V,bas|Q)
    out = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(a,b)⊻pcc ? Single{V}(-1,d) : d) : Single{V}((pcc ? -1 : 1)*parityinner(V,A,B),d)
    diffvars(V)≠0 && !iszero(Z) && (out = Single{V}(getbasis(loworder(V),Z),out))
    return cc ? (v=value(out);out+Single{V}(hasinforigin(V,A,B) ? -(v) : v,getbasis(V,conformalmask(V)⊻UInt(d)))) : out
end

function times(a::Single{V},b::Submanifold{V}) where V
    v = derive_mul(V,UInt(basis(a)),UInt(b),a.v,true)
    bas = mul(basis(a),b,v)
    order(a.v)+order(bas)>diffmode(V) ? Zero(V) : Single{V}(v,bas)
end
function times(a::Submanifold{V},b::Single{V}) where V
    v = derive_mul(V,UInt(a),UInt(basis(b)),b.v,false)
    bas = mul(a,basis(b),v)
    order(b.v)+order(bas)>diffmode(V) ? Zero(V) : Single{V}(v,bas)
end

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
@generated ∧(t::T) where T<:Values{N} where N = wedges([:(t[$i]) for i ∈ list(1,N)])
@generated ∧(t::T) where T<:FixedVector{N} where N = wedges([:(@inbounds t[$i]) for i ∈ list(1,N)])
∧(::Values{0,<:Chain{V}}) where V = One(V) # ∧() = 1
∧(::FixedVector{0,<:Chain{V}}) where V = One(V)
∧(t::Chain{V,1,<:Chain} where V) = ∧(value(t))
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

export ∧, ∨, ⟑, wedgedot, veedot, ⊗

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

import LinearAlgebra: dot, ⋅
export ⋅

"""
    contraction(ω::TensorAlgebra,η::TensorAlgebra)

Interior (right) contraction product: ω⋅η = ω∨⋆η
"""
@pure function contraction(a::Submanifold{V},b::Submanifold{V}) where V
    g,C,t,Z = interior(a,b)
    (!t || iszero(derive_mul(V,UInt(a),UInt(b),1,true))) && (return Zero(V))
    d = getbasis(V,C)
    istangent(V) && !iszero(Z) && (d = Single{V}(getbasis(loworder(V),Z),d))
    return isone(g) ? d : Single{V}(g,d)
end

function contraction(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    ba,bb = UInt(basis(a)),UInt(basis(b))
    g,C,t,Z = interior(V,ba,bb)
    !t && (return Zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.dot)
    if istangent(V) && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,ba,bb)
        v = !(typeof(v)<:TensorTerm) ? Single{V}(v,getbasis(V,Z)) : Single{V}(v,getbasis(loworder(V),Z))
        count_ones(Q)+order(v)>diffmode(V) && (return Zero(V))
    end
    return Single{V}(g*v,getbasis(V,C))
end

export ⨼, ⨽

@doc """
    dot(ω::TensorAlgebra,η::TensorAlgebra)

Interior (right) contraction product: ω⋅η = ω∨⋆η
""" dot

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

## veedot

veedot(a,b) = complementleft(complementright(a)*complementright(b))

## antidot

antidot(a,b) = complementleft(contraction(complementright(a),complementright(b)))

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
    return typeof(v)<:Submanifold ? out : out*AbstractTensors.:^(value(v),i)
end

function Base.:^(v::T,i::S) where {T<:TensorAlgebra,S<:Integer}
    V = Manifold(v)
    isone(i) && (return v)
    if T<:Chain && mdims(V)≤3 && diffvars(v)==0
        sq,d = contraction2(~v,v),i÷2
        val = isone(d) ? sq : sq^d
        return iszero(i%2) ? val : val*v
    elseif T<:Couple && value(basis(v)*basis(v))==-1
        return Couple{V,basis(v)}(v.v^i)
    end
    out = One(V)
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

@pure abs2_inv(::Submanifold{V,G,B} where G) where {V,B} = abs2(getbasis(V,grade_basis(V,B)))

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
        function $nv(m::Multivector{V,T}) where {V,T}
            rm = ~m
            d = rm*m
            fd = norm(d)
            sd = scalar(d)
            norm(sd) ≈ fd && (return $d(rm,sd))
            for k ∈ list(1,mdims(V))
                @inbounds AbstractTensors.norm(d[k]) ≈ fd && (return $d(rm,d(k)))
            end
            throw(error("inv($m) is undefined"))
        end
        function $nv(m::Multivector{V,Any}) where V
            rm = ~m
            d = rm*m
            fd = $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d)]...)
            sd = scalar(d)
            $Sym.:∏(value(sd),value(sd)) == fd && (return $d(rm,sd))
            for k ∈ list(1,mdims(V))
                @inbounds $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d[k])]...) == fd && (return $d(rm,d(k)))
            end
            throw(error("inv($m) is undefined"))
        end
        function $nv(m::Spinor{V,T}) where {V,T}
            rm = ~m
            d = rm*m
            fd = norm(d)
            sd = scalar(d)
            norm(sd) ≈ fd && (return $d(rm,sd))
            for k ∈ evens(2,mdims(V))
                @inbounds AbstractTensors.norm(d[k]) ≈ fd && (return $d(rm,d(k)))
            end
            throw(error("inv($m) is undefined"))
        end
        function $nv(m::Spinor{V,Any}) where V
            rm = ~m
            d = rm*m
            fd = $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d)]...)
            sd = scalar(d)
            $Sym.:∏(value(sd),value(sd)) == fd && (return $d(rm,sd))
            for k ∈ evens(2,mdims(V))
                @inbounds $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d[k])]...) == fd && (return $d(rm,d(k)))
            end
            throw(error("inv($m) is undefined"))
        end
        @pure $nv(b::Submanifold{V,0} where V) = b
        @pure function $nv(b::Submanifold{V,G,B}) where {V,G,B}
            $d(parityreverse(grade(V,B)) ? -1 : 1,value(abs2_inv(b)))*b
        end
        $nv(b::Single{V,0,B}) where {V,B} = Single{V,0,B}(AbstractTensors.inv(value(b)))
        function $nv(b::Single{V,G,B,T}) where {V,G,B,T}
            Single{V,G,B}($d(parityreverse(grade(V,B)) ? -one(T) : one(T),value(abs2_inv(B)*value(b))))
        end
        function $nv(b::Single{V,G,B,Any}) where {V,G,B}
            Single{V,G,B}($Sym.$d(parityreverse(grade(V,B)) ? -1 : 1,value($Sym.:∏(abs2_inv(B),value(b)))))
        end
    end
end

/(a::TensorTerm{V,0},b::Couple{V,B,S}) where {V,B,S} = (T = promote_type(valuetype(a),S); Couple{V,B}(value(a)*inv(T(b.v))))
/(a::Couple{V,B},b::TensorTerm{V,0}) where {V,B} = Couple{V,B}(Complex(a.v.re/value(b),a.v.im/value(b)))

function /(a::Couple{V,B,T}, b::Couple{V,B,T}) where {V,B,T<:Real}
    are,aim = reim(a); bre,bim = reim(b)
    B2 = value(abs2_inv(B))
    Couple{V,B}(if abs(bre) <= abs(bim)
        if isinf(bre) && isinf(bim)
            r = sign(bre)/sign(bim)
        else
            r = bre / bim
        end
        den = bim*B2 + r*bre
        Complex((are*r + aim*B2)/den, (aim*r - are)/den)
    else
        if isinf(bre) && isinf(bim)
            r = sign(bim)/sign(bre)
        else
            r = bim / bre
        end
        den = bre + (r*bim)*B2
        Complex((are + (aim*r)*B2)/den, (aim - are*r)/den)
    end)
end

inv(z::Couple{V,B,<:Union{Float16,Float32}}) where {V,B} =
    (w = inv(widen(z)); Couple{V,B}(oftype(z.v,w.v)))

/(z::Couple{V,B,T}, w::Couple{V,B,T}) where {V,B,T<:Union{Float16,Float32}} =
    (w = widen(z)*inv(widen(w)); Couple{V,B}(oftype(z.v, w.v)))

# robust complex division for double precision
# variables are scaled & unscaled to avoid over/underflow, if necessary
# based on arxiv.1210.4539
#             a + i*b
#  p + i*q = ---------
#             c + i*d
function /(z::Couple{V,B,Float64}, w::Couple{V,B,Float64}) where {V,B}
    a, b = reim(z); c, d = reim(w)
    absa = abs(a); absb = abs(b);  ab = absa >= absb ? absa : absb # equiv. to max(abs(a),abs(b)) but without NaN-handling (faster)
    absc = abs(c); absd = abs(d);  cd = absc >= absd ? absc : absd
    halfov = 0.5*floatmax(Float64)              # overflow threshold
    twounϵ = floatmin(Float64)*2.0/eps(Float64) # underflow threshold
    # actual division operations
    e = Val(float(value(abs2_inv(B))))
    if  ab>=halfov || ab<=twounϵ || cd>=halfov || cd<=twounϵ # over/underflow case
        p,q = scaling_cdiv(a,b,c,d,ab,cd,e) # scales a,b,c,d before division (unscales after)
    else
        p,q = cdiv(a,b,c,d,e)
    end
    return Couple{V,B}(ComplexF64(p,q))
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

function inv(z::Couple{V,B}) where {V,B}
    c, d = reim(z)
    (isinf(c) | isinf(d)) && (return Couple{V,B}(complex(copysign(zero(c), c), flipsign(-zero(d), d))))
    e = c*c + d*d*value(abs2_inv(B))
    Couple{V,B}(complex(c/e, parityreverse(grade(B)) ? -d/e : d/e))
end
inv(z::Couple{V,B,<:Integer}) where {V,B} = inv(Couple{V,B}(float(z.v)))

function inv(w::Couple{V,B,Float64}) where {V,B}
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
    a = Val(float(value(abs2_inv(B))))
    if absd <= absc
        p, q = robust_cinv(c, d, a)
    else
        q, p = robust_cinv_rev(-d, -c, a)
    end
    return Couple{V,B}(ComplexF64(p*s, q*s)) # undo scaling
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
        elseif !istangent(V) && !hasconformal(V) && L == 0 &&
                valuetype(a)<:Real && valuetype(b)<:Real
            :(Couple{V,basis(b)}(Complex(value(a),$bop(value(b)))))
        elseif !istangent(V) && !hasconformal(V) && G == 0 &&
                valuetype(a)<:Real && valuetype(b)<:Real
            :(Couple{V,basis(a)}(Complex($bop(value(b)),value(a))))
        elseif !istangent(V) && !hasconformal(V) && L == grade(V) &&
                valuetype(a)<:Real && valuetype(b)<:Real
            :(PseudoCouple{V,basis(b)}(Complex($bop(value(b)),value(a))))
        elseif !istangent(V) && !hasconformal(V) && G == grade(V) &&
                valuetype(a)<:Real && valuetype(b)<:Real
            :(PseudoCouple{V,basis(a)}(Complex(value(a),$bop(value(b)))))
        elseif L == G
            if binomial(mdims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:ib,:t),:mvec)...)
                out = svec(N,G,Any)(zeros(svec(N,G,t)))
                setblade!_pre(out,:(value(a,$t)),UInt(basis(a)),Val{N}())
                setblade!_pre(out,:($bop(value(b,$t))),UInt(basis(b)),Val{N}())
                return :(Chain{V,L}($(Expr(:call,tvec(N,G,t),out...))))
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
            out,ib = svecs(N,Any)(zeros(svecs(N,t))),indexbasis(N)
            setspin!_pre(out,:(value(a,$t)),UInt(basis(a)),Val{N}())
            setspin!_pre(out,:($bop(value(b,$t))),UInt(basis(b)),Val{N}())
            return :(Spinor{V}($(Expr(:call,tvecs(N,t),out...))))
        else quote
            $(insert_expr((:N,:t),VEC)...)
            out = zeros(mvecs(N,t))
            setspin!(out,value(a,t),UInt(basis(a)),Val{N}())
            setspin!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
            return Spinor{V}(out)
        end end
    end
    @noinline function adderanti(a::Type{<:TensorTerm{V,L}},b::Type{<:TensorTerm{V,G}},op) where {V,L,G}
        (iseven(L) || iseven(G)) && (return :(error("$(basis(a)) and $(basis(b)) are not expressible as AntiSpinor")))
        left,bop,VEC = addvec(a,b,false,op)
        if mdims(V)-1<cache_limit
            $(insert_expr((:N,),:svecs)...)
            t = promote_type(valuetype(a),valuetype(b))
            out,ib = svecs(N,Any)(zeros(svecs(N,t))),indexbasis(N)
            setanti!_pre(out,:(value(a,$t)),UInt(basis(a)),Val{N}())
            setanti!_pre(out,:($bop(value(b,$t))),UInt(basis(b)),Val{N}())
            return :(AntiSpinor{V}($(Expr(:call,tvecs(N,t),out...))))
        else quote
            $(insert_expr((:N,:t),VEC)...)
            out = zeros(mvecs(N,t))
            setanti!(out,value(a,t),UInt(basis(a)),Val{N}())
            setanti!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
            return AntiSpinor{V}(out)
        end end
    end
    @noinline function addermulti(a::Type{<:TensorTerm{V,L}},b::Type{<:TensorTerm{V,G}},op) where {V,L,G}
        left,bop,VEC = addvec(a,b,false,op)
        if mdims(V)<cache_limit
            $(insert_expr((:N,:t),:mvec)...)
            out,ib = svec(N,Any)(zeros(svec(N,t))),indexbasis(N)
            setmulti!_pre(out,:(value(a,$t)),UInt(basis(a)),Val{N}())
            setmulti!_pre(out,:($bop(value(b,$t))),UInt(basis(b)),Val{N}())
            return :(Multivector{V}($(Expr(:call,tvec(N,t),out...))))
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
            out = svec(N,G,Any)(zeros(svec(N,G,t)))
            X = UInt(basis(a))
            for k ∈ list(1,binomial(N,G))
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
        if !istangent(V) && !hasconformal(V) && L == 0 && G == mdims(V) &&
                valuetype(a)<:Real && valuetype(b)<:Real
            if swap
                :(Couple{V,basis(V)}(Complex($right(value(a)),b.v[1])))
            else
                :(Couple{V,basis(V)}(Complex(value(a),$right(b.v[1]))))
            end
        elseif !istangent(V) && !hasconformal(V) && G == 0 &&
                valuetype(a)<:Real && valuetype(b)<:Real
            if swap
                :(Couple{V,basis(a)}(Complex(b.v[1],$right(value(a)))))
            else
                :(Couple{V,basis(a)}(Complex($right(b.v[1]),value(a))))
            end
        elseif !istangent(V) && !hasconformal(V) && G == grade(V) &&
                valuetype(a)<:Real && valuetype(b)<:Real
            if swap
                :(PseudoCouple{V,basis(a)}(Complex($right(value(a)),b.v[1])))
            else
                :(PseudoCouple{V,basis(a)}(Complex(value(a),$right(b.v[1]))))
            end
        elseif iseven(L) && iseven(G)
            if mdims(V)-1<cache_limit
                $(insert_expr((:N,:ib,:bn,:t),:mvecs)...)
                out = svecs(N,Any)(zeros(svecs(N,t)))
                X = UInt(basis(a))
                for k ∈ list(1,binomial(N,G))
                    B = @inbounds ib[k]
                    val = :(@inbounds $right(b.v[$k]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    @inbounds setspin!_pre(out,val,B,Val(N))
                end
                val = :(@inbounds $left(value(a,$t)))
                setspin!_pre(out,val,X,Val(N))
                return :(Spinor{V}($(Expr(:call,tvecs(N,t),out...))))
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
                out = svecs(N,Any)(zeros(svecs(N,t)))
                X = UInt(basis(a))
                for k ∈ list(1,binomial(N,G))
                    B = @inbounds ib[k]
                    val = :(@inbounds $right(b.v[$k]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    @inbounds setanti!_pre(out,val,B,Val(N))
                end
                val = :(@inbounds $left(value(a,$t)))
                setanti!_pre(out,val,X,Val(N))
                return :(AntiSpinor{V}($(Expr(:call,tvecs(N,t),out...))))
            else return if !swap; quote
                $(insert_expr((:N,:t,:out,:rrr,:bng),VECS)...)
                @inbounds out[rrr+1:rrr+bng] = $(bcast(right,:(value(b,$VEC(N,G,t)),)))
                addpseudo(out,value(a,t),UInt(basis(a)),Val(N))
                return AntiSpinor{V}(out)
            end; else quote
                $(insert_expr((:N,:t,:out,:rrr,:bng),VECS)...)
                @inbounds out[rrr+1:rrr+bng] = value(a,$VEC(N,G,t))
                addanti!(out,$left(value(b,t)),UInt(basis(b)),Val(N))
                return AntiSpinor{V}(out)
            end end end
        else
            if mdims(V)<cache_limit
                $(insert_expr((:N,:ib,:bn,:t),:mvec)...)
                out = svec(N,Any)(zeros(svec(N,t)))
                X = UInt(basis(a))
                for k ∈ list(1,binomial(N,G))
                    B = @inbounds ib[k]
                    val = :(@inbounds $right(b.v[$k]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    @inbounds setmulti!_pre(out,val,B,Val(N))
                end
                val = :(@inbounds $left(value(a,$t)))
                setmulti!_pre(out,val,X,Val(N))
                return :(Multivector{V}($(Expr(:call,tvec(N,t),out...))))
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
            out = svec(N,Any)(zeros(svec(N,t)))
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
            return :(Multivector{V}($(Expr(:call,tvec(N,t),out...))))
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
            out = svecs(N,Any)(zeros(svecs(N,t)))
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
            return :(Spinor{V}($(Expr(:call,tvecs(N,t),out...))))
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
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:AntiSpinor{V,T}},op,swap=false) where {V,G,T}
        left,right,VEC = addvec(a,b,swap,op)
        VECS = Symbol(string(VEC)*"s")
        !isodd(G) && (return swap ? :($op(Multivector(b),a)) : :($op(a,Multivector(b))))
        if mdims(V)<cache_limit
            $(insert_expr((:N,:ps,:bn,:t),:mvecs)...)
            out = svecs(N,Any)(zeros(svecs(N,t)))
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
            return :(AntiSpinor{V}($(Expr(:call,tvecs(N,t),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t),VEC)...)
            out = convert($VECS(N,t),$(bcast(right,:(value(b,$VECS(N,t)),))))
            addanti!(out,value(a,t),UInt(basis(a)),Val(N))
            return AntiSpinor{V}(out)
        end; else quote
            $(insert_expr((:N,:t),VEC)...)
            out = value(a,$VECS(N,t))
            addpseudo(out,$left(value(b,t)),UInt(basis(b)),Val(N))
            return AntiSpinor{V}(out)
        end end end
    end
    @noinline function product(a::Type{S},b::Type{<:Chain{V,G,T}},swap=false) where S<:TensorGraded{V,L} where {V,G,L,T}
        MUL,VEC = mulvecs(a,b)
        anti = isodd(L) ≠ isodd(G)
        type = anti ? :AntiSpinor : :Spinor
        if G == 0
            return S<:Chain ? :(Chain{V,L}(broadcast($MUL,a.v,Ref(@inbounds b[1])))) : swap ? :(Single(b)⟑a) : :(a⟑Single(b))
        elseif S<:Chain && L == 0
            return :(Chain{V,G}(broadcast($MUL,Ref(@inbounds a[1]),b.v)))
        elseif (swap ? L : G) == mdims(V) && !istangent(V)
            return swap ? (S<:Single ? :(⋆(~b)*value(a)) : :(⋆(~b))) : :(@inbounds ⋆(~a)*b[1])
        elseif (swap ? G : L) == mdims(V) && !istangent(V)
            return swap ? :(b[1]*complementlefthodge(~a)) : S<:Single ? :(value(a)*complementlefthodge(~b)) : S<:Chain ? :(@inbounds a[1]*complementlefthodge(~b)) : :(complementlefthodge(~b))
        elseif binomial(mdims(V),G)*(S<:Chain ? binomial(mdims(V),L) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:bng,:ib,:μ),:mvecs)...)
                out = svecs(N,Any)(zeros(svecs(N,t)))
                B = indexbasis(N,L)
                for i ∈ list(1,binomial(N,L))
                    @inbounds v,ibi = :(@inbounds a[$i]),B[i]
                    for j ∈ 1:bng
                        @inbounds (anti ? geomaddanti!_pre : geomaddspin!_pre)(V,out,ibi,ib[j],derive_pre(V,ibi,ib[j],v,:(@inbounds b[$j]),MUL))
                    end
                end
            else
                $(insert_expr((:N,:t,:ib,:μ),:mvecs)...)
                out = svecs(N,Any)(zeros(svecs(N,t)))
                U = UInt(basis(a))
                for i ∈ list(1,binomial(N,G))
                    A,B = swap ? (@inbounds ib[i],U) : (U,@inbounds ib[i])
                    if S<:Single
                        @inbounds (anti ? geomaddanti!_pre : geomaddspin!_pre)(V,out,A,B,derive_pre(V,A,B,:(a.v),:(@inbounds b[$i]),MUL))
                    else
                        @inbounds (anti ? geomaddanti!_pre : geomaddspin!_pre)(V,out,A,B,derive_pre(V,A,B,:(@inbounds b[$i]),false))
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
    @noinline function product_contraction(a::Type{S},b::Type{<:Chain{V,G,T}},swap=false,contr=:contraction) where S<:TensorGraded{V,L} where {V,G,T,L}
        MUL,VEC = mulvec(a,b,contr)
        (swap ? G<L : L<G) && (!istangent(V)) && (return Zero(V))
        if (G==0 || G==mdims(V)) && (!istangent(V))
            return swap ? :(contraction(Single(b),a)) : :(contraction(a,Single(b)))
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
                            @inbounds skewaddmulti!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(@inbounds b[$j]),MUL))
                        else
                            @inbounds skewaddblade!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(@inbounds b[$j]),MUL))
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

for input ∈ (:Multivector,:Spinor,:AntiSpinor)
    inspin,inanti = input==:Spinor,input==:AntiSpinor
for (op,product) ∈ ((:∧,:exteradd),(:*,:geomadd),
                     (:∨,:meetadd),(:contraction,:skewadd))
    outspin = product ∈ (:exteradd,:geomadd,:skewadd)
    outmulti = input == :Multivector
    outype = outmulti ? :Multivector : outspin ? :($(inspin ? :isodd : :iseven)(G) ? AntiSpinor : Spinor) : inspin ?  :(isodd(G)⊻isodd(N) ? AntiSpinor : Spinor) : :(isodd(G)⊻isodd(N) ? Spinor : AntiSpinor)
    product! = outmulti ? Symbol(product,:multi!) : outspin ? :($(inspin ? :isodd : :iseven)(G) ? $(Symbol(product,:anti!)) : $(Symbol(product,:spin!))) : :(isodd(G)⊻isodd(N) ? $(Symbol(product,outspin⊻inspin ? :anti! : :spin!)) : $(Symbol(product,outspin⊻inspin ? :spin! : :anti!)))
    preproduct! = outmulti ? Symbol(product,:multi!_pre) : outspin ? :($(inspin ? :isodd : :iseven)(G) ? $(Symbol(product,:anti!_pre)) : $(Symbol(product,:spin!_pre))) : :(isodd(G)⊻isodd(N) ? $(Symbol(product,outspin⊻inspin ? :anti!_pre : :spin!_pre)) : $(Symbol(product,outspin⊻inspin ? :spin!_pre : :anti!_pre)))
    prop = op≠:* ? Symbol(:product_,op) : :product
    outmulti && @eval $prop(a,b,swap=false) = $prop(typeof(a),typeof(b),swap)
    mgrade,nmgrade = op≠:∧ ? (:maxgrade,:nextmaxgrade) : (:mingrade,:nextmingrade)
    @eval @noinline function $prop(a::Type{S},b::Type{<:$input{V,T}},swap=false) where S<:TensorGraded{V,G} where {V,G,T}
        MUL,VEC = mulvec(a,b,$(QuoteNode(op)))
        VECS = isodd(G) ? VEC : string(VEC)*"s"
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
                return swap ? :($$op(b(Val(maxgrade(b))),a)) : :($$op(a,b(Val(mingrade(b)))))
            elseif (swap ? nextmaxgrade(b)==G : G+nextmaxpseudograde(b)==mdims(V)) && (!istangent(V))
                return swap ? :($$op(b(Val(maxgrade(b))),a)+$$op(b(Val(nextmaxgrade(b))),a)) : :($$op(a,b(Val(mingrade(b))))+$$op(a,b(Val(nextmingrade(b)))))
            end
        end; elseif op == :*; quote
            if S<:Chain && G == 0
                return :($input{V,G}(broadcast($MUL,Ref(@inbounds a[1]),b.v)))
            elseif G == mdims(V) && !istangent(V)
                return if swap
                    S<:Single ? :(⋆(~b)*value(a)) : :(⋆(~b))
                else
                    S<:Single ? :(value(a)*complementlefthodge(~b)) : S<:Chain ? :(@inbounds a[1]*complementlefthodge(~b)) : :(complementlefthodge(~b))
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
                        for j ∈ 1:bn[G+1]
                            A,B = swapper(ib[j],ia[i],swap)
                            X,Y = swapper(:(@inbounds a[$j]),val,swap)
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,X,Y,MUL))
                        end
                    else
                        U = UInt(basis(a))
                        A,B = swapper(U,ia[i],swap)
                        if S<:Single
                            X,Y = swapper(:(a.v),:(@inbounds b.v[$(bs[g]+i)]),swap)
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,X,Y,MUL))
                        else
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,:(@inbounds b.v[$(bs[g]+i)]),false))
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
        end end
    end
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

for com ∈ (:spinor,:s_m,:m_s,:anti,:a_m,:m_a,:multivector,:s_a,:a_s)
    outspin = com ∈ (:spinor,:anti,:s_a,:a_s)
    left,leftspin,right,rightspin,br = leftrightsym(com)
    VEC = outspin ? :svecs : :svec
    genloop = Symbol(:generate_loop_,com)
    @eval @noinline function $genloop(V,a,b,t,MUL,product!,preproduct!,d=nothing)
        if mdims(V)<cache_limit/2
            $(insert_expr((:N,br...,:bn),:mvec)...)
            out = $VEC(N,Any)(zeros($VEC(N,t)))
            for g ∈ $leftspin
                X = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    @inbounds val = nothing≠d ? :(@inbounds $a[$($left[g]+i)]/$d) : :(@inbounds $a[$($left[g]+i)])
                    for G ∈ $rightspin
                        @inbounds R = $right[G]
                        Y = indexbasis(N,G-1)
                        @inbounds for j ∈ 1:bn[G]
                            @inbounds preproduct!(V,out,X[i],Y[j],derive_pre(V,X[i],Y[j],val,:(@inbounds $b[$(R+j)]),MUL))
                        end
                    end
                end
            end
            (:N,:t,:out), :(out = $(Expr(:call,$(outspin ? :tvecs : :tvec)(N,t),out...)))
        else
            (:N,:t,:out,br...,:bn,:μ), quote
                for g ∈ $leftspin
                    X = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds val = $(nothing≠d ? :(@inbounds $a[$left[g]+i]/$d) : :(@inbounds $a[$left[g]+i]))
                        val≠0 && for G ∈ $rightspin
                            @inbounds R = $right[G]
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
end
