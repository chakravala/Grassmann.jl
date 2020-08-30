
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

function mul(a::SubManifold{V},b::SubManifold{V},der=derive_mul(V,bits(a),bits(b),1,true)) where V
    ba,bb = bits(a),bits(b)
    (diffcheck(V,ba,bb) || iszero(der)) && (return g_zero(V))
    A,B,Q,Z = symmetricmask(V,bits(a),bits(b))
    pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(V,A,B) : (false,A⊻B,false)
    d = getbasis(V,bas|Q)
    out = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(a,b)⊻pcc ? Simplex{V}(-1,d) : d) : Simplex{V}((pcc ? -1 : 1)*parityinner(V,A,B),d)
    diffvars(V)≠0 && !iszero(Z) && (out = Simplex{V}(getbasis(loworder(V),Z),out))
    return cc ? (v=value(out);out+Simplex{V}(hasinforigin(V,A,B) ? -(v) : v,getbasis(V,conformalmask(V)⊻bits(d)))) : out
end

function *(a::Simplex{V},b::SubManifold{V}) where V
    v = derive_mul(V,bits(basis(a)),bits(b),a.v,true)
    bas = mul(basis(a),b,v)
    order(a.v)+order(bas)>diffmode(V) ? zero(V) : Simplex{V}(v,bas)
end
function *(a::SubManifold{V},b::Simplex{V}) where V
    v = derive_mul(V,bits(a),bits(basis(b)),b.v,false)
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
@generated ∧(t::T) where T<:FixedVector{N} where N = wedges([:(t[$i]) for i ∈ 1:N])
∧(::Values{0,<:Chain{V}}) where V = one(V) # ∧() = 1
∧(::FixedVector{0,<:Chain{V}}) where V = one(V)
∧(t::Chain{V,1,<:Chain} where V) = ∧(value(t))
∧(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = ∧(a∧b,c...)

wedges(x,i=length(x)-1) = i ≠ 0 ? Expr(:call,:∧,wedges(x,i-1),x[1+i]) : x[1+i]

@pure function ∧(a::SubManifold{V},b::SubManifold{V}) where V
    ba,bb = bits(a),bits(b)
    A,B,Q,Z = symmetricmask(V,ba,bb)
    ((count_ones(A&B)>0) || diffcheck(V,ba,bb) || iszero(derive_mul(V,ba,bb,1,true))) && (return g_zero(V))
    d = getbasis(V,(A⊻B)|Q)
    diffvars(V)≠0 && !iszero(Z) && (d = Simplex{V}(getbasis(loworder(V),Z),d))
    return parity(a,b) ? Simplex{V}(-1,d) : d
end

function ∧(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    x,y = basis(a), basis(b)
    ba,bb = bits(x),bits(y)
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

⊗(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = a∧b

## regressive product: (L = grade(a) + grade(b); (-1)^(L*(L-mdims(V)))*⋆(⋆(a)∧⋆(b)))

@pure function ∨(a::SubManifold{V},b::SubManifold{V}) where V
    p,C,t,Z = regressive(a,b)
    (!t || iszero(derive_mul(V,bits(a),bits(b),1,true))) && (return g_zero(V))
    d = getbasis(V,C)
    istangent(V) && !iszero(Z) && (d = Simplex{V}(getbasis(loworder(V),Z),d))
    return isone(p) ? d : Simplex{V}(p,d)
end

function ∨(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    ba,bb = bits(basis(a)),bits(basis(b))
    p,C,t,Z = regressive(V,ba,bb)
    !t  && (return g_zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.∏)
    if istangent(V) && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,bits(basis(a)),bits(basis(b)))
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
    (!t || iszero(derive_mul(V,bits(a),bits(b),1,true))) && (return g_zero(V))
    d = getbasis(V,C)
    istangent(V) && !iszero(Z) && (d = Simplex{V}(getbasis(loworder(V),Z),d))
    return typeof(V) <: Signature ? (g ? Simplex{V}(-1,d) : d) : Simplex{V}(g,d)
end

function contraction(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    ba,bb = bits(basis(a)),bits(basis(b))
    g,C,t,Z = interior(V,ba,bb)
    !t && (return g_zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.∏)
    if istangent(V) && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,bits(basis(a)),bits(basis(b)))
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

contraction(a::Chain{W,G},b::Chain{V,1,<:Chain}) where {W,G,V} = Chain{V,1}(column(Ref(a).⋅value(b)))
contraction(a::Chain{W,G,<:Chain},b::Chain{V,1,<:Chain}) where {W,G,V} = Chain{V,1}(Ref(a).⋅value(b))
Base.:(:)(a::Chain{V,1,<:Chain},b::Chain{V,1,<:Chain}) where V = sum(value(a).⋅value(b))

# dyadic identity element

Base.:+(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling{Bool}) where V = t+g
Base.:+(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling) where V = t+g
Base.:-(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling{Bool}) where V = t+g
Base.:-(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling) where V = t+g
@generated Base.:+(t::LinearAlgebra.UniformScaling{Bool},g::Chain{V,1,<:Chain{V,1}}) where V = :(Chain{V,1}($(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)]).+value(g)))
@generated Base.:+(t::LinearAlgebra.UniformScaling,g::Chain{V,1,<:Chain{V,1}}) where V = :(Chain{V,1}(t.λ*$(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)]).+value(g)))
@generated Base.:-(t::LinearAlgebra.UniformScaling{Bool},g::Chain{V,1,<:Chain{V,1}}) where V = :(Chain{V,1}($(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)]).-value(g)))
@generated Base.:-(t::LinearAlgebra.UniformScaling,g::Chain{V,1,<:Chain{V,1}}) where V = :(Chain{V,1}(t.λ*$(getalgebra(V).b[Grassmann.list(2,mdims(V)+1)]).-value(g)))

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

for X ∈ TAG, Y ∈ TAG
    @eval ⊘(x::X,y::Y) where {X<:$X{V},Y<:$Y{V}} where V = diffvars(V)≠0 ? conj(y)*x*y : y\x*involute(y)
end
for Z ∈ TAG
    @eval ⊘(x::Chain{V,G},y::T) where {V,G,T<:$Z} = diffvars(V)≠0 ? conj(y)*x*y : ((~y)*x*involute(y))(Val(G))/abs2(y)
end


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
    out = basis(v)
    for k ∈ 1:(i-1)%4
        out *= basis(v)
    end
    return typeof(v)<:SubManifold ? out : out*AbstractTensors.:^(value(v),i)
end

function Base.:^(v::T,i::S) where {T<:TensorAlgebra,S<:Integer}
    V = Manifold(v)
    isone(i) && (return v)
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
            value(sd) ≈ fd && (return $d(rm,sd))
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

### Sum Algebra Constructor

const NSE = Union{Symbol,Expr,<:Real,<:Complex}

Base.:-(t::SubManifold) = Simplex(-value(t),t)

for (op,eop) ∈ ((:+,:(+=)),(:-,:(-=)))
    for Term ∈ (:TensorGraded,:TensorMixed)
        @eval begin
            $op(a::T,b::NSE) where T<:$Term = iszero(b) ? a : $op(a,b*g_one(Manifold(a)))
            $op(a::NSE,b::T) where T<:$Term = iszero(a) ? $op(b) : $op(a*g_one(Manifold(b)),b)
        end
    end
    @eval begin
        function $op(a::SparseChain{V,G,T},b::SparseChain{V,G,S}) where {V,G,T,S}
            isempty(a.v.nzval) && (return b)
            isempty(b.v.nzval) && (return a)
            t = length(a.v.nzind) > length(b.v.nzind)
            bi,bv = value(t ? b : a).nzind,value(t ? b : a).nzval
            out = convert(SparseVector{promote_type(T,S),Int},copy(value(t ? a : b)))
            $(Expr(eop,:(out[bi]),:bv))
            SparseChain{V,G}(out)
        end
        function $op(a::SparseChain{V,G,S},b::T) where T<:TensorTerm{V,G} where {V,G,S}
            out = convert(SparseVector{promote_type(S,valuetype(b)),Int},copy(value(a)))
            $(Expr(eop,:(out[basisindex(mdims(V),bits(b))]),:(value(b))))
            SparseChain{V,G}(out)
        end
    end
    for Tens ∈ (:(TensorTerm{V,B}),:(Chain{T,V,B} where T))
        @eval $op(a::T,b::SparseChain{V,A}) where {T<:$Tens} where {V,A,B} = b+a
    end
    @eval begin
        @generated function $op(a::SubManifold{V,A},b::SubManifold{V,B}) where {V,A,B}
            adder(a,b,$(QuoteNode(op)),:mvec)
        end
        function $op(a::MultiGrade{V,A},b::MultiGrade{V,B}) where {V,A,B}
            at,bt = terms(a),terms(b)
            isempty(at) && (return b)
            isempty(bt) && (return a)
            bl = length(bt)
            out = convert(Vector{TensorGraded{V}},at)
            N = mdims(V)
            i,k,bk = 0,1,rank(out[1])
            while i < bl
                k += 1
                i += 1
                bas = rank(bt[i])
                if bas == bk
                    $(Expr(eop,:(out[k-1]),:(bt[i])))
                    k < length(out) ? (bk = rank(out[k])) : (k -= 1)
                elseif bas<bk
                    insert!(out,k-1,bt[i])
                elseif k ≤ length(out)
                    bk = rank(out[k])
                    i -= 1
                else
                    insert!(out,k,bt[i])
                end
            end
            G = A|B
            MultiGrade{V,G}(Values{count_ones(G),TensorGraded{V}}(out))
        end
        function $op(a::MultiGrade{V,A},b::T) where T<:TensorGraded{V,B} where {V,A,B}
            N = mdims(V)
            out = convert(Vector{TensorGraded{V}},terms(a))
            i,k,bk,bl = 0,1,rank(out[1]),length(out)
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
                    bk = rank(out[k])
                else
                    insert!(out,k,b)
                    break
                end
            end
            G = A|(UInt(1)<<B)
            MultiGrade{V,G}(Values{count_ones(G),TensorGraded{V}}(out))
        end
        $op(a::SparseChain{V,A},b::T) where T<:TensorGraded{V,B} where {V,A,B} = MultiGrade{V,(UInt(1)<<A)|(UInt(1)<<B)}(A<B ? Values(a,b) : Values(b,a))
    end
end

@inline swapper(a,b,swap) = swap ? (b,a) : (a,b)

adder(a,b,left=:+,right=:+) = adder(typeof(a),typeof(b),left,right)
adder(a::Type,b::Type,left=:+,right=:+) = adder(a,b,left,right,:mvec)

@eval begin
    @noinline function adder(a::Type{<:TensorTerm{V,A}},b::Type{<:TensorTerm{V,B}},bop=:+,VEC=:mvec) where {V,A,B}
        if basis(a) == basis(b)
            :(Simplex{V,A}($bop(value(a),value(b)),basis(a)))
        elseif A == B; G = A
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
                    :(Chain{V,A}($(Expr(:call,tvec(N,G,:t),out...)))))
            else return quote
                $(insert_expr((:N,:t))...)
                out = zeros($VEC(N,A,t))
                setblade!(out,value(a,t),UInt(basis(a)),Val{N}())
                setblade!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
                return Chain{V,A}(out)
            end end
        else quote
            #@warn("sparse MultiGrade{V} objects not properly handled yet")
            #return MultiGrade{V}(a,b)
            $(insert_expr((:N,:t,:out))...)
            setmulti!(out,value(a,t),UInt(basis(a)),Val{N}())
            setmulti!(out,$bop(value(b,t)),UInt(basis(b)),Val{N}())
            return MultiVector{V}(out)
        end end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:Chain{V,G,T}},left,right,VEC,swap=false) where {V,G,T}
        if binomial(mdims(V),G)<(1<<cache_limit)
            $(insert_expr((:N,:ib,:t),:svec)...)
            out = zeros(svec(N,G,Any))
            X = UInt(basis(a))
            for k ∈ 1:binomial(N,G)
                B = ib[k]
                val = :($right(b.v[$k]))
                val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                @inbounds setblade!_pre(out,val,ib[k],Val{N}())
            end
            return :(Chain{V,G}($(Expr(:call,tvec(N,G,:T),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t),VEC)...)
            out = convert($VEC(N,G,t),$(bcast(right,:(value(b,$VEC(N,G,t)),))))
            addblade!(out,value(a,t),basis(a),Val{N}())
            return Chain{V,G}(out)
        end; else quote
            $(insert_expr((:N,:t),VEC)...)
            out = value(b,$VEC(N,G,t))
            addblade!(out,$left(value(a,t)),bits(basis(a)),Val{N}())
            return Chain{V,G}(out)
        end end end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,L}},b::Type{<:Chain{V,G,T}},left,right,VEC,swap=false) where {V,G,T,L}
        if mdims(V)<cache_limit
            $(insert_expr((:N,:ib,:bn,:t),:svec)...)
            out = zeros(svec(N,Any))
            X = UInt(basis(a))
            for k ∈ 1:binomial(N,G)
                B = ib[k]
                val = :($right(b.v[$k]))
                val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                @inbounds setmulti!_pre(out,val,B,Val(N))
            end
            for g ∈ 1:N+1
                g-1 == G && continue
                ib = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    B = ib[i]
                    if B == X
                        val = :($left(value(a,$t)))
                        @inbounds setmulti!_pre(out,val,B,Val(N))
                    end
                end
            end
            return :(MultiVector{V}($(Expr(:call,tvec(N,:T),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
            @inbounds out[r+1:r+bng] = $(bcast(right,:(value(b,$VEC(N,G,t)),)))
            addmulti!(out,value(a,t),bits(basis(a)),Val(N))
            return MultiVector{V}(out)
        end; else quote
            $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
            @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
            addmulti!(out,$left(value(b,t)),bits(basis(b)),Val(N))
            return MultiVector{V}(out)
        end end end
    end
    @noinline function adder(a::Type{<:TensorTerm{V,G}},b::Type{<:MultiVector{V,T}},left,right,VEC,swap=false) where {V,G,T}
        if mdims(V)<cache_limit
            $(insert_expr((:N,:bs,:bn,:t),:svec)...)
            out = zeros(svec(N,Any))
            X = UInt(basis(a))
            for g ∈ 1:N+1
                ib = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    B = ib[i]
                    val = :($right(b.v[$(bs[g]+i)]))
                    val = B==X ? Expr(:call,left,val,:(value(a,$t))) :  val
                    @inbounds setmulti!_pre(out,val,B,Val(N))
                end
            end
            return :(MultiVector{V}($(Expr(:call,tvec(N,:T),out...))))
        else return if !swap; quote
            $(insert_expr((:N,:t),VEC)...)
            out = convert($VEC(N,t),$(bcast(right,:(value(b,$VEC(N,t)),))))
            addmulti!(out,value(a,t),bits(basis(a)),Val(N))
            return MultiVector{V}(out)
        end; else quote
            $(insert_expr((:N,:t),VEC)...)
            out = value(a,$VEC(N,t))
            addmulti!(out,$left(value(b,t)),bits(basis(b)),Val(N))
            return MultiVector{V}(out)
        end end end
    end
    @noinline function product(a::Type{S},b::Type{<:Chain{V,G,T}},MUL,VEC,swap=false) where S<:TensorGraded{V,L} where {V,G,L,T}
        if G == 0
            return S<:Chain ? :(Chain{V,L}(broadcast($MUL,a.v,Ref(b[1])))) : swap ? :(b[1]*a) : :(a*b[1])
        elseif S<:Chain && L == 0
            return :(Chain{V,G}(broadcast($MUL,Ref(a[1]),b.v)))
        elseif (swap ? L : G) == mdims(V) && !istangent(V)
            return swap ? (S<:Simplex ? :(⋆(~b)*value(a)) : :(⋆(~b))) : S<:Chain ? :(a[1]*complementlefthodge(~b)) : :(⋆(~a)*b[1])
        elseif (swap ? G : L) == mdims(V) && !istangent(V)
            return swap ? :(b[1]*complementlefthodge(~a)) : S<:Simplex ? :(value(a)*complementlefthodge(~b)) : S<:Chain ? :(⋆(~a)*b[1]) : :(complementlefthodge(~b))
        elseif binomial(mdims(V),G)*(S<:Chain ? binomial(ndims(V),L) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:bng,:ib,:μ),:svec)...)
                out = zeros(svec(N,t))
                B = indexbasis(N,L)
                for i ∈ 1:binomial(N,L)
                    @inbounds v,ibi = :(a[$i]),B[i]
                    for j ∈ 1:bng
                        @inbounds geomaddmulti!_pre(V,out,ibi,ib[j],derive_pre(V,ibi,ib[j],v,:(b[$j]),MUL))
                    end
                end
            else
                $(insert_expr((:N,:t,:out,:ib,:μ),:svec)...)
                U = UInt(basis(a))
                for i ∈ 1:binomial(N,G)
                    A,B = swap ? (ib[i],U) : (U,ib[i])
                    if S<:Simplex
                        @inbounds geomaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(b[$i]),MUL))
                    else
                        @inbounds geomaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(b[$i]),false))
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
                A,B = swap ? (ib[i],U) : (U,ib[i])
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
    @noinline function product_contraction(a::Type{S},b::Type{<:Chain{V,G,T}},MUL,VEC,swap=false) where S<:TensorGraded{V,L} where {V,G,T,L}
        (swap ? G<L : L<G) && (!istangent(V)) && (return g_zero(V))
        GL = swap ? G-L : L-G
        if binomial(mdims(V),G)*(S<:Chain ? binomial(mdims(V),L) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:bng,:bnl),:svec)...)
                μ = istangent(V)|hasconformal(V)
                ia = indexbasis(N,L)
                ib = indexbasis(N,G)
                out = zeros(μ ? svec(N,Any) : svec(N,G-L,Any))
                for i ∈ 1:bnl
                    @inbounds v,iai = :(a[$i]),ia[i]
                    for j ∈ 1:bng
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(b[$j]),MUL))
                        else
                            @inbounds skewaddblade!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(b[$j]),MUL))
                        end
                    end
                end
            else
                $(insert_expr((:N,:t,:ib,:bng,:μ),:svec)...)
                out = zeros(μ ? svec(N,Any) : svec(N,GL,Any))
                U = UInt(basis(a))
                for i ∈ 1:bng
                    A,B = swap ? (ib[i],U) : (U,ib[i])
                    if S<:Simplex
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(b[$i]),MUL))
                        else
                            @inbounds skewaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(b[$i]),MUL))
                        end
                    else
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(b[$i]),false))
                        else
                            @inbounds skewaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(b[$i]),false))
                        end
                    end
                end
            end
            #return :(value_diff(Simplex{V,0,$(getbasis(V,0))}($(value(mv)))))
            return if μ
                insert_t(:(MultiVector{$V}($(Expr(:call,istangent(V) ? tvec(N) : tvec(N,:t),out...)))))
            else
                insert_t(:(Chain{$V,$GL}($(Expr(:call,tvec(N,GL,:t),out...)))))
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
                    A,B = swap ? (ib[i],U) : (U,ib[i])
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
    @eval @noinline function $prop(a::Type{S},b::Type{<:Chain{R,L,T}},MUL,VEC,swap=false) where S<:TensorGraded{Q,G} where {Q,R,T,G,L}
        w,W = swap ? (R,Q) : (Q,R)
        V = w==W ? w : ((w==dual(W)) ? (dyadmode(w)≠0 ? W⊕w : w⊕W) : (return :(interop($$op,a,b))))
        $po(G+L,mdims(V)) && (!istangent(V)) && (return g_zero(V))
        if binomial(mdims(W),L)*(S<:Chain ? binomial(mdims(w),G) : 1)<(1<<cache_limit)
            if S<:Chain
                $(insert_expr((:N,:t,:μ),:mvec,:T,:S)...)
                ia = indexbasis(mdims(w),L)
                ib = indexbasis(mdims(W),G)
                out = zeros(μ ? svec(N,Any) : svec(N,$GL,Any))
                CA,CB = isdual(w),isdual(W)
                for i ∈ 1:binomial(mdims(w),L)
                    @inbounds v,iai = :(a[$i]),ia[i]
                    x = CA ? dual(V,iai) : iai
                    for j ∈ 1:binomial(mdims(W),G)
                        X = @inbounds CB ? dual(V,ib[j]) : ib[j]
                        if μ
                            $grassaddmulti!_pre(V,out,x,X,derive_pre(V,x,X,v,:(b[$j]),MUL))
                        else
                            $grassaddblade!_pre(V,out,x,X,derive_pre(V,x,X,v,:(b[$j]),MUL))
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
                            $grassaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(b[$i]),MUL))
                        else
                            $grassaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(a.v),:(b[$i]),MUL))
                        end
                    else
                        if μ
                            $grassaddmulti!_pre(V,out,A,B,derive_pre(V,A,B,:(b[$i]),false))
                        else
                            $grassaddblade!_pre(V,out,A,B,derive_pre(V,A,B,:(b[$i]),false))
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
            out = zeros(μ $VEC(N,t) : $VEC(N,$$GL,t))
            CA,CB = isdual(L),isdual(R)
            for i ∈ 1:binomial(mdims(w),L)
                @inbounds v,iai = a[i],ia[i]
                x = CA ? dual(V,iai) : iai
                v≠0 && for j ∈ 1:binomial(mdims(W),G)
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
    @eval $prop(a,b,MUL=:+) = $prop(typeof(a),typeof(b),MUL)
    @eval $prop(a::Type,b::Type,MUL=:+) = $prop(a,b,MUL,:mvec)
    @eval @noinline function $prop(a::Type{S},b::Type{<:MultiVector{V,T}},MUL,VEC,swap=false) where S<:TensorGraded{V,G} where {V,G,T}
        if mdims(V)<cache_limit
            $(insert_expr((:N,:t,:out,:ib,:bs,:bn,:μ),:svec)...)
            for g ∈ 1:N+1
                ia = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    if S<:Chain
                        @inbounds val = :(b.v[$(bs[g]+i)])
                        for j ∈ 1:bn[G+1]
                            A,B = swapper(ib[j],ia[i],swap)
                            X,Y = swapper(:(a[$j]),val,swap)
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,X,Y,MUL))
                        end
                    else
                        U = UInt(basis(a))
                        A,B = swapper(U,ia[i],swap)
                        if S<:Simplex
                            X,Y = swapper(:(a.v),:(b.v[$(bs[g]+i)]),swap)
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,X,Y,MUL))
                        else
                            @inbounds $preproduct!(V,out,A,B,derive_pre(V,A,B,:(b.v[$(bs[g]+i)]),false))
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
                            A,B = $(swap ? :((ia[i],ib[j])) : :((ib[j],ia[i])))
                            X,Y = $(swap ? :((val,a[j])) : :((a[j],val)))
                            if @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,X,Y,$MUL))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,X,Y,$MUL))
                            end
                        end end
                    else quote
                        A,B = $(swap ? :((ia[i],$(UInt(basis(a))))) : :(($(UInt(basis(a))),ia[i])))
                        $(if S<:Simplex; quote
                            X,Y=$(swap ? :((b.v[bs[g]+1],a.v)) : :((a.v,b.v[bs[g]+1])))
                            if @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,X,Y,$MUL))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,B,derive_mul(V,A,B,X,Y,$MUL))
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
                @inbounds val = nothing≠d ? :($a[$(bs[g]+i)]/$d) : :($a[$(bs[g]+i)])
                for G ∈ 1:N+1
                    @inbounds R = bs[G]
                    Y = indexbasis(N,G-1)
                    @inbounds for j ∈ 1:bn[G]
                        @inbounds preproduct!(V,out,X[i],Y[j],derive_pre(V,X[i],Y[j],val,:($b[$(R+j)]),MUL))
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
                    @inbounds val = $(nothing≠d ? :($a[bs[g]+i]/$d) : :($a[bs[g]+i]))
                    val≠0 && for G ∈ 1:N+1
                        @inbounds R = bs[G]
                        Y = indexbasis(N,G-1)
                        @inbounds for j ∈ 1:bn[G]
                            if @inbounds $product!(V,out,X[i],Y[j],derive_mul(V,X[i],Y[j],val,$b[R+j],$MUL))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $product!(V,out,X[i],Y[j],derive_mul(V,X[i],Y[j],val,$b[R+j],$MUL))
                            end
                        end
                    end
                end
            end
        end
    end
end
