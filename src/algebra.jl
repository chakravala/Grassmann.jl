
#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Base: +, -, *, ^, /, //, inv, <, >, <<, >>, >>>
import AbstractTensors: ∧, ∨, ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, ∗, rem, div, contraction, TAG
import DirectSum: diffcheck, diffmode, tangent, hasinforigin, hasorigininf, symmetricsplit
export tangent

## mutating operations

@pure tvec(N,G,t::Symbol=:Any) = :(SVector{$(binomial(N,G)),$t})
@pure tvec(N,t::Symbol=:Any) = :(SVector{$(1<<N),$t})
@pure tvec(N,μ::Bool) = tvec(N,μ ? :Any : :t)

import DirectSum: g_one, g_zero, Field, ExprField
const Sym = :AbstractTensors
const SymField = Any

@pure derive(n::N,b) where N<:Number = zero(typeof(n))
derive(n,b,a,t) = t ? (a,derive(n,b)) : (derive(n,b),a)
#derive(n,b,a::T,t) where T<:TensorAlgebra = t ? (a,derive(n,b)) : (derive(n,b),a)

@inline function derive_mul(V,A,B,v,x::Bool)
    if !(istangent(V) && isdyadic(V))
        return v
    else
        sa,sb = symmetricsplit(V,A),symmetricsplit(V,B)
        ca,cb = count_ones(sa[2]),count_ones(sb[2])
        return if (ca == cb == 0) || ((ca ≠ 0) && (cb ≠ 0))
            v
        else
            prev = ca == 0 ? (x ? one(typeof(v)) : v) : (x ? v : one(typeof(v)))
            for k ∈ DirectSum.indexsplit((ca==0 ? sa : sb)[1],ndims(V))
                prev = derive(prev,getbasis(V,k))
            end
            prev
        end
    end
end

@inline function derive_mul(V,A,B,a,b,*)
    if !(istangent(V) && isdyadic(V))
        return a*b
    else
        sa,sb = symmetricsplit(V,A),symmetricsplit(V,B)
        ca,cb = count_ones(sa[2]),count_ones(sb[2])
        α,β = if (ca == cb == 0) || ((ca ≠ 0) && (cb ≠ 0))
            a,b
        else
            prev = ca == 0 ? (a,b) : (b,a)
            for k ∈ DirectSum.indexsplit((ca==0 ? sa : sb)[1],ndims(V))
                prev = derive(prev[2],getbasis(V,k),prev[1],true)
            end
            #base = getbasis(V,0)
            while typeof(prev[1]) <: TensorTerm
                basi = basis(prev[1])
                #base *= basi
                inds = DirectSum.indexsplit(bits(basi),ndims(V))
                prev = (value(prev[1]),prev[2])
                for k ∈ inds
                    prev = derive(prev[2],getbasis(V,k),prev[1],true)
                end
            end
            #base ≠ getbasis(V,0) && (prev = (base*prev[1],prev[2]))
            ca == 0 ? prev : (prev[2],prev[1])
        end
        return α*β
    end
end

function derive_pre(V,A,B,v,x)
    if !(istangent(V) && isdyadic(V))
        return v
    else
        return :(derive_post($V,$(Val{A}()),$(Val{B}()),$v,$x))
    end
end

function derive_pre(V,A,B,a,b,p)
    if !(istangent(V) && isdyadic(V))
        return Expr(:call,p,a,b)
    else
        return :(derive_post($V,$(Val{A}()),$(Val{B}()),$a,$b,$p))
    end
end

function derive_post(V,::Val{A},::Val{B},v,x::Bool) where {A,B}
    sa,sb = symmetricsplit(V,A),symmetricsplit(V,B)
    ca,cb = count_ones(sa[2]),count_ones(sb[2])
    return if (ca == cb == 0) || ((ca ≠ 0) && (cb ≠ 0))
        v
    else
        prev = ca == 0 ? (x ? one(typeof(v)) : v) : (x ? v : one(typeof(v)))
        for k ∈ DirectSum.indexsplit((ca==0 ? sa : sb)[1],ndims(V))
            prev = derive(prev,getbasis(V,k))
        end
        prev
    end
end

function derive_post(V,::Val{A},::Val{B},a,b,*) where {A,B}
    sa,sb = symmetricsplit(V,A),symmetricsplit(V,B)
    ca,cb = count_ones(sa[2]),count_ones(sb[2])
    α,β = if (ca == cb == 0) || ((ca ≠ 0) && (cb ≠ 0))
        a,b
    else
        prev = ca == 0 ? (a,b) : (b,a)
        for k ∈ DirectSum.indexsplit((ca==0 ? sa : sb)[1],ndims(V))
            prev = derive(prev[2],getbasis(V,k),prev[1],true)
        end
        #base = getbasis(V,0)
        while typeof(prev[1]) <: TensorTerm
            basi = basis(prev[1])
            #base *= basi
            inds = DirectSum.indexsplit(bits(basi),ndims(V))
            prev = (value(prev[1]),prev[2])
            for k ∈ inds
                prev = derive(prev[2],getbasis(V,k),prev[1],true)
            end
        end
        #base ≠ getbasis(V,0) && (prev = (base*prev[1],prev[2]))
        ca == 0 ? prev : (prev[2],prev[1])
    end
    return α*β
end

import DirectSum: loworder, isnull

bcast(op,arg) = op ∈ (:(AbstractTensors.:∑),:(AbstractTensors.:-)) ? Expr(:.,op,arg) : Expr(:call,op,arg.args...)

set_val(set,expr,val) = Expr(:(=),expr,set≠:(=) ? Expr(:call,:($Sym.:∑),expr,val) : val)

pre_val(set,expr,val) = set≠:(=) ? :(isnull($expr) ? ($expr=Expr(:call,:($Sym.:∑),$val)) : push!($expr.args,$val)) : Expr(:(=),expr,val)

add_val(set,expr,val,OP) = Expr(OP∉(:-,:+) ? :.= : set,expr,OP∉(:-,:+) ? Expr(:.,OP,Expr(:tuple,expr,val)) : val)

function generate_mutators(M,F,set_val,SUB,MUL)
    for (op,set) ∈ ((:add,:(+=)),(:set,:(=)))
        sm = Symbol(op,:multi!)
        sb = Symbol(op,:blade!)
        for (s,index) ∈ ((sm,:basisindex),(sb,:bladeindex))
            spre = Symbol(s,:_pre)
            @eval @inline function $s(out::$M,val::S,i) where {M,T,S}
                @inbounds $(set_val(set,:(out[i]),:val))
                return out
            end
            for (i,B) ∈ ((:i,UInt),(:(bits(i)),SubManifold))
                @eval begin
                    @inline function $s(out::$M,val::S,i::$B) where {M,T<:$F,S<:$F}
                        @inbounds $(set_val(set,:(out[$index(intlog(M),$i)]),:val))
                        return out
                    end
                    @inline function $s(out::Q,val::S,i::$B,::Val{N}) where Q<:$M where {M,T<:$F,S<:$F,N}
                        @inbounds $(set_val(set,:(out[$index(N,$i)]),:val))
                        return out
                    end
                end
            end
            for (i,B) ∈ ((:i,UInt),(:(bits(i)),SubManifold))
                @eval begin
                    @inline function $spre(out::$M,val::S,i::$B) where {M,T<:$F,S<:$F}
                        ind = $index(intlog(M),$i)
                        @inbounds $(pre_val(set,:(out[ind]),:val))
                        return out
                    end
                    @inline function $spre(out::Q,val::S,i::$B,::Val{N}) where Q<:$M where {M,T<:$F,S<:$F,N}
                        ind = $index(N,$i)
                        @inbounds $(pre_val(set,:(out[ind]),:val))
                        return out
                    end
                end
            end
        end
        for s ∈ (sm,sb)
            spre = Symbol(s,:_pre)
            @eval begin
                @inline function $(Symbol(:join,s))(V::W,m::$M,a::UInt,b::UInt,v::S) where W<:Manifold{N} where {N,T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        val = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(A,B,V) ? $SUB(v) : v) : $MUL(parityinner(A,B,V),v)
                        if diffvars(V)≠0
                            !iszero(Z) && (T≠Any ? (return true) : (val *= getbasis(loworder(V),Z)))
                            count_ones(Q)+order(val)>diffmode(V) && (return false)
                        end
                        $s(m,val,(A⊻B)|Q,Val{N}())
                    end
                    return false
                end
                @inline function $(Symbol(:join,spre))(V::W,m::$M,a::UInt,b::UInt,v::S) where W<:Manifold{N} where {N,T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        val = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(A,B,V) ? :($$SUB($v)) : v) : :($$MUL($(parityinner(A,B,V)),$v))
                        if diffvars(V)≠0
                            !iszero(Z) && (val = Expr(:call,:*,val,getbasis(loworder(V),Z)))
                            val = :(h=$val;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                        end
                        $spre(m,val,(A⊻B)|Q,Val{N}())
                    end
                    return false
                end
                @inline function $(Symbol(:geom,s))(V::W,m::$M,a::UInt,b::UInt,v::S) where W<:Manifold{N} where {N,T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(A,B,V) : (false,A⊻B,false)
                        val = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(A,B,V)⊻pcc ? $SUB(v) : v) : $MUL(parityinner(A,B,V),pcc ? $SUB(v) : v)
                        if istangent(V)
                            !iszero(Z) && (T≠Any ? (return true) : (val *= getbasis(loworder(V),Z)))
                            count_ones(Q)+order(val)>diffmode(V) && (return false)
                        end
                        $s(m,val,bas|Q,Val{N}())
                        cc && $s(m,hasinforigin(V,A,B) ? $SUB(val) : val,(conformalmask(V)⊻bas)|Q,Val{N}())
                    end
                    return false
                end
                @inline function $(Symbol(:geom,spre))(V::W,m::$M,a::UInt,b::UInt,v::S) where W<:Manifold{N} where {N,T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(A,B,V) : (false,A⊻B,false)
                        val = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(A,B,V)⊻pcc ? :($$SUB($v)) : v) : :($$MUL($(parityinner(A,B,V)),$(pcc ? :($$SUB($v)) : v)))
                        if istangent(V)
                            !iszero(Z) && (val = Expr(:call,:*,val,getbasis(loworder(V),Z)))
                            val = :(h=$val;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                        end
                        $spre(m,val,bas|Q,Val{N}())
                        cc && $spre(m,hasinforigin(V,A,B) ? :($$SUB($val)) : val,(conformalmask(V)⊻bas)|Q,Val{N}())
                    end
                    return false
                end
            end
            for j ∈ (:join,:geom)
                for S ∈ (s,spre)
                    @eval @inline function $(Symbol(j,S))(m::$M,v::S,A::SubManifold{V},B::SubManifold{V}) where {V,T<:$F,S<:$F,M}
                        $(Symbol(j,S))(V,m,bits(A),bits(B),v)
                    end
                end
            end
            for (prod,uct) ∈ ((:meet,:regressive),(:skew,:interior))
                @eval begin
                    @inline function $(Symbol(prod,s))(V::W,m::$M,A::UInt,B::UInt,val::T) where W<:Manifold{N} where {N,T,M}
                        if val ≠ 0
                            g,C,t,Z = $uct(A,B,V)
                            v = val
                            if istangent(V)
                                if !iszero(Z)
                                    T≠Any && (return true)
                                    _,_,Q,_ = symmetricmask(V,A,B)
                                    v *= getbasis(loworder(V),Z)
                                    count_ones(Q)+order(v)>diffmode(V) && (return false)
                                end
                            end
                            t && $s(m,typeof(V) <: Signature ? g ? $SUB(v) : v : $MUL(g,v),C,Val{N}())
                        end
                        return false
                    end
                    @inline function $(Symbol(prod,spre))(V::W,m::$M,A::UInt,B::UInt,val::T) where W<:Manifold{N} where {N,T,M}
                        if val ≠ 0
                            g,C,t,Z = $uct(A,B,V)
                            v = val
                            if istangent(V)
                                if !iszero(Z)
                                    _,_,Q,_ = symmetricmask(V,A,B)
                                    v = Expr(:call,:*,v,getbasis(loworder(V),Z))
                                    v = :(h=$v;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                                end
                            end
                            t && $spre(m,typeof(V) <: Signature ? g ? :($$SUB($v)) : v : Expr(:call,$(QuoteNode(MUL)),g,v),C,Val{N}())
                        end
                        return false
                    end

                end
                for S ∈ (s,spre)
                    @eval @inline function $(Symbol(prod,S))(m::$M,A::SubManifold{V},B::SubManifold{V},v::T) where {V,T,M}
                        $(Symbol(prod,S))(V,m,bits(A),bits(B),v)
                    end
                end
            end
        end
    end
end

@inline exterbits(V,α,β) = diffvars(V)≠0 ? ((a,b)=symmetricmask(V,α,β);count_ones(a&b)==0) : count_ones(α&β)==0

@inline exteraddmulti!(V::W,out,α,β,γ) where W<:Manifold = exterbits(V,α,β) && joinaddmulti!(V,out,α,β,γ)

@inline outeraddblade!(V::W,out,α,β,γ) where W<:Manifold = exterbits(V,α,β) && joinaddblade!(V,out,α,β,γ)

@inline exteraddmulti!_pre(V::W,out,α,β,γ) where W<:Manifold = exterbits(V,α,β) && joinaddmulti!_pre(V,out,α,β,γ)

@inline outeraddblade!_pre(V::W,out,α,β,γ) where W<:Manifold = exterbits(V,α,β) && joinaddblade!_pre(V,out,α,β,γ)

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
    pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(A,B,V) : (false,A⊻B,false)
    d = getbasis(V,bas|Q)
    out = (typeof(V)<:Signature || count_ones(A&B)==0) ? (parity(a,b)⊻pcc ? Simplex{V}(-1,d) : d) : Simplex{V}((pcc ? -1 : 1)*parityinner(A,B,V),d)
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
@generated ∧(t::T) where T<:SVector = Expr(:call,:∧,[:(t[$k]) for k ∈ 1:length(t)]...)
∧(m::Vector{Chain{V,G,T,X}} where {G,T,X}) where V = [∧(V[k]) for k ∈ value.(m)]
∧(a::X,b::Y,c::Z...) where {X<:TensorAlgebra,Y<:TensorAlgebra,Z<:TensorAlgebra} = ∧(a∧b,c...)

export ∧, ∨, ⊗

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

⊗(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = a∧b

## regressive product: (L = grade(a) + grade(b); (-1)^(L*(L-ndims(V)))*⋆(⋆(a)∧⋆(b)))

@pure function ∨(a::SubManifold{V},b::SubManifold{V}) where V
    p,C,t,Z = regressive(a,b)
    (!t || iszero(derive_mul(V,bits(a),bits(b),1,true))) && (return g_zero(V))
    d = getbasis(V,C)
    istangent(V) && !iszero(Z) && (d = Simplex{V}(getbasis(loworder(V),Z),d))
    return p ? Simplex{V}(-1,d) : d
end

function ∨(a::X,b::Y) where {X<:TensorTerm{V},Y<:TensorTerm{V}} where V
    ba,bb = bits(basis(a)),bits(basis(b))
    p,C,t,Z = regressive(ba,bb,V)
    !t  && (return g_zero(V))
    v = derive_mul(V,ba,bb,value(a),value(b),AbstractTensors.∏)
    if istangent(V) && !iszero(Z)
        _,_,Q,_ = symmetricmask(V,bits(basis(a)),bits(basis(b)))
        v = !(typeof(v)<:TensorTerm) ? Simplex{V}(v,getbasis(V,Z)) : Simplex{V}(v,getbasis(loworder(V),Z))
        count_ones(Q)+order(v)>diffmode(V) && (return zero(V))
    end
    return Simplex{V}(p ? -v : v,getbasis(V,C))
end

"""
    ∨(ω::TensorAlgebra,η::TensorAlgebra)

Regressive product as defined by the DeMorgan's law: ∨(ω...) = ⋆⁻¹(∧(⋆.(ω)...))
"""
@inline ∨(a::X,b::Y) where {X<:TensorAlgebra,Y<:TensorAlgebra} = interop(∨,a,b)
@inline ∨(a::TensorAlgebra{V},b::UniformScaling{T}) where {V,T<:Field} = a∨V(b)
@inline ∨(a::UniformScaling{T},b::TensorAlgebra{V}) where {V,T<:Field} = V(a)∨b
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
    g,C,t,Z = interior(ba,bb,V)
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
    @eval ⊘(x::$X{V},y::$Y{V}) where V = diffvars(V)≠0 ? conj(y)*x*y : y\x*involute(y)
end

@doc """
    ⊘(ω::TensorAlgebra,η::TensorAlgebra)

Sandwich product: ω⊘η = (~ω)⊖η⊖ω
""" Grassmann.:⊘

for X ∈ TAG, Y ∈ TAG
    @eval >>>(x::$X{V},y::$Y{V}) where V = x * y * ~x
end

@doc """
    ⊘(ω::TensorAlgebra,η::TensorAlgebra)

Sandwich product: ω>>>η = ω⊖η⊖(~ω)
""" Grassmann.:>>>

## linear algebra

export ⟂, ∥

∥(a,b) = iszero(a∧b)

# algebra

@eval begin
    *(a::F,b::MultiVector{V,T}) where {F<:Number,V,T} = MultiVector{V,promote_type(T,F)}(broadcast($Sym.:∏,Ref(a),b.v))
    *(a::MultiVector{V,T},b::F) where {F<:Number,V,T} = MultiVector{V,promote_type(T,F)}(broadcast($Sym.:∏,a.v,Ref(b)))
    *(a::F,b::Simplex{V,G,B,T} where B) where {F<:Number,V,G,T} = Simplex{V,G}($Sym.:∏(a,b.v),basis(b))
    *(a::Simplex{V,G,B,T} where B,b::F) where {F<:Number,V,G,T} = Simplex{V,G}($Sym.:∏(a.v,b),basis(a))
    *(a::F,b::Chain{V,G,T}) where {F<:Number,V,G,T} = Chain{V,G,promote_type(T,F)}(broadcast($Sym.:∏,Ref(a),b.v))
    *(a::Chain{V,G,T},b::F) where {F<:Number,V,G,T,} = Chain{V,G,promote_type(T,F)}(broadcast($Sym.:∏,a.v,Ref(b)))
end

for F ∈ Fields
    @eval begin
        *(a::F,b::MultiVector{V,T}) where {F<:$F,V,T<:Number} = MultiVector{V,promote_type(T,F)}(broadcast(*,Ref(a),b.v))
        *(a::MultiVector{V,T},b::F) where {F<:$F,V,T<:Number} = MultiVector{V,promote_type(T,F)}(broadcast(*,a.v,Ref(b)))
        *(a::F,b::Simplex{V,G,B,T} where B) where {F<:$F,V,G,T<:Number} = Simplex{V,G}(*(a,b.v),basis(b))
        *(a::Simplex{V,G,B,T} where B,b::F) where {F<:$F,V,G,T<:Number} = Simplex{V,G}(*(a.v,b),basis(a))
        *(a::F,b::Chain{V,G,T}) where {F<:$F,V,G,T<:Number} = Chain{V,G,promote_type(T,F)}(broadcast(*,Ref(a),b.v))
        *(a::Chain{V,G,T},b::F) where {F<:$F,V,G,T<:Number} = Chain{V,G,promote_type(T,F)}(broadcast(*,a.v,Ref(b)))
    end
end

for A ∈ (SubManifold,Simplex,Chain,MultiVector)
    for B ∈ (SubManifold,Simplex,Chain,MultiVector)
        @eval @inline *(a::$A,b::$B) = interop(*,a,b)
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

function ^(v::T,i::S) where {T<:TensorTerm,S<:Integer}
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
        @pure function $nv(b::SubManifold{V,G,B}) where {V,G,B}
            $d(parityreverse(grade(V,B)) ? -1 : 1,value(abs2_inv(b)))*b
        end
        @pure $d(a,b::T) where T<:TensorAlgebra = a*$nv(b)
        @pure $d(a::N,b::T) where {N<:Number,T<:TensorAlgebra} = a*$nv(b)
        function $nv(m::MultiVector{V,T}) where {V,T}
            rm = ~m
            d = rm*m
            fd = norm(d)
            sd = scalar(d)
            value(sd) ≈ fd && (return $d(rm,sd))
            for k ∈ 1:ndims(V)
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
            for k ∈ 1:ndims(V)
                @inbounds $Sym.:∑([$Sym.:∏(a,a) for a ∈ value(d[k])]...) == fd && (return $d(rm,d(k)))
            end
            throw(error("inv($m) is undefined"))
        end
        function $nv(b::Simplex{V,G,B,T}) where {V,G,B,T}
            Simplex{V,G,B}($d(parityreverse(grade(V,B)) ? -one(T) : one(T),value(abs2_inv(B)*value(b))))
        end
        function $nv(b::Simplex{V,G,B,Any}) where {V,G,B}
            Simplex{V,G,B}($Sym.$d(parityreverse(grade(V,B)) ? -1 : 1,value($Sym.:∏(abs2_inv(B),value(b)))))
        end
        function $nv(a::Chain)
            r,v,q = ~a,abs2(a),diffvars(Manifold(a))≠0
            q&&!(typeof(v)<:TensorGraded && grade(v)==0) ? $d(r,v) : $d(r,value(scalar(v)))
        end
    end
    for Term ∈ (:TensorGraded,:TensorMixed)
        @eval @pure $d(a::S,b::UniformScaling) where S<:$Term = a*$nv(Manifold(a)(b))
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

for (op,eop) ∈ ((:+,:(+=)),(:-,:(-=)))
    for Term ∈ (:TensorGraded,:TensorMixed)
        @eval begin
            $op(a::T,b::NSE) where T<:$Term = iszero(b) ? a : $op(a,b*one(Manifold(a)))
            $op(a::NSE,b::T) where T<:$Term = iszero(a) ? $op(b) : $op(a*one(Manifold(b)),b)
        end
    end
    @eval begin
        $op(a::SubManifold{V,G,B} where G) where {V,B} = Simplex($op(value(a)),a)
        function $op(a::SubManifold{V,A},b::SubManifold{V,B}) where {V,A,B}
            if a == b
                return Simplex{V,A}($op(value(a),value(b)),basis(a))
            elseif A == B
                $(insert_expr((:N,:t))...)
                out = zeros(mvec(N,A,t))
                setblade!(out,value(a,t),bits(a),Val{N}())
                setblade!(out,$op(value(b,t)),bits(b),Val{N}())
                return Chain{V,A}(out)
            else
                #@warn("sparse MultiGrade{V} objects not properly handled yet")
                #return MultiGrade{V}(a,b)
                $(insert_expr((:N,:t,:out))...)
                setmulti!(out,value(a,t),bits(a),Val{N}())
                setmulti!(out,$op(value(b,t)),bits(b),Val{N}())
                return MultiVector{V}(out)
            end
        end
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
            $(Expr(eop,:(out[basisindex(ndims(V),bits(b))]),:(value(b))))
            SparseChain{V,G}(out)
        end
        function $op(a::MultiGrade{V,A},b::MultiGrade{V,B}) where {V,A,B}
            at,bt = terms(a),terms(b)
            isempty(at) && (return b)
            isempty(bt) && (return a)
            bl = length(bt)
            out = convert(Vector{TensorGraded{V}},at)
            N = ndims(V)
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
            MultiGrade{V,G}(SVector{count_ones(G),TensorGraded{V}}(out))
        end
        function $op(a::MultiGrade{V,A},b::T) where T<:TensorGraded{V,B} where {V,A,B}
            N = ndims(V)
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
            MultiGrade{V,G}(SVector{count_ones(G),TensorGraded{V}}(out))
        end
        $op(a::SparseChain{V,A},b::T) where T<:TensorGraded{V,B} where {V,A,B} = MultiGrade{V,(UInt(1)<<A)|(UInt(1)<<B)}(A<B ? SVector(a,b) : SVector(b,a))
    end
    for Tens ∈ (:(TensorTerm{V,B}),:(Chain{T,V,B} where T))
        @eval $op(a::T,b::SparseChain{V,A}) where {T<:$Tens} where {V,A,B} = b+a
    end
end

function generate_sums(Field=Field,VEC=:mvec,MUL=:*,ADD=:+,SUB=:-,CONJ=:conj,PAR=false)
    if Field == Grassmann.Field
        generate_mutators(:(MArray{Tuple{M},T,1,M}),Number,Expr,SUB,MUL)
    elseif Field ∈ (SymField,:(SymPy.Sym))
        generate_mutators(:(SizedArray{Tuple{M},T,1,1}),Field,set_val,SUB,MUL)
    end
    PAR && (DirectSum.extend_field(Field); parsym = (parsym...,Field))
    TF = Field ∉ Fields ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    Field ∉ Fields && @eval begin
        Base.:*(a::F,b::SubManifold{V}) where {F<:$EF,V} = Simplex{V}(a,b)
        Base.:*(a::SubManifold{V},b::F) where {F<:$EF,V} = Simplex{V}(b,a)
        Base.:*(a::F,b::Simplex{V,G,B,T} where B) where {F<:$Field,V,G,T<:$Field} = Simplex{V,G}($MUL(a,b.v),basis(b))
        Base.:*(a::Simplex{V,G,B,T} where B,b::F) where {F<:$Field,V,G,T<:$Field} = Simplex{V,G}($MUL(a.v,b),basis(a))
        Base.adjoint(b::Simplex{V,G,B,T}) where {V,G,B,T<:$Field} = Simplex{dual(V),G,B',$TF}($CONJ(value(b)))
        Base.promote_rule(::Type{Simplex{V,G,B,T}},::Type{S}) where {V,G,T,B,S<:$Field} = Simplex{V,G,B,promote_type(T,S)}
        Base.promote_rule(::Type{Chain{V,G,T,B}},::Type{S}) where {V,G,T,B,S<:$Field} = Chain{V,G,promote_type(T,S),B}
        Base.promote_rule(::Type{MultiVector{V,T,B}},::Type{S}) where {V,T,B,S<:$Field} = MultiVector{V,promote_type(T,S),B}
    end
    @eval begin
        *(a::F,b::Chain{V,G,T}) where {F<:$Field,V,G,T<:$Field} = Chain{V,G}(broadcast($MUL,Ref(a),b.v))
        *(a::Chain{V,G,T},b::F) where {F<:$Field,V,G,T<:$Field} = Chain{V,G}(broadcast($MUL,a.v,Ref(b)))
        *(a::F,b::MultiVector{V,T}) where {F<:$Field,T,V} = MultiVector{V}(broadcast($Sym.∏,Ref(a),b.v))
        *(a::MultiVector{V,T},b::F) where {F<:$Field,T,V} = MultiVector{V}(broadcast($Sym.∏,a.v,Ref(b)))
        *(a::F,b::MultiGrade{V,G}) where {F<:$EF,V,G} = MultiGrade{V,G}(broadcast($MUL,Ref(a),b.v))
        *(a::MultiGrade{V,G},b::F) where {F<:$EF,V,G} = MultiGrade{V,G}(broadcast($MUL,a.v,Ref(b)))
        @generated function adjoint(m::Chain{V,G,T}) where {V,G,T<:$Field}
            if binomial(ndims(V),G)<(1<<cache_limit)
                if isdyadic(V)
                    $(insert_expr((:N,:M,:ib),:svec)...)
                    out = zeros(svec(N,G,Any))
                    for i ∈ 1:binomial(N,G)
                        @inbounds setblade!_pre(out,:($$CONJ(m.v[$i])),dual(V,ib[i],M),Val{N}())
                    end
                    return :(Chain{$(dual(V)),G}($(Expr(:call,tvec(N,TF),out...))))
                else
                    return :(Chain{$(dual(V)),G}($$CONJ.(value(m))))
                end
            else return quote
                if isdyadic(V)
                    $(insert_expr((:N,:M,:ib),$(QuoteNode(VEC)))...)
                    out = zeros($$VEC(N,G,$$TF))
                    for i ∈ 1:binomial(N,G)
                        @inbounds setblade!(out,$$CONJ(m.v[i]),dual(V,ib[i],M),Val{N}())
                    end
                else
                    out = $$CONJ.(value(m))
                end
                Chain{dual(V),G}(out)
            end end
        end
        @generated function adjoint(m::MultiVector{V,T}) where {V,T<:$Field}
            if ndims(V)<cache_limit
                if isdyadic(V)
                    $(insert_expr((:N,:M,:bs,:bn),:svec)...)
                    out = zeros(svec(N,Any))
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds setmulti!_pre(out,:($$CONJ(m.v[$(bs[g]+i)])),dual(V,ib[i],M))
                        end
                    end
                    return :(MultiVector{$(dual(V))}($(Expr(:call,tvec(N,TF),out...))))
                else
                    return :(MultiVector{$(dual(V))}($$CONJ.(value(m))))
                end
            else return quote
                if isdyadic(V)
                    $(insert_expr((:N,:M,:bs,:bn),$(QuoteNode(VEC)))...)
                    out = zeros($$VEC(N,$$TF))
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds setmulti!(out,$$CONJ(m.v[bs[g]+i]),dual(V,ib[i],M))
                        end
                    end
                else
                    out = $$CONJ.(value(m))
                end
                MultiVector{dual(V)}(out)
            end end
        end
    end
    for (op,eop,bop) ∈ ((:+,:(+=),ADD),(:-,:(-=),SUB))
        @eval begin
            function $op(a::Simplex{V,A,X,T},b::Simplex{V,B,Y,S}) where {V,A,X,T<:$Field,B,Y,S<:$Field}
                if X == Y
                    return Simplex{V,A}($bop(value(a),value(b)),X)
                elseif A == B
                    $(insert_expr((:N,:t),VEC)...)
                    out = zeros($VEC(N,A,t))
                    setblade!(out,value(a,t),bits(X),Val{N}())
                    setblade!(out,$bop(value(b,t)),bits(Y),Val{N}())
                    return Chain{V,A}(out)
                else
                    #@warn("sparse MultiGrade{V} objects not properly handled yet")
                    #return MultiGrade{V}(a,b)
                    $(insert_expr((:N,:t,:out),VEC)...)
                    setmulti!(out,value(a,t),bits(X),Val{N}())
                    setmulti!(out,$bop(value(b,t)),bits(Y),Val{N}())
                    return MultiVector{V}(out)
                end
            end
            $op(a::Simplex{V,G,B,T}) where {V,G,B,T<:$Field} = Simplex{V,G,B,$TF}($bop(value(a)))
            function $op(a::Simplex{V,A,X,T},b::SubManifold{V,B,Y}) where {V,A,X,T<:$Field,B,Y}
                if X == b
                    return Simplex{V,A}($bop(value(a),value(b)),b)
                elseif A == B
                    $(insert_expr((:N,:t),VEC)...)
                    out = zeros($VEC(N,A,t))
                    setblade!(out,value(a,t),bits(X),Val{N}())
                    setblade!(out,$bop(value(b,t)),Y,Val{N}())
                    return Chain{V,A}(out)
                else
                    #@warn("sparse MultiGrade{V} objects not properly handled yet")
                    #return MultiGrade{V}(a,b)
                    $(insert_expr((:N,:t,:out),VEC)...)
                    setmulti!(out,value(a,t),bits(X),Val{N}())
                    setmulti!(out,$bop(value(b,t)),Y,Val{N}())
                    return MultiVector{V}(out)
                end
            end
            function $op(a::SubManifold{V,A,X},b::Simplex{V,B,Y,S}) where {V,A,X,B,Y,S<:$Field}
                if a == Y
                    return Simplex{V,A}($bop(value(a),value(b)),a)
                elseif A == B
                    $(insert_expr((:N,:t),VEC)...)
                    out = zeros($VEC(N,A,t))
                    setblade!(out,value(a,t),X,Val{N}())
                    setblade!(out,$bop(value(b,t)),bits(Y),Val{N}())
                    return Chain{V,A}(out)
                else
                    #@warn("sparse MultiGrade{V} objects not properly handled yet")
                    #return MultiGrade{V}(a,b)
                    $(insert_expr((:N,:t,:out),VEC)...)
                    setmulti!(out,value(a,t),X,Val{N}())
                    setmulti!(out,$bop(value(b,t)),bits(Y),Val{N}())
                    return MultiVector{V}(out)
                end
            end
            function $op(a::Simplex{V,G,A,S} where A,b::MultiVector{V,T}) where {V,T<:$Field,G,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = convert($VEC(N,t),$(bcast(bop,:(copy(value(b,$VEC(N,t))),))))
                addmulti!(out,value(a,t),bits(basis(a)),Val{N}())
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::Simplex{V,G,B,S} where B) where {V,T<:$Field,G,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = copy(value(a,$VEC(N,t)))
                addmulti!(out,$bop(value(b,t)),bits(basis(b)),Val{N}())
                return MultiVector{V}(out)
            end
            $op(a::MultiVector{V,T}) where {V,T<:$Field} = MultiVector{V,$TF}($(bcast(bop,:(value(a),))))
            function $op(a::SubManifold{V,G},b::MultiVector{V,T}) where {V,T<:$Field,G}
                $(insert_expr((:N,:t),VEC)...)
                out = convert($VEC(N,t),$(bcast(bop,:(copy(value(b,$VEC(N,t))),))))
                addmulti!(out,value(a,t),bits(basis(a)),Val{N}())
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::SubManifold{V,G}) where {V,T<:$Field,G}
                $(insert_expr((:N,:t),VEC)...)
                out = copy(value(a,$VEC(N,t)))
                addmulti!(out,$bop(value(b,t)),bits(basis(b)),Val{N}())
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::MultiVector{V,S}) where {V,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = copy(value(a,$VEC(N,t)))
                $(add_val(eop,:out,:(value(b,$VEC(N,t))),bop))
                return MultiVector{V}(out)
            end
            function $op(a::SparseChain{V,G,S},b::MultiVector{V,T}) where {V,T<:$Field,G,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                at = value(a)
                out = convert($VEC(N,t),$(bcast(bop,:(copy(value(b,$VEC(N,t))),))))
                addmulti!(out,at.nzval,binomsum(N,G).+at.nzind)
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::SparseChain{V,G}) where {V,T<:$Field,G}
                $(insert_expr((:N,:t),VEC)...)
                bt = value(b)
                out = copy(value(a,$VEC(N,t)))
                addmulti!(out,$bop.(bt.nzval),binomsum(N,G).+bt.nzind)
                return MultiVector{V}(out)
            end
            function $op(a::MultiGrade{V,G},b::MultiVector{V,T}) where {V,T<:$Field,G}
                $(insert_expr((:N,),VEC)...)
                at = terms(a)
                t = promote_type(T,valuetype.(at)...)
                out = convert($VEC(N,t),$(bcast(bop,:(copy(value(b,$VEC(N,t))),))))
                for A ∈ at
                    TA = typeof(A)
                    if TA <: TensorTerm
                        addmulti!(out,value(A,t),bits(A),Val{N}())
                    elseif TA <: SparseChain
                        vA = value(A,t)
                        addmulti!(out,vA.nzval,vA.nzind)
                    else
                        g = rank(A)
                        r = binomsum(N,g)
                        @inbounds $(add_val(eop,:(out[r+1:r+binomial(N,g)]),:(value(A,$VEC(N,g,t))),bop))
                    end
                end
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::MultiGrade{V,G}) where {V,T<:$Field,G}
                $(insert_expr((:N,),VEC)...)
                bt = terms(b)
                t = promote_type(T,valuetype.(bt)...)
                out = copy(value(a,$VEC(N,t)))
                for B ∈ bt
                    TB = typeof(B)
                    if TB <: TensorTerm
                        addmulti!(out,$bop(value(B,t)),bits(B),Val{N}())
                    elseif TB <: SparseChain
                        vB = value(B,t)
                        addmulti!(out,vB.nzval,vB.nzind)
                    else
                        g = rank(B)
                        r = binomsum(N,g)
                        @inbounds $(add_val(eop,:(out[r+1:r+binomial(N,g)]),:(value(B,$VEC(N,g,t))),bop))
                    end
                end
                return MultiVector{V}(out)
            end
            function $op(a::Chain{V,G,T},b::Chain{V,L,S}) where {V,G,T<:$Field,L,S<:$Field}
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                rb = binomsum(N,L)
                Rb = binomial(N,L)
                @inbounds out[rb+1:rb+Rb] = $(bcast(bop,:(value(b,$VEC(N,L,t)),)))
                return MultiVector{V}(out)
            end
            function $op(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T<:$Field,S<:$Field}
                return Chain{V,G,promote_type(valuetype(a),valuetype(b))}($(bcast(bop,:(a.v,b.v))))
            end
            function $op(a::Chain{V,G,T},b::Simplex{V,G,B,S} where B) where {V,G,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = copy(value(a,$VEC(N,G,t)))
                addblade!(out,$bop(value(b,t)),bits(basis(b)),Val{N}())
                return Chain{V,G}(out)
            end
            function $op(a::Simplex{V,G,A,S} where A,b::Chain{V,G,T}) where {V,G,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = convert($VEC(N,G,t),$(bcast(bop,:(copy(value(b,$VEC(N,G,t))),))))
                addblade!(out,value(a,t),basis(a),Val{N}())
                return Chain{V,G}(out)
            end
            function $op(a::Chain{V,G,T},b::Simplex{V,L,B,S} where B) where {V,G,T<:$Field,L,S<:$Field}
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                addmulti!(out,$bop(value(b,t)),bits(basis(b)),Val{N}())
                return MultiVector{V}(out)
            end
            function $op(a::Simplex{V,L,A,S} where A,b::Chain{V,G,T}) where {V,G,T<:$Field,L,S<:$Field}
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[r+1:r+bng] = $(bcast(bop,:(value(b,$VEC(N,G,t)),)))
                addmulti!(out,value(a,t),bits(basis(a)),Val{N}())
                return MultiVector{V}(out)
            end
            $op(a::Chain{V,G,T}) where {V,G,T<:$Field} = Chain{V,G,$TF}($(bcast(bop,:(value(a),))))
            function $op(a::$Chain{V,G,T},b::SubManifold{V,G}) where {V,G,T<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = copy(value(a,$VEC(N,G,t)))
                addblade!(out,$bop(value(b,t)),bits(basis(b)),Val{N}())
                return Chain{V,G}(out)
            end
            function $op(a::SubManifold{V,G},b::Chain{V,G,T}) where {V,G,T<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = convert($VEC(N,G,t),$(bcast(bop,:(copy(value(b,$VEC(N,G,t))),))))
                addblade!(out,value(a,t),basis(a),Val{N}())
                return Chain{V,G}(out)
            end
            function $op(a::Chain{V,G,T},b::SubManifold{V,L}) where {V,G,T<:$Field,L}
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                addmulti!(out,$bop(value(b,t)),bits(basis(b)),Val{N}())
                return MultiVector{V}(out)
            end
            function $op(a::SubManifold{V,L},b::Chain{V,G,T}) where {V,G,T<:$Field,L}
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[r+1:r+bng] = $(bcast(bop,:(copy(value(b,$VEC(N,G,t))),)))
                addmulti!(out,value(a,t),bits(basis(a)),Val{N}())
                return MultiVector{V}(out)
            end
            function $op(a::Chain{V,G,T},b::SparseChain{V,G}) where {V,G,T<:$Field}
                $(insert_expr((:N,),VEC)...)
                bt = terms(b)
                t = promote_type(T,valuetype.(bt)...)
                out = copy(value(a,$VEC(N,G,t)))
                addmulti!(out,bt.nzval,bt.nzind)
                return Chain{V,G}(out)
            end
            function $op(a::SparseChain{V,G},b::Chain{V,G,T}) where {V,G,T<:$Field}
                $(insert_expr((:N,),VEC)...)
                at = terms(a)
                t = promote_type(T,valuetype.(at)...)
                out = convert($VEC(N,G,t),$(bcast(bop,:(copy(value(b,$VEC(N,G,t))),))))
                addmulti!(out,at.nzval,at.nzind)
                return Chain{V,G}(out)
            end
            function $op(a::Chain{V,G,T},b::MultiVector{V,S}) where {V,G,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t,:r,:bng),VEC)...)
                out = convert($VEC(N,t),$(bcast(bop,:(copy(value(b,$VEC(N,t))),))))
                @inbounds $(add_val(:(+=),:(out[r+1:r+bng]),:(value(a,$VEC(N,G,t))),ADD))
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::Chain{V,G,S}) where {V,T<:$Field,G,S<:$Field}
                $(insert_expr((:N,:t,:r,:bng),VEC)...)
                out = copy(value(a,$VEC(N,t)))
                @inbounds $(add_val(eop,:(out[r+1:r+bng]),:(value(b,$VEC(N,G,t))),bop))
                return MultiVector{V}(out)
            end
        end
    end
end
