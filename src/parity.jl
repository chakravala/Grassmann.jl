
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

import Leibniz: parityreverse, parityinvolute, parityconj, parityclifford, parityright, parityleft, parityrighthodge, paritylefthodge, odd, even, involute
import Leibniz: complementleft, complementright, ⋆, complementlefthodge, complementrighthodge, complement, grade_basis
import DirectSum: antireverse, antiinvolute, anticlifford
## complement

export complementleft, complementright, ⋆, complementlefthodge, complementrighthodge

## reverse

import Base: reverse, conj, ~
export involute, clifford, antireverse, antiinvolute, anticlifford

## product parities

@pure parityjoin(N,a,b) = isodd(sum(digits_fast(a,N) .* cumsum(digits_fast(b<<1,N))))
@pure function parityjoin(N,S,a,b)
    isodd(sum(digits_fast(a,N) .* cumsum(digits_fast(b<<1,N)))+count_ones((a & b) & S))
end

@pure conformalmask(V) = UInt(2)^(hasinf(V)&&hasorigin(V) ? 2 : 0)-1

@pure function conformalcheck(V,A,B)
    bt = conformalmask(V)
    i2o,o2i = hasinf2origin(V,A,B),hasorigin2inf(V,A,B)
    A&bt, B&bt, i2o, o2i, i2o ⊻ o2i
end

@pure function parityconformal(V,A,B)
    C,cc = A ⊻ B, hasinforigin(V,A,B) || hasorigininf(V,A,B)
    A3,B3,i2o,o2i,xor = conformalcheck(V,A,B)
    pcc,bas = xor⊻i2o⊻(i2o&o2i), xor ? (A3|B3)⊻C : C
    return pcc, bas, cc, Zero(UInt)
end

function paritycomplementinverse(N,G)#,S)
    parityreverse(N-G)⊻parityreverse(G)⊻isodd(binomial(N,2))#⊻isodd(count_ones(S))
end

function cga(V,A,B)
    (hasinforigin(V,A,B) || hasorigininf(V,A,B)) && iszero(getbasis(V,A)∨⋆(getbasis(V,B)))
end

@pure parityregressive(V::Int,a,b,skew=Val(false)) = _parityregressive(V,a,b,skew)
@pure function _parityregressive(V,a,b,::Val{skew}=Val(false)) where skew
    N,M,S = mdims(V),options(V),metric(V)
    D,G = diffvars(V),typeof(V)<:Int ? V : grade(V)
    A,B,Q,Z = symmetricmask(V,a,b)
    α,β = complement(N,A,D),complement(N,B,D)
    cc = skew && (hasinforigin(V,A,β) || hasorigininf(V,A,β))
    if ((count_ones(α&β)==0) && !diffcheck(V,α,β)) || cc
        C,L = α ⊻ β, count_ones(A)+count_ones(B)
        bas = complement(N,C,D)
        pcc,bas = if skew
            A3,β3,i2o,o2i,xor = conformalcheck(V,A,β)
            cx = cc || xor
            cx && parity(V,A3,β3)⊻(i2o || o2i)⊻(xor&!i2o), cx ? (A3|β3)⊻bas : bas
        else
            false, A+B≠0 ? bas : Zero(UInt)
        end
        par = parityright(S,A,N)⊻parityright(S,B,N)⊻parityright(S,C,N)
        return (isodd(L*(L-G))⊻par⊻parity(N,S,α,β)⊻pcc)::Bool, bas|Q, true, Z
    else
        return false, Zero(UInt), false, Z
    end
end

@pure parityregressive(V::Signature,a,b,skew=Val(false)) = _parityregressive(V,a,b,skew)
@pure function parityregressive(V::M,A,B) where M<:Manifold
    p,C,t,Z = parityregressive(Signature(V),A,B)
    return p ? -1 : 1, C, t, Z
end

@pure function parityinterior(V::Int,a,b)
    A,B,Q,Z = symmetricmask(V,a,b)
    (diffcheck(V,A,B) || cga(V,A,B)) && (return false,Zero(UInt),false,Z)
    p,C,t = parityregressive(V,A,complement(V,B,diffvars(V)),Val(true))
    t ? (p⊻parityright(0,sum(indices(B,V)),count_ones(B)) ? -1 : 1) : 1, C|Q, t, Z
end

#=@pure function parityinterior(V::Signature{N,M,S},a,b) where {N,M,S}
    A,B,Q,Z = symmetricmask(V,a,b)
    diffcheck(V,A,B) && (return false,Zero(UInt),false,Z)
    p,C,t = parityregressive(V,A,complement(N,B,diffvars(V)),Val{true}())
    return t ? p⊻parityrighthodge(S,B,N) : p, C|Q, t, Z
end=#

@pure function parityinterior(V::M,a,b) where M<:Manifold
    A,B,Q,Z = symmetricmask(V,a,b); N = rank(V)
    (diffcheck(V,A,B) || cga(V,A,B)) && (return false,Zero(UInt),false,Z)
    p,C,t = parityregressive(Signature(V),A,complement(N,B,diffvars(V)),Val{true}())
    ind = indices(B,N); g = prod(V[ind])
    return t ? (p⊻parityright(0,sum(ind),count_ones(B)) ? -(g) : g) : g, C|Q, t, Z
end

@pure function parityinner(V::Int,a::UInt,b::UInt)
    A,B = symmetricmask(V,a,b)
    parity(V,A,B) ? -1 : 1
end

@pure function parityinner(V::M,a::UInt,b::UInt) where M<:Manifold
    A,B = symmetricmask(V,a,b)
    g = abs(prod(V[indices(A&B,mdims(V))]))
    parity(Signature(V),A,B) ? -(g) : g
end

### parity cache

const parity_cache = Dict{UInt,Vector{Vector{Bool}}}[]
const parity_extra = Dict{UInt,Dict{UInt,Dict{UInt,Bool}}}[]
@pure function parity(n,s,a,b)::Bool
    if n > sparse_limit
        N = n-sparse_limit
        for k ∈ length(parity_extra)+1:N
            push!(parity_extra,Dict{UInt,Dict{UInt,Dict{UInt,Bool}}}())
        end
        @inbounds !haskey(parity_extra[N],s) && push!(parity_extra[N],s=>Dict{UInt,Dict{UInt,Bool}}())
        @inbounds !haskey(parity_extra[N][s],a) && push!(parity_extra[N][s],a=>Dict{UInt,Bool}())
        @inbounds !haskey(parity_extra[N][s][a],b) && push!(parity_extra[N][s][a],b=>parityjoin(n,s,a,b))
        @inbounds parity_extra[N][s][a][b]
    elseif n==0
        parityjoin(n,s,a,b)
    else
        a1 = a+1
        for k ∈ length(parity_cache)+1:n
            push!(parity_cache,Dict{UInt,Vector{Bool}}())
        end
        @inbounds !haskey(parity_cache[n],s) && push!(parity_cache[n],s=>Vector{Bool}[])
        @inbounds for k ∈ length(parity_cache[n][s]):a
            @inbounds push!(parity_cache[n][s],Bool[])
        end
        @inbounds for k ∈ length(parity_cache[n][s][a1]):b
            @inbounds push!(parity_cache[n][s][a1],parityjoin(n,s,a,k))
        end
        @inbounds parity_cache[n][s][a1][b+1]
    end
end
@pure function parity(v::Signature,a::UInt,b::UInt)
    d = diffmask(v)
    D = ~(isdyadic(v) ? |(d...) : d)
    parity(mdims(v),metric(v),(a&D),(b&D))
end
@pure parity(v::Int,a::UInt,b::UInt) = parity(v,metric(v),a,b)
@pure parity(v::T,a::UInt,b::UInt) where T<:Manifold = parity(Signature(v),a,b)
@pure parity(a::Submanifold{V,G,B},b::Submanifold{V,L,C}) where {V,G,B,L,C} = parity(V,UInt(a),UInt(b))

### parity product caches

for par ∈ (:conformal,:regressive,:interior)
    calc = Symbol(:parity,par)
    T = Tuple{Any,UInt,Bool,UInt}
    extra = Symbol(par,:_extra)
    cache = Symbol(par,:_cache)
    @eval begin
        const $cache = Dict{UInt,Vector{Dict{UInt,Vector{Vector{$T}}}}}[]
        const $extra = Dict{UInt,Vector{Dict{UInt,Dict{UInt,Dict{UInt,$T}}}}}[]
        @pure function $par(V,a,b)::$T
            M,s = DirectSum.supermanifold(V),metric(V)
            n,m,S = mdims(M),DirectSum.options(M),metric(M)
            m1 = m+1
            if n > sparse_limit
                N = n-sparse_limit
                for k ∈ length($extra)+1:N
                    push!($extra,Dict{UInt,Vector{Dict{UInt,Dict{UInt,Dict{UInt,$T}}}}}())
                end
                if !haskey($extra[N],S)
                    push!($extra[N],S=>Dict{UInt,Dict{UInt,Dict{UInt,$T}}}[])
                end
                for k ∈ length($extra[N][S])+1:m1
                    @inbounds push!($extra[N][S],Dict{UInt,Dict{UInt,Dict{UInt,$T}}}())
                end
                @inbounds !haskey($extra[N][S][m1],s) && push!($extra[N][S][m1],s=>Dict{UInt,Dict{UInt,$T}}())
                @inbounds !haskey($extra[N][S][m1][s],a) && push!($extra[N][S][m1][s],a=>Dict{UInt,$T}())
                @inbounds !haskey($extra[N][S][m1][s][a],b) && push!($extra[N][S][m1][s][a],b=>$calc(V,a,b))
                @inbounds $extra[N][S][m1][s][a][b]
            elseif n==0
                $calc(V,a,b)
            else
                a1 = a+1
                for k ∈ length($cache)+1:n
                    push!($cache,Dict{UInt,Dict{UInt,Vector{Vector{$T}}}}())
                end
                if !haskey($cache[n],S)
                    push!($cache[n],S=>Dict{UInt,Vector{Vector{$T}}}[])
                end
                @inbounds for k ∈ length($cache[n][S])+1:m1
                    @inbounds push!($cache[n][S],Dict{UInt,Vector{Vector{$T}}}())
                end
                @inbounds !haskey($cache[n][S][m1],s) && push!($cache[n][S][m1],s=>Vector{$T}[])
                @inbounds for k ∈ length($cache[n][S][m1][s]):a
                    @inbounds push!($cache[n][S][m1][s],$T[])
                end
                @inbounds for k ∈ length($cache[n][S][m1][s][a1]):b
                    @inbounds push!($cache[n][S][m1][s][a1],$calc(V,a,k))
                end
                @inbounds $cache[n][S][m1][s][a1][b+1]
            end
        end
        @pure $par(a::Submanifold{V,G,B},b::Submanifold{V,L,C}) where {V,G,B,L,C} = $par(V,UInt(a),UInt(b))
    end
end

import Base: signbit, imag, real
export odd, even, angular, radial, ₊, ₋, ǂ

@pure signbit(V::T) where T<:Manifold = (ib=indexbasis(rank(V)); parity.(Ref(V),ib,ib))
@pure signbit(V::T,G) where T<:Manifold = (ib=indexbasis(rank(V),G); parity.(Ref(V),ib,ib))
@pure angular(V::T) where T<:Manifold = Values(findall(signbit(V))...)
@pure radial(V::T) where T<:Manifold = Values(findall(.!signbit(V))...)
@pure angular(V::T,G) where T<:Manifold = findall(signbit(V,G))
@pure radial(V::T,G) where T<:Manifold = findall(.!signbit(V,G))

for (op,other) ∈ ((:angular,:radial),(:radial,:angular))
    @eval begin
        $op(t::T) where T<:TensorTerm{V,G} where {V,G} = basisindex(mdims(V),UInt(basis(t))) ∈ $op(V,G) ? t : Zero(V)
        function $op(t::Chain{V,G,T}) where {V,G,T}
            out = copy(value(t,mvec(mdims(V),G,T)))
            for k ∈ $other(V,G)
                @inbounds out[k]≠0 && (out[k] = zero(T))
            end
            Chain{V,G}(out)
        end
    end
end

function odd(t::Multivector{V,T}) where {V,T}
    N = mdims(V)
    out = copy(value(t,mvec(N,T)))
    bs = binomsum_set(N)
    @inbounds out[1]≠0 && (out[1] = zero(T))
    for g ∈ 3:2:N+1
        @inbounds for k ∈ bs[g]+1:bs[g+1]
            @inbounds out[k]≠0 && (out[k] = zero(T))
        end
    end
    Multivector{V}(out)
end
odd(t::Spinor{V}) where V = Zero{V}()
even(t::Spinor) = t
function even(t::Multivector{V,T}) where {V,T}
    N = mdims(V)
    out = zeros(mvec(N-1,T))
    bs = binomsum_set(N)
    i = 1
    for g ∈ 1:2:N+1
        @inbounds for k ∈ bs[g]+1:bs[g+1]
            @inbounds out[i] = t.v[k]
            i += 1
        end
    end
    Spinor{V}(out)
end
#=function even(t::Multivector{V,T}) where {V,T}
    N = mdims(V)
    out = copy(value(t,mvec(N,T)))
    bs = binomsum_set(N)
    for g ∈ 2:2:N+1
        @inbounds for k ∈ bs[g]+1:bs[g+1]
            @inbounds out[k]≠0 && (out[k] = zero(T))
        end
    end
    Multivector{V}(out)
end=#

function imag(t::Multivector{V,T}) where {V,T}
    N = mdims(V)
    out = copy(value(t,mvec(N,T)))
    bs = binomsum_set(N)
    @inbounds out[1]≠0 && (out[1] = zero(T))
    for g ∈ 2:N+1
        @inbounds !parityreverse(g-1) && for k ∈ bs[g]+1:bs[g+1]
            @inbounds out[k]≠0 && (out[k] = zero(T))
        end
    end
    Multivector{V}(out)
end
function real(t::Multivector{V,T}) where {V,T}
    N = mdims(V)
    out = copy(value(t,mvec(N,T)))
    bs = binomsum_set(N)
    for g ∈ 3:N+1
        @inbounds parityreverse(g-1) && for k ∈ bs[g]+1:bs[g+1]
            @inbounds out[k]≠0 && (out[k] = zero(T))
        end
    end
    Multivector{V}(out)
end
function imag(t::Spinor{V,T}) where {V,T}
    N = mdims(V)
    out = copy(value(t,mvecs(N,T)))
    bs = spinsum_set(N)
    @inbounds out[1]≠0 && (out[1] = zero(T))
    for g ∈ 2:N+1
        @inbounds !parityreverse(g-1) && for k ∈ bs[g]+1:bs[g+1]
            @inbounds out[k]≠0 && (out[k] = zero(T))
        end
    end
    Spinor{V}(out)
end
function real(t::Spinor{V,T}) where {V,T}
    N = mdims(V)
    out = copy(value(t,mvecs(N,T)))
    bs = spinsum_set(N)
    for g ∈ 3:N+1
        @inbounds parityreverse(g-1) && for k ∈ bs[g]+1:bs[g+1]
            @inbounds out[k]≠0 && (out[k] = zero(T))
        end
    end
    Spinor{V}(out)
end
