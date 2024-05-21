
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
import DirectSum: antireverse, antiinvolute, anticlifford, paritymetric, parityanti
import DirectSum: complementleftanti, complementrightanti

## complement

export complementleft, complementright, ⋆, complementlefthodge, complementrighthodge
export complementleftanti, complementrightanti

## reverse

import Base: reverse, conj, ~
export involute, clifford, pseudoreverse, antireverse

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
    if ((count_ones(α&β)==0) && !diffcheck(V,α,β)) #|| cc
        C,L = α ⊻ β, count_ones(A)+count_ones(B)
        bas = complement(N,C,D)
        pcc,bas = if skew
            A3,β3,i2o,o2i,xor = UInt(0),UInt(0),false,false,false#conformalcheck(V,A,β)
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
@pure function parityregressive(::M,A,B) where M<:Manifold{V} where V
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

@pure function diffcheck2(V,A::UInt,B::UInt)
    d,db = diffvars(V),diffmask(V)
    v = isdyadic(V) ? db[1]|db[2] : db
    d≠0 && count_ones(A&v)+count_ones(B&v)>diffmode(V)
end

@pure function parityinterior(V::M,a,b,::Val{lim}=Val(false)) where {W,M<:Manifold{W},lim}
    A,B,Q,Z = symmetricmask(V,a,b); N = rank(V)
    diffcheck(V,A,B) && (return lim ? (Values{0,Tuple{UInt,Int}}(),Z) : (1,Zero(UInt),false,Z))
    gout,tout,G,diag = 0,false,count_ones(B),isdiag(V)
    bas = diag ? Values((B,)) : indexbasis(grade(V),G)
    g = if diag
        gg = V[indices(B,N)]
        Values((prod(isempty(gg) ? 1 : signbool.(gg)),))
    else
        value(metrictensor(V,G))[bladeindex(grade(V),B)]
    end
    bs,bg = (),()
    for i ∈ list(1,diag ? 1 : gdims(grade(V),G))
        if (!isnull(g[i])) && !(diffcheck2(V,A,bas[i]) || cga(V,A,bas[i]))
            p,C,t = parityregressive(Signature(W),A,complement(N,bas[i],diffvars(V)),Val{true}())
            CQ,tout = C|Q,tout|t
            if t
                ggg = (p⊻parityright(0,sum(indices(bas[i],N)),G)) ? -(g[i]) : g[i]
                if CQ ∉ bs
                    bs = (bs...,CQ)
                    bg = (bg...,(CQ,ggg))
                else
                    j = findfirst(x->x==CQ,bs)
                    bg = @inbounds (bg[1:j-1]...,(CQ,(bg[j][2]+ggg)),bg[j+1:length(bs)]...)
                end
                gout += ggg
            end
        end
    end
    if lim
        return (isempty(bg) ? Values{0,Tuple{UInt,Int}}() : Values(bg)), Z
    else
        length(bs) > 1 && throw("this is a limited variant of interior product")
        return @inbounds gout, isempty(bs) ? UInt(0) : bs[1], tout, Z
    end
end

@pure function parityinner(V::Int,a::UInt,b::UInt)
    A,B = symmetricmask(V,a,b)
    parity(V,A,B) ? -1 : 1
end

@pure function parityinner(V::M,a::UInt,b::UInt) where {W,M<:Manifold{W}}
    A,B = symmetricmask(V,a,b)
    C = A&B; G = count_ones(C)
    g = if isdiag(V) || hasconformal(V)
        gg = V[indices(C,mdims(V))]
        abs(prod(isempty(gg) ? 1 : signbool.(gg)))
    else
        value(metrictensor(V,G))[bladeindex(mdims(V),C)]
    end
    parity(Signature(W),A,B) ? -(g) : g
end

function parityseq(V,B)
    l,out = length(B),false
    (isdiag(V) || isone(l)) && (return 1)
    for i ∈ list(2,length(B))
        out = out⊻parity(grade(V),B[i-1],B[i])
    end
    return out ? -1 : 1
end

maxempty(x) = isempty(x) ? 0 : maximum(x)

function paritygeometric(V,A::UInt,B::UInt)
    #if iszero(B)
    #    Values(((A,1),))
    a,b = splitbasis(V,A),splitbasis(V,B)
    ga,gb = maxempty(count_ones.(a)),maxempty(count_ones.(b))
    if  (ga≤1 && gb≤1) ? count_ones(A) ≥ count_ones(B) : ga ≥ gb
        paritygeometric(V,A,b)
    else
        paritygeometric(V,a,B)
    end
end
function paritygeometric(V,A::UInt,b::Tuple)
    i = length(b)
    vals = if iszero(i)
        Values((((UInt(0),1),(A,1)),))
    else
        p = parityseq(V,b)
        out = paritygeometricright(V,((UInt(0),p),(A,1)),b[1])
        isone(i) ? out : paritygeometricright(V,out,b,2)
    end
    combinebasis(V,vals)
end
function paritygeometric(V,a::Tuple,B::UInt)
    i = length(a)
    vals = if iszero(i)
        Values((((UInt(0),1),(B,1)),))
    else
        p = parityseq(V,a)
        out = paritygeometricleft(V,a[i],((UInt(0),p),(B,1)))
        isone(i) ? out : paritygeometricleft(V,a,out,i-1)
    end
    combinebasis(V,vals)
end
function paritygeometricright(V,a,b::Tuple,i)
    out = vcat(paritygeometricright.(Ref(V),a,Ref(b[i]))...)
    i==length(b) ? out : paritygeometricright(V,out,b,i+1)
end
function paritygeometricright(V,aeai,B::UInt)
    ae,ai = @inbounds (aeai[1],aeai[2])
    Ae,Aeg = @inbounds (ae[1],ae[2])
    Ai,Aig = @inbounds (ai[1],ai[2])
    G = count_ones(B)
    CCg = if isdiag(V) || hasconformal(V)
        g,C,t,Z = interior(V,Ai,B)
        Cg = parityclifford(G)⊻isodd(G*count_ones(Ai)) ? -Aig*g : Aig*g
        t ? ((ae,(C,Cg)),) : ()
    else
        Cg,Z = parityinterior(V,Ai,B,Val(true))
        g = parityclifford(G)⊻isodd(G*count_ones(Ai)) ? -Aig*Aeg : Aig*Aeg
        ([((Ae,g),cg) for cg ∈ Cg]...,)
    end
    out = if iszero(Ai&B)
        p = parity(grade(V),Ai⊻Ae,B) ? -Aeg : Aeg
        (combinegeometric(V,(Ae⊻B,p),ai),CCg...)
    else
        CCg
    end
    isempty(out) ? Values{0,Tuple{Tuple{UInt,Int},Tuple{UInt,Int}}}() : Values(out)
end
function paritygeometricleft(V,a::Tuple,b,i)
    out = vcat(paritygeometricleft.(Ref(V),Ref(a[i]),b)...)
    isone(i) ? out : paritygeometricleft(V,a,out,i-1)
end
function paritygeometricleft(V,A::UInt,bebi)
    be,bi = @inbounds (bebi[1],bebi[2])
    Be,Beg = @inbounds (be[1],be[2])
    Bi,Big = @inbounds (bi[1],bi[2])
    G = count_ones(A)
    CCg = if isdiag(V) || hasconformal(V)
        g,C,t,Z = interior(V,Bi,A)
        Cg = parityreverse(G) ? -Big*g : Big*g
        t ? ((be,(C,Cg)),) : ()
    else
        Cg,Z = parityinterior(V,Bi,A,Val(true))
        g = parityreverse(G) ? -Big*Beg : Big*Beg
        ([((Be,g),cg) for cg ∈ Cg]...,)
    end
    out = if iszero(A&Bi)
        p = parity(grade(V),A,Be⊻Bi) ? -Beg : Beg
        (combinegeometric(V,(A⊻Be,p),bi),CCg...)
    else
        CCg
    end
    isempty(out) ? Values{0,Tuple{Tuple{UInt,Int},Tuple{UInt,Int}}}() : Values(out)
end

function splitbasis(V,ind::AbstractVector)
    g = metrictensor(V)
    f = [findall(.!iszero.(value(value(g)[i]))) for i ∈ 1:mdims(V)]
    for i ∈ list(1,length(f))
        i ∉ f[i] && push!(f[i],i)
    end
    j = 2
    while j ≤ length(f)
        t = false
        for k ∈ list(1,j-1)
            for q ∈ f[j]
                if q ∈ f[k]
                    t = true
                    for p ∈ f[j]
                        (p ∉ f[k]) && push!(f[k],p)
                    end
                    popat!(f,j)
                    break
                end
            end
            t && (break)
        end
        !t && (j += 1)
    end
    return getindex.(Ref(ind),f)
end

function splitbasis(V,B)
    iszero(B) && (return ())
    ind = indices(B,mdims(V))
    bs = ((UInt(1).<<(ind.-1))...,)
    if isdiag(V)
        return bs
    else
        f = splitbasis(V(ind...),ind)
        ([|((UInt(1).<<(j.-1))...) for j ∈ f]...,)
    end
end

function combinebasis(V,vals)
    result = ()
    for val ∈ vals
        cg = combinegeometric(V,val)
        !isnothing(cg) && (result = (result...,cg))
    end
    isempty(result) ? Values{0,Tuple{UInt,Int}}() : Values(result)
end
function combinegeometric(V,ei)
    e,i = @inbounds ei[1],ei[2]
    E,Eg = @inbounds e[1],e[2]
    I,Ig = @inbounds i[1],i[2]
    iszero(E&I) ? (E⊻I,Eg*Ig) : nothing
end
function combinegeometric(V,e,i)
    E,Eg = @inbounds e[1],e[2]
    I,Ig = @inbounds i[1],i[2]
    #p = parity(grade(V),E,I) ? -Eg*Ig : Eg*Ig
    return ((E,Eg*Ig),(I,1))
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

Base.iseven(t::Zero) = true
Base.isodd(t::Zero) = true
Base.iseven(t::TensorGraded{V,G}) where {V,G} = iseven(G) ? true : iszero(t)
Base.isodd(t::TensorGraded{V,G}) where {V,G} = isodd(G) ? true : iszero(t)
Base.iseven(t::Spinor) = true
Base.iseven(t::AntiSpinor) = iszero(t)
Base.isodd(t::Spinor) = iszero(t)
Base.isodd(t::AntiSpinor) = true
Base.iseven(t::Couple{V,B}) where {V,B} = iseven(grade(B)) ? true : iszero(imaginary(t))
Base.isodd(t::Couple{V,B}) where {V,B} = isodd(grade(B)) ? iszero(scalar(t)) : iszero(t)
Base.iseven(t::PseudoCouple{V,B}) where {V,B} = iseven(imaginary(t)) && iseven(volume(t))
Base.isodd(t::PseudoCouple{V,B}) where {V,B} = isodd(imaginary(t)) && isodd(volume(t))
Base.iseven(t::Multivector) = norm(t) ≈ norm(even(t))
Base.isodd(t::Multivector) = norm(t) ≈ norm(odd(t))

even(t::AntiSpinor{V}) where V = Zero{V}()
odd(t::Spinor{V}) where V = Zero{V}()
even(t::Spinor) = t
odd(t::AntiSpinor) = t
even(t::Couple{V,B}) where {V,B} = iseven(grade(B)) ? t : scalar(t)
odd(t::Couple{V,B}) where {V,B} = isodd(grade(B)) ? imaginary(t) : Zero{V}()
even(t::PseudoCouple{V,B}) where {V,B} = even(imaginary(t)) + even(volume(t))
odd(t::PseudoCouple{V,B}) where {V,B} = odd(imaginary(t)) + odd(volume(t))

function imag(t::Multivector{V,T}) where {V,T}
    N = mdims(V)
    out = copy(value(t,mvec(N,T)))
    bs = binomsum_set(N)
    @inbounds out[1]≠0 && (out[1] = zero(T))
    for g ∈ list(2,N+1)
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
    for g ∈ list(3,N+1)
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
    for g ∈ evens(2,N+1)
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
    for g ∈ evens(3,N+1)
        @inbounds parityreverse(g-1) && for k ∈ bs[g]+1:bs[g+1]
            @inbounds out[k]≠0 && (out[k] = zero(T))
        end
    end
    Spinor{V}(out)
end
