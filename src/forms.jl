#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

## conversions

@pure choicevec(M,G,T) = T ∈ (Any,BigFloat,BigInt,Complex{BigFloat},Rational{BigInt},Complex{BigInt}) ? svec(M,G,T) : mvec(M,G,T)
@pure choicevec(M,T) = T ∈ (Any,BigFloat,BigInt,Complex{BigFloat},Rational{BigInt},Complex{BigInt}) ? svec(M,T) : mvec(M,T)

#@pure supblade(N,S,B) = bladeindex(N,expandbits(N,S,B))
#@pure supmulti(N,S,B) = basisindex(N,expandbits(N,S,B))

(W::Submanifold)(V::ChainBundle) = W.(value(V))
(M::ChainBundle)(b::Int...) = Submanifold{M}(b)
(M::ChainBundle)(b::Tuple) = Submanifold{M}(b)
(M::ChainBundle)(b::UnitRange) = Submanifold{M}(b)
(M::ChainBundle)(b::T) where T<:AbstractVector = Submanifold{M}(b)

(W::Signature)(b::Chain{V,G,T}) where {V,G,T} = Submanifold(W)(b)
@generated function (w::Submanifold{Q,M})(b::Chain{V,G,T}) where {Q,M,V,G,T}
    W = Manifold(w)
    if isbasis(W)
        if Q == V
            if isdyadic(V)
                if G == M == 1
                    x = UInt(W)
                    X = isdyadic(V) ? x>>Int(mdims(V)/2) : x
                    Y = 0≠X ? X : x
                    out = :(@inbounds b.v[bladeindex($(mdims(V)),Y)])
                    return :(Single{V}(V[intlog(Y)+1] ? -($out) : $out,Submanifold{V}()))
                elseif G == 1 && M == 2
                    ib,(m1,m2) = indexbasis(N,1),DirectSum.eval_shift(W)
                    :(@inbounds $(V[m2] ? :(-(b.v[m2])) : :(b.v[m2]))*getbasis(V,ib[m1]))
                else
                    :(throw(error("not yet possible")))
                end
            else
                return :(contraction(w,b))
            end
        else
            return :(interform(w,b))
        end
    elseif V==W
        return V===W ? :b : :(Chain{w,G,T}(value(b)))
    elseif W⊆V
        if G == 1
            ind = Values{mdims(W),Int}(indices(UInt(W),mdims(V)))
            :(@inbounds Chain{w,1,T}(b.v[$ind]))
        else quote
            out,N = zeros(choicevec(M,G,valuetype(b))),mdims(V)
            ib,S = indexbasis(N,G),UInt(w)
            for k ∈ 1:length(ib)
                @inbounds if b[k] ≠ 0
                    @inbounds if count_ones(ib[k]&S) == G
                        @inbounds setblade!(out,b[k],lowerbits(M,S,ib[k]),Val{M}())
                    end
                end
            end
            return Chain{w,G}(out)
        end end
    elseif V⊆W
        quote
            WC,VC,N = isdyadic(w),isdyadic(V),mdims(V)
            #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
            #    return V0
            out = zeros(choicevec(M,G,valuetype(b)))
            ib = indexbasis(N,G)
            for k ∈ 1:length(ib)
                @inbounds if b[k] ≠ 0
                    @inbounds B = typeof(V)<:Submanifold ? expandbits(M,UInt(V),ib[k]) : ib[k]
                    if WC && (!VC)
                        @inbounds setblade!(out,b[k],mixed(V,B),Val{M}())
                    elseif (!WC) && (!VC)
                        @inbounds setblade!(out,b[k],B,Val{M}())
                    else
                        throw(error("arbitrary Manifold intersection not yet implemented."))
                    end
                end
            end
            return Chain{w,G}(out)
        end
    elseif V == V''
        return :(w(Chain{$(V''),G}(value(b))))
    else
        :(throw(error("cannot convert from $V to $w")))
    end
end

(W::Signature)(b::Multivector{V,T}) where {V,T} = Submanifold(W)(b)
function (W::Submanifold{Q,M,S})(m::Multivector{V,T}) where {Q,M,V,S,T}
    if isbasis(W)
        isdyadic(V) && throw(error("Multivector forms not yet supported"))
        return V==W ? contraction(W,m) : interform(W,m)
    elseif V==W
        return V===W ? m : Multivector{W,T}(value(m))
    elseif W⊆V
        out,N = zeros(choicevec(M,valuetype(m))),mdims(V)
        bs = binomsum_set(N)
        for i ∈ 1:N+1
            ib = indexbasis(N,i-1)
            for k ∈ 1:length(ib)
                @inbounds s = k+bs[i]
                @inbounds if m.v[s] ≠ 0
                    @inbounds if count_ones(ib[k]&S) == i-1
                        @inbounds setmulti!(out,m.v[s],lowerbits(N,S,ib[k]),Val{M}())
                    end
                end
            end
        end
        return Multivector{W}(out)
    elseif V⊆W
        WC,VC,N = isdyadic(W),isdyadic(V),mdims(V)
        #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
        #    return V0
        out = zeros(choicevec(M,valuetype(m)))
        bs = binomsum_set(N)
        for i ∈ 1:N+1
            ib = indexbasis(N,i-1)
            for k ∈ 1:length(ib)
                @inbounds s = k+bs[i]
                @inbounds if m.v[s] ≠ 0
                    @inbounds B = typeof(V)<:Submanifold ? expandbits(M,UInt(V),ib[k]) : ib[k]
                    if WC && (!VC)
                        @inbounds setmulti!(out,m.v[s],mixed(V,B),Val{M}())
                    elseif (!WC) && (!VC)
                        @inbounds setmulti!(out,m.v[s],B,Val{M}())
                    else
                        throw(error("arbitrary Manifold intersection not yet implemented."))
                    end
                end
            end
        end
        return Multivector{W}(out)
    elseif V == V''
        return W((V'')(m))
    else
        throw(error("cannot convert from $(V) to $(W)"))
    end
end

#### need separate storage for m and F for caching

const dualform_cache = Vector{Tuple{Int,Bool}}[]
const dualformC_cache = Vector{Tuple{Int,Bool}}[]
@pure function dualform(V::Manifold)
    N,C = rank(V),isdyadic(V)
    for n ∈ 2length(C ? dualformC_cache : dualform_cache)+2:2:N
        push!(C ? dualformC_cache : dualform_cache,Tuple{Int,Any}[])
    end
    M = Int(N/2)
    @inbounds if isempty((C ? dualformC_cache : dualindex_cache)[M])
        ib = indexbasis(N,1)
        mV = Array{Tuple{Int,Bool},1}(undef,M)
        for Q ∈ 1:M
            @inbounds x = ib[Q]
            X = C ? x<<M : x
            Y = X>(1<<N) ? x : X
            @inbounds mV[Q] = (intlog(Y)+1,V[intlog(x)+1])
        end
        @inbounds (C ? dualformC_cache : dualform_cache)[M] = mV
    end
    @inbounds (C ? dualformC_cache : dualform_cache)[M]
end

const dualindex_cache = Vector{Vector{Int}}[]
const dualindexC_cache = Vector{Vector{Int}}[]
@pure function dualindex(V::Manifold)
    N,C = rank(V),isdyadic(V)
    for n ∈ 2length(C ? dualindexC_cache : dualindex_cache)+2:2:N
        push!(C ? dualindexC_cache : dualindex_cache,Vector{Int}[])
    end
    M = Int(N/2)
    @inbounds if isempty((C ? dualindexC_cache : dualindex_cache)[M])
        #df = dualform(C ? Manifold(M)⊕Manifold(M)' : Manifold(n))
        di = Array{Vector{Int},1}(undef,M)
        x = M .+cumsum(collect(2(M-1):-1:M-1))
        @inbounds di[1] = [M;x;collect(x[end]+1:x[end]+M-1)]
        for Q ∈ 2:M
            @inbounds di[Q] = di[Q-1] .+ (Q-1)
            @inbounds di[Q][end-M+Q] = M+Q
            #@inbounds m = df[Q][1]
            #@inbounds di[Q] = [bladeindex(n,bit2int(indexbits(n,[i,m]))) for i ∈ 1:n]
        end
        @inbounds di[1][end-M+1] = M+1
        @inbounds (C ? dualindexC_cache : dualindex_cache)[M] = di
    end
    @inbounds (C ? dualindexC_cache : dualindex_cache)[M]
end

## Chain forms

(a::Chain)(b::T) where {T<:TensorAlgebra} = interform(a,b)
(a::Chain{V,1,<:Manifold} where V)(b::T) where {T<:TensorAlgebra} = contraction(a,b)
@eval begin
    function (a::Chain{V,2})(b::Chain{V,1}) where V
        (!isdyadic(V)) && (return contraction(a,b))
        $(insert_expr((:N,:t,:df,:di))...)
        out = zero(mvec(N,1,t))
        for Q ∈ 1:Int(N/2)
            @inbounds m,val = df[Q][1],df[Q][2]*b.v[Q]
            val≠0 && for i ∈ 1:N
                @inbounds i≠m && addblade!(out,a.v[di[Q][i]]*val,UInt(1)<<(i-1),Val{N}())
            end
        end
        return Chain{V,1}(out)
    end
    function Chain{V,T}(b::AbstractMatrix{T}) where {V,T}
        (!isdyadic(V)) && throw(error("$V does not support this conversion"))
        $(insert_expr((:N,:M))...)
        size(b) ≠ (M,M) && throw(error("dimension mismatch"))
        out = zeros(mvec(N,2,T))
        for i ∈ 1:M
            x = UInt(1)<<(i-1)
            for j ∈ 1:M
                @inbounds b[j,i]≠0 && setblade!(out,b[j,i],x⊻(UInt(1)<<(M+j-1)),Val{N}())
            end
        end
        return Chain{V,2}(out)
    end
end

# more forms

function (a::Chain{V,1})(b::Submanifold{V,1}) where V
    (!isdyadic(V)) && (return contraction(a,b))
    x = UInt(b)
    X = isdyadic(V) ? x<<Int(mdims(V)/2) : x
    Y = X>2^mdims(V) ? x : X
    @inbounds out = a.v[bladeindex(mdims(V),Y)]
    Single{V}((V[intlog(x)+1]*out),Submanifold{V}())
end
@eval begin
    function (a::Chain{V,2,T})(b::Submanifold{V,1}) where {V,T}
        (!isdyadic(V)) && (return contraction(a,b))
        $(insert_expr((:N,:df,:di))...)
        Q = bladeindex(N,UInt(b))
        @inbounds m,val = df[Q][1],df[Q][2]*value(b)
        out = zero(mvec(N,1,T))
        for i ∈ 1:N
            i≠m && @inbounds setblade!(out,a.v[di[Q][i]]*val,UInt(1)<<(i-1),Val{N}())
        end
        return Chain{V,1}(out)
    end
    function (a::Chain{V,1})(b::Single{V,1}) where V
        (!isdyadic(V)) && (return contraction(a,b))
        $(insert_expr((:t,))...)
        x = bits(b)
        X = isdyadic(V) ? x<<Int(mdims(V)/2) : x
        Y = X>2^mdims(V) ? x : X
        @inbounds out = a.v[bladeindex(mdims(V),Y)]
        Single{V}((V[intlog(x)+1]*out*b.v)::t,Submanifold{V}())
    end
    function (a::Single{V,1})(b::Chain{V,1}) where V
        (!isdyadic(V)) && (return contraction(a,b))
        $(insert_expr((:t,))...)
        x = UInt(a)
        X = isdyadic(V) ? x>>Int(mdims(V)/2) : x
        Y = 0≠X ? X : x
        @inbounds out = b.v[bladeindex(mdims(V),Y)]
        Single{V}((a.v*V[intlog(Y)+1]*out)::t,Submanifold{V}())
    end
    function (a::Single{V,2})(b::Chain{V,1}) where V
        (!isdyadic(V)) && (return contraction(a,b))
        $(insert_expr((:N,:t))...)
        ib,(m1,m2) = indexbasis(N,1),DirectSum.eval_shift(a)
        @inbounds ((V[m2]*a.v*b.v[m2])::t)*getbasis(V,ib[m1])
    end
    function (a::Chain{V,2})(b::Single{V,1}) where V
        (!isdyadic(V)) && (return contraction(a,b))
        $(insert_expr((:N,:t,:df,:di))...)
        Q = bladeindex(N,UInt(b))
        out = zero(mvec(N,1,T))
        @inbounds m,val = df[Q][1],df[Q][2]*b.v
        for i ∈ 1:N
            i≠m && @inbounds setblade!(out,a.v[di[Q][i]]*val,UInt(1)<<(i-1),Val{N}())
        end
        return Chain{V,1}(out)
    end
    function (a::Chain{V,1})(b::Chain{V,1}) where V
        (!isdyadic(V)) && (return contraction(a,b))
        $(insert_expr((:N,:M,:t,:df))...)
        out = zero(t)
        for Q ∈ 1:M
            @inbounds out += a.v[df[Q][1]]*(df[Q][2]*b.v[Q])
        end
        return Single{V}(out::t,Submanifold{V}())
    end
end
