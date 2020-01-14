#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

## conversions

@pure choicevec(M,G,T) = T ∈ (Any,BigFloat,BigInt,Complex{BigFloat},Rational{BigInt},Complex{BigInt}) ? svec(M,G,T) : mvec(M,G,T)
@pure choicevec(M,T) = T ∈ (Any,BigFloat,BigInt,Complex{BigFloat},Rational{BigInt},Complex{BigInt}) ? svec(M,T) : mvec(M,T)

#@pure supblade(N,S,B) = bladeindex(N,expandbits(N,S,B))
#@pure supmulti(N,S,B) = basisindex(N,expandbits(N,S,B))

function (W::Signature)(b::Chain{V,G,T}) where {V,G,T}
    V==W && (return b)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    N,M = ndims(V),ndims(W)
    out = zeros(choicevec(M,G,valuetype(b)))
    ib = indexbasis(N,G)
    for k ∈ 1:length(ib)
        @inbounds if b[k] ≠ 0
            @inbounds B = typeof(V)<:SubManifold ? expandbits(M,bits(V),ib[k]) : ib[k]
            if WC<0 && VC≥0
                @inbounds setblade!(out,b[k],mixed(V,B),Val{M}())
            elseif WC≥0 && VC≥0
                @inbounds setblade!(out,b[k],B,Val{M}())
            else
                throw(error("arbitrary Manifold intersection not yet implemented."))
            end
        end
    end
    return Chain{W,G,T}(out)
end
function (W::SubManifold{V,M,S})(b::Chain{V,1,T}) where {M,V,S,T}
    Chain{W,1,T}(b.v[indices(bits(W),ndims(V))])
end
function (W::SubManifold{V,M,S})(b::Chain{V,G,T}) where {M,V,S,T,G}
    out,N = zeros(choicevec(M,G,valuetype(b))),ndims(V)
    ib = indexbasis(N,G)
    for k ∈ 1:length(ib)
        @inbounds if b[k] ≠ 0
            @inbounds if count_ones(ib[k]&S) == G
                @inbounds setblade!(out,b[k],lowerbits(M,S,ib[k]),Val{M}())
            end
        end
    end
    return Chain{W,G,T}(out)
end

function (W::Signature)(m::MultiVector{V,T}) where {V,T}
    V==W && (return m)
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    N,M = ndims(V),ndims(W)
    out = zeros(choicevec(M,valuetype(m)))
    bs = binomsum_set(N)
    for i ∈ 1:N+1
        ib = indexbasis(N,i-1)
        for k ∈ 1:length(ib)
            @inbounds s = k+bs[i]
            @inbounds if m.v[s] ≠ 0
                @inbounds B = typeof(V)<:SubManifold ? expandbits(M,bits(V),ib[k]) : ib[k]
                if WC<0 && VC≥0
                    @inbounds setmulti!(out,m.v[s],mixed(V,B),Val{M}())
                elseif WC≥0 && VC≥0
                    @inbounds setmulti!(out,m.v[s],B,Val{M}())
                else
                    throw(error("arbitrary Manifold intersection not yet implemented."))
                end
            end
        end
    end
    return MultiVector{W,T}(out)
end

function (W::SubManifold{V,M,S})(m::MultiVector{V,T}) where {M,V,S,T}
    out,N = zeros(choicevec(M,valuetype(m))),ndims(V)
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
    return MultiVector{W,T}(out)
end

#### need separate storage for m and F for caching

const dualform_cache = Vector{Tuple{Int,Bool}}[]
const dualformC_cache = Vector{Tuple{Int,Bool}}[]
@pure function dualform(V::Signature{N}) where {N}
    C = mixedmode(V)<0
    for n ∈ 2length(C ? dualformC_cache : dualform_cache)+2:2:N
        push!(C ? dualformC_cache : dualform_cache,Tuple{Int,Bool}[])
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
@pure function dualindex(V::Manifold{N}) where N
    C = mixedmode(V)<0
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
function (a::SubManifold{V,1,A})(b::Chain{V,1,T}) where {V,A,T}
    x = bits(a)
    X = mixedmode(V)<0 ? x>>Int(ndims(V)/2) : x
    Y = 0≠X ? X : x
    @inbounds out = b.v[bladeindex(ndims(V),Y)]
    Simplex{V}((V[intlog(Y)+1] ? -(out) : out),SubManifold{V}())
end
function (a::Chain{V,1,T})(b::SubManifold{V,1,B}) where {T,V,B}
    x = bits(b)
    X = mixedmode(V)<0 ? x<<Int(ndims(V)/2) : x
    Y = X>2^ndims(V) ? x : X
    @inbounds out = a.v[bladeindex(ndims(V),Y)]
    Simplex{V}((V[intlog(x)+1] ? -(out) : out),SubManifold{V}())
end
@eval begin
    function (a::SubManifold{V,2,A})(b::Chain{V,1,T}) where {V,A,T}
        C = mixedmode(V)
        (C ≥ 0) && throw(error("wrong basis"))
        $(insert_expr((:N,:M))...)
        bi = indices(basis(a),N)
        ib = indexbasis(N,1)
        @inbounds m = bi[2]>M ? bi[2]-M : bi[2]
        @inbounds ((V[m] ? -(b.v[m]) : b.v[m])*getbasis(V,ib[bi[1]]))
    end
    function (a::Chain{V,2,T})(b::SubManifold{V,1,B}) where {T,V,B}
        C = mixedmode(V)
        (C ≥ 0) && throw(error("wrong basis"))
        $(insert_expr((:N,:df,:di))...)
        Q = bladeindex(N,bits(b))
        @inbounds m,val = df[Q][1],df[Q][2] ? -(value(b)) : value(b)
        out = zero(mvec(N,1,T))
        for i ∈ 1:N
            i≠m && @inbounds setblade!(out,a.v[di[Q][i]]*val,one(Bits)<<(i-1),Val{N}())
        end
        return Chain{V,1,T}(out)
    end
end
@eval begin
    function (a::Chain{V,1,T})(b::Simplex{V,1,X,S} where X) where {V,A,T,S}
        $(insert_expr((:t,))...)
        x = bits(basis(b))
        X = mixedmode(V)<0 ? x<<Int(ndims(V)/2) : x
        Y = X>2^ndims(V) ? x : X
        @inbounds out = a.v[bladeindex(ndims(V),Y)]
        Simplex{V}(((V[intlog(x)+1] ? -(out) : out)*b.v)::t,SubManifold{V}())
    end
    function (a::Simplex{V,1,X,T} where X)(b::Chain{V,1,S}) where {V,T,S}
        $(insert_expr((:t,))...)
        x = bits(basis(a))
        X = mixedmode(V)<0 ? x>>Int(ndims(V)/2) : x
        Y = 0≠X ? X : x
        @inbounds out = b.v[bladeindex(ndims(V),Y)]
        Simplex{V}((a.v*(V[intlog(Y)+1] ? -(out) : out))::t,SubManifold{V}())
    end
    function (a::Simplex{V,2,A,T})(b::Chain{V,1,S}) where {V,A,T,S}
        C = mixedmode(V)
        (C ≥ 0) && throw(error("wrong basis"))
        $(insert_expr((:N,:M,:t))...)
        bi = indices(basis(a),N)
        ib = indexbasis(N,1)
        @inbounds m = bi[2]>M ? bi[2]-M : bi[2]
        @inbounds (((V[m] ? -(a.v) : a.v)*b.v[m])::t)*getbasis(V,ib[bi[1]])
    end
    function (a::Chain{V,2,T})(b::Simplex{V,1,B,S}) where {V,T,S,B}
        C = mixedmode(V)
        (C ≥ 0) && throw(error("wrong basis"))
        $(insert_expr((:N,:t,:df,:di))...)
        Q = bladeindex(N,bits(basis(b)))
        out = zero(mvec(N,1,T))
        @inbounds m,val = df[Q][1],df[Q][2] ? -(b.v) : b.v
        for i ∈ 1:N
            i≠m && @inbounds setblade!(out,a.v[di[Q][i]]*val,one(Bits)<<(i-1),Val{N}())
        end
        return Chain{V,1,t}(out)
    end
    function (a::Chain{V,1,T})(b::Chain{V,1,S}) where {V,T,S}
        $(insert_expr((:N,:M,:t,:df))...)
        out = zero(t)
        for Q ∈ 1:M
            @inbounds out += a.v[df[Q][1]]*(df[Q][2] ? -(b.v[Q]) : b.v[Q])
        end
        return Simplex{V}(out::t,SubManifold{V}())
    end
    function (a::Chain{V,2,T})(b::Chain{V,1,S}) where {V,T,S}
        C = mixedmode(V)
        (C ≥ 0) && throw(error("wrong basis"))
        $(insert_expr((:N,:t,:df,:di))...)
        out = zero(mvec(N,1,t))
        for Q ∈ 1:Int(N/2)
            @inbounds m,val = df[Q][1],df[Q][2] ? -(b.v[Q]) : b.v[Q]
            val≠0 && for i ∈ 1:N
                @inbounds i≠m && addblade!(out,a.v[di[Q][i]]*val,one(Bits)<<(i-1),Val{N}())
            end
        end
        return Chain{V,1,t}(out)
    end
end

@eval begin
    function Chain{V,T}(b::Matrix{T}) where {V,T}
        mixedmode(V)≥0 && throw(error("$V does not support this conversion"))
        $(insert_expr((:N,:M))...)
        size(b) ≠ (M,M) && throw(error("dimension mismatch"))
        out = zeros(mvec(N,2,T))
        for i ∈ 1:M
            x = one(Bits)<<(i-1)
            for j ∈ 1:M
                @inbounds b[j,i]≠0 && setblade!(out,b[j,i],x⊻(one(Bits)<<(M+j-1)),Val{N}())
            end
        end
        return Chain{V,2,T}(out)
    end
end
