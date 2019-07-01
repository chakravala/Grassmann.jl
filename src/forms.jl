#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

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
@pure function dualindex(V::VectorBundle{N}) where N
    C = mixedmode(V)<0
    for n ∈ 2length(C ? dualindexC_cache : dualindex_cache)+2:2:N
        push!(C ? dualindexC_cache : dualindex_cache,Vector{Int}[])
    end
    M = Int(N/2)
    @inbounds if isempty((C ? dualindexC_cache : dualindex_cache)[M])
        #df = dualform(C ? VectorBundle(M)⊕VectorBundle(M)' : VectorBundle(n))
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

## Basis forms

(a::Basis)(b::T) where {T<:TensorAlgebra} = interform(a,b)
function (a::Basis{V,1,A})(b::Basis{V,1,B}) where {V,A,B}
    T = valuetype(a)
    x = bits(a)
    X = mixedmode(V)<0 ? x>>Int(ndims(V)/2) : x
    bits(b)∉(x,X) ? zero(V) : ((V[intlog(B)+1] ? -one(T) : one(T))*Basis{V}())
end
function (a::Basis{V,2,A})(b::Basis{V,1,B}) where {V,A,B}
    C = mixedmode(V)
    (C ≥ 0) && throw(error("wrong basis"))
    N = ndims(V)
    M = Int(N/2)
    T = valuetype(a)
    bi = indices(a,N)
    ib = indexbasis(N,1)
    @inbounds v = ib[bi[2]>M ? bi[2]-M : bi[2]]
    t = bits(b)≠v
    @inbounds t ? zero(V) : ((V[intlog(v)+1] ? -one(T) : one(T))*getbasis(V,ib[bi[1]]))
end

# Blade forms

for Blade ∈ MSB
    @eval begin
        (a::$Blade)(b::T) where {T<:TensorAlgebra} = interform(a,b)
        function (a::Basis{V,1,A})(b::$Blade{V,1,X,T} where X) where {V,A,T}
            x = bits(a)
            X = mixedmode(V)<0 ? x>>Int(ndims(V)/2) : x
            Y = bits(basis(b))
            Y∉(x,X) && (return zero(V))
            (V[intlog(Y)+1] ? -(b.v) : b.v) * Basis{V}()
        end
        function (a::$Blade{V,1,X,T} where X)(b::Basis{V,1,B}) where {V,T,B}
            x = bits(basis(a))
            X = mixedmode(V)<0 ? x>>Int(ndims(V)/2) : x
            Y = bits(b)
            Y∉(x,X) && (return zero(V))
            (V[intlog(Y)+1] ? -(a.v) : a.v) * Basis{V}()
        end
        function (a::$Blade{V,2,A,T})(b::Basis{V,1,B}) where {V,A,T,B}
            C = mixedmode(V)
            (C ≥ 0) && throw(error("wrong basis"))
            $(insert_expr((:N,:M))...)
            bi = indices(basis(a),N)
            ib = indexbasis(N,1)
            @inbounds v = ib[bi[2]>M ? bi[2]-M : bi[2]]
            t = bits(b)≠v
            @inbounds t ? zero(V) : ((V[intlog(v)+1] ? -(a.v) : a.v)*getbasis(V,ib[bi[1]]))
        end
        function (a::$Basis{V,2,A})(b::$Blade{V,1,B,T}) where {V,A,B,T}
            C = mixedmode(V)
            (C ≥ 0) && throw(error("wrong basis"))
            $(insert_expr((:N,:M))...)
            bi = indices(a,N)
            ib = indexbasis(N,1)
            @inbounds v = ib[bi[2]>M ? bi[2]-M : bi[2]]
            t = bits(basis(b))≠v
            @inbounds t ? zero(V) : ((V[intlog(v)+1] ? -(b.v) : b.v)*getbasis(V,ib[bi[1]]))
        end
    end
    for Other ∈ MSB
        @eval begin
            function (a::$Blade{V,1,X,T} where X)(b::$Other{V,1,Y,S} where Y) where {V,T,S}
                $(insert_expr((:t,))...)
                x = bits(basis(a))
                X = mixedmode(V)<0 ? x>>Int(ndims(V)/2) : x
                Y = bits(basis(b))
                Y∉(x,X) && (return zero(V))
                SBlade{V}((a.v*(V[intlog(Y)+1] ? -(b.v) : b.v))::t,Basis{V}())
            end
            function (a::$Blade{V,2,A,T})(b::$Other{V,1,B,S}) where {V,A,T,B,S}
                C = mixedmode(V)
                (C ≥ 0) && throw(error("wrong basis"))
                $(insert_expr((:N,:M,:t))...)
                bi = indices(basis(a),N)
                ib = indexbasis(N,1)
                @inbounds v = ib[bi[2]>M ? bi[2]-M : bi[2]]
                j = bits(basis(b))≠v
                @inbounds j ? zero(V) : (a.v*(V[intlog(v)+1] ? -(b.v) : b.v)*getbasis(V,ib[bi[1]]))
            end
        end
    end
end

## Chain forms

for Chain ∈ MSC
    @eval begin
        (a::$Chain)(b::T) where {T<:TensorAlgebra} = interform(a,b)
        function (a::Basis{V,1,A})(b::$Chain{T,V,1}) where {V,A,T}
            x = bits(a)
            X = mixedmode(V)<0 ? x>>Int(ndims(V)/2) : x
            Y = 0≠X ? X : x
            @inbounds out = b.v[bladeindex(ndims(V),Y)]
            SBlade{V}((V[intlog(Y)+1] ? -(out) : out),Basis{V}())
        end
        function (a::$Chain{T,V,1})(b::Basis{V,1,B}) where {T,V,B}
            x = bits(b)
            X = mixedmode(V)<0 ? x<<Int(ndims(V)/2) : x
            Y = X>2^ndims(V) ? x : X
            @inbounds out = a.v[bladeindex(ndims(V),Y)]
            SBlade{V}((V[intlog(x)+1] ? -(out) : out),Basis{V}())
        end
        function (a::Basis{V,2,A})(b::$Chain{T,V,1}) where {V,A,T}
            C = mixedmode(V)
            (C ≥ 0) && throw(error("wrong basis"))
            $(insert_expr((:N,:M))...)
            bi = indices(basis(a),N)
            ib = indexbasis(N,1)
            @inbounds m = bi[2]>M ? bi[2]-M : bi[2]
            @inbounds ((V[m] ? -(b.v[m]) : b.v[m])*getbasis(V,ib[bi[1]]))
        end
        function (a::$Chain{T,V,2})(b::Basis{V,1,B}) where {T,V,B}
            C = mixedmode(V)
            (C ≥ 0) && throw(error("wrong basis"))
            $(insert_expr((:N,:df,:di))...)
            Q = bladeindex(N,bits(b))
            @inbounds m,val = df[Q][1],df[Q][2] ? -(value(b)) : value(b)
            out = zero(mvec(N,1,T))
            for i ∈ 1:N
                i≠m && @inbounds setblade!(out,a.v[di[Q][i]]*val,one(Bits)<<(i-1),Dimension{N}())
            end
            return $Chain{T,V,1}(out)
        end
    end
    for Blade ∈ MSB
        @eval begin
            function (a::$Chain{T,V,1})(b::$Blade{V,1,X,S} where X) where {V,A,T,S}
                $(insert_expr((:t,))...)
                x = bits(basis(b))
                X = mixedmode(V)<0 ? x<<Int(ndims(V)/2) : x
                Y = X>2^ndims(V) ? x : X
                @inbounds out = a.v[bladeindex(ndims(V),Y)]
                SBlade{V}(((V[intlog(x)+1] ? -(out) : out)*b.v)::t,Basis{V}())
            end
            function (a::$Blade{V,1,X,T} where X)(b::$Chain{S,V,1}) where {V,T,S}
                $(insert_expr((:t,))...)
                x = bits(basis(a))
                X = mixedmode(V)<0 ? x>>Int(ndims(V)/2) : x
                Y = 0≠X ? X : x
                @inbounds out = b.v[bladeindex(ndims(V),Y)]
                SBlade{V}((a.v*(V[intlog(Y)+1] ? -(out) : out))::t,Basis{V}())
            end
            function (a::$Blade{V,2,A,T})(b::$Chain{S,V,1}) where {V,A,T,S}
                C = mixedmode(V)
                (C ≥ 0) && throw(error("wrong basis"))
                $(insert_expr((:N,:M,:t))...)
                bi = indices(basis(a),N)
                ib = indexbasis(N,1)
                @inbounds m = bi[2]>M ? bi[2]-M : bi[2]
                @inbounds (((V[m] ? -(a.v) : a.v)*b.v[m])::t)*getbasis(V,ib[bi[1]])
            end
            function (a::$Chain{T,V,2})(b::$Blade{V,1,B,S}) where {V,T,S,B}
                C = mixedmode(V)
                (C ≥ 0) && throw(error("wrong basis"))
                $(insert_expr((:N,:t,:df,:di))...)
                Q = bladeindex(N,bits(basis(b)))
                out = zero(mvec(N,1,T))
                @inbounds m,val = df[Q][1],df[Q][2] ? -(b.v) : b.v
                for i ∈ 1:N
                    i≠m && @inbounds setblade!(out,a.v[di[Q][i]]*val,one(Bits)<<(i-1),Dimension{N}())
                end
                return $Chain{t,V,1}(out)
            end
        end
    end
    for Other ∈ MSC
        Final = ((Chain == MSC[1]) && (Other == MSC[1])) ? MSB[1] : MSB[2]
        @eval begin
            function (a::$Chain{T,V,1})(b::$Other{S,V,1}) where {V,T,S}
                $(insert_expr((:N,:M,:t,:df))...)
                out = zero(t)
                for Q ∈ 1:M
                    @inbounds out += a.v[df[Q][1]]*(df[Q][2] ? -(b.v[Q]) : b.v[Q])
                end
                return $Final{V}(out::t,Basis{V}())
            end
            function (a::$Chain{T,V,2})(b::$Other{S,V,1}) where {V,T,S}
                C = mixedmode(V)
                (C ≥ 0) && throw(error("wrong basis"))
                $(insert_expr((:N,:t,:df,:di))...)
                out = zero(mvec(N,1,t))
                for Q ∈ 1:Int(N/2)
                    @inbounds m,val = df[Q][1],df[Q][2] ? -(b.v[Q]) : b.v[Q]
                    val≠0 && for i ∈ 1:N
                        @inbounds i≠m && addblade!(out,a.v[di[Q][i]]*val,one(Bits)<<(i-1),Dimension{N}())
                    end
                end
                return $Chain{t,V,1}(out)
            end
        end
    end
end


for Chain ∈ MSC
    @eval begin
        function $Chain{T,V}(b::Matrix{T}) where {T,V}
            mixedmode(V)≥0 && throw(error("$V does not support this conversion"))
            $(insert_expr((:N,:M))...)
            size(b) ≠ (M,M) && throw(error("dimension mismatch"))
            out = zeros(mvec(N,2,T))
            for i ∈ 1:M
                x = one(Bits)<<(i-1)
                for j ∈ 1:M
                    @inbounds b[j,i]≠0 && setblade!(out,b[j,i],x⊻(one(Bits)<<(M+j-1)),Dimension{N}())
                end
            end
            return $Chain{T,V,2}(out)
        end
    end
end
