#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

@inline interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = a(b)
@inline function interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{W}} where {V,W}
    VW = V∪W
    return VW(a)(VW(b))
end

#### need separate storage for m and F for caching

const dualform_cache = Vector{Tuple{Int,Bool}}[]
const dualformC_cache = Vector{Tuple{Int,Bool}}[]
function dualform(V::VectorSpace{N}) where {N}
    C = dualtype(V)<0
    for n ∈ 2length(C ? dualformC_cache : dualform_cache)+2:2:N
        M = Int(n/2)
        ib = indexbasis(n,1)
        mV = Array{Tuple{Int,Bool},1}(undef,M)
        for Q ∈ 1:M
            x = ib[Q]
            X = C ? x<<M : x
            Y = X>(1<<N) ? x : X
            mV[Q] = (intlog(Y)+1,V[intlog(x)+1])
        end
        push!(C ? dualformC_cache : dualform_cache,mV)
    end
    (C ? dualformC_cache : dualform_cache)[Int(N/2)]
end

const dualindex_cache = Vector{Vector{Int}}[]
const dualindexC_cache = Vector{Vector{Int}}[]
function dualindex(V::VectorSpace{N}) where N
    C = dualtype(V)<0
    for n ∈ 2length(C ? dualindexC_cache : dualindex_cache)+2:2:N
        M = Int(n/2)
        df = dualform(C ? VectorSpace(M)⊕VectorSpace(M)' : VectorSpace(n))
        di = Array{Vector{Int},1}(undef,M)
        for Q ∈ 1:M
            m = df[Q][1]
            di[Q] = [bladeindex(n,bit2int(basisbits(n,[i,m]))) for i ∈ 1:n]
        end
        push!(C ? dualindexC_cache : dualindex_cache,di)
    end
    (C ? dualindexC_cache : dualindex_cache)[Int(N/2)]
end

## Basis forms

(a::Basis)(b::T) where {T<:TensorAlgebra} = interform(a,b)
function (a::Basis{V,1,A})(b::Basis{V,1,B}) where {V,A,B}
    T = valuetype(a)
    x = bits(a)
    X = dualtype(V)<0 ? x>>Int(ndims(V)/2) : x
    bits(b)∉(x,X) ? zero(V) : ((V[intlog(B)+1] ? -one(T) : one(T))*Basis{V}())
end
function (a::Basis{V,2,A})(b::Basis{V,1,B}) where {V,A,B}
    C = dualtype(V)
    (C ≥ 0) && throw(error("wrong basis"))
    T = valuetype(a)
    bi = basisindices(a)
    ib = indexbasis(ndims(V),1)
    M = Int(ndims(V)/2)
    v = ib[bi[2]>M ? bi[2]-M : bi[2]]
    t = bits(b)≠v
    t ? zero(V) : ((V[intlog(v)+1] ? -one(T) : one(T))*getbasis(V,ib[bi[1]]))
end

# Value forms

for Value ∈ MSV
    @eval begin
        (a::$Value)(b::T) where {T<:TensorAlgebra} = interform(a,b)
        function (a::Basis{V,1,A})(b::$Value{V,1,X,T} where X) where {V,A,T}
            x = bits(a)
            X = dualtype(V)<0 ? x>>Int(ndims(V)/2) : x
            Y = bits(basis(b))
            Y∉(x,X) && (return zero(V))
            (V[intlog(Y)+1] ? -(b.v) : b.v) * Basis{V}()
        end
        function (a::$Value{V,1,X,T} where X)(b::Basis{V,1,B}) where {V,T,B}
            x = bits(basis(a))
            X = dualtype(V)<0 ? x>>Int(ndims(V)/2) : x
            Y = bits(b)
            Y∉(x,X) && (return zero(V))
            (V[intlog(Y)+1] ? -(a.v) : a.v) * Basis{V}()
        end
        function (a::$Value{V,2,A,T})(b::Basis{V,1,B}) where {V,A,T,B}
            C = dualtype(V)
            (C ≥ 0) && throw(error("wrong basis"))
            bi = basisindices(basis(a))
            ib = indexbasis(ndims(V),1)
            M = Int(ndims(V)/2)
            v = ib[bi[2]>M ? bi[2]-M : bi[2]]
            t = bits(b)≠v
            t ? zero(V) : ((V[intlog(v)+1] ? -(a.v) : a.v)*getbasis(V,ib[bi[1]]))
        end
        function (a::$Basis{V,2,A})(b::$Value{V,1,B,T}) where {V,A,B,T}
            C = dualtype(V)
            (C ≥ 0) && throw(error("wrong basis"))
            bi = basisindices(a)
            ib = indexbasis(ndims(V),1)
            M = Int(ndims(V)/2)
            v = ib[bi[2]>M ? bi[2]-M : bi[2]]
            t = bits(basis(b))≠v
            t ? zero(V) : ((V[intlog(v)+1] ? -(b.v) : b.v)*getbasis(V,ib[bi[1]]))
        end
    end
    for Other ∈ MSV
        @eval begin
            function (a::$Value{V,1,X,A} where X)(b::$Other{V,1,Y,B} where Y) where {V,A,B}
                T = promote_type(A,B)
                x = bits(basis(a))
                X = dualtype(V)<0 ? x>>Int(ndims(V)/2) : x
                Y = bits(basis(b))
                Y∉(x,X) && (return zero(V))
                SValue{V}((a.v*(V[intlog(Y)+1] ? -(b.v) : b.v))::T,Basis{V}())
            end
            function (a::$Value{V,2,A,T})(b::$Other{V,1,B,S}) where {V,A,T,B,S}
                C = dualtype(V)
                (C ≥ 0) && throw(error("wrong basis"))
                t = promote_type(T,S)
                bi = basisindices(basis(a))
                ib = indexbasis(ndims(V),1)
                M = Int(ndims(V)/2)
                v = ib[bi[2]>M ? bi[2]-M : bi[2]]
                t = bits(basis(b))≠v
                t ? zero(V) : (a.v*(V[intlog(v)+1] ? -(b.v) : b.v)*getbasis(V,ib[bi[1]]))
            end
        end
    end
end

## Blade forms

for Blade ∈ MSB
    @eval begin
        (a::$Blade)(b::T) where {T<:TensorAlgebra} = interform(a,b)
        function (a::Basis{V,1,A})(b::$Blade{T,V,1}) where {V,A,T}
            x = bits(a)
            X = dualtype(V)<0 ? x>>Int(ndims(V)/2) : x
            Y = 0≠X ? X : x
            out = b.v[bladeindex(ndims(V),Y)]
            SValue{V}((V[intlog(Y)+1] ? -(out) : out),Basis{V}())
        end
        function (a::$Blade{T,V,1})(b::Basis{V,1,B}) where {T,V,B}
            x = bits(b)
            X = dualtype(V)<0 ? x<<Int(ndims(V)/2) : x
            Y = X>2^ndims(V) ? x : X
            out = a.v[bladeindex(ndims(V),Y)]
            SValue{V}((V[intlog(x)+1] ? -(out) : out),Basis{V}())
        end
        function (a::Basis{V,2,A})(b::$Blade{T,V,1}) where {V,A,T}
            C = dualtype(V)
            (C ≥ 0) && throw(error("wrong basis"))
            bi = basisindices(basis(a))
            ib = indexbasis(ndims(V),1)
            M = Int(ndims(V)/2)
            m = bi[2]>M ? bi[2]-M : bi[2]
            ((V[m] ? -(b.v[m]) : b.v[m])*getbasis(V,ib[bi[1]]))
        end
        function (a::$Blade{T,V,2})(b::Basis{V,1,B}) where {T,V,B}
            C = dualtype(V)
            (C ≥ 0) && throw(error("wrong basis"))
            N = ndims(V)
            x = bits(b)
            X = dualtype(V)<0 ? x<<Int(N/2) : x
            Y = X>2^ndims(V) ? x : X
            m = intlog(Y)+1
            out = zero(mvec(N,1,T))
            for i ∈ 1:N
                if i≠m
                    F = bladeindex(N,bit2int(basisbits(N,[i,m])))
                    setblade!(out,V[intlog(x)+1] ? -(a.v[F]) : a.v[F],0x0001<<(i-1),Dimension{N}())
                end
            end
            return $Blade{T,V,1}(out)
        end
    end
    for Value ∈ MSV
        @eval begin
            function (a::$Blade{T,V,1})(b::$Value{V,1,X,S} where X) where {V,A,T,S}
                t = promote_type(T,S)
                x = bits(basis(b))
                X = dualtype(V)<0 ? x<<Int(ndims(V)/2) : x
                Y = X>2^ndims(V) ? x : X
                out = a.v[bladeindex(ndims(V),Y)]
                SValue{V}(((V[intlog(x)+1] ? -(out) : out)*b.v)::t,Basis{V}())
            end
            function (a::$Value{V,1,X,T} where X)(b::$Blade{S,V,1}) where {V,T,S}
                t = promote_type(T,S)
                x = bits(basis(a))
                X = dualtype(V)<0 ? x>>Int(ndims(V)/2) : x
                Y = 0≠X ? X : x
                out = b.v[bladeindex(ndims(V),Y)]
                SValue{V}((a.v*(V[intlog(Y)+1] ? -(out) : out))::t,Basis{V}())
            end
            function (a::$Value{V,2,A,T})(b::$Blade{S,V,1}) where {V,A,T,S}
                C = dualtype(V)
                (C ≥ 0) && throw(error("wrong basis"))
                t = promote_type(T,S)
                bi = basisindices(basis(a))
                ib = indexbasis(ndims(V),1)
                M = Int(ndims(V)/2)
                m = bi[2]>M ? bi[2]-M : bi[2]
                (((V[m] ? -(a.v) : a.v)*b.v[m])::t)*getbasis(V,ib[bi[1]])
            end
            function (a::$Blade{T,V,2})(b::$Value{V,1,B,S}) where {V,T,S,B}
                C = dualtype(V)
                (C ≥ 0) && throw(error("wrong basis"))
                t = promote_type(T,S)
                N = ndims(V)
                x = bits(basis(b))
                X = dualtype(V)<0 ? x<<Int(N/2) : x
                Y = X>2^ndims(V) ? x : X
                m = intlog(Y)+1
                out = zero(mvec(N,1,T))
                for i ∈ 1:N
                    if i≠m
                        F = bladeindex(N,bit2int(basisbits(N,[i,m])))
                        setblade!(out,a.v[F]*(V[intlog(x)+1] ? -(b.v) : b.v),one(Bits)<<(i-1),Dimension{N}())
                    end
                end
                return $Blade{t,V,1}(out)
            end
        end
    end
    for Other ∈ MSB
        Final = ((Blade == MSB[1]) && (Other == MSB[1])) ? MSV[1] : MSV[2]
        @eval begin
            function (a::$Blade{A,V,1})(b::$Other{B,V,1}) where {V,A,B}
                T = promote_type(A,B)
                N = ndims(V)
                M = Int(N/2)
                df = dualform(V)
                out = zero(T)
                for Q ∈ 1:M
                    out += a.v[df[Q][1]]*(df[Q][2] ? -(b.v[Q]) : b.v[Q])
                end
                return $Final{V}(out::T,Basis{V}())
            end
            function (a::$Blade{T,V,2})(b::$Other{S,V,1}) where {V,T,S}
                C = dualtype(V)
                (C ≥ 0) && throw(error("wrong basis"))
                t = promote_type(T,S)
                N = ndims(V)
                df = dualform(V)
                di = dualindex(V)
                out = zero(mvec(N,1,t))
                for Q ∈ 1:Int(N/2)
                    m = df[Q][1]
                    id = di[Q]
                    for i ∈ 1:N
                        i≠m && addblade!(out,a.v[id[i]]*(df[Q][2] ? -(b.v[Q]) : b.v[Q]),one(Bits)<<(i-1),Dimension{N}())
                    end
                end
                return $Blade{t,V,1}(out)
            end
        end
    end
end

