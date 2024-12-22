#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export TensorNested, Projector, Dyadic, Proj, outer, operator, gerschgorin
export DiagonalOperator, TensorOperator, Endomorphism, Outermorphism, outermorphism
export sylvester, characteristic, eigen, eigvecs, eigvals, eigpolys, eigprods, eigmults
export eigvalsreal, eigvalscomplex, eigvecsreal, eigvecscomplex, eigenreal, eigencomplex
export MetricTensor, metrictensor, metricextensor, InducedMetric
export @TensorOperator, @Endomorphism, @Outermorphism, @SpectralOperator
import LinearAlgebra: eigvals, eigvecs, eigen

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

#= need separate storage for m and F for caching

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
        x = UInt(b)
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
end=#

(t::TensorGraded)(y::TensorGraded...) = contraction(t,∧(y...))
(t::TensorMixed)(y::TensorGraded...) = contraction(t,∧(y...))

# Dyadic

abstract type TensorNested{V,T} <: Manifold{V,T} end
@pure Manifold(::T) where T<:TensorNested{V} where V = V
@pure Manifold(::Type{T}) where T<:TensorNested{V} where V = V

transpose_row(t::Values{N,<:Chain{V}},i,W=V) where {N,V} = Chain{W,1}(getindex.(t,i))
transpose_row(t::FixedVector{N,<:Chain{V}},i,W=V) where {N,V} = Chain{W,1}(getindex.(t,i))
transpose_row(t::Chain{V,1,<:Chain},i) where V = transpose_row(value(t),i,V)
@generated _transpose(t::Values{N,<:Chain{V,1}},W=V) where {N,V} = :(Chain{V,1}(transpose_row.(Ref(t),$(list(1,mdims(V))),W)))
@generated _transpose(t::FixedVector{N,<:Chain{V,1}},W=V) where {N,V} = :(Chain{V,1}(transpose_row.(Ref(t),$(list(1,mdims(V))),W)))
Base.transpose(t::Chain{V,1,<:Chain{V,1}}) where V = _transpose(value(t))
Base.transpose(t::Chain{V,1,<:Chain{W,1}}) where {V,W} = _transpose(value(t),V)
Base.inv(t::TensorNested,g) = inv(t)
@generated function LinearAlgebra.tr(m::Chain{V,G,<:Chain{V,G},N}) where {V,G,N}
    :(sum(Values($([:(m[$i][$i]) for i ∈ list(1,N)]...))))
end
for typ ∈ (:Spinor,:AntiSpinor,:Multivector)
    @eval @generated function LinearAlgebra.tr(m::$typ{V,<:$typ{V},N}) where {V,N}
        :(sum(Values($([:(m[$i][$i]) for i ∈ list(1,N)]...))))
    end
end

Base.Matrix(t::TensorAlgebra) = matrix(t)

matrix(m::Chain{V,G,<:TensorGraded{W,G}}) where {V,W,G} = hcat(value.(Chain.(value(m)))...)
matrix(m::Chain{V,G,<:Chain{W,G}}) where {V,W,G} = hcat(value.(value(m))...)
matrix(m::TensorGraded{V,G,<:Chain{W,G}}) where {V,W,G} = hcat(value.(value(Chain(m)))...)
matrix(m::TensorGraded{V,G,<:TensorGraded{W,G}}) where {V,W,G} = hcat(value.(Chain.(value(Chain(m))))...)

matrix(m::Multivector{V,<:TensorAlgebra{W}}) where {V,W} = hcat(value.(Multivector.(value(m)))...)
matrix(m::Multivector{V,<:Multivector{W}}) where {V,W} = hcat(value.(value(m))...)
matrix(m::TensorAlgebra{V,<:Multivector{W}}) where {V,W} = hcat(value.(Multivector.(value(Multivector(m))))...)

Chain(m::Matrix) = DyadicChain{Submanifold(size(m)[1])}(m)
Chain{V}(m::Matrix) where V = Chain{V,1}(m)
function Chain{V,G}(m::Matrix) where {V,G}
    N = size(m)[2]
    Chain{V,G,Chain{N≠mdims(V) ? Submanifold(N) : V,G}}(m)
end
Chain{V,G,<:Chain{W,G}}(m::Matrix) where {V,W,G} = Chain{V,G}(Chain{W,G}.(getindex.(Ref(m),:,list(1,size(m)[2]))))

Multivector(m::Matrix) = Multivector{log2sub(size(m)[1]),1}(m)
function Multivector{V}(m::Matrix) where V
    N = size(m)[2]
    Multivector{V,Multivector{Int(log2(N))≠mdims(V) ? log2sub(N) : V}}(m)
end
Multivector{V,<:Chain{W}}(m::Matrix) where {V,W} = Multivector{V}(Multivector{W}.(getindex.(Ref(m),:,list(1,size(m)[2]))))

for pinor ∈ (:Spinor,:AntiSpinor)#,:Multivector)
    @eval begin
        matrix(m::$pinor{V,<:TensorAlgebra{W}}) where {V,W} = hcat(value.($pinor.(value(m)))...)
        matrix(m::$pinor{V,<:$pinor{W}}) where {V,W} = hcat(value.(value(m))...)
        matrix(m::TensorGraded{V,G,<:$pinor{W}}) where {V,G,W} = hcat(value.(value($pinor(m)))...)
        $pinor(m::Matrix) = $pinor{log2sub(size(m)[1]),1}(m)
        function $pinor{V}(m::Matrix) where V
            N = size(m)[2]
            $pinor{V,$pinor{Int(log2(N))≠mdims(V) ? log2sub(N) : V}}(m)
        end
        $pinor{V,<:$pinor{W}}(m::Matrix) where {V,W} = $pinor{V}($pinor{W}.(getindex.(Ref(m),:,list(1,size(m)[2]))))
    end
end

display_matrix(m::Chain{V,G,<:TensorGraded{W,G}}) where {V,G,W} = vcat(transpose([Submanifold(V),chainbasis(V,G)...]),hcat(chainbasis(W,G),matrix(m)))
display_matrix(m::TensorGraded{V,G,<:Spinor{W}}) where {V,G,W} = vcat(transpose([Submanifold(V),evenbasis(V)...]),hcat(evenbasis(W),matrix(m)))
display_matrix(m::TensorGraded{V,G,<:AntiSpinor{W}}) where {V,G,W} = vcat(transpose([Submanifold(V),oddbasis(V)...]),hcat(oddbasis(W),matrix(m)))
display_matrix(m::TensorAlgebra{V,<:Multivector{W}}) where {V,W} = vcat(transpose([Submanifold(V),fullbasis(V)...]),hcat(fullbasis(W),matrix(m)))
for (pinor,bas) ∈ ((:Spinor,:evenbasis),(:AntiSpinor,:oddbasis),(:Multivector,:fullbasis))
    @eval display_matrix(m::$pinor{V,<:TensorAlgebra{W}}) where {V,W} = vcat(transpose([Submanifold(V),$bas(V)...]),hcat($bas(W),matrix(m)))
end

struct Projector{V,T,Λ} <: TensorNested{V,T}
    v::T
    λ::Λ
    Projector{V,T,Λ}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
    Projector{V,T}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
    Projector{V}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
end

const Proj = Projector
const SpectralOperator{V,T<:Simplex{V},Λ} = Projector{V,T,Λ}
export SpectralOperator

Proj(v::T,λ=1) where T<:TensorGraded{V} where V = Proj{V}(v/abs(v),λ)
Proj(v::Chain{W,1,<:Chain{V}},λ=1) where {V,W} = Proj{V}(Chain(value(v)./abs.(value(v))),λ)
#Proj(v::Chain{V,1,<:TensorNested},λ=1) where V = Proj{V}(v,λ)
SpectralOperator(t::AbstractMatrix) = SpectralOperator(Endomorphism(t))
SpectralOperator{V}(t::AbstractMatrix) where V = SpectralOperator(Endomorphism{V}(t))

macro SpectralOperator(ex)
    SpectralOperator(eval(ex))
end

(P::Projector)(x) = contraction(P,x)
function (T::Projector{V})(x::TensorAlgebra{V},y::TensorAlgebra{V}) where V
    vecdot(contraction(T,x),y)
end

function Base.exp(P::Proj{V}) where V
    out = exp(Chain(P))[1]
    Proj{V}(out/sqrt(out[1]))
end
function Base.log(P::Proj{V}) where V
    out = log(Endomorphism(P))[1]
    Proj{V}(out/sqrt(out[1]))
end
Base.exp(P::SpectralOperator{V}) where V = Proj{V}(P.v,map(exp,P.λ))
Base.log(P::SpectralOperator{V}) where V = Proj{V}(P.v,map(log,P.λ))
Base.inv(P::SpectralOperator{V}) where V = Proj{V}(P.v,map(inv,P.λ))

getindex(P::Proj,i::Int,j::Int) = P.v[i]*P.v[j]
getindex(P::Proj{V,<:Chain{W,1,<:Chain}},i::Int) where {V,W} = Proj{V}(P.v[i],P.λ[i])
getindex(P::Proj{V,<:Chain{W,1,<:Chain}} where {V,W},i::Int,j::Int) = sum(column(P.v,i).*column(P.v,j))
#getindex(P::Proj{V,<:Chain{V,1,<:TensorNested}} where V,i::Int,j::Int) = sum(getindex.(value(P.v),i,j))

Leibniz.extend_parnot(Projector)

show(io::IO,P::Proj{V,T,Λ}) where {V,T,Λ<:Real} = print(io,isone(P.λ) ? "" : P.λ,"Proj(",P.v,")")
show(io::IO,P::Proj{V,T,Λ}) where {V,T,Λ} = print(io,"(",P.λ,")Proj(",P.v,")")

#Chain{V}(P::Proj{V,T}) where {V,T<:Chain{V,1,<:TensorNested}} = sum(Chain.(value(P.v)))
Chain{V}(P::Proj{V,<:Chain{W,1,<:Chain}}) where {V,W} = sum(outer.(value(P.v).*value(P.λ),value(P.v)))
Chain{V}(P::Proj{V}) where V = outer(P.v*P.λ,P.v)
Chain(P::Proj{V}) where V = Chain{V}(P)

struct Dyadic{V,X,Y} <: TensorNested{V,X}
    x::X
    y::Y
    Dyadic{V,X,Y}(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = new{DirectSum.submanifold(V),X,Y}(x,y)
    Dyadic{V}(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = new{DirectSum.submanifold(V),X,Y}(x,y)
end

Dyadic(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = Dyadic{V}(x,y)
Dyadic(P::Projector) = Dyadic(P.v*P.λ,P.v)
Dyadic(D::Dyadic) = D

(P::Dyadic)(x) = contraction(P,x)
function (T::Dyadic{V})(x::TensorAlgebra{V},y::TensorAlgebra{V}) where V
    vecdot(contraction(T,x),y)
end

Base.expm1(P::Dyadic) = expm1(Endomorphism(P))
Base.exp(P::Dyadic) = exp(Endomorphism(P))
Base.log(P::Dyadic) = log(Endomorphism(P))

getindex(P::Dyadic,i::Int,j::Int) = P.x[i]*P.y[j]
Base.transpose(P::Dyadic) = Dyadic(P.y,P.x)

show(io::IO,P::Dyadic) = print(io,"(",P.x,")⊗(",P.y,")")

Chain{V}(P::Dyadic{V}) where V = outer(P.x,P.y)
Chain(P::Dyadic{V}) where V = Chain{V}(P)

# DiagonalOperator

struct DiagonalOperator{V,T<:TensorAlgebra{V}} <: TensorNested{V,T}
    v::T
    DiagonalOperator{V}(t::T) where {V,T<:TensorAlgebra{V}} = new{V,T}(t)
    DiagonalOperator(t::T) where {V,T<:TensorAlgebra{V}} = new{V,T}(t)
end
const DiagonalMorphism{V,T<:Chain{V,1}} = DiagonalOperator{V,T}
const DiagonalOutermorphism{V,T<:Multivector{V}} = DiagonalOperator{V,T}
export DiagonalMorphism, DiagonalOutermorphism
DiagonalMorphism(t::TensorGraded{V,1} where V) = DiagonalOperator(Chain(t))
DiagonalMorphism(t::DiagonalOutermorphism) = DiagonalOperator(value(t)(Val(1)))
DiagonalOutermorphism(t::Multivector) = DiagonalOperator(Multivector(t))
DiagonalOutermorphism(t::Chain{V,1} where V) = outermorphism(DiagonalOperator(t))
DiagonalOutermorphism(t::DiagonalMorphism) = outermorphism(t)
DiagonalOutermorphism(t::AbstractMatrix) = outermorphism(DiagonalOperator(t))
DiagonalMorphism(t::AbstractMatrix) = DiagonalOperator(t)
DiagonalOperator(t::AbstractMatrix) = DiagonalOperator{Submanifold(@inbounds size(t)[1])}(t)
DiagonalOutermorphism{V}(t::AbstractMatrix) where V = outermorphism(DiagonalOperator{V}(t))
DiagonalMorphism{V}(t::AbstractMatrix) where V = DiagonalOperator{V}(t)
DiagonalOperator{V}(m::AbstractMatrix) where V = DiagonalOperator(Chain{V}(getindex.(Ref(m),list(1,mdims(V)),list(1,mdims(V)))))

(T::DiagonalOperator{V})(x::TensorAlgebra{V}) where V = contraction(T,x)
function (T::DiagonalOperator{V})(x::TensorAlgebra{V},y::TensorAlgebra{V}) where V
    vecdot(contraction(T,x),y)
end

value(t::DiagonalOperator) = t.v
matrix(m::DiagonalOperator) = matrix(TensorOperator(m))
getindex(t::DiagonalOperator,i::Int,j::Int) = i≠j ? zero(valuetype(value(t))) : value(value(t))[i]
getindex(t::DiagonalOperator,i::Int) = value(t)(i)

Base.zero(t::DiagonalOperator) = DiagonalOperator(zero(value(t)))
Base.zero(t::Type{<:DiagonalOperator{V,T}}) where {V,T} = DiagonalOperator(zero(T))

scalar(m::DiagonalOperator) = tr(m)/length(value(m))
LinearAlgebra.tr(m::DiagonalOperator) = sum(value(value(m)))
LinearAlgebra.det(m::DiagonalMorphism{V}) where V = Chain{V,mdims(V)}(prod(value(value(m))))
LinearAlgebra.det(m::DiagonalOutermorphism{V}) where V = value(m)(Val(mdims(V)))
compound(m::DiagonalMorphism{V},::Val{0}) where V = DiagonalOperator(Chain{V,0}(1))
@generated function compound(m::DiagonalMorphism{V},::Val{G}) where {V,G}
    Expr(:call,:DiagonalOperator,Expr(:call,:(Chain{V,G}),Expr(:call,Values,[Expr(:call,:*,[:(@inbounds m.v[$i]) for i ∈ indices(j)]...) for j ∈ indexbasis(mdims(V),G)]...)))
end
@generated function outermorphism(m::DiagonalMorphism{V}) where V
    Expr(:call,:DiagonalOperator,Expr(:call,:(Multivector{V}),Expr(:call,Values,1,[Expr(:call,:*,[:(@inbounds m.v[$i]) for i ∈ indices(j)]...) for j ∈ indexbasis(mdims(V))[list(2,tdims(V))]]...)))
end

for op ∈ (:(Base.inv),:(Base.exp),:(Base.expm1),:(Base.log))
    @eval begin
        $op(t::DiagonalMorphism) = DiagonalOperator(map($op,value(t)))
        $op(t::DiagonalOutermorphism) = outermorphism(DiagonalOperator(map($op,value(t)(Val(1)))))
    end
end

_axes(t::DiagonalOperator) = (Base.OneTo(length(t.v)),Base.OneTo(length(t.v)))
_axes(t::DiagonalOperator{V,T}) where {V,G,T<:Chain{V,G}} = (Base.OneTo(gdims(T)),Base.OneTo(gdims(T)))

# anything array-like gets summarized e.g. 10-element Array{Int64,1}
Base.summary(io::IO, a::DiagonalOperator) = Base.array_summary(io, a, _axes(a))

show(io::IO,X::DiagonalOperator) = Base.show(io,TensorOperator(X))

function show(io::IO, ::MIME"text/plain", t::DiagonalOperator)
    X = display_matrix(value(TensorOperator(t)))
    if isempty(X) && get(io, :compact, false)::Bool
        return show(io, X)
    end
    show_matrix(io, t, X)
end

struct TensorOperator{V,W,T<:TensorAlgebra{V,<:TensorAlgebra{W}}} <: TensorNested{V,T}
    v::T
    TensorOperator{V,W}(t::T) where {V,W,T<:TensorAlgebra{V,<:TensorAlgebra{W}}} = new{V,W,T}(t)
    TensorOperator{V}(t::T) where {V,W,T<:TensorAlgebra{V,<:TensorAlgebra{W}}} = new{V,W,T}(t)
    TensorOperator(t::T) where {V,W,T<:TensorAlgebra{V,<:TensorAlgebra{W}}} = new{V,W,T}(t)
end

const Endomorphism{V,T<:TensorAlgebra{V,<:TensorAlgebra{V}}} = TensorOperator{V,V,T}
Endomorphism(t::TensorAlgebra{V,<:TensorAlgebra{V}}) where V = TensorOperator{V,V}(t)

(T::TensorOperator{V})(x::TensorAlgebra{V}) where V = contraction(T,x)
function (T::TensorOperator{V,W})(x::TensorAlgebra{V},y::TensorAlgebra{W}) where {V,W}
    vecdot(contraction(T,x),y)
end

macro TensorOperator(ex)
    TensorOperator(eval(ex))
end
macro Endomorphism(ex)
    Endomorphism(eval(ex))
end

value(t::TensorOperator) = t.v
matrix(m::TensorOperator) = matrix(value(m))
compound(m::TensorOperator,g) = TensorOperator(compound(value(m),g))
compound(m::TensorOperator,g::Integer) = TensorOperator(compound(value(m),g))
getindex(t::TensorOperator,i::Int,j::Int) = value(value(t.v)[j])[i]
getindex(t::TensorOperator,i::Int) = value(t.v)[i]
Base.transpose(t::TensorOperator) = TensorOperator(transpose(value(t)))
scalar(m::Endomorphism) = tr(m)/length(value(m))
LinearAlgebra.tr(m::Endomorphism) = tr(value(m))
LinearAlgebra.det(t::TensorOperator) = ∧(value(t))

Base.zero(t::TensorOperator) = TensorOperator(zero(value(t)))
Base.zero(t::Type{<:TensorOperator{V,W,T}}) where {V,W,T} = TensorOperator(zero(T))

Base.log(t::Endomorphism{V,<:Chain}) where V = Endomorphism{V}(log(Matrix(t)))
for op ∈ (:(Base.inv),:(Base.exp),:(Base.expm1))
    @eval $op(t::Endomorphism{V,<:Chain}) where V = TensorOperator($op(value(t)))
end

Endomorphism(t::Projector) = Endomorphism(Chain(t))
Endomorphism(t::Dyadic) = Endomorphism(Chain(t))
@generated Endomorphism(t::DiagonalOperator{V,<:Chain{V,G}}) where {V,G} = :(Endomorphism(Chain{V,G}(value(value(t)).*$(Chain.(chainbasis(V,G))))))
@generated Endomorphism(t::DiagonalOperator{V,<:Spinor{V,G}}) where {V,G} = :(Endomorphism(Spinor{V}(value(value(t)).*$(Spinor.(evenbasis(V))))))
@generated Endomorphism(t::DiagonalOperator{V,<:AntiSpinor{V,G}}) where {V,G} = :(Endomorphism(AntiSpinor{V}(value(value(t)).*$(CoSpinor.(oddbasis(V))))))
@generated Endomorphism(t::DiagonalOutermorphism{V}) where V = :(Endomorphism(Multivector{V}(value(value(t)).*$(Multivector(Λ(V).b)))))
Endomorphism(m::AbstractMatrix) = Endomorphism{Submanifold(@inbounds size(m)[1])}(m)

TensorOperator(t::Projector) = TensorOperator(Chain(t))
TensorOperator(t::Dyadic) = TensorOperator(Chain(t))
@generated TensorOperator(t::DiagonalOperator{V,<:Chain{V,G}}) where {V,G} = :(TensorOperator(Chain{V,G}(value(value(t)).*$(Chain.(chainbasis(V,G))))))
@generated TensorOperator(t::DiagonalOperator{V,<:Spinor{V,G}}) where {V,G} = :(TensorOperator(Spinor{V}(value(value(t)).*$(Spinor.(evenbasis(V))))))
@generated TensorOperator(t::DiagonalOperator{V,<:AntiSpinor{V,G}}) where {V,G} = :(TensorOperator(AntiSpinor{V}(value(value(t)).*$(CoSpinor.(oddbasis(V))))))
@generated TensorOperator(t::DiagonalOutermorphism{V}) where V = :(TensorOperator(Multivector{V}(value(value(t)).*$(Multivector(Λ(V).b)))))
TensorOperator(m::AbstractMatrix) = TensorOperator{Submanifold.(size(m))...}(m)
TensorOperator{V,W}(m::AbstractMatrix) where {V,W} = TensorOperator(Chain{V}(Chain{W,1}.(getindex.(Ref(m),:,list(1,mdims(V))))))

SpectralOperator(t::Endomorphism) = eigen(t)
@generated function DiagonalOperator(t::Endomorphism{V,<:Chain{V,G}}) where {V,G}
    Expr(:call,:DiagonalOperator,Expr(:call,:(Chain{V,G}),[:(t[$i,$i]) for i ∈ list(1,binomial(mdims(V),G))]...))
end
@generated function DiagonalOperator(t::Endomorphism{V,<:Spinor{V}}) where V
    Expr(:call,:DiagonalOperator,Expr(:call,:(Spinor{V}),[:(t[$i,$i]) for i ∈ list(1,binomial(mdims(V),G))]...))
end
@generated function DiagonalOperator(t::Endomorphism{V,<:AntiSpinor{V}}) where V
    Expr(:call,:DiagonalOperator,Expr(:call,:(AntiSpinor{V}),[:(t[$i,$i]) for i ∈ list(1,binomial(mdims(V),G))]...))
end
@generated function DiagonalOperator(t::Endomorphism{V,<:Multivector{V}}) where V
    Expr(:call,:DiagonalOperator,Expr(:call,:(Multivector{V}),[:(t[$i,$i]) for i ∈ list(1,1<<mdims(V))]...))
end

_axes(t::TensorOperator) = (Base.OneTo(length(t.v)),Base.OneTo(length(t.v)))
_axes(t::TensorOperator{V,W,T}) where {V,W,G,L,S<:Chain{W,L},T<:Chain{V,G,S}} = (Base.OneTo(gdims(S)),Base.OneTo(gdims(T)))
_axes(t::TensorOperator{V,W,T}) where {V,W,S<:Multivector{W},T<:TensorAlgebra{V,S}} = (Base.OneTo(tdims(S)),Base.OneTo(tdims(T)))

# anything array-like gets summarized e.g. 10-element Array{Int64,1}
Base.summary(io::IO, a::TensorOperator) = Base.array_summary(io, a, _axes(a))

show(io::IO,X::TensorOperator) = Base.show(io,X.v)

function show(io::IO, ::MIME"text/plain", t::TensorOperator)
    X = display_matrix(t.v)
    if isempty(X) && get(io, :compact, false)::Bool
        return show(io, X)
    end
    show_matrix(io, t, X)
end
function show_matrix(io::IO, t, X)
    # 0) show summary before setting :compact
    summary(io, t)
    isempty(X) && return
    print(io, ":")
    Base.show_circular(io, X) && return

    # 1) compute new IOContext
    if !haskey(io, :compact) && length(axes(X, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    if get(io, :limit, false)::Bool && eltype(X) === Method
        # override usual show method for Vector{Method}: don't abbreviate long lists
        io = IOContext(io, :limit => false)
    end

    if get(io, :limit, false)::Bool && displaysize(io)[1]-4 <= 0
        return print(io, " …")
    else
        println(io)
    end

    # 2) update typeinfo
    #
    # it must come after printing the summary, which can exploit :typeinfo itself
    # (e.g. views)
    # we assume this function is always called from top-level, i.e. that it's not nested
    # within another "show" method; hence we always print the summary, without
    # checking for current :typeinfo (this could be changed in the future)
    io = IOContext(io, :typeinfo => eltype(X))

    # 2) show actual content
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io,X)
end

# Outermorphism

struct Outermorphism{V,T<:Tuple} <: TensorNested{V,T}
    v::T
    Outermorphism{V}(t::T) where {V,T<:Tuple} = new{V,T}(t)
end

@generated function outermorphism(t::Simplex{V}) where V
    :(Outermorphism{V}($(Expr(:tuple,[:(compound(t,Val($g))) for g ∈ list(1,mdims(V))]...))))
end

(T::Outermorphism{V})(x::TensorAlgebra{V}) where V = contraction(T,x)
function (T::Outermorphism{V})(x::TensorAlgebra{V},y::TensorAlgebra{V}) where V
    vecdot(contraction(T,x),y)
end

export @Outermorphism
macro Outermorphism(ex)
    Outermorphism(Endomorphism(eval(ex)))
end

Outermorphism(t::AbstractMatrix) = Outermorphism(Endomorphism(t))
Outermorphism(t::Simplex) = outermorphism(t)
Outermorphism(t::Endomorphism{V,<:Simplex}) where V = outermorphism(value(t))
outermorphism(t::Endomorphism{V,<:Simplex}) where V = outermorphism(value(t))
DiagonalOperator(t::Outermorphism) = outermorphism(DiagonalOperator(TensorOperator(t.v[1])))
value(t::Outermorphism) = t.v
matrix(m::Outermorphism) = matrix(TensorOperator(m))
getindex(t::Outermorphism{V},i::Int) where V = iszero(i) ? Chain{V,0}((Chain(One(V)),)) : t.v[i]
Base.transpose(m::Outermorphism) = Outermorphism(map(transpose,value(m)))
scalar(m::Outermorphism{V}) where V = tr(m)/(1<<mdims(V))
LinearAlgebra.tr(m::Outermorphism) = 1+sum(map(tr,value(m)))
LinearAlgebra.det(m::Outermorphism) = (@inbounds value(value(m)[end])[1])

Base.zero(t::Outermorphism) = Outermorphism(zero.(value(t)))
@generated function Base.zero(t::Type{<:Outermorphism{V,T}}) where {V,T}
    :(Outermorphism{V}($(zero.(([fieldtype(typ,i) for i ∈ 1:fieldcount(typ)]...,)))))
end

compound(m::Outermorphism,::Val{g}) where g = TensorOperator(value(m)[g])
compound(m::Outermorphism,g::Integer) = TensorOperator(value(m)[g])

Endomorphism(t::Outermorphism) = TensorOperator(t)
TensorOperator(t::Outermorphism{V}) where V = TensorOperator(Multivector{V}(vcat(Multivector(One(V)),value.(map.(Multivector,value.(t.v)))...)))

for op ∈ (:(Base.inv),:(Base.exp),:(Base.expm1),:(Base.log))
    @eval $op(t::Outermorphism) = Outermorphism($op(@inbounds value(t)[1]))
end

_axes(t::Outermorphism{V}) where V = (Base.OneTo(tdims(V)),Base.OneTo(tdims(V)))

# anything array-like gets summarized e.g. 10-element Array{Int64,1}
Base.summary(io::IO, a::Outermorphism) = Base.array_summary(io, a, _axes(a))

show(io::IO,X::Outermorphism) = Base.show(io,TensorOperator(X))

function show(io::IO, ::MIME"text/plain", t::Outermorphism)
    X = display_matrix(TensorOperator(t))
    if isempty(X) && get(io, :compact, false)::Bool
        return show(io, X)
    end
    show_matrix(io, t, X)
end

"""
    cayley(V,op=*)

Compute the `cayley` table with `op(a,b)` for each `Submanifold` basis of `V`.
"""
function cayley(V,op=*)
    bas = Λ(V).b
    TensorOperator(Multivector{V}([Multivector{V}(op.(bas,b)) for b ∈ bas]))
end
cayley(V,G::Int,op=*) = cayley(V,Val(G),op)
function cayley(V,::Val{G},op=*) where G
    bas = chainbasis(V,G)
    TensorOperator(Chain{V,G}([Chain{V,G}(op.(bas,b)) for b ∈ bas]))
end
function cayleyeven(V,op=*)
    bas = evenbasis(V)
    TensorOperator(Spinor{V}([Spinor{V}(op.(bas,b)) for b ∈ bas]))
end
function cayleyodd(V,op=*)
    bas = oddbasis(V)
    TensorOperator(AntiSpinor{V}([AntiSpinor{V}(op.(bas,b)) for b ∈ bas]))
end
cayley(b::AbstractVector) = cayley(b,b)
cayley(b::AbstractVector,op) = cayley(b,b,op)
cayley(a::AbstractVector,b::AbstractVector) = a*transpose(b)
cayley(a::AbstractVector,b::AbstractVector,op) = TensorAlgebra{Manifold(Manifold(eltype(a)))}[op(x,y) for x ∈ a, y ∈ b]

companion(x...) = companion(Values(x...))
companion(x::Chain) = companion(value(x))
@generated function companion(x::Values{N}) where N
    V = Submanifold(N)
    Expr(:call,:TensorOperator,Expr(:call,:Chain,
        [i≠N ? Chain(Λ(V).b[i+2]) : :(-Chain(x)) for i ∈ list(1,N)]...))
end

# dyadic products

vecdot(x::Chain{V,G},y::Chain{V,G}) where {V,G} = value(x)⋅value(y)
vecdot(x::TensorGraded{V,G},y::TensorGraded{V,G}) where {V,G} = vecdot(Chain(x),Chain(y))
vecdot(x::TensorGraded{V,G},y::TensorGraded{V,L}) where {V,G,L} = vecdot(multispin(x),y)
vecdot(x::TensorGraded{V,G},y::Couple{V}) where {V,G} = vecdot(multispin(x),y)
vecdot(x::TensorGraded{V,G},y::PseudoCouple{V}) where {V,G} = vecdot(multispin(x),y)
vecdot(x::TensorGraded{V,G},y::Spinor{V}) where {V,G} = isodd(G) ? 0 : vecdot(x,y(Val(G)))
vecdot(x::TensorGraded{V,G},y::CoSpinor{V}) where {V,G} = isodd(G) ? vecdot(x,y(Val(G))) : 0
vecdot(x::TensorGraded{V,G},y::Multivector{V}) where {V,G} = vecdot(x,y(Val(G)))

vecdot(x::Couple{V},y::TensorGraded{V,G}) where {V,G} = vecdot(x,multispin(y))
vecdot(x::Couple{V},y::Couple{V}) where V = vecdot(x,multispin(y))
vecdot(x::Couple{V},y::PseudoCouple{V}) where V = vecdot(x,multispin(y))
vecdot(x::Couple{V},y::Spinor{V}) where V = vecdot(multispin(x),y)
vecdot(x::Couple{V},y::CoSpinor{V}) where V = vecdot(imaginary(x),y)
vecdot(x::Couple{V},y::Multivector{V}) where V = vecdot(multispin(x),y)

vecdot(x::PseudoCouple{V},y::TensorGraded{V,G}) where {V,G} = vecdot(x,multispin(y))
vecdot(x::PseudoCouple{V},y::Couple{V}) where V = vecdot(multispin(x),y)
vecdot(x::PseudoCouple{V},y::PseudoCouple{V}) where V = vecdot(multispin(x),y)
vecdot(x::PseudoCouple{V},y::Spinor{V}) where V = vecdot(multispin(x),y)
vecdot(x::PseudoCouple{V},y::CoSpinor{V}) where V = vecdot(multispin(x),y)
vecdot(x::PseudoCouple{V},y::Multivector{V}) where V = vecdot(multispin(x),y)

vecdot(x::Spinor{V},y::TensorGraded{V,G}) where {V,G} = isodd(G) ? 0 : vecdot(x(Val(G)),y)
vecdot(x::Spinor{V},y::Couple{V}) where V = vecdot(x,multispin(y))
vecdot(x::Spinor{V},y::PseudoCouple{V}) where V = vecdot(x,multispin(y))
vecdot(x::Spinor{V},y::Spinor{V}) where V = value(x)⋅value(y)
vecdot(x::Spinor{V},y::CoSpinor{V}) where V = 0
vecdot(x::Spinor{V},y::Multivector{V}) where V = vecdot(x,even(y))

vecdot(x::CoSpinor{V},y::TensorGraded{V,G}) where {V,G} = isodd(G) ? vecdot(x(Val(G)),y) : 0
vecdot(x::CoSpinor{V},y::Couple{V}) where V = vecdot(x,imaginary(y))
vecdot(x::CoSpinor{V},y::PseudoCouple{V}) where V = vecdot(x,multispin(y))
vecdot(x::CoSpinor{V},y::Spinor{V}) where V = 0
vecdot(x::CoSpinor{V},y::CoSpinor{V}) where V = value(x)⋅value(y)
vecdot(x::CoSpinor{V},y::Multivector{V}) where V = vecdot(x,even(y))

vecdot(x::Multivector{V},y::TensorGraded{V,G}) where {V,G} = vecdot(x(Val(G)),y)
vecdot(x::Multivector{V},y::Couple{V}) where V = vecdot(x,multispin(y))
vecdot(x::Multivector{V},y::PseudoCouple{V}) where V = vecdot(x,multispin(y))
vecdot(x::Multivector{V},y::Spinor{V}) where V = vecdot(even(x),y)
vecdot(x::Multivector{V},y::CoSpinor{V}) where V = vecdot(odd(x),y)
vecdot(x::Multivector{V},y::Multivector{V}) where V = value(x)⋅value(y)

outer(a::Leibniz.Derivation,b::Chain{V,1}) where V= outer(V(a),b)
outer(a::Chain{W},b::Leibniz.Derivation{T,1}) where {W,T} = outer(a,W(b))
outer(a::Chain{W},b::Chain{V,1}) where {W,V} = Chain{V,1}(a.*conj.(value(b)))

contraction_metric(a::TensorNested,b::TensorNested,g) = contraction(a,b)
contraction_metric(a::TensorNested,b::TensorAlgebra,g) = contraction(a,b)
contraction_metric(a::TensorAlgebra,b::TensorNested,g) = contraction(a,b)
wedgedot_metric(a::TensorNested,b::TensorNested,g) = a⟑b
wedgedot_metric(a::TensorNested,b::TensorAlgebra,g) = a⟑b
wedgedot_metric(a::TensorAlgebra,b::TensorNested,g) = a⟑b

contraction(a::Proj,b::TensorOperator) = contraction(a,value(b))
contraction(a::TensorOperator,b::Proj) = contraction(value(a),b)
contraction(a::Proj,b::Outermorphism) = contraction(a,(@inbounds value(b)[1]))
contraction(a::Outermorphism,b::Proj) = contraction((@inbounds value(a)[1]),b)
contraction(a::Proj,b::DiagonalMorphism) = contraction(a,TensorOperator(b))
contraction(a::DiagonalMorphism,b::Proj) = contraction(TensorOperator(a),b)
contraction(a::Proj,b::DiagonalOutermorphism) = contraction(a,DiagonalMorphism(b))
contraction(a::DiagonalOutermorphism,b::Proj) = contraction(DiagonalMorphism(a),b)
contraction(a::SpectralOperator,b::TensorOperator) = contraction(Endomorphism(a),b)
contraction(a::TensorOperator,b::SpectralOperator) = contraction(a,Endomorphism(b))
contraction(a::SpectralOperator,b::DiagonalOperator) = contraction(Endomorphism(a),b)
contraction(a::DiagonalOperator,b::SpectralOperator) = contraction(a,Endomorphism(b))
contraction(a::SpectralOperator,b::Outermorphism) = contraction(Endomorphism(a),b)
contraction(a::Outermorphism,b::SpectralOperator) = contraction(a,Endomorphism(b))
contraction(a::SpectralOperator,b::Dyadic) = contraction(Endomorphism(a),b)
contraction(a::Dyadic,b::SpectralOperator) = contraction(a,Endomorphism(b))
contraction(a::SpectralOperator,b::Projector) = contraction(Endomorphism(a),b)
contraction(a::Projector,b::SpectralOperator) = contraction(a,Endomorphism(b))
contraction(a::Dyadic,b::TensorOperator) = contraction(a,value(b))
contraction(a::TensorOperator,b::Dyadic) = contraction(value(a),b)
contraction(a::Dyadic,b::Outermorphism) = contraction(a,(@inbounds value(b)[1]))
contraction(a::Outermorphism,b::Dyadic) = contraction((@inbounds value(a)[1]),b)
contraction(a::Dyadic,b::DiagonalMorphism) = contraction(a,TensorOperator(b))
contraction(a::DiagonalMorphism,b::Dyadic) = contraction(TensorOperator(a),b)
contraction(a::Dyadic,b::DiagonalOutermorphism) = contraction(a,DiagonalMorphism(b))
contraction(a::DiagonalOutermorphism,b::Dyadic) = contraction(DiagonalMorphism(a),b)

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
Base.:(:)(a::Chain{V,1,<:Chain},b::Chain{V,1,<:Chain}) where V = sum(value(a).⋅value(b))
Base.:(:)(a::Chain{W,1,<:Dyadic{V}},b::Chain{V,1}) where {W,V} = sum(value(a).⋅Ref(b))
#Base.:(:)(a::Chain{W,1,<:Proj{V}},b::Chain{V,1}) where {W,V} = sum(broadcast(⋅,value(a),Ref(b)))

contraction(a::Chain{W},b::Chain{V,G,<:Chain}) where {W,G,V} = Chain{V,G}(Ref(value(a)).⋅value.(value(b)))
contraction(a::Chain{W,L,<:Chain,N},b::Chain{V,G,<:Chain{W,L},M}) where {W,L,G,V,N,M} = Chain{V,G}(value.(Ref(a).⋅value(b)))
contraction(a::Multivector{W,<:Multivector},b::Multivector{V,<:Multivector{W}}) where {W,V} = Multivector{V}(column(Ref(a).⋅value(b)))

contraction(a::TensorTerm{W},b::Chain{V,G,<:Chain}) where {W,G,V} = contraction(Chain(a),b)
contraction(x::Chain{V,G,<:Chain},y::Single{V,G}) where {V,G} = value(y)*x[bladeindex(mdims(V),UInt(basis(y)))]
contraction(x::Chain{V,G,<:Chain},y::Submanifold{V,G}) where {V,G} = x[bladeindex(mdims(V),UInt(y))]
#contraction(a::Chain{V,L,<:Chain{V,G},N},b::Chain{V,G,<:Chain{V},M}) where {V,G,L,N,M} = Chain{V,G}(contraction.(Ref(a),value(b)))
contraction(x::Chain{V,L,<:Chain{V,G},N},y::Chain{V,G,<:Chain{V,L},N}) where {L,N,V,G} = Chain{V,G}(contraction_mat.(Ref(x),value(y)))
contraction_mat(x::Chain{W,L,<:Chain{V,G},N},y::Chain{V,G,T,N}) where {W,L,N,V,G,T} = Chain{V,G}(matmul(value(x),value(y)))
contraction(x::Chain{W,L,<:Chain{V,G},N},y::Chain{V,G,T,N}) where {W,L,N,V,G,T} = Chain{V,G}(matmul(value(x),value(y)))
contraction(x::Chain{W,L,<:Multivector{V},N},y::Chain{V,G,T,N}) where {W,L,N,V,G,T} = Multivector{V}(matmul(value(x),value(y)))
contraction(x::Multivector{W,<:Chain{V,G},N},y::Multivector{V,T,N}) where {W,N,V,G,T} = Chain{V,G}(matmul(value(x),value(y)))
contraction(x::Multivector{W,<:Multivector{V},N},y::Multivector{V,T,N}) where {W,N,V,T} = Multivector{V}(matmul(value(x),value(y)))
@inline @generated function matmul(x::Values{N,<:Single{V,G}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,:(@inbounds y[$i]*value(x[$i]))) for i ∈ list(1,N)]...)
end
@inline @generated function matmul(x::Values{N,<:Chain{V,G}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds x[$i][$j]*y[$i]) for i ∈ list(1,N)]...) for j ∈ list(1,binomial(mdims(V),G))]...)
end
@inline @generated function matmul(x::Values{N,<:Multivector{V}},y::Values{N}) where {N,V}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds value(x[$i])[$j]*y[$i]) for i ∈ list(1,N)]...) for j ∈ list(1,1<<mdims(V))]...)
end
@inline @generated function matmul(x::Values{N,<:Spinor{V}},y::Values{N}) where {N,V}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds value(x[$i])[$j]*y[$i]) for i ∈ list(1,N)]...) for j ∈ list(1,1<<(mdims(V)-1))]...)
end
@inline @generated function matmul(x::Values{N,<:AntiSpinor{V}},y::Values{N}) where {N,V}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds value(x[$i])[$j]*y[$i]) for i ∈ list(1,N)]...) for j ∈ list(1,1<<(mdims(V)-1))]...)
end
@inline @generated function matwedge(x::Values{N,<:Chain{V,G}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds x[$i][$j]∧y[$i]) for i ∈ list(1,N)]...) for j ∈ list(1,binomial(mdims(V),G))]...)
end
@inline @generated function matvee(x::Values{N,<:Chain{V,G}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds x[$i][$j]∨y[$i]) for i ∈ list(1,N)]...) for j ∈ list(1,binomial(mdims(V),G))]...)
end

contraction(x::Spinor{W,<:Spinor{V},N},y::Spinor{V,T,N}) where {W,N,V,T} = Spinor{V}(matmul(value(x),value(y)))
contraction(x::AntiSpinor{W,<:AntiSpinor{V},N},y::AntiSpinor{V,T,N}) where {W,N,V,T} = AntiSpinor{V}(matmul(value(x),value(y)))

contraction(a::Dyadic{V,<:Chain{V,1,<:Chain},<:Chain{V,1,<:Chain}} where V,b::TensorGraded) = sum(value(a.x).⊗(value(a.y).⋅b))
contraction(a::Dyadic{V,<:Chain{V,1,<:Chain}} where V,b::TensorGraded) = sum(value(a.x).⊗(a.y.⋅b))
contraction(a::Dyadic{V,T,<:Chain{V,1,<:Chain}} where {V,T},b::TensorGraded) = sum(a.x.⊗(value(a.y).⋅b))
contraction(a::Proj{V,<:Chain{W,1,<:Chain} where W} where V,b::TensorGraded) = sum(value(a.v).⊗(value(a.λ).*value(a.v).⋅b))
contraction(a::Proj{V,<:Chain{W,1,<:Chain{V,1}} where W},b::TensorGraded{V,1}) where V = sum(value(a.v).⊗(value(a.λ).*column(value(a.v).⋅b)))

+(a::Proj{V}...) where V = Proj{V}(Chain(Values(eigvec.(a)...)),Chain(Values(eigval.(a)...)))
+(a::Dyadic{V}...) where V = Proj(Chain(a...))
+(a::TensorNested{V}...) where V = Proj(Chain(Dyadic.(a)...))
plus(a::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W,b::TensorNested{V}) where V = +(value(a.v)...,b)
plus(a::TensorNested{V},b::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W) where V = +(a,value(b.v)...)
+(a::Proj{M,<:Chain{M,1,<:TensorNested{V}}} where M,b::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W) where V = +(value(a.v)...,value(b.v)...)
+(a::Proj{M,<:Chain{M,1,<:Chain{V}}} where M,b::Proj{W,<:Chain{W,1,<:Chain{V}}} where W) where V = Chain(Values(value(a.v)...,value(b.v)...))
#+(a::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W,b::TensorNested{V}) where V = +(b,Proj.(value(a.v),value(a.λ))...)
#+(a::TensorNested{V},b::Proj{W,<:Chain{W,1,<:TensorNested{V}}} where W) where V = +(a,value(b.v)...)

+(a::TensorNested,b::TensorNested,c::TensorNested...) = +(a+b,c...)
+(a::TensorNested) = a
-(a::TensorNested) = -1a
minus(a::TensorNested,b::TensorNested) = a+(-b)
*(a::Number,b::TensorNested{V}) where V = (a*One(V))*b
*(a::TensorNested{V},b::Number) where V = a*(b*One(V))
@inline ⟑(a::TensorGraded{V,0},b::TensorNested{V}) where V = b⋅a
@inline ⟑(a::TensorNested{V},b::TensorGraded{V,0}) where V = a⋅b
@inline ⟑(a::TensorGraded{V,0},b::Proj{V,<:Chain{V,1,<:TensorNested}}) where V = Proj{V}(a*b.v)
@inline ⟑(a::Proj{V,<:Chain{V,1,<:TensorNested}},b::TensorGraded{V,0}) where V = Proj{V}(a.v*b)

@inline ⟑(a::Chain,b::Chain{V,G,<:Chain{V,G}} where {V,G}) = contraction(a,b)
@inline ⟑(a::Chain{V,G,<:Chain{V,G}} where {V,G},b::Chain) = contraction(a,b)
@inline ⟑(a::Chain{V,G,<:Chain{V,G}},b::Chain{V,G,<:Chain{V,G}}) where {V,G} = contraction(a,b)
@inline ⟑(a::Chain{V,G,<:Chain{V,G}} where {V,G},b::TensorTerm) = contraction(a,b)
@inline ⟑(a::TensorGraded,b::Chain{V,G,<:Chain{V,G}} where {V,G}) = contraction(a,b)
@inline ⟑(a::Chain{V,G,<:Chain{V,G}} where {V,G},b::TensorNested) = contraction(a,b)
@inline ⟑(a::TensorNested,b::Chain{V,G,<:Chain{V,G}} where {V,G}) = contraction(a,b)

for (op,po) ∈ ((:∧,:.∧),(:∨,:.∨))
    @eval begin
        $op(a::DiagonalOperator{V,<:Chain{V,G}},b::Chain{V,G}) where {V,G} = Chain{V,G}($po(value(value(a)),value(b)))
        $op(a::DiagonalOperator{V,<:Chain{V,G}},b::DiagonalOperator{V,<:Chain{V,G}}) where {V,G} = DiagonalOperator(Chain{V,G}($po(value(value(a)),value(value(b)))))
        #$op(a::Endomorphism{W,<:TensorGraded},b::Endomorphism{V,<:Chain{V,G}}) where {W,V,G} = TensorOperator(Chain{V,G}($po(Ref(value(a)),value(value(b)))))
        $op(a::Endomorphism{W,<:TensorGraded},b::Endomorphism{V,<:Chain{V,G}}) where {W,V,G} = TensorOperator(Chain{V,G}($po(Ref(a),value(value(b)))))
        #$op(x::Chain{W,L,<:Chain{V,G},N},y::Chain{V,G,<:Chain{X,F} where {X,F},N}) where {W,L,N,V,G} = Chain{V,G}($po(Ref(x),value(y)))
    end
end
∧(a::Endomorphism{W,<:Chain},b::Chain{V,G}) where {W,V,G} = Chain{V,G}(matwedge(value(value(a)),value(b)))
∨(a::Endomorphism{W,<:Chain},b::Chain{V,G}) where {W,V,G} = Chain{V,G}(matvee(value(value(a)),value(b)))
for op ∈ (:complementright,:complementleft,:complementrighthodge,:complementlefthodge,:metric,:cometric)
    @eval $op(a::Endomorphism{V,<:Chain}) where V = map($op,a)
end

contraction(a::DiagonalOperator{V,<:Chain{V,G}},b::Chain{V,G}) where {V,G} = Chain{V,G}(value(value(a)).*value(b))
contraction(a::Chain{V,G},b::DiagonalOperator{V,<:Chain{V,G}}) where {V,G} = Chain{V,G}(value(a).*value(value(b)))
contraction(a::DiagonalOperator{V,<:Chain{V,G}},b::DiagonalOperator{V,<:Chain{V,G}}) where {V,G} = DiagonalOperator(Chain{V,G}(value(value(a)).*value(value(b))))

contraction(a::DiagonalOutermorphism{V},b::Chain{V,G}) where {V,G} = Chain{V,G}(value(value(a)(Val(G))).*value(b))
contraction(a::Chain{V,G},b::DiagonalOutermorphism{V}) where {V,G} = Chain{V,G}(value(a).*value(value(b)(Val(G))))
contraction(a::DiagonalOutermorphism{V},b::Spinor{V}) where V = Spinor{V}(value(even(value(a))).*value(b))
contraction(a::Spinor{V},b::DiagonalOutermorphism{V}) where V = Spinor{V}(value(a).*value(even(value(b))))
contraction(a::DiagonalOutermorphism{V},b::AntiSpinor{V}) where V = AntiSpinor{V}(value(even(value(a))).*value(b))
contraction(a::AntiSpinor{V},b::DiagonalOutermorphism{V}) where V = AntiSpinor{V}(value(a).*value(even(value(b))))
contraction(a::DiagonalOutermorphism{V},b::Multivector{V}) where V = Multivector{V}(value(value(a)).*value(b))
contraction(a::Multivector{V},b::DiagonalOutermorphism{V}) where V = Multivector{V}(value(a).*value(value(b)))
contraction(a::DiagonalOutermorphism{V},b::DiagonalOutermorphism{V}) where V = DiagonalOperator(Multivector{V}(value(value(a)).*value(value(b))))

contraction(a::Outermorphism{V},b::Endomorphism{V,<:Chain{V,G}}) where {V,G} = contraction(TensorOperator(a[G]),b)
contraction(a::Endomorphism{V,<:Chain{V,G}},b::Outermorphism{V}) where {V,G} = contraction(a,TensorOperator(b[G]))
contraction(a::DiagonalOutermorphism{V},b::Endomorphism{V,<:Chain{V,G}}) where {V,G} = contraction(DiagonalOperator(value(a)(Val(G))),b)
contraction(a::Endomorphism{V,<:Chain{V,G}},b::DiagonalOutermorphism{V}) where {V,G} = contraction(a,DiagonalOperator(value(b)(Val(G))))
contraction(a::DiagonalOperator{V,<:Chain{V,G}},b::Endomorphism{V,<:Chain{V,G}}) where {V,G} = contraction(TensorOperator(a),b)
contraction(a::Endomorphism{V,<:Chain{V,G}},b::DiagonalOperator{V,<:Chain{V,G}}) where {V,G} = TensorOperator(transpose(Chain{V,G}(value(value(a)).*value(value(b)))))

contraction(a::Outermorphism{V},b::TensorGraded{V,G}) where {V,G} = contraction(a[G],b)
contraction(a::TensorGraded{V,G},b::Outermorphism{V}) where {V,G} = contraction(a,b[G])
contraction(a::Outermorphism{V},b::Outermorphism{V}) where V = Outermorphism(contraction.(a.v,b.v))

@generated function contraction(a::Outermorphism{V},b::Spinor{V}) where V
    Expr(:call,:(Spinor{V}),Expr(:call,:Values,:(@inbounds value(b)[1]),[:(value(contraction((@inbounds a.v[$g]),b(Val($g))))...) for g ∈ evens(2,mdims(V))]...))
end
@generated function contraction(a::Outermorphism{V},b::AntiSpinor{V}) where V
    Expr(:call,:(AntiSpinor{V}),Expr(:call,:Values,[:(value(contraction((@inbounds a.v[$g]),b(Val($g))))...) for g ∈ evens(1,mdims(V))]...))
end
@generated function contraction(a::Outermorphism{V},b::Multivector{V}) where V
    Expr(:call,:(Multivector{V}),Expr(:call,:Values,:(@inbounds value(b)[1]),[:(value(contraction((@inbounds a.v[$g]),b(Val($g))))...) for g ∈ list(1,mdims(V))]...))
end

scalarcheck(x) = isscalar(x) ? value(scalar(x)) : x
for (op,args) ∈ ((:contraction,()),(:contraction_metric,(:g,)))
    @eval @generated function $op(x::Endomorphism{V,<:Chain{V,G,<:Chain{V,G,<:TensorGraded{W,L}}}},y::TensorGraded{W,L},$(args...)) where {W,L,V,G}
    Expr(:call,:TensorOperator,Expr(:call,:(Chain{V,G}),[Expr(:call,:(Chain{V,G}),
        [:(@inbounds scalarcheck($$op(x[$i][$j],y,$($args...)))) for i ∈ list(1,gdims(mdims(V),G))]...) for j ∈ list(1,gdims(mdims(V),G))]...))
    end
end

+(a::Endomorphism,b::DiagonalOperator) = a + TensorOperator(b)
+(a::DiagonalOperator,b::Endomorphism) = TensorOperator(a) + b
for op ∈ (:plus,:minus,:+,:-)
    @eval @generated function $op(a::Outermorphism{V},b::Outermorphism{V}) where V
        Expr(:call,:(Outermorphism{V}),Expr(:tuple,[:($$op(value(a)[$g],value(b)[$g])) for g ∈ list(1,mdims(V))]...))
    end
    for operator ∈ (:TensorOperator,:DiagonalOperator)
        @eval $op(a::$operator,b::$operator) = $operator($op(value(a),value(b)))
    end
end
for op ∈ (:⟑,:*)
    for type ∈ (:DiagonalOperator,:Outermorphism)
        @eval begin
            $op(a::$type,b::TensorAlgebra) = contraction(a,b)
            $op(a::TensorAlgebra,b::$type) = contraction(a,b)
            $op(a::$type,b::$type) = contraction(a,b)
        end
    end
end
for op ∈ (:*,:⟑,:contraction,:/)
    @eval begin
        $op(a::TensorAlgebra,b::TensorOperator) = $op(a,value(b))
        $op(a::TensorOperator,b::TensorAlgebra) = $op(value(a),b)
        $op(a::TensorOperator,b::TensorOperator) = TensorOperator($op(value(a),value(b)))
    end
end
for F ∈ Fields
    @eval begin
        *(a::F,b::Outermorphism{V}) where {V,F<:$F} = Outermorphism{V}(a.*value(b))
        *(a::Outermorphism{V},b::F) where {V,F<:$F} = Outermorphism{V}(value(a).*b)
        /(a::Outermorphism{V},b::F) where {V,F<:$F} = Outermorphism{V}(value(a)./b)
    end
    for operator ∈ (:TensorOperator,:DiagonalOperator)
        @eval begin
            *(a::F,b::$operator) where F<:$F = $operator(a*value(b))
            *(a::$operator,b::F) where F<:$F = $operator(value(a)*b)
            /(a::$operator,b::F) where F<:$F = $operator(value(a)/b)
        end
    end
end

Base.map(fn, x::DiagonalOperator) = DiagonalOperator(map(fn,value(x)))
Base.map(fn, x::TensorOperator{V,W,<:Chain{V,G}}) where {V,W,G} = TensorOperator(Chain{V,G}(map.(fn,value(value(x)))))
Base.map(fn, x::TensorOperator{V,W,<:Spinor}) where {V,W} = TensorOperator(Spinor{V}(map.(fn,value(value(x)))))
Base.map(fn, x::TensorOperator{V,W,<:AntiSpinor}) where {V,W} = TensorOperator(AntiSpinor{V}(map.(fn,value(value(x)))))
Base.map(fn, x::TensorOperator{V,W,<:Multivector}) where {V,W} = TensorOperator(Multivector{V}(map.(fn,value(value(x)))))

# dyadic identity element

for op ∈ (:(Base.:+),:(Base.:-))
    for tensor ∈ (:Projector,:Dyadic)
        @eval begin
            $op(a::A,b::B) where {A<:$tensor,B<:TensorAlgebra} = $op(Chain(a),b)
            $op(a::A,b::B) where {A<:TensorAlgebra,B<:$tensor} = $op(a,Chain(b))
            $op(t::LinearAlgebra.UniformScaling,g::$tensor) = $op(t,Chain(g))
            $op(g::$tensor,t::LinearAlgebra.UniformScaling) = $op(Chain(g),t)
        end
    end
end
Base.:+(g::Endomorphism,t::LinearAlgebra.UniformScaling) = t+g
Base.:+(t::LinearAlgebra.UniformScaling,g::Endomorphism) = TensorOperator(t+value(g))
Base.:-(g::Endomorphism,t::LinearAlgebra.UniformScaling) = TensorOperator(value(g)-t)
Base.:-(t::LinearAlgebra.UniformScaling,g::Endomorphism) = TensorOperator(t-value(g))
Base.:+(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling) where V = t+g
Base.:+(t::LinearAlgebra.UniformScaling{Bool},g::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1}(chainbasis(V).+value(g))
Base.:+(t::LinearAlgebra.UniformScaling,g::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1}(t.λ*chainbasis(V).+value(g))
Base.:-(t::LinearAlgebra.UniformScaling{Bool},g::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1}(chainbasis(V).-value(g))
Base.:-(t::LinearAlgebra.UniformScaling,g::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1}(t.λ*chainbasis(V).-value(g))
Base.:-(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling{Bool}) where V = Chain{V,1}(value(g).-(chainbasis(V)))
Base.:-(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling) where V = Chain{V,1}(value(g).-t.λ*(chainbasis(V)))

# representation

⊘(a::TensorOperator{V,W,<:Chain{V,G}},b::TensorAlgebra{W}) where {V,W,G} = TensorOperator(Chain{V,G}(value(value(a)) .⊘ Ref(b)))
for t ∈ (:Spinor,:AntiSpinor,:Multivector)
    @eval begin
        ⊘(a::TensorOperator{V,W,<:$t},b::TensorAlgebra{W}) where {V,W} = TensorOperator($t{V}(value(value(a)) .⊘ Ref(b)))
        @pure $t{V}(t::LinearAlgebra.UniformScaling) where V = t.λ*$t{V}(I)
    end
end
@pure Chain{V}(t::LinearAlgebra.UniformScaling) where V = Chain{V,1}(t)
@pure Chain{V,G}(t::LinearAlgebra.UniformScaling) where {V,G} = t.λ*Chain{V,G}(I)
@pure Chain{V,G}(t::LinearAlgebra.UniformScaling{Bool}) where {V,G} = TensorOperator(Chain{V,G}(chaindyad(V,G)))
@pure Spinor{V}(t::LinearAlgebra.UniformScaling{Bool}) where V = TensorOperator(Spinor{V}(evendyad(V)))
@pure AntiSpinor{V}(t::LinearAlgebra.UniformScaling{Bool}) where V = TensorOperator(AntiSpinor{V}(odddyad(V)))
@pure Multivector{V}(t::LinearAlgebra.UniformScaling{Bool}) where V = TensorOperator(Multivector{V}(fulldyad(V)))

@pure fulldyad(V) = Multivector.(fullbasis(V))
@pure fullbasis(V) = Λ(V).b

@pure chaindyad(V,G=Val(1)) = Chain.(chainbasis(V,G))
@pure chainbasis(V,G) = chainbasis(V,Val(G))
@pure function chainbasis(V,::Val{G}=Val(1)) where G
    N = mdims(V)
    r,b = binomsum(N,G),binomial(N,G)
    bas = Values{b,Submanifold{V,G}}(Λ(V).b[list(r+1,r+b)])
end

operator(t::TensorAlgebra,G::Int) = operator(t,Val(G))
function operator(t::TensorTerm{V},G::Val=Val(1)) where V
    isdiag(V) ? DiagonalOperator(operator(Chain(t),G)) : operator(Chain(t),G)
end
function operator(t::TensorAlgebra{V},g::Val{G}=Val(1)) where {V,G}
    TensorOperator(Chain{V,G}(chainbasis(V,g) .⊘ Ref(t)))
end

outermorphism(t::TensorAlgebra) = gradedoperator(t)
function gradedoperator(t::TensorTerm{V}) where V
    isdiag(V) ? outermorphism(operator(t)) : gradedoperator(Chain(t))
end
@generated function gradedoperator(t::TensorAlgebra{V}) where V
    Expr(:call,:(Outermorphism{V}),Expr(:tuple,[:(value(operator(t,Val($G)))) for G ∈ list(1,mdims(V))]...))
end

#=function operator(fun,V,::Val{G}=Val(1)) where G
    TensorOperator(Chain{V,G}(fun.(chainbasis(V,G))))
end
operator(fun,V,G::Int) = operator(fun,V,Val(G))
gradedoperator(fun,V) = outermorphism(operator(fun,V))=#
#gradedoperator(fun,V) = TensorOperator(Multivector{V}(fun.(Λ(V).b)))

@pure odddyad(V) = AntiSpinor.(oddbasis(V))
@pure oddbasis(V) = evenbasis(V,false)
@pure evendyad(V) = Spinor.(evenbasis(V))
@pure evenbasis(V,even::Bool) = evenbasis(V,Val(even))
@pure function evenbasis(V,::Val{even}=Val(true)) where even
    N = mdims(V)
    r,b = binomsum_set(N),gdimsall(N)
    vcat([Λ(V).b[list(r[g]+1,r[g]+b[g])] for g ∈ evens(even ? 1 : 2,N+1)]...)
end
#evenoperator(t::TensorAlgebra{V}) where V = TensorOperator(Spinor{V}(evenbasis(V) .⊘ Ref(t)))
#oddoperator(t::TensorAlgebra{V}) where V = TensorOperator(AntiSpinor{V}(oddbasis(V) .⊘ Ref(t)))

function getpairs(n,i)
    ind = indices(basis(!Λ(Submanifold(n)).b[i+1]))
    Expr(:call,:Values,[:(@inbounds value(x)[$j]-value(x)[$i]) for j ∈ ind]...)
end
nozero(x::T) where T = iszero(x) ? one(T) : x
getmult(n,i) = :(+(true,iszero.($(getpairs(n,i)))...))
getdiff(n,i) = :(*(nozero.($(getpairs(n,i)))...))
sylvester(X::TensorNested) = sylvester(eigvals(X))
@generated function sylvester(x::Chain{V,1}) where V
    Expr(:call,:(Chain{V}),getdiff.(mdims(V),list(1,mdims(V)))...)
end
eigmults(X::TensorNested) = eigmults(eigvals(X))
@generated function eigmults(x::Chain{V,1}) where V
    Expr(:call,:(Chain{V}),getmult.(mdims(V),list(1,mdims(V)))...)
end

function eigpolys(X::TensorNested{V}) where V
    if mdims(V)≠2
        eigpolys(eigvalsreal(X))
    else
        Chain{V}(eigpolys(X,Val(1)),eigpolys(X,Val(2)))
    end
end
@generated function eigpolys(x::Chain{V,1}) where V
    Expr(:call,:(Chain{V}),[:(eigpolys(x,Val($i))) for i ∈ list(1,mdims(V))]...)
end
eigpolys(X,G::Int) = eigpolys(X,Val(G))
eigpolys(::TensorAlgebra{V} where V,::Val{0}) = 1
function eigpolys(X::TensorNested{V},g::Val{G}) where {V,G}
    mdims(V)≠G ? sum(value(eigprods(X,g)))/binomial(mdims(V),G) : Real(det(X))
end
function eigpolys(x::Chain{V,1},g::Val{G}) where {V,G}
    mdims(V)≠G ? sum(value(eigprods(x,g)))/binomial(mdims(V),G) : prod(value(x))
end
eigpolys(X::Outermorphism,G::Val{1}) = scalar(TensorOperator(@inbounds value(x)[1]))
eigpolys(X::Endomorphism{V,<:Simplex} where V,::Val{1}) = scalar(X)
eigpolys(X::DiagonalMorphism,::Val{1}) = scalar(X)
eigpolys(X::DiagonalMorphism,G::Val) = eigpolys(value(X),G)
eigpolys(X::DiagonalMorphism) = eigpolys(value(X))
eigpolys(X::DiagonalOutermorphism,G::Val{1}) = scalar(DiagonalOperator(value(X)(G)))
eigpolys(X::DiagonalOutermorphism,G::Val) = eigpolys(value(X)(Val(1)),G)
eigpolys(X::DiagonalOutermorphism) = eigpolys(value(X)(Val(1)))

eigprods(X,G::Int) = eigprods(X,Val(G))
eigprods(X::TensorNested) = eigprods(eigvalsreal(X))
eigprods(X::TensorNested{V},g::Val{G}) where {V,G} = mdims(V)≠G ? eigprods(eigvalsreal(X),g) : det(X)
eigprods(X::TensorAlgebra{V},::Val{0}) where V = Chain{V,0}(1)
eigprods(x::Chain{V,1} where V,::Val{1}) = x
@generated function eigprods(x::Chain{V,1},::Val{G}) where {V,G}
    Expr(:call,:(Chain{V,G}),getprod.(mdims(V),list(1,binomial(mdims(V),G)),G)...)
end
@generated function eigprods(x::Chain{V,1,T}) where {V,T}
    N = mdims(V)
    Expr(:call,:(Multivector{V}),one(T),vcat([[getprod(N,i,G) for i ∈ list(1,binomial(N,G))] for G ∈ list(1,N)]...)...)
end
function getprod(n,i,g)
    ind = indices(basis(Λ(Submanifold(n)).b[i+binomsum(n,g)]))
    Expr(:call,:*,[:(@inbounds value(x)[$j]) for j ∈ ind]...)
end

for fun ∈ (:eigvecs,:eigvecsreal,:eigvecscomplex,:eigvals,:eigvalsreal,:eigvalscomplex,:eigen,:eigenreal,:eigencomplex)
    @eval begin
        $fun(X,i::Int) = $fun(X)[i]
        $fun(X,::Val{i}) where i = $fun(X)[i]
    end
end
eigvecs(X::TensorAlgebra) = eigvecs(operator(X))
eigvecs(X::SpectralOperator) = TensorOperator(X.v)
eigvecs(X::DiagonalMorphism) = DiagonalOperator(map(unit,value(X)))
eigvecs(X::DiagonalOutermorphism) = DiagonalOperator(map(unit,value(X)(Val(1))))
eigvecs(X::Endomorphism{V,<:Simplex}) where V = Endomorphism{V}(eigvecs(Matrix(X)))
eigvecs(X::Outermorphism) = (@inbounds eigvecs(TensorOperator(value(X)[1])))
eigvecsreal(X::DiagonalMorphism) = DiagonalOperator(map(unit,value(X)))
eigvecsreal(X::DiagonalOutermorphism) = DiagonalOperator(map(unit,value(X)(Val(1))))
eigvecsreal(X::Endomorphism{V,<:Simplex}) where V = Endomorphism{V}(map(Float64,eigvecs(Matrix(X))))
eigvecsreal(X::Outermorphism) = (@inbounds eigvecsreal(TensorOperator(value(X)[1])))
eigvecscomplex(X::TensorAlgebra) = eigvecscomplex(operator(X))
eigvecscomplex(X::DiagonalMorphism) = DiagonalOperator(map(Complex,map(unit,value(X))))
eigvecscomplex(X::DiagonalOutermorphism) = DiagonalOperator(map(Complex,map(unit,value(X)(Val(1)))))
eigvecscomplex(X::Endomorphism{V,<:Simplex}) where V = Endomorphism{V}(map(Complex,eigvecs(Matrix(X))))
eigvecscomplex(X::Outermorphism) = (@inbounds eigvecscomplex(TensorOperator(value(X)[1])))
eigvals(X::SpectralOperator) = X.λ
eigvals(X::TensorAlgebra) = eigvals(operator(X))
eigvals(X::Scalar{V}) where V = Chain{V}(value(abs2(X))*ones(Values{mdims(V)}))
function eigvals(X::T) where {V,G,T<:TensorGraded{V,G}}
    S = supermanifold(V)
    if (S==2 || S===S"2" || S===3 || S===S"3") && iseven(G)
        eigvals(T<:Chain ? Spinor(X) : Couple(X))
    else
        eigvals(operator(X))
    end
end
function eigvals(X::Union{<:Couple{V},<:Spinor{V}}) where V
    S = supermanifold(V)
    if S===2 || S===S"2"
        X2 = X*X
        re,sq = Real(scalar(X2)),sqrt(Real(abs2(imaginary(X2))))
        Chain{V}(Complex(re,-sq),Complex(re,sq))
    elseif S===3 || S===S"3"
        X2 = X*X
        re,sq = Real(scalar(X2)),sqrt(Real(abs2(imaginary(X2))))
        Chain{V}(Complex(re,-sq),Complex(re,sq),Real(abs2(X)))
    else
        eigvals(operator(X))
    end
end
function eigvals(X::TensorNested{V}) where V
    N = mdims(V)
    if N == 1
        X[1]
    elseif N < 5
        Chain{V}(monicroots(characteristic(X)))
    else
        Chain{V}(Values{N}(eigvals(eigen(Matrix(X)))))
    end
end
function eigvalsreal(X::TensorNested{V}) where V
    N = mdims(V)
    if N == 1
        X[1]
    elseif N < 5
        Chain{V}(monicrootsreal(characteristic(X)))
    else
        Chain{V}(Values{N,Float64}(eigvals(eigen(Matrix(X)))))
    end
end
eigvalscomplex(X::TensorAlgebra) = eigvalscomplex(operator(X))
eigvalscomplex(X::Scalar{V}) where V = Chain{V}(value(abs2(X))*Complex.(ones(Values{mdims(V)})))
function eigvalscomplex(X::T) where {V,G,T<:TensorGraded{V,G}}
    S = supermanifold(V)
    if (S==2 || S===S"2" || S===3 || S===S"3") && iseven(G)
        eigvalscomplex(T<:Chain ? Spinor(X) : Couple(X))
    else
        eigvalscomplex(operator(X))
    end
end
function eigvalscomplex(X::Union{<:Couple{V},<:Spinor{V}}) where V
    S = supermanifold(V)
    if S===2 || S===S"2"
        X2 = X*X
        re,sq = Real(scalar(X2)),sqrt(Real(abs2(imaginary(X2))))
        Chain{V}(Complex(re,-sq),Complex(re,sq))
    elseif S===3 || S===S"3"
        X2 = X*X
        re,sq = Real(scalar(X2)),sqrt(Real(abs2(imaginary(X2))))
        Chain{V}(Complex(re,-sq),Complex(re,sq),Real(abs2(X)))
    else
        eigvalscomplex(operator(X))
    end
end
function eigvalscomplex(X::TensorNested{V}) where V
    N = mdims(V)
    if N == 1
        Complex(X[1])
    elseif N < 5
        Chain{V}(monicrootscomplex(characteristic(X)))
    else
        Chain{V}(Complex.(Values{N}(eigvals(eigen(Matrix(X))))))
    end
end
function eigen(X::TensorNested{V}) where V
    eig = eigen(Matrix(X))
    Proj(value(Endomorphism{V}(eigvecs(eig))),Chain{V}(Values{mdims(V)}(eigvals(eig))))
end
function eigenreal(X::TensorNested{V}) where V
    eig = eigen(Matrix(X))
    Proj(value(Endomorphism{V}(map(Float64,eigvecs(eig)))),Chain{V}(Values{mdims(V),Float64}(eigvals(eig))))
end
function eigencomplex(X::TensorNested{V}) where V
    eig = eigen(Matrix(X))
    Proj(value(Endomorphism{V}(map(Complex,eigvecs(eig)))),Chain{V}(Complex.(Values{mdims(V)}(eigvals(eig)))))
end

characteristic(X::DiagonalOperator) = characteristic_exact(X)
function characteristic(X::Outermorphism{V}) where V
    mdims(V) < 5 ? (@inbounds characteristic(TensorOperator(value(X)[1]))) : characteristic_exact(X)
end
function characteristic(X::Endomorphism{V,<:Simplex}) where V
    N = mdims(V)
    if N == 1
        -X[1]
    elseif N == 2
        Chain{V}(Real(det(X)),-tr(X))
    elseif N == 3
        a0,a2 = -Real(det(X)),tr(X)
        Chain{V}(a0,(a2*a2-tr(X⋅X))/2,-a2)
    elseif N == 4
        a3,a0,X2 = tr(X),Real(det(X)),X⋅X
        a32,trX2 = a3*a3,tr(X2)
        a2,a1 = (a32 - trX2)/2,(a3*(a32 - 3trX2) + 2tr(X2⋅X))/-6
        Chain{V}(a0,a1,a2,-a3)
    else
        characteristic_exact(X)
    end
end

characteristic_exact(X::Endomorphism{V,<:Simplex} where V) = characteristic_exact(outermorphism(X))
characteristic_exact(X::DiagonalMorphism) = characteristic_exact(outermorphism(X))
@generated function characteristic_exact(X::DiagonalOutermorphism{V}) where V
    N = mdims(V)
    Expr(:block,:(out = sum.(getindex.(value(X),$(Val.(list(1,N)))))),
        Expr(:call,:(Chain{V}),[isodd(N-k) ? :(@inbounds out[$(N-k+1)]) : :(@inbounds -out[$(N-k+1)]) for k ∈ list(1,N)]...))
end
@generated function characteristic_exact(X::Outermorphism{V}) where V
    N = mdims(V)
    Expr(:block,:(out = Values(map(tr,value(X)))),
        Expr(:call,:(Chain{V}),[isodd(N-k) ? :(@inbounds out[$(N-k+1)]) : :(@inbounds -out[$(N-k+1)]) for k ∈ list(1,N)]...))
end

gerschgorin(x::DiagonalOperator{V}) where V = zero(Chain{V,1,Int})
gerschgorin(x::Outermorphism) = gerschgorin(@inbounds value(x)[1])
function gerschgorin(x::Endomorphism)
    sum.(map(value,map.(abs,value(value(transpose(x-DiagonalOperator(x)))))))
end

export 𝓛, Lie, LieBracket, LieDerivative, bracket

struct LieBracket end
struct LieDerivative{X}
    v::X
end
const 𝓛 = LieBracket()
const Lie = LieBracket()

Base.show(io::IO,::LieBracket) = print(io,"LieBracket[...]")
Base.getindex(::LieBracket,X...) = bracket(X...)
(::LieBracket)(X,Y...) = LieDerivative(bracket(X,Y...))
(::LieBracket)(X) = LieDerivative(X)
(X::LieDerivative)(Y...) = bracket(X.v,Y...)
(X::LieDerivative)(Y::LieDerivative) = LieDerivative(X.v(Y.v))
LieBracket(X...) = bracket(X...)
bracket(X) = X
bracket(X,Y) = X(Y) - Y(X)
bracket(X,Y,Z) = X(bracket(Y,Z)) + Y(bracket(Z,X)) + Z(bracket(X,Y))
bracket(W,X,Y,Z) = W(bracket(X,Y,Z)) + X(bracket(W,Z,Y)) + Y(bracket(W,X,Z)) + Z(bracket(W,Y,X))
bracket(V,W,X,Y,Z) = V(bracket(W,X,Y,Z)) + W(bracket(V,X,Z,Y)) + X(bracket(V,W,Y,Z)) + Y(bracket(V,W,Z,X)) + Z(bracket(V,W,X,Y))
@generated function bracket(X::Vararg{T,N}) where {N,T}
    Expr(:call,:+,[:($(isodd(i) ? :+ : :-)(X[$i](bracket($(vcat([j≠i ? [:(X[$j])] : [] for j ∈ list(1,N)]...)...))))) for i ∈ list(1,N)]...)
end

+(X::LieDerivative) = X
-(X::LieDerivative) = LieDerivative(-X.v)
+(X::LieDerivative,Y::LieDerivative) = LieDerivative(X.v+Y.v)
-(X::LieDerivative,Y::LieDerivative) = LieDerivative(X.v-Y.v)
*(n,X::LieDerivative) = LieDerivative(n*X.v)
*(X::LieDerivative,n) = LieDerivative(X.v*n)
/(X::LieDerivative,n) = LieDerivative(X.v/n)

# metric tensor

antimetrictensor(V,G) = compound(metrictensor(V),grade(V)-G)
antimetrictensor(V,::Val{G}=Val(1)) where G = compound(metrictensor(V),Val(grade(V)-G))
metrictensor(V::TensorBundle) = TensorOperator(map(Chain,value(metricdyad(V))))
metrictensor(V::Int) = TensorOperator(map(Chain,value(metricdyad(V))))
metricdyad(V::TensorBundle) = metricdyad(Submanifold(V))
metricdyad(V::Int) = metricdyad(Submanifold(V))
function metricdyad(V)
    if hasconformal(V)
        N = mdims(V)
        TensorOperator(Chain{V}(Single{V}((UInt(2),-1)),Single{V}((UInt(1),-1)),[Single{V}((UInt(1)<<i,1)) for i ∈ list(2,N-1)]...))
    else
        cayley(V,1,(x,y)->value(contraction(x,y)))
    end
end

applyf(f,mat::TensorOperator) = f.(value(value(mat)))

@pure metricextensor(V::TensorAlgebra) = Outermorphism(metrictensor(V))
const metriceven,metricodd = metricextensor,metricextensor
#metricfull(V) = TensorOperator(Multivector{V}(vcat(value.(map.(Multivector,value.(metrictensor.(V,list(0,mdims(V))))))...)))
#metriceven(V) = TensorOperator(Spinor{V}(vcat(applyf.(Spinor,metrictensor.(V,evens(0,mdims(V))))...)))
#metricodd(V) = TensorOperator(AntiSpinor{V}(vcat(applyf.(AntiSpinor,metrictensor.(V,evens(1,mdims(V))))...)))

struct MetricTensor{n,ℙ,g,Vars,Diff,Name} <: TensorBundle{n,ℙ,g,Vars,Diff,Name}
    @pure MetricTensor{N,M,S,F,D,L}() where {N,M,S,F,D,L} = new{N,M,S,F,D,L}()
end

@pure MetricTensor{N,M,S,F,D}() where {N,M,S,F,D} = MetricTensor{N,M,S,F,D,1}()
@pure MetricTensor{N,M,S}() where {N,M,S} = MetricTensor{N,M,S,0,0}()
@pure MetricTensor{N,M}(b::Values{N,<:Tuple}) where {N,M} = MetricTensor{N,M,metricsig(M,Values.(b))}()
@pure MetricTensor{N,M}(b::Values{N,<:Values}) where {N,M} = MetricTensor{N,M,metricsig(M,b)}()
@pure MetricTensor{N,M}(b::Values{N,<:Chain}) where {N,M} = MetricTensor{N,M,metricsig(M,value.(b))}()
@pure MetricTensor{N,M}(b::Values{N,<:AbstractVector}) where {N,M} = MetricTensor{N,M,metricsig(M,Values{N}.(b))}()
MetricTensor{N,M}(b::Vector) where {N,M} = MetricTensor{N,M}(Values(b...))
@pure MetricTensor(b::Tuple) = MetricTensor(Values(b))
@pure MetricTensor(b::Values{N}) where N = MetricTensor{N,0}(b)
MetricTensor(b::Values{N,<:Real}) where N = DiagonalForm(b)
MetricTensor(b::AbstractVector{<:Real}) = DiagonalForm(b)
MetricTensor(b::AbstractVector) = MetricTensor{length(b),0}(b)
MetricTensor(b::AbstractMatrix) = MetricTensor(getindex.(Ref(b),:,1:(size(b)[1])))
MetricTensor(b::Chain{V,G,<:Chain} where {V,G}) = MetricTensor(value(b))
MetricTensor(b::Chain) = DiagonalForm(value(b))
MetricTensor(b::TensorOperator) = MetricTensor(value(b))
MetricTensor(b...) = MetricTensor(b)

@pure Manifold(::Type{T}) where T<:MetricTensor = T()

construct_cache(:MetricTensor)
@pure metrictensor(b::Submanifold{V}) where V = isbasis(b) ? metrictensor(V) : TensorOperator(map(b,b(value(metrictensor(V)))))
@pure metrictensor(V,G) = compound(metrictensor(V),G)
@pure function metrictensor(V::MetricTensor{N,M,S} where N) where {M,S}
    out = Chain{Submanifold(V)}.(metrictensor_cache[S])
    TensorOperator(Chain{Submanifold(V)}(isdual(V) ? SUB(out) : out))
end
const metrictensor_cache = Values[]
@pure function metricsig(M,b::Values)
    a = dyadmode(M)>0 ? SUB(b) : b
    if a ∈ metrictensor_cache
        findfirst(x->x==a,metrictensor_cache)
    else
        push!(metrictensor_cache,a)
        length(metrictensor_cache)
    end
end

@pure DirectSum.getalgebra(V::MetricTensor) = DirectSum.getalgebra(Submanifold(V))

for t ∈ (Any,Integer)
    @eval @inline getindex(s::MetricTensor{N,M,S} where {N,M},i::T) where {S,T<:$t} = value(value(metrictensor(s))[i])
end
@inline getindex(vs::MetricTensor,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::MetricTensor,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::MetricTensor{N,M,S} where M,i::Colon) where {N,S} = Vector(value(metrictensor(vs)))

@pure Signature(V::MetricTensor{N,M,S,F,D}) where {N,M,S,F,D} = Signature{N,M,UInt(0),F,D}()

# anything array-like gets summarized e.g. 10-element Array{Int64,1}
Base.summary(io::IO, a::MetricTensor) = Base.array_summary(io, a, _axes(metrictensor(a)))

show(io::IO,M::MetricTensor) = Base.show(io,Submanifold(M))
function show(io::IO, ::MIME"text/plain", M::MetricTensor)
    X = display_matrix(value(metrictensor(M)))
    if isempty(X) && get(io, :compact, false)::Bool
        return show(io, X)
    end
    show_matrix(io,M,X)
end

(M::MetricTensor)(b::Int...) = Submanifold{M}(b)
(M::MetricTensor)(b::T) where T<:AbstractVector{Int} = Submanifold{M}(b)
(M::MetricTensor)(b::T) where T<:AbstractRange{Int} = Submanifold{M}(b)

isdiag(::MetricTensor) = false

function DirectSum.TensorBundle(b::Submanifold{V}) where V
    if isbasis(b)
        TensorBundle(V)
    elseif typeof(V) <: Int
        Signature(mdims(b))
    elseif typeof(V) <: Signature
        Signature(b)
    elseif typeof(V) <: DiagonalForm
        M = options(V)
        DiagonalForm{mdims(b),M,DirectSum.diagsig(M,Values(DirectSum.diagonalform(V)[indices(b)]...)),diffvars(V),diffmode(V)}()
    elseif typeof(V) <: MetricTensor
        M = options(V)
        MetricTensor{mdims(b),M,metricsig(M,value.(value(value(metrictensor(b))))),diffvars(V),diffmode(V)}()
    end
end

# InducedMetric

struct InducedMetric end
#=struct InducedMetric{V} <: TensorNested{V,Multivector{V}} end
metrictensor(::InducedMetric{V}) where V = metrictensor(V)
metricextensor(::InducedMetric{V}) where V = metricextensor(V)
Base.show(io::IO,x::InducedMetric) = show(io,metricextensor(x))=#

isinduced(x) = false
isinduced(::InducedMetric) = true
isinduced(x::Submanifold) = !isbasis(x)
isinduced(::TensorBundle) = true
isinduced(x::Type{<:Submanifold}) = !isbasis(x)
isinduced(::Type{<:TensorBundle}) = true
isinduced(::Type{<:InducedMetric}) = true

@inline Base.log(t::Real,g::InducedMetric) = Base.log(t)
@inline Base.log(t::Complex,g::InducedMetric) = Base.log(t)

