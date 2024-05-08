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

# Dyadic

export TensorNested
abstract type TensorNested{V,T} <: Manifold{V,T} end

transpose_row(t::Values{N,<:Chain{V}},i,W=V) where {N,V} = Chain{W,1}(getindex.(t,i))
transpose_row(t::FixedVector{N,<:Chain{V}},i,W=V) where {N,V} = Chain{W,1}(getindex.(t,i))
transpose_row(t::Chain{V,1,<:Chain},i) where V = transpose_row(value(t),i,V)
@generated _transpose(t::Values{N,<:Chain{V,1}},W=V) where {N,V} = :(Chain{V,1}(transpose_row.(Ref(t),$(list(1,mdims(V))),W)))
@generated _transpose(t::FixedVector{N,<:Chain{V,1}},W=V) where {N,V} = :(Chain{V,1}(transpose_row.(Ref(t),$(list(1,mdims(V))),W)))
Base.transpose(t::Chain{V,1,<:Chain{V,1}}) where V = _transpose(value(t))
Base.transpose(t::Chain{V,1,<:Chain{W,1}}) where {V,W} = _transpose(value(t),V)

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

display_matrix(m::Chain{V,G,<:TensorGraded{W,G}}) where {V,G,W} = vcat(transpose([V,chainbasis(V,G)...]),hcat(chainbasis(W,G),matrix(m)))
display_matrix(m::TensorGraded{V,G,<:Spinor{W}}) where {V,G,W} = vcat(transpose([V,evenbasis(V)...]),hcat(evenbasis(W),matrix(m)))
display_matrix(m::TensorGraded{V,G,<:AntiSpinor{W}}) where {V,G,W} = vcat(transpose([V,oddbasis(V)...]),hcat(oddbasis(W),matrix(m)))
display_matrix(m::TensorAlgebra{V,<:Multivector{W}}) where {V,W} = vcat(transpose([V,fullbasis(V)...]),hcat(fullbasis(W),matrix(m)))
for (pinor,bas) ∈ ((:Spinor,:evenbasis),(:AntiSpinor,:oddbasis),(:Multivector,:fullbasis))
    @eval display_matrix(m::$pinor{V,<:TensorAlgebra{W}}) where {V,W} = vcat(transpose([V,$bas(V)...]),hcat($bas(W),matrix(m)))
end

export Projector, Dyadic, Proj

struct Projector{V,T,Λ} <: TensorNested{V,T}
    v::T
    λ::Λ
    Projector{V,T,Λ}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
    Projector{V,T}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
    Projector{V}(v::T,λ::Λ=1) where {T<:Manifold{V},Λ} where V = new{DirectSum.submanifold(V),T,Λ}(v,λ)
end

const Proj = Projector

Proj(v::T,λ=1) where T<:TensorGraded{V} where V = Proj{V}(v/abs(v),λ)
Proj(v::Chain{W,1,<:Chain{V}},λ=1) where {V,W} = Proj{V}(Chain(value(v)./abs.(value(v))),λ)
#Proj(v::Chain{V,1,<:TensorNested},λ=1) where V = Proj{V}(v,λ)

(P::Projector)(x) = contraction(P,x)

getindex(P::Proj,i::Int,j::Int) = P.v[i]*P.v[j]
getindex(P::Proj{V,<:Chain{W,1,<:Chain}} where {V,W},i::Int,j::Int) = sum(column(P.v,i).*column(P.v,j))
#getindex(P::Proj{V,<:Chain{V,1,<:TensorNested}} where V,i::Int,j::Int) = sum(getindex.(value(P.v),i,j))

Leibniz.extend_parnot(Projector)

show(io::IO,P::Proj{V,T,Λ}) where {V,T,Λ<:Real} = print(io,isone(P.λ) ? "" : P.λ,"Proj(",P.v,")")
show(io::IO,P::Proj{V,T,Λ}) where {V,T,Λ} = print(io,"(",P.λ,")Proj(",P.v,")")

#Chain{V}(P::Proj{V,T}) where {V,T<:Chain{V,1,<:TensorNested}} = sum(Chain.(value(P.v)))
Chain{V}(P::Proj{V,T}) where {V,T<:Simplex{V}} = sum(outer.(value(P.v).*value(P.λ),P.v))
Chain{V}(P::Proj{V}) where V = outer(P.v*P.λ,P.v)
Chain(P::Proj{V}) where V = Chain{V}(P)

struct Dyadic{V,X,Y} <: TensorNested{V,X}
    x::X
    y::Y
    Dyadic{V,X,Y}(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = new{DirectSum.submanifold(V),X,Y}(x,y)
    Dyadic{V}(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = new{DirectSum.submanifold(V),X,Y}(x,y)
end

Dyadic(x::X,y::Y) where {X<:TensorGraded,Y<:TensorGraded{V}} where V = Dyadic{V}(x,y)
Dyadic(P::Projector) = Dyadic(P.v,P.v)
Dyadic(D::Dyadic) = D

(P::Dyadic)(x) = contraction(P,x)

getindex(P::Dyadic,i::Int,j::Int) = P.x[i]*P.y[j]

show(io::IO,P::Dyadic) = print(io,"(",P.x,")⊗(",P.y,")")

Chain{V}(P::Dyadic{V}) where V = outer(P.x,P.y)
Chain(P::Dyadic{V}) where V = Chain{V}(P)

export TensorOperator, Endomorphism

struct TensorOperator{V,W,T<:TensorAlgebra{V,<:TensorAlgebra{W}}} <: TensorNested{V,T}
    v::T
    TensorOperator{V,W}(t::T) where {V,W,T<:TensorAlgebra{V,<:TensorAlgebra{W}}} = new{V,W,T}(t)
    TensorOperator{V}(t::T) where {V,W,T<:TensorAlgebra{V,<:TensorAlgebra{W}}} = new{V,W,T}(t)
    TensorOperator(t::T) where {V,W,T<:TensorAlgebra{V,<:TensorAlgebra{W}}} = new{V,W,T}(t)
end

Endomorphism{V,T<:TensorAlgebra{V,<:TensorAlgebra{V}}} = TensorOperator{V,V,T}

value(t::TensorOperator) = t.v
matrix(m::TensorOperator) = matrix(value(m))
getindex(t::TensorOperator,i::Int,j::Int) = value(value(t.v)[j])[i]

for op ∈ (:(Base.inv),)
    @eval $op(t::Endomorphism{V,<:Chain}) where V = TensorOperator($op(value(t)))
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

# dyadic products

export outer

outer(a::Leibniz.Derivation,b::Chain{V,1}) where V= outer(V(a),b)
outer(a::Chain{W},b::Leibniz.Derivation{T,1}) where {W,T} = outer(a,W(b))
outer(a::Chain{W},b::Chain{V,1}) where {W,V} = Chain{V,1}(a.*value(b))

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

contraction(a::Chain{W},b::Chain{V,G,<:Chain}) where {W,G,V} = Chain{V,G}(value.(Single.(Ref(a).⋅value(b))))
contraction(a::Chain{W,L,<:Chain,N},b::Chain{V,G,<:Chain{W,L},M}) where {W,L,G,V,N,M} = Chain{V,G}(value.(Ref(a).⋅value(b)))
contraction(a::Multivector{W,<:Multivector},b::Multivector{V,<:Multivector{W}}) where {W,V} = Multivector{V}(column(Ref(a).⋅value(b)))

contraction(a::Submanifold{W},b::Chain{V,G,<:Chain}) where {W,G,V} = Chain{V,G}(column(Ref(a).⋅value(b)))
contraction(a::Single{W},b::Chain{V,G,<:Chain}) where {W,G,V} = Chain{V,G}(column(Ref(a).⋅value(b)))
contraction(x::Chain{V,G,<:Chain},y::Single{V,G}) where {V,G} = value(y)*x[bladeindex(mdims(V),UInt(basis(y)))]
contraction(x::Chain{V,G,<:Chain},y::Submanifold{V,G}) where {V,G} = x[bladeindex(mdims(V),UInt(y))]
contraction(a::Chain{V,L,<:Chain{V,G},N},b::Chain{V,G,<:Chain{V},M}) where {V,G,L,N,M} = Chain{V,G}(contraction.(Ref(a),value(b)))
contraction(x::Chain{V,L,<:Chain{V,G},N},y::Chain{V,G,<:Chain{V,L},N}) where {L,N,V,G} = Chain{V,G}(contraction.(Ref(x),value(y)))
contraction(x::Chain{W,L,<:Chain{V,G},N},y::Chain{V,G,T,N}) where {W,L,N,V,G,T} = Chain{V,G}(matmul(value(x),value(y)))
contraction(x::Chain{W,L,<:Multivector{V},N},y::Chain{V,G,T,N}) where {W,L,N,V,G,T} = Multivector{V}(matmul(value(x),value(y)))
contraction(x::Multivector{W,<:Chain{V,G},N},y::Multivector{V,T,N}) where {W,N,V,G,T} = Chain{V,G}(matmul(value(x),value(y)))
contraction(x::Multivector{W,<:Multivector{V},N},y::Multivector{V,T,N}) where {W,N,V,T} = Multivector{V}(matmul(value(x),value(y)))
@inline @generated function matmul(x::Values{N,<:Single{V,G}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,:(@inbounds y[$i]*value(x[$i]))) for i ∈ list(1,N)]...)
end
@inline @generated function matmul(x::Values{N,<:Chain{V,G}},y::Values{N}) where {N,V,G}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds y[$i]*x[$i][$j]) for i ∈ list(1,N)]...) for j ∈ list(1,binomial(mdims(V),G))]...)
end
@inline @generated function matmul(x::Values{N,<:Multivector{V}},y::Values{N}) where {N,V}
    Expr(:call,:Values,[Expr(:call,:+,[:(@inbounds y[$i]*value(x[$i])[$j]) for i ∈ list(1,N)]...) for j ∈ list(1,1<<mdims(V))]...)
end

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
@inline ⟑(a::Chain{V,G,<:Chain{V,G}} where {V,G},b::TensorTerm) = contraction(a,b)
@inline ⟑(a::TensorGraded,b::Chain{V,G,<:Chain{V,G}} where {V,G}) = contraction(a,b)
@inline ⟑(a::Chain{V,G,<:Chain{V,G}} where {V,G},b::TensorNested) = contraction(a,b)
@inline ⟑(a::TensorNested,b::Chain{V,G,<:Chain{V,G}} where {V,G}) = contraction(a,b)

for op ∈ (:*,:⟑,:contraction,:plus,:minus,:+,:-,:/)
    @eval begin
        $op(a::TensorAlgebra,b::TensorOperator) = $op(a,value(b))
        $op(a::TensorOperator,b::TensorAlgebra) = $op(value(a),b)
        $op(a::TensorOperator,b::TensorOperator) = TensorOperator($op(value(a),value(b)))
    end
end
for F ∈ Fields
    @eval begin
        *(a::F,b::TensorOperator) where F<:$F = TensorOperator(a*value(b))
        *(a::TensorOperator,b::F) where F<:$F = TensorOperator(value(a)*b)
    end
end

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
Base.:+(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling) where V = t+g
Base.:+(t::LinearAlgebra.UniformScaling{Bool},g::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1}(chainbasis(V).+value(g))
Base.:+(t::LinearAlgebra.UniformScaling,g::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1}(t.λ*chainbasis(V).+value(g))
Base.:-(t::LinearAlgebra.UniformScaling{Bool},g::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1}(chainbasis(V).-value(g))
Base.:-(t::LinearAlgebra.UniformScaling,g::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1}(t.λ*chainbasis(V).-value(g))
Base.:-(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling{Bool}) where V = Chain{V,1}(value(g).-(chainbasis(V)))
Base.:-(g::Chain{V,1,<:Chain{V,1}},t::LinearAlgebra.UniformScaling) where V = Chain{V,1}(value(g).-t.λ*(chainbasis(V)))

# representation

export operator, gradedoperator, evenoperator, oddoperator

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

@pure chaindyad(V,G) = Chain.(chainbasis(V,G))
@pure function chainbasis(V,G=1)
    N = mdims(V)
    r,b = binomsum(N,G),binomial(N,G)
    bas = Values{b,Submanifold{V,G}}(Λ(V).b[list(r+1,r+b)])
end

function operator(t::TensorAlgebra{V},::Val{G}=Val(1)) where {V,G}
    TensorOperator(Chain{V,G}(chainbasis(V,G) .⊘ Ref(t)))
end
operator(t::TensorAlgebra,G::Int) = operator(t,Val(G))
gradedoperator(t::TensorAlgebra{V}) where V = TensorOperator(Multivector{V}(Λ(V).b .⊘ Ref(t)))

function operator(fun,V,::Val{G}=Val(1)) where G
    TensorOperator(Chain{V,G}(fun.(chainbasis(V,G))))
end
operator(fun,V,G::Int) = operator(fun,V,Val(G))
gradedoperator(fun,V) = TensorOperator(Multivector{V}(fun.(Λ(V).b)))

@pure odddyad(V) = AntiSpinor.(oddbasis(V))
@pure oddbasis(V) = evenbasis(V,false)
@pure evendyad(V) = Spinor.(evenbasis(V))
@pure function evenbasis(V,even=true)
    N = mdims(V)
    r,b = binomsum_set(N),binomial_set(N)
    vcat([Λ(V).b[list(r[g]+1,r[g]+b[g])] for g ∈ evens(even ? 1 : 2,N+1)]...)
end
evenoperator(t::TensorAlgebra{V}) where V = TensorOperator(Spinor{V}(evenbasis(V) .⊘ Ref(t)))
oddoperator(t::TensorAlgebra{V}) where V = TensorOperator(AntiSpinor{V}(oddbasis(V) .⊘ Ref(t)))
