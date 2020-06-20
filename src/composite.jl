
#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export exph, log_fast, logh_fast

## exponential & logarithm function

@inline Base.expm1(t::SubManifold{V,0}) where V = Simplex{V}(ℯ-1)
@inline Base.expm1(t::T) where T<:TensorGraded{V,0} where V = Simplex{Manifold(t)}(AbstractTensors.expm1(value(T<:TensorTerm ? t : scalar(t))))

function Base.expm1(t::T) where T<:TensorAlgebra
    V = Manifold(t)
    S,term,f = t,(t^2)/2,norm(t)
    norms = SizedVector{3}(f,norm(term),f)
    k::Int = 3
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        term *= t/k
        @inbounds norms .= (norms[2],norm(term),ns)
        k += 1
    end
    return S
end

@eval @generated function Base.expm1(b::MultiVector{V,T}) where {V,T}
    loop = generate_loop_multivector(V,:term,:B,:*,:geomaddmulti!,geomaddmulti!_pre,:k)
    return quote
        B = value(b)
        sb,nb = scalar(b),AbstractTensors.norm(B)
        sb ≈ nb && (return Simplex{V}(AbstractTensors.expm1(value(sb))))
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        S = zeros(mvec(N,t))
        term = zeros(mvec(N,t))
        S .= B
        out .= value(b^2)/2
        norms = SizedVector{3}(nb,norm(out),norm(term))
        k::Int = 3
        @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
            S += out
            ns = norm(S)
            @inbounds ns ≈ norms[3] && break
            term .= out
            out .= 0
            # term *= b/k
            $(loop[2])
            @inbounds norms .= (norms[2],norm(out),ns)
            k += 1
        end
        return MultiVector{V}(S)
    end
end

@inline unabs!(t) = t
@inline unabs!(t::Expr) = (t.head == :call && t.args[1] == :abs) ? t.args[2] : t

function Base.exp(t::T) where T<:TensorGraded
    S,B = T<:SubManifold,T<:TensorTerm
    i = B ? basis(t) : t
    sq = i*i
    if B && isnull(t)
        return one(V)
    elseif isscalar(sq)
        hint = value(scalar(sq))
        isnull(hint) && (return 1+t)
        grade(t)==0 && (return Simplex{Manifold(t)}(AbstractTensors.exp(value(S ? t : scalar(t)))))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(t))))))
        hint<0 ? AbstractTensors.cos(θ)+t*(S ? AbstractTensors.sin(θ) : AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+t*(S ? AbstractTensors.sinh(θ) : AbstractTensors.:/(AbstractTensors.sinh(θ),θ))
    else
        return 1+expm1(t)
    end
end

function Base.exp(t::T,::Val{hint}) where T<:TensorGraded where hint
    S = T<:SubManifold
    i = T<:TensorTerm ? basis(t) : t
    sq = i*i
    if isscalar(sq)
        isnull(hint) && (return 1+t)
        grade(t)==0 && (return Simplex{Manifold(t)}(AbstractTensors.exp(value(S ? t : scalar(t)))))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(t))))))
        hint<0 ? AbstractTensors.cos(θ)+t*(S ? AbstractTensors.sin(θ) : AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+t*(S ? AbstractTensors.sinh(θ) : AbstractTensors.:/(AbstractTensors.sinh(θ),θ))
    else
        return 1+expm1(t)
    end
end

function Base.exp(t::MultiVector)
    st = scalar(t)
    mt = t-scalar(t)
    sq = mt*mt
    if isscalar(sq)
        hint = value(scalar(sq))
        isnull(hint) && (return AbstractTensors.exp(value(st))*(1+t))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(mt))))))
        return AbstractTensors.exp(value(st))*(hint<0 ? AbstractTensors.cos(θ)+mt*(AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+mt*(AbstractTensors.:/(AbstractTensors.sinh(θ),θ)))
    else
        return 1+expm1(t)
    end
end

function Base.exp(t::MultiVector,::Val{hint}) where hint
    st = scalar(t)
    mt = t-scalar(t)
    sq = mt*mt
    if isscalar(sq)
        isnull(hint) && (return AbstractTensors.exp(value(st))*(1+t))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(mt))))))
        return AbstractTensors.exp(value(st))*(hint<0 ? AbstractTensors.cos(θ)+mt*(AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+mt*(AbstractTensors.:/(AbstractTensors.sinh(θ),θ)))
    else
        return 1+expm1(t)
    end
end

function qlog(w::T,x::Int=10000) where T<:TensorAlgebra
    V = Manifold(w)
    w2,f = w^2,norm(w)
    prod = w*w2
    S,term = w,prod/3
    norms = SizedVector{3}(f,norm(term),f)
    k::Int = 5
    @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ x
        S += term
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        prod *= w2
        term = prod/k
        @inbounds norms .= (norms[2],norm(term),ns)
        k += 2
    end
    return 2S
end # http://www.netlib.org/cephes/qlibdoc.html#qlog

@eval @generated function qlog_fast(b::MultiVector{V,T,E},x::Int=10000) where {V,T,E}
    loop = generate_loop_multivector(V,:prod,:B,:*,:geomaddmulti!,geomaddmulti!_pre)
    return quote
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        f = norm(b)
        w2::MultiVector{V,T,E} = b^2
        B = value(w2)
        S = zeros(mvec(N,t))
        prod = zeros(mvec(N,t))
        term = zeros(mvec(N,t))
        S .= value(b)
        out .= value(b*w2)
        term .= out/3
        norms = SizedVector{3}(f,norm(term),f)
        k::Int = 5
        @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ x
            S += term
            ns = norm(S)
            @inbounds ns ≈ norms[3] && break
            prod .= out
            out .= 0
            # prod *= w2
            $(loop[2])
            term .= out/k
            @inbounds norms .= (norms[2],norm(term),ns)
            k += 2
        end
        S *= 2
        return MultiVector{V}(S)
    end
end

@inline Base.log(t::T) where T<:TensorAlgebra = qlog((t-1)/(t+1))
@inline Base.log1p(t::T) where T<:TensorAlgebra = qlog(t/(t+2))

for (qrt,n) ∈ ((:sqrt,2),(:cbrt,3))
    @eval begin
        @inline Base.$qrt(t::SubManifold{V,0} where V) = t
        @inline Base.$qrt(t::T) where T<:TensorGraded{V,0} where V = Simplex{V}($Sym.$qrt(value(T<:TensorTerm ? t : scalar(t))))
        @inline function Base.$qrt(t::T) where T<:TensorAlgebra
            isscalar(t) ? $qrt(scalar(t)) : exp(log(t)/$n)
        end
    end
end

## trigonometric

@inline Base.cosh(t::T) where T<:TensorGraded{V,0} where V = Simplex{Manifold(t)}(AbstractTensors.cosh(value(T<:TensorTerm ? t : scalar(t))))

function Base.cosh(t::T) where T<:TensorAlgebra
    V = Manifold(t)
    τ = t^2
    S,term = τ/2,(τ^2)/24
    f = norm(S)
    norms = SizedVector{3}(f,norm(term),f)
    k::Int = 6
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        term *= τ/(k*(k-1))
        @inbounds norms .= (norms[2],norm(term),ns)
        k += 2
    end
    return 1+S
end

@eval @generated function Base.cosh(b::MultiVector{V,T,E}) where {V,T,E}
    loop = generate_loop_multivector(V,:term,:B,:*,:geomaddmulti!,geomaddmulti!_pre,:(k*(k-1)))
    return quote
        sb,nb = scalar(b),norm(b)
        sb ≈ nb && (return Simplex{V}(AbstractTensors.cosh(value(sb))))
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        τ::MultiVector{V,T,E} = b^2
        B = value(τ)
        S = zeros(mvec(N,t))
        term = zeros(mvec(N,t))
        S .= value(τ)/2
        out .= value((τ^2))/24
        norms = SizedVector{3}(norm(S),norm(out),norm(term))
        k::Int = 6
        @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
            S += out
            ns = norm(S)
            @inbounds ns ≈ norms[3] && break
            term .= out
            out .= 0
            # term *= τ/(k*(k-1))
            $(loop[2])
            @inbounds norms .= (norms[2],norm(out),ns)
            k += 2
        end
        @inbounds S[1] += 1
        return MultiVector{V}(S)
    end
end

@inline Base.sinh(t::T) where T<:TensorGraded{V,0} where V = Simplex{Manifold(t)}(AbstractTensors.sinh(value(T<:TensorTerm ? t : scalar(t))))

function Base.sinh(t::T) where T<:TensorAlgebra
    V = Manifold(t)
    τ,f = t^2,norm(t)
    S,term = t,(t*τ)/6
    norms = SizedVector{3}(f,norm(term),f)
    k::Int = 5
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        term *= τ/(k*(k-1))
        @inbounds norms .= (norms[2],norm(term),ns)
        k += 2
    end
    return S
end

@eval @generated function Base.sinh(b::MultiVector{V,T,E}) where {V,T,E}
    loop = generate_loop_multivector(V,:term,:B,:*,:geomaddmulti!,geomaddmulti!_pre,:(k*(k-1)))
    return quote
        sb,nb = scalar(b),norm(b)
        sb ≈ nb && (return Simplex{V}(AbstractTensors.sinh(value(sb))))
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        τ::MultiVector{V,T,E} = b^2
        B = value(τ)
        S = zeros(mvec(N,t))
        term = zeros(mvec(N,t))
        S .= value(b)
        out .= value(b*τ)/6
        norms = SizedVector{3}(norm(S),norm(out),norm(term))
        k::Int = 5
        @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
            S += out
            ns = norm(S)
            @inbounds ns ≈ norms[3] && break
            term .= out
            out .= 0
            # term *= τ/(k*(k-1))
            $(loop[2])
            @inbounds norms .= (norms[2],norm(out),ns)
            k += 2
        end
        return MultiVector{V}(S)
    end
end

exph(t) = Base.cosh(t)+Base.sinh(t)

for (logfast,expf) ∈ ((:log_fast,:exp),(:logh_fast,:exph))
    @eval function $logfast(t::T) where T<:TensorAlgebra
        V = Manifold(t)
        term = zero(V)
        norm = SizedVector{2}(0.,0.)
        while true
            en = $expf(term)
            term -= 2(en-t)/(en+t)
            @inbounds norm .= (norm[2],norm(term))
            @inbounds norm[1] ≈ norm[2] && break
        end
        return term
    end
end

#=function log(t::T) where T<:TensorAlgebra
    V = Manifold(t)
    norms::Tuple = (norm(t),0)
    k::Int = 3
    τ = t-1
    if true #norms[1] ≤ 5/4
        prods = τ^2
        terms = TensorAlgebra{V}[τ,prods/2]
        norms = (norms[1],norm(terms[2]))
        while (norms[2]<norms[1] || norms[2]>1) && k ≤ 3000
            prods = prods*t
            push!(terms,prods/(k*(-1)^(k+1)))
            norms = (norms[2],norm(terms[end]))
            k += 1
        end
    else
        s = inv(t*inv(τ))
        prods = s^2
        terms = TensorAlgebra{V}[s,2prods]
        norms = (norm(terms[1]),norm(terms[2]))
        while (norms[2]<norms[1] || norms[2]>1) && k ≤ 3000
            prods = prods*s
            push!(terms,k*prods)
            norms = (norms[2],norm(terms[end]))
            k += 1
        end
    end
    return sum(terms[1:end-1])
end=#

function Cramer(N::Int)
    x,y = SVector{N}([Symbol(:x,i) for i ∈ 1:N]),SVector{N}([Symbol(:y,i) for i ∈ 1:N])
    xy = [:(($(x[1+i]),$(y[1+i])) = ($(x[i])∧t[$(1+i)],t[end-$i]∧$(y[i]))) for i ∈ 1:N-1]
    return x,y,xy
end

@generated function Base.:\(t::SVector{N,<:Chain{V,1}} where N,v::Chain{V,1}) where V
    N = ndims(V)-1 # paste this into the REPL for faster eval
    x,y,xy = Grassmann.Cramer(N)
    mid = [:($(x[i])∧v∧$(y[end-i])) for i ∈ 1:N-1]
    out = Expr(:call,:SVector,:(v∧$(y[end])),mid...,:($(x[end])∧v))
    return Expr(:block,:((x1,y1)=(t[1],t[end])),xy...,
        :(Chain{V,1}(getindex.($(Expr(:call,:./,out,:(t[1]∧$(y[end])))),1))))
end

@generated function Base.in(v::Chain{V,1},t::SVector{N,<:Chain{V,1}} where N) where V
    N = ndims(V)-1
    x,y,xy = Grassmann.Cramer(N)
    out = Expr(:call,:SVector,:(s==signbit((v∧$(y[end]))[1])),[:(s==signbit(($(x[i])∧v∧$(y[end-i]))[1])) for i ∈ 1:N-1]...,:(s==signbit(($(x[end])∧v)[1])))
    return Expr(:block,:((x1,y1)=(t[1],t[end])),xy...,:(s=signbit((t[1]∧$(y[end]))[1])),
        Expr(:call,:prod,out))
end

@generated function Base.inv(t::SVector{N,<:Chain{V,1}} where N) where V
    N = ndims(V)-1
    x,y,xy = Grassmann.Cramer(N)
    out = if iseven(N)
        Expr(:call,:SVector,y[end],[:($(y[end-i])∧$(x[i])) for i ∈ 1:N-1]...,x[end])
    else
        Expr(:call,:SVector,:(-$(y[end])),[:($(isodd(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ 1:N-1]...,x[end])
    end
    return Expr(:block,:((x1,y1)=(t[1],t[end])),xy...,:(_transpose(.⋆($(Expr(:call,:./,out,:((t[1]∧$(y[end]))[1])))))))
end

Base.:\(t::Chain{V,1,<:Chain{V,1}},v::Chain{V,1}) where V = value(t)\v
Base.in(v::Chain{V,1},t::Chain{V,1,<:Chain{V,1}}) where V = v ∈ value(t)
Base.inv(t::Chain{V,1,<:Chain{V,1}}) where V = inv(value(t))
INV(m::Chain{V,1,<:Chain{V,1}}) where V = Chain{V,1,Chain{V,1}}(inv(SMatrix(m)))

export detsimplex, initmesh, refinemesh, refinemesh!, select, submesh

detsimplex(m::Vector{<:Chain{V}}) where V = ∧(m)/factorial(ndims(V)-1)
detsimplex(m::ChainBundle) = detsimplex(value(m))
mean(m::Vector{<:Chain}) = sum(m)/length(m)
mean(m::T) where T<:SVector = sum(m)/length(m)
mean(m::Chain{V,1,<:Chain} where V) = mean(value(m))
barycenter(m::SVector{N,<:Chain}) where N = (s=sum(m);s/s[1])
barycenter(m::Vector{<:Chain}) = (s=sum(m);s/s[1])
barycenter(m::Chain{V,1,<:Chain} where V) = barycenter(value(m))
curl(m::SVector{N,<:Chain{V}} where N) where V = curl(Chain{V,1}(m))
curl(m::T) where T<:TensorAlgebra = Manifold(m)(∇)×m
LinearAlgebra.det(t::Chain{V,1,<:Chain} where V) = ∧(t)
LinearAlgebra.det(V::ChainBundle,m::Vector) = .∧(getindex.(Ref(V),value.(m)))
∧(m::Vector{<:Chain{V}}) where V = LinearAlgebra.det(V,m)
∧(m::ChainBundle) = LinearAlgebra.det(Manifold(m),value(m))
for op ∈ (:mean,:barycenter,:curl)
    ops = Symbol(op,:s)
    @eval begin
        export $op, $ops
        $ops(m::Vector{<:Chain{p}}) where p = $ops(m,p)
        @pure $ops(m::ChainBundle{p}) where p = $ops(m,p)
        @pure $ops(m,::SubManifold{p}) where p = $ops(m,p)
        @pure $ops(m,p) = $op.(getindex.(Ref(p),value.(value(m))))
    end
end

function initmesh(r::R) where R<:AbstractRange
    G = Λ(ℝ^2)
    p = ChainBundle(collect(r).*G.v2.+G.v1)
    e = ChainBundle(Chain{p(2),1,Int}.([(1,),(length(p),)]))
    t = ChainBundle(Chain{p,1,Int}.([(i,i+1) for i ∈ 1:length(p)-1]))
    return p,e,t
end

select(η,ϵ=sqrt(norm(η)^2/length(η))) = sort!(findall(x->x>ϵ,η))
refinemesh(g::R,args...) where R<:AbstractRange = (g,initmesh(g,args...)...)
function refinemesh!(::R,p::ChainBundle{W},e,t,η,_=nothing) where {W,R<:AbstractRange}
    p = points(t)
    x,T,V = value(p),value(t),Manifold(p)
    for i ∈ η
        push!(x,Chain{V,1}(SVector(1,(x[i+1][2]+x[i][2])/2)))
    end
    sort!(x,by=x->x[2]); submesh!(p)
    e[end] = Chain{p(2),1}(SVector(length(x)))
    for i ∈ length(t)+2:length(x)
        push!(T,Chain{p,1}(SVector{2,Int}(i-1,i)))
    end
end

const array_cache = (Array{T,2} where T)[]
function array(m::ChainBundle{V,G,T,B} where {V,G,T}) where B
    for k ∈ length(array_cache):B
        push!(array_cache,Array{Any,2}(undef,0,0))
    end
    isempty(array_cache[B]) && (array_cache[B] = [m[i][j] for i∈1:length(m),j∈1:ndims(m)])
    return array_cache[B]
end
function array!(m::ChainBundle{V,G,T,B} where {V,G,T}) where B
    length(array_cache) ≥ B && (array_cache[B] = Array{Any,2}(undef,0,0))
end

const submesh_cache = (Array{T,2} where T)[]
submesh(m) = [m[i][j] for i∈1:length(m),j∈2:ndims(Manifold(m))]
function submesh(m::ChainBundle{V,G,T,B} where {V,G,T}) where B
    for k ∈ length(submesh_cache):B
        push!(submesh_cache,Array{Any,2}(undef,0,0))
    end
    isempty(submesh_cache[B]) && (submesh_cache[B] = submesh(value(m)))
    return submesh_cache[B]
end
function submesh!(m::ChainBundle{V,G,T,B} where {V,G,T}) where B
    length(submesh_cache) ≥ B && (submesh_cache[B] = Array{Any,2}(undef,0,0))
end

for op ∈ (:div,:rem,:mod,:mod1,:fld,:fld1,:cld,:ldexp)
    @eval begin
        Base.$op(a::Chain{V,G,T},m::S) where {V,G,T,S} = Chain{V,G}($op.(value(a),m))
        Base.$op(a::MultiVector{V,T},m::S) where {T,V,S} = MultiVector{V}($op.(value(a),m))
    end
end
for op ∈ (:mod2pi,:rem2pi,:rad2deg,:deg2rad,:round)
    @eval begin
        Base.$op(a::Chain{V,G,T}) where {V,G,T} = Chain{V,G}($op.(value(a)))
        Base.$op(a::MultiVector{V,T}) where {V,T} = MultiVector{V}($op.(value(a)))
    end
end
Base.isfinite(a::Chain) = prod(isfinite.(value(a)))
Base.isfinite(a::MultiVector) = prod(isfinite.(value(a)))
Base.rationalize(t::Type,a::Chain{V,G,T};tol::Real=eps(T)) where {V,G,T} = Chain{V,G}(rationalize.(t,value(a),tol))
Base.rationalize(t::Type,a::MultiVector{V,T};tol::Real=eps(T)) where {V,T} = MultiVector{V}(rationalize.(t,value(a),tol))
Base.rationalize(t::T;kvs...) where T<:TensorAlgebra = rationalize(Int,t;kvs...)

*(A::SparseMatrixCSC{TA,S}, x::StridedVector{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
    (T = promote_type(TA, Chain{V,G,𝕂,X}); SparseArrays.mul!(similar(x, T, A.m), A, x, 1, 0))
*(A::SparseMatrixCSC{TA,S}, B::StridedMatrix{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
    (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(B, T, (A.m, size(B, 2))), A, B, 1, 0))
*(adjA::LinearAlgebra.Adjoint{<:Any,<:SparseMatrixCSC{TA,S}}, x::StridedVector{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
    (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(x, T, size(adjA, 1)), adjA, x, 2, 0))
*(transA::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC{TA,S}}, x::StridedVector{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
    (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(x, T, size(transA, 1)), transA, x, 1, 0))
if VERSION >= v"1.4"
    *(adjA::LinearAlgebra.Adjoint{<:Any,<:SparseMatrixCSC{TA,S}}, B::SparseArrays.AdjOrTransStridedOrTriangularMatrix{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
        (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(B, T, (size(adjA, 1), size(B, 2))), adjA, B, 1, 0))
    *(transA::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC{TA,S}}, B::SparseArrays.AdjOrTransStridedOrTriangularMatrix{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
        (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(B, T, (size(transA, 1), size(B, 2))), transA, B, 1, 0))
end

@generated function StaticArrays._diff(::Size{S}, a::SVector{Q,<:Chain}, ::Val{D}) where {S,D,Q}
    N = length(S)
    Snew = ([n==D ? S[n]-1 : S[n] for n = 1:N]...,)

    exprs = Array{Expr}(undef, Snew)
    itr = [1:n for n = Snew]

    for i1 = Base.product(itr...)
        i2 = copy([i1...])
        i2[D] = i1[D] + 1
        exprs[i1...] = :(a[$(i2...)] - a[$(i1...)])
    end

    return quote
        Base.@_inline_meta
        T = eltype(a)
        @inbounds return similar_type(a, T, Size($Snew))(tuple($(exprs...)))
    end
end

Base.map(fn, x::MultiVector{V}) where V = MultiVector{V}(map(fn, value(x)))
Base.map(fn, x::Chain{V,G}) where {V,G} = Chain{V,G}(map(fn,value(x)))
Base.map(fn, x::Simplex{V,G,B}) where {V,G,B} = fn(value(x))*B

import Random: SamplerType, AbstractRNG
Base.rand(::AbstractRNG,::SamplerType{Chain}) = rand(Chain{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Chain{V}}) where V = rand(Chain{V,rand(0:ndims(V))})
Base.rand(::AbstractRNG,::SamplerType{Chain{V,G}}) where {V,G} = Chain{V,G}(DirectSum.orand(svec(ndims(V),G,Float64)))
Base.rand(::AbstractRNG,::SamplerType{Chain{V,G,T}}) where {V,G,T} = Chain{V,G}(rand(svec(ndims(V),G,T)))
Base.rand(::AbstractRNG,::SamplerType{Chain{V,G,T} where G}) where {V,T} = rand(Chain{V,rand(0:ndims(V)),T})
Base.rand(::AbstractRNG,::SamplerType{MultiVector}) = rand(MultiVector{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{MultiVector{V}}) where V = MultiVector{V}(DirectSum.orand(svec(ndims(V),Float64)))
Base.rand(::AbstractRNG,::SamplerType{MultiVector{V,T}}) where {V,T} = MultiVector{V}(rand(svec(ndims(V),T)))
