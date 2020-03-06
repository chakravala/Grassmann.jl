
#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export exph, log_fast, logh_fast

## exponential & logarithm function

@inline Base.expm1(t::SubManifold{V,0}) where V = Simplex{V}(ℯ-1)
@inline Base.expm1(t::T) where T<:TensorGraded{V,0} where V = Simplex{V}(AbstractTensors.expm1(value(T<:TensorTerm ? t : scalar(t))))

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

function Base.exp(t::T) where T<:TensorGraded{V,G} where {V,G}
    S = T<:SubManifold
    i = T<:TensorTerm ? basis(t) : t
    sq = i*i
    if isscalar(sq)
        hint = value(scalar(sq))
        isnull(hint) && (return 1+t)
        G==0 && (return Simplex{V}(AbstractTensors.exp(value(S ? t : scalar(t)))))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(t))))))
        hint<0 ? AbstractTensors.cos(θ)+t*(S ? AbstractTensors.sin(θ) : AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+t*(S ? AbstractTensors.sinh(θ) : AbstractTensors.:/(AbstractTensors.sinh(θ),θ))
    else
        return 1+expm1(t)
    end
end

function Base.exp(t::T,::Val{hint}) where T<:TensorGraded{V,G} where {V,G,hint}
    S = T<:SubManifold
    i = T<:TensorTerm ? basis(t) : t
    sq = i*i
    if isscalar(sq)
        isnull(hint) && (return 1+t)
        G==0 && (return Simplex{V}(AbstractTensors.exp(value(S ? t : scalar(t)))))
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

@inline Base.cosh(t::T) where T<:TensorGraded{V,0} where V = Simplex{V}(AbstractTensors.cosh(value(T<:TensorTerm ? t : scalar(t))))

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

@inline Base.sinh(t::T) where T<:TensorGraded{V,0} where V = Simplex{V}(AbstractTensors.sinh(value(T<:TensorTerm ? t : scalar(t))))

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

export detsimplex, initmesh

detsimplex(m::Vector{Chain{V,G,T,X}} where {G,T,X}) where V = ∧(m)/factorial(ndims(V)-1)
mean(m::Vector{Chain{V,G,T,X}} where {V,G,T,X}) = sum(m)/length(m)
mean(m::T) where T<:SVector = sum(m)/length(m)
barycenter(m::SVector{N,Chain{V,G,T,X}} where {V,G,T,X}) where N = (s=sum(m);s/s[1])
barycenter(m::Vector{Chain{V,G,T,X}} where {V,G,T,X}) = (s=sum(m);s/s[1])
curl(m::SVector{N,Chain{V,G,T,X}} where {N,G,T,X}) where V = curl(Chain{V,1}(m))
curl(m::T) where T<:TensorAlgebra = Manifold(m)(∇)×m
for op ∈ (:∧,:detsimplex)
    @eval @pure $op(m::ChainBundle) = ChainBundle($op(value(m)))
end
for op ∈ (:mean,:barycenter,:curl)
    ops = Symbol(op,:s)
    @eval begin
        export $op, $ops
        @pure $ops(m::ChainBundle{p}) where p = $ops(m,p)
        @pure $ops(m::ChainBundle,::SubManifold{p}) where p = $ops(m,p)
        @pure $ops(m::ChainBundle,p) = [$op(p[k]) for k ∈ value.(value(m))]
    end
end

function initmesh(r::T) where T<:AbstractRange
    G = Λ(ℝ^2)
    p = ChainBundle(collect(r).*G.v2.+G.v1)
    e = ChainBundle(Chain{p(2),1,Int}.([(1,),(length(p),)]))
    t = ChainBundle(Chain{p,1,Int}.([(i,i+1) for i ∈ 1:length(p)-1]))
    return p,e,t
end

const array_cache = (Array{T,2} where T)[]
function array(m::ChainBundle{V,G,T,B} where {V,G,T}) where B
    for k ∈ length(array_cache):B
        push!(array_cache,Array{Any,2}(undef,0,0))
    end
    isempty(array_cache[B]) && (array_cache[B] = [m[i][j] for i∈1:length(m),j∈1:ndims(Manifold(m))])
    return array_cache[B]
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
