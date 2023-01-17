
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

export exph, log_fast, logh_fast

## exponential & logarithm function

@inline Base.expm1(t::Submanifold{V,0}) where V = Single{V}(ℯ-1)
@inline Base.expm1(t::T) where T<:TensorGraded{V,0} where V = Single{Manifold(t)}(AbstractTensors.expm1(value(T<:TensorTerm ? t : scalar(t))))

Base.expm1(t::Chain) = expm1(Multivector(t))
function Base.expm1(t::T) where T<:TensorAlgebra
    V = Manifold(t)
    if T<:Couple
        B = basis(t); BB = value(B*B)
        if BB == -1
            return Couple{V,B}(expm1(t.v))
        end
    end
    S,term,f = t,(t^2)/2,norm(t)
    norms = FixedVector{3}(f,norm(term),f)
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

@eval @generated function Base.expm1(b::Multivector{V,T}) where {V,T}
    loop = generate_loop_multivector(V,:term,:B,:*,:geomaddmulti!,geomaddmulti!_pre,:k)
    return quote
        B = value(b)
        sb,nb = scalar(b),AbstractTensors.norm(B)
        sb ≈ nb && (return Single{V}(AbstractTensors.expm1(value(sb))))
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        S = zeros(mvec(N,t))
        term = zeros(mvec(N,t))
        S .= B
        out .= value(b^2)/2
        norms = FixedVector{3}(nb,norm(out),norm(term))
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
        return Multivector{V}(S)
    end
end

@eval @generated function Base.expm1(b::Spinor{V,T}) where {V,T}
    loop = generate_loop_spinor(V,:term,:B,:*,:geomaddspin!,geomaddspin!_pre,:k)
    return quote
        B = value(b)
        sb,nb = scalar(b),AbstractTensors.norm(B)
        sb ≈ nb && (return Single{V}(AbstractTensors.expm1(value(sb))))
        $(insert_expr(loop[1],:mvecs,:T,Float64)...)
        S = zeros(mvecs(N,t))
        term = zeros(mvecs(N,t))
        S .= B
        out .= value(b^2)/2
        norms = FixedVector{3}(nb,norm(out),norm(term))
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
        return Spinor{V}(S)
    end
end

function Base.exp(t::Multivector)
    st = scalar(t)
    mt = t-st
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

function Base.exp(t::Spinor)
    st = scalar(t)
    mt = t-st
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

function Base.exp(t::Multivector,::Val{hint}) where hint
    st = scalar(t)
    mt = t-st
    sq = mt*mt
    if isscalar(sq)
        isnull(hint) && (return AbstractTensors.exp(value(st))*(1+t))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(mt))))))
        return AbstractTensors.exp(value(st))*(hint<0 ? AbstractTensors.cos(θ)+mt*(AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+mt*(AbstractTensors.:/(AbstractTensors.sinh(θ),θ)))
    else
        return 1+expm1(t)
    end
end

function Base.exp(t::Couple{V,B}) where {V,B}
    st,mt = scalar(t),imaginary(t)
    if isscalar(B*B)
        hint = value(scalar(B*B))
        isnull(hint) && (return AbstractTensors.exp(value(st))*(1+t))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(mt))))))
        return AbstractTensors.exp(value(st))*(hint<0 ? AbstractTensors.cos(θ)+mt*(AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+mt*(AbstractTensors.:/(AbstractTensors.sinh(θ),θ)))
    else
        return 1+expm1(t)
    end
end

@pure isR301(V::DiagonalForm) = DirectSum.diagonalform(V) == Values(1,1,1,0)
@pure isR301(::Submanifold{V}) where V = isR301(V)
@pure isR301(V) = false

@inline unabs!(t) = t
@inline unabs!(t::Expr) = (t.head == :call && t.args[1] == :abs) ? t.args[2] : t

function Base.exp(t::T) where T<:TensorGraded
    S,B = T<:Submanifold,T<:TensorTerm
    if B && isnull(t)
        return One(Manifold(t))
    elseif isR301(Manifold(t)) && grade(t)==2 # && abs(t[0])<1e-9 && !options.over
        u = sqrt(abs(abs2(t)[1]))
        u<1e-5 && (return 1+t)
        v,cu,su = (t∧t)*(-0.5/u),cos(u),sin(u)
        return (cu-v*su)+((su+v*cu)*t)*(inv(u)-v/u^2)
    end # need general inv(u+v) ~ inv(u)-v/u^2
    i = B ? basis(t) : t
    sq = i*i
    if isscalar(sq)
        hint = value(scalar(sq))
        isnull(hint) && (return 1+t)
        grade(t)==0 && (return Single{Manifold(t)}(AbstractTensors.exp(value(S ? t : scalar(t)))))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(t))))))
        hint<0 ? AbstractTensors.cos(θ)+t*(S ? AbstractTensors.sin(θ) : AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+t*(S ? AbstractTensors.sinh(θ) : AbstractTensors.:/(AbstractTensors.sinh(θ),θ))
    else
        return 1+expm1(t)
    end
end

function Base.exp(t::T,::Val{hint}) where T<:TensorGraded where hint
    S = T<:Submanifold
    i = T<:TensorTerm ? basis(t) : t
    sq = i*i
    if isscalar(sq)
        isnull(hint) && (return 1+t)
        grade(t)==0 && (return Single{Manifold(t)}(AbstractTensors.exp(value(S ? t : scalar(t)))))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(t))))))
        hint<0 ? AbstractTensors.cos(θ)+t*(S ? AbstractTensors.sin(θ) : AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+t*(S ? AbstractTensors.sinh(θ) : AbstractTensors.:/(AbstractTensors.sinh(θ),θ))
    else
        return 1+expm1(t)
    end
end

function qlog(w::T,x::Int=10000) where T<:TensorAlgebra
    V = Manifold(w)
    w2,f = w^2,norm(w)
    prod = w*w2
    S,term = w,prod/3
    norms = FixedVector{3}(f,norm(term),f)
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

@eval @generated function qlog_fast(b::Multivector{V,T,E},x::Int=10000) where {V,T,E}
    loop = generate_loop_multivector(V,:prod,:B,:*,:geomaddmulti!,geomaddmulti!_pre)
    return quote
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        f = norm(b)
        w2::Multivector{V,T,E} = b^2
        B = value(w2)
        S = zeros(mvec(N,t))
        prod = zeros(mvec(N,t))
        term = zeros(mvec(N,t))
        S .= value(b)
        out .= value(b*w2)
        term .= out/3
        norms = FixedVector{3}(f,norm(term),f)
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
        return Multivector{V}(S)
    end
end

@eval @generated function qlog_fast(b::Spinor{V,T,E},x::Int=10000) where {V,T,E}
    loop = generate_loop_spinor(V,:prod,:B,:*,:geomaddspin!,geomaddspin!_pre)
    return quote
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        f = norm(b)
        w2::Spinor{V,T,E} = b^2
        B = value(w2)
        S = zeros(mvecs(N,t))
        prod = zeros(mvecs(N,t))
        term = zeros(mvecs(N,t))
        S .= value(b)
        out .= value(b*w2)
        term .= out/3
        norms = FixedVector{3}(f,norm(term),f)
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
        return Spinor{V}(S)
    end
end

Base.log(t::Couple{V,B}) where {V,B} = value(B*B)==-1 ? Couple{V,B}(log(t.v)) : qlog((t-1)/(t+1))
Base.log1p(t::Couple{V,B}) where {V,B} = value(B*B)==-1 ? Couple{V,B}(log1p(t.v)) : qlog(t/(t+2))
@inline Base.log(t::T) where T<:TensorAlgebra = qlog((t-1)/(t+1))
@inline Base.log1p(t::T) where T<:TensorAlgebra = qlog(t/(t+2))

@inline Base.sinh(::Zero{V}) where V = Zero(V)

for op ∈ (:log,:exp,:asin,:acos,:atan,:acot,:sinc,:cosc)
    @eval @inline Base.$op(t::T) where T<:TensorGraded{V,0} where V = Single{V}($op(value(t)))
end

for op ∈ (:exp,:exp2,:exp10,:cosh,:sinc) # exp
    @eval @inline Base.$op(::Zero{V}) where V = One(V)
end
for op ∈ (:log,:log2,:log10,:asech,:acosh,:sinc)
    @eval @inline Base.$op(::One{V}) where V = Zero(V)
end
for op ∈ (:acos,:asin,:atan,:asinh,:atanh,:cosc,:sqrt,:cbrt)
    @eval @inline Base.$op(t::Zero) = t
end

for (qrt,n) ∈ ((:sqrt,2),(:cbrt,3))
    @eval begin
        @inline function Base.$qrt(t::T) where T<:TensorAlgebra
            isscalar(t) ? $qrt(scalar(t)) : exp(log(t)/$n)
        end
        @inline function Base.$qrt(t::Couple{V,B}) where {V,B}
            value(B*B)==-1 ? Couple{V,B}($qrt(t.v)) :
                isscalar(t) ? $qrt(scalar(t)) : exp(log(t)/$n)
        end
        @inline Base.$qrt(t::Submanifold{V,0} where V) = t
        @inline Base.$qrt(t::T) where T<:TensorGraded{V,0} where V = Single{V}($Sym.$qrt(value(T<:TensorTerm ? t : scalar(t))))
    end
end

## trigonometric

@inline Base.cosh(t::T) where T<:TensorGraded{V,0} where V = Single{Manifold(t)}(AbstractTensors.cosh(value(T<:TensorTerm ? t : scalar(t))))

function Base.cosh(t::T) where T<:TensorAlgebra
    V = Manifold(t)
    if T<:Couple
        B = basis(t); BB = value(B*B)
        if BB == -1
            return Couple{V,B}(cosh(t.v))
        end
    end
    τ = t^2
    S,term = τ/2,(τ^2)/24
    f = norm(S)
    norms = FixedVector{3}(f,norm(term),f)
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

@eval @generated function Base.cosh(b::Multivector{V,T,E}) where {V,T,E}
    loop = generate_loop_multivector(V,:term,:B,:*,:geomaddmulti!,geomaddmulti!_pre,:(k*(k-1)))
    return quote
        sb,nb = scalar(b),norm(b)
        sb ≈ nb && (return Single{V}(AbstractTensors.cosh(value(sb))))
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        τ::Multivector{V,T,E} = b^2
        B = value(τ)
        S = zeros(mvec(N,t))
        term = zeros(mvec(N,t))
        S .= value(τ)/2
        out .= value((τ^2))/24
        norms = FixedVector{3}(norm(S),norm(out),norm(term))
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
        return Multivector{V}(S)
    end
end

@eval @generated function Base.cosh(b::Spinor{V,T,E}) where {V,T,E}
    loop = generate_loop_spinor(V,:term,:B,:*,:geomaddspin!,geomaddspin!_pre,:(k*(k-1)))
    return quote
        sb,nb = scalar(b),norm(b)
        sb ≈ nb && (return Single{V}(AbstractTensors.cosh(value(sb))))
        $(insert_expr(loop[1],:mvecs,:T,Float64)...)
        τ::Spinor{V,T,E} = b^2
        B = value(τ)
        S = zeros(mvecs(N,t))
        term = zeros(mvecs(N,t))
        S .= value(τ)/2
        out .= value((τ^2))/24
        norms = FixedVector{3}(norm(S),norm(out),norm(term))
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
        return Spinor{V}(S)
    end
end


@inline Base.sinh(t::T) where T<:TensorGraded{V,0} where V = Single{Manifold(t)}(AbstractTensors.sinh(value(T<:TensorTerm ? t : scalar(t))))

function Base.sinh(t::T) where T<:TensorAlgebra
    V = Manifold(t)
    if T<:Couple
        B = basis(t); BB = value(B*B)
        if BB == -1
            return Couple{V,B}(sinh(t.v))
        end
    end
    τ,f = t^2,norm(t)
    S,term = t,(t*τ)/6
    norms = FixedVector{3}(f,norm(term),f)
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

@eval @generated function Base.sinh(b::Multivector{V,T,E}) where {V,T,E}
    loop = generate_loop_multivector(V,:term,:B,:*,:geomaddmulti!,geomaddmulti!_pre,:(k*(k-1)))
    return quote
        sb,nb = scalar(b),norm(b)
        sb ≈ nb && (return Single{V}(AbstractTensors.sinh(value(sb))))
        $(insert_expr(loop[1],:mvec,:T,Float64)...)
        τ::Multivector{V,T,E} = b^2
        B = value(τ)
        S = zeros(mvec(N,t))
        term = zeros(mvec(N,t))
        S .= value(b)
        out .= value(b*τ)/6
        norms = FixedVector{3}(norm(S),norm(out),norm(term))
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
        return Multivector{V}(S)
    end
end

@eval @generated function Base.sinh(b::Spinor{V,T,E}) where {V,T,E}
    loop = generate_loop_spinor(V,:term,:B,:*,:geomaddspin!,geomaddspin!_pre,:(k*(k-1)))
    return quote
        sb,nb = scalar(b),norm(b)
        sb ≈ nb && (return Single{V}(AbstractTensors.sinh(value(sb))))
        $(insert_expr(loop[1],:mvecs,:T,Float64)...)
        τ::Spinor{V,T,E} = b^2
        B = value(τ)
        S = zeros(mvecs(N,t))
        term = zeros(mvecs(N,t))
        S .= value(b)
        out .= value(b*τ)/6
        norms = FixedVector{3}(norm(S),norm(out),norm(term))
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
        return Spinor{V}(S)
    end
end

exph(t) = Base.cosh(t)+Base.sinh(t)

for (logfast,expf) ∈ ((:log_fast,:exp),(:logh_fast,:exph))
    @eval function $logfast(t::T) where T<:TensorAlgebra
        V = Manifold(t)
        term = Zero(V)
        nrm = FixedVector{2}(0.,0.)
        while true
            en = $expf(term)
            term -= 2(en-t)/(en+t)
            @inbounds nrm .= (nrm[2],norm(term))
            @inbounds nrm[1] ≈ nrm[2] && break
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

function Cramer(N::Int,j=0)
    t = j ≠ 0 ? :T : :t
    x,y = Values{N}([Symbol(:x,i) for i ∈ 1:N]),Values{N}([Symbol(:y,i) for i ∈ 1:N])
    xy = [:(($(x[1+i]),$(y[1+i])) = ($(x[i])∧$t[$(1+i-j)],$t[end-$i]∧$(y[i]))) for i ∈ 1:N-1-j]
    return x,y,xy
end

DirectSum.Λ(x::Chain{V,1,<:Chain{V,1}},G) where V = compound(x,G)
compound(x,G::T) where T<:Integer = compound(x,Val(G))
@generated function compound(x::Chain{V,1,<:Chain{V,1}},::Val{G}) where {V,G}
    Expr(:call,:(Chain{V,G}),Expr(:call,:Values,[Expr(:call,:∧,[:(x[$i]) for i ∈ indices(j)]...) for j ∈ indexbasis(mdims(V),G)]...))
end

@generated function Base.:\(t::Values{M,<:Chain{V,1}},v::Chain{V,1}) where {M,V}
    W = M≠mdims(V) ? Submanifold(M) : V; N = M-1
    if M == 1 && (V === ℝ1 || V == 1)
        return :(@inbounds Chain{V,1}(Values(v[1]/t[1][1])))
    elseif M == 2 && (V === ℝ2 || V == 2)
        return quote
            (a,A),(b,B),(c,C) = value(t[1]),value(t[2]),value(v)
            x1 = (c-C*(b/B))/(a-A*(b/B))
            return Chain{V,1}(x1,(C-A*x1)/B)
        end
    elseif M == 3 && (V === ℝ3 || V == 3)
        return quote
            dv = @inbounds v/∧(t)[1]; c1,c2,c3 = value(t)
            return @inbounds Chain{V,1}(
                (c2[2]*c3[3] - c3[2]*c2[3])*dv[1] +
                    (c3[1]*c2[3] - c2[1]*c3[3])*dv[2] +
                    (c2[1]*c3[2] - c3[1]*c2[2])*dv[3],
                (c3[2]*c1[3] - c1[2]*c3[3])*dv[1] +
                    (c1[1]*c3[3] - c3[1]*c1[3])*dv[2] +
                    (c3[1]*c1[2] - c1[1]*c3[2])*dv[3],
                (c1[2]*c2[3] - c2[2]*c1[3])*dv[1] +
                    (c2[1]*c1[3] - c1[1]*c2[3])*dv[2] +
                    (c1[1]*c2[2] - c2[1]*c1[2])*dv[3])
        end
    end
    N<1 && (return :(inv(t)⋅v))
    M > mdims(V) && (return :(tt=_transpose(t,$W); tt⋅(inv(Chain{$W,1}(t)⋅tt)⋅v)))
    x,y,xy = Grassmann.Cramer(N) # paste this into the REPL for faster eval
    mid = [:($(x[i])∧v∧$(y[end-i])) for i ∈ 1:N-1]
    out = Expr(:call,:Values,:(v∧$(y[end])),mid...,:($(x[end])∧v))
    detx = :(detx = @inbounds (t[1]∧$(y[end])))
    return Expr(:block,:((x1,y1)=@inbounds (t[1],t[end])),xy...,detx,
        :(Chain{$W,1}(column($(Expr(:call,:.⋅,out,:(Ref(detx))))./abs2(detx)))))
end

@generated function Base.in(v::Chain{V,1},t::Values{N,<:Chain{V,1}}) where {V,N}
    if N == mdims(V)
        x,y,xy = Grassmann.Cramer(N-1)
        mid = [:(s==signbit(@inbounds ($(x[i])∧v∧$(y[end-i]))[1])) for i ∈ 1:N-2]
        out = Values(:(s==signbit(@inbounds (v∧$(y[end]))[1])),mid...,:(s==signbit(@inbounds ($(x[end])∧v)[1])))
        return Expr(:block,:((x1,y1)=@inbounds (t[1],t[end])),xy...,:(s=signbit(@inbounds (t[1]∧$(y[end]))[1])),ands(out))
    else
        x,y,xy = Grassmann.Cramer(N-1,1)
        mid = [:(signscalar(($(x[i])∧(v-x1)∧$(y[end-i]))/d)) for i ∈ 1:N-2]
        out = Values(:(signscalar((v∧∧(vectors(t,v)))/d)),mid...,:(signscalar(($(x[end])∧(v-x1))/d)))
        return Expr(:block,:(T=vectors(t)),:((x1,y1)=@inbounds (t[1],T[end])),xy...,
            :($(x[end])=$(x[end-1])∧T[end-1];d=$(x[end])∧T[end]),ands(out))
    end
end

@generated function Base.inv(t::Values{M,<:Chain{V,1}}) where {M,V}
    W = M≠mdims(V) ? Submanifold(M) : V; N = M-1
    N<1 && (return :(_transpose(Values(inv(@inbounds t[1])),$W)))
    M > mdims(V) && (return :(tt = _transpose(t,$W); tt⋅inv(Chain{$W,1}(t)⋅tt)))
    x,y,xy = Grassmann.Cramer(N)
    val = if iseven(N)
        Expr(:call,:Values,y[end],[:($(y[end-i])∧$(x[i])) for i ∈ 1:N-1]...,x[end])
    elseif M≠mdims(V)
        Expr(:call,:Values,y[end],[:($(iseven(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ 1:N-1]...,:(-$(x[end])))
    else
        Expr(:call,:Values,:(-$(y[end])),[:($(isodd(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ 1:N-1]...,x[end])
    end
    out = if M≠mdims(V)
        :(vector.($(Expr(:call,:./,val,:(@inbounds (t[1]∧$(y[end])))))))
    else
        :(.⋆($(Expr(:call,:./,val,:(@inbounds (t[1]∧$(y[end]))[1])))))
    end
    return Expr(:block,:((x1,y1)=@inbounds (t[1],t[end])),xy...,:(_transpose($out,$W)))
end

@generated function grad(T::Values{M,<:Chain{V,1}}) where {M,V}
    W = M≠mdims(V) ? Submanifold(M) : V; N = mdims(V)-1
    M < mdims(V) && (return :(ct = Chain{$W,1}(T); map(↓(V),ct⋅inv(_transpose(T,$W)⋅ct))))
    x,y,xy = Grassmann.Cramer(N)
    val = if iseven(N)
        Expr(:call,:Values,[:($(y[end-i])∧$(x[i])) for i ∈ 1:N-1]...,x[end])
    elseif M≠mdims(V)
        Expr(:call,:Values,y[end],[:($(iseven(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ 1:N-1]...,:(-$(x[end])))
    else
        Expr(:call,:Values,[:($(isodd(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ 1:N-1]...,x[end])
    end
    out = if M≠mdims(V)
        :(vector.($(Expr(:call,:./,val,:(@inbounds (t[1]∧$(y[end])))))))
    else
        :(.⋆($(Expr(:call,:./,val,:(@inbounds (t[1]∧$(y[end]))[1])))))
    end
    return Expr(:block,:(t=_transpose(T,$W)),:((x1,y1)=@inbounds (t[1],t[end])),xy...,:(_transpose($out,↓(V))))
end

@generated function Base.:\(t::Values{N,<:Chain{M,1}},v::Chain{V,1}) where {N,M,V}
    W = M≠mdims(V) ? Submanifold(N) : V
    if mdims(M) > mdims(V)
        :(ct=Chain{$W,1}(t); ct⋅(inv(_transpose(t,$W)⋅ct)⋅v))
    else # mdims(M) < mdims(V) ? inv(tt⋅t)⋅(tt⋅v) : tt⋅(inv(t⋅tt)⋅v)
        :(_transpose(t,$W)\v)
    end
end
function inv_approx(t::Chain{M,1,<:Chain{V,1}}) where {M,V}
    tt = transpose(t)
    mdims(M) < mdims(V) ? (inv(tt⋅t))⋅tt : tt⋅inv(t⋅tt)
end

Base.:\(t::Chain{M,1,<:Chain{W,1}},v::Chain{V,1}) where {M,W,V} = value(t)\v
Base.in(v::Chain{V,1},t::Chain{W,1,<:Chain{V,1}}) where {V,W} = v ∈ value(t)
Base.inv(t::Chain{V,1,<:Chain{W,1}}) where {W,V} = inv(value(t))
grad(t::Chain{V,1,<:Chain{W,1}}) where {V,W} = grad(value(t))

export vandermonde

@generated approx(x,y::Chain{V}) where V = :(polynom(x,$(Val(mdims(V))))⋅y)
approx(x,y::Values{N}) where N = value(polynom(x,Val(N)))⋅y
approx(x,y::AbstractVector) = [x^i for i ∈ 0:length(y)-1]⋅y

vandermonde(x::Array,y::Array,N::Int) = vandermonde(x,N)\y[:] # compute ((inv(X'*X))*X')*y
function vandermonde(x::Array,N)
    V = zeros(length(x),N)
    for d ∈ 0:N-1
        V[:,d+1] = x.^d
    end
    return V # Vandermonde
end

vandermonde(x,y,V) = (length(x)≠mdims(V) ? _vandermonde(x,V) : vandermonde(x,V))\y
vandermonde(x,V) = transpose(_vandermonde(x,V))
_vandermonde(x::Chain,V) = _vandermonde(value(x),V)
@generated _vandermonde(x::Values{N},V) where N = :(Chain{$(Submanifold(N)),1}(polynom.(x,$(Val(mdims(V))))))
@generated polynom(x,::Val{N}) where N = Expr(:call,:(Chain{$(Submanifold(N)),1}),Expr(:call,:Values,[:(x^$i) for i ∈ 0:N-1]...))

function vandermondeinterp(x,y,V,grid) # grid=384
    coef = vandermonde(x,y,V) # Vandermonde ((inv(X'*X))*X')*y
    minx,maxx = minimum(x),maximum(x)
    xp,yp = [minx:(maxx-minx)/grid:maxx...],coef[1]*ones(grid+1)
    for d ∈ 1:length(coef)-1
        yp += coef[d+1].*xp.^d
    end # fill in polynomial terms
    return coef,xp,yp # coefficients, interpolation
end

@generated function vectors(t,c=columns(t))
    v = Expr(:tuple,[:(M.(p[c[$i]]-A)) for i ∈ 2:mdims(t)]...)
    quote
        p = points(t)
        M,A = ↓(Manifold(p)),p[c[1]]
        Chain{M,1}.($(Expr(:.,:Values,v)))
    end
end
@pure list(a::Int,b::Int) = Values{max(0,b-a+1),Int}(a:b...)
@pure evens(a::Int,b::Int) = Values{((b-a)÷2)+1,Int}(a:2:b...)
vectors(x::Values{N,<:Chain{V}},y=x[1]) where {N,V} = ↓(V).(x[list(2,N)].-y)
vectors(x::Chain{V,1},y=x[1]) where V = vectors(value(x),y)
#point(x,y=x[1]) = y∧∧(vectors(x))

signscalar(x::Submanifold{V,0} where V) = true
signscalar(x::Single{V,0} where V) = !signbit(value(x))
signscalar(x::Single) = false
signscalar(x::Chain) = false
signscalar(x::Chain{V,0} where V) = !signbit(@inbounds x[1])
signscalar(x::Multivector) = isscalar(x) && !signbit(value(scalar(x)))
ands(x,i=length(x)-1) = i ≠ 0 ? Expr(:&&,x[end-i],ands(x,i-1)) : x[end-i]

function Base.findfirst(P,t::Vector{<:Chain{V,1,<:Chain}} where V)
    for i ∈ 1:length(t)
        @inbounds P ∈ t[i] && (return i)
    end
    return 0
end
function Base.findfirst(P,t::ChainBundle)
    p = points(t)
    for i ∈ 1:length(t)
        P ∈ p[t[i]] && (return i)
    end
    return 0
end
function Base.findlast(P,t::Vector{<:Chain{V,1,<:Chain}} where V)
    for i ∈ length(t):-1:1
        @inbounds P ∈ t[i] && (return i)
    end
    return 0
end
function Base.findlast(P,t::ChainBundle)
    p = points(t)
    for i ∈ length(t):-1:1
        P ∈ p[t[i]] && (return i)
    end
    return 0
end
Base.findall(P,t) = findall(P .∈ getindex.(points(t),value(t)))

export volumes, detsimplex, initmesh, refinemesh, refinemesh!, select, submesh

edgelength(e) = (v=points(e)[value(e)]; value(abs(v[2]-v[1])))
volumes(m,dets) = value.(abs.(.⋆(dets)))
volumes(m) = mdims(Manifold(m))≠2 ? volumes(m,detsimplex(m)) : edgelength.(value(m))
detsimplex(m::Vector{<:Chain{V}}) where V = ∧(m)/factorial(mdims(V)-1)
detsimplex(m::ChainBundle) = detsimplex(value(m))
mean(m::T) where T<:AbstractVector{<:Chain} = sum(m)/length(m)
mean(m::T) where T<:Values = sum(m)/length(m)
mean(m::Chain{V,1,<:Chain} where V) = mean(value(m))
barycenter(m::Values{N,<:Chain}) where N = (s=sum(m);@inbounds s/s[1])
barycenter(m::Vector{<:Chain}) = (s=sum(m);@inbounds s/s[1])
barycenter(m::Chain{V,1,<:Chain} where V) = barycenter(value(m))
curl(m::FixedVector{N,<:Chain{V}} where N) where V = curl(Chain{V,1}(m))
curl(m::Values{N,<:Chain{V}} where N) where V = curl(Chain{V,1}(m))
curl(m::T) where T<:TensorAlgebra = Manifold(m)(∇)×m
LinearAlgebra.det(t::Chain{V,1,<:Chain} where V) = ∧(t)
LinearAlgebra.det(m::Vector{<:Chain{V}}) where V = ∧(m)
LinearAlgebra.det(m::ChainBundle) = ∧(m)
∧(m::ChainBundle) = ∧(value(m))
function ∧(m::Vector{<:Chain{V}}) where V
    p = points(m); pm = p[m]
    if mdims(p)>mdims(V)
        .∧(vectors.(pm))
    else
        Chain{↓(Manifold(V)),mdims(V)-1}.(value.(.∧(pm)))
    end
end
for op ∈ (:mean,:barycenter,:curl)
    ops = Symbol(op,:s)
    @eval begin
        export $op, $ops
        $ops(m::Vector{<:Chain{p}}) where p = $ops(m,p)
        @pure $ops(m::ChainBundle{p}) where p = $ops(m,p)
        @pure $ops(m,::Submanifold{p}) where p = $ops(m,p)
        @pure $ops(m,p) = $op.(getindex.(Ref(p),value.(value(m))))
    end
end

function area(m::Vector{<:Chain})
    S = m[end]∧m[1]
    for i ∈ 1:length(m)-1
        S += m[i]∧m[i+1]
    end
    return value(abs(⋆(S))/2)
end

initedges(p::ChainBundle) = Chain{p,1}.(1:length(p)-1,2:length(p))
initedges(r::R) where R<:AbstractVector = initedges(ChainBundle(initpoints(r)))
function initmesh(r::R) where R<:AbstractVector
    t = initedges(r); p = points(t)
    p,ChainBundle(Chain{↓(p),1}.([1,length(p)])),t
end

select(η,ϵ=sqrt(norm(η)^2/length(η))) = sort!(findall(x->x>ϵ,η))
refinemesh(g::R,args...) where R<:AbstractRange = (g,initmesh(g,args...)...)
function refinemesh!(::R,p::ChainBundle{W},e,t,η,_=nothing) where {W,R<:AbstractRange}
    p = points(t)
    x,T,V = value(p),value(t),Manifold(p)
    for i ∈ η
        push!(x,Chain{V,1}(Values(1,(x[i+1][2]+x[i][2])/2)))
    end
    sort!(x,by=x->x[2]); submesh!(p)
    e[end] = Chain{p(2),1}(Values(length(x)))
    for i ∈ length(t)+2:length(x)
        push!(T,Chain{p,1}(Values{2,Int}(i-1,i)))
    end
end

const array_cache = (Array{T,2} where T)[]
array(m::Vector{<:Chain}) = [m[i][j] for i∈1:length(m),j∈1:mdims(Manifold(m))]
function array(m::ChainBundle{V,G,T,B} where {V,G,T}) where B
    for k ∈ length(array_cache):B
        push!(array_cache,Array{Any,2}(undef,0,0))
    end
    isempty(array_cache[B]) && (array_cache[B] = array(value(m)))
    return array_cache[B]
end
function array!(m::ChainBundle{V,G,T,B} where {V,G,T}) where B
    length(array_cache) ≥ B && (array_cache[B] = Array{Any,2}(undef,0,0))
end

const submesh_cache = (Array{T,2} where T)[]
submesh(m) = [m[i][j] for i∈1:length(m),j∈2:mdims(Manifold(m))]
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
        Base.$op(a::Chain{V,G,T},m) where {V,G,T} = Chain{V,G}($op.(value(a),m))
        Base.$op(a::Multivector{V,T},m) where {T,V} = Multivector{V}($op.(value(a),m))
    end
end
for op ∈ (:mod2pi,:rem2pi,:rad2deg,:deg2rad,:round)
    @eval begin
        Base.$op(a::Chain{V,G,T}) where {V,G,T} = Chain{V,G}($op.(value(a)))
        Base.$op(a::Multivector{V,T}) where {V,T} = Multivector{V}($op.(value(a)))
    end
end
Base.isfinite(a::Chain) = prod(isfinite.(value(a)))
Base.isfinite(a::Multivector) = prod(isfinite.(value(a)))
Base.rationalize(t::Type,a::Chain{V,G,T};tol::Real=eps(T)) where {V,G,T} = Chain{V,G}(rationalize.(t,value(a),tol))
Base.rationalize(t::Type,a::Multivector{V,T};tol::Real=eps(T)) where {V,T} = Multivector{V}(rationalize.(t,value(a),tol))
Base.rationalize(t::T;kvs...) where T<:TensorAlgebra = rationalize(Int,t;kvs...)

*(A::SparseMatrixCSC{TA,S}, x::StridedVector{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
    (T = promote_type(TA, Chain{V,G,𝕂,X}); SparseArrays.mul!(similar(x, T, A.m), A, x, 1, 0))
*(A::SparseMatrixCSC{TA,S}, B::StridedMatrix{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
    (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(B, T, (A.m, size(B, 2))), A, B, 1, 0))
*(adjA::LinearAlgebra.Adjoint{<:Any,<:SparseMatrixCSC{TA,S}}, x::StridedVector{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
    (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(x, T, size(adjA, 1)), adjA, x, 2, 0))
*(transA::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC{TA,S}}, x::StridedVector{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
    (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(x, T, size(transA, 1)), transA, x, 1, 0))
if VERSION >= v"1.4" && VERSION < v"1.6"
    *(adjA::LinearAlgebra.Adjoint{<:Any,<:SparseMatrixCSC{TA,S}}, B::SparseArrays.AdjOrTransStridedOrTriangularMatrix{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
        (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(B, T, (size(adjA, 1), size(B, 2))), adjA, B, 1, 0))
    *(transA::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC{TA,S}}, B::SparseArrays.AdjOrTransStridedOrTriangularMatrix{Chain{V,G,𝕂,X}}) where {TA,S,V,G,𝕂,X} =
        (T = promote_type(TA, Chain{V,G,𝕂,X}); mul!(similar(B, T, (size(transA, 1), size(B, 2))), transA, B, 1, 0))
end

@generated function AbstractTensors._diff(::Val{N}, a::Values{Q,<:Chain}, ::Val{1}) where {N,Q}
    Snew = N-1
    exprs = Array{Expr}(undef, Snew)
    for i1 = Base.product(1:Snew)
        i2 = copy([i1...])
        i2[1] = i1[1] + 1
        exprs[i1...] = :(a[$(i2...)] - a[$(i1...)])
    end
    return quote
        Base.@_inline_meta
        elements = tuple($(exprs...))
        @inbounds return AbstractTensors.similar_type(a, eltype(a), Val($Snew))(elements)
    end
end

Base.map(fn, x::Multivector{V}) where V = Multivector{V}(map(fn, value(x)))
Base.map(fn, x::Spinor{V}) where V = Spinor{V}(map(fn, value(x)))
Base.map(fn, x::Chain{V,G}) where {V,G} = Chain{V,G}(map(fn,value(x)))
Base.map(fn, x::Single{V,G,B}) where {V,G,B} = fn(value(x))*B
Base.map(fn, x::Couple{V,B}) where {V,B} = Couple{V,B}(Complex(fn(x.v.re),fn(x.v.im)))

import Random: SamplerType, AbstractRNG
Base.rand(::AbstractRNG,::SamplerType{Chain}) = rand(Chain{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Chain{V}}) where V = rand(Chain{V,rand(0:mdims(V))})
Base.rand(::AbstractRNG,::SamplerType{Chain{V,G}}) where {V,G} = Chain{V,G}(DirectSum.orand(svec(mdims(V),G,Float64)))
Base.rand(::AbstractRNG,::SamplerType{Chain{V,G,T}}) where {V,G,T} = Chain{V,G}(rand(svec(mdims(V),G,T)))
Base.rand(::AbstractRNG,::SamplerType{Chain{V,G,T} where G}) where {V,T} = rand(Chain{V,rand(0:mdims(V)),T})
Base.rand(::AbstractRNG,::SamplerType{Multivector}) = rand(Multivector{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Multivector{V}}) where V = Multivector{V}(DirectSum.orand(svec(mdims(V),Float64)))
Base.rand(::AbstractRNG,::SamplerType{Multivector{V,T}}) where {V,T} = Multivector{V}(rand(svec(mdims(V),T)))
Base.rand(::AbstractRNG,::SamplerType{Spinor}) = rand(Spinor{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Spinor{V}}) where V = Spinor{V}(DirectSum.orand(svecs(mdims(V),Float64)))
Base.rand(::AbstractRNG,::SamplerType{Spinor{V,T}}) where {V,T} = Spinor{V}(rand(svecs(mdims(V),T)))
Base.rand(::AbstractRNG,::SamplerType{Couple}) = rand(Couple{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Couple{V}}) where V = rand(Couple{V,rand(Single{V})})
Base.rand(::AbstractRNG,::SamplerType{Couple{V,B}}) where {V,B} = Couple{V,G}(rand(Complex{Float64}))
Base.rand(::AbstractRNG,::SamplerType{Couple{V,B,T}}) where {V,B,T} = Couple{V,G}(rand(Complex{T}))
Base.rand(::AbstractRNG,::SamplerType{Couple{V,B,T} where B}) where {V,T} = rand(Couple{V,rand(Single{V}),T})

export Orthogrid

@computed struct Orthogrid{V,T} # <: TensorGraded{V,1} mess up collect?
    v::Dyadic{V,Chain{V,1,T,mdims(V)},Chain{V,1,T,mdims(V)}}
    n::Chain{V,1,Int}
    s::Chain{V,1,Float64}
    Orthogrid{V,T}(v,n,s) where {V,T} = new{DirectSum.submanifold(V),T}(v,n,s)
end

Orthogrid{V,T}(v,n) where {V,T} = Orthogrid{V,T}(v,n,Chain{V,1}(value(v.x-v.y)./(value(n)-1)))

Base.show(io::IO,t::Orthogrid) = println('(',t.v.x,"):(",t.s,"):(",t.v.y,')')

zeroinf(f) = iszero(f) ? Inf : f

(::Base.Colon)(min::Chain{V,1,T},step::Chain{V,1,T},max::Chain{V,1,T}) where {V,T} = Orthogrid{V,T}(min⊗max,Chain{V,1}(Int.(round.(value(max-min)./zeroinf.(value(step)))).+1),step)

Base.iterate(t::Orthogrid) = (getindex(t,1),1)
Base.iterate(t::Orthogrid,state) = (s=state+1; s≤length(t) ? (getindex(t,s),s) : nothing)
@pure Base.eltype(::Type{Orthogrid{V,T}}) where {V,T} = Chain{V,1,T,mdims(V)}
@pure Base.step(t::Orthogrid) = value(t.s)
@pure Base.size(t::Orthogrid) = value(t.n).v
@pure Base.length(t::Orthogrid) = prod(size(t))
@pure Base.lastindex(t::Orthogrid) = length(t)
@pure Base.lastindex(t::Orthogrid,i::Int) = size(t)[i]
@pure Base.getindex(t::Orthogrid,i::CartesianIndex) = getindex(t,i.I...)
@pure Base.getindex(t::Orthogrid{V},i::Vararg{Int}) where V = Chain{V,1}(value(t.v.x)+(Values(i).-1).*step(t))

Base.IndexStyle(::Orthogrid) = IndexCartesian()
function Base.getindex(A::Orthogrid, I::Int)
    Base.@_inline_meta
    @inbounds getindex(A, Base._to_subscript_indices(A, I)...)
end
Base._to_subscript_indices(A::Orthogrid, i::Integer) = (Base.@_inline_meta; Base._unsafe_ind2sub(A, i))
function Base._ind2sub(A::Orthogrid, ind)
    Base.@_inline_meta
    Base._ind2sub(axes(A), ind)
end
