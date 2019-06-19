
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export exph, log_fast, logh_fast

## exponential & logarithm function

function Base.expm1(t::T) where T<:TensorAlgebra{V} where V
    S,term,f = t,(t^2)/2,frobenius(t)
    norms = MVector(f,frobenius(term),f)
    k::Int = 3
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = frobenius(S)
        @inbounds ns ≈ norms[3] && break
        term *= t/k
        @inbounds norms .= (norms[2],frobenius(term),ns)
        k += 1
    end
    return S
end

@eval function Base.expm1(b::MultiVector{T,V}) where {T,V}
    $(insert_expr((:N,:t,:out,:bs,:bn),:mvec,:T,Float64)...)
    B = value(b)
    S = zeros(mvec(N,t))
    term = zeros(mvec(N,t))
    S .= value(b)
    out .= value(b^2)/2
    norms = MVector(norm(S),norm(out),norm(term))
    k::Int = 3
    @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
        S += out
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        term .= out
        out .= 0
        # term *= b/k
        for g ∈ 1:N+1
            Y = indexbasis(N,g-1)
            @inbounds for i ∈ 1:bn[g]
                @inbounds val = B[bs[g]+i]/k
                val≠0 && for G ∈ 1:N+1
                    @inbounds R = bs[G]
                    X = indexbasis(N,G-1)
                    @inbounds for j ∈ 1:bn[G]
                        @inbounds geomaddmulti!(V,out,X[j],Y[i],*(term[R+j],val))
                    end
                end
            end
        end
        @inbounds norms .= (norms[2],norm(out),ns)
        k += 1
    end
    return MultiVector{t,V}(S)
end

@inline Base.exp(t::T) where T<:TensorAlgebra = 1+expm1(t)

@inline ^(b::S,t::T) where {S<:TensorAlgebra{V},T<:TensorAlgebra{V}} where V = exp(t*log(b))
@inline ^(b::S,t::T) where {S<:Number,T<:TensorAlgebra} = exp(t*log(b))

function qlog(w::T,x::Int=10000) where T<:TensorAlgebra{V} where V
    w2,f = w^2,frobenius(w)
    prod = w*w2
    S,term = w,prod/3
    norms = MVector(f,frobenius(term),f)
    k::Int = 5
    @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ x
        S += term
        ns = frobenius(S)
        @inbounds ns ≈ norms[3] && break
        prod *= w2
        term = prod/k
        @inbounds norms .= (norms[2],frobenius(term),ns)
        k += 2
    end
    return 2S
end # http://www.netlib.org/cephes/qlibdoc.html#qlog

@eval function qlog_fast(b::MultiVector{T,V,E},x::Int=10000) where {T,V,E}
    $(insert_expr((:N,:t,:out,:bs,:bn),:mvec,:T,Float64)...)
    f = frobenius(b)
    w2::MultiVector{T,V,E} = b^2
    B = value(w2)
    S = zeros(mvec(N,t))
    prod = zeros(mvec(N,t))
    term = zeros(mvec(N,t))
    S .= value(b)
    out .= value(b*w2)
    term .= out/3
    norms = MVector(f,norm(term),f)
    k::Int = 5
    @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ x
        S += term
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        prod .= out
        out .= 0
        # prod *= w2
        for g ∈ 1:N+1
            Y = indexbasis(N,g-1)
            @inbounds for i ∈ 1:bn[g]
                @inbounds val = B[bs[g]+i]
                val≠0 && for G ∈ 1:N+1
                    @inbounds R = bs[G]
                    X = indexbasis(N,G-1)
                    @inbounds for j ∈ 1:bn[G]
                        @inbounds geomaddmulti!(V,out,X[j],Y[i],*(prod[R+j],val))
                    end
                end
            end
        end
        term .= out/k
        @inbounds norms .= (norms[2],norm(term),ns)
        k += 2
    end
    S *= 2
    return MultiVector{t,V}(S)
end

@inline Base.log(b,t::T) where T<:TensorAlgebra = log(t)/log(b)
@inline Base.log(t::T) where T<:TensorAlgebra = qlog((t-1)/(t+1))
@inline Base.log1p(t::T) where T<:TensorAlgebra = qlog(t/(t+2))

for base ∈ (2,10)
    fl,fe = (Symbol(:log,base),Symbol(:exp,base))
    @eval Base.$fl(t::T) where T<:TensorAlgebra = $fl(ℯ)*log(t)
    @eval Base.$fe(t::T) where T<:TensorAlgebra = exp(log($base)*t)
end

@inline Base.sqrt(t::T) where T<:TensorAlgebra = exp(log(t)/2)
@inline Base.cbrt(t::T) where T<:TensorAlgebra = exp(log(t)/3)

## trigonometric

function Base.cosh(t::T) where T<:TensorAlgebra{V} where V
    τ = t^2
    S,term = τ/2,(τ^2)/24
    f = frobenius(S)
    norms = MVector(f,frobenius(term),f)
    k::Int = 6
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = frobenius(S)
        @inbounds ns ≈ norms[3] && break
        term *= τ/(k*(k-1))
        @inbounds norms .= (norms[2],frobenius(term),ns)
        k += 2
    end
    return 1+S
end

@eval function Base.cosh(b::MultiVector{T,V,E}) where {T,V,E}
    $(insert_expr((:N,:t,:out,:bs,:bn),:mvec,:T,Float64)...)
    τ::MultiVector{T,V,E} = b^2
    B = value(τ)
    S = zeros(mvec(N,t))
    term = zeros(mvec(N,t))
    S .= value(τ)/2
    out .= value((τ^2))/24
    norms = MVector(norm(S),norm(out),norm(term))
    k::Int = 6
    @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
        S += out
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        term .= out
        out .= 0
        # term *= τ/(k*(k-1))
        for g ∈ 1:N+1
            Y = indexbasis(N,g-1)
            @inbounds for i ∈ 1:bn[g]
                @inbounds val = B[bs[g]+i]/(k*(k-1))
                val≠0 && for G ∈ 1:N+1
                    @inbounds R = bs[G]
                    X = indexbasis(N,G-1)
                    @inbounds for j ∈ 1:bn[G]
                        @inbounds geomaddmulti!(V,out,X[j],Y[i],*(term[R+j],val))
                    end
                end
            end
        end
        @inbounds norms .= (norms[2],norm(out),ns)
        k += 2
    end
    @inbounds S[1] += 1
    return MultiVector{t,V}(S)
end

function Base.sinh(t::T) where T<:TensorAlgebra{V} where V
    τ,f = t^2,frobenius(t)
    S,term = t,(t*τ)/6
    norms = MVector(f,frobenius(term),f)
    k::Int = 5
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = frobenius(S)
        @inbounds ns ≈ norms[3] && break
        term *= τ/(k*(k-1))
        @inbounds norms .= (norms[2],frobenius(term),ns)
        k += 2
    end
    return S
end

@eval function Base.sinh(b::MultiVector{T,V,E}) where {T,V,E}
    $(insert_expr((:N,:t,:out,:bs,:bn),:mvec,:T,Float64)...)
    τ::MultiVector{T,V,E} = b^2
    B = value(τ)
    S = zeros(mvec(N,t))
    term = zeros(mvec(N,t))
    S .= value(b)
    out .= value(b*τ)/6
    norms = MVector(norm(S),norm(out),norm(term))
    k::Int = 5
    @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
        S += out
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        term .= out
        out .= 0
        # term *= τ/(k*(k-1))
        for g ∈ 1:N+1
            Y = indexbasis(N,g-1)
            @inbounds for i ∈ 1:bn[g]
                @inbounds val = B[bs[g]+i]/(k*(k-1))
                val≠0 && for G ∈ 1:N+1
                    @inbounds R = bs[G]
                    X = indexbasis(N,G-1)
                    @inbounds for j ∈ 1:bn[G]
                        @inbounds geomaddmulti!(V,out,X[j],Y[i],*(term[R+j],val))
                    end
                end
            end
        end
        @inbounds norms .= (norms[2],norm(out),ns)
        k += 2
    end
    return MultiVector{t,V}(S)
end

@inline Base.cos(t::T) where T<:TensorAlgebra{V} where V = cosh(V(I)*t)
@inline Base.sin(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);sinh(i*t)/i)
@inline Base.tan(t::T) where T<:TensorAlgebra = sin(t)/cos(t)
@inline Base.cot(t::T) where T<:TensorAlgebra = cos(t)/sin(t)
@inline Base.sec(t::T) where T<:TensorAlgebra = inv(cos(t))
@inline Base.csc(t::T) where T<:TensorAlgebra = inv(sin(t))
@inline Base.asec(t::T) where T<:TensorAlgebra = acos(inv(t))
@inline Base.acsc(t::T) where T<:TensorAlgebra = asin(inv(t))
@inline Base.sech(t::T) where T<:TensorAlgebra = inv(cosh(t))
@inline Base.csch(t::T) where T<:TensorAlgebra = inv(sinh(t))
@inline Base.asech(t::T) where T<:TensorAlgebra = acosh(inv(t))
@inline Base.acsch(t::T) where T<:TensorAlgebra = asinh(inv(t))
@inline Base.tanh(t::T) where T<:TensorAlgebra = sinh(t)/cosh(t)
@inline Base.coth(t::T) where T<:TensorAlgebra = cosh(t)/sinh(t)
@inline Base.asinh(t::T) where T<:TensorAlgebra = log(t+sqrt(1+t^2))
@inline Base.acosh(t::T) where T<:TensorAlgebra = log(t+sqrt(t^2-1))
@inline Base.atanh(t::T) where T<:TensorAlgebra = (log(1+t)-log(1-t))/2
@inline Base.acoth(t::T) where T<:TensorAlgebra = (log(t+1)-log(t-1))/2
Base.asin(t::T) where T<:TensorAlgebra{V} where V =(i=V(I);-i*log(i*t+sqrt(1-t^2)))
Base.acos(t::T) where T<:TensorAlgebra{V} where V =(i=V(I);-i*log(t+i*sqrt(1-t^2)))
Base.atan(t::T) where T<:TensorAlgebra{V} where V =(i=V(I);(-i/2)*(log(1+i*t)-log(1-i*t)))
Base.acot(t::T) where T<:TensorAlgebra{V} where V =(i=V(I);(-i/2)*(log(t-i)-log(t+i)))
Base.sinc(t::T) where T<:TensorAlgebra{V} where V = iszero(t) ? one(V) : (x=(1*π)*t;sin(x)/x)
Base.cosc(t::T) where T<:TensorAlgebra{V} where V = iszero(t) ? zero(V) : (x=(1*π)*t; cos(x)/t - sin(x)/(x*t))

exph(t) = cosh(t)+sinh(t)

function log_fast(t)
    term = zero(V)
    norm = MVector(0.,0.)
    while true
        en = exp(term)
        term -= 2(en-t)/(en+t)
        @inbounds norm .= (norm[2],frobenius(term))
        @inbounds norm[1] ≈ norm[2] && break
    end
    return term
end

function logh_fast(t)
    term = zero(V)
    norm = MVector(0.,0.)
    while true
        en = exph(term)
        term -= 2(en-t)/(en+t)
        @inbounds norm .= (norm[2],frobenius(term))
        @inbounds norm[1] ≈ norm[2] && break
    end
    return term
end

#=function log(t::T) where T<:TensorAlgebra{V} where V
    norms::Tuple = (frobenius(t),0)
    k::Int = 3
    τ = t-1
    if true #norms[1] ≤ 5/4
        prods = τ^2
        terms = TensorAlgebra{V}[τ,prods/2]
        norms = (norms[1],frobenius(terms[2]))
        while (norms[2]<norms[1] || norms[2]>1) && k ≤ 3000
            prods = prods*t
            push!(terms,prods/(k*(-1)^(k+1)))
            norms = (norms[2],frobenius(terms[end]))
            k += 1
        end
    else
        s = inv(t*inv(τ))
        prods = s^2
        terms = TensorAlgebra{V}[s,2prods]
        norms = (frobenius(terms[1]),frobenius(terms[2]))
        while (norms[2]<norms[1] || norms[2]>1) && k ≤ 3000
            prods = prods*s
            push!(terms,k*prods)
            norms = (norms[2],frobenius(terms[end]))
            k += 1
        end
    end
    return sum(terms[1:end-1])
end=#
