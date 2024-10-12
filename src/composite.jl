
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

export exph, log_fast, logh_fast, pseudoexp, pseudolog, pseudometric, pseudodot, @pseudo
export pseudoabs, pseudoabs2, pseudosqrt, pseudocbrt, pseudoinv, pseudoscalar
export pseudocos, pseudosin, pseudotan, pseudocosh, pseudosinh, pseudotanh
export vandermonde, volumes, detsimplex, submesh

## exponential & logarithm function

for (op,field) ∈ ((:⟑,false),(:wedgedot_metric,true));
    args = field ? (:g,) : ()
    indu(t=:(exp(t))) = field ? :(isinduced(g) && (return :($$t))) : nothing
@eval @inline Base.expm1(t::Submanifold{V,0},$(args...)) where V = Single{V}(ℯ-1)
@eval @inline Base.expm1(t::T,$(args...)) where T<:TensorGraded{V,0} where V = Single{Manifold(t)}(AbstractTensors.expm1(value(T<:TensorTerm ? t : scalar(t))))

@eval Base.expm1(t::Chain,$(args...)) = expm1(multispin(t),$(args...))
@eval function Base.expm1(t::T,$(args...)) where T<:TensorAlgebra
    $(indu(:(expm1(t))))
    V = Manifold(t)
    if T<:Couple
        B = basis(t); BB = value($op(B,B,$(args...)))
        if BB == -1
            return Couple{V,B}(expm1(t.v))
        end
    end
    S,term,f = t,$op(t,t,$(args...))/2,norm(t)
    norms = FixedVector{3}(f,norm(term),f)
    k::Int = 3
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        $(field ? :(term = $op(term,t/k,g)) : :(term *= t/k))
        @inbounds norms .= (norms[2],norm(term),ns)
        k += 1
    end
    return S
end

for pinor ∈ (:Multivector,:Spinor); VEC = pinor≠:Spinor ? :mvec : :mvecs
@eval @generated function Base.expm1(b::$pinor{V,T},$(args...)) where {V,T}
    $(indu(:(expm1(b))))
    loop = $(pinor≠:Spinor ? :generate_loop_multivector : :generate_loop_spinor)(V,:term,:B,promote_type(T,Float64),:*,$(QuoteNode(pinor≠:Spinor ? :geomaddmulti! : :geomaddspin!)),$(pinor≠:Spinor ? :geomaddmulti!_pre : :geomaddspin!_pre),$field,:k)
    return quote
        B = value(b)
        sb,nb = scalar(b),AbstractTensors.norm(B)
        sb ≈ nb && (return Single{V}(AbstractTensors.expm1(value(sb))))
        $(insert_expr(loop[1],$(QuoteNode(VEC)),:T,Float64)...)
        S = zeros($$VEC(N,t))
        term = zeros($$VEC(N,t))
        S .= B
        out .= value($$op(b,b,$($args...)))/2
        norms = FixedVector{3}(nb,norm(out),norm(term))
        k::Int = 3
        @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
            S += out
            ns = norm(S)
            @inbounds ns ≈ norms[3] && break
            term .= out
            # term *= b/k
            $(loop[2])
            @inbounds norms .= (norms[2],norm(out),ns)
            k += 1
        end
        return $$pinor{V}(S)
    end
end

@eval function Base.exp(t::$pinor{V},$(args...)) where V
    $(indu(:(exp(t))))
    st = scalar(t)
    mt = t-st
    sq = $op(mt,mt,$(args...))
    if isscalar(sq)
        hint = value(scalar(sq))
        isnull(hint) && (return AbstractTensors.exp(value(st))*(One(V)+t))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(mt,$(args...)))))))
        return AbstractTensors.exp(value(st))*(hint<0 ? AbstractTensors.cos(θ)+$op(mt,AbstractTensors.:/(AbstractTensors.sin(θ),θ),$(args...)) : AbstractTensors.cosh(θ)+$op(mt,AbstractTensors.:/(AbstractTensors.sinh(θ),θ),$(args...)))
    else
        return One(V)+expm1(t)
    end
end
end

@eval function Base.exp(t::Couple{V,B},$(args...)) where {V,B}
    $(indu(:(exp(t))))
    st,mt = scalar(t),imaginary(t)
    if isscalar($op(B,B,$(args...)))
        hint = value(scalar($op(B,B,$(args...))))
        isnull(hint) && (return AbstractTensors.exp(value(st))*(One(V)+t))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(mt,$(args...)))))))
        return AbstractTensors.exp(value(st))*(hint<0 ? AbstractTensors.cos(θ)+$op(mt,AbstractTensors.:/(AbstractTensors.sin(θ),θ),$(args...)) : AbstractTensors.cosh(θ)+$op(mt,AbstractTensors.:/(AbstractTensors.sinh(θ),θ),$(args...)))
    else
        return One(V)+expm1(t)
    end
end

@eval function Base.expm1(t::PseudoCouple{V,B},$(args...)) where {V,B}
    $(indu(:(expm1(t))))
    if isscalar(B)
        exp(t,$(args...))-One(V)
    else
        expm1(multispin(t),$(args...))
    end
end
@eval function Base.exp(t::PseudoCouple{V,B},$(args...)) where {V,B}
    $(indu(:(exp(t))))
    if isscalar(B)
        out = exp(Couple{V,Submanifold(V)}(realvalue(t),imagvalue(t)),$(args...))
        PseudoCouple{V,B}(realvalue(out),imagvalue(out))
    else
        exp(multispin(t),$(args...))
    end
end

@eval Base.expm1(t::Phasor{V},$(args...)) where V = exp(t,$(args...))-One(V)
@eval function Base.exp(t::Phasor{V,B},$(args...)) where {V,B}
    z = exp(angle(t,$(args...)))
    Phasor{V,B}(exp(radius(t,$(args...))+realvalue(z)),imagvalue(z))
end

@eval function Base.exp(t::T,$(args...)) where T<:TensorGraded
    $(indu(:(exp(t))))
    S,B,V = T<:Submanifold,T<:TensorTerm,Manifold(t)
    if B && isnull(t)
        vt = valuetype(t)
        return Couple{V,basis(t)}(one(vt),zero(vt))
    elseif !$field && isR301(V) && grade(t)==2 # && abs(t[0])<1e-9 && !options.over
        u = sqrt(abs(abs2(t)[1]))
        u<1e-5 && (return One(V)+t)
        v,cu,su = (t∧t)*(-0.5/u),cos(u),sin(u)
        return (cu-v*su)+((su+v*cu)*t)*(inv(u)-v/(u*u))
    end # need general inv(u+v) ~ inv(u)-v/u^2
    i = B ? basis(t) : t
    sq = $op(i,i,$(args...))
    if isscalar(sq)
        hint = value(scalar(sq))
        isnull(hint) && (return One(V)+t)
        grade(t)==0 && (return Single{Manifold(t)}(AbstractTensors.exp(value(S ? t : scalar(t)))))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(t,$(args...)))))))
        hint<0 ? AbstractTensors.cos(θ)+$op(t,S ? AbstractTensors.sin(θ) : AbstractTensors.:/(AbstractTensors.sin(θ),θ),$(args...)) : AbstractTensors.cosh(θ)+$op(t,S ? AbstractTensors.sinh(θ) : AbstractTensors.:/(AbstractTensors.sinh(θ),θ),$(args...))
    else
        return One(V)+expm1(t,$(args...))
    end
end
end

function Base.exp(t::Multivector{V},::Val{hint}) where {V,hint}
    st = scalar(t)
    mt = t-st
    sq = mt⟑mt
    if isscalar(sq)
        isnull(hint) && (return AbstractTensors.exp(value(st))*(One(V)+t))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(mt))))))
        return AbstractTensors.exp(value(st))*(hint<0 ? AbstractTensors.cos(θ)+mt*(AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+mt*(AbstractTensors.:/(AbstractTensors.sinh(θ),θ)))
    else
        return One(V)+expm1(t)
    end
end

@pure isR301(V::DiagonalForm) = DirectSum.diagonalform(V) == Values(1,1,1,0)
@pure isR301(::Submanifold{V}) where V = isR301(V)
@pure isR301(V) = false

@inline unabs!(t) = t
@inline unabs!(t::Expr) = (t.head == :call && t.args[1] == :abs) ? t.args[2] : t

function Base.exp(t::T,::Val{hint}) where T<:TensorGraded{V} where {V,hint}
    S = T<:Submanifold
    i = T<:TensorTerm ? basis(t) : t
    sq = i⟑i
    if isscalar(sq)
        isnull(hint) && (return One(V)+t)
        grade(t)==0 && (return Single{V}(AbstractTensors.exp(value(S ? t : scalar(t)))))
        θ = unabs!(AbstractTensors.sqrt(AbstractTensors.abs(value(scalar(abs2(t))))))
        hint<0 ? AbstractTensors.cos(θ)+t*(S ? AbstractTensors.sin(θ) : AbstractTensors.:/(AbstractTensors.sin(θ),θ)) : AbstractTensors.cosh(θ)+t*(S ? AbstractTensors.sinh(θ) : AbstractTensors.:/(AbstractTensors.sinh(θ),θ))
    else
        return One(V)+expm1(t)
    end
end

@inline Base.expm1(A::Chain{V,G,<:Chain{V,G}},_=nothing) where {V,G} = exp(A)-I
@inline Base.exp(A::Chain{V,G,<:Chain{V,G},1},_=nothing) where {V,G} = Chain{V,G}(Values(Chain{V,G}(exp(A[1][1]))))

@inline function Base.exp(A::Chain{V,G,Chain{V,G,<:Real,2},2},_=nothing) where {V,G}
    T = typeof(exp(zero(valuetype(A))))
    @inbounds a = A[1][1]
    @inbounds c = A[1][2]
    @inbounds b = A[2][1]
    @inbounds d = A[2][2]
    v = (a-d)^2 + 4*b*c
    if v > 0
        z = sqrt(v)
        z1 = cosh(z / 2)
        z2 = sinh(z / 2) / z
    elseif v < 0
        z = sqrt(-v)
        z1 = cos(z / 2)
        z2 = sin(z / 2) / z
    else # if v == 0
        z1 = T(1.0)
        z2 = T(0.5)
    end
    r = exp((a + d) / 2)
    m11 = r * (z1 + (a - d) * z2)
    m12 = r * 2b * z2
    m21 = r * 2c * z2
    m22 = r * (z1 - (a - d) * z2)
    Chain{V,G}(Chain{V,G}(m11, m21), Chain{V,G}(m12, m22))
end

@inline function Base.exp(A::Chain{V,G,Chain{V,G,<:Complex,2},2},_=nothing) where {V,G}
    T = typeof(exp(zero(valuetype(A))))
    @inbounds a = A[1][1]
    @inbounds c = A[1][2]
    @inbounds b = A[2][1]
    @inbounds d = A[2][2]
    z = sqrt((a - d)*(a - d) + 4*b*c )
    e = expm1((a + d - z) / 2)
    f = expm1((a + d + z) / 2)
    ϵ = eps()
    g = abs2(z) < ϵ^2 ? exp((a + d) / 2) * (1 + z^2 / 24) : (f - e) / z
    m11 = (g * (a - d) + f + e) / 2 + 1
    m12 = g * b
    m21 = g * c
    m22 = (-g * (a - d) + f + e) / 2 + 1
    Chain{V,G}(Chain{V,G}(m11, m21), Chain{V,G}(m12, m22))
end

# Adapted from implementation in Base; algorithm from
# Higham, "Functions of Matrices: Theory and Computation", SIAM, 2008
function Base.exp(_A::Chain{W,G,Chain{W,G,T,N},N},_=nothing) where {W,G,T,N}
    S = typeof((zero(T)*zero(T) + zero(T)*zero(T))/one(T))
    A = Chain{W,G}(map.(S,value(_A)))
    # omitted: matrix balancing, i.e., LAPACK.gebal!
    nA = maximum(sum.(value.(map.(abs,value(A)))))
    # marginally more performant than norm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        A2 = A*A
        if nA > 0.95
            U = S(8821612800)*I+A2*(S(302702400)*I+A2*(S(2162160)*I+A2*(S(3960)*I+A2)))
            U = A*U
            V = S(17643225600)*I+A2*(S(2075673600)*I+A2*(S(30270240)*I+A2*(S(110880)*I+S(90)*A2)))
        elseif nA > 0.25
            U = S(8648640)*I+A2*(S(277200)*I+A2*(S(1512)*I+A2))
            U = A*U
            V = S(17297280)*I+A2*(S(1995840)*I+A2*(S(25200)*I+S(56)*A2))
        elseif nA > 0.015
            U = S(15120)*I+A2*(S(420)*I+A2)
            U = A*U
            V = S(30240)*I+A2*(S(3360)*I+S(30)*A2)
        else
            U = S(60)*I+A2
            U = A*U
            V = S(120)*I+S(12)*A2
        end
        expA = (V - U) \ (V + U)
    else
        s  = log2(nA/5.4)               # power of 2 later reversed by squaring
        if s > 0
            si = ceil(Int,s)
            A = A / S(2^si)
        end
        A2 = A*A
        A4 = A2*A2
        A6 = A2*A4
        U = A6*(A6 + S(16380)*A4 + S(40840800)*A2) +
            (S(33522128640)*A6 + S(10559470521600)*A4 + S(1187353796428800)*A2) +
            S(32382376266240000)*I
        U = A*U
        V = A6*(S(182)*A6 + S(960960)*A4 + S(1323241920)*A2) +
            (S(670442572800)*A6 + S(129060195264000)*A4 + S(7771770303897600)*A2) +
            S(64764752532480000)*I
        expA = (V - U) \ (V + U)
        if s > 0            # squaring to reverse dividing by power of 2
            for t=1:si
                expA = expA*expA
            end
        end
    end
    expA
end

for (op,logm,field) ∈ ((:⟑,:(Base.log),false),(:wedgedot_metric,:log_metric,true));
    args = field ? (:g,) : ()
    indu(t=:(log(t))) = field ? :(isinduced(g) && (return :($$t))) : nothing
@eval qlog(b::PseudoCouple,$(args...),x::Int=10000) = qlog(multispin(b),$(args...),x)
@eval qlog(b::CoSpinor,$(args...),x::Int=10000) = qlog(Multivector(b),$(args...),x)
@eval function qlog(w::T,$(args...),x::Int=10000) where T<:TensorAlgebra
    $(indu(:(qlog(w,x))))
    V = Manifold(w)
    w2,f = $op(w,w,$(args...)),norm(w)
    prod = $op(w,w2,$(args...))
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

@eval qlog_fast(b::PseudoCouple,$(args...),x::Int=10000) = qlog_fast(multispin(b),$(args...),x)
@eval qlog_fast(b::CoSpinor,$(args...),x::Int=10000) = qlog_fast(Multivector(b),$(args...),x)
for pinor ∈ (:Multivector,:Spinor); VEC = pinor≠:Spinor ? :mvec : :mvecs
@eval @generated function qlog_fast(b::$pinor{V,T,E},$(args...),x::Int=10000) where {V,T,E}
    $(indu(:(qlog_fast(b,x))))
    loop = $(pinor≠:Spinor ? :generate_loop_multivector : :generate_loop_spinor)(V,:prod,:B,promote_type(T,Float64),:*,$(QuoteNode(pinor≠:Spinor ? :geomaddmulti! : :geomaddspin!)),$(pinor≠:Spinor ? :geomaddmulti!_pre : :geomaddspin!_pre),$field)
    return quote
        $(insert_expr(loop[1],$(QuoteNode(VEC)),:T,Float64)...)
        f = norm(b)
        w2::$pinor{V,T,E} = $op(b,b,$(args...))
        B = value(w2)
        S = zeros($VEC(N,t))
        prod = zeros($VEC(N,t))
        term = zeros($VEC(N,t))
        S .= value(b)
        out .= value($op(b,w2,$(args...)))
        term .= out/3
        norms = FixedVector{3}(f,norm(term),f)
        k::Int = 5
        @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ x
            S += term
            ns = norm(S)
            @inbounds ns ≈ norms[3] && break
            prod .= out
            # prod *= w2
            $(loop[2])
            term .= out/k
            @inbounds norms .= (norms[2],norm(term),ns)
            k += 2
        end
        S *= 2
        return $pinor{V}(S)
    end
end
end

@eval begin
    $logm(A::Chain{V,G,<:Chain{V,G}},$(args...)) where {V,G} = Chain{V,G,Chain{V,G}}(log(Matrix(A)))
    $logm(t::TensorTerm,$(args...)) = $logm(Couple(t),$(args...))
    $logm(t::Phasor,$(args...)) = (r=radius(t,$(args...)); log(r)+angle(t,$(args...)))
    Base.log1p(t::Phasor{V},$(args...)) where V = $logm(One(V)+t,$(args...))
    $logm(t::Couple{V,B},$(args...)) where {V,B} = value($op(B,B,$(args...)))==-1 ? Couple{V,B}(log(Complex(t))) : $logm(radius(t,$(args...)))+angle(t,$(args...))
    Base.log1p(t::Couple{V,B},$(args...)) where {V,B} = value($op(B,B,$(args...)))==-1 ? Couple{V,B}(log1p(Complex(t))) : $logm(One(V)+t,$(args...))
    $logm(t::Quaternion{V},$(args...)) where V = iszero(metric(V)) ? (r=radius(t,$(args...)); $logm(radius(t,$(args...)))+angle(t,$(args...),r)) : qlog(/(t-One(V),t+One(V),$(args...)),$(args...))
    Base.log1p(t::Quaternion{V},$(args...)) where V = iszero(metric(V)) ? $logm(One(V)+t,$(args...)) : qlog(/(t,t+2,$(args...)),$(args...))
    @inline $logm(t::T,$(args...)) where T<:TensorAlgebra{V} where V = qlog(/(t-One(V),t+One(V),$(args...)),$(args...))
    @inline Base.log1p(t::T,$(args...)) where T<:TensorAlgebra = qlog(/(t,t+2,$(args...)),$(args...))
end

@eval function $logm(t::PseudoCouple{V,B},$(args...)) where {V,B}
    $(indu(:(log(t))))
    if isscalar(B)
        out = $logm(Couple{V,Submanifold(V)}(realvalue(t),imagvalue(t)),$(args...))
        PseudoCouple{V,B}(realvalue(out),imagvalue(out))
    else
        $logm(multispin(t),$(args...))
    end
end
@eval function Base.log1p(t::PseudoCouple{V,B},$(args...)) where {V,B}
    $(indu(:(log1p(t))))
    if isscalar(B)
        out = log1p(Couple{V,Submanifold(V)}(realvalue(t),imagvalue(t)),$(args...))
        PseudoCouple{V,B}(realvalue(out),imagvalue(out))
    else
        log1p(multispin(t),$(args...))
    end
end

@eval begin
    Base.exp(t::CoSpinor,$(args...)) = exp(Multivector(t),$(args...))
    Base.expm1(t::CoSpinor,$(args...)) = expm1(Multivector(t),$(args...))
    $logm(t::CoSpinor,$(args...)) = $logm(Multivector(t),$(args...))
    Base.log1p(t::CoSpinor,$(args...)) = log1p(Multivector(t),$(args...))
    log_fast(t::CoSpinor,$(args...)) = log_fast(Multivector(t),$(args...))
    logh_fast(t::CoSpinor,$(args...)) = logh_fast(Multivector(t),$(args...))
end
for op ∈ (:cosh,:sinh)
    @eval begin
        Base.$op(t::PseudoCouple,$(args...)) = $op(multispin(t),$(args...))
        Base.$op(t::CoSpinor,$(args...)) = $op(Multivector(t),$(args...))
    end
end

for op ∈ (logm,:(Base.exp),:(Base.asin),:(Base.acos),:(Base.atan),:(Base.acot),:(Base.sinc),:(Base.cosc))
    @eval @inline $op(t::T,$(args...)) where T<:TensorGraded{V,0} where V = Single{V}($op(value(t)))
end

for op ∈ (logm,:(Base.log2),:(Base.log10),:(Base.asech),:(Base.acosh),:(Base.acos),:(Basesinc))
    @eval @inline $op(::One{V},$(args...)) where V = Zero(V)
end
for op ∈ (:atanh,:acoth)
    @eval @inline Base.$op(::One{V},$(args...)) where V = Infinity(V)
end

@eval @inline Base.sinh(::Zero{V},$(args...)) where V = Zero(V)
for op ∈ (:exp,:exp2,:exp10,:cosh,:sinc) # exp
    @eval @inline Base.$op(::Zero{V},$(args...)) where V = One(V)
end
for op ∈ (:asin,:atan,:asinh,:atanh,:cosc,:sqrt,:cbrt)
    @eval @inline Base.$op(t::Zero,$(args...)) = t
end

for op ∈ (:tanh,:coth)
    @eval @inline Base.$op(::Infinity{V},$(args...)) where V = One(V)
end
for op ∈ (:acoth,:acot,:sinc,:cosc)
    @eval @inline Base.$op(::Infinity{V},$(args...)) where V = Zero(V)
end
for op ∈ (:(Base.exp),:(Base.exp2),:(Base.exp10),logm,:(Base.log2),:(Base.log10),:(Base.cosh),:(Base.sinh),:(Base.acosh),:(Base.asinh),:(Base.sqrt),:(Base.cbrt))
    @eval @inline $op(t::Infinity,$(args...)) = t
end

for (qrt,n) ∈ ((:sqrt,2),(:cbrt,3))
    @eval begin
        @inline function Base.$qrt(t::T,$(args...)) where T<:TensorAlgebra
            isscalar(t) ? $qrt(scalar(t)) : exp(log(t,$(args...))/$n,$(args...))
        end
        @inline function Base.$qrt(t::Quaternion{V},$(args...)) where V
            iszero(metric(V)) ? $qrt(radius(t))*exp(angle(t)/$n,$(args...)) : exp(log(t,$(args...))/$n,$(args...))
        end
        @inline function Base.$qrt(t::Couple{V,B},$(args...)) where {V,B}
            value(B*B)==-1 ? Couple{V,B}($qrt(Complex(t))) :
                $qrt(radius(t))*exp(angle(t)/$n,$(args...))
        end
        @inline Base.$qrt(t::Phasor,$(args...)) = Phasor($qrt(radius(t)),angle(t)/$n)
        @inline Base.$qrt(t::Submanifold{V,0} where V,$(args...)) = t
        @inline Base.$qrt(t::T,$(args...)) where T<:TensorGraded{V,0} where V = Single{V}($Sym.$qrt(value(T<:TensorTerm ? t : scalar(t))))
    end
end

## trigonometric

@eval @inline Base.cosh(t::T,$(args...)) where T<:TensorGraded{V,0} where V = Single{Manifold(t)}(AbstractTensors.cosh(value(T<:TensorTerm ? t : scalar(t))))

@eval function Base.cosh(t::T,$(args...)) where T<:TensorAlgebra
    $(indu(:(cosh(t))))
    V = Manifold(t)
    if T<:Couple
        B = basis(t); BB = value($op(B,B,$(args...)))
        if BB == -1
            return Couple{V,B}(cosh(t.v))
        end
    end
    τ = $op(t,t,$(args...))
    S,term = τ/2,$op(τ,τ,$(args...))/24
    f = norm(S)
    norms = FixedVector{3}(f,norm(term),f)
    k::Int = 6
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        $(field ? :(term = $op(term,τ/(k*(k-1)),g)) : :(term *= τ/(k*(k-1))))
        @inbounds norms .= (norms[2],norm(term),ns)
        k += 2
    end
    return One(V)+S
end

for pinor ∈ (:Multivector,:Spinor); VEC = pinor≠:Spinor ? :mvec : :mvecs
@eval @generated function Base.cosh(b::$pinor{V,T,E},$(args...)) where {V,T,E}
    $(indu(:(cosh(b))))
    loop = $(pinor≠:Spinor ? :generate_loop_multivector : :generate_loop_spinor)(V,:term,:B,promote_type(T,Float64),:*,$(QuoteNode(pinor≠:Spinor ? :geomaddmulti! : :geomaddspin!)),$(pinor≠:Spinor ? :geomaddmulti!_pre : :geomaddspin!_pre),$field,:(k*(k-1)))
    return quote
        sb,nb = scalar(b),norm(b)
        sb ≈ nb && (return Single{V}(AbstractTensors.cosh(value(sb))))
        $(insert_expr(loop[1],$(QuoteNode(VEC)),:T,Float64)...)
        τ::Multivector{V,T,E} = $op(b,b,$(args...))
        B = value(τ)
        S = zeros($VEC(N,t))
        term = zeros($VEC(N,t))
        S .= value(τ)/2
        out .= value($op(τ,τ,$(args...)))/24
        norms = FixedVector{3}(norm(S),norm(out),norm(term))
        k::Int = 6
        @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
            S += out
            ns = norm(S)
            @inbounds ns ≈ norms[3] && break
            term .= out
            # term *= τ/(k*(k-1))
            $(loop[2])
            @inbounds norms .= (norms[2],norm(out),ns)
            k += 2
        end
        @inbounds S[1] += One(V)
        return $pinor{V}(S)
    end
end
end

@eval @inline Base.sinh(t::T,$(args...)) where T<:TensorGraded{V,0} where V = Single{Manifold(t)}(AbstractTensors.sinh(value(T<:TensorTerm ? t : scalar(t))))

@eval  function Base.sinh(t::T,$(args...)) where T<:TensorAlgebra
    $(indu(:(sinh(t))))
    V = Manifold(t)
    if T<:Couple
        B = basis(t); BB = value($op(B,B,$(args...)))
        if BB == -1
            return Couple{V,B}(sinh(t.v))
        end
    end
    τ,f = $op(t,t,$(args...)),norm(t)
    S,term = t,$op(t,τ,$(args...))/6
    norms = FixedVector{3}(f,norm(term),f)
    k::Int = 5
    @inbounds while norms[2]<norms[1] || norms[2]>1
        S += term
        ns = norm(S)
        @inbounds ns ≈ norms[3] && break
        $(field ? :(term = $op(term,τ/(k*(k-1)),g)) : :(term *= τ/(k*(k-1))))
        @inbounds norms .= (norms[2],norm(term),ns)
        k += 2
    end
    return S
end

for pinor ∈ (:Multivector,:Spinor); VEC = pinor≠:Spinor ? :mvec : :mvecs
@eval @generated function Base.sinh(b::$pinor{V,T,E},$(args...)) where {V,T,E}
    $(indu(:(sinh(b))))
    loop = $(pinor≠:Spinor ? :generate_loop_multivector : :generate_loop_spinor)(V,:term,:B,promote_type(T,Float64),:*,$(QuoteNode(pinor≠:Spinor ? :geomaddmulti! : :geomaddspin!)),$(pinor≠:Spinor ? :geomaddmulti!_pre : :geomaddspin!_pre),$field,:(k*(k-1)))
    return quote
        sb,nb = scalar(b),norm(b)
        sb ≈ nb && (return Single{V}(AbstractTensors.sinh(value(sb))))
        $(insert_expr(loop[1],$(QuoteNode(VEC)),:T,Float64)...)
        τ::Multivector{V,T,E} = $op(b,b,$(args...))
        B = value(τ)
        S = zeros($VEC(N,t))
        term = zeros($VEC(N,t))
        S .= value(b)
        out .= value($op(b,τ,$(args...)))/6
        norms = FixedVector{3}(norm(S),norm(out),norm(term))
        k::Int = 5
        @inbounds while (norms[2]<norms[1] || norms[2]>1) && k ≤ 10000
            S += out
            ns = norm(S)
            @inbounds ns ≈ norms[3] && break
            term .= out
            # term *= τ/(k*(k-1))
            $(loop[2])
            @inbounds norms .= (norms[2],norm(out),ns)
            k += 2
        end
        return $pinor{V}(S)
    end
end
end

@eval exph(t,$(args...)) = Base.cosh(t,$(args...))+Base.sinh(t,$(args...))

for (logfast,expf) ∈ ((:log_fast,:exp),(:logh_fast,:exph))
    @eval function $logfast(t::T,$(args...)) where T<:TensorAlgebra
        V = Manifold(t)
        term = Zero(V)
        nrm = FixedVector{2}(0.,0.)
        while true
            en = $expf(term,$(args...))
            term -= /(2(en-t),en+t,$(args...))
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

@eval function Base.angle(z::Couple{V,B},$(args...)) where {V,B}
    if value($op(B,B,$(args...))) == -1
        atan(imagvalue(z),realvalue(z))*B
    elseif value($op(B,B,$(args...))) == 1
        atanh(imagvalue(z),realvalue(z))*B
    else
        error("Unsupported trigonometric angle")
    end
end

@eval radius(z::Quaternion,$(args...)) = value(scalar(abs(z,$(args...))))
@eval function Base.angle(z::Quaternion,$(args...),r::Real=radius(z,$(args...)))
    b = bivector(z)
    (acos(value(scalar(z))/r)/value(abs(b,$(args...))))*b
end
end

Base.atanh(y::Real, x::Real) = atanh(promote(float(y),float(x))...)
Base.atanh(y::T, x::T) where {T<:AbstractFloat} = Base.no_op_err("atanh", T)

function Base.atanh(y::T, x::T) where T<:Union{Float32, Float64}
    # Method :
    #    M1) Reduce y to positive by atanh2(y,x)=-atanh2(-y,x).
    #    M2) Reduce x to positive by (if x and y are unexceptional):
    #        ARGH (x+iy) = arctanh(y/x)     ... if x > 0,
    #        ARGH (x+iy) = arctanh[y/(-x)]  ... if x < 0,
    #
    # Special cases:
    #
    #    S1) ATANH2((anything), NaN ) is NaN;
    #    S2) ATANH2(NAN , (anything) ) is NaN;
    #    S3) ATANH2(+-0, +-(anything but NaN)) is +-0;
    #    S4) ATANH2(+-(anything but 0 and NaN), 0) is ERROR;
    #    S5) ATANH2(+-(anything but INF and NaN), +-INF) is +-0;
    #    S6) ATANH2(+-INF,+-INF ) is +-Inf;
    #    S7) ATANH2(+-INF, (anything but,0,NaN, and INF)) is ERROR;
    if isnan(x) || isnan(y) # S1 or S2
        return T(NaN)
    end

    if x == T(1.0) || x == T(-1.0) # then y/x = y, see M2
        return atanh(y)
    end
    # generate an m ∈ {0, 1, 2, 3} to branch off of
    m = 2*signbit(x) + 1*signbit(y)

    if iszero(y)
        return y # atanh(+-0, +-anything) = +-0
    elseif iszero(x)
        return atanh(copysign(T(Inf), y))
    end

    if isinf(x)
        if isinf(y)
            return y # atanh(+-Inf, +-Inf)
        else
            if m == 0 || m == 2
                return zero(T)  # atanh(+...,+-Inf) */
            elseif m == 1 || m == 3
                return -zero(T) # atanh(-...,+-Inf) */
            end
        end
    end

    # x wasn't Inf, but y is
    isinf(y) && return atanh(y)

    ypw = Base.Math.poshighword(y)
    xpw = Base.Math.poshighword(x)
    # compute y/x for Float32
    k = reinterpret(Int32, ypw-xpw)>>Base.Math.ATAN2_RATIO_BIT_SHIFT(T)

    if k > Base.Math.ATAN2_RATIO_THRESHOLD(T) # |y/x| >  threshold
        z=T(pi)/2+T(0.5)*Base.Math.ATAN2_PI_LO(T)
        m&=1;
    elseif x<0 && k < -Base.Math.ATAN2_RATIO_THRESHOLD(T) # 0 > |y|/x > threshold
        z = zero(T)
    else #safe to do y/x
        z = atanh(abs(y/x))
    end

    if m == 0 || m == 2
        return z # atanh(+,+-)
    else # default case m == 1 || m == 3
        return -z # atanh(-,+-)
    end
end

function Cramer(N::Int,j=0)
    t = j ≠ 0 ? :T : :t
    x,y = Values{N}([Symbol(:x,i) for i ∈ list(1,N)]),Values{N}([Symbol(:y,i) for i ∈ list(1,N)])
    xy = [:(($(x[1+i]),$(y[1+i])) = ($(x[i])∧$t[$(1+i-j)],$t[end-$i]∧$(y[i]))) for i ∈ list(1,N-1-j)]
    return x,y,xy
end

DirectSum.Λ(x::Chain{V,1,<:Chain{V,1}},G) where V = compound(x,G)
compound(x,G::T) where T<:Integer = compound(x,Val(G))
compound(x::Chain{V,1,<:Chain{V,1}},::Val{0}) where V = Chain{V,0}(Values(Chain{V,0}(1)))
@generated function compound(x::Chain{V,1,<:Chain{V,1}},::Val{G}) where {V,G}
    Expr(:call,:(Chain{V,G}),Expr(:call,:Values,[Expr(:call,:∧,[:(@inbounds x[$i]) for i ∈ indices(j)]...) for j ∈ indexbasis(mdims(V),G)]...))
end

@generated function Base.:\(t::Values{M,<:Chain{V,1}},v::Chain{V,1}) where {M,V}
    W = M≠mdims(V) ? Submanifold(M) : V; N = M-1
    if M == 1 && (V === ℝ1 || V == 1)
        return :(@inbounds Chain{V,1}(Values(v[1]/t[1][1])))
    elseif M == 2 && (V === ℝ2 || V == 2)
        return quote
            @inbounds (a,A),(b,B),(c,C) = value(t[1]),value(t[2]),value(v)
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
    mid = [:($(x[i])∧v∧$(y[end-i])) for i ∈ list(1,N-1)]
    out = Expr(:call,:Values,:(v∧$(y[end])),mid...,:($(x[end])∧v))
    detx = :(detx = @inbounds (t[1]∧$(y[end])))
    return Expr(:block,:((x1,y1)=@inbounds (t[1],t[end])),xy...,detx,
        :(Chain{$W,1}(column($(Expr(:call,:.⋅,out,:(Ref(detx))))./abs2(detx)))))
end

@generated function Base.in(v::Chain{V,1},t::Values{N,<:Chain{V,1}}) where {V,N}
    if N == mdims(V)
        x,y,xy = Grassmann.Cramer(N-1)
        mid = [:(s==signbit(@inbounds ($(x[i])∧v∧$(y[end-i]))[1])) for i ∈ list(1,N-2)]
        out = Values(:(s==signbit(@inbounds (v∧$(y[end]))[1])),mid...,:(s==signbit(@inbounds ($(x[end])∧v)[1])))
        return Expr(:block,:((x1,y1)=@inbounds (t[1],t[end])),xy...,:(s=signbit(@inbounds (t[1]∧$(y[end]))[1])),ands(out))
    else
        x,y,xy = Grassmann.Cramer(N-1,1)
        mid = [:(signscalar(($(x[i])∧(v-x1)∧$(y[end-i]))/d)) for i ∈ list(1,N-2)]
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
        Expr(:call,:Values,y[end],[:($(y[end-i])∧$(x[i])) for i ∈ list(1,N-1)]...,x[end])
    elseif M≠mdims(V)
        Expr(:call,:Values,y[end],[:($(iseven(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ list(1,N-1)]...,:(-$(x[end])))
    else
        Expr(:call,:Values,:(-$(y[end])),[:($(isodd(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ list(1,N-1)]...,x[end])
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
        Expr(:call,:Values,[:($(y[end-i])∧$(x[i])) for i ∈ list(1,N-1)]...,x[end])
    elseif M≠mdims(V)
        Expr(:call,:Values,y[end],[:($(iseven(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ list(1,N-1)]...,:(-$(x[end])))
    else
        Expr(:call,:Values,[:($(isodd(i) ? :+ : :-)($(y[end-i])∧$(x[i]))) for i ∈ list(1,N-1)]...,x[end])
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

Base.:\(t::LinearAlgebra.UniformScaling,v::Chain{V,G,<:Chain}) where {V,G} = inv(v)#value(Chain{V,G}(t))\v
Base.:\(t::Chain{M,1,<:Chain{W,1}},v::Chain{V,1,<:Chain}) where {M,W,V} = t*inv(v)#Chain{V,1}(t.\value(v))
Base.:\(t::Chain{M,1,<:Chain{W,1}},v::Chain{V,1}) where {M,W,V} = value(t)\v
Base.in(v::Chain{V,1},t::Chain{W,1,<:Chain{V,1}}) where {V,W} = v ∈ value(t)
#Base.inv(t::Chain{V,1,<:Chain{V,G}}) where {V,G} = value(Chain{V,G}(I))\t
Base.inv(t::Chain{V,1,<:Chain{W,1}}) where {W,V} = inv(value(t))
grad(t::Chain{V,1,<:Chain{W,1}}) where {V,W} = grad(value(t))

@generated approx(x,y::Chain{V}) where V = :(polynom(x,$(Val(mdims(V))))⋅y)
approx(x,y::Values{N}) where N = value(polynom(x,Val(N)))⋅y
approx(x,y::AbstractVector) = [x^i for i ∈ 0:length(y)-1]⋅y

vandermonde(x::Array,y::Array,N::Int) = vandermonde(x,N)\y[:] # compute ((inv(X'*X))*X')*y
function vandermonde(x::Array,N)
    V = zeros(length(x),N)
    for d ∈ list(0,N-1)
        V[:,d+1] = x.^d
    end
    return V # Vandermonde
end

vandermonde(x,y,V) = (length(x)≠mdims(V) ? _vandermonde(x,V) : vandermonde(x,V))\y
vandermonde(x,V) = transpose(_vandermonde(x,V))
_vandermonde(x::Chain,V) = _vandermonde(value(x),V)
@generated _vandermonde(x::Values{N},V) where N = :(Chain{$(Submanifold(N)),1}(polynom.(x,$(Val(mdims(V))))))
@generated polynom(x,::Val{N}) where N = Expr(:call,:(Chain{$(Submanifold(N)),1}),Expr(:call,:Values,[:(x^$i) for i ∈ list(0,N-1)]...))

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
    v = Expr(:tuple,[:(M.(p[c[$i]]-A)) for i ∈ list(2,mdims(t))]...)
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
        Base.$op(a::Couple{V,B},m) where {V,B} = Couple{V,B}($op(value(a),m))
        Base.$op(a::PseudoCouple{V,B},m) where {V,B} = PseudoCouple{V,B}($op(value(a),m))
        Base.$op(a::Chain{V,G,T},m) where {V,G,T} = Chain{V,G}($op.(value(a),m))
        Base.$op(a::Spinor{V,T},m) where {T,V} = Spinor{V}($op.(value(a),m))
        Base.$op(a::CoSpinor{V,T},m) where {T,V} = CoSpinor{V}($op.(value(a),m))
        Base.$op(a::Multivector{V,T},m) where {T,V} = Multivector{V}($op.(value(a),m))
    end
end
for op ∈ (:mod2pi,:rem2pi,:rad2deg,:deg2rad,:round)
    @eval begin
        Base.$op(a::Couple{V,B};args...) where {V,B} = Couple{V,B}($op(value(a);args...))
        Base.$op(a::PseudoCouple{V,B};args...) where {V,B} = PseudoCouple{V,B}($op(value(a);args...))
        Base.$op(a::Chain{V,G,T};args...) where {V,G,T} = Chain{V,G}($op.(value(a);args...))
        Base.$op(a::Spinor{V,T};args...) where {V,T} = Spinor{V}($op.(value(a);args...))
        Base.$op(a::CoSpinor{V,T};args...) where {V,T} = CoSpinor{V}($op.(value(a);args...))
        Base.$op(a::Multivector{V,T};args...) where {V,T} = Multivector{V}($op.(alue(a);args...))
    end
end
Base.isfinite(a::Chain) = prod(isfinite.(value(a)))
Base.isfinite(a::Spinor) = prod(isfinite.(value(a)))
Base.isfinite(a::CoSpinor) = prod(isfinite.(value(a)))
Base.isfinite(a::Multivector) = prod(isfinite.(value(a)))
Base.isfinite(a::Couple) = isfinite(value(a))
Base.isfinite(a::PseudoCouple) = isfinite(value(a))
Base.rationalize(t::Type,a::Chain{V,G,T};tol::Real=eps(T)) where {V,G,T} = Chain{V,G}(rationalize.(t,value(a),tol))
Base.rationalize(t::Type,a::Multivector{V,T};tol::Real=eps(T)) where {V,T} = Multivector{V}(rationalize.(t,value(a),tol))
Base.rationalize(t::Type,a::Spinor{V,T};tol::Real=eps(T)) where {V,T} = Spinor{V}(rationalize.(t,value(a),tol))
Base.rationalize(t::Type,a::CoSpinor{V,T};tol::Real=eps(T)) where {V,T} = CoSpinor{V}(rationalize.(t,value(a),tol))
Base.rationalize(t::Type,a::Couple{V,B};tol::Real=eps(T)) where {V,B} = Couple{V,B}(rationalize(t,value(a),tol))
Base.rationalize(t::Type,a::PseudoCouple{V,B};tol::Real=eps(T)) where {V,B} = PseudoCouple{V,B}(rationalize(t,value(a),tol))
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
Base.map(fn, x::CoSpinor{V}) where V = CoSpinor{V}(map(fn, value(x)))
Base.map(fn, x::Chain{V,G}) where {V,G} = Chain{V,G}(map(fn,value(x)))
Base.map(fn, x::TensorTerm) = fn(value(x))*basis(x)
Base.map(fn, x::Couple{V,B}) where {V,B} = Couple{V,B}(Complex(fn(x.v.re),fn(x.v.im)))
Base.map(fn, x::PseudoCouple{V,B}) where {V,B} = PseudoCouple{V,B}(Complex(fn(x.v.re),fn(x.v.im)))

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
Base.rand(::AbstractRNG,::SamplerType{CoSpinor}) = rand(CoSpinor{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{CoSpinor{V}}) where V = CoSpinor{V}(DirectSum.orand(svecs(mdims(V),Float64)))
Base.rand(::AbstractRNG,::SamplerType{CoSpinor{V,T}}) where {V,T} = CoSpinor{V}(rand(svecs(mdims(V),T)))
Base.rand(::AbstractRNG,::SamplerType{Couple}) = rand(Couple{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Couple{V}}) where V = rand(Couple{V,Submanifold{V}(UInt(rand(1:1<<mdims(V)-1)))})
Base.rand(::AbstractRNG,::SamplerType{Couple{V,B}}) where {V,B} = Couple{V,B}(rand(Complex{Float64}))
Base.rand(::AbstractRNG,::SamplerType{Couple{V,B,T}}) where {V,B,T} = Couple{V,B}(rand(Complex{T}))
Base.rand(::AbstractRNG,::SamplerType{Couple{V,B,T} where B}) where {V,T} = rand(Couple{V,Submanifold{V}(UInt(rand(1:1<<mdims(V)-1))),T})
Base.rand(::AbstractRNG,::SamplerType{PseudoCouple}) = rand(PseudoCouple{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{PseudoCouple{V}}) where V = rand(PseudoCouple{V,Submanifold{V}(UInt(rand(0:(1<<mdims(V)-1)-1)))})
Base.rand(::AbstractRNG,::SamplerType{PseudoCouple{V,B}}) where {V,B} = PseudoCouple{V,B}(rand(Complex{Float64}))
Base.rand(::AbstractRNG,::SamplerType{PseudoCouple{V,B,T}}) where {V,B,T} = PseudoCouple{V,B}(rand(Complex{T}))
Base.rand(::AbstractRNG,::SamplerType{PseudoCouple{V,B,T} where B}) where {V,T} = rand(PseudoCouple{V,Submanifold{V}(UInt(rand(0:(1<<mdims(V)-1)-1))),T})
