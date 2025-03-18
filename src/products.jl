
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

@pure tvec(N,G,t=Any) = :(Values{$(binomial(N,G)),$t})
@pure tvec(N,t::Type=Any) = :(Values{$(1<<N),$t})
@pure tvec(N,t::Symbol) = :(Values{$(1<<N),$t})
@pure tvec(N,μ::Bool) = tvec(N,μ ? :Any : :t)
@pure tvecs(N,t::Type=Any) = :(Values{$(1<<(N-1)),$t})
@pure tvecs(N,t::Symbol) = :(Values{$(1<<(N-1)),$t})
@pure tvecs(N,μ::Bool) = tvecs(N,μ ? :Any : :t)

# mutating operations

@pure derive(n::N,b) where N<:Number = zero(typeof(n))
derive(n,b,a,t) = t ? (a,derive(n,b)) : (derive(n,b),a)
#derive(n,b,a::T,t) where T<:TensorAlgebra = t ? (a,derive(n,b)) : (derive(n,b),a)

@inline @generated function derive_mul(V,A,B,v,x::Bool)
    if !(istangent(V) && isdyadic(V))
        return :v
    else quote
        sa,sb = symmetricsplit(V,A),symmetricsplit(V,B)
        ca,cb = count_ones(@inbounds sa[2]),count_ones(@inbounds sb[2])
        return if (ca == cb == 0) || ((ca ≠ 0) && (cb ≠ 0))
            v
        else
            prev = ca == 0 ? (x ? one(typeof(v)) : v) : (x ? v : one(typeof(v)))
            for k ∈ Leibniz.indexsplit(@inbounds (ca==0 ? sa : sb)[1],mdims(V))
                prev = derive(prev,getbasis(V,k))
            end
            prev
        end
    end end
end

@inline @generated function derive_mul(V,A,B,a,b,*)
    if !(istangent(V) && isdyadic(V))
        return :(a*b)
    else quote
        sa,sb = symmetricsplit(V,A),symmetricsplit(V,B)
        ca,cb = count_ones(@inbounds sa[2]),count_ones(@inbounds sb[2])
        α,β = if (ca == cb == 0) || ((ca ≠ 0) && (cb ≠ 0))
            a,b
        else
            prev = ca == 0 ? (a,b) : (b,a)
            for k ∈ Leibniz.indexsplit(@inbounds (ca==0 ? sa : sb)[1],mdims(V))
                prev = @inbounds derive(prev[2],getbasis(V,k),prev[1],true)
            end
            #base = getbasis(V,0)
            while typeof(@inbounds prev[1]) <: TensorTerm
                basi = @inbounds basis(prev[1])
                #base *= basi
                inds = Leibniz.indexsplit(UInt(basi),mdims(V))
                prev = @inbounds (value(prev[1]),prev[2])
                for k ∈ inds
                    prev = @inbounds derive(prev[2],getbasis(V,k),prev[1],true)
                end
            end
            #base ≠ getbasis(V,0) && (prev = (base*prev[1],prev[2]))
            ca == 0 ? prev : (@inbounds prev[2],@inbounds prev[1])
        end
        return α*β
    end end
end

function derive_pre(V,A,B,v,x)
    if !(istangent(V) && isdyadic(V))
        return v
    else
        return :(derive_post($V,$(Val{A}()),$(Val{B}()),$v,$x))
    end
end

function derive_pre(V,A,B,a,b,p)
    if !(istangent(V) && isdyadic(V))
        return Expr(:call,p,a,b)
    else
        return :(derive_post($V,$(Val{A}()),$(Val{B}()),$a,$b,$p))
    end
end

function derive_post(V,::Val{A},::Val{B},v,x::Bool) where {A,B}
    sa,sb = symmetricsplit(V,A),symmetricsplit(V,B)
    ca,cb = count_ones(@inbounds sa[2]),count_ones(@inbounds sb[2])
    return if (ca == cb == 0) || ((ca ≠ 0) && (cb ≠ 0))
        v
    else
        prev = ca == 0 ? (x ? one(typeof(v)) : v) : (x ? v : one(typeof(v)))
        for k ∈ Leibniz.indexsplit(@inbounds (ca==0 ? sa : sb)[1],mdims(V))
            prev = derive(prev,getbasis(V,k))
        end
        prev
    end
end

function derive_post(V,::Val{A},::Val{B},a,b,*) where {A,B}
    sa,sb = symmetricsplit(V,A),symmetricsplit(V,B)
    ca,cb = count_ones(@inbounds sa[2]),count_ones(@inbounds sb[2])
    α,β = if (ca == cb == 0) || ((ca ≠ 0) && (cb ≠ 0))
        a,b
    else
        prev = ca == 0 ? (a,b) : (b,a)
        for k ∈ Leibniz.indexsplit(@inbounds (ca==0 ? sa : sb)[1],mdims(V))
            prev = @inbounds derive(prev[2],getbasis(V,k),prev[1],true)
        end
        #base = getbasis(V,0)
        while typeof(@inbounds prev[1]) <: TensorTerm
            basi = @inbounds basis(prev[1])
            #base *= basi
            inds = Leibniz.indexsplit(UInt(basi),mdims(V))
            prev = @inbounds (value(prev[1]),prev[2])
            for k ∈ inds
                prev = @inbounds derive(prev[2],getbasis(V,k),prev[1],true)
            end
        end
        #base ≠ getbasis(V,0) && (prev = (base*prev[1],prev[2]))
        ca == 0 ? prev : (@inbounds prev[2],@inbounds prev[1])
    end
    return α*β
end

bcast(op,arg) = op ∈ (:(AbstractTensors.:∑),:(AbstractTensors.:-)) ? Expr(:.,op,arg) : Expr(:call,op,arg.args...)

set_val(set,expr,val) = Expr(:(=),expr,set≠:(=) ? Expr(:call,:($Sym.:∑),expr,val) : val)

pre_val(set,expr,val) = set≠:(=) ? :(isnull($expr) ? ($expr=Expr(:call,:($Sym.:∑),$val)) : push!($expr.args,$val)) : Expr(:(=),expr,val)

add_val(set,expr,val,OP) = Expr(OP∉(:-,:+) ? :.= : set,expr,OP∉(:-,:+) ? Expr(:.,OP,Expr(:tuple,expr,val)) : val)

function generate_mutators(M,F,set_val,SUB,MUL,i,B)
    for (op,set) ∈ ((:add,:(+=)),(:set,:(=)))
        for (s,index) ∈ ((Symbol(op,:multi!),:basisindex),(Symbol(op,:blade!),:bladeindex),(Symbol(op,:spin!),:spinindex),(Symbol(op,:anti!),:antiindex))
            spre = Symbol(s,:_pre)
            @eval begin
                @inline function $s(out::$M,val::S,i::$B) where {M,T<:$F,S<:$F}
                    @inbounds $(set_val(set,:(out[$index(intlog(M),$i)]),:val))
                    return out
                end
                @inline function $s(out::Q,val::S,i::$B,::Val{N}) where Q<:$M where {M,T<:$F,S<:$F,N}
                    @inbounds $(set_val(set,:(out[$index(N,$i)]),:val))
                    return out
                end
                @inline function $spre(out::$M,val::S,i::$B) where {M,T<:$F,S<:$F}
                    ind = $index(intlog(M),$i)
                    @inbounds $(pre_val(set,:(out[ind]),:val))
                    return out
                end
                @inline function $spre(out::Q,val::S,i::$B,::Val{N}) where Q<:$M where {M,T<:$F,S<:$F,N}
                    ind = $index(N,$i)
                    @inbounds $(pre_val(set,:(out[ind]),:val))
                    return out
                end
            end
        end
    end
end

function generate_mutators(M,F,set_val,SUB,MUL)
    generate_mutators(M,F,set_val,SUB,MUL,:i,UInt)
    generate_mutators(M,F,set_val,SUB,MUL,:(UInt(i)),Submanifold)
    for (op,set) ∈ ((:add,:(+=)),(:set,:(=)))
        for s ∈ (Symbol(op,:multi!),Symbol(op,:blade!),Symbol(op,:spin!),Symbol(op,:anti!))
            @eval @inline function $s(out::$M,val::S,i) where {M,T,S}
                @inbounds $(set_val(set,:(out[i]),:val))
                return out
            end
            spre = Symbol(s,:_pre)
            for j ∈ (:join,:geom)
                for S ∈ (s,spre)
                    @eval @inline function $(Symbol(j,S))(m::$M,v::S,A::Submanifold{V},B::Submanifold{V}) where {V,T<:$F,S<:$F,M}
                        $(Symbol(j,S))(V,m,UInt(A),UInt(B),v)
                    end
                end
            end
            @eval begin
                @inline function $(Symbol(:join,s))(V,m::$M,a::UInt,b::UInt,v::S) where {T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        val = $MUL(parityinner(grade(V),A,B),v)
                        if diffvars(V)≠0
                            !iszero(Z) && (T≠Any ? (return true) : (val *= getbasis(loworder(V),Z)))
                            count_ones(Q)+order(val)>diffmode(V) && (return false)
                        end
                        $s(m,val,(A⊻B)|Q,Val(mdims(V)))
                    end
                    return false
                end
                @inline function $(Symbol(:join,spre))(V,m::$M,a::UInt,b::UInt,v::S,field=nothing) where {T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        val = :($$MUL($(parityinner(grade(V),A,B)),$v))
                        if diffvars(V)≠0
                            !iszero(Z) && (val = Expr(:call,:*,val,getbasis(loworder(V),Z)))
                            val = :(h=$val;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                        end
                        $spre(m,val,(A⊻B)|Q,Val(mdims(V)))
                    end
                    return false
                end
                @inline function $(Symbol(:geom,s))(V,m::$M,a::UInt,b::UInt,v::S) where {T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        basg = if isdiag(V)
                            Values(((A⊻B,parityinner(V,A,B)),))
                        else
                            paritygeometric(V,A,B)
                        end
                        for (bas,g) ∈ basg
                            val = $MUL(g,v)
                            if istangent(V)
                                !iszero(Z) && (T≠Any ? (return true) : (val *= getbasis(loworder(V),Z)))
                                count_ones(Q)+order(val)>diffmode(V) && (return false)
                            end
                            $s(m,val,bas|Q,Val(mdims(V)))
                        end
                    end
                    return false
                end
                @inline function $(Symbol(:geom,spre))(V,m::$M,a::UInt,b::UInt,v::S,vfield::Val{field}=Val(false)) where {T<:$F,S<:$F,M,field}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        basg = if isdiag(V)
                            Values(((A⊻B,parityinner(V,A,B,vfield)),))
                        else
                            paritygeometric(V,A,B,vfield)
                        end
                        for (bas,g) ∈ basg
                            val = :($$MUL($g,$v))
                            if istangent(V)
                                !iszero(Z) && (val = Expr(:call,:*,val,getbasis(loworder(V),Z)))
                                val = :(h=$val;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                            end
                            $spre(m,val,bas|Q,Val(mdims(V)))
                        end
                    end
                    return false
                end
            end
            for (prod,uct) ∈ ((:meet,:regressive),(:skew,:interior))
                for S ∈ (s,spre)
                    @eval @inline function $(Symbol(prod,S))(m::$M,A::Submanifold{V},B::Submanifold{V},v::T) where {V,T,M}
                        $(Symbol(prod,S))(V,m,UInt(A),UInt(B),v)
                    end
                end
            end
            @eval begin
                @inline function $(Symbol(:meet,s))(V,m::$M,A::UInt,B::UInt,val::T) where {T,M}
                    if val ≠ 0
                        g,C,t,Z = regressive(V,A,B)
                        v = val
                        if istangent(V) && !iszero(Z)
                            T≠Any && (return true)
                            _,_,Q,_ = symmetricmask(V,A,B)
                            v *= getbasis(loworder(V),Z)
                            count_ones(Q)+order(v)>diffmode(V) && (return false)
                        end
                        t && $s(m,$MUL(g,v),C,Val(mdims(V)))
                    end
                    return false
                end
                @inline function $(Symbol(:meet,spre))(V,m::$M,A::UInt,B::UInt,val::T,field::Val=Val(false)) where {T,M}
                    if val ≠ 0
                        g,C,t,Z = regressive(V,A,B)
                        v = val
                        if istangent(V) && !iszero(Z)
                            _,_,Q,_ = symmetricmask(V,A,B)
                            v = Expr(:call,:*,v,getbasis(loworder(V),Z))
                            v = :(h=$v;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                        end
                        t && $spre(m,Expr(:call,$(QuoteNode(MUL)),g,v),C,Val(mdims(V)))
                    end
                    return false
                end
                @inline function $(Symbol(:skew,s))(V,m::$M,A::UInt,B::UInt,val::T) where {T,M}
                    if val ≠ 0
                        if isdiag(V)
                            g,C,t,Z = interior(V,A,B)
                            v = val
                            if istangent(V) && !iszero(Z)
                                T≠Any && (return true)
                                _,_,Q,_ = symmetricmask(V,A,B)
                                v *= getbasis(loworder(V),Z)
                                count_ones(Q)+order(v)>diffmode(V) && (return false)
                            end
                            t && $s(m,$MUL(g,v),C,Val(mdims(V)))
                        else
                            Cg,Z = parityinterior(V,A,B,Val(true),false)
                            v = val
                            if istangent(V) && !iszero(Z)
                                T≠Any && (return true)
                                _,_,Q,_ = symmetricmask(V,A,B)
                                v *= getbasis(loworder(V),Z)
                                count_ones(Q)+order(v)>diffmode(V) && (return false)
                            end
                            for (C,g) ∈ Cg
                                $s(m,$MUL(g,v),C,Val(mdims(V)))
                            end
                        end
                    end
                    return false
                end
                @inline function $(Symbol(:skew,spre))(V,m::$M,A::UInt,B::UInt,val::T,field::Val=Val(false)) where {T,M}
                    if val ≠ 0
                        if isdiag(V)
                            g,C,t,Z = interior(V,A,B,Val(false),field)
                            v = val
                            if istangent(V) && !iszero(Z)
                                _,_,Q,_ = symmetricmask(V,A,B)
                                v = Expr(:call,:*,v,getbasis(loworder(V),Z))
                                v = :(h=$v;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                            end
                            t && $spre(m,Expr(:call,$(QuoteNode(MUL)),g,v),C,Val(mdims(V)))
                        else
                            Cg,Z = parityinterior(V,A,B,Val(true),field)
                            v = val
                            if istangent(V) && !iszero(Z)
                                _,_,Q,_ = symmetricmask(V,A,B)
                                v = Expr(:call,:*,v,getbasis(loworder(V),Z))
                                v = :(h=$v;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                            end
                            for (C,g) ∈ Cg
                                $spre(m,Expr(:call,$(QuoteNode(MUL)),g,v),C,Val(mdims(V)))
                            end
                        end
                    end
                    return false
                end


            end

        end
    end
end

@inline exterbits(V,α,β) = diffvars(V)≠0 ? ((a,b)=symmetricmask(V,α,β);iszero(a&b)) : iszero(α&β)
@inline exteraddmulti!(V,out,α,β,γ) = exterbits(V,α,β) && joinaddmulti!(V,out,α,β,γ)
@inline exteraddblade!(V,out,α,β,γ) = exterbits(V,α,β) && joinaddblade!(V,out,α,β,γ)
@inline exteraddspin!(V,out,α,β,γ) = exterbits(V,α,β) && joinaddspin!(V,out,α,β,γ)
@inline exteraddanti!(V,out,α,β,γ) = exterbits(V,α,β) && joinaddanti!(V,out,α,β,γ)
@inline exteraddmulti!_pre(V,out,α,β,γ,f=Val(false)) = exterbits(V,α,β) && joinaddmulti!_pre(V,out,α,β,γ,f)
@inline exteraddblade!_pre(V,out,α,β,γ,f=Val(false)) = exterbits(V,α,β) && joinaddblade!_pre(V,out,α,β,γ,f)
@inline exteraddspin!_pre(V,out,α,β,γ,f=Val(false)) = exterbits(V,α,β) && joinaddspin!_pre(V,out,α,β,γ,f)
@inline exteraddanti!_pre(V,out,α,β,γ,f=Val(false)) = exterbits(V,α,β) && joinaddanti!_pre(V,out,α,β,γ,f)

# algebra

const FieldsBig = (Fields...,BigFloat,BigInt,Complex{BigFloat},Complex{BigInt},Rational{BigInt})

*(a::UniformScaling,b::Single{V}) where V = V(a)*b
*(a::Single{V},b::UniformScaling) where V = a*V(b)
*(a::UniformScaling,b::Chain{V}) where V = V(a)*b
*(a::Chain{V},b::UniformScaling) where V = a*V(b)

+(a::Zero{V},::Zero{V}) where V = a
-(a::Zero{V},::Zero{V}) where V = a
⟑(a::Zero{V},::Zero{V}) where V = a
∧(a::Zero{V},::Zero{V}) where V = a
∨(a::Zero{V},::Zero{V}) where V = a
contraction(a::Zero{V},::Zero{V}) where V = a
contraction_metric(a::Zero{V},::Zero{V},g) where V = a
wedgedot_metric(a::Zero{V},::Zero{V},g) where V = a

⟑(a::Infinity{V},::Infinity{V}) where V = a
∧(a::Infinity{V},::Infinity{V}) where V = a
∨(a::Infinity{V},::Infinity{V}) where V = a
contraction(a::Infinity{V},::Infinity{V}) where V = a
contraction_metric(a::Infinity{V},::Infinity{V},g) where V = a
wedgedot_metric(a::Infinity{V},::Infinity{V}) where V = a

+(a::T,b::Zero) where T<:TensorAlgebra{V} where V = a
+(a::Zero,b::T) where T<:TensorAlgebra{V} where V = b
-(a::T,b::Zero) where T<:TensorAlgebra{V} where V = a
-(a::Zero,b::T) where T<:TensorAlgebra{V} where V = -b
#/(a::T,b::Zero) where T<:TensorAlgebra = inv(b)
/(a::Zero,b::T) where T<:Number = iszero(b) ? Single{V}(0/0) : a
/(a::Zero,b::T) where T<:TensorAlgebra{V} where V = iszero(b) ? Single{V}(0/0) : Zero(V)
/(a::Zero,b::T) where T<:Couple{V} where V = iszero(b) ? Single{V}(0/0) : Zero(V)
#/(a::Zero{V},b::Zero) where V = Single{V}(0/0)
/(a::One,b::Zero) = inv(b)
inv(a::Zero{V}) where V = Infinity(V)

+(a::T,b::Infinity) where T<:TensorAlgebra = b
+(a::Infinity,b::T) where T<:TensorAlgebra = a
-(a::T,b::Infinity) where T<:TensorAlgebra = b
-(a::Infinity,b::T) where T<:TensorAlgebra = a
#/(a::T,b::Infinity) where T<:TensorAlgebra = inv(b)
/(a::Infinity,b::T) where T<:Number = isinf(b) ? Single{V}(Inf/Inf) : a
/(a::Infinity,b::T) where T<:TensorAlgebra{V} where V = isinf(norm(b)) ? Single{V}(Inf/Inf) : Infinity(V)
/(a::Infinity,b::T) where T<:Couple{V} where V = isinf(value(b)) ? Single{V}(Inf/Inf) : Infinity(V)
#/(a::Infinity{V},b::Infinity) where V = Single{V}(Inf/Inf)
/(a::One,b::Infinity) = inv(b)
inv(a::Infinity{V}) where V = Zero(V)

for type ∈ (:Zero,:Infinity)
    for tensor ∈ (:TensorTerm,:Couple,:PseudoCouple)
        for (op,args) ∈ ((:⟑,()),(:wedgedot_metric,(:g,)))
            @eval begin
                $op(a::$tensor,b::$type,$(args...)) = b
                $op(a::$type,b::$tensor,$(args...)) = a
            end
        end
    end
end

for T ∈ (:TensorTerm,:Couple,:PseudoCouple)
    for type ∈ (:Zero,:Infinity)
        @eval begin
            ∧(a::T,b::$type) where T<:$T = b
            ∧(a::$type,b::T) where T<:$T = a
            ∨(a::T,b::$type) where T<:$T = b
            ∨(a::$type,b::T) where T<:$T = a
            contraction(a::$T,b::$type) = b
            contraction(a::$type,b::$T) = a
            contraction_metric(a::$T,b::$type,g) = b
            contraction_metric(a::$type,b::$T,g) = a
        end
    end
end

+(a::T,b::Zero{V}) where {T<:Number,V} = Single{V}(a)
+(a::Zero{V},b::T) where {T<:Number,V} = Single{V}(b)
-(a::T,b::Zero{V}) where {T<:Number,V} = Single{V}(a)
-(a::Zero{V},b::T) where {T<:Number,V} = Single{V}(-b)
*(a::T,b::Zero) where T<:Real = b
*(a::Zero,b::T) where T<:Real = a
*(a::T,b::Zero) where T<:Complex = b
*(a::Zero,b::T) where T<:Complex = a

+(a::T,b::Infinity) where T<:Number = b
+(a::Infinity,b::T) where T<:Number = a
-(a::T,b::Infinity) where T<:Number = b
-(a::Infinity,b::T) where T<:Number = a
*(a::T,b::Infinity) where T<:Real = b
*(a::Infinity,b::T) where T<:Real = a
*(a::T,b::Infinity) where T<:Complex = b
*(a::Infinity,b::T) where T<:Complex = a

@inline Base.:^(a::T,b::Zero,_=nothing) where T<:TensorAlgebra{V} where V = One(V)
@inline Base.:^(a::Zero,b::Single{V,0},_=nothing) where V = iszero(b) ? One(V) : isless(b,0) ? Infinity(V) : Zero(V)
@inline Base.:^(a::T,b::Zero{V},_=nothing) where {T<:Number,V} = One(V)
@inline Base.:^(a::Zero{V},b::T,_=nothing) where {T<:Number,V} = iszero(b) ? One(V) : isless(b,zero(b)) ? Infinity(V) : a
@inline Base.:^(a::Zero{V},::Zero,_=nothing) where V = One(V)
@inline Base.:^(a::Zero,::Infinity,_=nothing) = a
@inline Base.:^(a::Zero{V},b::T,_=nothing) where {V,T<:Integer} = iszero(b) ? One(V) : isless(b,zero(b)) ? Infinity(V) : a

@inline Base.:^(a::Single{V,0},b::Infinity,_=nothing) where V = (c=abs(value(a)); isone(c) ? One(V) : isless(c,1) ? Zero(V) : b)
@inline Base.:^(a::Infinity,b::T,_=nothing) where T<:TensorTerm{V,0} where V = iszero(b) ? One(V) : isless(b,0) ? Zero(V) : Infinity(V)
@inline Base.:^(a::T,b::Infinity{V},_=nothing) where {T<:Number,V} = (c=abs(a); isone(c) ? One(V) : isless(c,1) : Zero(V) : b)
@inline Base.:^(a::Infinity{V},b::T,_=nothing) where {T<:Number,V} = iszero(b) ? One(V) : isless(b,zero(b)) ? Zero(V) : a
@inline Base.:^(a::Infinity{V},::Zero,_=nothing) where V = One(V)
@inline Base.:^(a::Infinity,::Infinity,_=nothing) = a
@inline Base.:^(a::Infinity{V},b::T,_=nothing) where {V,T<:Integer} = iszero(b) ? One(V) : isless(b,zero(b)) ? Zero(V) : a

+(a::T,b::Phasor) where T<:TensorAlgebra{V} where V = a+Couple(b)
+(a::Phasor,b::T) where T<:TensorAlgebra{V} where V = Couple(a)+b
-(a::T,b::Phasor) where T<:TensorAlgebra{V} where V = a-Couple(b)
-(a::Phasor,b::T) where T<:TensorAlgebra{V} where V = Couple(a)-b
⟑(a::T,b::Phasor) where T<:TensorAlgebra{V} where V = a⟑Couple(b)
⟑(a::Phasor,b::T) where T<:TensorAlgebra{V} where V = Couple(a)⟑b
/(a::T,b::Phasor) where T<:TensorAlgebra{V} where V = a/Couple(b)
/(a::Phasor,b::T) where T<:TensorAlgebra{V} where V = Couple(a)/b
∧(a::T,b::Phasor) where T<:TensorAlgebra{V} where V = a∧Couple(b)
∧(a::Phasor,b::T) where T<:TensorAlgebra{V} where V = Couple(a)∧b
∨(a::T,b::Phasor) where T<:TensorAlgebra{V} where V = a∨Couple(b)
∨(a::Phasor,b::T) where T<:TensorAlgebra{V} where V = Couple(a)∨b
contraction(a::T,b::Phasor) where T<:TensorAlgebra{V} where V = contraction(a,Couple(b))
contraction(a::Phasor,b::T) where T<:TensorAlgebra{V} where V = contraction(Couple(a),b)
contraction_metric(a::T,b::Phasor,g) where T<:TensorAlgebra{V} where V = contraction_metric(a,Couple(b),g)
contraction_metric(a::Phasor,b::T,g) where T<:TensorAlgebra{V} where V = contraction_metric(Couple(a),b,g)
wedgedot_metric(a::T,b::Phasor,g) where T<:TensorAlgebra{V} where V = wedgedot_metric(a,Couple(b),g)
wedgedot_metric(a::Phasor,b::T,g) where T<:TensorAlgebra{V} where V = wedgedot_metric(Couple(a),b,g)

plus(b::Chain{V,G},a::Submanifold{V,G}) where {V,G} = plus(a,b)
plus(b::Chain{V,G},a::Submanifold{V,L}) where {V,G,L} = plus(a,b)
plus(b::Chain{V,G},a::Single{V,G}) where {V,G} = plus(a,b)
plus(b::Chain{V,G},a::Single{V,L}) where {V,G,L} = plus(a,b)
plus(b::Multivector{V},a::Submanifold{V,G}) where {V,G} = plus(a,b)
plus(b::Multivector{V},a::Single{V,G}) where {V,G} = plus(a,b)
plus(b::Spinor{V},a::Submanifold{V,G}) where {V,G} = plus(a,b)
plus(b::Spinor{V},a::Single{V,G}) where {V,G} = plus(a,b)
plus(b::CoSpinor{V},a::Submanifold{V,G}) where {V,G} = plus(a,b)
plus(b::CoSpinor{V},a::Single{V,G}) where {V,G} = plus(a,b)
-(t::Submanifold) = Single(-value(t),t)
-(a::Chain{V,G}) where {V,G} = Chain{V,G}(-value(a))
-(a::Multivector{V}) where V = Multivector{V}(-value(a))
-(a::Spinor{V}) where V = Spinor{V}(-value(a))
-(a::CoSpinor{V}) where V = CoSpinor{V}(-value(a))
-(a::Couple{V,B}) where {V,B} = Couple{V,B}(-a.v)
-(a::PseudoCouple{V,B}) where {V,B} = PseudoCouple{V,B}(-a.v)
⟑(a::Single{V,0},b::Chain{V,G}) where {V,G} = Chain{V,G}(a.v*b.v)
⟑(a::Chain{V,G},b::Single{V,0}) where {V,G} = Chain{V,G}(a.v*b.v)
⟑(a::Submanifold{V,0},b::Chain{W,G}) where {V,W,G} = b
⟑(a::Chain{V,G},b::Submanifold{W,0}) where {V,W,G} = a
wedgedot_metric(a::Single{V,0},b::Chain{V,G},g) where {V,G} = Chain{V,G}(a.v*b.v)
wedgedot_metric(a::Chain{V,G},b::Single{V,0},g) where {V,G} = Chain{V,G}(a.v*b.v)
wedgedot_metric(a::Submanifold{V,0},b::Chain{W,G},g) where {V,W,G} = b
wedgedot_metric(a::Chain{V,G},b::Submanifold{W,0},g) where {V,W,G} = a

for (couple,calar) ∈ ((:Couple,:scalar),(:PseudoCouple,:volume))
    @eval begin
        plus(a::Multivector{V},b::$couple{V}) where V = (a+$calar(b))+imaginary(b)
        plus(a::$couple{V},b::Multivector{V}) where V = (b+$calar(a))+imaginary(a)
        minus(a::Multivector{V},b::$couple{V}) where V = (a-$calar(b))-imaginary(b)
        minus(a::$couple{V},b::Multivector{V}) where V = ($calar(a)-b)+imaginary(a)
        plus(a::Spinor{V},b::$couple{V}) where V = (a+$calar(b))+imaginary(b)
        plus(a::$couple{V},b::Spinor{V}) where V = (b+$calar(a))+imaginary(a)
        minus(a::Spinor{V},b::$couple{V}) where V = (a-$calar(b))-imaginary(b)
        minus(a::$couple{V},b::Spinor{V}) where V = ($calar(a)-b)+imaginary(a)
        plus(a::CoSpinor{V},b::$couple{V}) where V = (a+$calar(b))+imaginary(b)
        plus(a::$couple{V},b::CoSpinor{V}) where V = (b+$calar(a))+imaginary(a)
        minus(a::CoSpinor{V},b::$couple{V}) where V = (a-$calar(b))-imaginary(b)
        minus(a::$couple{V},b::CoSpinor{V}) where V = ($calar(a)-b)+imaginary(a)
        plus(a::Chain{V,0},b::$couple{V}) where V = (a+$calar(b))+imaginary(b)
        plus(a::$couple{V},b::Chain{V,0}) where V = (b+$calar(a))+imaginary(a)
        minus(a::Chain{V,0},b::$couple{V}) where V = (a-$calar(b))-imaginary(b)
        minus(a::$couple{V},b::Chain{V,0}) where V = ($calar(a)-b)+imaginary(a)
        plus(a::Chain{V},b::$couple{V}) where V = (a+imaginary(b))+$calar(b)
        plus(a::$couple{V},b::Chain{V}) where V = (b+imaginary(a))+$calar(a)
        minus(a::Chain{V},b::$couple{V}) where V = (a-imaginary(b))-$calar(b)
        minus(a::$couple{V},b::Chain{V}) where V = (imaginary(a)-b)+$calar(a)
    end
end

for (op,po) ∈ ((:plus,:+),(:minus,:-))
    @eval begin
        $op(a::Couple{V,B},b::Couple{V,B}) where {V,B} = Couple{V,B}($po(realvalue(a),realvalue(b)),$po(imagvalue(a),imagvalue(b)))
        $op(a::PseudoCouple{V,B},b::PseudoCouple{V,B}) where {V,B} = PseudoCouple{V,B}($po(realvalue(a),imagvalue(b)),$po(imagvalue(a),imagvalue(b)))
        $op(a::Phasor{V,B},b::Phasor{V,B}) where {V,B} = Phasor($op(Couple(a),Couple(b)))
        $op(a::Couple{V},b::Phasor{V}) where V = $op(a,Couple(b))
        $op(a::Phasor{V},b::Couple{V}) where V = $op(Couple(a),b)
        $op(a::Couple{V},b::Couple{V}) where V = $op($op(a,scalar(b)),imaginary(b))
        $op(a::PseudoCouple{V},b::PseudoCouple{V}) where V = $op($op(a,volume(b)),imaginary(b))
        $op(a::Couple{V},b::PseudoCouple{V}) where V = $op($op(a,imaginary(b)),volume(b))
        $op(a::PseudoCouple{V},b::Couple{V}) where V = $op(imaginary(a),b)+volume(a)
        $op(a::Phasor{V},b::Phasor{V}) where V = $op(Couple(a),Couple(b))
        $op(a::Spinor{V},b::CoSpinor{V}) where V = $op(Multivector(a),Multivector(b))
        $op(a::CoSpinor{V},b::Spinor{V}) where V = $op(Multivector(a),Multivector(b))
    end
end

for (op,args) ∈ ((:⟑,()),(:wedgedot_metric,(:g,)))
    @eval begin
        function $op(a::Couple{V,B},b::Couple{V,B},$(args...)) where {V,B}
            Couple{V,B}(realvalue(a)*realvalue(b)+(imagvalue(a)*imagvalue(b))*value($op(B,B,$(args...))),realvalue(a)*imagvalue(b)+imagvalue(a)*realvalue(b))
        end
        function $op(a::PseudoCouple{V,B},b::PseudoCouple{V,B},$(args...)) where {V,B}
            out = imaginary(a)*volume(b)+volume(a)*imaginary(b)
            Couple{V,basis(out)}((realvalue(a)*realvalue(b))*value($op(B,B,$(args...)))+value(volume(a)*volume(b)),value(out))
        end
        function $op(a::Phasor{V,B},b::Phasor{V,B},$(args...)) where {V,B}
            Phasor{V,B}(realvalue(a)*realvalue(b),imagvalue(a)+imagvalue(b))
        end
        $op(a::Couple{V},b::Couple{V},$(args...)) where V = (a⟑scalar(b))+$op(a,imaginary(b),$(args...))
        $op(a::PseudoCouple{V},b::PseudoCouple{V},$(args...)) where V = ($op(volume(a),volume(b),$(args...))+$op(imaginary(a),imaginary(b),$(args...)))+$op(imaginary(a),volume(b),$(args...))+$op(volume(b),imaginary(a),$(args...))
        $op(a::Couple{V},b::PseudoCouple{V},$(args...)) where V = (scalar(a)⟑b)+$op(imaginary(a),b,$(args...))
        $op(a::PseudoCouple{V},b::Couple{V},$(args...)) where V = (a⟑scalar(b))+$op(a,imaginary(b),$(args...))
    end
end

function ∧(a::Couple{V,B},b::Couple{V,B}) where {V,B}
    Couple{V,B}(realvalue(a)*realvalue(b),realvalue(a)*imagvalue(b)+imagvalue(a)*realvalue(b))
end
function ∧(a::PseudoCouple{V,B},b::PseudoCouple{V,B}) where {V,B}
    grade(B)==0 ? PseudoCouple{V,B}(realvalue(a)*realvalue(b),realvalue(a)*imagvalue(b)+imagvalue(a)*realvalue(b)) : Zero(V)
end
∧(a::Phasor{V,B},b::Phasor{V,B}) where {V,B} = Phasor(Couple(a)∧Couple(b))

function ∨(a::Couple{V,B},b::Couple{V,B}) where {V,B}
    grade(B)==grade(V) ? Couple{V,B}(realvalue(a)*imagvalue(b)+imagvalue(a)*realvalue(b),imagvalue(a)*imagvalue(b)) : Zero(V)
end
function ∨(a::PseudoCouple{V,B},b::PseudoCouple{V,B}) where {V,B}
    PseudoCouple{V,B}(realvalue(a)*imagvalue(b)+imagvalue(a)*realvalue(b),imagvalue(a)*imagvalue(b))
end
∨(a::Phasor{V,B},b::Phasor{V,B}) where {V,B} = Phasor(Couple(a)∨Couple(b))

for (op,args) ∈ ((:contraction,()),(:contraction_metric,(:g,)))
    @eval begin
        function $op(a::Couple{V,B},b::Couple{V,B},$(args...)) where {V,B}
            Couple{V,B}(realvalue(a)*realvalue(b)+(imagvalue(a)*imagvalue(b))*value(abs2_inv(B,$(args...))),imagvalue(a)*realvalue(b))
        end
        function $op(a::PseudoCouple{V,B},b::PseudoCouple{V,B},$(args...)) where {V,B}
            out = $op(volume(a),imaginary(b),$(args...))
            Couple{V,basis(out)}((realvalue(a)*realvalue(b))*value(abs2_inv(B,$(args...)))+(imagvalue(a)*imagvalue(b))*value(abs2_inv(V,$(args...))),value(out))
        end
        $op(a::Phasor{V,B},b::Phasor{V,B},$(args...)) where {V,B} = Phasor($op(Couple(a),Couple(b),$(args...)))
        $op(a::Couple{V},b::Couple{V},$(args...)) where V = $op(a,imaginary(b),$(args...))+contraction(a,scalar(b))
        $op(a::PseudoCouple{V},b::PseudoCouple{V},$(args...)) where V = $op(imaginary(a),b,$(args...))+$op(volume(a),b,$(args...))
        $op(a::Couple{V},b::PseudoCouple{V},$(args...)) where V = contractn(scalar(a),imaginary(b))+$op(imaginary(a),b,$(args...))
        $op(a::PseudoCouple{V},b::Couple{V},$(args...)) where V = contraction(a,scalar(b))+$op(a,imaginary(b),$(args...))
    end
end

∧(a::Couple{V},b::Couple{V}) where V = (a∧scalar(b))+(a∧imaginary(b))
∧(a::PseudoCouple{V},b::PseudoCouple{V}) where V = imaginary(a)∧imaginary(b)
∧(a::Couple{V},b::PseudoCouple{V}) where V = (scalar(a)∧b)+(imaginary(a)∧imaginary(b))
∧(a::PseudoCouple{V},b::Couple{V}) where V = (a∧scalar(b))+(imaginary(a)∧imaginary(b))
∨(a::Couple{V},b::Couple{V}) where V = imaginary(a)∨imaginary(b)
∨(a::PseudoCouple{V},b::PseudoCouple{V}) where V = (a∨imaginary(b))+(a∨volume(b))
∨(a::Couple{V},b::PseudoCouple{V}) where V = (scalar(a)∨volume(b))+(imaginary(a)∨b)
∨(a::PseudoCouple{V},b::Couple{V}) where V = (volume(a)∨scalar(b))+(a∨imaginary(b))

plus(a::TensorTerm{V,0},b::Couple{V,B}) where {V,B} = Couple{V,B}(AbstractTensors.:∑(value(a),realvalue(b)),imagvalue(b))
plus(a::Couple{V,B},b::TensorTerm{V,0}) where {V,B} = Couple{V,B}(AbstractTensors.:∑(realvalue(a),value(b)),imagvalue(a))
function plus(a::TensorTerm{V},b::Couple{V,B}) where {V,B}
    if basis(a) == B
        Couple{V,B}(realvalue(b),AbstractTensors.:∑(value(a),imagvalue(b)))
    else
        a+multispin(b)
    end
end
function plus(a::Couple{V,B},b::TensorTerm{V}) where {V,B}
    if B == basis(b)
        Couple{V,B}(realvalue(a),AbstractTensors.:∑(imagvalue(a),value(b)))
    else
        multispin(a)+b
    end
end
function plus(a::TensorTerm{V},b::PseudoCouple{V,B}) where {V,B}
    if basis(a) == B
        PseudoCouple{V,B}(AbstractTensors.:∑(value(a),realvalue(b)),imagvalue(b))
    elseif basis(a) == Subamnifold(V)
        PseudoCouple{V,B}(realvalue(b),AbstractTensors.:∑(value(a),imagvalue(b)))
    else
        a+multispin(b)
    end
end
function plus(a::PseudoCouple{V,B},b::TensorTerm{V}) where {V,B}
    if B == basis(b)
        PseudoCouple{V,B}(AbstractTensors.:∑(realvalue(a),value(b)),imagvalue(a))
    elseif Submanifold(V) == basis(b)
        PseudoCouple{V,B}(realvalue(a),AbstractTensors.:∑(imagvalue(a),value(b)))
    else
        multispin(a)+b
    end
end

minus(a::TensorTerm{V,0},b::Couple{V,B}) where {V,B} = (re = AbstractTensors.:-(value(a),realvalue(b)); Couple{V,B}(re,AbstractTensors.:-(imagvalue(b))))
minus(a::Couple{V,B},b::TensorTerm{V,0}) where {V,B} = Couple{V,B}(AbstractTensors.:-(realvalue(a),value(b)),imagvalue(a))
function minus(a::TensorTerm{V},b::Couple{V,B}) where {V,B}
    if basis(a) == B
        re = AbstractTensors.:-(value(a),imagvalue(b))
        Couple{V,B}(AbstractTensors.:-(realvalue(b)),re)
    else
        a-multispin(b)
    end
end
function minus(a::Couple{V,B},b::TensorTerm{V}) where {V,B}
    if B == basis(b)
        Couple{V,B}(realvalue(a),AbstractTensors.:-(imagvalue(a),value(b)))
    else
        multispin(a)-b
    end
end
function minus(a::TensorTerm{V},b::PseudoCouple{V,B}) where {V,B}
    if basis(a) == B
        re = AbstractTensors.:-(value(a),realvalue(b))
        PseudoCouple{V,B}(re,AbstractTensors.:-(imagvalue(b)))
    elseif basis(a) == Submanifold(V)
        re = AbstractTensors.:-(value(a),imagvalue(b))
        PseudoCouple{V,B}(AbstractTensors.:-(realvalue(b)),re)
    else
        a-multispin(b)
    end
end
function minus(a::PseudoCouple{V,B},b::TensorTerm{V}) where {V,B}
    if B == basis(b)
        PseudoCouple{V,B}(AbstractTensors.:-(realvalue(a),value(b)),imagvalue(a))
    elseif Submanifold(V) == basis(b)
        PseudoCouple{V,B}(realvalue(a),AbstractTensors.:-(imagvalue(a),value(b)))
    else
        multispin(a)-b
    end
end

for (op,args) ∈ ((:⟑,()),(:wedgedot_metric,(:g,)))
for (couple,calar) ∈ ((:Couple,:scalar),(:PseudoCouple,:volume))
    @eval begin
        $op(a::Multivector{V},b::$couple{V},$(args...)) where V = $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::Multivector{V},$(args...)) where V = $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
        $op(a::Spinor{V},b::$couple{V},$(args...)) where V = $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::Spinor{V},$(args...)) where V = $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
        $op(a::CoSpinor{V},b::$couple{V},$(args...)) where V = $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::CoSpinor{V},$(args...)) where V = $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
        $op(a::Chain{V,G},b::$couple{V},$(args...)) where {V,G} = (G==0 || G==mdims(V)) ? $op(Single(a),b,$(args...)) : $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::Chain{V,G},$(args...)) where {V,G} = (G==0 || G==mdims(V)) ? $op(a,Single(b),$(args...)) : $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
        $op(a::TensorTerm{V},b::$couple{V},$(args...)) where V = $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::TensorTerm{V},$(args...)) where V = $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
    end
end
@eval begin
    $op(a::TensorTerm{V,0},b::Couple{V,B},$(args...)) where {V,B} = Couple{V,B}(AbstractTensors.:∏(value(a),realvalue(b)),AbstractTensors.:∏(value(a),imagvalue(b)))
    $op(a::Couple{V,B},b::TensorTerm{V,0},$(args...)) where {V,B} = Couple{V,B}(AbstractTensors.:∏(realvalue(a),value(b)),AbstractTensors.:∏(imagvalue(a),value(b)))
    $op(a::TensorTerm{V,0},b::PseudoCouple{V,B},$(args...)) where {V,B} = PseudoCouple{V,B}(AbstractTensors.:∏(value(a),realvalue(b)),AbstractTensors.:∏(value(a),imagvalue(b)))
    $op(a::PseudoCouple{V,B},b::TensorTerm{V,0},$(args...)) where {V,B} = PseudoCouple{V,B}(AbstractTensors.:∏(realvalue(a),value(b)),AbstractTensors.:∏(imagvalue(a),value(b)))
end
end

for (couple,calar) ∈ ((:Couple,:scalar),(:PseudoCouple,:volume))
    @eval begin
        ∧(a::Multivector{V},b::$couple{V}) where V = (a∧$calar(b)) + (a∧imaginary(b))
        ∧(a::$couple{V},b::Multivector{V}) where V = ($calar(a)∧b) + (imaginary(a)∧b)
        ∧(a::Spinor{V},b::$couple{V}) where V = (a∧$calar(b)) + (a∧imaginary(b))
        ∧(a::$couple{V},b::Spinor{V}) where V = ($calar(a)∧b) + (imaginary(a)∧b)
        ∧(a::CoSpinor{V},b::$couple{V}) where V = (a∧$calar(b)) + (a∧imaginary(b))
        ∧(a::$couple{V},b::CoSpinor{V}) where V = ($calar(a)∧b) + (imaginary(a)∧b)
        ∧(a::Chain{V},b::$couple{V}) where V = (a∧$calar(b)) + (a∧imaginary(b))
        ∧(a::$couple{V},b::Chain{V}) where V = ($calar(a)∧b) + (imaginary(a)∧b)
        ∧(a::TensorTerm{V,0},b::$couple{V}) where V = a⟑b
        ∧(a::$couple{V},b::TensorTerm{V,0}) where V = a⟑b
    end
end
function ∧(a::TensorTerm{V,G},b::Couple{V,B}) where {V,G,B}
    basis(a) == B ? a∧scalar(b) : (a∧scalar(b)) + (a∧imaginary(b))
end
function ∧(a::Couple{V,B},b::TensorTerm{V,G}) where {V,G,B}
    B == basis(b) ? scalar(a)∧b : (scalar(a)∧b) + (imaginary(a)∧b)
end
function ∧(a::TensorTerm{V,G},b::PseudoCouple{V,B}) where {V,G,B}
    basis(a) == B ? Zero(V) : a∧imaginary(b)
end
function ∧(a::PseudoCouple{V,B},b::TensorTerm{V,G}) where {V,G,B}
    B == basis(b) ? Zero(V) : imaginary(a)∧b
end

for (couple,calar) ∈ ((:Couple,:scalar),(:PseudoCouple,:volume))
    @eval begin
        ∨(a::Multivector{V},b::$couple{V}) where V = (a∨$calar(b)) + (a∨imaginary(b))
        ∨(a::$couple{V},b::Multivector{V}) where V = ($calar(a)∨b) + (imaginary(a)∨b)
        ∨(a::Spinor{V},b::$couple{V}) where V = (a∨$calar(b)) + (a∨imaginary(b))
        ∨(a::$couple{V},b::Spinor{V}) where V = ($calar(a)∨b) + (imaginary(a)∨b)
        ∨(a::CoSpinor{V},b::$couple{V}) where V = (a∨$calar(b)) + (a∨imaginary(b))
        ∨(a::$couple{V},b::CoSpinor{V}) where V = ($calar(a)∨b) + (imaginary(a)∨b)
        ∨(a::Chain{V},b::$couple{V}) where V = (a∨$calar(b)) + (a∨imaginary(b))
        ∨(a::$couple{V},b::Chain{V}) where V = ($calar(a)∨b) + (imaginary(a)∨b)
    end
end
∨(a::TensorTerm{V,0},b::Couple{V}) where V = grade(B)==grade(V) ? a∨imaginary(b) : Zero(V)
∨(a::Couple{V},b::TensorTerm{V,0}) where V = grade(B)==grade(V) ? imaginary(a)∨b : Zero(V)
∨(a::TensorTerm{V,0},b::PseudoCouple{V}) where V = a∨volume(b)
∨(a::PseudoCouple{V},b::TensorTerm{V,0}) where V = volume(a)∨b
function ∨(a::TensorTerm{V,G},b::Couple{V,B}) where {V,G,B}
    if basis(a)==B && grade(B)==grade(V)
        (a∨scalar(b)) + (a∨imaginary(b))
    else
        a∨imaginary(b)
    end
end
function ∨(a::Couple{V,B},b::TensorTerm{V,G}) where {V,G,B}
    if B == basis(b) && grade(B)==grade(V)
        (scalar(a)∨b) + (imaginary(a)∨b)
    else
        imaginary(a)∨b
    end
end
∨(a::TensorTerm{V},b::PseudoCouple{V}) where V = (a∨imaginary(b)) + (a∨volume(b))
∨(a::PseudoCouple{V},b::TensorTerm{V}) where V = (imaginary(a)∨b) + (volume(a)∨b)

for (op,args) ∈ ((:contraction,()),(:contraction_metric,(:g,)))
for (couple,calar) ∈ ((:Couple,:scalar),(:PseudoCouple,:volume))
    @eval begin
        $op(a::Multivector{V},b::$couple{V},$(args...)) where V = $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::Multivector{V},$(args...)) where V = $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
        $op(a::Spinor{V},b::$couple{V},$(args...)) where V = $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::Spinor{V},$(args...)) where V = $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
        $op(a::CoSpinor{V},b::$couple{V},$(args...)) where V = $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::CoSpinor{V},$(args...)) where V = $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
        $op(a::Chain{V},b::$couple{V},$(args...)) where V = $op(a,$calar(b),$(args...)) + $op(a,imaginary(b),$(args...))
        $op(a::$couple{V},b::Chain{V},$(args...)) where V = $op($calar(a),b,$(args...)) + $op(imaginary(a),b,$(args...))
        $op(a::$couple{V},b::TensorTerm{V,0},$(args...)) where V = a⟑b
    end
end
@eval begin
    $op(a::TensorTerm{V,0},b::Couple{V},$(args...)) where V = Single{V}(value(a)*realvalue(b))
    $op(a::TensorTerm{V,0},b::PseudoCouple{V},$(args...)) where V = contraction(a,imaginary(b))
    function $op(a::TensorTerm{V,G},b::Couple{V,B},$(args...)) where {V,G,B}
        contraction(a,scalar(b)) + $op(a,imaginary(b),$(args...))
    end
    function $op(a::Couple{V,B},b::TensorTerm{V,G},$(args...)) where {V,G,B}
        if basis(a) == One(V)
            contraction(scalar(a),b) + $op(imaginary(a),b,$(args...))
        else
            $op(imaginary(a),b,$(args...))
        end
    end
    function $op(a::TensorTerm{V,G},b::PseudoCouple{V,B},$(args...)) where {V,G,B}
        if basis(a) == Submanifold(V)
            $op(a,imaginary(b),$(args...)) + $op(a,volume(b),$(args...))
        else
            $op(a,imaginary(b),$(args...))
        end
    end
    function $op(a::PseudoCouple{V,B},b::TensorTerm{V,G},$(args...)) where {V,G,B}
        $op(imaginary(a),b,$(args...)) + $op(volume(a),b,$(args...))
    end
end
end

# more algebra

for F ∈ Fields
    @eval begin
        *(a::F,b::Couple{V,B}) where {F<:$F,V,B} = Couple{V,B}(a*realvalue(b),a*imagvalue(b))
        *(a::Couple{V,B},b::F) where {F<:$F,V,B} = Couple{V,B}(realvalue(a)*b,imagvalue(a)*b)
        *(a::F,b::PseudoCouple{V,B}) where {F<:$F,V,B} = PseudoCouple{V,B}(a*realvalue(b),a*imagvalue(b))
        *(a::PseudoCouple{V,B},b::F) where {F<:$F,V,B} = PseudoCouple{V,B}(realvalue(a)*b,imagvalue(a)*b)
        *(a::F,b::Phasor{V,B}) where {F<:$F,V,B} = Phasor{V,B}(a*realvalue(b),imagvalue(b))
        *(a::Phasor{V,B},b::F) where {F<:$F,V,B} = Phasor{V,B}(realvalue(a)*b,imagvalue(a))
        *(a::F,b::Multivector{V}) where {F<:$F,V} = Multivector{V}(a*b.v)
        *(a::Multivector{V},b::F) where {F<:$F,V} = Multivector{V}(a.v*b)
        *(a::F,b::Spinor{V}) where {F<:$F,V} = Spinor{V}(a*b.v)
        *(a::Spinor{V},b::F) where {F<:$F,V} = Spinor{V}(a.v*b)
        *(a::F,b::CoSpinor{V}) where {F<:$F,V} = CoSpinor{V}(a*b.v)
        *(a::CoSpinor{V},b::F) where {F<:$F,V} = CoSpinor{V}(a.v*b)
        *(a::F,b::Chain{V,G}) where {F<:$F,V,G} = Chain{V,G}(a*b.v)
        *(a::Chain{V,G},b::F) where {F<:$F,V,G} = Chain{V,G}(a.v*b)
        *(a::F,b::Single{V,G,B,T} where B) where {F<:$F,V,G,T} = Single{V,G}($Sym.:∏(a,b.v),basis(b))
        *(a::Single{V,G,B,T} where B,b::F) where {F<:$F,V,G,T} = Single{V,G}($Sym.:∏(a.v,b),basis(a))
        *(a::F,b::Single{V,G,B,T} where B) where {F<:$F,V,G,T<:Number} = Single{V,G}(*(a,b.v),basis(b))
        *(a::Single{V,G,B,T} where B,b::F) where {F<:$F,V,G,T<:Number} = Single{V,G}(*(a.v,b),basis(a))
    end
end
for op ∈ (:+,:-)
    for Term ∈ (:TensorGraded,:TensorMixed,:Zero)
        @eval begin
            $op(a::T,b::NSE) where T<:$Term = iszero(b) ? a : $op(a,b*One(Manifold(a)))
            $op(a::NSE,b::T) where T<:$Term = iszero(a) ? $op(b) : $op(a*One(Manifold(b)),b)
        end
    end
end
for (op,po) ∈ ((:+,:plus),(:-,:minus))
    @eval begin
        @generated function $po(a::TensorTerm{V,L},b::TensorTerm{V,G}) where {V,L,G}
            adder(a,b,$(QuoteNode(op)))
        end
        @generated function $po(a::TensorTerm{V,G},b::Chain{V,G,T}) where {V,G,T}
            adder(a,b,$(QuoteNode(op)))
        end
        @generated function $po(a::TensorTerm{V,L},b::Chain{V,G,T}) where {V,G,T,L}
            adder(a,b,$(QuoteNode(op)))
        end
        @generated function $po(a::TensorTerm{V,G},b::Multivector{V,T}) where {V,G,T}
            adder(a,b,$(QuoteNode(op)))
        end
        @generated function $po(a::TensorTerm{V,G},b::Spinor{V,T}) where {V,G,T}
            adder(a,b,$(QuoteNode(op)))
        end
        @generated function $po(a::TensorTerm{V,G},b::CoSpinor{V,T}) where {V,G,T}
            adder(a,b,$(QuoteNode(op)))
        end
    end
end
@generated minus(b::Chain{V,G,T},a::TensorTerm{V,G}) where {V,G,T} = adder(a,b,:-,true)
@generated minus(b::Chain{V,G,T},a::TensorTerm{V,L}) where {V,G,T,L} = adder(a,b,:-,true)
@generated minus(b::Spinor{V,T},a::TensorTerm{V,G}) where {V,G,T} = adder(a,b,:-,true)
@generated minus(b::CoSpinor{V,T},a::TensorTerm{V,G}) where {V,G,T} = adder(a,b,:-,true)
@generated minus(b::Multivector{V,T},a::TensorTerm{V,G}) where {V,G,T} = adder(a,b,:-,true)

@eval begin
    @generated function Base.adjoint(m::Chain{V,G,T}) where {V,G,T}
        CONJ,VEC = conjvec(m)
        TF = T ∉ FieldsBig ? :Any : :T
        if binomial(mdims(V),G)<(1<<cache_limit)
            if isdyadic(V)
                $(insert_expr((:N,:M,:ib,:t),:mvec)...)
                out = svec(N,G,Any)(zeros(svec(N,G,t)))
                for i ∈ list(1,binomial(N,G))
                    @inbounds setblade!_pre(out,:($CONJ(@inbounds m.v[$i])),dual(V,ib[i],M),Val{N}())
                end
                return :(Chain{$(dual(V)),G}($(Expr(:call,tvec(N,TF),out...))))
            else
                return :(Chain{$(dual(V)),G}($CONJ.(value(m))))
            end
        else return quote
            if isdyadic(V)
                $(insert_expr((:N,:M,:ib),:svec)...)
                out = zeros($VEC(N,G,$TF))
                for i ∈ 1:binomial(N,G)
                    @inbounds setblade!(out,$CONJ(m.v[i]),dual(V,ib[i],M),Val{N}())
                end
            else
                out = $CONJ.(value(m))
            end
            Chain{dual(V),G}(out)
        end end
    end
    @generated function Base.adjoint(m::Multivector{V,T}) where {V,T}
        CONJ,VEC = conjvec(m)
        TF = T ∉ FieldsBig ? :Any : :T
        if mdims(V)<cache_limit
            if isdyadic(V)
                $(insert_expr((:N,:M,:bs,:bn,:t),:mvec)...)
                out = svec(N,Any)(zeros(svec(N,t)))
                for g ∈ list(1,N+1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds setmulti!_pre(out,:($CONJ(@inbounds m.v[$(bs[g]+i)])),dual(V,ib[i],M))
                    end
                end
                return :(Multivector{$(dual(V))}($(Expr(:call,tvec(N,TF),out...))))
            else
                return :(Multivector{$(dual(V))}($CONJ.(value(m))))
            end
        else return quote
            if isdyadic(V)
                $(insert_expr((:N,:M,:bs,:bn),:svec)...)
                out = zeros($VEC(N,$TF))
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds setmulti!(out,$CONJ(m.v[bs[g]+i]),dual(V,ib[i],M))
                    end
                end
            else
                out = $CONJ.(value(m))
            end
            Multivector{dual(V)}(out)
        end end
    end
    @generated function Base.adjoint(m::Spinor{V,T}) where {V,T}
        CONJ,VEC = conjvec(m)
        TF = T ∉ FieldsBig ? :Any : :T
        if mdims(V)<cache_limit
            if isdyadic(V)
                $(insert_expr((:N,:M,:rs,:bn,:t),:mvec)...)
                out = svecs(N,Any)(zeros(svecs(N,t)))
                for g ∈ evens(1,N+1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds setspin!_pre(out,:($CONJ(@inbounds m.v[$(rs[g]+i)])),dual(V,ib[i],M))
                    end
                end
                return :(Spinor{$(dual(V))}($(Expr(:call,tvecs(N,TF),out...))))
            else
                return :(Spinor{$(dual(V))}($CONJ.(value(m))))
            end
        else return quote
            if isdyadic(V)
                $(insert_expr((:N,:M,:rs,:bn),:svec)...)
                out = zeros($VECS(N,$TF))
                for g ∈ 1:2:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds setspin!(out,$CONJ(m.v[rs[g]+i]),dual(V,ib[i],M))
                    end
                end
            else
                out = $CONJ.(value(m))
            end
            Spinor{dual(V)}(out)
        end end
    end
    @generated function Base.adjoint(m::CoSpinor{V,T}) where {V,T}
        CONJ,VEC = conjvec(m)
        TF = T ∉ FieldsBig ? :Any : :T
        if mdims(V)<cache_limit
            if isdyadic(V)
                $(insert_expr((:N,:M,:ps,:bn,:t),:mvec)...)
                out = svecs(N,Any)(zeros(svecs(N,t)))
                for g ∈ evens(2,N+1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds setanti!_pre(out,:($CONJ(@inbounds m.v[$(ps[g]+i)])),dual(V,ib[i],M))
                    end
                end
                return :(CoSpinor{$(dual(V))}($(Expr(:call,tvecs(N,TF),out...))))
            else
                return :(CoSpinor{$(dual(V))}($CONJ.(value(m))))
            end
        else return quote
            if isdyadic(V)
                $(insert_expr((:N,:M,:ps,:bn),:svec)...)
                out = zeros($VECS(N,$TF))
                for g ∈ 2:2:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds setanti!(out,$CONJ(m.v[ps[g]+i]),dual(V,ib[i],M))
                    end
                end
            else
                out = $CONJ.(value(m))
            end
            CoSpinor{dual(V)}(out)
        end end
    end
end

function generate_products(Field=Field,VEC=:mvec,MUL=:*,ADD=:+,SUB=:-,CONJ=:conj,PAR=false)
    VECS = Symbol(string(VEC)*"s")
    if Field == Grassmann.Field
        generate_mutators(:(Variables{M,T}),Number,Expr,SUB,MUL)
    elseif Field ∈ (SymField,:(SymPy.Sym))
        generate_mutators(:(FixedVector{M,T}),Field,set_val,SUB,MUL)
    end
    PAR && (Leibniz.extend_field(eval(Field)); global parsym = (parsym...,eval(Field)))
    TF = Field ∉ FieldsBig ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    Field ∉ Fields && @eval begin
        *(a::F,b::Submanifold{V}) where {F<:$EF,V} = Single{V}(a,b)
        *(a::Submanifold{V},b::F) where {F<:$EF,V} = Single{V}(b,a)
        *(a::F,b::Couple{V,B}) where {F<:$EF,V,B} = Couple{V,B}($Sym.:∏(a,realvalue(b)),$Sym.:∏(a,imagvalue(b)))
        *(a::Couple{V,B},b::F) where {F<:$EF,V,B} = Couple{V,B}($Sym.:∏(realvalue(a),b),$Sym.:∏(imagvalue(a),b))
        *(a::F,b::PseudoCouple{V,B}) where {F<:$EF,V,B} = PseudoCouple{V,B}($Sym.:∏(a,realvalue(b)),$Sym.:∏(a,imagvalue(b)))
        *(a::PseudoCouple{V,B},b::F) where {F<:$EF,V,B} = PseudoCouple{V,B}($Sym.:∏(realvalue(a),b),$Sym.:∏(imagvalue(a),b))
        *(a::F,b::Phasor{V,B}) where {F<:$EF,V,B} = Phasor{V,B}($Sym.:∏(a,realvalue(b)),imagvalue(b))
        *(a::Phasor{V,B},b::F) where {F<:$EF,V,B} = Phasor{V,B}($Sym.:∏(realvalue(a),b),imagvalue(a))
        *(a::F,b::Multivector{V}) where {F<:$EF,V} = Multivector{V}(a*b.v)
        *(a::Multivector{V},b::F) where {F<:$EF,V} = Multivector{V}(a.v*b)
        *(a::F,b::Spinor{V}) where {F<:$EF,V} = Spinor{V}(a*b.v)
        *(a::Spinor{V},b::F) where {F<:$EF,V} = Spinor{V}(a.v*b)
        *(a::F,b::CoSpinor{V}) where {F<:$EF,V} = CoSpinor{V}(a*b.v)
        *(a::CoSpinor{V},b::F) where {F<:$EF,V} = CoSpinor{V}(a.v*b)
        *(a::F,b::Chain{V,G}) where {F<:$EF,V,G} = Chain{V,G}(a*b.v)
        *(a::Chain{V,G},b::F) where {F<:$EF,V,G} = Chain{V,G}(a.v*b)
        *(a::F,b::Single{V,G,B,T} where B) where {F<:$EF,V,G,T} = Single{V,G}($Sym.:∏(a,b.v),basis(b))
        *(a::Single{V,G,B,T} where B,b::F) where {F<:$EF,V,G,T} = Single{V,G}($Sym.:∏(a.v,b),basis(a))
        *(a::F,b::Single{V,G,B,T} where B) where {F<:$EF,V,G,T<:Number} = Single{V,G}($Sym.:∏(a,b.v),basis(b))
        *(a::Single{V,G,B,T} where B,b::F) where {F<:$EF,V,G,T<:Number} = Single{V,G}($Sym.:∏(a.v,b),basis(a))
        adjoint(b::Single{V,G,B,T}) where {V,G,B,T<:$Field} = Single{dual(V),G,B',$TF}($CONJ(value(b)))
        #Base.promote_rule(::Type{Single{V,G,B,T}},::Type{S}) where {V,G,T,B,S<:$Field} = Single{V,G,B,promote_type(T,S)}
        #Base.promote_rule(::Type{Multivector{V,T,B}},::Type{S}) where {V,T,B,S<:$Field} = Multivector{V,promote_type(T,S),B}
    end
    #=Field ∉ Fields && Field≠Any && @eval begin
        Base.promote_rule(::Type{Chain{V,G,T,B}},::Type{S}) where {V,G,T,B,S<:$Field} = Chain{V,G,promote_type(T,S),B}
    end=#
    @eval begin
        Base.:-(a::Single{V,G,B,T}) where {V,G,B,T<:$Field} = Single{V,G,B,$TF}($SUB(value(a)))
        function ⟑(a::Single{V,G,A,T} where {G,A},b::Single{V,L,B,S} where {L,B}) where {V,T<:$Field,S<:$Field}
            ba,bb = basis(a),basis(b)
            v = derive_mul(V,UInt(ba),UInt(bb),a.v,b.v,$MUL)
            v*mul(ba,bb,v)
        end
        function wedgedot_metric(a::Single{V,G,A,T} where {G,A},b::Single{V,L,B,S} where {L,B},g) where {V,T<:$Field,S<:$Field}
            ba,bb = basis(a),basis(b)
            v = derive_mul(V,UInt(ba),UInt(bb),a.v,b.v,$MUL)
            v*mul_metric(ba,bb,g,v)
        end
        ∨(a::$Field,b::$Field) = zero($Field)
        ∧(a::$Field,b::$Field) = $MUL(a,b)
        ∧(a::F,b::B) where B<:TensorTerm{V,G} where {F<:$EF,V,G} = Single{V,G}(a,b)
        ∧(a::A,b::F) where A<:TensorTerm{V,G} where {F<:$EF,V,G} = Single{V,G}(b,a)
        #=∧(a::$Field,b::Chain{V,G,T}) where {V,G,T<:$Field} = Chain{V,G,T}(a.*b.v)
        ∧(a::Chain{V,G,T},b::$Field) where {V,G,T<:$Field} = Chain{V,G,T}(a.v.*b)
        ∧(a::$Field,b::Multivector{V,T}) where {V,T<:$Field} = Multivector{V,T}(a.*b.v)
        ∧(a::Multivector{V,T},b::$Field) where {V,T<:$Field} = Multivector{V,T}(a.v.*b)=#
    end
    for (op,po,eop,bop) ∈ ((:+,:plus,:(+=),ADD),(:-,:minus,:(-=),SUB))
        @eval begin
            function $po(a::Chain{V,G,T},b::Chain{V,L,S}) where {V,G,T<:$Field,L,S<:$Field}
                (G == 0 || G == mdims(V)) && (return $po(Single(a),b))
                (L == 0 || L == mdims(V)) && (return $po(a,Single(b)))
                ((isodd(G) && isodd(L))||(iseven(G) && iseven(L))) && (return $po(multispin(a),multispin(b)))
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[list(r+1,r+bng)] = value(a,$VEC(N,G,t))
                rb = binomsum(N,L)
                Rb = binomial(N,L)
                @inbounds out[list(rb+1,rb+Rb)] = $(bcast(bop,:(value(b,$VEC(N,L,t)),)))
                return Multivector{V}(out)
            end
            function $po(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T<:$Field,S<:$Field}
                return Chain{V,G}($(bcast(bop,:(a.v,b.v))))
            end
            function $po(a::Multivector{V,T},b::Multivector{V,S}) where {V,T<:$Field,S<:$Field}
                #=$(insert_expr((:N,:t),VEC)...)
                out = value(a,$VEC(N,t))
                $(add_val(eop,:out,:(value(b,$VEC(N,t))),bop))
                return Multivector{V}(out)=#
                return Multivector{V}($(bcast(bop,:(a.v,b.v))))
            end
            function $po(a::Chain{V,G,T},b::Multivector{V,S}) where {V,G,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t,:r,:bng),VEC)...)
                out = convert($VEC(N,t),$(bcast(bop,:(value(b,$VEC(N,t)),))))
                @inbounds $(add_val(:(+=),:(out[list(r+1,r+bng)]),:(value(a,$VEC(N,G,t))),ADD))
                return Multivector{V}(out)
            end
            function $po(a::Multivector{V,T},b::Chain{V,G,S}) where {V,T<:$Field,G,S<:$Field}
                $(insert_expr((:N,:t,:r,:bng),VEC)...)
                out = value(a,$VEC(N,t))
                @inbounds $(add_val(eop,:(out[list(r+1,r+bng)]),:(value(b,$VEC(N,G,t))),bop))
                return Multivector{V}(out)
            end
            function $po(a::Spinor{V,T},b::Spinor{V,S}) where {V,T<:$Field,S<:$Field}
                #=$(insert_expr((:N,:t),VEC)...)
                out = value(a,$VECS(N,t))
                $(add_val(eop,:out,:(value(b,$VECS(N,t))),bop))
                return Spinor{V}(out)=#
                return Spinor{V}($(bcast(bop,:(a.v,b.v))))
            end
            function $po(a::CoSpinor{V,T},b::CoSpinor{V,S}) where {V,T<:$Field,S<:$Field}
                #=$(insert_expr((:N,:t),VEC)...)
                out = value(a,$VECS(N,t))
                $(add_val(eop,:out,:(value(b,$VECS(N,t))),bop))
                return CoSpinor{V}(out)=#
                return CoSpinor{V}($(bcast(bop,:(a.v,b.v))))
            end
            function $po(a::Spinor{V,T},b::Multivector{V,S}) where {V,T<:$Field,S<:$Field}
                return $po(Multivector{V}(a),b)
            end
            function $po(a::Multivector{V,T},b::Spinor{V,S}) where {V,T<:$Field,S<:$Field}
                return $po(a,Multivector{V}(b))
            end
            function $po(a::CoSpinor{V,T},b::Multivector{V,S}) where {V,T<:$Field,S<:$Field}
                return $po(Multivector{V}(a),b)
            end
            function $po(a::Multivector{V,T},b::CoSpinor{V,S}) where {V,T<:$Field,S<:$Field}
                return $po(a,Multivector{V}(b))
            end
            function $po(a::Chain{V,G,T},b::Spinor{V,S}) where {V,G,T<:$Field,S<:$Field}
                if iseven(G)
                    $(insert_expr((:N,:t,:rr,:bng),VEC)...)
                    out = convert($VECS(N,t),$(bcast(bop,:(value(b,$VECS(N,t)),))))
                    @inbounds $(add_val(:(+=),:(out[list(rr+1,rr+bng)]),:(value(a,$VEC(N,G,t))),ADD))
                    return Spinor{V}(out)
                else
                    return $po(a,Multivector{V}(b))
                end
            end
            function $po(a::Spinor{V,T},b::Chain{V,G,S}) where {V,T<:$Field,G,S<:$Field}
                if iseven(G)
                    $(insert_expr((:N,:t,:rr,:bng),VEC)...)
                    out = value(a,$VECS(N,t))
                    @inbounds $(add_val(eop,:(out[list(rr+1,rr+bng)]),:(value(b,$VEC(N,G,t))),bop))
                    return Spinor{V}(out)
                else
                    return $po(Multivector{V}(a),b)
                end
            end
            function $po(a::Chain{V,G,T},b::CoSpinor{V,S}) where {V,G,T<:$Field,S<:$Field}
                if isodd(G)
                    $(insert_expr((:N,:t,:rrr,:bng),VEC)...)
                    out = convert($VECS(N,t),$(bcast(bop,:(value(b,$VECS(N,t)),))))
                    @inbounds $(add_val(:(+=),:(out[list(rrr+1,rrr+bng)]),:(value(a,$VEC(N,G,t))),ADD))
                    return CoSpinor{V}(out)
                else
                    return $po(a,Multivector{V}(b))
                end
            end
            function $po(a::CoSpinor{V,T},b::Chain{V,G,S}) where {V,T<:$Field,G,S<:$Field}
                if isodd(G)
                    $(insert_expr((:N,:t,:rrr,:bng),VEC)...)
                    out = value(a,$VECS(N,t))
                    @inbounds $(add_val(eop,:(out[list(rrr+1,rrr+bng)]),:(value(b,$VEC(N,G,t))),bop))
                    return CoSpinor{V}(out)
                else
                    return $po(Multivector{V}(a),b)
                end
            end
        end
    end
end

### Product Algebra

const product_contraction_metric = product_contraction
#=@generated function contraction2(b::Chain{V,G,T},a::TensorTerm{V,L}) where {V,G,L,T}
    product_contraction(a,b,true,false,:product)
end=#
@generated function contraction2(a::TensorGraded{V,L},b::Chain{V,G,T}) where {V,G,L,T}
    product_contraction(a,b,false,false,:product)
end
@generated function contraction2_metric(a::TensorGraded{V,L},b::Chain{V,G,T},g) where {V,G,L,T}
    product_contraction(a,b,false,true,:product)
end
for (mop,prop,field) ∈ ((:⟑,:product,false),(:contraction,:product_contraction,false),(:wedgedot,:product,true),(:contraction,:product_contraction,true))
    op = field ? Symbol(mop,:_metric) : mop
    args = field ? (:g,) : ()
    indu = field ? :(isinduced(g) && (return :($$mop(a,b)))) : nothing
    @eval begin
        @generated function $op(b::Chain{V,G,T},a::TensorTerm{V,L},$(args...)) where {V,G,L,T}
            $indu
            $prop(a,b,true,$field)
        end
        @generated function $op(a::TensorGraded{V,L},b::Chain{V,G,T},$(args...)) where {V,G,L,T}
            $indu
            $prop(a,b,false,$field)
        end
    end
end
for op ∈ (:∧,:∨)
    prop = Symbol(:product_,op)
    @eval begin
        @generated function $op(a::Chain{w,G,T},b::Chain{W,L,S}) where {T,w,S,W,G,L}
            $prop(a,b,false)
        end
        @generated function $op(b::Chain{Q,G,T},a::TensorTerm{R,L}) where {Q,G,T,R,L}
            $prop(a,b,true)
        end
        @generated function $op(a::TensorTerm{Q,G},b::Chain{R,L,T}) where {Q,R,T,G,L}
            $prop(a,b,false)
        end
    end
end
for (mop,product!,field) ∈ ((:∧,:exteraddmulti!,false),(:⟑,:geomaddmulti!,false),
                     (:∨,:meetaddmulti!,false),(:contraction,:skewaddmulti!,false),
                     (:wedgedot,:geomaddmulti!,true),(:contraction,:skewaddmulti!,true))
    op = field ? Symbol(mop,:_metric) : mop
    preproduct! = Symbol(product!,:_pre)
    prop = op∉(:⟑,:wedgedot_metric) ? Symbol(:product_,op) : :product
    args = field ? (:g,) : ()
    indu = field ? :(isinduced(g) && (return :($$mop(a,b)))) : nothing
    @eval begin
        @generated function $op(b::Multivector{V,T},a::TensorGraded{V,G},$(args...)) where {V,T,G}
            $indu
            $prop(a,b,true,$field)
        end
        @generated function $op(a::TensorGraded{V,G},b::Multivector{V,S},$(args...)) where {V,G,S}
            $indu
            $prop(a,b,false,$field)
        end
        @generated function $op(a::Multivector{V,T},b::Multivector{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_multivector(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:Multivector,loop,VEC)
        end
        @generated function $op(a::Spinor{V,T},b::Multivector{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_s_m(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:Multivector,loop,VEC)
        end
        @generated function $op(a::Multivector{V,T},b::Spinor{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_m_s(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:Multivector,loop,VEC)
        end
        @generated function $op(a::CoSpinor{V,T},b::Multivector{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_a_m(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:Multivector,loop,VEC)
        end
        @generated function $op(a::Multivector{V,T},b::CoSpinor{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_m_a(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:Multivector,loop,VEC)
        end
    end
end
for (mop,product!,field) ∈ ((:∧,:exteraddspin!,false),(:⟑,:geomaddspin!,false),
                     (:∨,:meetaddspin!,false),(:contraction,:skewaddspin!,false),
                     (:wedgedot,:geomaddspin!,true),(:contraction,:skewaddspin!,true))
    op = field ? Symbol(mop,:_metric) : mop
    preproduct! = Symbol(product!,:_pre)
    prop = op∉(:⟑,:wedgedot_metric) ? Symbol(:product_,op) : :product
    args = field ? (:g,) : ()
    indu = field ? :(isinduced(g) && (return :($$mop(a,b)))) : nothing
    @eval begin
        @generated function $op(b::Spinor{V,T},a::TensorGraded{V,G},$(args...)) where {V,T,G}
            $indu
            Grassmann.$prop(a,b,true,$field)
        end
        @generated function $op(a::TensorGraded{V,G},b::Spinor{V,S},$(args...)) where {V,G,S}
            $indu
            Grassmann.$prop(a,b,false,$field)
        end
        @generated function $op(b::CoSpinor{V,T},a::TensorGraded{V,G},$(args...)) where {V,T,G}
            $indu
            Grassmann.$prop(a,b,true,$field)
        end
        @generated function $op(a::TensorGraded{V,G},b::CoSpinor{V,S},$(args...)) where {V,G,S}
            $indu
            Grassmann.$prop(a,b,false,$field)
        end
    end
end
for (mop,product!,field) ∈ ((:∧,:exteraddspin!,false),(:⟑,:geomaddspin!,false),(:contraction,:skewaddspin!,false),(:wedgedot,:geomaddspin!,true),(:contraction,:skewaddspin!,true))
    op = field ? Symbol(mop,:_metric) : mop
    preproduct! = Symbol(product!,:_pre)
    prop = op∉(:⟑,:wedgedot_metric) ? Symbol(:product_,op) : :product
    args = field ? (:g,) : ()
    indu = field ? :(isinduced(g) && (return :($$mop(a,b)))) : nothing
    @eval begin
        @generated function $op(a::Spinor{V,T},b::Spinor{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_spinor(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:Spinor,loop,Symbol(string(VEC)*"s"))
        end
        @generated function $op(a::CoSpinor{V,T},b::CoSpinor{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_anti(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:Spinor,loop,Symbol(string(VEC)*"s"))
        end
    end
end
for (mop,product!,field) ∈ ((:∧,:exteraddanti!,false),(:⟑,:geomaddanti!,false),(:contraction,:skewaddanti!,false),(:wedgedot,:geomaddanti!,true),(:contraction,:skewaddanti!,true))
    op = field ? Symbol(mop,:_metric) : mop
    preproduct! = Symbol(product!,:_pre)
    prop = op∉(:⟑,:wedgedot_metric) ? Symbol(:product_,op) : :product
    args = field ? (:g,) : ()
    indu = field ? :(isinduced(g) && (return :($$mop(a,b)))) : nothing
    @eval begin
        @generated function $op(a::Spinor{V,T},b::CoSpinor{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_s_a(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:CoSpinor,loop,Symbol(string(VEC)*"s"))
        end
        @generated function $op(a::CoSpinor{V,T},b::Spinor{V,S},$(args...)) where {V,T,S}
            $indu
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_a_s(V,:(a.v),:(b.v),promote_type(T,S),MUL,$product!,$preproduct!,$field)
            product_loop(V,:CoSpinor,loop,Symbol(string(VEC)*"s"))
        end
    end
end
@generated function ∨(a::Spinor{V,T},b::Spinor{V,S}) where {V,T,S}
    MUL,VEC = mulvec(a,b)
    loop = generate_loop_spinor(V,:(a.v),:(b.v),promote_type(T,S),MUL,isodd(mdims(V)) ? meetaddanti! : meetaddspin!,isodd(mdims(V)) ? meetaddanti!_pre : meetaddspin!_pre)
    product_loop(V,isodd(mdims(V)) ? :CoSpinor : :Spinor,loop,Symbol(string(VEC)*"s"))
end
@generated function ∨(a::CoSpinor{V,T},b::CoSpinor{V,S}) where {V,T,S}
    MUL,VEC = mulvec(a,b)
    loop = generate_loop_anti(V,:(a.v),:(b.v),promote_type(T,S),MUL,isodd(mdims(V)) ? meetaddanti! : meetaddspin!,isodd(mdims(V)) ? meetaddanti!_pre : meetaddspin!_pre)
    product_loop(V,isodd(mdims(V)) ? :CoSpinor : :Spinor,loop,Symbol(string(VEC)*"s"))
end
@generated function ∨(a::Spinor{V,T},b::CoSpinor{V,S}) where {V,T,S}
    MUL,VEC = mulvec(a,b)
    loop = generate_loop_s_a(V,:(a.v),:(b.v),promote_type(T,S),MUL,isodd(mdims(V)) ? meetaddspin! : meetaddanti!,isodd(mdims(V)) ? meetaddspin!_pre : meetaddanti!_pre)
    product_loop(V,isodd(mdims(V)) ? :Spinor : :CoSpinor,loop,Symbol(string(VEC)*"s"))
end
@generated function ∨(a::CoSpinor{V,T},b::Spinor{V,S}) where {V,T,S}
    MUL,VEC = mulvec(a,b)
    loop = generate_loop_a_s(V,:(a.v),:(b.v),promote_type(T,S),MUL,isodd(mdims(V)) ? meetaddspin! : meetaddanti!,isodd(mdims(V)) ? meetaddspin!_pre : meetaddanti!_pre)
    product_loop(V,isodd(mdims(V)) ? :Spinor : :CoSpinor,loop,Symbol(string(VEC)*"s"))
end

for side ∈ (:left,:right)
    cc,p = Symbol(:complement,side),Symbol(:parity,side)
    h,pg,pn = Symbol(cc,:hodge),Symbol(p,:hodge),Symbol(p,:null)
    pnp = :(Leibniz.$(Symbol(pn,:pre)))
    for (c,p,ff) ∈ ((cc,p,false),(h,pg,false),(h,pg,true))
        args = ff ? (:g,) : ()
        adj = c≠cc ? :conj : :identity
        @eval begin
            $c(z::Phasor,$(args...)) = $c(Couple(z),$(args...))
            function $c(z::Couple{V},$(args...)) where V
                G = grade(V)
                Single{V,G,getbasis(V,UInt(1)<<G-1)}(realvalue(z)) + $c(imaginary(z),$(args...))
            end
            $c(z::PseudoCouple{V},$(args...)) where V = $c(volume(z),$(args...)) + $c(imaginary(z),$(args...))
            @generated function $c(b::Chain{V,G,T},$(args...)) where {V,G,T}
                isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
                istangent(V) && (return :($$c(Multivector(b),$(args...))))
                $(c≠h ? nothing : :(((!isdiag(V)) || ($ff && !isinduced(g))) && (return :($$cc(metric(b,$($args...))))) ))
                SUB,VEC,MUL = subvec(b)
                if binomial(mdims(V),G)<(1<<cache_limit)
                    $(insert_expr((:N,:ib,:D),:svec)...)
                    #P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = svec(N,G,Any)(zeros(svec(N,G,T)))
                    D = diffvars(V)
                    for k ∈ list(1,binomial(N,G))
                        B = @inbounds ib[k]
                        val = :($$adj(@inbounds b.v[$k]))
                        v = Expr(:call,MUL,$p(V,B),val)#$(c≠h ? :($pnp(V,B,val)) : :val))
                        setblade!_pre(out,v,complement(N,B,D),Val{N}())
                    end
                    return :(Chain{V,$(N-G)}($(Expr(:call,tvec(N,N-G,T),out...))))
                else return quote
                    $(insert_expr((:N,:ib,:D),:svec)...)
                    #P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = zeros($VEC(N,G,T))
                    D = diffvars(V)
                    for k ∈ 1:binomial(N,G)
                        @inbounds val = b.v[k]
                        if val≠0
                            @inbounds ibk = ib[k]
                            v = $$adj($MUL($$p(V,ibk),val))#$(c≠h ? :($$pn(V,ibk,val)) : :val)))
                            setblade!(out,v,complement(N,ibk,D),Val{N}())
                        end
                    end
                    return Chain{V,N-G}(out)
                end end
            end
            @generated function $c(m::Multivector{V,T},$(args...)) where {V,T}
                isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
                $(c≠h ? nothing : :(((!isdiag(V)) || ($ff && !isinduced(g))) && (return :($$cc(metric(m,$($args...))))) ))
                SUB,VEC = subvec(m)
                if mdims(V)<cache_limit
                    $(insert_expr((:N,:bs,:bn),:svec)...)
                    #P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = svec(N,Any)(zeros(svec(N,T)))
                    D = diffvars(V)
                    for g ∈ list(1,N+1)
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            ibi = @inbounds ib[i]
                            val = :($$adj(@inbounds m.v[$(bs[g]+i)]))
                            v = Expr(:call,:*,$p(V,ibi),val)#$(c≠h ? :($pnp(V,ibi,val)) : :val))
                            @inbounds setmulti!_pre(out,v,complement(N,ibi,D),Val{N}())
                        end
                    end
                    return :(Multivector{V}($(Expr(:call,tvec(N,T),out...))))
                else return quote
                    $(insert_expr((:N,:bs,:bn),:svec)...)
                    #P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = zeros($VEC(N,T))
                    D = diffvars(V)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = m.v[bs[g]+i]
                            if val≠0
                                ibi = @inbounds ib[i]
                                v = $$adj($$p(V,ibi)*val)#$(c≠h ? :($$pn(V,ibi,val)) : :val))
                                setmulti!(out,v,complement(N,ibi,D),Val{N}())
                            end
                        end
                    end
                    return Multivector{V}(out)
                end end
            end
            @generated function $c(m::Spinor{V,T},$(args...)) where {V,T}
                isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
                $(c≠h ? nothing : :(((!isdiag(V)) || ($ff && !isinduced(g))) && (return :($$cc(metric(m,$($args...))))) ))
                SUB,VEC = subvecs(m)
                if mdims(V)<cache_limit
                    $(insert_expr((:N,:rs,:bn),:svec)...)
                    #P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = svecs(N,Any)(zeros(svecs(N,T)))
                    D = diffvars(V)
                    for g ∈ evens(1,N+1)
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            ibi = @inbounds ib[i]
                            val = :($$adj(@inbounds m.v[$(rs[g]+i)]))
                            v = Expr(:call,:*,$p(V,ibi),val)#$(c≠h ? :($pnp(V,ibi,val)) : :val))
                            @inbounds (isodd(N) ? setanti!_pre : setspin!_pre)(out,v,complement(N,ibi,D),Val{N}())
                        end
                    end
                    return :($(isodd(N) ? :CoSpinor : :Spinor){V}($(Expr(:call,tvecs(N,T),out...))))
                else return quote
                    $(insert_expr((:N,:rs,:bn),:svec)...)
                    #P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = zeros($VEC(N,T))
                    D = diffvars(V)
                    for g ∈ 1:2:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = m.v[rs[g]+i]
                            if val≠0
                                ibi = @inbounds ib[i]
                                v = $$adj($$p(V,ibi)*val)#$(c≠h ? :($$pn(V,ibi,val)) : :val))
                                $(isodd(N) ? setanti! : setspin!)(out,v,complement(N,ibi,D),Val{N}())
                            end
                        end
                    end
                    return $(isodd(N) ? :CoSpinor : :Spinor){V}(out)
                end end
            end
            @generated function $c(m::CoSpinor{V,T},$(args...)) where {V,T}
                isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
                $(c≠h ? nothing : :(((!isdiag(V)) || ($ff && !isinduced(g))) && (return :($$cc(metric(m,$($args...))))) ))
                SUB,VEC = subvecs(m)
                if mdims(V)<cache_limit
                    $(insert_expr((:N,:ps,:bn),:svec)...)
                    #P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = svecs(N,Any)(zeros(svecs(N,T)))
                    D = diffvars(V)
                    for g ∈ evens(2,N+1)
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            ibi = @inbounds ib[i]
                            val = :($$adj(@inbounds m.v[$(ps[g]+i)]))
                            v = Expr(:call,:*,$p(V,ibi),val)#$(c≠h ? :($pnp(V,ibi,val)) : :val))
                            @inbounds (isodd(N) ? setspin!_pre : setanti!_pre)(out,v,complement(N,ibi,D),Val{N}())
                        end
                    end
                    return :($(isodd(N) ? :Spinor : :CoSpinor){V}($(Expr(:call,tvecs(N,T),out...))))
                else return quote
                    $(insert_expr((:N,:ps,:bn),:svec)...)
                    #P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = zeros($VEC(N,T))
                    D = diffvars(V)
                    for g ∈ 2:2:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = m.v[ps[g]+i]
                            if val≠0
                                ibi = @inbounds ib[i]
                                v = $$adj($$p(V,ibi)*val)#$(c≠h ? :($$pn(V,ibi,val)) : :val))
                                $(isodd(N) ? setspin! : setanti!)(out,v,complement(N,ibi,D),Val{N}())
                            end
                        end
                    end
                    return $(isodd(N) ? :Spinor : :CoSpinor){V}(out)
                end end
            end
        end
    end
end
for c ∈ (:even,:odd)
    anti = c ≠ :even
    @eval begin
        @generated function $c(m::Multivector{V,T}) where {V,T}
            SUB,VEC = subvecs(m)
            if mdims(V)<cache_limit
                $(insert_expr((:N,:bs,:bn),:svecs)...)
                out = svecs(N,Any)(zeros(svecs(N,T)))
                for g ∈ $(anti ? :(evens(2,N+1)) : :(evens(1,N+1)))
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        ibi = @inbounds ib[i]
                        v = :(@inbounds m.v[$(bs[g]+i)])
                        @inbounds $(anti ? :setanti!_pre : :setspin!_pre)(out,v,ibi,Val{N}())
                    end
                end
                return :($$(anti ? :CoSpinor : :Spinor){V}($(Expr(:call,tvecs(N,T),out...))))
            else return quote
                $(insert_expr((:N,:bs,:bn),:svecs)...)
                out = zeros($VEC(N,T))
                for g ∈ $(anti ? :(2:2:N+1) : :(1:2:N+1))
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds v = m.v[bs[g]+i]
                        if v≠0
                            ibi = @inbounds ib[i]
                            $(anti ? :setanti! : :setspin!)(out,v,ibi,Val{N}())
                        end
                    end
                end
                return $(anti ? :CoSpinor : :Spinor){V}(out)
            end end
        end
    end
end
for c ∈ (:real,:imag)
    par = c≠:real ? :(parityreverse(g-1)) : :(!parityreverse(g-1))
    @eval begin
        @generated function $c(m::Multivector{V,T}) where {V,T}
            SUB,VEC = subvecs(m)
            if mdims(V)<cache_limit
                $(insert_expr((:N,:bs,:bn),:svec)...)
                out = svec(N,Any)(zeros(svec(N,T)))
                for g ∈ list(1,N+1)
                    if $par
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            ibi = @inbounds ib[i]
                            v = :(@inbounds m.v[$(bs[g]+i)])
                            @inbounds setmulti!_pre(out,v,ibi,Val{N}())
                        end
                    end
                end
                return :(Multivector{V}($(Expr(:call,tvec(N,T),out...))))
            else return quote
                $(insert_expr((:N,:bs,:bn),:svec)...)
                out = zeros($VEC(N,T))
                for g ∈ list(1,N+1)
                    if $$par
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds v = m.v[bs[g]+i]
                            if v≠0
                                ibi = @inbounds ib[i]
                                setmulti!(out,v,ibi,Val{N}())
                            end
                        end
                    end
                end
                return Multivector{V}(out)
            end end
        end
        @generated function $c(m::Spinor{V,T}) where {V,T}
            SUB,VEC = subvecs(m)
            if mdims(V)<cache_limit
                $(insert_expr((:N,:rs,:bn),:svecs)...)
                out = svecs(N,Any)(zeros(svecs(N,T)))
                for g ∈ evens(1,N+1)
                    if $par
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            ibi = @inbounds ib[i]
                            v = :(@inbounds m.v[$(rs[g]+i)])
                            @inbounds setspin!_pre(out,v,ibi,Val{N}())
                        end
                    end
                end
                return :(Spinor{V}($(Expr(:call,tvecs(N,T),out...))))
            else return quote
                $(insert_expr((:N,:rs,:bn),:svecs)...)
                out = zeros($VEC(N,T))
                for g ∈ 1:2:N+1
                    if $$par
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds v = m.v[rs[g]+i]
                            if v≠0
                                ibi = @inbounds ib[i]
                                setspin!(out,v,ibi,Val{N}())
                            end
                        end
                    end
                end
                return Spinor{V}(out)
            end end
        end
        @generated function $c(m::CoSpinor{V,T}) where {V,T}
            SUB,VEC = subvecs(m)
            if mdims(V)<cache_limit
                $(insert_expr((:N,:ps,:bn),:svecs)...)
                out = svecs(N,Any)(zeros(svecs(N,T)))
                for g ∈ evens(2,N+1)
                    if $par
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            ibi = @inbounds ib[i]
                            v = :(@inbounds m.v[$(ps[g]+i)])
                            @inbounds setanti!_pre(out,v,ibi,Val{N}())
                        end
                    end
                end
                return :(CoSpinor{V}($(Expr(:call,tvecs(N,T),out...))))
            else return quote
                $(insert_expr((:N,:ps,:bn),:svecs)...)
                out = zeros($VEC(N,T))
                for g ∈ 2:2:N+1
                    if $$par
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds v = m.v[ps[g]+i]
                            if v≠0
                                ibi = @inbounds ib[i]
                                setanti!(out,v,ibi,Val{N}())
                            end
                        end
                    end
                end
                return CoSpinor{V}(out)
            end end
        end
    end
end
for (side,field) ∈ ((:metric,false),(:anti,false),(:metric,true),(:anti,true))
    c,p = (side≠:anti ? side : Symbol(side,:metric)),Symbol(:parity,side)
    tens,tensfull = Symbol(side,:tensor),Symbol(side,:extensor)
    tenseven,tensodd = Symbol(side,:even),Symbol(side,:odd)
    args = field ? (:g,) : ()
    @eval begin
        $c(z::Phasor,$(args...)) = $c(Couple(z),$(args...))
        $c(z::Couple{V},$(args...)) where V = scalar(z) + $c(imaginary(z),$(args...))
        $c(z::PseudoCouple{V},$(args...)) where V = $c(imaginary(z),$(args...)) + $c(volume(z),$(args...))
        @generated function $c(b::Chain{V,G,T},$(args...)) where {V,G,T}
            isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
            istangent(V) && (return :($$c(Multivector(b),$$(args...))))
            ($field && !isinduced(g)) && (return :(contraction(g,b)))
            (!isdiag(V)) && (return :(contraction($($tens(V,G)),b)))
            SUB,VEC,MUL = subvec(b)
            if binomial(mdims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:ib),:svec)...)
                out = svec(N,G,Any)(zeros(svec(N,G,T)))
                for k ∈ list(1,binomial(N,G))
                    B = @inbounds ib[k]
                    val = :(conj(@inbounds b.v[$k]))
                    par = $p(V,B)
                    v = if typeof(par)==Bool
                        par ? :($SUB($val)) : val
                    else
                        Expr(:call,MUL,par,val)
                    end
                    setblade!_pre(out,v,B,Val{N}())
                end
                return :(Chain{V,G}($(Expr(:call,tvec(N,G,T),out...))))
            else return quote
                $(insert_expr((:N,:ib),:svec)...)
                out = zeros($VEC(N,G,T))
                for k ∈ 1:binomial(N,G)
                    @inbounds val = conj(b.v[k])
                    if val≠0
                        @inbounds ibk = ib[k]
                        par = $$p(V,ibk)
                        v = if typeof(par)==Bool
                            par ? $SUB(val) : val
                        else
                            $MUL(par,val)
                        end
                        setblade!(out,v,ibk,Val{N}())
                    end
                end
                return Chain{V,G}(out)
            end end
        end
        @generated function $c(m::Multivector{V,T},$(args...)) where {V,T}
            isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
            ($field && !isinduced(g)) && (return :(contraction(g,m)))
            (!isdiag(V)) && (return :(contraction($($tensfull(V)),m)))
            SUB,VEC = subvec(m)
            if mdims(V)<cache_limit
                $(insert_expr((:N,:bs,:bn),:svec)...)
                out = svec(N,Any)(zeros(svec(N,T)))
                for g ∈ list(1,N+1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        ibi = @inbounds ib[i]
                        val = :(@inbounds conj(m.v[$(bs[g]+i)]))
                        par = $p(V,ibi)
                        v = if typeof(par)==Bool
                            par ? :($SUB($val)) : val
                        else
                            Expr(:call,:*,par,val)
                        end
                        @inbounds setmulti!_pre(out,v,ibi,Val{N}())
                    end
                end
                return :(Multivector{V}($(Expr(:call,tvec(N,T),out...))))
            else return quote
                $(insert_expr((:N,:bs,:bn),:svec)...)
                out = zeros($VEC(N,T))
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds val = conj(m.v[bs[g]+i])
                        if val≠0
                            ibi = @inbounds ib[i]
                            par = $$p(V,ibi)
                            v = if typeof(par)==Bool
                                par ? $SUB(val) : val
                            else
                                par*val
                            end
                            setmulti!(out,v,ibi,Val{N}())
                        end
                    end
                end
                return Multivector{V}(out)
            end end
        end
        @generated function $c(m::Spinor{V,T},$(args...)) where {V,T}
            isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
            ($field && !isinduced(g)) && (return :(contraction(g,m)))
            (!isdiag(V)) && (return :(contraction($($tenseven(V)),m)))
            SUB,VEC = subvecs(m)
            if mdims(V)<cache_limit
                $(insert_expr((:N,:rs,:bn),:svecs)...)
                out = svecs(N,Any)(zeros(svecs(N,T)))
                for g ∈ evens(1,N+1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        ibi = @inbounds ib[i]
                        val = :(@inbounds conj(m.v[$(rs[g]+i)]))
                        par = $p(V,ibi)
                        v = if typeof(par)==Bool
                            par ? :($SUB($val)) : val
                        else
                            Expr(:call,:*,par,val)
                        end
                        @inbounds setspin!_pre(out,v,ibi,Val{N}())
                    end
                end
                return :(Spinor{V}($(Expr(:call,tvecs(N,T),out...))))
            else return quote
                $(insert_expr((:N,:rs,:bn),:svecs)...)
                out = zeros($VEC(N,T))
                for g ∈ 1:2:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds val = conj(m.v[rs[g]+i])
                        if val≠0
                            ibi = @inbounds ib[i]
                            par = $$p(V,ibi)
                            v = if typeof(par)==Bool
                                par ? $SUB(val) : val
                            else
                                par*val
                            end
                            setspin!(out,v,ibi,Val{N}())
                        end
                    end
                end
                return Spinor{V}(out)
            end end
        end
        @generated function $c(m::CoSpinor{V,T},$(args...)) where {V,T}
            isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
            ($field && !isinduced(g)) && (return :(contraction(g,m)))
            (!isdiag(V)) && (return :(contraction($($tensodd(V)),m,)))
            SUB,VEC = subvecs(m)
            if mdims(V)<cache_limit
                $(insert_expr((:N,:ps,:bn),:svecs)...)
                out = svecs(N,Any)(zeros(svecs(N,T)))
                for g ∈ evens(2,N+1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        ibi = @inbounds ib[i]
                        val = :(@inbounds conj(m.v[$(ps[g]+i)]))
                        par = $p(V,ibi)
                        v = if typeof(par)==Bool
                            par ? :($SUB($val)) : val
                        else
                            Expr(:call,:*,par,val)
                        end
                        @inbounds setanti!_pre(out,v,ibi,Val{N}())
                    end
                end
                return :(CoSpinor{V}($(Expr(:call,tvecs(N,T),out...))))
            else return quote
                $(insert_expr((:N,:ps,:bn),:svecs)...)
                out = zeros($VEC(N,T))
                for g ∈ 2:2:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds val = conj(m.v[ps[g]+i])
                        if val≠0
                            ibi = @inbounds ib[i]
                            par = $$p(V,ibi)
                            v = if typeof(par)==Bool
                                par ? $SUB(val) : val
                            else
                                par*val
                            end
                            setanti!(out,v,ibi,Val{N}())
                        end
                    end
                end
                return CoSpinor{V}(out)
            end end
        end
    end
end
for reverse ∈ (:reverse,:involute,:conj,:clifford,:antireverse)
    p = Symbol(:parity,reverse≠:antireverse ? reverse : :reverse)
    g = reverse≠:antireverse ? :grade : :antigrade
    @eval begin
        function $reverse(z::Couple{V,B}) where {V,B}
            Couple{V,B}(realvalue(z),$p($g(B)) ? -imagvalue(z) : imagvalue(z))
        end
        function $reverse(z::PseudoCouple{V,B}) where {V,B}
            PseudoCouple{V,B}($p($g(B)) ? -realvalue(z) : reavalue(z),$p($g(V)) ? -imagvalue(z) : imagvalue(z))
        end
        function $reverse(z::Phasor{V,B}) where {V,B}
            Phasor{V,B}(realvalue(z),$p($g(B)) ? -imagvalue(z) : imagvalue(z))
        end
        @generated function $reverse(b::Chain{V,G,T}) where {V,G,T}
            SUB,VEC = subvec(b)
            if binomial(mdims(V),G)<(1<<cache_limit)
                D = diffvars(V)
                D==0 && !$p($g(b)) && (return :b)
                $(insert_expr((:N,:ib),:svec)...)
                out = svec(N,G,Any)(zeros(svec(N,G,T)))
                for k ∈ list(1,binomial(N,G))
                    v = :(@inbounds b.v[$k])
                    if D==0
                        @inbounds setblade!_pre(out,:($SUB($v)),ib[k],Val{N}())
                    else
                        @inbounds B = ib[k]
                        setblade!_pre(out,$p($g(V,B)) ? :($SUB($v)) : v,B,Val{N}())
                    end
                end
                return :(Chain{V,G}($(Expr(:call,tvec(N,G,T),out...))))
            else return quote
                D = diffvars(V)
                D==0 && !$$p($g(b)) && (return b)
                $(insert_expr((:N,:ib),:svec)...)
                out = zeros($VEC(N,G,T))
                for k ∈ 1:binomial(N,G)
                    @inbounds v = b.v[k]
                    v≠0 && if D==0
                        @inbounds setblade!(out,$SUB(v),ib[k],Val{N}())
                    else
                        @inbounds B = ib[k]
                        setblade!(out,$$p($$g(V,B)) ? $SUB(v) : v,B,Val{N}())
                    end
                end
                return Chain{V,G}(out)
            end end
        end
        @generated function $reverse(m::Multivector{V,T}) where {V,T}
            if mdims(V)<cache_limit
                $(insert_expr((:N,:bs,:bn,:D),:svec)...)
                out = svec(N,Any)(zeros(svec(N,T)))
                for g ∈ list(1,N+1)
                    pg = $p($(reverse≠:antireverse ? :(g-1) : :(N+1-g)))
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        v = :(@inbounds m.v[$(@inbounds bs[g]+i)])
                        if D==0
                            @inbounds setmulti!(out,pg ? :($SUB($v)) : v,ib[i],Val{N}())
                        else
                            @inbounds B = ib[i]
                            setmulti!(out,$p($g(V,B)) ? :($SUB($v)) : v,B,Val{N}())
                        end
                    end
                end
                return :(Multivector{V}($(Expr(:call,tvec(N,T),out...))))
            else return quote
                $(insert_expr((:N,:bs,:bn,:D),:svec)...)
                out = zeros($VEC(N,T))
                for g ∈ 1:N+1
                    pg = $$p($$(reverse≠:antireverse ? :(g-1) : :(N+1-g)))
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds v = m.v[bs[g]+i]
                        v≠0 && if D==0
                            @inbounds setmulti!(out,pg ? $SUB(v) : v,ib[i],Val{N}())
                        else
                            @inbounds B = ib[i]
                            setmulti!(out,$$p($$g(V,B)) ? $SUB(v) : v,B,Val{N}())
                        end
                    end
                end
                return Multivector{V}(out)
            end end
        end
        @generated function $reverse(m::Spinor{V,T}) where {V,T}
            if mdims(V)<cache_limit
                $(insert_expr((:N,:rs,:bn,:D),:svecs)...)
                out = svecs(N,Any)(zeros(svecs(N,T)))
                for g ∈ evens(1,N+1)
                    pg = $p($(reverse≠:antireverse ? :(g-1) : :(N+1-g)))
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        v = :(@inbounds m.v[$(@inbounds rs[g]+i)])
                        if D==0
                            @inbounds setspin!(out,pg ? :($SUB($v)) : v,ib[i],Val{N}())
                        else
                            @inbounds B = ib[i]
                            setspin!(out,$p($g(V,B)) ? :($SUB($v)) : v,B,Val{N}())
                        end
                    end
                end
                return :(Spinor{V}($(Expr(:call,tvecs(N,T),out...))))
            else return quote
                $(insert_expr((:N,:rs,:bn,:D),:svecs)...)
                out = zeros($VECS(N,T))
                for g ∈ 1:2:N+1
                    pg = $$p($$(reverse≠:antireverse ? :(g-1) : :(N+1-g)))
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds v = m.v[rs[g]+i]
                        v≠0 && if D==0
                            @inbounds setspin!(out,pg ? $SUB(v) : v,ib[i],Val{N}())
                        else
                            @inbounds B = ib[i]
                            setspin!(out,$$p($$g(V,B)) ? $SUB(v) : v,B,Val{N}())
                        end
                    end
                end
                return Spinor{V}(out)
            end end
        end
        @generated function $reverse(m::CoSpinor{V,T}) where {V,T}
            if mdims(V)<cache_limit
                $(insert_expr((:N,:ps,:bn,:D),:svecs)...)
                out = svecs(N,Any)(zeros(svecs(N,T)))
                for g ∈ evens(2,N+1)
                    pg = $p($(reverse≠:antireverse ? :(g-1) : :(N+1-g)))
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        v = :(@inbounds m.v[$(@inbounds ps[g]+i)])
                        if D==0
                            @inbounds setanti!(out,pg ? :($SUB($v)) : v,ib[i],Val{N}())
                        else
                            @inbounds B = ib[i]
                            setanti!(out,$p($g(V,B)) ? :($SUB($v)) : v,B,Val{N}())
                        end
                    end
                end
                return :(CoSpinor{V}($(Expr(:call,tvecs(N,T),out...))))
            else return quote
                $(insert_expr((:N,:ps,:bn,:D),:svecs)...)
                out = zeros($VECS(N,T))
                for g ∈ 2:2:N+1
                    pg = $$p($$(reverse≠:antireverse ? :(g-1) : :(N+1-g)))
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds v = m.v[ps[g]+i]
                        v≠0 && if D==0
                            @inbounds setanti!(out,pg ? $SUB(v) : v,ib[i],Val{N}())
                        else
                            @inbounds B = ib[i]
                            setanti!(out,$$p($$g(V,B)) ? $SUB(v) : v,B,Val{N}())
                        end
                    end
                end
                return CoSpinor{V}(out)
            end end
        end
    end
end
