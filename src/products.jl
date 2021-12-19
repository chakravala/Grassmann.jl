
#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

@pure tvec(N,G,t=Any) = :(Values{$(binomial(N,G)),$t})
@pure tvec(N,t::Type=Any) = :(Values{$(1<<N),$t})
@pure tvec(N,t::Symbol) = :(Values{$(1<<N),$t})
@pure tvec(N,μ::Bool) = tvec(N,μ ? :Any : :t)

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
        sm = Symbol(op,:multi!)
        sb = Symbol(op,:blade!)
        for (s,index) ∈ ((sm,:basisindex),(sb,:bladeindex))
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
    generate_mutators(M,F,set_val,SUB,MUL,:(UInt(i)),SubManifold)
    for (op,set) ∈ ((:add,:(+=)),(:set,:(=)))
        sm = Symbol(op,:multi!)
        sb = Symbol(op,:blade!)
        for s ∈ (sm,sb)
            @eval @inline function $s(out::$M,val::S,i) where {M,T,S}
                @inbounds $(set_val(set,:(out[i]),:val))
                return out
            end
            spre = Symbol(s,:_pre)
            for j ∈ (:join,:geom)
                for S ∈ (s,spre)
                    @eval @inline function $(Symbol(j,S))(m::$M,v::S,A::SubManifold{V},B::SubManifold{V}) where {V,T<:$F,S<:$F,M}
                        $(Symbol(j,S))(V,m,UInt(A),UInt(B),v)
                    end
                end
            end
            @eval begin
                @inline function $(Symbol(:join,s))(V,m::$M,a::UInt,b::UInt,v::S) where {T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        val = $MUL(parityinner(V,A,B),v)
                        if diffvars(V)≠0
                            !iszero(Z) && (T≠Any ? (return true) : (val *= getbasis(loworder(V),Z)))
                            count_ones(Q)+order(val)>diffmode(V) && (return false)
                        end
                        $s(m,val,(A⊻B)|Q,Val(mdims(V)))
                    end
                    return false
                end
                @inline function $(Symbol(:join,spre))(V,m::$M,a::UInt,b::UInt,v::S) where {T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        val = :($$MUL($(parityinner(V,A,B)),$v))
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
                        pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(V,A,B) : (false,A⊻B,false)
                        val = $MUL(parityinner(V,A,B),pcc ? $SUB(v) : v)
                        if istangent(V)
                            !iszero(Z) && (T≠Any ? (return true) : (val *= getbasis(loworder(V),Z)))
                            count_ones(Q)+order(val)>diffmode(V) && (return false)
                        end
                        $s(m,val,bas|Q,Val(mdims(V)))
                        cc && $s(m,hasinforigin(V,A,B) ? $SUB(val) : val,(conformalmask(V)⊻bas)|Q,Val(mdims(V)))
                    end
                    return false
                end
                @inline function $(Symbol(:geom,spre))(V,m::$M,a::UInt,b::UInt,v::S) where {T<:$F,S<:$F,M}
                    if v ≠ 0 && !diffcheck(V,a,b)
                        A,B,Q,Z = symmetricmask(V,a,b)
                        pcc,bas,cc = (hasinf(V) && hasorigin(V)) ? conformal(V,A,B) : (false,A⊻B,false)
                        val = :($$MUL($(parityinner(V,A,B)),$(pcc ? :($$SUB($v)) : v)))
                        if istangent(V)
                            !iszero(Z) && (val = Expr(:call,:*,val,getbasis(loworder(V),Z)))
                            val = :(h=$val;iszero(h)||$(count_ones(Q))+order(h)>$(diffmode(V)) ? 0 : h)
                        end
                        $spre(m,val,bas|Q,Val(mdims(V)))
                        cc && $spre(m,hasinforigin(V,A,B) ? :($$SUB($val)) : val,(conformalmask(V)⊻bas)|Q,Val(mdims(V)))
                    end
                    return false
                end
            end
            for (prod,uct) ∈ ((:meet,:regressive),(:skew,:interior))
                for S ∈ (s,spre)
                    @eval @inline function $(Symbol(prod,S))(m::$M,A::SubManifold{V},B::SubManifold{V},v::T) where {V,T,M}
                        $(Symbol(prod,S))(V,m,UInt(A),UInt(B),v)
                    end
                end
                @eval begin
                    @inline function $(Symbol(prod,s))(V,m::$M,A::UInt,B::UInt,val::T) where {T,M}
                        if val ≠ 0
                            g,C,t,Z = $uct(V,A,B)
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
                    @inline function $(Symbol(prod,spre))(V,m::$M,A::UInt,B::UInt,val::T) where {T,M}
                        if val ≠ 0
                            g,C,t,Z = $uct(V,A,B)
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
                end
            end
        end
    end
end

@inline exterbits(V,α,β) = diffvars(V)≠0 ? ((a,b)=symmetricmask(V,α,β);count_ones(a&b)==0) : count_ones(α&β)==0
@inline exteraddmulti!(V,out,α,β,γ) = exterbits(V,α,β) && joinaddmulti!(V,out,α,β,γ)
@inline exteraddblade!(V,out,α,β,γ) = exterbits(V,α,β) && joinaddblade!(V,out,α,β,γ)
@inline exteraddmulti!_pre(V,out,α,β,γ) = exterbits(V,α,β) && joinaddmulti!_pre(V,out,α,β,γ)
@inline exteraddblade!_pre(V,out,α,β,γ) = exterbits(V,α,β) && joinaddblade!_pre(V,out,α,β,γ)

# algebra

const FieldsBig = (Fields...,BigFloat,BigInt,Complex{BigFloat},Complex{BigInt},Rational{BigInt})

+(b::Chain{V,G},a::SubManifold{V,G}) where {V,G} = a+b
+(b::Chain{V,G},a::SubManifold{V,L}) where {V,G,L} = a+b
+(b::Chain{V,G},a::Simplex{V,G}) where {V,G} = a+b
+(b::Chain{V,G},a::Simplex{V,L}) where {V,G,L} = a+b
+(b::MultiVector{V},a::SubManifold{V,G}) where {V,G} = a+b
+(b::MultiVector{V},a::Simplex{V,G}) where {V,G} = a+b
-(t::SubManifold) = Simplex(-value(t),t)
-(a::Chain{V,G}) where {V,G} = Chain{V,G}(-value(a))
-(a::MultiVector{V,T}) where {V,T} = MultiVector{V}(-value(a))
*(a::Simplex{V,0},b::Chain{V,G}) where {V,G} = Chain{V,G}(a.v*b.v)
*(a::Chain{V,G},b::Simplex{V,0}) where {V,G} = Chain{V,G}(a.v*b.v)
*(a::SubManifold{V,0},b::Chain{W,G}) where {V,W,G} = b
*(a::Chain{V,G},b::SubManifold{W,0}) where {V,W,G} = a

for F ∈ Fields
    @eval begin
        *(a::F,b::MultiVector{V}) where {F<:$F,V} = MultiVector{V}(a*b.v)
        *(a::MultiVector{V},b::F) where {F<:$F,V} = MultiVector{V}(a.v*b)
        *(a::F,b::Chain{V,G}) where {F<:$F,V,G} = Chain{V,G}(a*b.v)
        *(a::Chain{V,G},b::F) where {F<:$F,V,G} = Chain{V,G}(a.v*b)
        *(a::F,b::Simplex{V,G,B,T} where B) where {F<:$F,V,G,T} = Simplex{V,G}($Sym.:∏(a,b.v),basis(b))
        *(a::Simplex{V,G,B,T} where B,b::F) where {F<:$F,V,G,T} = Simplex{V,G}($Sym.:∏(a.v,b),basis(a))
        *(a::F,b::Simplex{V,G,B,T} where B) where {F<:$F,V,G,T<:Number} = Simplex{V,G}(*(a,b.v),basis(b))
        *(a::Simplex{V,G,B,T} where B,b::F) where {F<:$F,V,G,T<:Number} = Simplex{V,G}(*(a.v,b),basis(a))
    end
end
for op ∈ (:+,:-)
    for Term ∈ (:TensorGraded,:TensorMixed)
        @eval begin
            $op(a::T,b::NSE) where T<:$Term = iszero(b) ? a : $op(a,b*g_one(Manifold(a)))
            $op(a::NSE,b::T) where T<:$Term = iszero(a) ? $op(b) : $op(a*g_one(Manifold(b)),b)
        end
    end
    @eval begin
        @generated function $op(a::TensorTerm{V,L},b::TensorTerm{V,G}) where {V,L,G}
            adder(a,b,$(QuoteNode(op)))
        end
        @generated function $op(a::TensorTerm{V,G},b::Chain{V,G,T}) where {V,G,T}
            adder(a,b,$(QuoteNode(op)))
        end
        @generated function $op(a::TensorTerm{V,L},b::Chain{V,G,T}) where {V,G,T,L}
            adder(a,b,$(QuoteNode(op)))
        end
        @generated function $op(a::TensorTerm{V,G},b::MultiVector{V,T}) where {V,G,T}
            adder(a,b,$(QuoteNode(op)))
        end
    end
end
@generated -(b::Chain{V,G,T},a::TensorTerm{V,G}) where {V,G,T} = adder(a,b,:-,true)
@generated -(b::Chain{V,G,T},a::TensorTerm{V,L}) where {V,G,T,L} = adder(a,b,:-,true)
@generated -(b::MultiVector{V,T},a::TensorTerm{V,G}) where {V,G,T} = adder(a,b,:-,true)

@eval begin
    @generated function Base.adjoint(m::Chain{V,G,T}) where {V,G,T}
        CONJ,VEC = conjvec(m)
        TF = T ∉ FieldsBig ? :Any : :T
        if binomial(mdims(V),G)<(1<<cache_limit)
            if isdyadic(V)
                $(insert_expr((:N,:M,:ib),:svec)...)
                out = zeros(svec(N,G,Any))
                for i ∈ 1:binomial(N,G)
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
    @generated function Base.adjoint(m::MultiVector{V,T}) where {V,T}
        CONJ,VEC = conjvec(m)
        TF = T ∉ FieldsBig ? :Any : :T
        if mdims(V)<cache_limit
            if isdyadic(V)
                $(insert_expr((:N,:M,:bs,:bn),:svec)...)
                out = zeros(svec(N,Any))
                for g ∈ 1:N+1
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds setmulti!_pre(out,:($CONJ(@inbounds m.v[$(bs[g]+i)])),dual(V,ib[i],M))
                    end
                end
                return :(MultiVector{$(dual(V))}($(Expr(:call,tvec(N,TF),out...))))
            else
                return :(MultiVector{$(dual(V))}($CONJ.(value(m))))
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
            MultiVector{dual(V)}(out)
        end end
    end
end

function generate_products(Field=Field,VEC=:mvec,MUL=:*,ADD=:+,SUB=:-,CONJ=:conj,PAR=false)
    if Field == Grassmann.Field
        generate_mutators(:(Variables{M,T}),Number,Expr,SUB,MUL)
    elseif Field ∈ (SymField,:(SymPy.Sym))
        generate_mutators(:(FixedVector{M,T}),Field,set_val,SUB,MUL)
    end
    PAR && (Leibniz.extend_field(eval(Field)); global parsym = (parsym...,eval(Field)))
    TF = Field ∉ FieldsBig ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    Field ∉ Fields && @eval begin
        *(a::F,b::SubManifold{V}) where {F<:$EF,V} = Simplex{V}(a,b)
        *(a::SubManifold{V},b::F) where {F<:$EF,V} = Simplex{V}(b,a)
        adjoint(b::Simplex{V,G,B,T}) where {V,G,B,T<:$Field} = Simplex{dual(V),G,B',$TF}($CONJ(value(b)))
        Base.promote_rule(::Type{Simplex{V,G,B,T}},::Type{S}) where {V,G,T,B,S<:$Field} = Simplex{V,G,B,promote_type(T,S)}
        Base.promote_rule(::Type{MultiVector{V,T,B}},::Type{S}) where {V,T,B,S<:$Field} = MultiVector{V,promote_type(T,S),B}
    end
    Field ∉ Fields && Field≠Any && @eval begin
        Base.promote_rule(::Type{Chain{V,G,T,B}},::Type{S}) where {V,G,T,B,S<:$Field} = Chain{V,G,promote_type(T,S),B}
    end
    @eval begin
        Base.:-(a::Simplex{V,G,B,T}) where {V,G,B,T<:$Field} = Simplex{V,G,B,$TF}($SUB(value(a)))
        function *(a::Simplex{V,G,A,T} where {G,A},b::Simplex{V,L,B,S} where {L,B}) where {V,T<:$Field,S<:$Field}
            ba,bb = basis(a),basis(b)
            v = derive_mul(V,UInt(ba),UInt(bb),a.v,b.v,$MUL)
            Simplex(v,mul(ba,bb,v))
        end
        ∧(a::$Field,b::$Field) = $MUL(a,b)
        ∧(a::F,b::B) where B<:TensorTerm{V,G} where {F<:$EF,V,G} = Simplex{V,G}(a,b)
        ∧(a::A,b::F) where A<:TensorTerm{V,G} where {F<:$EF,V,G} = Simplex{V,G}(b,a)
        #=∧(a::$Field,b::Chain{V,G,T}) where {V,G,T<:$Field} = Chain{V,G,T}(a.*b.v)
        ∧(a::Chain{V,G,T},b::$Field) where {V,G,T<:$Field} = Chain{V,G,T}(a.v.*b)
        ∧(a::$Field,b::MultiVector{V,T}) where {V,T<:$Field} = MultiVector{V,T}(a.*b.v)
        ∧(a::MultiVector{V,T},b::$Field) where {V,T<:$Field} = MultiVector{V,T}(a.v.*b)=#
    end
    for (op,eop,bop) ∈ ((:+,:(+=),ADD),(:-,:(-=),SUB))
        @eval begin
            function $op(a::Chain{V,G,T},b::Chain{V,L,S}) where {V,G,T<:$Field,L,S<:$Field}
                $(insert_expr((:N,:t,:out,:r,:bng),VEC)...)
                @inbounds out[r+1:r+bng] = value(a,$VEC(N,G,t))
                rb = binomsum(N,L)
                Rb = binomial(N,L)
                @inbounds out[rb+1:rb+Rb] = $(bcast(bop,:(value(b,$VEC(N,L,t)),)))
                return MultiVector{V}(out)
            end
            function $op(a::Chain{V,G,T},b::Chain{V,G,S}) where {V,G,T<:$Field,S<:$Field}
                return Chain{V,G}($(bcast(bop,:(a.v,b.v))))
            end
            function $op(a::MultiVector{V,T},b::MultiVector{V,S}) where {V,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                out = value(a,$VEC(N,t))
                $(add_val(eop,:out,:(value(b,$VEC(N,t))),bop))
                return MultiVector{V}(out)
            end
            function $op(a::Chain{V,G,T},b::MultiVector{V,S}) where {V,G,T<:$Field,S<:$Field}
                $(insert_expr((:N,:t,:r,:bng),VEC)...)
                out = convert($VEC(N,t),$(bcast(bop,:(value(b,$VEC(N,t)),))))
                @inbounds $(add_val(:(+=),:(out[r+1:r+bng]),:(value(a,$VEC(N,G,t))),ADD))
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::Chain{V,G,S}) where {V,T<:$Field,G,S<:$Field}
                $(insert_expr((:N,:t,:r,:bng),VEC)...)
                out = value(a,$VEC(N,t))
                @inbounds $(add_val(eop,:(out[r+1:r+bng]),:(value(b,$VEC(N,G,t))),bop))
                return MultiVector{V}(out)
            end
        end
    end
end

### Product Algebra

@generated function contraction2(a::TensorGraded{V,L},b::Chain{V,G,T}) where {V,G,L,T}
    product_contraction(a,b,false,:product)
end
for (op,prop) ∈ ((:*,:product),(:contraction,:product_contraction))
    @eval begin
        @generated function $op(b::Chain{V,G,T},a::TensorTerm{V,L}) where {V,G,L,T}
            $prop(a,b,true)
        end
        @generated function $op(a::TensorGraded{V,L},b::Chain{V,G,T}) where {V,G,L,T}
            $prop(a,b)
        end
    end
end
for op ∈ (:∧,:∨)
    prop = Symbol(:product_,op)
    @eval begin
        @generated function $op(a::Chain{w,G,T},b::Chain{W,L,S}) where {T,w,S,W,G,L}
            $prop(a,b)
        end
        @generated function $op(b::Chain{Q,G,T},a::TensorTerm{R,L}) where {Q,G,T,R,L}
            $prop(a,b,true)
        end
        @generated function $op(a::TensorTerm{Q,G},b::Chain{R,L,T}) where {Q,R,T,G,L}
            $prop(a,b)
        end
    end
end
for (op,product!) ∈ ((:∧,:exteraddmulti!),(:*,:geomaddmulti!),
                     (:∨,:meetaddmulti!),(:contraction,:skewaddmulti!))
    preproduct! = Symbol(product!,:_pre)
    prop = op≠:* ? Symbol(:product_,op) : :product
    @eval begin
        @generated function $op(b::MultiVector{V,T},a::TensorGraded{V,G}) where {V,T,G}
            $prop(a,b,true)
        end
        @generated function $op(a::TensorGraded{V,G},b::MultiVector{V,S}) where {V,G,S}
            $prop(a,b)
        end
        @generated function $op(a::MultiVector{V,T},b::MultiVector{V,S}) where {V,T,S}
            MUL,VEC = mulvec(a,b)
            loop = generate_loop_multivector(V,:(a.v),:(b.v),MUL,$product!,$preproduct!)
            if mdims(V)<cache_limit/2
                return insert_t(:(MultiVector{V}($(loop[2].args[2]))))
            else return quote
                $(insert_expr(loop[1],VEC)...)
                $(loop[2])
                return MultiVector{V,t}(out)
            end end
        end
    end
end

for side ∈ (:left,:right)
    c,p = Symbol(:complement,side),Symbol(:parity,side)
    h,pg,pn = Symbol(c,:hodge),Symbol(p,:hodge),Symbol(p,:null)
    pnp = :(Leibniz.$(Symbol(pn,:pre)))
    for (c,p) ∈ ((c,p),(h,pg))
        @eval begin
            @generated function $c(b::Chain{V,G,T}) where {V,G,T}
                isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
                istangent(V) && (return :($$c(MultiVector(b))))
                SUB,VEC,MUL = subvec(b)
                if binomial(mdims(V),G)<(1<<cache_limit)
                    $(insert_expr((:N,:ib,:D),:svec)...)
                    P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = zeros(svec(N,G,Any))
                    D = diffvars(V)
                    for k ∈ 1:binomial(N,G)
                        B = @inbounds ib[k]
                        val = :(conj(@inbounds b.v[$k]))
                        v = Expr(:call,MUL,$p(V,B),$(c≠h ? :($pnp(V,B,val)) : :val))
                        setblade!_pre(out,v,complement(N,B,D,P),Val{N}())
                    end
                    return :(Chain{V,$(N-G)}($(Expr(:call,tvec(N,N-G,:T),out...))))
                else return quote
                    $(insert_expr((:N,:ib,:D),:svec)...)
                    P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = zeros($VEC(N,G,T))
                    D = diffvars(V)
                    for k ∈ 1:binomial(N,G)
                        @inbounds val = b.v[k]
                        if val≠0
                            @inbounds ibk = ib[k]
                            v = conj($MUL($$p(V,ibk),$(c≠h ? :($$pn(V,ibk,val)) : :val)))
                            setblade!(out,v,complement(N,ibk,D,P),Val{N}())
                        end
                    end
                    return Chain{V,N-G}(out)
                end end
            end
            @generated function $c(m::MultiVector{V,T}) where {V,T}
                isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
                SUB,VEC = subvec(m)
                if mdims(V)<cache_limit
                    $(insert_expr((:N,:bs,:bn),:svec)...)
                    P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = zeros(svec(N,Any))
                    D = diffvars(V)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            ibi = @inbounds ib[i]
                            val = :(conj(@inbounds m.v[$(bs[g]+i)]))
                            v = Expr(:call,:*,$p(V,ibi),$(c≠h ? :($pnp(V,ibi,val)) : :val))
                            @inbounds setmulti!_pre(out,v,complement(N,ibi,D,P),Val{N}())
                        end
                    end
                    return :(MultiVector{V}($(Expr(:call,tvec(N,:T),out...))))
                else return quote
                    $(insert_expr((:N,:bs,:bn),:svec)...)
                    P = $(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))
                    out = zeros($VEC(N,T))
                    D = diffvars(V)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = m.v[bs[g]+i]
                            if val≠0
                                ibi = @inbounds ib[i]
                                v = conj($$p(V,ibi)*$(c≠h ? :($$pn(V,ibi,val)) : :val))
                                setmulti!(out,v,complement(N,ibi,D,P),Val{N}())
                            end
                        end
                    end
                    return MultiVector{V}(out)
                end end
            end
        end
    end
end
for reverse ∈ (:reverse,:involute,:conj,:clifford)
    p = Symbol(:parity,reverse)
    @eval begin
        @generated function $reverse(b::Chain{V,G,T}) where {V,G,T}
            SUB,VEC = subvec(b)
            if binomial(mdims(V),G)<(1<<cache_limit)
                D = diffvars(V)
                D==0 && !$p(G) && (return :b)
                $(insert_expr((:N,:ib),:svec)...)
                out = zeros(svec(N,G,Any))
                for k ∈ 1:binomial(N,G)
                    v = :(@inbounds b.v[$k])
                    if D==0
                        @inbounds setblade!_pre(out,:($SUB($v)),ib[k],Val{N}())
                    else
                        @inbounds B = ib[k]
                        setblade!_pre(out,$p(grade(V,B)) ? :($SUB($v)) : v,B,Val{N}())
                    end
                end
                return :(Chain{V,G}($(Expr(:call,tvec(N,G,:T),out...))))
            else return quote
                D = diffvars(V)
                D==0 && !$$p(G) && (return b)
                $(insert_expr((:N,:ib),:svec)...)
                out = zeros($VEC(N,G,T))
                for k ∈ 1:binomial(N,G)
                    @inbounds v = b.v[k]
                    v≠0 && if D==0
                        @inbounds setblade!(out,$SUB(v),ib[k],Val{N}())
                    else
                        @inbounds B = ib[k]
                        setblade!(out,$$p(grade(V,B)) ? $SUB(v) : v,B,Val{N}())
                    end
                end
                return Chain{V,G}(out)
            end end
        end
        @generated function $reverse(m::MultiVector{V,T}) where {V,T}
            if mdims(V)<cache_limit
                $(insert_expr((:N,:bs,:bn,:D),:svec)...)
                out = zeros(svec(N,Any))
                for g ∈ 1:N+1
                    pg = $p(g-1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        v = :(@inbounds m.v[$(@inbounds bs[g]+i)])
                        if D==0
                            @inbounds setmulti!(out,pg ? :($SUB($v)) : v,ib[i],Val{N}())
                        else
                            @inbounds B = ib[i]
                            setmulti!(out,$p(grade(V,B)) ? :($SUB($v)) : v,B,Val{N}())
                        end
                    end
                end
                return :(MultiVector{V}($(Expr(:call,tvec(N,:T),out...))))
            else return quote
                $(insert_expr((:N,:bs,:bn,:D),:svec)...)
                out = zeros($VEC(N,T))
                for g ∈ 1:N+1
                    pg = $$p(g-1)
                    ib = indexbasis(N,g-1)
                    @inbounds for i ∈ 1:bn[g]
                        @inbounds v = m.v[bs[g]+i]
                        v≠0 && if D==0
                            @inbounds setmulti!(out,pg ? $SUB(v) : v,ib[i],Val{N}())
                        else
                            @inbounds B = ib[i]
                            setmulti!(out,$$p(grade(V,B)) ? $SUB(v) : v,B,Val{N}())
                        end
                    end
                end
                return MultiVector{V}(out)
            end end
        end
    end
end
