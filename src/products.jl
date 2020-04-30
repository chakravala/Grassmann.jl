
#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

### Product Algebra Constructor

@eval function generate_loop_multivector(V,a,b,MUL,product!,preproduct!,d=nothing)
    if ndims(V)<cache_limit/2
        $(insert_expr((:N,:t,:out,:bs,:bn),:svec)...)
        for g ∈ 1:N+1
            X = indexbasis(N,g-1)
            @inbounds for i ∈ 1:bn[g]
                @inbounds val = nothing≠d ? :($a[$(bs[g]+i)]/$d) : :($a[$(bs[g]+i)])
                for G ∈ 1:N+1
                    @inbounds R = bs[G]
                    Y = indexbasis(N,G-1)
                    @inbounds for j ∈ 1:bn[G]
                        @inbounds preproduct!(V,out,X[i],Y[j],derive_pre(V,X[i],Y[j],val,:($b[$(R+j)]),MUL))
                    end
                end
            end
        end
        (:N,:t,:out), :(out .= $(Expr(:call,tvec(N,:t),out...)))
    else
        (:N,:t,:out,:bs,:bn,:μ), quote
            for g ∈ 1:N+1
                X = indexbasis(N,g-1)
                @inbounds for i ∈ 1:bn[g]
                    @inbounds val = $(nothing≠d ? :($a[bs[g]+i]/$d) : :($a[bs[g]+i]))
                    val≠0 && for G ∈ 1:N+1
                        @inbounds R = bs[G]
                        Y = indexbasis(N,G-1)
                        @inbounds for j ∈ 1:bn[G]
                            if @inbounds $product!(V,out,X[i],Y[j],derive_mul(V,X[i],Y[j],val,$b[R+j],$MUL))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $product!(V,out,X[i],Y[j],derive_mul(V,X[i],Y[j],val,$b[R+j],$MUL))
                            end
                        end
                    end
                end
            end
        end
    end
end

insert_t(x) = Expr(:block,:(t=promote_type(valuetype(a),valuetype(b))),x)

function generate_products(Field=Field,VEC=:mvec,MUL=:*,ADD=:+,SUB=:-,CONJ=:conj,PAR=false)
    TF = Field ∉ Fields ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    generate_sums(Field,VEC,MUL,ADD,SUB,CONJ,PAR)
    @eval begin
        function *(a::Simplex{V,G,A,T} where {G,A},b::Simplex{V,L,B,S} where {L,B}) where {V,T<:$Field,S<:$Field}
            ba,bb = basis(a),basis(b)
            v = derive_mul(V,bits(ba),bits(bb),a.v,b.v,$MUL)
            Simplex(v,mul(ba,bb,v))
        end
        @generated function *(a::Chain{V,G,T},b::SubManifold{V,L}) where {V,G,L,T<:$Field}
            if G == 0
                return :(a[1]*b)
            elseif L == ndims(V) && !istangent(V)
                return :(⋆(~a))
            elseif G == ndims(V) && !istangent(V)
                return :(a[1]*complementlefthodge(~b))
            elseif binomial(ndims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:t,:out,:ib,:μ),:svec)...)
                for i ∈ 1:binomial(N,G)
                    @inbounds geomaddmulti!_pre(V,out,ib[i],bits(b),derive_pre(V,ib[i],bits(b),:(a[$i]),true))
                end
                return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
            else return quote
                $(insert_expr((:N,:t,:out,:ib,:μ),$(QuoteNode(VEC)))...)
                for i ∈ 1:binomial(N,G)
                    if @inbounds geomaddmulti!(V,out,ib[i],bits(b),derive_mul(V,ib[i],bits(b),a[i],true))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds geomaddmulti!(V,out,ib[i],bits(b),derive_mul(V,ib[i],bits(b),a[i],true))
                    end
                end
                return MultiVector{V}(out)
            end end
        end
        @generated function *(a::SubManifold{V,L},b::Chain{V,G,T}) where {V,G,L,T<:$Field}
            if G == 0
                return :(a*b[1])
            elseif G == ndims(V) && !istangent(V)
                return :(⋆(~a)*b[1])
            elseif L == ndims(V) && !istangent(V)
                return :(complementlefthodge(~b))
            elseif binomial(ndims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:t,:out,:ib,:μ),:svec)...)
                for i ∈ 1:binomial(N,G)
                    @inbounds geomaddmulti!_pre(V,out,bits(a),ib[i],derive_pre(V,bits(a),ib[i],:(b[$i]),false))
                end
                return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
            else return quote
                $(insert_expr((:N,:t,:out,:ib,:μ),$(QuoteNode(VEC)))...)
                for i ∈ 1:binomial(N,G)
                    if @inbounds geomaddmulti!(V,out,bits(a),ib[i],derive_mul(V,bits(a),ib[i],b[i],false))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds geomaddmulti!(V,out,bits(a),ib[i],derive_mul(V,bits(a),ib[i],b[i],false))
                    end
                end
                return MultiVector{V}(out)
            end end
        end
        @generated function *(a::Chain{V,G,T},b::Simplex{V,L,B,S}) where {V,G,T<:$Field,L,B,S<:$Field}
            if G == 0
                return :(a[1]*b)
            elseif L == ndims(V) && !istangent(V)
                return :(⋆(~a)*value(b))
            elseif G == ndims(V) && !istangent(V)
                return :(a[1]*complementlefthodge(~b))
            elseif ndims(V)<cache_limit
                $(insert_expr((:N,:t,:out,:ib,:μ),:svec)...)
                X = bits(B)
                for i ∈ 1:binomial(N,G)
                    @inbounds geomaddmulti!_pre(V,out,ib[i],X,derive_pre(V,ib[i],B,:(a[$i]),:(b.v),$(QuoteNode(MUL))))
                end
                return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
            else return quote
                $(insert_expr((:N,:t,:out,:ib,:μ),$(QuoteNode(VEC)))...)
                X = bits(basis(b))
                for i ∈ 1:binomial(N,G)
                    if @inbounds geomaddmulti!(V,out,ib[i],X,derive_mul(V,ib[i],B,a[i],b.v,$$MUL))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds geomaddmulti!(V,out,ib[i],X,derive_mul(V,ib[i],B,a[i],b.v,$$MUL))
                    end
                end
                return MultiVector{V}(out)
            end end
        end
        @generated function *(a::Simplex{V,L,B,S},b::Chain{V,G,T}) where {V,G,T<:$Field,L,B,S<:$Field}
            if G == 0
                return :(a*b[1])
            elseif G == ndims(V) && !istangent(V)
                return :(⋆(~a)*b[1])
            elseif L == ndims(V) && !istangent(V)
                return :(value(a)*complementlefthodge(~b))
            elseif ndims(V)<cache_limit
                $(insert_expr((:N,:t,:out,:ib,:μ),:svec)...)
                A = bits(B)
                for i ∈ 1:binomial(N,G)
                    @inbounds geomaddmulti!_pre(V,out,A,ib[i],derive_pre(V,A,ib[i],:(a.v),:(b[$i]),$(QuoteNode(MUL))))
                end
                return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
            else return quote
                $(insert_expr((:N,:t,:out,:ib,:μ),$(QuoteNode(VEC)))...)
                A = bits(basis(a))
                for i ∈ 1:binomial(N,G)
                    if @inbounds geomaddmulti!(V,out,A,ib[i],derive_mul(V,A,ib[i],a.v,b[i],$$MUL))&μ
                        $(insert_expr((:out,);mv=:out)...)
                        @inbounds geomaddmulti!(V,out,A,ib[i],derive_mul(V,A,ib[i],a.v,b[i],$$MUL))
                    end
                end
                return MultiVector{V}(out)
            end end
        end
        @generated function *(a::Chain{V,G,T},b::Chain{V,L,S}) where {V,G,T<:$Field,L,S<:$Field}
            if G == 0
                return :(Chain{V,G}(broadcast($$MUL,Ref(a[1]),b.v)))
            elseif L == 0
                return :(Chain{V,G}(broadcast($$MUL,a.v,Ref(b[1]))))
            elseif G == ndims(V) && !istangent(V)
                return :(a[1]*complementlefthodge(~b))
            elseif L == ndims(V) && !istangent(V)
                return :(⋆(~a)*b[1])
            elseif binomial(ndims(V),G)*binomial(ndims(V),L)<(1<<cache_limit)
                $(insert_expr((:N,:t,:bnl,:ib,:μ),:svec)...)
                out = zeros(svec(N,t))
                B = indexbasis(N,L)
                for i ∈ 1:binomial(N,G)
                    @inbounds v,ibi = :(a[$i]),ib[i]
                    for j ∈ 1:bnl
                        @inbounds geomaddmulti!_pre(V,out,ibi,B[j],derive_pre(V,ibi,B[j],v,:(b[$j]),$(QuoteNode(MUL))))
                    end
                end
                return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
            else return quote
                $(insert_expr((:N,:t,:bnl,:ib,:μ),$(QuoteNode(VEC)))...)
                out = zeros($$VEC(N,t))
                B = indexbasis(N,L)
                for i ∈ 1:binomial(N,G)
                    @inbounds v,ibi = a[i],ib[i]
                    v≠0 && for j ∈ 1:bnl
                        if @inbounds geomaddmulti!(V,out,ibi,B[j],derive_mul(V,ibi,B[j],v,b[j],$$MUL))&μ
                            $(insert_expr((:out,);mv=:out)...)
                            @inbounds geomaddmulti!(V,out,ibi,B[j],derive_mul(V,ibi,B[j],v,b[j],$$MUL))
                        end
                    end
                end
                return MultiVector{V}(out)
            end end
        end
        #=function *(a::Chain{V,1,T},b::Chain{W,1,S}) where {V,T<:$Field,W,S<:$Field}
            !(V == dual(W) && V ≠ W) && throw(error())
            $(insert_expr((:N,:t,:bnl,:ib),VEC)...)
            out = zeros($VEC(N,2,t))
            B = indexbasis(N,L)
            for i ∈ 1:binomial(N,G)
                for j ∈ 1:bnl
                    @inbounds geomaddmulti!(V,out,ib[i],B[j],$MUL(a[i],b[j]))
                end
            end
            return MultiVector{V}(out)
        end=#
        @generated function contraction(a::Chain{V,G,T},b::SubManifold{V,L}) where {V,G,T<:$Field,L}
            G<L && (!istangent(V)) && (return g_zero(V))
            if binomial(ndims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:t,:ib,:bng,:μ),:svec)...)
                out = zeros(μ ? svec(N,Any) : svec(N,G-L,Any))
                for i ∈ 1:bng
                    if μ
                        @inbounds skewaddmulti!_pre(V,out,ib[i],bits(b),derive_pre(V,ib[i],bits(b),:(a[$i]),true))
                    else
                        @inbounds skewaddblade!_pre(V,out,ib[i],bits(b),derive_pre(V,ib[i],bits(b),:(a[$i]),true))
                    end
                end
                #return :(value_diff(Simplex{V,0,$(getbasis(V,0))}($(value(mv)))))
                return if μ
                    insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                else
                    insert_t(:(Chain{$V,G-L}($(Expr(:call,tvec(N,G-L,:t),out...)))))
                end
            else return quote
                $(insert_expr((:N,:t,:ib,:bng,:μ),$(QuoteNode(VEC)))...)
                out = zeros(μ ? $$VEC(N,t) : $$VEC(N,G-L,t))
                for i ∈ 1:bng
                    if μ
                        if @inbounds skewaddmulti!(V,out,ib[i],bits(b),derive_mul(V,ib[i],bits(b),a[i],true))
                            #$(insert_expr((:out,);mv=:(value(mv)))...)
                            out,t = zeros(svec(N,G-L,Any)) .+ out,Any
                            @inbounds skewaddmulti!(V,out,ib[i],bits(b),derive_mul(V,ib[i],bits(b),a[i],true))
                        end
                    else
                        @inbounds skewaddblade!(V,out,ib[i],bits(b),derive_mul(V,ib[i],bits(b),a[i],true))
                    end
                end
                return μ ? MultiVector{V}(out) : value_diff(Chain{V,L-G}(out))
            end end
        end
        @generated function contraction(a::SubManifold{V,L},b::Chain{V,G,T}) where {V,G,T<:$Field,L}
            L<G && (!istangent(V)) && (return g_zero(V))
            if binomial(ndims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:t,:ib,:bng,:μ),:svec)...)
                out = zeros(μ ? svec(N,Any) : svec(N,L-G,Any))
                for i ∈ 1:bng
                    if μ
                        @inbounds skewaddmulti!_pre(V,out,bits(a),ib[i],derive_pre(V,bits(a),ib[i],:(b[$i]),false))
                    else
                        @inbounds skewaddblade!_pre(V,out,bits(a),ib[i],derive_pre(V,bits(a),ib[i],:(b[$i]),false))
                    end
                end
                return if μ
                    insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                else
                    insert_t(:(Chain{$V,L-G}($(Expr(:call,tvec(N,L-G,:t),out...)))))
                end
            else return quote
                $(insert_expr((:N,:t,:ib,:bng,:μ),$(QuoteNode(VEC)))...)
                out = zeros(μ ? $$VEC(N,t) : $$VEC(N,L-G,t))
                for i ∈ 1:bng
                    if μ
                        if @inbounds skewaddmulti!(V,out,bits(a),ib[i],derive_mul(V,bits(a),ib[i],b[i],false))
                            out,t = zeros(svec(N,Any)) .+ out,Any
                            @inbounds skewaddmulti!(V,out,bits(a),ib[i],derive_mul(V,bits(a),ib[i],b[i],false))
                        end
                    else
                        @inbounds skewaddblade!(V,out,bits(a),ib[i],derive_mul(V,bits(a),ib[i],b[i],false))
                    end
                end
                return μ ? MultiVector{V}(out) : value_diff(Chain{V,L-G}(out))
            end end
        end
        @generated function contraction(a::Chain{V,G,T},b::Simplex{V,L,B,S}) where {V,G,T<:$Field,B,S<:$Field,L}
            G<L && (!istangent(V)) && (return g_zero(V))
            if binomial(ndims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:t,:ib,:bng,:μ),:svec)...)
                out,X = zeros(μ ? svec(N,Any) : svec(N,G-L,Any)),bits(B)
                for i ∈ 1:bng
                    @inbounds skewaddblade!_pre(V,out,ib[i],X,derive_pre(V,ib[i],B,:(a[$i]),:(b.v),$(QuoteNode(MUL))))
                end
                return if μ
                    insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                else
                    insert_t(:(value_diff(Chain{$V,G-L}($(Expr(:call,tvec(N,G-L,:t),out...))))))
                end
            else return quote
                $(insert_expr((:N,:t,:ib,:bng,:μ),$(QuoteNode(VEC)))...)
                out,X = zeros($$VEC(N,G-L,t)),bits(B)
                for i ∈ 1:bng
                    if μ
                        if @inbounds skewaddmulti!(V,out,ib[i],X,derive_mul(V,ib[i],B,a[i],b.v,$$MUL))
                            out,t = zeros(svec(N,Any)) .+ out,Any
                            @inbounds skewaddmulti!(V,out,ib[i],X,derive_mul(V,ib[i],B,a[i],b.v,$$MUL))
                        end
                    else
                        @inbounds skewaddblade!(V,out,ib[i],X,derive_mul(V,ib[i],B,a[i],b.v,$$MUL))
                    end
                end
                return μ ? MultiVector{V}(out) : value_diff(Chain{V,G-L}(out))
            end end
        end
        @generated function contraction(a::Simplex{V,L,B,S},b::Chain{V,G,T}) where {V,G,T<:$Field,B,S<:$Field,L}
            L<G && (!istangent(V)) && (return g_zero(V))
            if binomial(ndims(V),G)<(1<<cache_limit)
                $(insert_expr((:N,:t,:ib,:bng,:μ),:svec)...)
                out,A = zeros(μ ? svec(N,Any) : svec(N,L-G,Any)),bits(B)
                for i ∈ 1:bng
                    if μ
                        @inbounds skewaddmulti!_pre(V,out,A,ib[i],derive_pre(V,A,ib[i],:(a.v),:(b[$i]),$(QuoteNode(MUL))))
                    else
                        @inbounds skewaddblade!_pre(V,out,A,ib[i],derive_pre(V,A,ib[i],:(a.v),:(b[$i]),$(QuoteNode(MUL))))
                    end
                end
                return if μ
                    insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                else
                    insert_t(:(value_diff(Chain{$V,L-G}($(Expr(:call,tvec(N,L-G,:t),out...))))))
                end
            else return quote
                $(insert_expr((:N,:t,:ib,:bng,:μ),$(QuoteNode(VEC)))...)
                out,A = zeros(μ ? $$VEC(N,t) : $$VEC(N,L-G,t)),bits(B)
                for i ∈ 1:bng
                    if μ
                        if @inbounds skewaddblade!(V,out,A,ib[i],derive_mul(V,A,ib[i],a.v,b[i],$$MUL))
                            out,t = zeros(svec(N,Any)) .+ out,Any
                            @inbounds skewaddblade!(V,out,A,ib[i],derive_mul(V,A,ib[i],a.v,b[i],$$MUL))
                        end
                    else
                        @inbounds skewaddblade!(V,out,A,ib[i],derive_mul(V,A,ib[i],a.v,b[i],$$MUL))
                    end
                end
                return μ ? MultiVector{V}(out) : value_diff(Chain{V,L-G}(out))
            end end
        end
        @generated function contraction(a::Chain{V,G,T},b::Chain{V,L,S}) where {V,G,L,T<:$Field,S<:$Field}
            G<L && (!istangent(V)) && (return g_zero(V))
            if binomial(ndims(V),G)*binomial(ndims(V),L)<(1<<cache_limit)
                $(insert_expr((:N,:t,:bng,:bnl),:svec)...)
                μ = istangent(V)|DirectSum.hasconformal(V)
                ia = indexbasis(N,G)
                ib = indexbasis(N,L)
                out = zeros(μ ? svec(N,Any) : svec(N,G-L,Any))
                for i ∈ 1:bng
                    @inbounds v,iai = :(a[$i]),ia[i]
                    for j ∈ 1:bnl
                        if μ
                            @inbounds skewaddmulti!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(b[$j]),$(QuoteNode(MUL))))
                        else
                            @inbounds skewaddblade!_pre(V,out,iai,ib[j],derive_pre(V,iai,ib[j],v,:(b[$j]),$(QuoteNode(MUL))))
                        end
                    end
                end
                return if μ
                    if !istangent(V)
                        insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N,:t),out...)))))
                    else
                        insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                    end
                else
                    insert_t(:(value_diff(Chain{$V,G-L}($(Expr(:call,tvec(N,G-L,:t),out...))))))
                end
            else return quote
                $(insert_expr((:N,:t,:bng,:bnl,:μ),$(QuoteNode(VEC)))...)
                ia = indexbasis(N,G)
                ib = indexbasis(N,L)
                out = zeros(μ ? $$VEC(N,t) : $$VEC(N,G-L,t))
                for i ∈ 1:bng
                    @inbounds v,iai = a[i],ia[i]
                    v≠0 && for j ∈ 1:bnl
                        if μ
                            if @inbounds skewaddmulti!(V,out,iai,ib[j],derive_mul(V,iai,ib[j],v,b[j],$$MUL))
                                out,t = zeros(svec(N,Any)) .+ out,Any
                                @inbounds skewaddmulti!(V,out,iai,ib[j],derive_mul(V,iai,ib[j],v,b[j],$$MUL))
                            end
                        else
                            @inbounds skewaddblade!(V,out,iai,ib[j],derive_mul(V,iai,ib[j],v,b[j],$$MUL))
                        end
                    end
                end
                return μ ? MultiVector{V}(out) : value_diff(Chain{V,G-L}(out))
            end end
        end
        ∧(a::$Field,b::$Field) = $MUL(a,b)
        ∧(a::F,b::B) where B<:TensorTerm{V,G} where {F<:$EF,V,G} = Simplex{V,G}(a,b)
        ∧(a::A,b::F) where A<:TensorTerm{V,G} where {F<:$EF,V,G} = Simplex{V,G}(b,a)
        #=∧(a::$Field,b::Chain{V,G,T}) where {V,G,T<:$Field} = Chain{V,G,T}(a.*b.v)
        ∧(a::Chain{V,G,T},b::$Field) where {V,G,T<:$Field} = Chain{V,G,T}(a.v.*b)
        ∧(a::$Field,b::MultiVector{V,T}) where {V,T<:$Field} = MultiVector{V,T}(a.*b.v)
        ∧(a::MultiVector{V,T},b::$Field) where {V,T<:$Field} = MultiVector{V,T}(a.v.*b)
        ∧(a::$Field,b::MultiGrade{V,G}) where V = MultiGrade{V,G}(a.*b.v)
        ∧(a::MultiGrade{V,G},b::$Field) where V = MultiGrade{V,G}(a.v.*b)=#
    end
    for (op,po,GL,grass) ∈ ((:∧,:>,:(G+L),:exter),(:∨,:<,:(G+L-ndims(V)),:meet))
        grassaddmulti! = Symbol(grass,:addmulti!)
        grassaddblade! = Symbol(grass,:addblade!)
        grassaddmulti!_pre = Symbol(grassaddmulti!,:_pre)
        grassaddblade!_pre = Symbol(grassaddblade!,:_pre)
        @eval begin
            @generated function $op(a::Chain{w,G,T},b::SubManifold{W,L}) where {w,G,T<:$Field,W,L}
                V = w==W ? w : ((w==dual(W)) ? (dyadmode(w)≠0 ? W⊕w : w⊕W) : (return :(interop($$op,a,b))))
                $po(G+L,ndims(V)) && (!istangent(V)) && (return g_zero(V))
                if binomial(ndims(w),G)<(1<<cache_limit)
                    $(insert_expr((:N,:t,:μ),VEC,:T,Int)...)
                    ib = indexbasis(ndims(w),G)
                    out = zeros(μ ? svec(N,Any) : svec(N,$GL,Any))
                    C,y = isdual(w),isdual(W) ? dual(V,bits(b)) : bits(b)
                    for i ∈ 1:binomial(ndims(w),G)
                        X = @inbounds C ? dual(V,ib[i]) : ib[i]
                        if μ
                            $grassaddmulti!_pre(V,out,X,y,derive_pre(V,X,y,:(a[$i]),true))
                        else
                            $grassaddblade!_pre(V,out,X,y,derive_pre(V,X,y,:(a[$i]),true))
                        end
                    end
                    return if μ
                        insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                    else
                        insert_t(:(Chain{$V,$$GL}($(Expr(:call,tvec(N,$GL,:t),out...)))))
                    end
                else return quote
                    V = $V
                    $(insert_expr((:N,:t,:μ),$(QuoteNode(VEC)))...)
                    ib = indexbasis(ndims(w),G)
                    out = zeros(μ ? $$VEC(N,t) : $$VEC(N,$$GL,t))
                    C,y = isdual(w),isdual(W) ? dual(V,bits(b)) : bits(b)
                    for i ∈ 1:binomial(ndims(w),G)
                        X = @inbounds C ? dual(V,ib[i]) : ib[i]
                        if μ
                            if @inbounds $$grassaddmulti!(V,out,X,y,derive_mul(V,X,y,a[i],true))
                                out,t = zeros(svec(N,Any)) .+ out,Any
                                @inbounds $$grassaddmulti!(V,out,X,y,derive_mul(V,X,y,a[i],true))
                            end
                        else
                            @inbounds $$grassaddblade!(V,out,X,y,derive_mul(V,X,y,a[i],true))
                        end
                    end
                    return μ ? MultiVector{V}(out) : Chain{V,$$GL}(out)
                end end
            end
            @generated function $op(a::SubManifold{w,G},b::Chain{W,L,T}) where {w,W,T<:$Field,G,L}
                V = w==W ? w : ((w==dual(W)) ? (dyadmode(w)≠0 ? W⊕w : w⊕W) : (return :(interop($$op,a,b))))
                $po(G+L,ndims(V)) && (!istangent(V)) && (return g_zero(V))
                if binomial(ndims(W),L)<(1<<cache_limit)
                    $(insert_expr((:N,:t,:μ),VEC,Int,:T)...)
                    ib = indexbasis(ndims(W),L)
                    out = zeros(μ ? svec(N,Any) : svec(N,$GL,Any))
                    C,x = isdual(W),isdual(w) ? dual(V,bits(a)) : bits(a)
                    for i ∈ 1:binomial(ndims(W),L)
                        X = @inbounds C ? dual(V,ib[i]) : ib[i]
                        if μ
                            $grassaddmulti!_pre(V,out,x,X,derive_pre(V,x,X,:(b[$i]),false))
                        else
                            $grassaddblade!_pre(V,out,x,X,derive_pre(V,x,X,:(b[$i]),false))
                        end
                    end
                    return if μ
                        insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                    else
                        insert_t(:(Chain{$V,$$GL}($(Expr(:call,tvec(N,$GL,:t),out...)))))
                    end
                else return quote
                    V = $V
                    $(insert_expr((:N,:t,:μ),$(QuoteNode(VEC)))...)
                    ib = indexbasis(ndims(W),L)
                    out = zeros(μ ? $$VEC(N,t) : $$VEC(N,$$GL,t))
                    C,x = isdual(W),isdual(w) ? dual(V,bits(a)) : bits(a)
                    for i ∈ 1:binomial(ndims(W),L)
                        X = @inbounds C ? dual(V,ib[i]) : ib[i]
                        if μ
                            if @inbounds $$grassaddmulti!(V,out,x,X,derive_mul(V,x,X,b[i],false))
                                out,t = zeros(svec(N,Any)) .+ out,Any
                                @inbounds $$grassaddmulti!(V,out,x,X,derive_mul(V,x,X,b[i],false))
                            end
                        else
                            @inbounds $$grassaddblade!(V,out,x,X,derive_mul(V,x,X,b[i],false))
                        end
                    end
                    return μ ? MultiVector{V}(out) : Chain{V,$$GL}(out)
                end end
            end
            @generated function $op(a::Chain{w,G,T},b::Simplex{W,L,B,S}) where {w,G,T<:$Field,W,B,S<:$Field,L}
                V = w==W ? w : ((w==dual(W)) ? (dyadmode(w)≠0 ? W⊕w : w⊕W) : (return :(interop($$op,a,b))))
                $po(G+L,ndims(V)) && (!istangent(V)) && (return g_zero(V))
                if binomial(ndims(w),G)<(1<<cache_limit)
                    $(insert_expr((:N,:t,:μ),VEC,:T,:S)...)
                    ib = indexbasis(ndims(w),G)
                    out = zeros(μ ? svec(N,Any) : svec(N,$GL,Any))
                    C,y = isdual(w),isdual(W) ? dual(V,bits(B)) : bits(B)
                    for i ∈ 1:binomial(ndims(w),G)
                        X = @inbounds C ? dual(V,ib[i]) : ib[i]
                        if μ
                            $grassaddmulti!_pre(V,out,X,y,derive_pre(V,X,y,:(a[$i]),:(b.v),$(QuoteNode(MUL))))
                        else
                            $grassaddblade!_pre(V,out,X,y,derive_pre(V,X,y,:(a[$i]),:(b.v),$(QuoteNode(MUL))))
                        end
                    end
                    return if μ
                        insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                    else
                        insert_t(:(Chain{$V,$$GL}($(Expr(:call,tvec(N,$GL,:t),out...)))))
                    end
                else return quote
                    $(insert_expr((:N,:t,:μ),$(QuoteNode(VEC)))...)
                    ib = indexbasis(ndims(w),G)
                    out = zeros(μ ? $$VEC(N,t) : $$VEC(N,$$GL,t))
                    C,y = isdual(w),isdual(W) ? dual(V,bits(B)) : bits(B)
                    for i ∈ 1:binomial(ndims(w),G)
                        X = @inbounds C ? dual(V,ib[i]) : ib[i]
                        if μ
                            if @inbounds $$grassaddmulti!(V,out,X,y,derive_mul(V,X,y,a[i],b.v,$$MUL))
                                out,t = zeros(svec(N,Any)) .+ out,Any
                                @inbounds $$grassaddmulti!(V,out,X,y,derive_mul(V,X,y,a[i],b.v,$$MUL))
                            end
                        else
                            @inbounds $$grassaddblade!(V,out,X,y,derive_mul(V,X,y,a[i],b.v,$$MUL))
                        end
                    end
                    return μ ? MultiVector{V}(out) : Chain{V,$$GL}(out)
                end end
            end
            @generated function $op(a::Simplex{w,G,B,S},b::Chain{W,L,T}) where {T<:$Field,w,W,B,S<:$Field,G,L}
                V = w==W ? w : ((w==dual(W)) ? (dyadmode(w)≠0 ? W⊕w : w⊕W) : (return :(interop($$op,a,b))))
                $po(G+L,ndims(V)) && (!istangent(V)) && (return g_zero(V))
                if binomial(ndims(W),L)<(1<<cache_limit)
                    $(insert_expr((:N,:t,:μ),VEC,:S,:T)...)
                    ib = indexbasis(ndims(W),L)
                    out = zeros(μ ? svec(N,Any) : svec(N,$GL,Any))
                    C,x = isdual(W),isdual(w) ? dual(V,bits(B)) : bits(B)
                    for i ∈ 1:binomial(ndims(W),L)
                        X = @inbounds C ? dual(V,ib[i]) : ib[i]
                        if μ
                            $grassaddmulti!_pre(V,out,x,X,derive_pre(V,x,X,:(a.v),:(b[$i]),$(QuoteNode(MUL))))
                        else
                            $grassaddblade!_pre(V,out,x,X,derive_pre(V,x,X,:(a.v),:(b[$i]),$(QuoteNode(MUL))))
                        end
                    end
                    return if μ
                        insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                    else
                        insert_t(:(Chain{$V,$$GL}($(Expr(:call,tvec(N,$GL,:t),out...)))))
                    end
                else return quote
                    $(insert_expr((:N,:t,:μ),$(QuoteNode(VEC)))...)
                    ib = indexbasis(ndims(W),L)
                    out = zeros(μ ? $$VEC(N,t) : $$VEC(N,$$GL,t))
                    C,x = isdual(W),isdual(w) ? dual(V,bits(B)) : bits(B)
                    for i ∈ 1:binomial(ndims(W),L)
                        X = @inbounds C ? dual(V,ib[i]) : ib[i]
                        if μ
                            if @inbounds $$grassaddmulti!(V,out,x,X,derive_mul(V,x,X,a.v,b[i],$$MUL))
                                out,t = zeros(svec(N,Any)) .+ out,Any
                                @inbounds $$grassaddmulti!(V,out,x,X,derive_mul(V,x,X,a.v,b[i],$$MUL))
                            end
                        else
                            @inbounds $$grassaddblade!(V,out,x,X,derive_mul(V,x,X,a.v,b[i],$$MUL))
                        end
                    end
                    return μ ? MultiVector{V}(out) : Chain{V,$$GL}(out)
                end end
            end
            @generated function $op(a::Chain{w,G,T},b::Chain{W,L,S}) where {T<:$Field,w,S<:$Field,W,G,L}
                V = w==W ? w : ((w==dual(W)) ? (dyadmode(w)≠0 ? W⊕w : w⊕W) : (return :(interop($$op,a,b))))
                $po(G+L,ndims(V)) && (!istangent(V)) && (return g_zero(V))
                if binomial(ndims(w),G)*binomial(ndims(W),L)<(1<<cache_limit)
                    $(insert_expr((:N,:t,:μ),VEC,:T,:S)...)
                    ia = indexbasis(ndims(w),G)
                    ib = indexbasis(ndims(W),L)
                    out = zeros(μ ? svec(N,Any) : svec(N,$GL,Any))
                    CA,CB = isdual(w),isdual(W)
                    for i ∈ 1:binomial(ndims(w),G)
                        @inbounds v,iai = :(a[$i]),ia[i]
                        x = CA ? dual(V,iai) : iai
                        for j ∈ 1:binomial(ndims(W),L)
                            X = @inbounds CB ? dual(V,ib[j]) : ib[j]
                            if μ
                                $grassaddmulti!_pre(V,out,x,X,derive_pre(V,x,X,v,:(b[$j]),$(QuoteNode(MUL))))
                            else
                                $grassaddblade!_pre(V,out,x,X,derive_pre(V,x,X,v,:(b[$j]),$(QuoteNode(MUL))))
                            end
                        end
                    end
                    return if μ
                        insert_t(:(MultiVector{$V}($(Expr(:call,tvec(N),out...)))))
                    else
                        insert_t(:(Chain{$V,$$GL}($(Expr(:call,tvec(N,$GL,:t),out...)))))
                    end
                else return quote
                    $(insert_expr((:N,:t,:μ),$(QuoteNode(VEC)))...)
                    ia = indexbasis(ndims(w),G)
                    ib = indexbasis(ndims(W),L)
                    out = zeros(μ $$VEC(N,t) : $$VEC(N,$$GL,t))
                    CA,CB = isdual(w),isdual(W)
                    for i ∈ 1:binomial(ndims(w),G)
                        @inbounds v,iai = a[i],ia[i]
                        x = CA ? dual(V,iai) : iai
                        v≠0 && for j ∈ 1:binomial(ndims(W),L)
                            X = @inbounds CB ? dual(V,ib[j]) : ib[j]
                            if μ
                                if @inbounds $$grassaddmulti!(V,out,x,X,derive_mul(V,x,X,v,b[j],$$MUL))
                                    out,t = zeros(svec(N,promote_type,Any)) .+ out,Any
                                    @inbounds $$grassaddmulti!(V,out,x,X,derive_mul(V,x,X,v,b[j],$$MUL))
                                end
                            else
                                @inbounds $$grassaddblade!(V,out,x,X,derive_mul(V,x,X,v,b[j],$$MUL))
                            end
                        end
                    end
                    return μ ? MultiVector{V}(out) : Chain{V,$$GL}(out)
                end end
            end
        end
    end
    for (op,product!) ∈ ((:∧,:exteraddmulti!),(:*,:geomaddmulti!),
                         (:∨,:meetaddmulti!),(:contraction,:skewaddmulti!))
        preproduct! = Symbol(product!,:_pre)
        @eval begin
            @generated function $op(a::MultiVector{V,T},b::SubManifold{V,G,B}) where {V,T<:$Field,G,B}
                if ndims(V)<cache_limit
                    $(insert_expr((:N,:t,:out,:bs,:bn),:svec)...)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds $preproduct!(V,out,ib[i],B,derive_pre(V,ib[i],B,:(a.v[$(bs[g]+i)]),true))
                        end
                    end
                    return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,:t),out...)))))
                else return quote
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),$(QuoteNode(VEC)))...)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            if @inbounds $$product!(V,out,ib[i],B,derive_mul(V,ib[i],B,a.v[bs[g]+i],true))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,ib[i],B,derive_mul(V,ib[i],B,a.v[bs[g]+i],true))
                            end
                        end
                    end
                    return MultiVector{V}(out)
                end end
            end
            @generated function $op(a::SubManifold{V,G,A},b::MultiVector{V,T}) where {V,G,A,T<:$Field}
                if ndims(V)<cache_limit
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),:svec)...)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds $preproduct!(V,out,A,ib[i],derive_pre(V,A,ib[i],:(b.v[$(bs[g]+i)]),false))
                        end
                    end
                    return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
                else return quote
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),$(QuoteNode(VEC)))...)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            if @inbounds $$product!(V,out,A,ib[i],derive_mul(V,A,ib[i],b.v[bs[g]+i],false))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,ib[i],derive_mul(V,A,ib[i],b.v[bs[g]+i],false))
                            end
                        end
                    end
                    return MultiVector{V}(out)
                end end
            end
            @generated function $op(a::MultiVector{V,T},b::Simplex{V,G,B,S}) where {V,T<:$Field,G,B,S<:$Field}
                if ndims(V)<cache_limit
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),:svec)...)
                    X = bits(B)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds $preproduct!(V,out,ib[i],X,derive_pre(V,ib[i],B,:(a.v[$(bs[g]+i)]),:(b.v),$(QuoteNode(MUL))))
                        end
                    end
                    return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
                else return quote
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),VEC)...)
                    X = bits(basis(b))
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            if @inbounds $$product!(V,out,ib[i],X,derive_mul(V,ib[i],B,a.v[bs[g]+i],b.v,$$MUL))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,ib[i],X,derive_mul(V,ib[i],B,a.v[bs[g]+i],b.v,$$MUL))
                            end
                        end
                    end
                    return MultiVector{V}(out)
                end end
            end
            @generated function $op(a::Simplex{V,G,B,T},b::MultiVector{V,S}) where {V,G,B,T<:$Field,S<:$Field}
                if ndims(V)<cache_limit
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),:svec)...)
                    A = bits(B)
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds $preproduct!(V,out,A,ib[i],derive_pre(V,A,ib[i],:(a.v),:(b.v[$(bs[g]+i)]),$(QuoteNode(MUL))))
                        end
                    end
                    return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
                else return quote
                    $(insert_expr((:N,:t,:out,:bs,:bn,:μ),$(QuoteNode(VEC)))...)
                    A = bits(basis(a))
                    for g ∈ 1:N+1
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            if @inbounds $$product!(V,out,A,ib[i],derive_mul(V,A,ib[i],a.v,b.v[bs[g]+i],$$MUL))&μ
                                $(insert_expr((:out,);mv=:out)...)
                                @inbounds $$product!(V,out,A,ib[i],derive_mul(V,A,ib[i],a.v,b.v[bs[g]+i],$$MUL))
                            end
                        end
                    end
                    return MultiVector{V}(out)
                end end
            end
            @generated function $op(a::MultiVector{V,T},b::Chain{V,G,S}) where {V,T<:$Field,S<:$Field,G}
                if binomial(ndims(V),G)*(1<<ndims(V))<(1<<cache_limit)
                    $(insert_expr((:N,:t,:out,:bng,:ib,:bs,:bn,:μ),:svec)...)
                    for g ∈ 1:N+1
                        A = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = :(a.v[$(bs[g]+i)])
                            for j ∈ 1:bng
                                @inbounds $preproduct!(V,out,A[i],ib[j],derive_pre(V,A[i],ib[j],val,:(b[$j]),$(QuoteNode(MUL))))
                            end
                        end
                    end
                    return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
                else return quote
                    $(insert_expr((:N,:t,:out,:bng,:ib,:bs,:bn,:μ),$(QuoteNode(VEC)))...)
                    for g ∈ 1:N+1
                        A = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = a.v[bs[g]+i]
                            val≠0 && for j ∈ 1:bng
                                if @inbounds $$product!(V,out,A[i],ib[j],derive_mul(V,A[i],ib[j],val,b[j],$$MUL))&μ
                                    $(insert_expr((:out,);mv=:out)...)
                                    @inbounds $$product!(V,out,A[i],ib[j],derive_mul(V,A[i],ib[j],val,b[j],$$MUL))
                                end
                            end
                        end
                    end
                    return MultiVector{V}(out)
                end end
            end
            @generated function $op(a::Chain{V,G,T},b::MultiVector{V,S}) where {V,G,S<:$Field,T<:$Field}
                if binomial(ndims(V),G)*(1<<ndims(V))<(1<<cache_limit)
                    $(insert_expr((:N,:t,:out,:bng,:ib,:bs,:bn,:μ),:svec)...)
                    for g ∈ 1:N+1
                        B = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = :(b.v[$(bs[g]+i)])
                            for j ∈ 1:bng
                                @inbounds $preproduct!(V,out,ib[j],B[i],derive_pre(V,ib[j],B[i],:(a[$j]),val,$(QuoteNode(MUL))))
                            end
                        end
                    end
                    return insert_t(:(MultiVector{V}($(Expr(:call,tvec(N,μ),out...)))))
                else return quote
                    $(insert_expr((:N,:t,:out,:bng,:ib,:bs,:bn,:μ),$(QuoteNode(VEC)))...)
                    for g ∈ 1:N+1
                        B = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds val = b.v[bs[g]+i]
                            val≠0 && for j ∈ 1:bng
                                if @inbounds $$product!(V,out,ib[j],B[i],derive_mul(V,ib[j],B[i],a[j],val,$$MUL))&μ
                                    $(insert_expr((:out,);mv=:out)...)
                                    @inbounds $$product!(V,out,ib[j],B[i],derive_mul(V,ib[j],B[i],a[j],val,$$MUL))
                                end
                            end
                        end
                    end
                    return MultiVector{V}(out)
                end end
            end
            @generated function $op(a::MultiVector{V,T},b::MultiVector{V,S}) where {V,T<:$Field,S<:$Field}
                loop = generate_loop_multivector(V,:(a.v),:(b.v),$(QuoteNode(MUL)),$product!,$preproduct!)
                if ndims(V)<cache_limit/2
                    return insert_t(:(MultiVector{V}($(loop[2].args[2]))))
                else return quote
                    $(insert_expr(loop[1],$(QuoteNode(VEC)))...)
                    $(loop[2])
                    return MultiVector{V,t}(out)
                end end
            end
        end
    end
    for side ∈ (:left,:right)
        c,p = Symbol(:complement,side),Symbol(:parity,side)
        h,pg,pn = Symbol(c,:hodge),Symbol(p,:hodge),Symbol(p,:null)
        pnp = :(DirectSum.$(Symbol(pn,:pre)))
        for (c,p) ∈ ((c,p),(h,pg))
            @eval begin
                @generated function $c(b::Chain{V,G,T}) where {V,G,T<:$Field}
                    isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
                    if binomial(ndims(V),G)<(1<<cache_limit)
                        $(insert_expr((:N,:ib,:D,:P),:svec)...)
                        out = zeros(svec(N,G,Any))
                        D = diffvars(V)
                        for k ∈ 1:binomial(N,G)
                            val = :(b.v[$k])
                            @inbounds p = $p(V,ib[k])
                            v = $(c≠h ? :($pnp(V,ib[k],val)) : :val)
                            v = typeof(V)<:Signature ? (p ? :($$SUB($v)) : v) : Expr(:call,$MUL,p,v)
                            @inbounds setblade!_pre(out,v,complement(N,ib[k],D,P),Val{N}())
                        end
                        return :(Chain{V,$(N-G)}($(Expr(:call,tvec(N,N-G,:T),out...))))
                    else return quote
                        $(insert_expr((:N,:ib,:D,:P),$(QuoteNode(VEC)))...)
                        out = zeros($$VEC(N,G,T))
                        D = diffvars(V)
                        for k ∈ 1:binomial(N,G)
                            @inbounds val = b.v[k]
                            if val≠0
                                @inbounds p = $$p(V,ib[k])
                                v = $(c≠h ? :($$pn(V,ib[k],val)) : :val)
                                v = typeof(V)<:Signature ? (p ? $$SUB(v) : v) : $$MUL(p,v)
                                @inbounds setblade!(out,v,complement(N,ib[k],D,P),Val{N}())
                            end
                        end
                        return Chain{V,N-G}(out)
                    end end
                end
                @generated function $c(m::MultiVector{V,T}) where {V,T<:$Field}
                    isdyadic(V) && throw(error("Complement for dyadic tensors is undefined"))
                    if ndims(V)<cache_limit
                        $(insert_expr((:N,:bs,:bn,:P),:svec)...)
                        out = zeros(svec(N,Any))
                        D = diffvars(V)
                        for g ∈ 1:N+1
                            ib = indexbasis(N,g-1)
                            @inbounds for i ∈ 1:bn[g]
                                val = :(m.v[$(bs[g]+i)])
                                v = $(c≠h ? :($pnp(V,ib[i],val)) : :val)
                                v = typeof(V)<:Signature ? ($p(V,ib[i]) ? :($$SUB($v)) : v) : Expr(:call,:*,$p(V,ib[i]),v)
                                @inbounds setmulti!_pre(out,v,complement(N,ib[i],D,P),Val{N}())
                            end
                        end
                        return :(MultiVector{V}($(Expr(:call,tvec(N,:T),out...))))
                    else return quote
                        $(insert_expr((:N,:bs,:bn,:P),$(QuoteNode(VEC)))...)
                        out = zeros($$VEC(N,T))
                        D = diffvars(V)
                        for g ∈ 1:N+1
                            ib = indexbasis(N,g-1)
                            @inbounds for i ∈ 1:bn[g]
                                @inbounds val = m.v[bs[g]+i]
                                if val≠0
                                    v = $(c≠h ? :($$pn(V,ib[i],val)) : :val)
                                    v = typeof(V)<:Signature ? ($$p(V,ib[i]) ? $$SUB(v) : v) : $$p(V,ib[i])*v
                                    @inbounds setmulti!(out,v,complement(N,ib[i],D,P),Val{N}())
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
            @generated function $reverse(b::Chain{V,G,T}) where {V,G,T<:$Field}
                if binomial(ndims(V),G)<(1<<cache_limit)
                    D = diffvars(V)
                    D==0 && !$p(G) && (return :b)
                    $(insert_expr((:N,:ib),:svec)...)
                    out = zeros(svec(N,G,Any))
                    for k ∈ 1:binomial(N,G)
                        @inbounds v = :(b.v[$k])
                        if D==0
                            @inbounds setblade!_pre(out,:($$SUB($v)),ib[k],Val{N}())
                        else
                            @inbounds B = ib[k]
                            setblade!_pre(out,$p(grade(V,B)) ? :($$SUB($v)) : v,B,Val{N}())
                        end
                    end
                    return :(Chain{V,G}($(Expr(:call,tvec(N,G,:T),out...))))
                else return quote
                    D = diffvars(V)
                    D==0 && !$$p(G) && (return b)
                    $(insert_expr((:N,:ib),$(QuoteNode(VEC)))...)
                    out = zeros($$VEC(N,G,T))
                    for k ∈ 1:binomial(N,G)
                        @inbounds v = b.v[k]
                        v≠0 && if D==0
                            @inbounds setblade!(out,$$SUB(v),ib[k],Val{N}())
                        else
                            @inbounds B = ib[k]
                            setblade!(out,$$p(grade(V,B)) ? $$SUB(v) : v,B,Val{N}())
                        end
                    end
                    return Chain{V,G}(out)
                end end
            end
            @generated function $reverse(m::MultiVector{V,T}) where {V,T<:$Field}
                if ndims(V)<cache_limit
                    $(insert_expr((:N,:bs,:bn,:D),:svec)...)
                    out = zeros(svec(N,Any))
                    for g ∈ 1:N+1
                        pg = $p(g-1)
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds v = :(m.v[$(bs[g]+i)])
                            if D==0
                                @inbounds setmulti!(out,pg ? :($$SUB($v)) : v,ib[i],Val{N}())
                            else
                                @inbounds B = ib[i]
                                setmulti!(out,$p(grade(V,B)) ? :($$SUB($v)) : v,B,Val{N}())
                            end
                        end
                    end
                    return :(MultiVector{V}($(Expr(:call,tvec(N,:T),out...))))
                else return quote
                    $(insert_expr((:N,:bs,:bn,:D),$(QuoteNode(VEC)))...)
                    out = zeros($$VEC(N,T))
                    for g ∈ 1:N+1
                        pg = $$p(g-1)
                        ib = indexbasis(N,g-1)
                        @inbounds for i ∈ 1:bn[g]
                            @inbounds v = m.v[bs[g]+i]
                            v≠0 && if D==0
                                @inbounds setmulti!(out,pg ? $$SUB(v) : v,ib[i],Val{N}())
                            else
                                @inbounds B = ib[i]
                                setmulti!(out,$$p(grade(V,B)) ? $$SUB(v) : v,B,Val{N}())
                            end
                        end
                    end
                    return MultiVector{V}(out)
                end end
            end
        end
    end
end
