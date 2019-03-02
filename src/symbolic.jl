
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

const Sym = :(Reduce.Algebra)
const SymField = Any

set_val(set,expr) = Expr(:(=),expr,set≠:(=) ? Expr(:call,:($Sym.:+),expr,:val) : :val)

for (op,set) ∈ ((:add,:(+=)),(:set,:(=)))
    sm = Symbol(op,:multi!)
    sb = Symbol(op,:blade!)
    for (s,index) ∈ ((sm,:basisindex),(sb,:bladeindex))
        for (i,B) ∈ ((:i,Bits),(:(bits(i)),Basis))
            @eval begin
                @inline function $s(out::SizedArray{Tuple{M},T,1,1},val::T,i::$B) where {M,T<:SymField}
                    @inbounds $(set_val(set,:(out[$index(intlog(M),$i)])))
                    return out
                end
                @inline function $s(out::Q,val::T,i::$B,::Dimension{N}) where Q<:SizedArray{Tuple{M},T,1,1} where {M,T<:SymField,N}
                    @inbounds $(set_val(set,:(out[$index(N,$i)])))
                    return out
                end
            end
        end
    end
    for s ∈ (sm,sb)
        @eval begin
            @inline function $(Symbol(:join,s))(V::VectorSpace{N,D},m::SizedArray{Tuple{M},T,1,1},A::Bits,B::Bits,v::T) where {N,D,T<:SymField,M}
                if  v ≠ 0 && !(Bool(D) && isodd(A) && isodd(B))
                    $s(m,parity(A,B,V) ? $Sym.:-(v) : v,A ⊻ B,Dimension{N}())
                end
                return m
            end
            @inline function $(Symbol(:join,s))(m::SizedArray{Tuple{M},T,1,1},A::Basis{V},B::Basis{V},v::T) where {V,T<:SymField,M}
                if v ≠ 0 && !(hasdual(V) && hasdual(A) && hasdual(B))
                    $s(m,parity(A,B) ? $Sym.:-(v) : v,bits(A) ⊻ bits(B),Dimension{ndims(V)}())
                end
                return m
            end
        end
        for (prod,uct) ∈ ((:meet,:regressive),(:skew,:interior),(:cross,:crossprod))
            @eval begin
                @inline function $(Symbol(prod,s))(V::VectorSpace{N,D},m::SizedArray{Tuple{M},T,1,1},A::Bits,B::Bits,v::T) where {N,D,T<:SymField,M}
                    if v ≠ 0 && !(hasdual(V) && hasdual(A) && hasdual(B))
                        p,C,t = $uct(A,B,V)
                        t && $s(m,p ? $Sym.:-(v) : v,C,Dimension{N}())
                    end
                    return m
                end
                @inline function $(Symbol(prod,s))(m::SizedArray{Tuple{M},T,1,1},A::Basis{V},B::Basis{V},v::T) where {V,T<:SymField,M}
                    if v ≠ 0 && !(hasdual(V) && hasdual(A) && hasdual(B))
                        p,C,t = $uct(bits(A),bits(B),V)
                        t && $s(m,p ? $Sym.:-(v) : v,C,Dimension{N}())
                    end
                    return m
                end
            end
        end
    end
end

generate_product_algebra(SymField,:($Sym.:*),:($Sym.:+),:($Sym.:-),:svec,:($Sym.conj))
