
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

const Sym = :(Reduce.Algebra)
const SymField = Any
const ExprField = Union{Expr,Symbol}

set_val(set,expr) = Expr(:(=),expr,set≠:(=) ? Expr(:call,:($Sym.:+),expr,:val) : :val)

for (op,set) ∈ ((:add,:(+=)),(:set,:(=)))
    sm = Symbol(op,:multi!)
    sb = Symbol(op,:blade!)
    @eval begin
        @inline function $sm(out::SizedArray{Tuple{M},T,1,1},val::T,i::Bits) where {M,T<:SymField}
            @inbounds $(set_val(set,:(out[basisindex(intlog(M),i)])))
            return out
        end
        @inline function $sm(out::Q,val::T,i::Bits,::Dimension{N}) where Q<:SizedArray{Tuple{M},T,1,1} where {M,T<:SymField,N}
            @inbounds $(set_val(set,:(out[basisindex(N,i)])))
            return out
        end
        @inline function $(Symbol(:join,sm))(V::VectorSpace{N,D},m::SizedArray{Tuple{M},T,1,1},v::T,A::Bits,B::Bits) where {N,D,T<:SymField,M}
            !(Bool(D) && isodd(A) && isodd(B)) && $sm(m,parity(A,B,V) ? $Sym.:-(v) : v,A .⊻ B,Dimension{N}())
            return m
        end
        @inline function $(Symbol(:join,sm))(m::SizedArray{Tuple{M},T,1,1},v::T,A::Basis{V},B::Basis{V}) where {V,T<:SymField,M}
            !(hasdual(V) && hasdual(A) && hasdual(B)) && $sm(m,parity(A,B) ? $Sym.:-(v) : v,bits(A) .⊻ bits(B),Dimension{ndims(V)}())
            return m
        end
        @inline function $sb(out::SizedArray{Tuple{M},T,1,1},val::T,i::Basis) where {M,T<:SymField}
            @inbounds $(set_val(set,:(out[bladeindex(intlog(M),bits(i))])))
            return out
        end
        @inline function $sb(out::Q,val::T,i::Basis,::Dimension{N}) where Q<:SizedArray{Tuple{M},T,1,1} where {M,T<:SymField,N}
            @inbounds $(set_val(set,:(out[bladeindex(N,bits(i))])))
            return out
        end
        @inline function $sb(out::SizedArray{Tuple{M},T,1,1},val::T,i::Bits) where {M,T<:SymField}
            @inbounds $(set_val(set,:(out[bladeindex(intlog(M),i)])))
            return out
        end
        @inline function $sb(out::Q,val::T,i::Bits,::Dimension{N}) where Q<:SizedArray{Tuple{M},T,1,1} where {M,T<:SymField,N}
            @inbounds $(set_val(set,:(out[bladeindex(N,i)])))
            return out
        end
    end
end

generate_product_algebra(SymField,:($Sym.:*),:($Sym.:+),:($Sym.:-),:svec,:($Sym.conj))
