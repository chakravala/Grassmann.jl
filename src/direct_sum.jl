
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export VectorSpace, vectorspace, @V_str, ℝ, ⊕
import Base: getindex, @pure, +, *, ^, ∪, ∩, ⊆, ⊇

## utilities

Bits = UInt

bit2int(b::BitArray{1}) = parse(Bits,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

@pure doc2m(d,o,c=0) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))

## VectorSpace{N}

struct VectorSpace{Indices,Options,Signatures}
    @pure VectorSpace{N,M,S}() where {N,M,S} = new{N,M,S}()
end

@pure VectorSpace{N,M}(b::BitArray{1}) where {N,M} = VectorSpace{N,M,bit2int(b[1:N])}()
@pure VectorSpace{N,M}(b::Array{Bool,1}) where {N,M} = VectorSpace{N,M}(convert(BitArray{1},b))
@pure VectorSpace{N,M}(s::String) where {N,M} = VectorSpace{N,M}([k=='-' for k∈s])
@pure VectorSpace(n::Int,d::Int=0,o::Int=0,s::Bits=zero(Bits)) = VectorSpace{n,doc2m(d,o),s}()
@pure VectorSpace(str::String) = VectorSpace{length(str)}(str)

@pure function VectorSpace{N}(s::String) where N
    ms = match(r"[0-9]+",s)
    if ms ≠ nothing && String(ms.match) == s
        length(s) < 4 && (s *= join(zeros(Int,5-length(s))))
        VectorSpace(parse(Int,s[1]),parse(Int,s[2]),parse(Int,s[3]),UInt(parse(Int,s[4:end])))
    else
        VectorSpace{N,doc2m(Int('ϵ'∈s),Int('o'∈s))}(replace(replace(s,'ϵ'=>'+'),'o'=>'+'))
    end
end

@pure function getindex(::VectorSpace{N,M,S} where {N,M},i::Int) where S
    d = one(Bits) << (i-1)
    return (d & S) == d
end

@pure getindex(vs::VectorSpace{N,M,S} where {N,M},i::UnitRange{Int}) where S = [getindex(vs,j) for j ∈ i]
@pure getindex(vs::VectorSpace{N,M,S} where M,i::Colon) where {N,S} = [getindex(vs,j) for j ∈ 1:N]
Base.firstindex(m::VectorSpace) = 1
Base.lastindex(m::VectorSpace{N}) where N = N
Base.length(s::VectorSpace{N}) where N = N

@inline sig(s::Bool) = s ? '-' : '+'

@inline function Base.show(io::IO,s::VectorSpace)
    print(io,'⟨')
    hasdual(s) && print(io,'ϵ')
    hasorigin(s) && print(io,'o')
    print(io,sig.(s[hasdual(s)+hasorigin(s)+1:ndims(s)])...)
    print(io,'⟩')
    C = dualtype(s)
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
end

macro V_str(str)
    VectorSpace(str)
end

@pure Base.ndims(::VectorSpace{N}) where N = N
@pure hasdual(::VectorSpace{N,M} where N) where M = M ∈ (1,3,5,7,9,11)
@pure hasorigin(::VectorSpace{N,M} where N) where M = M ∈ (2,3,6,7,10,11)
@pure dualtype(::VectorSpace{N,M} where N) where M = M ∈ 8:11 ? -1 : Int(M ∈ (4,5,6,7))
@pure options(::VectorSpace{N,M} where N) where M = M
@pure options_list(V::VectorSpace) = hasdual(V),hasorigin(V),dualtype(V)
@inline value(::VectorSpace{N,M,S} where {N,M}) where S = S

# dual involution

dual(V::VectorSpace) = dualtype(V)<0 ? V : V'
dual(V::VectorSpace{N},B,M=Int(N/2)) where N = ((B<<M)&((1<<N)-1))|(B>>M)

@pure flip_sig(N,S::Bits) = Bits(2^N-1) & (~S)

@pure function Base.adjoint(V::VectorSpace{N,M,S}) where {N,M,S}
    C = dualtype(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    VectorSpace{N,doc2m(hasdual(V),hasorigin(V),Int(!Bool(C))),flip_sig(N,S)}()
end

## default definitions

function vectorspace end
const V0 = VectorSpace(0)
const ℝ = VectorSpace(1)

export ⊕

# direct sum ⨁

for op ∈ (:+,:⊕)
    @eval begin
        @pure function $op(a::VectorSpace{N,X,A},b::VectorSpace{M,Y,B}) where {N,X,A,M,Y,B}
            D1,O1,C1 = options_list(a)
            D2,O2,C2 = options_list(b)
            NM = N == M
            opt = if (D1,O1,C1,D2,O2,C2) == (0,0,0,0,0,0)
                doc2m(0,0,0)
            elseif (D1,O1,C1,D2,O2,C2) == (0,0,1,0,0,1)
                doc2m(0,0,1)
            elseif (D1,O1,C1,D2,O2,C2) == (0,0,0,0,0,1)
                doc2m(0,0,NM ? (B ≠ flip_sig(N,A) ? 0 : -1) : 0)
            elseif (D1,O1,C1,D2,O2,C2) == (0,0,1,0,0,0)
                doc2m(0,0,NM ? (A ≠ flip_sig(N,B) ? 0 : -1) : 0)
            else
                throw(error("arbitrary VectorSpace direct-sums not yet implemented"))
            end
            VectorSpace{N+M,opt,bit2int(BitArray([a[:]; b[:]]))}()
        end
    end
end
for M ∈ (0,4)
    @eval begin
        @pure function ^(v::VectorSpace{N,$M,S},i::I) where {N,S,I<:Integer}
            let V = v
                for k ∈ 2:i
                    V = V⊕v
                end
                return V
            end
        end
    end
end

## set theory ∪,∩,⊆,⊇

for op ∈ (:*,:∪)
    @eval begin
        @pure $op(a::VectorSpace{N,M,S},::VectorSpace{N,M,S}) where {N,M,S} = a
        @pure function $op(a::VectorSpace{N1,M1,S1},b::VectorSpace{N2,M2,S2}) where {N1,M1,S1,N2,M2,S2}
            D1,O1,C1 = options_list(a)
            D2,O2,C2 = options_list(b)
            if ((C1≠C2)&&(C1≥0)&&(C2≥0)) && a==b'
                return C1>0 ? b⊕a : a⊕b
            elseif min(C1,C2)<0 && max(C1,C2)≥0
                Y = C1<0 ? b⊆a : a⊆b
                !Y && throw(error("VectorSpace union $(a)∪$(b) incompatible!"))
                return C1<0 ? a : b
            elseif ((N1,D1,O1)==(N2,D2,O2)) || (N1==N2)
                throw(error("VectorSpace intersection $(a)∩$(b) incompatible!"))
            else
                throw(error("arbitrary VectorSpace union not yet implemented."))
            end
        end
    end
end

∩(a::VectorSpace{N,M,S},::VectorSpace{N,M,S}) where {N,M,S} = a
∩(a::VectorSpace{N},::VectorSpace{N}) where N = V0
@pure function ∩(a::VectorSpace{N1,M1,S1},b::VectorSpace{N2,M2,S2}) where {N1,M1,S1,N2,M2,S2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if ((C1≠C2)&&(C1≥0)&&(C2≥0))
        return V0
    elseif min(C1,C2)<0 && max(C1,C2)≥0
        Y = C1<0
        return (Y ? b⊕b' : a⊕a') == (Y ? a : b) ? Y ? b : a : V0
    else
        throw(error("arbitrary VectorSpace intersection not yet implemented."))
    end
end

@pure ⊇(a::VectorSpace,b::VectorSpace) = b ⊆ a
⊆(::VectorSpace{N,M,S},::VectorSpace{N,M,S}) where {N,M,S} = true
⊆(::VectorSpace{N},::VectorSpace{N}) where N = false
@pure function ⊆(a::VectorSpace{N1,M1,S1},b::VectorSpace{N2,M2,S2}) where {N1,M1,S1,N2,M2,S2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if ((C1≠C2)&&(C1≥0)&&(C2≥0)) || ((C1<0)&&(C2≥0))
        return false
    elseif C2<0 && C1≥0
        return (C1>0 ? a'⊕a : a⊕a') == b
    else
        throw(error("arbitrary VectorSpace subsets not yet implemented."))
    end
end

const pre = ("v","w")
const vsn = (:V,:VV,:W)
const digs = "1234567890"
const low_case,upp_case = "abcdefghijklmnopqrstuvwxyz","ABCDEFGHIJKLMNOPQRSTUVWXYZ"
const low_greek,upp_greek = "αβγδϵζηθικλμνξοπρστυφχψω","ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΡΣΤΥΦΨΩ"
const alphanumv = digs*low_case*upp_case #*low_greek*upp_greek
const alphanumw = digs*upp_case*low_case #*upp_greek*low_greek

const subs = Dict{Int,Char}(
   -1 => 'ϵ',
    0 => 'o',
    1 => '₁',
    2 => '₂',
    3 => '₃',
    4 => '₄',
    5 => '₅',
    6 => '₆',
    7 => '₇',
    8 => '₈',
    9 => '₉',
    10 => '₀',
    [j=>alphanumv[j] for j ∈ 11:36]...
)

const sups = Dict{Int,Char}(
   -1 => 'ϵ',
    0 => 'o',
    1 => '¹',
    2 => '²',
    3 => '³',
    4 => '⁴',
    5 => '⁵',
    6 => '⁶',
    7 => '⁷',
    8 => '⁸',
    9 => '⁹',
    10 => '⁰',
    [j=>alphanumw[j] for j ∈ 11:36]...
)

const VTI = Union{Vector{Int},Tuple,NTuple}

@inline function indexbits(d::Integer,b::VTI)
    out = falses(d)
    for k ∈ b
        out[k] = true
    end
    return out
end

@inline indices(b::Bits) = findall(digits(b,base=2).==1)
@inline shift_indices(V::VectorSpace,b::Bits) = shift_indices(V,indices(b))
function shift_indices(s::VectorSpace{N,M} where N,set::Vector{Int}) where M
    if !isempty(set)
        k = 1
        hasdual(s) && set[1] == 1 && (set[1] = -1; k += 1)
        shift = hasdual(s) + hasorigin(s)
        hasorigin(s) && length(set)>=k && set[k]==shift && (set[k]=0;k+=1)
        shift > 0 && (set[k:end] .-= shift)
    end
    return set
end

@inline printindex(i,e::String=pre[1],t=i>36) = (e≠pre[1])⊻t ? sups[t ? i-26 : i] : subs[t ? i-26 : i]
@inline printindices(io::IO,b::VTI,e::String=pre[1]) = print(io,e,[printindex(i,e) for i ∈ b]...)
@inline function printindices(io::IO,a::VTI,b::VTI,e::String=pre[1],f::String=pre[2])
    F = !isempty(b)
    !(F && isempty(a)) && printindices(io,a,e)
    F && printindices(io,b,f)
end
@inline function printindices(io::IO,V::VectorSpace,e::Bits)
    C = dualtype(V)
    if C < 0
        N = Int(ndims(V)/2)
        printindices(io,shift_indices(V,e & Bits(2^N-1)),shift_indices(V,e>>N))
    else
        printindices(io,shift_indices(V,e),C>0 ? pre[2] : pre[1])
    end
end

# universal root Tensor type

abstract type TensorAlgebra{V} end

# parameters accessible from anywhere

Base.@pure vectorspace(::T) where T<:TensorAlgebra{V} where V = V

# universal vector space interopability

@inline interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = op(a,b)

# ^^ identity ^^ | vv union vv #

@inline function interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{W}} where {V,W}
    VW = V∪W
    return op(VW(a),VW(b))
end

# abstract tensor form evaluation

@inline interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = a(b)
@inline function interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{W}} where {V,W}
    VW = V∪W
    return VW(a)(VW(b))
end

# extended compatibility interface

export interop, TensorAlgebra, interform, ⊗

# some shared presets

for op ∈ (:(Base.:+),:(Base.:-),:(Base.:*),:⊗)
    @eval begin
        @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop($op,a,b)
    end
end

