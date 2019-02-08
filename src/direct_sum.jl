
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export VectorSpace

## VectorSpace{N}

struct VectorSpace{Indices,Options,Signatures}
    @pure VectorSpace{N,M,S}() where {N,M,S} = new{N,M,S}()
end

@pure doc2m(d,o,c=0) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))

#@pure VectorSpace{N,D,O,S}() where {N,D,O,S} = VectorSpace{N,doc2m(D,O,0),S}()

@pure ndims(::VectorSpace{N}) where N = N
@pure hasdual(::VectorSpace{N,M} where N) where M = M ∈ (1,3,5,7,9,11)
@pure hasorigin(::VectorSpace{N,M} where N) where M = M ∈ (2,3,6,7,10,11)
@pure dualtype(::VectorSpace{N,M} where N) where M = M ∈ 8:11 ? -1 : Int(M ∈ (4,5,6,7))
@pure options(::VectorSpace{N,M} where N) where M = M
@pure options_list(V::VectorSpace) = hasdual(V),hasorigin(V),dualtype(V)
@inline value(::VectorSpace{N,M,S} where {N,M}) where S = S
dual(V::VectorSpace) = dualtype(V)<0 ? V : V'
dual(V::VectorSpace{N},B,M=Int(N/2)) where N = ((B<<M)&((1<<N)-1))|(B>>M)
sigcheck(a::VectorSpace,b::VectorSpace) = a ≠ b && throw(error("$(a.s) ≠ $(b.s)"))
@inline sigcheck(a,b) = sigcheck(sig(a),sig(b))

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

@inline function show(io::IO,s::VectorSpace)
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

@pure flip_sig(N,S::Bits) = Bits(2^N-1) & (~S)

@pure function Base.adjoint(V::VectorSpace{N,M,S}) where {N,M,S}
    C = dualtype(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    VectorSpace{N,doc2m(hasdual(V),hasorigin(V),Int(!Bool(C))),flip_sig(N,S)}()
end

# direct sum ⨁

export ⊕, ℝ #, ℂ, ℍ

for op ∈ [:(Base.:+),:⊕]
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
        @pure function Base.:^(v::VectorSpace{N,$M,S},i::I) where {N,S,I<:Integer}
            let V = v
                for k ∈ 2:i
                    V = V⊕v
                end
                return V
            end
        end
    end
end

const ℝ = VectorSpace(1)
#const ℂ = VectorSpace(2)
#const ℍ = VectorSpace(4)

@pure Base.zero(V::VectorSpace) = 0*one(V)
@pure Base.one(V::VectorSpace) = Basis{V}()

## set theory ∪,∩,⊆,⊇

for op ∈ (:(Base.:*),:(Base.:∪))
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

Base.:∩(a::VectorSpace{N,M,S},::VectorSpace{N,M,S}) where {N,M,S} = a
#Base.:∩(a::VectorSpace{N,D,O},::VectorSpace{N,D,O}) where {N,D,O} = V0
Base.:∩(a::VectorSpace{N},::VectorSpace{N}) where N = V0
@pure function Base.:∩(a::VectorSpace{N1,M1,S1},b::VectorSpace{N2,M2,S2}) where {N1,M1,S1,N2,M2,S2}
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

@pure Base.:⊇(a::VectorSpace,b::VectorSpace) = b ⊂ a
Base.:⊆(::VectorSpace{N,M,S},::VectorSpace{N,M,S}) where {N,M,S} = true
#Base.:⊆(::VectorSpace{N,D,O},::VectorSpace{N,D,O}) where {N,D,O} = false
Base.:⊆(::VectorSpace{N},::VectorSpace{N}) where N = false
@pure function Base.:⊆(a::VectorSpace{N1,M1,S1},b::VectorSpace{N2,M2,S2}) where {N1,M1,S1,N2,M2,S2}
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
