
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export VectorSpace

## VectorSpace{N}

struct VectorSpace{N,D,O,S,C}
    @pure VectorSpace{N,D,O,S,C}() where {N,D,O,S,C} = new{N,D,O,S,C}()
end

@pure VectorSpace{N,D,O,S}() where {N,D,O,S} = VectorSpace{N,D,O,S,0}()

@pure function getindex(::VectorSpace{N,D,O,S} where {N,D,O},i::Int) where S
    d = one(Bits) << (i-1)
    return (d & S) == d
end

@pure getindex(vs::VectorSpace{N,D,O,S} where {N,D,O},i::UnitRange{Int}) where S = [getindex(vs,j) for j ∈ i]
@pure getindex(vs::VectorSpace{N,D,O,S} where {D,O},i::Colon) where {N,S} = [getindex(vs,j) for j ∈ 1:N]
Base.firstindex(m::VectorSpace) = 1
Base.lastindex(m::VectorSpace{N}) where N = N
Base.length(s::VectorSpace{N}) where N = N

@inline sig(s::Bool) = s ? '-' : '+'

@pure VectorSpace{N,D,O}(b::BitArray{1}) where {N,D,O} = VectorSpace{N,D,O,bit2int(b[1:N])}()
@pure VectorSpace{N,D,O}(b::Array{Bool,1}) where {N,D,O} = VectorSpace{N,D,O}(convert(BitArray{1},b))
@pure VectorSpace{N,D,O}(s::String) where {N,D,O} = VectorSpace{N,D,O}([k=='-' for k∈s])
@pure VectorSpace(n::Int,d::Int=0,o::Int=0,s::Bits=zero(Bits)) = VectorSpace{n,d,o,s}()
@pure VectorSpace(str::String) = VectorSpace{length(str)}(str)

@pure function VectorSpace{N}(s::String) where N
    ms = match(r"[0-9]+",s)
    if ms ≠ nothing && String(ms.match) == s
        length(s) < 4 && (s *= join(zeros(Int,5-length(s))))
        VectorSpace(parse(Int,s[1]),do2m(parse(Int,s[2]),parse(Int,s[3]),0),parse(Int,s[4:end]))
    else
        VectorSpace{N,Int('ϵ'∈s),Int('o'∈s)}(replace(replace(s,'ϵ'=>'+'),'o'=>'+'))
    end
end

@inline function show(io::IO,s::VectorSpace)
    hasdual(s) && print(io,'ϵ')
    hasorigin(s) && print(io,'o')
    print(io,sig.(s[hasdual(s)+hasorigin(s)+1:ndims(s)])...)
    C = dualtype(s)
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
end

macro V_str(str)
    VectorSpace(str)
end

@pure flip_sig(N,S::Bits) = Bits(2^N-1) & (~S)

@pure function Base.adjoint(V::VectorSpace{N,D,O,S,C}) where {N,D,O,S,C}
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    VectorSpace{N,D,O,flip_sig(N,S),Int(!Bool(C))}()
end

# direct sum ⨁

export ⊕, ℝ #, ℂ, ℍ

for op ∈ [:(Base.:+),:⊕]
    @eval begin
        @pure function $op(a::VectorSpace{N,0,0,A,0},b::VectorSpace{N,0,0,B,1}) where {N,A,B}
            VectorSpace{2N,0,0,bit2int(BitArray([a[:]; b[:]])),B ≠ flip_sig(N,A) ? 0 : -1}()
        end
        @pure function $op(a::VectorSpace{N,0,0,A,1},b::VectorSpace{N,0,0,B,0}) where {N,A,B}
            VectorSpace{2N,0,0,bit2int(BitArray([a[:]; b[:]])),A ≠ flip_sig(N,B) ? 0 : -1}()
        end
        @pure function $op(a::VectorSpace{N,0,0,A,0},b::VectorSpace{M,0,0,B,0}) where {N,A,M,B}
            VectorSpace{N+M,0,0,bit2int(BitArray([a[:]; b[:]])),0}()
        end
        @pure function $op(a::VectorSpace{N,0,0,A,1},b::VectorSpace{M,0,0,B,1}) where {N,A,M,B}
            VectorSpace{N+M,0,0,bit2int(BitArray([a[:]; b[:]])),1}()
        end
        @pure function $op(a::VectorSpace{N,0,0,A,0},b::VectorSpace{M,0,0,B,1}) where {N,A,M,B}
            VectorSpace{N+M,0,0,bit2int(BitArray([a[:]; b[:]])),0}()
        end
        @pure function $op(a::VectorSpace{N,0,0,A,1},b::VectorSpace{M,0,0,B,0}) where {N,A,M,B}
            VectorSpace{N+M,0,0,bit2int(BitArray([a[:]; b[:]])),0}()
        end
    end
end
for C ∈ [0,1]
    @eval begin
        @pure function Base.:^(v::VectorSpace{N,0,0,S,$C},i::I) where {N,S,I<:Integer}
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

dualtype(V::VectorSpace{N,D,O,S,C} where {N,D,O,S}) where C = C

@pure Base.zero(V::VectorSpace) = 0*one(V)
@pure Base.one(V::VectorSpace) = Basis{V}()

## set theory ∪,∩,⊂,⊆,⊃,⊇

for op ∈ (:(Base.:*),:(Base.:∪))
    @eval begin
        @pure $op(a::VectorSpace{N,D,O,S,C},::VectorSpace{N,D,O,S,C}) where {N,D,O,S,C} = a
        @pure function $op(a::VectorSpace{N1,D1,O1,S1,C1},b::VectorSpace{N2,D2,O2,S2,C2}) where {N1,D1,O1,S1,C1,N2,D2,O2,S2,C2}
            if a==b'
                return a⊕b
            elseif ((N1,D1,O1)==(N2,D2,O2)) || (N1==N2)
                throw(error("VectorSpace intersection ⟨$(a)⟩∩⟨$(b)⟩ incompatible!"))
            else
                throw(error("Arbitrary VectorSpace union not yet implemented."))
            end
        end
    end
end

