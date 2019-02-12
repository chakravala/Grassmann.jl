
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

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

