module FewSpecialFunctionsExt

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

using Grassmann
isdefined(Grassmann, :Requires) ? (import Grassmann: FewSpecialFunctions) : (using FewSpecialFunctions)

for fun ∈ (:η,:FresnelC,:FresnelS,:FresnelE,:Ci_complex)
    @eval begin
        FewSpecialFunctions.$fun(x::TensorTerm{V}) where V = FewSpecialFunctions.$fun(Couple(x))
        FewSpecialFunctions.$fun(x::TensorTerm{V,0}) where V = Single{V}(FewSpecialFunctions.$fun(Real(x)))
        FewSpecialFunctions.$fun(x::Couple{V,B}) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(Complex(x)))
        FewSpecialFunctions.$fun(x::Chain) = vectorize(FewSpecialFunctions.$fun(complexify(x)))
        FewSpecialFunctions.$fun(x::Phasor) = FewSpecialFunctions.$fun(complexify(x))
    end
end
for fun ∈ (:C,:η)
    @eval begin
        FewSpecialFunctions.$fun(x,y::TensorTerm{V}) where V = FewSpecialFunctions.$fun(x,Couple(y))
        FewSpecialFunctions.$fun(x,y::TensorTerm{V,0}) where V = Single{V}(FewSpecialFunctions.$fun(x,Real(y)))
        FewSpecialFunctions.$fun(x,y::Couple{V,B}) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(x,Complex(y)))
        FewSpecialFunctions.$fun(x,y::Chain) = vectorize(FewSpecialFunctions.$fun(x,complexify(y)))
        FewSpecialFunctions.$fun(x,y::Phasor) = FewSpecialFunctions.$fun(x,complexify(y))
        FewSpecialFunctions.$fun(x::TensorTerm{V},y) where V = FewSpecialFunctions.$fun(Couple(x),y)
        FewSpecialFunctions.$fun(x::TensorTerm{V,0},y) where V = Single{V}(FewSpecialFunctions.$fun(Real(x),y))
        FewSpecialFunctions.$fun(x::Couple{V,B},y) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(Complex(x),y))
        FewSpecialFunctions.$fun(x::Chain,y) = vectorize(FewSpecialFunctions.$fun(complexify(x),y))
        FewSpecialFunctions.$fun(x::Phasor,y) = FewSpecialFunctions.$fun(complexify(x),y)
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V}) where V = FewSpecialFunctions.$fun(Couple(x),Couple(y))
        FewSpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0}) where V = Single{V}(FewSpecialFunctions.$fun(Real(x),Real(y)))
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V,0}) where V = FewSpecialFunctions.$fun(Couple(x),Real(y))
        FewSpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V}) where V = FewSpecialFunctions.$fun(Couple(x),Real(y))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(Complex(x),Complex(y)))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V,0}) where {V,B} = FewSpecialFunctions.$fun(x,Real(y))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V}) where {V,B} = FewSpecialFunctions.$fun(x,Couple(y))
        FewSpecialFunctions.$fun(x::TensorTerm{V,0},y::Couple{V,B}) where {V,B} = FewSpecialFunctions.$fun(Real(x),y)
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B}) where {V,B} = FewSpecialFunctions.$fun(Couple(x),y)
    end
end
for fun ∈ (:F,:G,:H⁺,:H⁻,:MarcumQ,:dQdb,:F_clausen)
    @eval begin
        FewSpecialFunctions.$fun(x::TensorTerm{V},y,z) where V = FewSpecialFunctions.$fun(Couple(x),y,z)
        FewSpecialFunctions.$fun(x::TensorTerm{V,0},y,z) where V = Single{V}(FewSpecialFunctions.$fun(Real(x),k,z))
        FewSpecialFunctions.$fun(x::Couple{V,B},y,z) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(Complex(x),y,z))
        FewSpecialFunctions.$fun(x,y::TensorTerm{V},z) where V = FewSpecialFunctions.$fun(Couple(x),y,z)
        FewSpecialFunctions.$fun(x,y::TensorTerm{V,0},z) where V = Single{V}(FewSpecialFunctions.$fun(Real(x),y,z))
        FewSpecialFunctions.$fun(x,y::Couple{V,B},z) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(Complex(x),y,z))
        FewSpecialFunctions.$fun(x,y,z::TensorTerm{V}) where V = FewSpecialFunctions.$fun(x,k,Couple(y))
        FewSpecialFunctions.$fun(x,y,z::TensorTerm{V,0}) where V = Single{V}(FewSpecialFunctions.$fun(x,k,Real(y)))
        FewSpecialFunctions.$fun(x,y,z::Couple{V,B}) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(x,k,Complex(y)))
        FewSpecialFunctions.$fun(x::Number,y::TensorTerm{V},z::TensorTerm{V}) where V = FewSpecialFunctions.$fun(x,Couple(y),Couple(z))
        FewSpecialFunctions.$fun(x::Number,y::TensorTerm{V,0},z::TensorTerm{V,0}) where V = Single{V}(FewSpecialFunctions.$fun(x,Real(y),Real(z)))
        FewSpecialFunctions.$fun(x::Number,y::Couple{V,B},z::Couple{V,B}) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(x,Complex(y),Complex(z)))
        FewSpecialFunctions.$fun(x::Number,y::Couple{V,B},z::TensorTerm{V}) where {V,B} = FewSpecialFunctions.$fun(x,y,Couple(z))
        FewSpecialFunctions.$fun(x::Number,y::TensorTerm{V},z::Couple{V,B}) where {V,B} = FewSpecialFunctions.$fun(x,Couple(y),z)
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::Number,z::TensorTerm{V}) where V = FewSpecialFunctions.$fun(Couple(x),y,Couple(z))
        FewSpecialFunctions.$fun(x::TensorTerm{V,0},y::Number,z::TensorTerm{V,0}) where V = Single{V}(FewSpecialFunctions.$fun(Real(x),y,Real(z)))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::Number,z::Couple{V,B}) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(Complex(x),y,Complex(z)))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::Number,z::TensorTerm{V}) where {V,B} = FewSpecialFunctions.$fun(x,y,Couple(z))
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::Number,z::Couple{V,B}) where {V,B} = FewSpecialFunctions.$fun(Couple(x),y,z)
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::Number) where V = FewSpecialFunctions.$fun(x,Couple(x),Couple(y),z)
        FewSpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0},z::Number) where V = Single{V}(FewSpecialFunctions.$fun(Real(x),Real(y),z))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::Number) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(Complex(x),Complex(y),z))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z::Number) where {V,B} = FewSpecialFunctions.$fun(x,Couple(y),z)
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z::Number) where {V,B} = FewSpecialFunctions.$fun(Couple(x),y,z)
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::TensorTerm{V}) where V = FewSpecialFunctions.$fun(Couple(x),Couple(y),Couple(z))
        FewSpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0},z::TensorTerm{V,0}) where V = Single{V}(FewSpecialFunctions.$fun(Real(x),Real(y),Real(z)))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::Couple{V,B}) where {V,B} = Couple{V,B}(FewSpecialFunctions.$fun(Complex(x),Complex(y),Complex(z)))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::TensorTerm{V}) where {V,B} = FewSpecialFunctions.$fun(x,y,Couple(z))
        FewSpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z::Couple{V,B}) where {V,B} = FewSpecialFunctions.$fun(x,Couple(y),z)
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z::Couple{V,B}) where {V,B} = FewSpecialFunctions.$fun(Couple(x),y,z)
        FewSpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z::TensorTerm{V}) where {V,B} = FewSpecialFunctions.$fun(x,Couple(y),Couple(z))
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z::TensorTerm{V}) where {V,B} = FewSpecialFunctions.$fun(Couple(x),y,Couple(z))
        FewSpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::Couple{V,B}) where {V,B} = FewSpecialFunctions.$fun(Couple(x),Couple(y),z)
    end
end

end # module
