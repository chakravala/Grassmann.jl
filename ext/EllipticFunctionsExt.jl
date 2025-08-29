module EllipticFunctionsExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: EllipticFunctions) : (using EllipticFunctions)

for fun ∈ (:qfromtau,:taufromq,:etaDedekind,:lambda,:kleinj,:kleinjinv,:ellipticE,:ellipticK,:EisensteinE2,:EisensteinE4,:EisensteinE6)
    @eval begin
        EllipticFunctions.$fun(x::TensorTerm{V}) where V = EllipticFunctions.$fun(Couple(x))
        EllipticFunctions.$fun(x::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(Real(x)))
        EllipticFunctions.$fun(x::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(x)))
        EllipticFunctions.$fun(x::Chain) = vectorize(EllipticFunctions.$fun(complexify(x)))
        EllipticFunctions.$fun(x::Phasor) = EllipticFunctions.$fun(complexify(x))
    end
end
for fun ∈ (:ellipticE,:ellipticF,:ellipticZ,:ljtheta1,:jtheta1,:ljtheta2,:jtheta2,:ljtheta3,:jtheta3,:ljtheta4,:jtheta4,:jtheta1dash,:am,:agm,:CarlsonRC,:ellipticInvariants,:halfPeriods)
    @eval begin
        EllipticFunctions.$fun(x::Number,y::TensorTerm{V}) where V = EllipticFunctions.$fun(x,Couple(y))
        EllipticFunctions.$fun(x::Number,y::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(x,Real(y)))
        EllipticFunctions.$fun(x::Number,y::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(x,Complex(y)))
        EllipticFunctions.$fun(x::Number,y::Chain) = vectorize(EllipticFunctions.$fun(x,complexify(y)))
        EllipticFunctions.$fun(x::Number,y::Phasor) = EllipticFunctions.$fun(x,complexify(y))
        EllipticFunctions.$fun(x::TensorTerm{V},y::Number) where V = EllipticFunctions.$fun(Couple(x),y)
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::Number) where V = Single{V}(EllipticFunctions.$fun(Real(x),y))
        EllipticFunctions.$fun(x::Couple{V,B},y::Number) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(x),y))
        EllipticFunctions.$fun(x::Chain,y::Number) = vectorize(EllipticFunctions.$fun(complexify(x),y))
        EllipticFunctions.$fun(x::Phasor,y::Number) = EllipticFunctions.$fun(complexify(x),y)
        EllipticFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V}) where V = EllipticFunctions.$fun(Couple(x),Couple(y))
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(Real(x),Real(y)))
        EllipticFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V,0}) where V = EllipticFunctions.$fun(Couple(x),Real(y))
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V}) where V = EllipticFunctions.$fun(Real(x),Couple(y))
        EllipticFunctions.$fun(x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(x),Complex(y)))
        EllipticFunctions.$fun(x::Couple{V,B},y::TensorTerm{V,0}) where {V,B} = EllipticFunctions.$fun(x,Real(y))
        EllipticFunctions.$fun(x::Couple{V,B},y::TensorTerm{V}) where {V,B} = EllipticFunctions.$fun(x,Couple(y))
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(Real(x),y)
        EllipticFunctions.$fun(x::TensorTerm{V},y::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(Couple(x),y)
    end
end
for fun ∈ (:ljtheta1,:jtheta1,:ljtheta2,:jtheta2,:ljtheta3,:jtheta3,:ljtheta4,:jtheta4,:jtheta1dash,:am)
    @eval begin
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V}},y::Number) where V = EllipticFunctions.$fun(Couple.(x),y)
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V,0}},y::Number) where V = Single{V}.(EllipticFunctions.$fun(Real.(x),y))
        EllipticFunctions.$fun(x::Array{<:Couple{V,B}},y::Number) where {V,B} = Couple{V,B}.(EllipticFunctions.$fun(Complex.(x),y))
        EllipticFunctions.$fun(x::Array{<:Chain},y::Number) = vectorize.(EllipticFunctions.$fun(complexify.(x),y))
        EllipticFunctions.$fun(x::Array{<:Phasor},y::Number) = EllipticFunctions.$fun(complexify.(x),y)
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V}},y::TensorTerm{V}) where V = EllipticFunctions.$fun(Couple.(x),Couple(y))
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V,0}},y::TensorTerm{V,0}) where V = Single{V}.(EllipticFunctions.$fun(Real.(x),Real(y)))
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V}},y::TensorTerm{V,0}) where V = EllipticFunctions.$fun(Couple.(x),Real(y))
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V,0}},y::TensorTerm{V}) where V = EllipticFunctions.$fun(Real.(x),Couple(y))
        EllipticFunctions.$fun(x::Array{<:Couple{V,B}},y::Couple{V,B}) where {V,B} = Couple{V,B}.(EllipticFunctions.$fun(Complex.(x),Complex(y)))
        EllipticFunctions.$fun(x::Array{<:Couple{V,B}},y::TensorTerm{V,0}) where {V,B} = EllipticFunctions.$fun(x,Real(y))
        EllipticFunctions.$fun(x::Array{<:Couple{V,B}},y::TensorTerm{V}) where {V,B} = EllipticFunctions.$fun(x,Couple(y))
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V,0}},y::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(Real.(x),y)
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V}},y::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(Couple.(x),y)
    end
end
for fun ∈ (:CarlsonRD,:CarlsonRF,:CarlsonRG,:ellipticPI)
    @eval begin
        EllipticFunctions.$fun(x::Number,y::Number,z::TensorTerm{V}) where V = EllipticFunctions.$fun(x,y,Couple(z))
        EllipticFunctions.$fun(x::Number,y::Number,z::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(x,y,Real(z)))
        EllipticFunctions.$fun(x::Number,y::Number,z::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(x,y,Complex(z)))
        EllipticFunctions.$fun(x::Number,y::TensorTerm{V},z::Number) where V = EllipticFunctions.$fun(x,Couple(y),z)
        EllipticFunctions.$fun(x::Number,y::TensorTerm{V,0},z::Number) where V = Single{V}(EllipticFunctions.$fun(x,Real(y),z))
        EllipticFunctions.$fun(x::Number,y::Couple{V,B},z::Number) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(x,Complex(y),z))
        EllipticFunctions.$fun(x::TensorTerm{V},y::Number,z::Number) where V = EllipticFunctions.$fun(Couple(x),y,z)
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::Number,z::Number) where V = Single{V}(EllipticFunctions.$fun(Real(x),y,z))
        EllipticFunctions.$fun(x::Couple{V,B},y::Number,z::Number) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(x),y,z))
        EllipticFunctions.$fun(x::Number,y::TensorTerm{V},z::TensorTerm{V}) where V = EllipticFunctions.$fun(x,Couple(y),Couple(z))
        EllipticFunctions.$fun(x::Number,y::TensorTerm{V,0},z::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(x,Real(y),Real(z)))
        EllipticFunctions.$fun(x::Number,y::Couple{V,B},z::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(x,Complex(y),Complex(z)))
        EllipticFunctions.$fun(x::Number,y::Couple{V,B},z::TensorTerm{V}) where {V,B} = EllipticFunctions.$fun(x,y,Couple(z))
        EllipticFunctions.$fun(x::Number,y::TensorTerm{V},z::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(x,Couple(y),z)
        EllipticFunctions.$fun(x::TensorTerm{V},y::Number,z::TensorTerm{V}) where V = EllipticFunctions.$fun(Couple(x),y,Couple(z))
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::Number,z::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(Real(x),y,Real(z)))
        EllipticFunctions.$fun(x::Couple{V,B},y::Number,z::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(x),y,Complex(z)))
        EllipticFunctions.$fun(x::Couple{V,B},y::Number,z::TensorTerm{V}) where {V,B} = EllipticFunctions.$fun(x,y,Couple(z))
        EllipticFunctions.$fun(x::TensorTerm{V},y::Number,z::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(Couple(x),y,z)
        EllipticFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::Number) where V = EllipticFunctions.$fun(x,Couple(x),Couple(y),z)
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0},z::Number) where V = Single{V}(EllipticFunctions.$fun(Real(x),Real(y),z))
        EllipticFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::Number) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(x),Complex(y),z))
        EllipticFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z::Number) where {V,B} = EllipticFunctions.$fun(x,Couple(y),z)
        EllipticFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z::Number) where {V,B} = EllipticFunctions.$fun(Couple(x),y,z)
        EllipticFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::TensorTerm{V}) where V = EllipticFunctions.$fun(Couple(x),Couple(y),Couple(z))
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0},z::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(Real(x),Real(y),Real(z)))
        EllipticFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(x),Complex(y),Complex(z)))
        EllipticFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::TensorTerm{V}) where {V,B} = EllipticFunctions.$fun(x,y,Couple(z))
        EllipticFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(x,Couple(y),z)
        EllipticFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(Couple(x),y,z)
        EllipticFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z::TensorTerm{V}) where {V,B} = EllipticFunctions.$fun(x,Couple(y),Couple(z))
        EllipticFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z::TensorTerm{V}) where {V,B} = EllipticFunctions.$fun(Couple(x),y,Couple(z))
        EllipticFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(Couple(x),Couple(y),z)
    end
end
for fun ∈ (:CarlsonRJ,:jtheta_ab)
    @eval begin
        #EllipticFunctions.$fun(a::TensorTerm{V},b::TensorTerm{V},x::TensorTerm{V},y::TensorTerm{V}) where V = EllipticFunctions.$fun(Couple(a),Couple(b),Couple(x),Couple(y))
        EllipticFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::TensorTerm{V,0},y::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(Real(a),Real(b),Real(x),Real(y)))
        EllipticFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(a),Complex(b),Complex(x),Complex(y)))
        EllipticFunctions.$fun(a::TensorTerm{V},b::TensorTerm{V},x::Number,y::Number) where V = EllipticFunctions.$fun(Couple(a),Couple(b),x,y)
        EllipticFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::Number,y::Number) where V = Single{V}(EllipticFunctions.$fun(Real(a),Real(b),x,y))
        EllipticFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::Number,y::Number) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(a),Complex(b),x,y))
        EllipticFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::TensorTerm{V,0},y::Number) where V = Single{V}(EllipticFunctions.$fun(Real(a),Real(b),Real(x),y))
        EllipticFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::TensorTerm{V},y::Number) where V = EllipticFunctions.$fun(Real(a),Real(b),Couple(x),y)
        EllipticFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::Couple{V},y::Number) where V = EllipticFunctions.$fun(Real(a),Real(b),x,y)
        EllipticFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::Couple{V,B},y::Number) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(a),Complex(b),Complex(x),y))
        EllipticFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::TensorTerm{V},y::Number) where {V,B} = EllipticFunctions.$fun(a,b,Couple(x),y)
        EllipticFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::TensorTerm{V,0},y::Number) where {V,B} = EllipticFunctions.$fun(a,b,Real(x),y)
        EllipticFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::Number,y::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(Real(a),Real(b),x,Real(y)))
        EllipticFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::Number,y::TensorTerm{V}) where V = Single{V}(EllipticFunctions.$fun(Real(a),Real(b),x,Couple(y)))
        EllipticFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::Number,y::Couple{V}) where V = Single{V}(EllipticFunctions.$fun(Real(a),Real(b),x,y))
        EllipticFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::Number,y::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(a),Complex(b),Complex(x),Complex(y)))
        EllipticFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::Number,y::TensorTerm{V}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(a,b,x,Couple(y)))
        EllipticFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::Number,y::TensorTerm{V,0}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(a,b,x,Real(y)))
    end
    for (A,B) ∈ ((:Number,:Number),(:(Couple{V}),:(Couple{V})),(:(TensorTerm{V}),:(TensorTerm{V})))
        @eval begin
            EllipticFunctions.$fun(a::$A,b::$B,x::TensorTerm{V},y::Number) where V = EllipticFunctions.$fun(a,b,Couple(x),y)
            EllipticFunctions.$fun(a::$A,b::$B,x::TensorTerm{V,0},y::Number) where V = Single{V}(EllipticFunctions.$fun(a,b,Real(x),y))
            EllipticFunctions.$fun(a::$A,b::$B,x::Couple{V,B},y::Number) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(a,b,Complex(x),y))
            EllipticFunctions.$fun(a::$A,b::$B,x::Number,y::TensorTerm{V}) where V = EllipticFunctions.$fun(a,b,x,Couple(y))
            EllipticFunctions.$fun(a::$A,b::$B,x::Number,y::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(a,b,x,Real(y)))
            EllipticFunctions.$fun(a::$A,b::$B,x::Number,y::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(a,b,x,Complex(y)))
            EllipticFunctions.$fun(a::$A,b::$B,x::TensorTerm{V},y::TensorTerm{V}) where V = EllipticFunctions.$fun(a,b,Couple(x),Couple(y))
            EllipticFunctions.$fun(a::$A,b::$B,x::TensorTerm{V,0},y::TensorTerm{V,0}) where V = Single{V}(EllipticFunctions.$fun(a,b,Real(x),Real(y)))
            EllipticFunctions.$fun(a::$A,b::$B,x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(a,b,Complex(x),Complex(y)))
            EllipticFunctions.$fun(a::$A,b::$B,x::Couple{V},y::TensorTerm{V}) where V = EllipticFunctions.$fun(a,b,x,Couple(y))
            EllipticFunctions.$fun(a::$A,b::$B,x::TensorTerm{V},y::Couple{V}) where V = EllipticFunctions.$fun(a,b,Couple(x),y)
        end
    end
end
for fun ∈ (:jtheta_ab,)
    @eval begin
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:TensorTerm{V}},y::Number) where V = EllipticFunctions.$fun(a,b,Couple.(x),y)
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:TensorTerm{V,0}},y::Number) where V = Single{V}.(EllipticFunctions.$fun(a,b,Real.(x),y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:Couple{V,B}},y::Number) where {V,B} = Couple{V,B}.(EllipticFunctions.$fun(a,b,Complex.(x),y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:Chain},y::Number) = vectorize.(EllipticFunctions.$fun(a,b,complexify.(x),y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:Phasor},y::Number) = EllipticFunctions.$fun(a,b,complexify.(x),y)
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:TensorTerm{V}},y::TensorTerm{V}) where V = EllipticFunctions.$fun(a,b,Couple.(x),Couple(y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:TensorTerm{V,0}},y::TensorTerm{V,0}) where V = Single{V}.(EllipticFunctions.$fun(a,b,Real.(x),Real(y)))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:TensorTerm{V}},y::TensorTerm{V,0}) where V = EllipticFunctions.$fun(a,b,Couple.(x),Real(y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:TensorTerm{V,0}},y::TensorTerm{V}) where V = EllipticFunctions.$fun(a,b,Real.(x),Couple(y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:Couple{V,B}},y::Couple{V,B}) where {V,B} = Couple{V,B}.(EllipticFunctions.$fun(a,b,Complex.(x),Complex(y)))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:Couple{V,B}},y::TensorTerm{V,0}) where {V,B} = EllipticFunctions.$fun(a,b,x,Real(y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:Couple{V,B}},y::TensorTerm{V}) where {V,B} = EllipticFunctions.$fun(a,b,x,Couple(y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:TensorTerm{V,0}},y::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(a,b,Real.(x),y)
        EllipticFunctions.$fun(a::Number,b::Number,x::Array{<:TensorTerm{V}},y::Couple{V,B}) where {V,B} = EllipticFunctions.$fun(a,b,Couple.(x),y)
    end
end
for fun ∈ (:wp,:wsigma,:wzeta,:thetaC,:thetaD,:thetaN,:thetaS)
    @eval begin
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V}};args...) where V = EllipticFunctions.$fun(Couple.(x);args...)
        EllipticFunctions.$fun(x::Array{<:TensorTerm{V,0}};args...) where V = Single{V}.(EllipticFunctions.$fun(Real.(x);args...))
        EllipticFunctions.$fun(x::Array{<:Couple{V,B}};args...) where {V,B} = Couple{V,B}.(EllipticFunctions.$fun(Complex.(x);args...))
        EllipticFunctions.$fun(x::Array{<:Chain};args...) = vectorize.(EllipticFunctions.$fun(complexify.(x);args...))
        EllipticFunctions.$fun(x::Array{<:Phasor};args...) = EllipticFunctions.$fun(complexify.(x);args...)
        EllipticFunctions.$fun(x::TensorTerm{V};args...) where V = EllipticFunctions.$fun(Couple(x);args...)
        EllipticFunctions.$fun(x::TensorTerm{V,0};args...) where V = Single{V}(EllipticFunctions.$fun(Real(x);args...))
        EllipticFunctions.$fun(x::Couple{V,B};args...) where {V,B} = Couple{V,B}(EllipticFunctions.$fun(Complex(x);args...))
        EllipticFunctions.$fun(x::Chain;args...) = vectorize(EllipticFunctions.$fun(complexify(x);args...))
        EllipticFunctions.$fun(x::Phasor;args...) = EllipticFunctions.$fun(complexify(x);args...)
    end
end
EllipticFunctions.jellip(kind::String,x::Array{<:TensorTerm{V}};args...) where V = EllipticFunctions.jellip(kind,Couple.(x);args...)
EllipticFunctions.jellip(kind::String,x::Array{<:TensorTerm{V,0}};args...) where V = Single{V}.(kind,jellip(Real.(x);args...))
EllipticFunctions.jellip(kind::String,x::Array{<:Couple{V,B}};args...) where {V,B} = Couple{V,B}.(EllipticFunctions.jellip(kind,Complex.(x);args...))
EllipticFunctions.jellip(kind::String,x::Array{<:Chain};args...) = vectorize.(EllipticFunctions.jellip(kind,complexify.(x);args...))
EllipticFunctions.jellip(kind::String,x::Array{<:Phasor};args...) = EllipticFunctions.jellip(kind,complexify.(x);args...)
EllipticFunctions.jellip(kind::String,x::TensorTerm{V};args...) where V = EllipticFunctions.jellip(kind,Couple(x);args...)
EllipticFunctions.jellip(kind::String,x::TensorTerm{V,0};args...) where V = Single{V}(EllipticFunctions.jellip(kind,Real(x);args...))
EllipticFunctions.jellip(kind::String,x::Couple{V,B};args...) where {V,B} = Couple{V,B}(EllipticFunctions.jellip(kind,Complex(x);args...))
EllipticFunctions.jellip(kind::String,x::Chain;args...) = vectorize(EllipticFunctions.jellip(kind,complexify(x);args...))
EllipticFunctions.jellip(kind::String,x::Phasor;args...) = EllipticFunctions.jellip(kind,complexify(x);args...)

end # module
