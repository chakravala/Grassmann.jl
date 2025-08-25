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
        EllipticFunctions.$fun(t::TensorTerm{V}) where V = $fun(Couple(t))
        EllipticFunctions.$fun(t::TensorTerm{V,0}) where V = Single{V}($fun(Real(t)))
        EllipticFunctions.$fun(t::Couple{V,B}) where {V,B} = Couple{V,B}($fun(Complex(t)))
        EllipticFunctions.$fun(t::Chain) = vectorize($fun(complexify(t)))
        EllipticFunctions.$fun(t::Phasor) = $fun(complexify(t))
    end
end
for fun ∈ (:ellipticE,:ellipticF,:ellipticZ)
    @eval begin
        EllipticFunctions.$fun(m,t::TensorTerm{V}) where V = $fun(m,Couple(t))
        EllipticFunctions.$fun(m,t::TensorTerm{V,0}) where V = Single{V}($fun(m,Real(t)))
        EllipticFunctions.$fun(m,t::Couple{V,B}) where {V,B} = Couple{V,B}($fun(m,Complex(t)))
        EllipticFunctions.$fun(m,t::Chain) = vectorize($fun(m,complexify(t)))
        EllipticFunctions.$fun(m,t::Phasor) = $fun(m,complexify(t))
    end
end
for fun ∈ (:ljtheta1,:jtheta1,:ljtheta2,:jtheta2,:ljtheta3,:jtheta3,:ljtheta4,:jtheta4,:jtheta1dash,:am)
    @eval begin
        EllipticFunctions.$fun(t::Array{<:TensorTerm{V}},q) where V = $fun(Couple.(t),q)
        EllipticFunctions.$fun(t::Array{<:TensorTerm{V,0}},q) where V = Single{V}.($fun(Real.(t),q))
        EllipticFunctions.$fun(t::Array{<:Couple{V,B}},q) where {V,B} = Couple{V,B}.($fun(Complex.(t),q))
        EllipticFunctions.$fun(t::Array{<:Chain},q) = vectorize.($fun(complexify.(t),q))
        EllipticFunctions.$fun(t::Array{<:Phasor},q) = $fun(complexify.(t),q)
        EllipticFunctions.$fun(t::TensorTerm{V},q) where V = $fun(Couple(t),q)
        EllipticFunctions.$fun(t::TensorTerm{V,0},q) where V = Single{V}($fun(Real(t),q))
        EllipticFunctions.$fun(t::Couple{V,B},q) where {V,B} = Couple{V,B}($fun(Complex(t),q))
        EllipticFunctions.$fun(t::Chain,q) = vectorize($fun(complexify(t),q))
        EllipticFunctions.$fun(t::Phasor,q) = $fun(complexify(t),q)
    end
end
for fun ∈ (:agm,:CarlsonRC,:ellipticInvariants,:halfPeriods)
    @eval begin
        EllipticFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V}) where V = $fun(Couple(x),Couple(y))
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0}) where V = Single{V}($fun(Real(t),Real(t)))
        EllipticFunctions.$fun(x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}($fun(Complex(x),Complex(y)))
        EllipticFunctions.$fun(x::Chain,y::Chain) = vectorize($fun(complexify(x),complexify(y)))
        EllipticFunctions.$fun(x::Phasor,y::Phasor) = $fun(complexify(x),complexify(y))
    end
end
for fun ∈ (:CarlsonRD,:CarlsonRF,:CarlsonRG)
    @eval begin
        EllipticFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::TensorTerm{V}) where V = $fun(Couple(x),Couple(y),Couple(z))
        EllipticFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0},z::TensorTerm{V,0}) where V = Single{V}($fun(Real(x),Real(y),Real(z)))
        EllipticFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::Couple{V,B}) where {V,B} = Couple{V,B}($fun(Complex(x),Complex(y),Complex(z)))
        EllipticFunctions.$fun(x::Chain,y::Chain,z::Chain) = vectorize($fun(complexify(x),complexify(y),complexify(z)))
        EllipticFunctions.$fun(x::Phasor,y::Phasor,z::Phasor) = $fun(complexify(x),complexify(y),complexify(z))
    end
end
EllipticFunctions.CarlsonRJ(x::TensorTerm{V},y::TensorTerm{V},z::TensorTerm{V},p::TensorTerm{V}) where V = CarlsonRJ(Couple(x),Couple(y),Couple(z),Couple(p))
EllipticFunctions.CarlsonRJ(x::TensorTerm{V,0},y::TensorTerm{V,0},z::TensorTerm{V,0},p::TensorTerm{V,0}) where V = Single{V}(CarlsonRJ(Real(x),Real(y),Real(z),Real(p)))
EllipticFunctions.CarlsonRJ(x::Couple{V,B},y::Couple{V,B},z::Couple{V,B},p::Couple{V,B}) where {V,B} = Couple{V,B}(CarlsonRJ(Complex(x),Complex(y),Complex(z),Complex(p)))
EllipticFunctions.CarlsonRJ(x::Chain,y::Chain,z::Chain,p::Chain) = vectorize(CarlsonRJ(complexify(x),complexify(y),complexify(z),complexify(p)))
EllipticFunctions.CarlsonRJ(x::Phasor,y::Phasor,z::Phasor,p::Phasor) = CarlsonRJ(complexify(x),complexify(y),complexify(z),complexify(p))
EllipticFunctions.jtheta_ab(a,b,z::Array{<:TensorTerm{V}},q) where V = jtheta_ab(a,b,Couple.(z),q)
EllipticFunctions.jtheta_ab(a,b,z::Array{<:TensorTerm{V,0}},q) where V = Single{V}.(jtheta_ab(a,b,Real.(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Array{<:Couple{V,B}},q) where {V,B} = Couple{V,B}.(jtheta_ab(a,b,Complex.(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Array{<:Chain},q) = vectorize.(jtheta_ab(a,b,complexify.(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Array{<:Phasor},q) = jtheta_ab(a,b,complexify.(z),q)
EllipticFunctions.jtheta_ab(a,b,z::TensorTerm{V},q) where V = jtheta_ab(a,b,Couple(z),q)
EllipticFunctions.jtheta_ab(a,b,z::TensorTerm{V,0},q) where V = Single{V}(jtheta_ab(a,b,Real(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Couple{V,B},q) where {V,B} = Couple{V,B}(jtheta_ab(a,b,Complex(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Chain,q) = vectorize(jtheta_ab(a,b,complexify(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Phasor,q) = jtheta_ab(a,b,complexify(z),q)
EllipticFunctions.ellipticPI(nu,k,t::TensorTerm{V}) where V = ellipticPI(nu,k,Couple(t))
EllipticFunctions.ellipticPI(nu,k,t::TensorTerm{V,0}) where V = Single{V}(nu,k,ellipticPI(Real(t)))
EllipticFunctions.ellipticPI(nu,k,t::Couple{V,B}) where {V,B} = Couple{V,B}(ellipticPI(nu,k,Complex(t)))
EllipticFunctions.ellipticPI(nu,k,t::Chain) = vectorize(ellipticPI(nu,k,complexify(t)))
EllipticFunctions.ellipticPI(nu,k,t::Phasor) = ellipticPI(nu,k,complexify(t))
for fun ∈ (:wp,:wsigma,:wzeta,:thetaC,:thetaD,:thetaN,:thetaS)
    @eval begin
        EllipticFunctions.$fun(t::Array{<:TensorTerm{V}};args...) where V = $fun(Couple.(t);args...)
        EllipticFunctions.$fun(t::Array{<:TensorTerm{V,0}};args...) where V = Single{V}.($fun(Real.(t);args...))
        EllipticFunctions.$fun(t::Array{<:Couple{V,B}};args...) where {V,B} = Couple{V,B}.($fun(Complex.(t);args...))
        EllipticFunctions.$fun(t::Array{<:Chain};args...) = vectorize.($fun(complexify.(t);args...))
        EllipticFunctions.$fun(t::Array{<:Phasor};args...) = $fun(complexify.(t);args...)
        EllipticFunctions.$fun(t::TensorTerm{V};args...) where V = $fun(Couple(t);args...)
        EllipticFunctions.$fun(t::TensorTerm{V,0};args...) where V = Single{V}($fun(Real(t);args...))
        EllipticFunctions.$fun(t::Couple{V,B};args...) where {V,B} = Couple{V,B}($fun(Complex(t);args...))
        EllipticFunctions.$fun(t::Chain;args...) = vectorize($fun(complexify(t);args...))
        EllipticFunctions.$fun(t::Phasor;args...) = $fun(complexify(t);args...)
    end
end
EllipticFunctions.jellip(kind,t::Array{<:TensorTerm{V}};args...) where V = jellip(kind,Couple.(t);args...)
EllipticFunctions.jellip(kind,t::Array{<:TensorTerm{V,0}};args...) where V = Single{V}.(kind,jellip(Real.(t);args...))
EllipticFunctions.jellip(kind,t::Array{<:Couple{V,B}};args...) where {V,B} = Couple{V,B}.(jellip(kind,Complex.(t);args...))
EllipticFunctions.jellip(kind,t::Array{<:Chain};args...) = vectorize.(jellip(kind,complexify.(t);args...))
EllipticFunctions.jellip(kind,t::Array{<:Phasor};args...) = jellip(kind,complexify.(t);args...)
EllipticFunctions.jellip(kind,t::TensorTerm{V};args...) where V = jellip(kind,Couple(t);args...)
EllipticFunctions.jellip(kind,t::TensorTerm{V,0};args...) where V = Single{V}(jellip(kind,Real(t);args...))
EllipticFunctions.jellip(kind,t::Couple{V,B};args...) where {V,B} = Couple{V,B}(jellip(kind,Complex(t);args...))
EllipticFunctions.jellip(kind,t::Chain;args...) = vectorize(jellip(kind,complexify(t);args...))
EllipticFunctions.jellip(kind,t::Phasor) = jellip(kind,complexify(t);args...)

end # module
