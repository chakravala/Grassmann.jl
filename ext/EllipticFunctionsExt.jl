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
        EllipticFunctions.$fun(t::Couple{V,B}) where {V,B} = Couple{V,B}($fun(Complex(t)))
        EllipticFunctions.$fun(t::Chain) = vectorize($fun(complexify(t)))
        EllipticFunctions.$fun(t::Phasor) = $fun(complexify(t))
    end
end
for fun ∈ (:ellipticE,:ellipticF,:ellipticZ)
    @eval begin
        EllipticFunctions.$fun(m,t::Couple{V,B}) where {V,B} = Couple{V,B}($fun(m,Complex(t)))
        EllipticFunctions.$fun(m,t::Chain) = vectorize($fun(m,complexify(t)))
        EllipticFunctions.$fun(m,t::Phasor) = $fun(m,complexify(t))
    end
end
for fun ∈ (:ljtheta1,:jtheta1,:ljtheta2,:jtheta2,:ljtheta3,:jtheta3,:ljtheta4,:jtheta4,:jtheta1dash,:am)
    @eval begin
        EllipticFunctions.$fun(t::Array{<:Couple{V,B}},q) where {V,B} = Couple{V,B}.($fun(Complex.(t),q))
        EllipticFunctions.$fun(t::Array{<:Chain},q) = vectorize.($fun(complexify.(t),q))
        EllipticFunctions.$fun(t::Array{<:Phasor},q) = $fun(complexify.(t),q)
        EllipticFunctions.$fun(t::Couple{V,B},q) where {V,B} = Couple{V,B}($fun(Complex(t),q))
        EllipticFunctions.$fun(t::Chain,q) = vectorize($fun(complexify(t),q))
        EllipticFunctions.$fun(t::Phasor,q) = $fun(complexify(t),q)
    end
end
for fun ∈ (:agm,:CarlsonRC,:ellipticInvariants,:halfPeriods)
    @eval begin
        EllipticFunctions.$fun(x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}($fun(Complex(x),Complex(y)))
        EllipticFunctions.$fun(x::Chain,y::Chain) = vectorize($fun(complexify(x),complexify(y)))
        EllipticFunctions.$fun(x::Phasor,y::Phasor) = $fun(complexify(x),complexify(y))
    end
end
for fun ∈ (:CarlsonRD,:CarlsonRF,:CarlsonRG)
    @eval begin
        EllipticFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::Couple{V,B}) where {V,B} = Couple{V,B}($fun(Complex(x),Complex(y),Complex(z)))
        EllipticFunctions.$fun(x::Chain,y::Chain,z::Chain) = vectorize($fun(complexify(x),complexify(y),complexify(z)))
        EllipticFunctions.$fun(x::Phasor,y::Phasor,z::Phasor) = $fun(complexify(x),complexify(y),complexify(z))
    end
end
EllipticFunctions.CarlsonRJ(x::Couple{V,B},y::Couple{V,B},z::Couple{V,B},p::Couple{V,B}) where {V,B} = Couple{V,B}(CarlsonRJ(Complex(x),Complex(y),Complex(z),Complex(p)))
EllipticFunctions.CarlsonRJ(x::Chain,y::Chain,z::Chain,p::Chain) = vectorize(CarlsonRJ(complexify(x),complexify(y),complexify(z),complexify(p)))
EllipticFunctions.CarlsonRJ(x::Phasor,y::Phasor,z::Phasor,p::Phasor) = CarlsonRJ(complexify(x),complexify(y),complexify(z),complexify(p))
EllipticFunctions.jtheta_ab(a,b,z::Array{<:Couple{V,B}},q) where {V,B} = Couple{V,B}.(jtheta_ab(a,b,Complex.(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Array{<:Chain},q) = vectorize.(jtheta_ab(a,b,complexify.(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Array{<:Phasor},q) = jtheta_ab(a,b,complexify.(z),q)
EllipticFunctions.jtheta_ab(a,b,z::Couple{V,B},q) where {V,B} = Couple{V,B}(jtheta_ab(a,b,Complex(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Chain,q) = vectorize(jtheta_ab(a,b,complexify(z),q))
EllipticFunctions.jtheta_ab(a,b,z::Phasor,q) = jtheta_ab(a,b,complexify(z),q)
EllipticFunctions.ellipticPI(nu,k,t::Couple{V,B}) where {V,B} = Couple{V,B}(ellipticPI(nu,k,Complex(t)))
EllipticFunctions.ellipticPI(nu,k,t::Chain) = vectorize(ellipticPI(nu,k,complexify(t)))
for fun ∈ (:wp,:wsigma,:wzeta,:thetaC,:thetaD,:thetaN,:thetaS)
    @eval begin
        EllipticFunctions.$fun(t::Array{<:Couple{V,B}};args...) where {V,B} = Couple{V,B}.($fun(Complex.(t);args...))
        EllipticFunctions.$fun(t::Array{<:Chain};args...) = vectorize.($fun(complexify.(t);args...))
        EllipticFunctions.$fun(t::Array{<:Phasor};args...) = $fun(complexify.(t);args...)
        EllipticFunctions.$fun(t::Couple{V,B};args...) where {V,B} = Couple{V,B}($fun(Complex(t);args...))
        EllipticFunctions.$fun(t::Chain;args...) = vectorize($fun(complexify(t);args...))
        EllipticFunctions.$fun(t::Phasor;args...) = $fun(complexify(t);args...)
    end
end
EllipticFunctions.jellip(kind,t::Array{<:Couple{V,B}};args...) where {V,B} = Couple{V,B}.(jellip(kind,Complex.(t);args...))
EllipticFunctions.jellip(kind,t::Array{<:Chain};args...) = vectorize.(jellip(kind,complexify.(t);args...))
EllipticFunctions.jellip(kind,t::Array{<:Phasor};args...) = jellip(kind,complexify.(t);args...)
EllipticFunctions.jellip(kind,t::Couple{V,B};args...) where {V,B} = Couple{V,B}(jellip(kind,Complex(t);args...))
EllipticFunctions.jellip(kind,t::Chain;args...) = vectorize(jellip(kind,complexify(t);args...))
EllipticFunctions.jellip(kind,t::Phasor) = jellip(kind,complexify(t);args...)

end # module
