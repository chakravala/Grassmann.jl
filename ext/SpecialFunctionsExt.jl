module SpecialFunctionsExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: SpecialFunctions) : (using SpecialFunctions)

for fun ∈ (:gamma,:loggamma,:logabsgamma,:loggamma1p,:logfactorial,:digamma,:invdigamma,:trigamma,:expint,:expintx,:sinint,:cosint,:erf,:erfc,:erfcinv,:erfcx,:logerfc,:logerfcx,:erfi,:erfinv,:dawson,:faddeeva,:airyai,:airyaiprime,:airybi,:airybiprime,:airyaix,:airyaiprimex,:airybix,:airybiprimex,:besselj0,:besselj1,:bessely0,:bessely1,:jinc,:ellipk,:ellipe,:eta,:zeta)
    @eval begin
        SpecialFunctions.$fun(x::TensorTerm{V}) where V = SpecialFunctions.$fun(Couple(x))
        SpecialFunctions.$fun(x::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(Real(x)))
        SpecialFunctions.$fun(x::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x)))
        SpecialFunctions.$fun(x::Chain) = vectorize(SpecialFunctions.$fun(complexify(x)))
        SpecialFunctions.$fun(x::Phasor) = SpecialFunctions.$fun(complexify(x))
    end
end
for fun ∈ (:polygamma,:gamma,:gamma_inc,:gamma_inc_asym,:gamma_inc_cf,:gamma_inc_fsum,:loggamma,:beta,:logbeta,:logabsbeta,:logabsbinomial,:expint,:expintx,:erf,:besselj,:besseljx,:sphericalbesselj,:bessely,:besselyx,:sphericalbessely,:hankelh1,:hankelh1x,:hankelh2,:hankelh2x,:besseli,:besselix,:besselk,:besselkx)
    @eval begin
        SpecialFunctions.$fun(x::Number,y::TensorTerm{V}) where V = SpecialFunctions.$fun(x,Couple(y))
        SpecialFunctions.$fun(x::Number,y::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(x,Real(y)))
        SpecialFunctions.$fun(x::Number,y::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(x,Complex(y)))
        SpecialFunctions.$fun(x::Number,y::Chain) = vectorize(SpecialFunctions.$fun(x,complexify(y)))
        SpecialFunctions.$fun(x::Number,y::Phasor) = SpecialFunctions.$fun(x,complexify(y))
        SpecialFunctions.$fun(x::TensorTerm{V},y::Number) where V = SpecialFunctions.$fun(Couple(x),y)
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::Number) where V = Single{V}(SpecialFunctions.$fun(Real(x),y))
        SpecialFunctions.$fun(x::Couple{V,B},y::Number) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),y))
        SpecialFunctions.$fun(x::Chain,y::Number) = vectorize(SpecialFunctions.$fun(complexify(x),y))
        SpecialFunctions.$fun(x::Phasor,y::Number) = SpecialFunctions.$fun(complexify(x),y)
        SpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V}) where V = SpecialFunctions.$fun(Couple(x),Couple(y))
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(Real(x),Real(y)))
        SpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V,0}) where V = SpecialFunctions.$fun(Couple(x),Real(y))
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V}) where V = SpecialFunctions.$fun(Couple(x),Real(y))
        SpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),Complex(y)))
        SpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V,0}) where {V,B} = SpecialFunctions.$fun(x,Real(y))
        SpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V}) where {V,B} = SpecialFunctions.$fun(x,Couple(y))
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::Couple{V,B}) where {V,B} = SpecialFunctions.$fun(Real(x),y)
        SpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B}) where {V,B} = SpecialFunctions.$fun(Couple(x),y)
    end
end
for fun ∈ (:gamma_inc,:gamma_inc_asym,:gamma_inc_cf)
    @eval begin
        SpecialFunctions.$fun(x::TensorTerm{V},y,z) where V = SpecialFunctions.$fun(Couple(x),y,z)
        SpecialFunctions.$fun(x::TensorTerm{V,0},y,z) where V = Single{V}(SpecialFunctions.$fun(Real(x),k,z))
        SpecialFunctions.$fun(x::Couple{V,B},y,z) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),y,z))
        SpecialFunctions.$fun(x::Chain,y,z) = vectorize(SpecialFunctions.$fun(complexify(x),y,z))
        SpecialFunctions.$fun(x::Phasor,y,z) = SpecialFunctions.$fun(complexify(x),y,z)
        SpecialFunctions.$fun(x,y::TensorTerm{V},z) where V = SpecialFunctions.$fun(Couple(x),y,z)
        SpecialFunctions.$fun(x,y::TensorTerm{V,0},z) where V = Single{V}(SpecialFunctions.$fun(Real(x),y,z))
        SpecialFunctions.$fun(x,y::Couple{V,B},z) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),y,z))
        SpecialFunctions.$fun(x,y::Chain,z) = vectorize(SpecialFunctions.$fun(complexify(x),y,z))
        SpecialFunctions.$fun(x,y::Phasor,z) = SpecialFunctions.$fun(complexify(x),y,z)
        SpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z) where V = SpecialFunctions.$fun(Couple(x),Couple(y),z)
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0},z) where V = Single{V}(SpecialFunctions.$fun(Real(x),Real(y),z))
        SpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V,0},z) where V = SpecialFunctions.$fun(Couple(x),Real(y),z)
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V},z) where V = SpecialFunctions.$fun(Real(x),Couple(y),z)
        SpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),Complex(y),z))
        SpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V,0},z) where {V,B} = SpecialFunctions.$fun(x,Real(y),z)
        SpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z) where {V,B} = SpecialFunctions.$fun(x,Couple(y),z)
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::Couple{V,B},z) where {V,B} = SpecialFunctions.$fun(Real(x),y,z)
        SpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z) where {V,B} = SpecialFunctions.$fun(Couple(x),y,z)
    end
end
for fun ∈ (:gamma_inc_inv,:beta_inc,:beta_inc_inv)
    @eval begin
        SpecialFunctions.$fun(x::TensorTerm{V},y,z) where V = SpecialFunctions.$fun(Couple(x),y,z)
        SpecialFunctions.$fun(x::TensorTerm{V,0},y,z) where V = Single{V}(SpecialFunctions.$fun(Real(x),k,z))
        SpecialFunctions.$fun(x::Couple{V,B},y,z) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),y,z))
        SpecialFunctions.$fun(x,y::TensorTerm{V},z) where V = SpecialFunctions.$fun(Couple(x),y,z)
        SpecialFunctions.$fun(x,y::TensorTerm{V,0},z) where V = Single{V}(SpecialFunctions.$fun(Real(x),y,z))
        SpecialFunctions.$fun(x,y::Couple{V,B},z) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),y,z))
        SpecialFunctions.$fun(x,y,z::TensorTerm{V}) where V = SpecialFunctions.$fun(x,k,Couple(y))
        SpecialFunctions.$fun(x,y,z::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(x,k,Real(y)))
        SpecialFunctions.$fun(x,y,z::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(x,k,Complex(y)))
        SpecialFunctions.$fun(x,y::TensorTerm{V},z::TensorTerm{V}) where V = SpecialFunctions.$fun(x,Couple(y),Couple(z))
        SpecialFunctions.$fun(x,y::TensorTerm{V,0},z::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(x,Real(y),Real(z)))
        SpecialFunctions.$fun(x,y::Couple{V,B},z::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(x,Complex(y),Complex(z)))
        SpecialFunctions.$fun(x,y::Couple{V,B},z::TensorTerm{V}) where {V,B} = SpecialFunctions.$fun(x,y,Couple(z))
        SpecialFunctions.$fun(x,y::TensorTerm{V},z::Couple{V,B}) where {V,B} = SpecialFunctions.$fun(x,Couple(y),z)
        SpecialFunctions.$fun(x::TensorTerm{V},y,z::TensorTerm{V}) where V = SpecialFunctions.$fun(Couple(x),y,Couple(z))
        SpecialFunctions.$fun(x::TensorTerm{V,0},y,z::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(Real(x),y,Real(z)))
        SpecialFunctions.$fun(x::Couple{V,B},y,z::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),y,Complex(z)))
        SpecialFunctions.$fun(x::Couple{V,B},y,z::TensorTerm{V}) where {V,B} = SpecialFunctions.$fun(x,y,Couple(z))
        SpecialFunctions.$fun(x::TensorTerm{V},y,z::Couple{V,B}) where {V,B} = SpecialFunctions.$fun(Couple(x),y,z)
        SpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z) where V = SpecialFunctions.$fun(x,Couple(x),Couple(y),z)
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0},z) where V = Single{V}(SpecialFunctions.$fun(Real(x),Real(y),z))
        SpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),Complex(y),z))
        SpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z) where {V,B} = SpecialFunctions.$fun(x,Couple(y),z)
        SpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z) where {V,B} = SpecialFunctions.$fun(Couple(x),y,z)
        SpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::TensorTerm{V}) where V = SpecialFunctions.$fun(Couple(x),Couple(y),Couple(z))
        SpecialFunctions.$fun(x::TensorTerm{V,0},y::TensorTerm{V,0},z::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(Real(x),Real(y),Real(z)))
        SpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(x),Complex(y),Complex(z)))
        SpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B},z::TensorTerm{V}) where {V,B} = SpecialFunctions.$fun(x,y,Couple(z))
        SpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z::Couple{V,B}) where {V,B} = SpecialFunctions.$fun(x,Couple(y),z)
        SpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z::Couple{V,B}) where {V,B} = SpecialFunctions.$fun(Couple(x),y,z)
        SpecialFunctions.$fun(x::Couple{V,B},y::TensorTerm{V},z::TensorTerm{V}) where {V,B} = SpecialFunctions.$fun(x,Couple(y),Couple(z))
        SpecialFunctions.$fun(x::TensorTerm{V},y::Couple{V,B},z::TensorTerm{V}) where {V,B} = SpecialFunctions.$fun(Couple(x),y,Couple(z))
        SpecialFunctions.$fun(x::TensorTerm{V},y::TensorTerm{V},z::Couple{V,B}) where {V,B} = SpecialFunctions.$fun(Couple(x),Couple(y),z)
    end
end
for fun ∈ (:beta_inc,:beta_inc_inv)
    @eval begin
        #SpecialFunctions.$fun(a::TensorTerm{V},b::TensorTerm{V},x::TensorTerm{V},y::TensorTerm{V}) where V = SpecialFunctions.$fun(Couple(a),Couple(b),Couple(x),Couple(y))
        SpecialFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::TensorTerm{V,0},y::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(Real(a),Real(b),Real(x),Real(y)))
        SpecialFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(a),Complex(b),Complex(x),Complex(y)))
        SpecialFunctions.$fun(a::TensorTerm{V},b::TensorTerm{V},x,y) where V = SpecialFunctions.$fun(Couple(a),Couple(b),x,y)
        SpecialFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x,y) where V = Single{V}(SpecialFunctions.$fun(Real(a),Real(b),x,y))
        SpecialFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x,y) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(a),Complex(b),x,y))
        SpecialFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::TensorTerm{V,0},y) where V = Single{V}(SpecialFunctions.$fun(Real(a),Real(b),Real(x),y))
        SpecialFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::TensorTerm{V},y) where V = SpecialFunctions.$fun(Real(a),Real(b),Couple(x),y)
        SpecialFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x::Couple{V},y) where V = SpecialFunctions.$fun(Real(a),Real(b),x,y)
        SpecialFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::Couple{V,B},y) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(a),Complex(b),Complex(x),y))
        SpecialFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::TensorTerm{V},y) where {V,B} = SpecialFunctions.$fun(a,b,Couple(x),y)
        SpecialFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x::TensorTerm{V,0},y) where {V,B} = SpecialFunctions.$fun(a,b,Real(x),y)
        SpecialFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x,y::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.$fun(Real(a),Real(b),x,Real(y)))
        SpecialFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x,y::TensorTerm{V}) where V = Single{V}(SpecialFunctions.$fun(Real(a),Real(b),x,Couple(y)))
        SpecialFunctions.$fun(a::TensorTerm{V,0},b::TensorTerm{V,0},x,y::Couple{V}) where V = Single{V}(SpecialFunctions.$fun(Real(a),Real(b),x,y))
        SpecialFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x,y::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(Complex(a),Complex(b),Complex(x),Complex(y)))
        SpecialFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x,y::TensorTerm{V}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(a,b,x,Couple(y)))
        SpecialFunctions.$fun(a::Couple{V,B},b::Couple{V,B},x,y::TensorTerm{V,0}) where {V,B} = Couple{V,B}(SpecialFunctions.$fun(a,b,x,Real(y)))
    end
end
SpecialFunctions.besselh(x,k,y::TensorTerm{V}) where V = SpecialFunctions.besselh(x,k,Couple(y))
SpecialFunctions.besselh(x,k,y::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.besselh(x,k,Real(y)))
SpecialFunctions.besselh(x,k,y::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.besselh(x,k,Complex(y)))
SpecialFunctions.besselh(x,k,y::Chain) = vectorize(SpecialFunctions.besselh(x,k,complexify(y)))
SpecialFunctions.besselh(x,k,y::Phasor) = SpecialFunctions.besselh(x,k,complexify(y))
SpecialFunctions.besselh(x::TensorTerm{V},k,y) where V = SpecialFunctions.besselh(Couple(x),k,y)
SpecialFunctions.besselh(x::TensorTerm{V,0},k,y) where V = Single{V}(SpecialFunctions.besselh(Real(x),k,y))
SpecialFunctions.besselh(x::Couple{V,B},k,y) where {V,B} = Couple{V,B}(SpecialFunctions.besselh(Complex(x),k,y))
SpecialFunctions.besselh(x::Chain,k,y) = vectorize(SpecialFunctions.besselh(complexify(x),k,y))
SpecialFunctions.besselh(x::Phasor,k,y) = SpecialFunctions.besselh(complexify(x),k,y)
SpecialFunctions.besselh(x::TensorTerm{V},k,y::TensorTerm{V}) where V = SpecialFunctions.besselh(Couple(x),k,Couple(y))
SpecialFunctions.besselh(x::TensorTerm{V,0},k,y::TensorTerm{V,0}) where V = Single{V}(SpecialFunctions.besselh(Real(x),k,Real(y)))
SpecialFunctions.besselh(x::TensorTerm{V},k,y::TensorTerm{V,0}) where V = SpecialFunctions.besselh(Couple(x),k,Real(y))
SpecialFunctions.besselh(x::TensorTerm{V,0},k,y::TensorTerm{V}) where V = SpecialFunctions.besselh(Real(x),k,Couple(y))
SpecialFunctions.besselh(x::Couple{V,B},k,y::Couple{V,B}) where {V,B} = Couple{V,B}(SpecialFunctions.besselh(Complex(x),k,Complex(y)))
SpecialFunctions.besselh(x::Couple{V,B},k,y::TensorTerm{V,0}) where {V,B} = SpecialFunctions.besselh(x,k,Real(y))
SpecialFunctions.besselh(x::Couple{V,B},k,y::TensorTerm{V}) where {V,B} = SpecialFunctions.besselh(x,k,Couple(y))
SpecialFunctions.besselh(x::TensorTerm{V,0},k,y::Couple{V,B}) where {V,B} = SpecialFunctions.besselh(Real(x),k,y)
SpecialFunctions.besselh(x::TensorTerm{V},k,y::Couple{V,B}) where {V,B} = SpecialFunctions.besselh(Couple(x),k,y)

end # module
