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

for fun ∈ (:gamma,:loggamma,:logfactorial,:digamma,:invdigamma,:trigamma,:expinti,:expintx,:sinint,:cosint,:erf,:erfc,:erfcinv,:erfcx,:logerfc,:logerfcx,:erfi,:erfinv,:dawson,:faddeeva,:airyai,:airyaiprime,:airybi,:airybiprime,:airyaix,:airyaiprimex,:airybix,:airybiprimex,:besselj0,:besselj1,:bessely0,:bessely1,:jinc,:ellipk,:ellipe,:eta,:zeta)
    @eval begin
        SpecialFunctions.$fun(t::Couple{V,B}) where {V,B} = Couple{V,B}($fun(Complex(t)))
        SpecialFunctions.$fun(t::Chain) = vectorize($fun(complexify(t)))
        SpecialFunctions.$fun(t::Phasor) = $fun(complexify(t))
    end
end
for fun ∈ (:polygamma,:gamma,:loggamma,:besselj,:besseljx,:sphericalbesselj,:bessely,:besselyx,:sphericalbessely,:hankelh1,:hankelh1x,:hankelh2,:hankelh2x,:besseli,:besselix,:besselk,:besselkx)
    @eval begin
        SpecialFunctions.$fun(m,t::Couple{V,B}) where {V,B} = Couple{V,B}($fun(m,Complex(t)))
        SpecialFunctions.$fun(m,t::Chain) = vectorize($fun(m,complexify(t)))
        SpecialFunctions.$fun(m,t::Phasor) = $fun(m,complexify(t))
    end
end
for fun ∈ (:gamma,:loggamma)
    @eval begin
        SpecialFunctions.$fun(a::Couple{V,B},z) where {V,B} = Couple{V,B}($fun(Complex(a),z))
        SpecialFunctions.$fun(a::Chain,z) = vectorize($fun(complexify(a),z))
        SpecialFunctions.$fun(a::Phasor,z) = $fun(complexify(a),z)
    end
end
for fun ∈ (:gamma,:loggamma,:beta,:logbeta,:logabsbeta,:logabsbinomial,:expint,:erf)
    @eval begin
        SpecialFunctions.$fun(x::Couple{V,B},y::Couple{V,B}) where {V,B} = Couple{V,B}($fun(Complex(x),Complex(y)))
        SpecialFunctions.$fun(x::Chain,y::Chain) = vectorize($fun(complexify(x),complexify(y)))
        SpecialFunctions.$fun(x::Phasor,y::Phasor) = $fun(complexify(x),complexify(y))
    end
end
SpecialFunctions.besselh(nu,k,t::Couple{V,B}) where {V,B} = Couple{V,B}(besselh(nu,k,Complex(t)))
SpecialFunctions.besselh(nu,k,t::Chain) = vectorize(besselh(nu,k,complexify(t)))
SpecialFunctions.besselh(nu,k,t::Phasor) = besselh(nu,k,complexify(t))

end # module
