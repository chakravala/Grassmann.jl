module SymbolicsExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: Symbolics) : (using Symbolics)
import Grassmann: ⟑, wedgedot_metric, ∨, ∧, realvalue, imagvalue, intlog
import Grassmann: getbasis, order, diffvars, diffmode, loworder
import Base: *, adjoint

eval(Grassmann.generate_algebra(:Symbolics,:Num,Symbolics.Num))
eval(Grassmann.generate_symbolic_methods(:Symbolics,:Num, (:expand,),(:simplify,:substitute)))
Base.:*(a::Symbolics.Num,b::Single{V,G,B,T}) where {V,G,B,T<:Real} = Single{V}(a,b)
Base.:*(a::Single{V,G,B,T},b::Symbolics.Num) where {V,G,B,T<:Real} = Single{V}(b,a)
Base.iszero(a::Single{V,G,B,Symbolics.Num}) where {V,G,B} = false
Grassmann.isfixed(::Type{Symbolics.Num}) = true
for op ∈ (:+,:-)
    for Term ∈ (:TensorGraded,:TensorMixed)
        @eval begin
            Base.$op(a::T,b::Symbolics.Num) where T<:$Term = $op(a,b*One(Manifold(a)))
            Base.$op(a::Symbolics.Num,b::T) where T<:$Term = $op(a*One(Manifold(b)),b)
        end
    end
end

end # module
