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
import Grassmann: getbasis, order, diffvars, diffmode, loworder, FixedVector
import Base: *, adjoint

eval(Grassmann.generate_algebra(:Symbolics,:Num,Symbolics.Num))
eval(Grassmann.generate_symbolic_methods(:Symbolics,:Num, (:expand,),(:simplify,:substitute)))
Grassmann.isfixed(::Type{<:Symbolics.Num}) = true

end # module
