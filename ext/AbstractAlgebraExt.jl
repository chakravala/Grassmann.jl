module AbstractAlgebraExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: AbstractAlgebra) : (using AbstractAlgebra)
import Grassmann: ⟑, wedgedot_metric, ∨, ∧, realvalue, imagvalue, intlog
import Grassmann: getbasis, order, diffvars, diffmode, loworder
import Base: *, adjoint

eval(Grassmann.generate_algebra(:AbstractAlgebra,:SetElem,AbstractAlgebra.SetElem))

end # module
