module SymPyExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: SymPy) : (using SymPy)
import Grassmann: ⟑, wedgedot_metric, ∨, ∧, realvalue, imagvalue, intlog
import Grassmann: getbasis, order, diffvars, diffmode, loworder
import Base: *, adjoint

Grassmann.symfields = (Grassmann.symfields...,SymPy.Sym)
eval(Grassmann.generate_algebra(:SymPy,:Sym,SymPy.Sym,:diff,:symbols))
eval(Grassmann.generate_symbolic_methods(:SymPy,:Sym, (:expand,:factor,:together,:apart,:cancel), (:N,:subs)))
for T ∈ (   Chain{V,G,SymPy.Sym} where {V,G},
            Multivector{V,SymPy.Sym} where V,
            Single{V,G,SymPy.Sym} where {V,G} )
    SymPy.collect(x::T, args...) = map(v -> typeof(v) == SymPy.Sym ? SymPy.collect(v, args...) : v, x)
end

end # module
