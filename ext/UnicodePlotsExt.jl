module UnicodePlotsExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: UnicodePlots) : (using UnicodePlots)

Grassmann.vandermonde(x::Chain,y,V,grid) = Grassmann.vandermonde(value(x),y,V,grid)
function Grassmann.vandermonde(x,y,V,grid) # grid=384
    coef,xp,yp = Grassmann.vandermondeinterp(x,y,V,grid)
    p = UnicodePlots.scatterplot(x,value(y)) # overlay points
    display(UnicodePlots.lineplot!(p,xp,yp)) # plot polynomial
    println("||ϵ||: ",norm(Grassmann.approx.(x,Ref(value(coef))).-value(y)))
    return coef # polynomial coefficients
end

end # module
