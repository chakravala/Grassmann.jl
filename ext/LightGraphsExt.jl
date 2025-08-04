module LightGraphsExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: LightGraphs) : (using LightGraphs)

function LightGraphs.SimpleDiGraph(x::T,g=LightGraphs.SimpleDiGraph(Grassmann.rank(V))) where T<:TensorTerm{V} where V
   ind = (Grassmann.signbit(value(x)) ? reverse : identity)(indices(basis(x)))
   Grassmann.rank(x) == 2 ? LightGraphs.add_edge!(g,ind...) : LightGraphs.SimpleDiGraph(∂(x),g)
   return g
end
function LightGraphs.SimpleDiGraph(x::Chain{V},g=LightGraphs.SimpleDiGraph(Grassmann.rank(V))) where V
    N,G = mdims(V),Grassmann.rank(x)
    ib = Grassmann.indexbasis(N,G)
    for k ∈ 1:Grassmann.binomial(N,G)
        if !iszero(x.v[k])
            B = Grassmann.symmetricmask(V,ib[k],ib[k])[1]
            count_ones(B) ≠1 && LightGraphs.SimpleDiGraph(x.v[k]*Grassmann.getbasis(V,B),g)
        end
    end
    return g
end
function LightGraphs.SimpleDiGraph(x::Multivector{V},g=LightGraphs.SimpleDiGraph(Grassmann.rank(V))) where V
   N = mdims(V)
   for i ∈ 2:N
        R = Grassmann.binomsum(N,i)
        ib = Grassmann.indexbasis(N,i)
        for k ∈ 1:Grassmann.binomial(N,i)
            if !iszero(x.v[k+R])
                B = Grassmann.symmetricmask(V,ib[k],ib[k])[1]
                count_ones(B) ≠ 1 && LightGraphs.SimpleDiGraph(x.v[k+R]*Grassmann.getbasis(V,B),g)
            end
        end
    end
    return g
end

end # module
