module StaticArraysExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: StaticArrays) : (using StaticArrays)

StaticArrays.SMatrix(m::Chain{V,G,<:Chain{W,G}}) where {V,W,G} = StaticArrays.SMatrix{Grassmann.binomial(mdims(W),G),Grassmann.binomial(mdims(V),G)}(vcat(value.(value(m))...))
Grassmann.Chain(m::StaticArrays.SMatrix{N,N}) where N = Chain{Submanifold(N),1}(m)
Grassmann.Chain{V,G}(m::StaticArrays.SMatrix{N,N}) where {V,G,N} = Chain{V,G}(Chain{V,G}.(getindex.(Ref(m),:,StaticArrays.SVector{N}(1:N))))
Grassmann.Chain{V,G,<:Chain{W,G}}(m::StaticArrays.SMatrix{M,N}) where {V,W,G,M,N} = Chain{V,G}(Chain{W,G}.(getindex.(Ref(m),:,StaticArrays.SVector{N}(1:N))))

end # module
