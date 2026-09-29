module MakieExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: Makie) : (using Makie)

Makie.convert_arguments(P::Makie.PointBased, a::AbstractArray{<:Chain}) = Makie.convert_arguments(P, Makie.Point.(a))
Makie.convert_single_argument(a::Chain) = convert_arguments(P,Makie.Point(a))
Makie.convert_single_argument(a::TensorTerm) = convert_arguments(P,value(a))
Makie.convert_single_argument(a::Chain{V,G,T,1} where {V,G,T}) = convert_arguments(P,a[1])

Makie.arrows(p::Vector{<:Chain{V}},v;args...) where V = Makie.arrows(Makie.Point.(↓(V).(p)),Makie.Point.(value(v));args...)
Makie.arrows!(p::Vector{<:Chain{V}},v;args...) where V = Makie.arrows!(Makie.Point.(↓(V).(p)),Makie.Point.(value(v));args...)

Makie.convert_arguments(P::Type{<:Makie.Lines}, p::AbstractVector{<:TensorAlgebra}) = (Makie.Point.(p),)
Makie.convert_arguments(P::Type{<:Makie.Lines}, p::AbstractVector{<:TensorTerm}) = (value.(p),)
Makie.convert_arguments(P::Type{<:Makie.Lines}, p::AbstractVector{<:Chain{V,G,T,1} where {V,G,T}}) = (getindex.(p,1),)

end # module
