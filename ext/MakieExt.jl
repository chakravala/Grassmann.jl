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

Makie.convert_arguments(P::Makie.PointBased, a::Vector{<:Chain}) = Makie.convert_arguments(P, Makie.Point.(a))
Makie.convert_single_argument(a::Chain) = convert_arguments(P,Makie.Point(a))
Makie.arrows(p::Vector{<:Chain{V}},v;args...) where V = Makie.arrows(Makie.Point.(↓(V).(p)),Makie.Point.(value(v));args...)
Makie.arrows!(p::Vector{<:Chain{V}},v;args...) where V = Makie.arrows!(Makie.Point.(↓(V).(p)),Makie.Point.(value(v));args...)
Makie.lines(p::Vector{<:TensorAlgebra};args...) = Makie.lines(Makie.Point.(p);args...)
Makie.lines!(p::Vector{<:TensorAlgebra};args...) = Makie.lines!(Makie.Point.(p);args...)
Makie.lines(p::Vector{<:TensorTerm};args...) = Makie.lines(value.(p);args...)
Makie.lines!(p::Vector{<:TensorTerm};args...) = Makie.lines!(value.(p);args...)
Makie.lines(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines(getindex.(p,1);args...)
Makie.lines!(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines!(getindex.(p,1);args...)

end # module
