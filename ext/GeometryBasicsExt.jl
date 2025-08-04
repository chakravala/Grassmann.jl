module GeometryBasicsExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: GeometryBasics) : (using GeometryBasics)

GeometryBasics.Point(t::Values) = GeometryBasics.Point(Tuple(t.v))
GeometryBasics.Point(t::Grassmann.Variables) = GeometryBasics.Point(Tuple(t.v))
Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorTerm{V} where V = convert(GeometryBasis.Point,Chain(t))
Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorTerm{V,0} where V = GeometryBasics.Point(zeros(valuetype(t),mdims(V))...)
Base.convert(::Type{GeometryBasics.Point},t::T) where T<:TensorAlgebra = GeometryBasics.Point(value(vector(t)))
Base.convert(::Type{GeometryBasics.Point},t::T) where T<:Couple = GeometryBasics.Point(t.v.re,t.v.im)
Base.convert(::Type{GeometryBasics.Point},t::T) where T<:Phasor = GeometryBasics.Point(t.v.re,t.v.im)
Base.convert(::Type{GeometryBasics.Point},t::Chain{V,G,T}) where {V,G,T} = GeometryBasics.Point(value(t))
GeometryBasics.Point(t::T) where T<:TensorAlgebra = convert(GeometryBasics.Point,t)
Grassmann.pointpair(p,V) = Pair(GeometryBasics.Point.(V.(value(p)))...)
Base.@pure Grassmann.ptype(::GeometryBasics.Point{N,T} where N) where T = T
Grassmann.pointfield(t,V=Manifold(t),W=V) = p->GeometryBasics.Point(V(vector(↓(↑((V∪Manifold(t))(Chain{W,1,Grassmann.ptype(p)}(p.data)))⊘t))))
function Grassmann.pointfield(t,ϕ::T) where T<:AbstractVector
    M = Manifold(t)
    V = Manifold(M)
    z = mdims(V) ≠ 4 ? GeometryBasics(0.0,0.0) : GeometryBasics.Point(0.0,0.0,0.0)
    p->begin
        P = Chain{V,1}(one(Grassmann.ptype(p)),p.data...)
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return GeometryBasics.Point((Pi\P)⋅Chain{V,1}(ϕ[ti])))
        end
        return z
    end
end

end # module
