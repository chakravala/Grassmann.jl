module MeshesExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: Meshes) : (using Meshes)

Meshes.Point(t::Values) = Meshes.Point(Tuple(t.v))
Meshes.Point(t::Variables) = Meshes.Point(Tuple(t.v))
Base.convert(::Type{Meshes.Point},t::T) where T<:TensorTerm{V} where V = Meshes.Point(value(Chain{V,valuetype(t)}(vector(t))))
Base.convert(::Type{Meshes.Point},t::T) where T<:TensorTerm{V,0} where V = Meshes.Point(zeros(valuetype(t),mdims(V))...)
Base.convert(::Type{Meshes.Point},t::T) where T<:TensorAlgebra = Meshes.Point(value(vector(t)))
Base.convert(::Type{Meshes.Point},t::Chain{V,G,T}) where {V,G,T} = G == 1 ? Meshes.Point(value(vector(t))) : Meshes.Point(zeros(T,mdims(V))...)
Meshes.Point(t::T) where T<:TensorAlgebra = convert(Meshes.Point,t)
Grassmann.pointpair(p,V) = Pair(Meshes.Point.(V.(value(p)))...)
@pure Grassmann.ptype(::Meshes.Point{N,T} where N) where T = T
export pointfield
Grassmann.pointfield(t,V=Manifold(t),W=V) = p->Meshes.Point(V(vector(↓(↑((V∪Manifold(t))(Chain{W,1,Grassmann.ptype(p)}(p.data)))⊘t))))
function Grassmann.pointfield(t,ϕ::T) where T<:AbstractVector
    M = Manifold(t)
    V = Manifold(M)
    z = mdims(V) ≠ 4 ? Meshes(0.0,0.0) : Meshes.Point(0.0,0.0,0.0)
    p->begin
        P = Chain{V,1}(one(Grassmann.ptype(p)),p.data...)
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return Meshes.Point((Pi\P)⋅Chain{V,1}(ϕ[ti])))
        end
        return z
    end
end

end # module
