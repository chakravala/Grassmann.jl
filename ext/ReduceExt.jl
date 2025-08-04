module ReduceExt

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
isdefined(Grassmann, :Requires) ? (import Grassmann: Reduce) : (using Reduce)

Base.:*(a::Reduce.RExpr,b::Submanifold{V}) where V = Single{V}(a,b)
Base.:*(a::Submanifold{V},b::Reduce.RExpr) where V = Single{V}(b,a)
Base.:*(a::Reduce.RExpr,b::Multivector{V,T}) where {V,T} = Multivector{V}(broadcast(Reduce.Algebra.:*,Ref(a),b.v))
Base.:*(a::Multivector{V,T},b::Reduce.RExpr) where {V,T} = Multivector{V}(broadcast(Reduce.Algebra.:*,a.v,Ref(b)))
Grassmann.:∧(a::Reduce.RExpr,b::Reduce.RExpr) = Reduce.Algebra.:*(a,b)
Grassmann.:∧(a::Reduce.RExpr,b::B) where B<:TensorTerm{V,G} where {V,G} = Single{V,G}(a,b)
Grassmann.:∧(a::A,b::Reduce.RExpr) where A<:TensorTerm{V,G} where {V,G} = Single{V,G}(b,a)
Grassmann.Leibniz.extend_field(Reduce.RExpr)
Grassmann.parsym = (Grassmann.parsym...,Reduce.RExpr)
for T ∈ (:RExpr,:Symbol,:Expr)
    @eval Base.:*(a::Reduce.$T,b::Chain{V,G,Any}) where {V,G} = (a*One(V))*b
    @eval Base.:*(a::Chain{V,G,Any},b::Reduce.$T) where {V,G} = a*(b*One(V))
    eval(Grassmann.generate_inverses(:(Reduce.Algebra),T))
    eval(Grassmann.generate_derivation(:(Reduce.Algebra),T,:df,:RExpr))
    #eval(Grassmann.generate_algebra(:(Reduce.Algebra),T,:df,:RExpr))
end

end # module
