
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

import Reduce: RExpr

*(a::RExpr,b::Basis{V}) where V = SValue{V}(a,b)
*(a::Basis{V},b::RExpr) where V = SValue{V}(b,a)
*(a::RExpr,b::MultiVector{T,V}) where {T,V} = MultiVector{promote_type(T,F),V}(broadcast(Reduce.Algebra.:*,a,b.v))
*(a::MultiVector{T,V},b::RExpr) where {T,V} = MultiVector{promote_type(T,F),V}(broadcast(Reduce.Algebra.:*,a.v,b))
*(a::RExpr,b::MultiGrade{V}) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,a,b.v))
*(a::MultiGrade{V},b::RExpr) where V = MultiGrade{V}(broadcast(Reduce.Algebra.:*,a.v,b))
∧(a::RExpr,b::RExpr) = Reduce.Algebra.:*(a,b)
∧(a::RExpr,b::B) where B<:TensorTerm{V,G} where {V,G} = SValue{V,G}(a,b)
∧(a::A,b::RExpr) where A<:TensorTerm{V,G} where {V,G} = SValue{V,G}(b,a)

for par ∈ (:parany,:parval,:parsym)
    @eval $par = ($par...,RExpr)
end
