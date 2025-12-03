module Grassmann

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
#                                              ___
#                 /\                  _____   / _ \
#   __   __      /  \      __   __   (_____) | | | |
#  / / _ \ \    / /\ \    / / _ \ \   _____  | | | |
# | |_/ \_| |  / /  \ \  | |_/ \_| | (_____) | |_| |
#  \___^___/  /_/    \_\  \___^___/           \___/

using SparseArrays, ComputedFieldTypes, AbstractFFTs
using AbstractTensors, Leibniz, DirectSum
import AbstractTensors: Values, Variables, FixedVector, clifford, hodge, wedge, vee

export ⊕, ℝ, @V_str, @S_str, @D_str, Manifold, Submanifold, Signature, DiagonalForm, value
export @basis, @basis_str, @dualbasis, @dualbasis_str, @mixedbasis, @mixedbasis_str, Λ
export ℝ0, ℝ1, ℝ2, ℝ3, ℝ4, ℝ5, ℝ6, ℝ7, ℝ8, ℝ9, mdims, tangent, metric, antimetric, cometric
export hodge, wedge, vee, complement, dot, antidot, istangent, Values, divergence, grad

import Base: @pure, ==, isapprox
import Base: print, show, getindex, setindex!, promote_rule, convert, adjoint
import DirectSum: V0, ⊕, generate, basis, getalgebra, getbasis, dual, Zero, One, Zero, One
import Leibniz: hasinf, hasorigin, dyadmode, value, pre, vsn, metric, mdims, gdims
import Leibniz: bit2int, indexbits, indices, diffvars, diffmask, hasconformal
import Leibniz: symmetricmask, indexstring, indexsymbol, combo, digits_fast
import DirectSum: Basis, Single, Signature, metrichash, antimetric, cometric, signbool

import AbstractTensors: valuetype, scalar, isscalar, trivector, istrivector, ⊗, complement
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume
import AbstractTensors: wedgedot_metric, contraction_metric, log_metric, pseudoscalar

## cache

import Leibniz: algebra_limit, sparse_limit, cache_limit, fill_limit, gdimsall, spincumsum
import Leibniz: binomial, binomial_set, binomsum, binomsum_set, lowerbits, expandbits
import Leibniz: bladeindex, basisindex, indexbasis, indexbasis_set, loworder, intlog
import Leibniz: antisum, antisum_set, anticumsum, antiindex, spinindex, binomcumsum
import Leibniz: promote_type, mvec, svec, intlog, insert_expr, supermanifold

include("multivectors.jl")
include("parity.jl")
include("algebra.jl")
include("products.jl")
include("composite.jl")
include("forms.jl")

## fundamentals

export cayley, hyperplanes, points, TensorAlgebra

@pure hyperplanes(V::Manifold) = map(n->UniformScaling{Bool}(false)*getbasis(V,1<<n),0:rank(V)-1-diffvars(V))

for M ∈ (:Signature,:DiagonalForm)
    @eval (::$M)(::S) where S<:SubAlgebra{V} where V = Multivector{V,Int}(ones(Int,1<<mdims(V)))
end

points(f::F,r=-2π:0.0001:2π) where F<:Function = vector.(f.(r))

export 𝕚,𝕛,𝕜
const 𝕚,𝕛,𝕜 = hyperplanes(ℝ3)

using Leibniz
import Leibniz: ∇, Δ, d, ∂, δ
export ∇, Δ, ∂, d, δ, ↑, ↓, differential, codifferential, boundary, project, reject
export nabla, Nabla, Laplacian

#generate_products(:(Leibniz.Operator),:svec)
for T ∈ (:(Chain{V}),:(Multivector{V}))
    @eval begin
        *(a::Derivation,b::$T) where V = V(a)*b
        *(a::$T,b::Derivation) where V = a*V(b)
    end
end
⊘(x::T,y::Derivation) where T<:TensorAlgebra{V} where V = x⊘V(y)
⊘(x::Derivation,y::T) where T<:TensorAlgebra{V} where V = V(x)⊘y

@pure function (V::Signature{N})(d::Leibniz.Derivation{T,O}) where {N,T,O}
    (O<1||diffvars(V)==0) && (return Chain{V,1,Int}(d.v.λ*ones(Values{N,Int})))
    G,D,C = grade(V),diffvars(V)==1,isdyadic(V)
    G2 = (C ? Int(G/2) : G)-1
    ∇ = sum([getbasis(V,1<<(D ? G : k+G))*getbasis(V,1<<k) for k ∈ 0:G2])
    isone(O) && (return ∇)
    x = (∇⋅∇)^div(isodd(O) ? O-1 : O,2)
    isodd(O) ? sum([(x*getbasis(V,1<<(k+G)))*getbasis(V,1<<k) for k ∈ 0:G2]) : x
end

@pure function (M::Submanifold{W,N})(d::Leibniz.Derivation{T,O}) where {W,N,T,O}
    V = isbasis(M) ? W : M
    (O<1||diffvars(V)==0) && (return Chain{V,1,Int}(d.v.λ*ones(Values{N,Int})))
    G,D,C = grade(V),diffvars(V)==1,isdyadic(V)
    G2 = (C ? Int(G/2) : G)-1
    ∇ = sum([getbasis(V,1<<(D ? G : k+G))*getbasis(V,1<<k) for k ∈ 0:G2])
    isone(O) && (return ∇)
    x = (∇⋅∇)^div(isodd(O) ? O-1 : O,2)
    isodd(O) ? sum([(x*getbasis(V,1<<(k+G)))*getbasis(V,1<<k) for k ∈ 0:G2]) : x
end

@generated ∂(ω::Chain{V,1,<:Chain{W,1}}) where {V,W} = :(∧(ω)⋅$(Λ(W).v1))
∂(ω::T) where T<:TensorAlgebra = ω⋅Manifold(ω)(∇)
d(ω::T) where T<:TensorAlgebra = Manifold(ω)(∇)∧ω
δ(ω::T) where T<:TensorAlgebra = -∂(ω)

"""
    boundary_rank(t)

Returns the rank of the combinatorial boundary operator evaluated on `t`, which can be an element of the `TensorAlgebra`.
Internally the function counts how many graded dimensions of `t` remain after applying the boundary operator `∂` on it, minimized to not exceed the original graded space.
This results in a list of `Values` whose entries `rₖ` satisfy `0 ≤ rₖ ≤ dₖ` computed with `d = count_gdims(t)` as utility methods.
"""
function boundary_rank(t,d=count_gdims(t))
    out = count_gdims(∂(t))
    out[1] = 0
    for k ∈ 2:length(out)-1
        @inbounds out[k] = min(out[k],d[k+1])
    end
    return Values(out)
end

"""
    boundary_null(t)

Dimension of the nullspace (kernel) of the combinatorial boundary operator on `t`, which can be an element of the `TensorAlgebra`.
For each grade `k` the function lists `Values` with entries `dₖ₊₁ - rₖ`, where the computation follows from `d = count_gdims(t)` and `r = boundary_rank(t)` utility methods.
Intuitively, this measures how many `k`‑chains are boundaries of (k+1)‑chains and therefore vanish under the `∂` operation.
"""
function boundary_null(t)
    d = count_gdims(t)
    r = boundary_rank(t,d)
    l = length(d)
    out = zeros(Variables{l,Int})
    for k ∈ 1:l-1
        @inbounds out[k] = d[k+1] - r[k]
    end
    return Values(out)
end

"""
    betti(::TensorAlgebra)

Compute combinatoric Betti numbers based on the `count_gdims` and `boundary_rank` methods.
"""
function betti(t::T) where T<:TensorAlgebra
    d = count_gdims(t)
    r = boundary_rank(t,d)
    l = length(d)-1
    out = zeros(Variables{l,Int})
    for k ∈ 1:l
        @inbounds out[k] = d[k+1] - r[k] - r[k+1]
    end
    return Values(out)
end

@generated function project(ω::T) where T<:TensorAlgebra
    V = Manifold(ω)
    T<:Submanifold && !isbasis(ω) && (return Leibniz.supermanifold(V))
    !(hasinf(V)||hasorigin(V)) && (return :ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        :((($G.v∞*(one(valuetype(ω))/2))*((~ω)⋅ω)+$G.v∅)+ω)
        #:((($G.v∞*(one(valuetype(ω))/2))*ω^2+$G.v∅)+ω)
    else
        quote
            ω2 = (~ω)⋅ω # ω^2
            iω2 = inv(ω2+1)
            (hasinf($V) ? $G.v∞ : $G.v∅)*((ω2-1)*iω2) + (2iω2)*ω
        end
    end
end
#project(ω::ChainBundle) = ω
function project(ω,b)
    ω2 = (~ω)⋅ω # ω^2
    iω2 = inv(ω2+1)
    (2iω2)*ω + ((ω2-1)*iω2)*b
end
function project(ω,p,m)
    ω2 = scalar((~ω)⋅ω) # ω^2
    iω2 = inv(ω2+1)
    (2iω2)*ω + ((ω2-1)*iω2)*p + ((ω2+1)*iω2)*m
end

@generated function reject(ω::T) where T<:TensorAlgebra
    V,M = Manifold(ω),T<:Submanifold && !isbasis(ω)
    !(hasinf(V)||hasorigin(V)) && (return M ? V(2:mdims(V)) : :ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        M && (return ω(3:mdims(V)))
        #:(inv(one(valuetype(ω))*$G.v∞∅)*($G.v∞∅∧ω)/(-ω⋅$G.v∞))
        :((($G.v∞∅∧ω)⋅inv(one(valuetype(ω))*~$G.v∞∅))/(-ω⋅$G.v∞))
    else
        M && (return V(2:mdims(V)))
        quote
            b = hasinf($V) ? $G.v∞ : $G.v∅
            (~(ω∧b)⋅b)/(1-b⋅ω) # ((ω∧b)*b)/(1-b⋅ω)
        end
    end
end
#reject(ω::ChainBundle) = ω(list(2,mdims(ω)))
reject(ω,b) = (~(b∧ω)⋅b)/(1-ω⋅b) # ((b∧ω)*b)/(1-ω⋅b)
reject(ω,∞,∅) = (m=∞∧∅;((m∧ω)⋅~inv(m))/(-ω⋅∞)) #(m=∞∧∅;inv(m)*(m∧ω)/(-ω⋅∞))

const ↑,↓ = project,reject

@doc """
    ↑(ω::TensorAlgebra{V}) where V # project

Canonical up-`project` operation from the space `V`, based on either Euclidean projective geometry, or the Riemann sphere, or conformal geometric algebra, or potentially other future canonical specifications.
Optional arguments expose lower-level building blocks: `↑(ω,b)` — use an explicit projective basis element `b`, or in the CGA context `↑(ω,p,m)` — parameterise the conformal split with point-like part `p` and Minkowski part `m`.
See also `↓` for the inverse down-`reject` operation.
""" project, ↑

@doc """
    ↓(ω::TensorAlgebra{V}) where V # reject

Canonical down-`reject` operation from the space `V`, based on either Euclidean projective geometry, or the Riemann sphere, or conformal geometric algebra, or potentially other future canonical specifications.
Optional arguments expose lower-level building blocks: `↓(ω,b)` — use an explicit projective basis element `b`, or in the CGA context `↓(ω,p,m)` — parameterise the conformal split with point-like part `p` and Minkowski part `m`.
See also `↑` for the inverse up-`project` operation.
""" reject, ↓

## skeleton / subcomplex

export skeleton, 𝒫, collapse, subcomplex, chain, path

absym(t) = abs(t)
absym(t::Submanifold) = t
absym(t::T) where T<:TensorTerm{V,G} where {V,G} = Single{V,G}(absym(value(t)),basis(t))
absym(t::Chain{V,G,T}) where {V,G,T} = Chain{V,G}(absym.(value(t)))
absym(t::Multivector{V,T}) where {V,T} = Multivector{V}(absym.(value(t)))

collapse(a,b) = a⋅absym(∂(b))

function chain(t::S,::Val{T}=Val{true}()) where S<:TensorTerm{V} where {V,T}
    N,B,v = mdims(V),UInt(basis(t)),value(t)
    C = symmetricmask(V,B,B)[1]
    G = count_ones(C)
    G < 2 && (return t)
    out,ind = zeros(mvec(N,2,Int)), indices(C,N)
    if T || G == 2
        setblade!(out,G==2 ? v : -v,bit2int(indexbits(N,[ind[1],ind[end]])),Val{N}())
    end
    for k ∈ 2:G
        setblade!(out,v,bit2int(indexbits(N,ind[[k-1,k]])),Val{N}())
    end
    return Chain{V,2}(out)
end
path(t) = chain(t,Val{false}())

#@inline (::Leibniz.Derivation{Bool,2})(x::T,v=Val{true}()) where T<:TensorAlgebra = skeleton(x,v)
𝒫(t::T) where T<:TensorAlgebra = Δ(t,Val{false}())
subcomplex(x::S,v=Val{true}()) where S<:TensorAlgebra = Δ(absym(∂(x)),v)
function skeleton(x::S,v::Val{T}=Val{true}()) where S<:TensorTerm{V} where {V,T}
    B = UInt(basis(x))
    count_ones(symmetricmask(V,B,B)[1])>0 ? absym(x)+skeleton(absym(∂(x)),v) : (T ? Zero(V) : absym(x))
end
function skeleton(x::Chain{V},v::Val{T}=Val{true}()) where {V,T}
    N,G,g = mdims(V),rank(x),0
    ib = indexbasis(N,G)
    for k ∈ 1:binomial(N,G)
        if !iszero(x.v[k]) && (!T || count_ones(symmetricmask(V,ib[k],ib[k])[1])>0)
            g += skeleton(Single{V,G}(x.v[k],getbasis(V,ib[k])),v)
        end
    end
    return g
end
function skeleton(x::Multivector{V},v::Val{T}=Val{true}()) where {V,T}
    N,g = mdims(V),0
    for i ∈ 0:N
        R = binomsum(N,i)
        ib = indexbasis(N,i)
        for k ∈ 1:binomial(N,i)
            if !iszero(x.v[k+R]) && (!T || count_ones(symmetricmask(V,ib[k],ib[k])[1])>0)
                g += skeleton(Single{V,i}(x.v[k+R],getbasis(V,ib[k])),v)
            end
        end
    end
    return g
end

# mesh

export column, columns

column(t,i=1) = getindex.(value(t),i)
columns(t,i=1,j=mdims(Manifold(t))) = column.(Ref(value(t)),list(i,j))

rows(a::T) where T<:AbstractMatrix = getindex.(Ref(a),list(1,3),:)

function pointset(e)
    mdims(Manifold(e)) == 1 && (return column(e))
    out = Int[]
    for i ∈ value(e)
        for k ∈ value(i)
            k ∉ out && push!(out,k)
        end
    end
    return out
end

export scalarfield, vectorfield, pointfield, chainfield, rectanglefield # rectangle

function pointfield end; const vectorfield = pointfield # deprecate ?
function pointpair end; function ptype end

chainfield(t,V=Manifold(t),W=V) = p->V(vector(↓(↑((V∪Manifold(t))(p))⊘t)))
function scalarfield(t,ϕ::T) where T<:AbstractVector
    M = Manifold(t)
    V = Manifold(M)
    P->begin
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return ((Pi\P)⋅Chain{V,1}(ϕ[ti]))[1])
        end
        return 0.0
    end
end
function chainfield(t,ϕ::T) where T<:AbstractVector
    M = Manifold(t)
    V = Manifold(M)
    z = mdims(V) ≠ 4 ? Chain{V,1}(1.0,0.0,0.0) : Chain{V,1}(1.0,0.0,0.0,0.0)
    P->begin
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return (Pi\P)⋅Chain{V,1}(ϕ[ti]))
        end
        return z
    end
end

function rectangle(p,nx=100,ny=nx)
    px,py = column(p,2),column(p,3)
    x = range(minimum(px),maximum(px),length=nx)
    y = range(minimum(py),maximum(py),length=ny)
    z = x' .+ im*y
    Chain{Manifold(p),1}.(1.0,real.(z),imag.(z))
end
rectanglefield(t,ϕ,nx=100,ny=nx) = chainfield(t,ϕ).(rectangle(points(t),nx,ny))

for fun ∈ (:fft,:fft!,:ifft,:ifft!,:bfft,:bfft!,:rfft,:irfft,:brfft)
    @eval begin
        AbstractFFTs.$fun(t::AbstractArray{<:TensorGraded{V,0}},args...) where V = $fun(Real.(t),args...)
        AbstractFFTs.$fun(t::AbstractArray{<:TensorGraded{V}},args...) where V = $fun(Couple.(t),args...)
        AbstractFFTs.$fun(t::AbstractArray{<:Couple{V,B}},args...) where {V,B} = Couple{V,B}.($fun(Complex.(t),args...))
        AbstractFFTs.$fun(t::AbstractArray{<:Chain},args...) = $fun(complexify.(t),args...)
        AbstractFFTs.$fun(t::AbstractArray{<:Phasor},args...) = $fun(complexify.(t),args...)
    end
end

eval(generate_products())
eval(generate_products(Complex))
eval(generate_products(Rational{BigInt},:svec))
for Big ∈ (BigFloat,BigInt)
    eval(generate_products(Big,:svec))
    eval(generate_products(Complex{Big},:svec))
end
eval(generate_products(SymField,:svec,:($Sym.:∏),:($Sym.:∑),:($Sym.:-),:($Sym.conj)))
function generate_derivation(m,t,d,c)
    :(Grassmann.derive(n::$(:($m.$t)),b) = $m.$d(n,$m.$c(Grassmann.indexsymbol(Manifold(b),UInt(b)))))
end
function generate_algebra(m,t,mt,d=nothing,c=nothing)
    out = Any[quote
        Base.:*(a::$m.$t,b::Single{V,G,B,T}) where {V,G,B,T<:Real} = Single{V}(a,b)
        Base.:*(a::Single{V,G,B,T},b::$m.$t) where {V,G,B,T<:Real} = Single{V}(b,a)
        Base.iszero(a::Single{V,G,B,$m.$t}) where {V,G,B} = false
    end]
    for op ∈ (:+,:-)
        for Term ∈ (:TensorGraded,:TensorMixed)
            push!(out,quote
                Base.$op(a::T,b::$m.$t) where T<:$Term = $op(a,b*One(Manifold(a)))
                Base.$op(a::$m.$t,b::T) where T<:$Term = $op(a*One(Manifold(b)),b)
            end)
        end
    end
    push!(out,generate_products(:($m.$t),:svec,:($m.:*),:($m.:+),:($m.:-),:($m.conj),true,mt))
    push!(out,generate_inverses(m,t))
    !isnothing(d) && push!(out,generate_derivation(m,t,d,c))
    Expr(:block,out...)
end
function generate_symbolic_methods(mod, symtype, methods_noargs, methods_args)
    n,m = length(methods_noargs),length(methods_args)
    out = Vector{Any}(undef,n+m)
    for i ∈ 1:n
        method = methods_noargs[i]
        out[i] = quote
            $mod.$method(x::T) where T<:TensorGraded = map(v -> typeof(v) == $mod.$symtype ? $mod.$method(v) : v, x)
            $mod.$method(x::T) where T<:TensorMixed = map(v -> typeof(v) == $mod.$symtype ? $mod.$method(v) : v, x)

        end
    end
    for i ∈ 1:m
        method = methods_args[i]
        out[n+i] = quote
            $mod.$method(x::T, args...) where T<:TensorGraded = map(v -> typeof(v) == $mod.$symtype ? $mod.$method(v, args...) : v, x)
            $mod.$method(x::T, args...) where T<:TensorMixed = map(v -> typeof(v) == $mod.$symtype ? $mod.$method(v, args...) : v, x)
        end
    end
    Expr(:block,out...)
end

check_parsym(Field) = Leibniz.check_field(Field)
check_parsym(::Type{<:Symbol}) = true

extend_parsym(Field) = (global parsym = (parsym...,Field))

if !isdefined(Base, :get_extension)
using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
    @require Reduce="93e0c654-6965-5f22-aba9-9c1ae6b3c259" include("../ext/ReduceExt.jl")
    @require Symbolics="0c5d862f-8b57-4792-8d23-62f2024744c7" include("../ext/SymbolicsExt.jl")
    @require SymPy="24249f21-da20-56a4-8eb1-6a02cf4ae2e6" include("../ext/SymPyExt.jl")
    @require SymEngine="123dc426-2d89-5057-bbad-38513e3affd8" include("../ext/SymEngineExt.jl")
    @require AbstractAlgebra="c3fe647b-3220-5bb0-a1ea-a7954cac585d" include("../ext/AbstractAlgebraExt.jl")
    @require GaloisFields="8d0d7f98-d412-5cd4-8397-071c807280aa" include("../ext/GaloisFieldsExt.jl")
    @require LightGraphs="093fc24a-ae57-5d10-9952-331d41423f4d" include("../ext/LightGraphsExt.jl")
    #=@require Compose="a81c6b42-2e10-5240-aca2-a61377ecd94b" begin
        import LightGraphs, GraphPlot, Cairo
        viewer = Base.Process(`$(haskey(ENV,"VIEWER") ? ENV["VIEWER"] : "xdg-open") simplex.pdf`,Ptr{Nothing}())
        function Compose.draw(img,x::T,l=layout=GraphPlot.circular_layout) where T<:TensorAlgebra
            Compose.draw(img,GraphPlot.gplot(LightGraphs.SimpleDiGraph(x),layout=l,nodelabel=collect(1:mdims(Manifold(x)))))
        end
        function graph(x,n="simplex.pdf",l=GraphPlot.circular_layout)
            cmd = `$(haskey(ENV,"VIEWER") ? ENV["VIEWER"] : "xdg-open") $n`
            global viewer
            viewer.cmd == cmd && kill(viewer)
            Compose.draw(Compose.PDF(n,16Compose.cm,16Compose.cm),x,l)
            viewer = run(cmd,(devnull,stdout,stderr),wait=false)
        end
    end=#
    @require StaticArrays="90137ffa-7385-5640-81b9-e52037218182" include("../ext/StaticArraysExt.jl")
    @require Meshes="eacbb407-ea5a-433e-ab97-5258b1ca43fa" include("../ext/MeshesExt.jl")
    @require GeometryBasics="5c1252a2-5f33-56bf-86c9-59e7332b4326" include("../ext/GeometryBasicsExt.jl")
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("../ext/MakieExt.jl")
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" include("../ext/UnicodePlotsExt.jl")
    @require SpecialFunctions="276daf66-3868-5448-9aa4-cd146d93841b" include("../ext/SpecialFunctionsExt.jl")
    @require EllipticFunctions="6a4e32cb-b31a-4929-85af-fb29d9a80738" include("../ext/EllipticFunctionsExt.jl")
    @require FewSpecialFunctions="6fcbd3ca-4273-49c4-98b3-81b765566de6" include("../ext/FewSpecialFunctionsExt.jl")
end
end

#   ____  ____    ____   _____  _____ ___ ___   ____  ____   ____
#  /    T|    \  /    T / ___/ / ___/|   T   T /    T|    \ |    \
# Y   __j|  D  )Y  o  |(   \_ (   \_ | _   _ |Y  o  ||  _  Y|  _  Y
# |  T  ||    / |     | \__  T \__  T|  \_/  ||     ||  |  ||  |  |
# |  l_ ||    \ |  _  | /  \ | /  \ ||   |   ||  _  ||  |  ||  |  |
# |     ||  .  Y|  |  | \    | \    ||   |   ||  |  ||  |  ||  |  |
# l___,_jl__j\_jl__j__j  \___j  \___jl___j___jl__j__jl__j__jl__j__j

end # module
