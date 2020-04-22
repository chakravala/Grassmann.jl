# Installation

*Grassmann.jl* is a package for the [Julia language](https://julialang.org), which can be obtained from their website or the recommended method for your operating system (GNU/Linux/Mac/Windows). Go to [docs.julialang.org](https://docs.julialang.org) for documentation.

Availability of this package and its subpackages is automatically handled with Julia's package manager `using Pkg` and `Pkg.add("Grassmann")` or by entering into `]` mode:
```Julia
pkg> add Grassmann
```
If you would like to keep up to date with the latest commits, instead use
```Julia
pkg> add Grassmann#master
```
which is not recommended if you want to use a stable release.

## Requirements

When the `master` branch is used it is possible that some of the dependencies also require a development branch before the release. This may include (but is not limited to) the following packages:

This requires a merged version of `ComputedFieldTypes` at [ComputedFieldTypes.jl](https://github.com/vtjnash/ComputedFieldTypes.jl).

Interoperability of `TensorAlgebra` with other packages is enabled by [DirectSum.jl](https://github.com/chakravala/DirectSum.jl) and [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl).

The package is compatible via [Requires.jl](https://github.com/MikeInnes/Requires.jl) with 
[Reduce.jl](https://github.com/chakravala/Reduce.jl),
[SymPy.jl](https://github.com/JuliaPy/SymPy.jl),
[SymEngine.jl](https://github.com/symengine/SymEngine.jl),
[AbstractAlgebra.jl](https://github.com/wbhart/AbstractAlgebra.jl),
[Nemo.jl](https://github.com/wbhart/Nemo.jl),
[GaloisFields.jl](https://github.com/tkluck/GaloisFields.jl),
[LightGraphs.jl](https://github.com/JuliaGraphs/LightGraphs.jl),
[Compose.jl](https://github.com/GiovineItalia/Compose.jl),
[GeometryTypes.jl](https://github.com/JuliaGeometry/GeometryTypes.jl),
[Makie.jl](https://github.com/JuliaPlots/Makie.jl).

## Grassmann for enterprise

Sponsor this at [liberapay](https://liberapay.com/chakravala), [GitHub Sponsors](https://github.com/sponsors/chakravala), [Patreon](https://patreon.com/dreamscatter), or [Bandcamp](https://music.crucialflow.com); also available as part of the [Tidelift](https://tidelift.com/funding/github/julia/Grassmann) Subscription:

The maintainers of Grassmann and thousands of other packages are working with Tidelift to deliver commercial support and maintenance for the open source dependencies you use to build your applications. Save time, reduce risk, and improve code health, while paying the maintainers of the exact dependencies you use. [Learn more.](https://tidelift.com/subscription/pkg/julia-grassmann?utm_source=julia-grassmann&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)
