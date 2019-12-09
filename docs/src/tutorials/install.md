# Installation

Availability of this package and its subpackages can be automatically handled with the Julia package manager with `using Pkg` and `Pkg.add("Grassmann")` or by entering:
```Julia
pkg> add Grassmann
```

## Requirements

When the `master` branch is used it is possible that some of the dependencies also require a development branch before the release. This may include (but is not limited to) the following packages:

This requires a merged version of `ComputedFieldTypes` at [ComputedFieldTypes.jl](https://github.com/vtjnash/ComputedFieldTypes.jl).

Interoperability of `TensorAlgebra` with other packages is automatically enabled by [DirectSum.jl](https://github.com/chakravala/DirectSum.jl) and [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl).

The package is compatible via [Requires.jl](https://github.com/MikeInnes/Requires.jl) with 
[Reduce.jl](https://github.com/chakravala/Reduce.jl),
[SymPy.jl](https://github.com/JuliaPy/SymPy.jl),
[SymEngine.jl](https://github.com/symengine/SymEngine.jl),
[AbstractAlgebra.jl](https://github.com/wbhart/AbstractAlgebra.jl),
[Nemo.jl](https://github.com/wbhart/Nemo.jl),
[GaloisFields.jl](https://github.com/tkluck/GaloisFields.jl),
[LightGraphs,jl](https://github.com/JuliaGraphs/LightGraphs.jl),
[Compose.jl](https://github.com/GiovineItalia/Compose.jl),
[GeometryTypes,jl](https://github.com/JuliaGeometry/GeometryTypes.jl),
[Makie.jl](https://github.com/JuliaPlots/Makie.jl).
