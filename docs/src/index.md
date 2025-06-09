# Grassmann.jl

*⟨Grassmann-Clifford-Hodge⟩ multilinear differential geometric algebra*

[![YouTube](https://img.shields.io/badge/JuliaCon%202019-YouTube-red)](https://www.youtube.com/watch?v=eQjDN0JQ6-s)
[![DropBox](https://img.shields.io/badge/download_PDF-DropBox-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-juliacon-2019.pdf)
[![DOI](https://zenodo.org/badge/101519786.svg)](https://zenodo.org/badge/latestdoi/101519786)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

The [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) package provides tools for doing computations based on multi-linear algebra, differential geometry, and spin groups using the extended tensor algebra known as Leibniz-Grassmann-Clifford-Hestenes geometric algebra.
Combinatorial products include ``\wedge, \vee, \cdot, *, \star, ', \sim, d, \partial`` (which are the exterior, regressive, inner, and geometric products; along with the Hodge star, adjoint, reversal, differential and boundary operators).
Kernelized operations are built up from composite sparse tensor products and Hodge duality, with high dimensional support for up to 62 indices using staged caching and precompilation. Code generation enables concise yet highly extensible definitions.
[DirectSum.jl](https://github.com/chakravala/DirectSum.jl) multivector parametric type polymorphism is based on tangent bundle vector spaces and conformal projective geometry to make the dispatch highly extensible for many applications.
Additionally, universal interoperability between different sub-algebras is enabled by [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl), on which the type system is built.

```@contents
Pages = ["design.md","algebra.md","library.md","references.md"]
```

This `Grassmann` package for the Julia language was created by [github.com/chakravala](https://github.com/chakravala) for mathematics and computer algebra research with differential geometric algebras.
These projects and repositories were started entirely independently and are available as free software to help spread the ideas to a wider audience.
Please consider donating to show your thanks and appreciation to this project at [liberapay](https://liberapay.com/chakravala), [GitHub Sponsors](https://github.com/sponsors/chakravala), [Patreon](https://patreon.com/dreamscatter), [Tidelift](https://tidelift.com/funding/github/julia/Grassmann), [Bandcamp](https://music.crucialflow.com) or [contribute](https://github.com/chakravala/Grassmann.jl/graphs/contributors) (documentation, tests, examples) in the repositories.
