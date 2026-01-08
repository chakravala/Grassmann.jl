# Grassmann.jl

*⟨Grassmann-Clifford-Hodge⟩ multilinear differential geometric algebra*

[![JuliaCon 2019](https://img.shields.io/badge/JuliaCon-2019-red)](https://www.youtube.com/watch?v=eQjDN0JQ6-s)
[![Grassmann.jl YouTube](https://img.shields.io/badge/Grassmann.jl-YouTube-red)](https://youtu.be/worMICG1MaI)
[![PDF 2019](https://img.shields.io/badge/PDF-2019-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-juliacon-2019.pdf)
[![PDF 2021](https://img.shields.io/badge/PDF-2021-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=differential-geometric-algebra-2021.pdf)
[![PDF 2025](https://img.shields.io/badge/PDF-2025-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-cartan-2025.pdf)
[![Hardcover 2025](https://img.shields.io/badge/Hardcover-2025-blue.svg)](https://www.lulu.com/shop/michael-reed/principal-differential-geometric-algebra/hardcover/product-kv6n8j8.html)
[![Paperback 2025](https://img.shields.io/badge/Paperback-2025-blue.svg)](https://www.lulu.com/shop/michael-reed/principal-differential-geometric-algebra/paperback/product-yvk7zqr.html)
[![DOI](https://zenodo.org/badge/101519786.svg)](https://zenodo.org/badge/latestdoi/101519786)

The [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) package provides tools for computations based on multi-linear algebra and spin groups using the extended geometric algebra known as Grassmann-Clifford-Hodge algebra.
Algebra operations include exterior, regressive, inner, and geometric, along with the Hodge star and boundary operators.
Code generation enables concise usage of the algebra syntax.
[DirectSum.jl](https://github.com/chakravala/DirectSum.jl) multivector parametric type polymorphism is based on tangent vector spaces and conformal projective geometry.
Additionally, the universal interoperability between different sub-algebras is enabled by [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl), on which the type system is built.
The design is based on `TensorAlgebra{V}` abstract type interoperability from *AbstractTensors.jl* with a ``\mathbb{K}``-module type parameter ``V`` from *DirectSum.jl*.
Abstract vector space type operations happen at compile-time, resulting in a differential geometric algebra of multivectors.

```@contents
Pages = ["design.md","algebra.md","videos.md","library.md","references.md"]
```

* Michael Reed, [Principal Differential Geometric Algebra: compute using Grassmann.jl, Cartan.jl](https://www.lulu.com/shop/michael-reed/principal-differential-geometric-algebra/hardcover/product-kv6n8j8.html) (Hardcover, 2025)
* Michael Reed, [Principal Differential Geometric Algebra: compute using Grassmann.jl, Cartan.jl](https://www.lulu.com/shop/michael-reed/principal-differential-geometric-algebra/paperback/product-yvk7zqr.html) (Paperback, 2025)

Please consider donating to show your thanks and appreciation to this project at [liberapay](https://liberapay.com/chakravala), [GitHub Sponsors](https://github.com/sponsors/chakravala), [Patreon](https://patreon.com/dreamscatter), or [contribute](https://github.com/chakravala/Grassmann.jl/graphs/contributors) (documentation, tests, examples) in the repositories.
