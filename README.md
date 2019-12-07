*⟨Leibniz-Grassmann-Clifford-Hestenes⟩ differential geometric algebra / multivector simplicial complex*

<p align="center">
  <img src="./dev/assets/logo.png" alt="Grassmann.jl"/>
</p>

The [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) package provides tools for doing computations based on multi-linear algebra, differential geometry, and spin groups using the extended tensor algebra known as Leibniz-Grassmann-Clifford-Hestenes geometric algebra.
Combinatorial products include the exterior, regressive, inner, and geometric products; along with the Hodge star, adjoint, reversal, and boundary operators).
The kernelized operations are built up from composite sparse tensor products and Hodge duality, with high dimensional support for up to 62 indices using staged caching and precompilation. Code generation enables concise yet highly extensible definitions.
The [DirectSum.jl](https://github.com/chakravala/DirectSum.jl) multivector parametric type polymorphism is based on tangent bundle vector spaces and conformal projective geometry to make the dispatch highly extensible for many applications.
Additionally, the universal interoperability between different sub-algebras is enabled by [AbstractTensors.jl](https://github.com/chakravala/AbstractTensors.jl), on which the type system is built.

[![YouTube](https://img.shields.io/badge/JuliaCon%202019-YouTube-red)](https://www.youtube.com/watch?v=eQjDN0JQ6-s)
[![DropBox](https://img.shields.io/badge/download_PDF-DropBox-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-juliacon-2019.pdf)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://grassmann.crucialflow.com/stable)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://grassmann.crucialflow.com/latest)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![BiVector](https://img.shields.io/badge/bivector.net-Discourse-blueviolet)](https://bivector.net)

This `Grassmann` package for the Julia language was created by [github.com/chakravala](https://github.com/chakravala), a mathematics and computer algebra researcher for differential geometric algebras.
These projects and repositories were started entirely independently and are available as free software to help spread the ideas to a wider audience.
Please consider donating to show your thanks and appreciation to this project at [liberapay](https://liberapay.com/chakravala), [GitHub Sponsors](https://github.com/sponsors/chakravala), [Patreon](https://patreon.com/dreamscatter), [Tidelift](https://tidelift.com/funding/github/julia/Grassmann), [Bandcamp](https://music.crucialflow.com) or contribute (documentation, tests, examples) in the repositories.

[![Build Status](https://travis-ci.org/chakravala/Grassmann.jl.svg?branch=master)](https://travis-ci.org/chakravala/Grassmann.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/c36u0rgtm2rjcquk?svg=true)](https://ci.appveyor.com/project/chakravala/grassmann-jl)
[![Coverage Status](https://coveralls.io/repos/chakravala/Grassmann.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chakravala/Grassmann.jl?branch=master)
[![codecov.io](http://codecov.io/github/chakravala/Grassmann.jl/coverage.svg?branch=master)](http://codecov.io/github/chakravala/Grassmann.jl?branch=master)
[![DOI](https://zenodo.org/badge/101519786.svg)](https://zenodo.org/badge/latestdoi/101519786)
[![Liberapay patrons](https://img.shields.io/liberapay/patrons/chakravala.svg)](https://liberapay.com/chakravala)
