#   This file is part of Grassmann.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

push!(LOAD_PATH, "..")
using Documenter, AbstractTensors, DirectSum, Leibniz, Grassmann

makedocs(
    # options
    modules=[AbstractTensors, DirectSum, Leibniz, Grassmann],#Adapode],
    doctest=false,
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
    sitename="Grassmann.jl",
    authors="Michael Reed",
    pages=Any[
        "Home"=>"index.md",
        "Design"=>"design.md",
        "Algebra"=>"algebra.md",
        "Library"=>"library.md",
        "AGPL-3.0"=>"agpl.md",
        "Tutorials"=>Any[
            "tutorials/install.md",
            "tutorials/quick-start.md",
            "tutorials/algebra-of-space.md",
            "tutorials/dyadic-tensors.md"
        ],
        "References"=>"references.md"
    ]
)

deploydocs(
    repo="github.com/chakravala/Grassmann.jl.git",
    target="build",
    deps=nothing,
    make=nothing
)
