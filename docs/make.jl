#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

using Documenter, Grassmann

makedocs(
    # options
    modules = [Grassmann],
    doctest = false,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "Grassmann.jl",
    authors = "Michael Reed",
    pages = Any[
        "Home" => "index.md",
        "Design" => "design.md",
        "Algebra" => "algebra.md",
        "Library" => "library.md",
        "Tutorials" => Any[
            "tutorials/install.md",
            "tutorials/quick-start.md",
            "tutorials/forms.md"
            ],
        "References" => "references.md"
        ]
)

deploydocs(
    repo   = "github.com/chakravala/Grassmann.jl.git",
    target = "build",
    deps = nothing,
    make = nothing
)
