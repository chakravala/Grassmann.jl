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
        "Library" => "library.md" 
        ]
)

deploydocs(
    repo   = "github.com/chakravala/Reduce.jl.git",
    target = "build",
    deps = nothing,
    make = nothing
)
