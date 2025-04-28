"""
Alias definitions providing plain‑ASCII names for selected unicode-heavy
functions in Grassmann.jl.  Each alias is documented and simply forwards all
its arguments to the canonical unicode implementation so that the two names
can be used interchangeably.
"""

# Gradient / Laplacian -------------------------------------------------------

"""
    nabla(xs...)

English alias for [`∇`](@ref).  Accepts the same arguments and returns the
same result.
"""
nabla(xs...) = ∇(xs...)

"""
    laplacian(xs...)

English alias for [`Δ`](@ref), the Laplace operator.  Accepts the same
arguments and returns the same result.
"""
laplacian(xs...) = Δ(xs...)

# Exterior & co‑derivatives ---------------------------------------------------

"""
    exterior(xs...)

English alias for [`d`](@ref), the exterior derivative.
"""
exterior(xs...) = d(xs...)

"""
    boundary(xs...)

English alias for [`∂`](@ref), the (geometric) boundary operator.
"""
boundary(xs...) = ∂(xs...)

"""
    codifferential(xs...)

English alias for [`δ`](@ref), the co‑differential.
"""
codifferential(xs...) = δ(xs...)

# Conformal projections -------------------------------------------------------

"""
    up(xs...)

English alias for [`↑`](@ref) (conformal up‑projection).  All arguments are
forwarded verbatim.
"""
up(xs...) = ↑(xs...)

"""
    down(xs...)

English alias for [`↓`](@ref) (conformal down‑projection).  All arguments are
forwarded verbatim.
"""
down(xs...) = ↓(xs...)