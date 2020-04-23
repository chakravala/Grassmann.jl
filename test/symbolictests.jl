
"""
Smoke tests for common SymPy and SymEngine functions.
"""
module SymEngineTests
    using Grassmann
    using Test

    @testset "Test SymEngine" begin
        using SymEngine
        @basis S"+++"
        x,y,z = symbols("x y z")

        # expanding the symbolic coefficient of a `Simplex`
        simp = (x+1)^2*v1
        @show expand(simp)

        # expansion/substitution on each symbolic coefficient of a `MultiVector`
        mv = (x+y)^3 * v12 + (y+z) * v123
        @show expand(mv)
        @show numeric_mv = N(subs(mv, Dict(x=>2, y=>2, z=>2)))
        @show map(typeof, numeric_mv.v)

        # expanding each symbolic coefficient of a `Chain`
        @show expand((x+1)*(x+2)*(v1+v2))
    end
end

module SymPyTests
    using Grassmann
    using Test

    @testset "Test SymPy" begin
        using SymPy
        @basis S"+++"
        x,y,z = symbols("x,y,z")

        @show expand((x+y+z)^3*(v1+v12+v123))

        expanded = expand((x+1)*(x+y)^5)
        @show unwieldly_multivector = z*v1 + expanded*v12
        @show clean_multivector = factor(unwieldly_multivector)

        mv = (x+y)^3 * v12 + (y+z) * v123

        @show numeric_mv = N(subs(mv, Dict(x=>2, y=>2, z=>2)))
        @show map(typeof, numeric_mv.v)


    end
end
