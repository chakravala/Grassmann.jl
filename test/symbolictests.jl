
"""
Smoke tests for common SymPy and SymEngine functions.
"""
module SymbolicTests
using Grassmann
    using Test

    @testset "Test SymEngine" begin
        using SymEngine
        @basis S"+++"
        x,y,z = SymEngine.symbols("x y z")

        # expanding the symbolic coefficient of a `Simplex`
        simp = (x+1)^2*v1
        @show SymEngine.expand(simp)

        # expansion/substitution on each symbolic coefficient of a `MultiVector`
        mv = (x+y)^3 * v12 + (y+z) * v123
        @show SymEngine.subs(mv, Dict(x=>2, y=>2, z=>2))
        @show SymEngine.expand(mv)

        # expanding each symbolic coefficient of a `Chain`
        @show expand((x+1)*(x+2)*(v1+v2))
    end

    @testset "Test SymPy" begin
        using SymPy
        @basis S"+++"
        x,y,z = SymPy.symbols("x,y,z")

        @show SymPy.expand((x+y+z)^3*(v1+v12+v123))

        expanded = SymPy.expand((x+1)*(x+y)^5)
        @show unwieldly_multivector = z*v1 + expanded*v12
        @show clean_multivector = SymPy.factor(unwieldly_multivector)
    end

end
