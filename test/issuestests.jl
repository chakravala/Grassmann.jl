"""
Unittests for issues from Github
"""
module IssuesTests

using Grassmann
using Test

@testset "Issue #19: Conformal split example" begin
    @basis S"∞∅++"
    @test (v∞^2, v∅^2, v1^2, v2^2) == (0v, 0v, v, v)
    @test v∞ ⋅ v∅ == -1v
    @test v∞∅^2 == v
    @test (v∞∅ * v∞, v∞∅ * v∅) == (-1v∞, v∅)
    @test (v∞ * v∅, v∅ * v∞) == (-1 + 1v∞∅, -1 - 1v∞∅)
end

@testset "Issue #17: Equality between Number and MultiVector" begin
    basis"2"
    
    a = v + v1 - v1
    @test a == v
    @test typeof(a) <: MultiVector
    @test a == 1
    
    b = a - 1
    @test b == 0
    @test b == 0

    @test a - 1 == 0
end

@testset "Issue #16: Adding basis vector to multivector" begin
    basis"2"

    A = 2v1 + v2
    B = v1 + v2

    @test A + B == 3v1 + 2v2
    @test A == 2v1 + v2
    @test B == v1 + v2
    
    @test v1 + A == 3v1 + 1v2
    @test A == 2v1 + v2
end

@testset "Issue #14: Error when adding MultiVectors" begin
    basis"+++"
    @test (v1+v2) + (v1+v2)*(v1+v2) == 2 + 1v1 + 1v2
end

@testset "Issue #15: generate rotation quaternions using exp" begin
    basis"3"
    i, j, k = hyperplanes(ℝ^3)
    alpha = 0.5π

    @test exp(alpha/2*(i)) ≈ sqrt(2)*(1+i)/2

    a, b, c = 1/sqrt(2) * [1, 1, 0]
    @test exp(alpha/2*(a*i + b*j + c*k)) ≈ (sqrt(2)+j+i)/2
end

@testset "Issue #20: geometric product of null basis and negative origin" begin
    @basis S"∞∅+"
    
    @test v∅*v∞ == -1 - v∞∅
    @test v∅*(-v∞) == 1 + v∞∅
    
    a = v∅*basis(-v∞)
    @test a == -1 - v∞∅
    @test Simplex{V}(-1, a) == -a
end

@testset "Issue #22: Error in MultiVector constructor for Chains" begin
    basis"++"

    a = v1 + v2
    @test typeof(a) <: Chain

    @test MultiVector(a) == v1 + v2
    @test Chain(v) == v
end
end
