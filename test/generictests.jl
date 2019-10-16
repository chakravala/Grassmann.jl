
"""
Tests based on mathematic properties of a Geometric Algebra that should pass
for any signature or field.
"""
module GenericTests
using Grassmann
using Test
using LinearAlgebra

# TODO: move tests for isapprox and scalar to a more appropriate file
#       since they are no general mathematical properties of an GA
@testset "Test isapprox" begin
    basis"2"

    # basis
    @test v ≈ v
    @test v1 ≈ v1
    @test v2 ≈ v2
    @test v12 ≈ v12
    # S/MBlade
    @test 2v ≈ 2v
    @test 2v1 ≈ 2v1
    # chain
    @test v1 + v2 ≈ v1 + v2
    # multivector
    @test v + v2 ≈ v + v2
    @test v + v12 ≈ v + v12
    @test v + v2 + v12 ≈ v + v2 + v12

    # basis and others
    @test !(v ≈ v1)
    @test !(v ≈ v12)
    @test !(v ≈ v1+v2)
    @test !(v ≈ v1+v)
    @test !(v ≈ v1+v12)
    @test !(v ≈ v+v1+v12)

    # S/MBlade and others
    @test !(2v ≈ v1)
    @test !(2v ≈ v12)
    @test !(2v ≈ v1+v2)
    @test !(2v ≈ v1+v)
    @test !(2v ≈ v1+v12)
    @test !(2v ≈ v+v1+v12)

    # Chain and others
    @test !(v1 + v2 ≈ v1)
    @test !(v1 + v2 ≈ v12)
    @test !(v1 + v2 ≈ v1+v)
    @test !(v1 + v2 ≈ v1+v12)
    @test !(v1 + v2 ≈ v+v1+v12)

    # multivector and others
    @test !(v+v1+v12 ≈ v1)
    @test !(v+v1+v12 ≈ v12)
    @test !(v+v1+v12 ≈ v1+v)
    @test !(v+v1+v12 ≈ v1+v12)
end

@testset "method: scalar" begin
    basis"2"
    @test scalar(v) == 1v
    @test scalar(2v) == 2v
    @test scalar(v1) == 0v
    @test scalar(-v+v1) == -1v
    @test scalar(v-v) == 0v
    @test scalar(v1+v2) == 0v
end

for 𝔽 in [Float64]
    e = one(𝔽)
    α = rand(𝔽)
    @test α*e == α
    @testset "Field: $(𝔽)" begin
        for G in [V"+++", S"∞+", S"∅+", V"-+++", S"∞∅+"]
            @testset "Algebra: $(G)" begin
                dims = ndims(G)

                # all basis elements of the whole algebra
                basis = collect(G)[1:end]

                # all basis vectors of the generating vectorspace
                basisvecs = basis[2:dims+1]

                # test set of vectors
                b = collect(1:dims)
                B = sum(b.*basisvecs)
                vectors = Any[basisvecs...]
                push!(vectors, B)

                # test set of multivectors
                a = rand(𝔽, 2^dims)
                A = sum(a.*basis)
                multivectors = Any[basis...]
                push!(multivectors, A)
                push!(multivectors, B)

                @testset "Existence of unity" begin
                    for A in multivectors
                        @test e*A == A == A*e
                    end
                end

                @testset "a² ∈  𝔽" begin
                    for a in vectors
                        @test a^2 ≈ scalar(a^2)*basis[1]
                    end
                end

                @testset "Associativity" begin
                    for A in multivectors,
                        B in multivectors,
                        C in multivectors

                        @test (A*(B*C)) ≈ ((A*B)*C)
                    end
                end

                @testset "Distributivity" begin
                    for A in multivectors,
                        B in multivectors,
                        C in multivectors

                        @test A*(B + C) ≈ (A*B) + (A*C)
                    end
                end

                @testset "a⋅b = 0.5(ab + ba)" begin
                    for a in vectors, b in vectors
                        @test a⋅b == 0.5*(a*b + b*a)
                    end
                end

                @testset "a∧b = 0.5(ab - ba)" begin
                    for a in vectors, b in vectors
                        @test a∧b == 0.5*(a*b - b*a)
                    end
                end

                @testset "ab = a⋅b + a∧b" begin
                    for a in vectors, b in vectors
                        @test a*b == a⋅b + a∧b
                    end
                end

                @testset "ab = 2a⋅b - ba" begin
                    for a in vectors, b in vectors
                        @test a*b == 2*a⋅b - b*a
                    end
                end

                @testset "aa = a⋅a" begin
                    for a in vectors
                        @test a*a == a⋅a
                    end
                end
            end
        end
    end
end

end
