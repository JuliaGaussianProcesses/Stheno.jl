using Random, LinearAlgebra
using Stheno: WoodburyMat, Xt_invA_X, LazyPDMat

@testset "woodbury" begin

    let
        rng, N, D = MersenneTwister(123456), 10, 5
        X, Σ_, σ = randn(rng, D, N), randn(rng, D, D), exp(randn(rng))
        Σ = LazyPDMat(Σ_ * Σ_' + 1e-9I, 0)
        A = WoodburyMat(X, Σ, σ)
        B, b = randn(N, 2), randn(N)

        # Test that conversion and indexing are consistent with one another.
        Am, Al = Matrix(A), LazyPDMat(A)
        @test size(A) == (N, N)
        @test Matrix(Al) == Am
        @test Am[2, 3] ≈ A[2, 3]
        @test Am[3, 2] ≈ A[3, 2]

        # Check unary operations.
        @test logdet(A) ≈ logdet(Matrix(A))

        # Test basics of matrix multiplication.
        @test Matrix(A * B) ≈ Matrix(A) * B
        @test Vector(A * b) ≈ Matrix(A) * b
        Bt = B'
        @test Matrix(Bt * A) ≈ B' * Matrix(A)

        # Test basics of linear systems solving.
        @test Matrix(A \ B) ≈ Matrix(A) \ B
        @test Vector(A \ b) ≈ Matrix(A) \ b

        @test Matrix(B' / A) ≈ B' / Matrix(A)

        # Test quadratic form solving.
        @test Xt_invA_X(A, B) isa Matrix{<:Real}
        @test Xt_invA_X(A, B) ≈ B' * (Matrix(A) \ B)

        @test Xt_invA_X(A, b) isa Real
        @test Xt_invA_X(A, b) ≈ b' * (Matrix(A) \ b)
    end

end
