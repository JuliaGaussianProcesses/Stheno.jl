using Stheno: WoodburyLazyPDMat

@testset "woodbury" begin

    let
        rng, N, D = MersenneTwister(123456), 1000, 5
        X, Σ_, σ = randn(rng, D, N), randn(rng, D, D), exp(randn(rng))
        Σ = LazyPDMat(Σ_ * Σ_' + 1e-9I, 0)
        A = WoodburyLazyPDMat(X, Σ, σ)
        B, b = randn(N, 2), randn(N)

        # Test that conversion and indexing are consistent with one another.
        Am, Al = Matrix(A), LazyPDMat(A)
        @test size(A) == (N, N)
        @test Matrix(Al) == Am
        @test Am[2, 3] ≈ A[2, 3]
        @test Am[3, 2] ≈ A[3, 2]

        # Test basics of matrix multiplication.
        @test Matrix(A * B) ≈ Matrix(A) * B
        @test Vector(A * b) ≈ Matrix(A) * b

        # Test basics of linear systems solving.
        @test Matrix(A \ B) ≈ Matrix(A) \ B
        @test Vector(A \ b) ≈ Matrix(A) \ b

        @test Matrix(B' / A) ≈ B' / Matrix(A)
    end

end
