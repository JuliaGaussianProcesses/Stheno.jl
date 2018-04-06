@testset "finite" begin

    # Tests for FiniteKernel.
    let N = 5, rng = MersenneTwister(123456), D = 2

        # Construct kernels.
        Xt = randn(rng, D, N)
        X = Xt'
        k = FiniteKernel(EQ(), X)

        # Check for correctness relative to base kernel.
        @test k(1, 1) == k.k(view(X, 1, :), view(X, 1, :))
        @test k(N, N) == k.k(view(X, N, :), view(X, N, : ))
        @test cov(k) == cov(k.k, X)

        # Test utility.
        @test size(k) == (N, N)
        @test size(k, 1) == N && size(k, 2) == N && size(k, 3) == 1
        @test isfinite(k) == true
        @test isstationary(k) == false
    end

    # Tests for FiniteCrossKernel.
    let N = 5, N′ = 6, rng = MersenneTwister(123456), D = 2

    end
end
