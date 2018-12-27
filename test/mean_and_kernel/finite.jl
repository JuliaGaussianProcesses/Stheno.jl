using Stheno: FiniteMean, OneMean, AM, AV, pairwise

@testset "finite" begin

    # Tests for FiniteMean.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        μ, X, x = OneMean(), ColsAreObs(randn(rng, D, N)), randn(rng, N)
        μX, μx = FiniteMean(μ, X), FiniteMean(μ, x)

        # Recusion depth 1.
        for (μ′, X′) in zip([μX, μx], [X, x])
            # @test length(μ′) == N
            # @test eachindex(μ′) == 1:N
            @test map(μ′, :) == map(μ, X′)
        end
    end

    # Tests for FiniteKernel.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        k, X, x = EQ(), ColsAreObs(randn(rng, D, N)), randn(rng, N)
        kX, kx = FiniteKernel(k, X), FiniteKernel(k, x)

        for (k′, X′) in zip((kX, kx), (X, x))

            # Check for correctness relative to base kernel.
            # @test size(k′) == (N, N)
            # @test size(k′, 1) == N
            # @test size(k′, 2) == N
            # @test eachindex(k′) == 1:N
            @test pairwise(k′, :) == pairwise(k, X′)
            @test pairwise(k′, :) ≈ pairwise(k′, :, :)
        end
    end

    # Tests for LhsFiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 4, 5, 2
        k, X, X′ = EQ(), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        x, x′ = randn(rng, N), randn(rng, N′)
        k′, kx = LhsFiniteCrossKernel(k, X), LhsFiniteCrossKernel(k, x)

        # @test size(k′) == (N, Inf)
        # @test size(k′, 1) == N
        # @test size(k′, 2) == Inf
        # @test eachindex(k′, 1) == 1:N
        @test pairwise(k′, :, X′) ≈ pairwise(k, X, X′)
    end

    # Tests for RhsFiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 4, 5, 2
        X, Y, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        x, y, x′ = randn(rng, N), randn(rng, N), randn(rng, N′)
        k = EQ()
        k′, kx = RhsFiniteCrossKernel(k, X′), RhsFiniteCrossKernel(k, x′)

        # @test size(k′) == (Inf, N′)
        # @test size(k′, 1) == Inf
        # @test size(k′, 2) == N′
        # @test eachindex(k′, 2) == 1:N′
        @test pairwise(k′, X, :) ≈ pairwise(k, X, X′)
    end

    # Tests for FiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 7, 2
        x, x′ = randn(rng, N), randn(rng, N′)
        k, X, X′ = EQ(), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        k′, kx = FiniteCrossKernel(k, X, X′), FiniteCrossKernel(k, x, x′)

        # @test size(k′) == (N, N′)
        # @test size(k′, 1) == N
        # @test size(k′, 2) == N′
        # @test eachindex(k′, 1) == 1:N
        # @test eachindex(k′, 2) == 1:N′
        @test pairwise(k′, :, :) == pairwise(k, X, X′)
    end
end
