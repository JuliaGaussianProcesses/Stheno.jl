@testset "finite" begin

    # Tests for FiniteMean.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        μ, X = ConstantMean(1.5), randn(rng, N, D)
        μ̂ = FiniteMean(μ, X)

        # Recusion depth 1.
        @test size(μ̂, 1) == N
        @test size(μ̂, 2) == 1
        @test length(μ̂) == N
        @test eachindex(μ̂) == 1:N
        @test μ̂(1) == 1.5
        @test μ̂(N) == 1.5
        @test mean(μ̂) == mean(μ̂, eachindex(μ̂))
        @test mean(μ̂) == mean(μ, X)
        @test mean(μ̂, 1:N-1) == mean(μ, X[1:N-1, :])
        @test mean(μ̂, 2:N) == mean(μ, X[2:N, :])
        @test mean(μ̂, 2:N-2) == mean(μ, X[2:N-2, :])

        # Recursion depth 2.
        r = 1:N-1
        μ2 = FiniteMean(μ̂, r)
        @test size(μ2, 1) == N - 1
        @test length(μ2) == N - 1
        @test eachindex(μ2) == 1:N-1
        @test mean(μ̂)[r] == mean(μ2)
        @test mean(μ̂)[r] == mean(μ2, eachindex(μ2))
    end

    # Tests for FiniteKernel.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        k, X = EQ(), randn(rng, N, D)
        k′ = FiniteKernel(k, X)

        # Check for correctness relative to base kernel.
        @test size(k′) == (N, N)
        @test size(k′, 1) == N
        @test size(k′, 2) == N
        @test isstationary(k′) == false
        @test eachindex(k′) == 1:N
        @test k′(3, 4) == k(X[3, :], X[4, :])
        @test k′(1, 1) == k(X[1, :], X[1, :])
        @test k′(N, N) == k(X[N, :], X[N, :])
        @test cov(k′) == cov(k′, eachindex(k′))
        @test Matrix(cov(k′)) == xcov(k′, eachindex(k′), eachindex(k′))
        @test cov(k′) == cov(k, X)
        @test xcov(k′) == xcov(k, X)
        @test xcov(k′, 1:N-1, 2:N) ≈ xcov(k, X[1:N-1, :], X[2:N, :])
        @test xcov(k′, eachindex(k′), 1:N-1) ≈ xcov(k, X, X[1:N-1, :])
        @test xcov(k′, 2:N, eachindex(k′)) ≈ xcov(k, X[2:N, :], X)

        # Recursion depth 2.
        r = 1:N-1
        k2 = FiniteKernel(k′, r)
        @test size(k2) == (N-1, N-1)
        @test eachindex(k2) == r
        @test xcov(k2) == xcov(k′, r)
    end

    # Tests for LhsFiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 4, 5, 2
        k, X, X′ = EQ(), randn(rng, N, D), randn(rng, N′, D)
        k′ = LhsFiniteCrossKernel(k, X)

        @test size(k′) == (N, Inf)
        @test size(k′, 1) == N
        @test size(k′, 2) == Inf
        @test isstationary(k′) == false
        @test eachindex(k′, 1) == 1:N
        @test k′(3, X′[4, :]) == k(X[3, :], X′[4, :])
        @test xcov(k′, eachindex(k′, 1), X′) == xcov(k, X, X′)
        @test xcov(k′, 1:N-1, X′) == xcov(k, X[1:N-1, :], X′)
    end

    # Tests for RhsFiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 4, 5, 2
        k, X, X′ = EQ(), randn(rng, N, D), randn(rng, N′, D)
        k′ = RhsFiniteCrossKernel(k, X′)

        @test size(k′) == (Inf, N′)
        @test size(k′, 1) == Inf
        @test size(k′, 2) == N′
        @test isstationary(k′) == false
        @test eachindex(k′, 2) == 1:N′
        @test k′(X[3, :], 4) == k(X[3, :], X′[4, :])
        @test xcov(k′, X, eachindex(k′, 2)) == xcov(k, X, X′)
        @test xcov(k′, X, 1:N′-1) == xcov(k, X, X′[1:N′-1, :])
    end

    # Tests for FiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        k, X, X′ = EQ(), randn(rng, N, D), randn(rng, N′, D)
        k′ = FiniteCrossKernel(k, X, X′)

        @test size(k′) == (N, N′)
        @test size(k′, 1) == N
        @test size(k′, 2) == N′
        @test isstationary(k′) == false
        @test eachindex(k′, 1) == 1:N
        @test eachindex(k′, 2) == 1:N′
        @test k′(3, 4) == k(X[3, :], X′[4, :])
        @test k′(1, 1) == k(X[1, :], X′[1, :])
        @test k′(N, N′) == k(X[N, :], X′[N′, :])
        @test xcov(k′) == xcov(k′, eachindex(k′, 1), eachindex(k′, 2))
        @test xcov(k′) == xcov(k, X, X′)
        @test xcov(k′, 1:N-1, 2:N) == xcov(k, X[1:N-1, :], X′[2:N, :])
        @test xcov(k′, eachindex(k′, 1), 1:N-1) == xcov(k, X, X′[1:N-1, :])
        @test xcov(k′, 2:N, eachindex(k′, 2)) == xcov(k, X[2:N, :], X′)

        # Recursion depth == 2.
        r, c = 1:N-1, 1:N′-2
        k2 = FiniteCrossKernel(k′, r, c)
        @test size(k2) == (length(r), length(c))
        @test xcov(k2) == xcov(k2, r, c)
        @test xcov(k2) == xcov(k′, r, c)
        @test xcov(k2, 2:N-1, c) == xcov(k′, 2:N-1, c)
    end
end
