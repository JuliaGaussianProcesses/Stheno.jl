using Stheno: FiniteMean, ConstantMean, AM, AV, pairwise

@testset "finite" begin

    # Tests for FiniteMean.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        μ, X, x = ConstantMean(1.5), ColsAreObs(randn(rng, D, N)), randn(rng, N)
        μX, μx = FiniteMean(μ, X), FiniteMean(μ, x)

        # Recusion depth 1.
        for (μ′, X′) in zip([μX, μx], [X, x])
            @test length(μ′) == N
            @test eachindex(μ′) == 1:N
            @test AbstractVector(μ′) == map(μ′, eachindex(μ′))
            @test AbstractVector(μ′) == map(μ, X′)
            mean_function_tests(μ′, eachindex(μ′))
            mean_function_tests(μ′, 1:N-1)
            mean_function_tests(μ′, BlockData([1:N-1, eachindex(μ′)]))

            # Recursion depth 2.
            r = 1:N-1
            μ2 = FiniteMean(μ′, r)
            @test length(μ2) == N - 1
            @test eachindex(μ2) == 1:N-1
            @test AbstractVector(μ′)[r] == AbstractVector(μ2)
            @test AbstractVector(μ′)[r] == map(μ2, eachindex(μ2))
            mean_function_tests(μ2, eachindex(μ2))
            mean_function_tests(μ2, 1:N-2)
            mean_function_tests(μ2, 1:N-2)
        end

        @test map(μX, 1:N-1) == map(μ, X[1:N-1])
        @test map(μX, 2:N) == map(μ, X[2:N])
        @test map(μX, 2:N-2) == map(μ, X[2:N-2])

        @test map(μx, 1:N-1) == map(μ, x[1:N-1])
        @test map(μx, 2:N) == map(μ, x[2:N])
        @test map(μx, 2:N-2) == map(μ, x[2:N-2])

        show(IOBuffer(), μX)
    end

    # Tests for FiniteKernel.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        k, X, x = EQ(), ColsAreObs(randn(rng, D, N)), randn(rng, N)
        kX, kx = FiniteKernel(k, X), FiniteKernel(k, x)

        for (k′, X′) in zip((kX, kx), (X, x))

            # Check for correctness relative to base kernel.
            @test size(k′) == (N, N)
            @test size(k′, 1) == N
            @test size(k′, 2) == N
            @test isstationary(k′) == false
            @test eachindex(k′) == 1:N
            @test AM(k′) == pairwise(k′, eachindex(k′))
            @test Matrix(AM(k′)) == pairwise(k′, eachindex(k′), eachindex(k′))
            @test AM(k′) ≈ pairwise(k, X′)
            @test AM(k′) ≈ pairwise(k, X′)

            @test pairwise(k′, 1:N-1, 2:N) ≈ pairwise(k, X′[1:N-1], X′[2:N])
            @test pairwise(k′, eachindex(k′), 1:N-1) ≈ pairwise(k, X′, X′[1:N-1])
            @test pairwise(k′, 2:N, eachindex(k′)) ≈ pairwise(k, X′[2:N], X′)

            kernel_tests(k′, 1:N-1, 2:N, eachindex(k′))
            d0 = BlockData([1:N-1, 2:N])
            d1 = BlockData([2:N, 1:N-1])
            d2 = BlockData([eachindex(k′)])
            kernel_tests(k′, d0, d1, d2)

            # Recursion depth 2.
            r = 1:N-1
            k2 = FiniteKernel(k′, r)
            @test size(k2) == (N-1, N-1)
            @test eachindex(k2) == r
            @test AM(k2) == pairwise(k′, r)

            # Known issue: offsetting doesn't really work.
            binary_map_tests(k2, r, r)
            pairwise_tests(k′, r, r)
        end

        show(IOBuffer(), kX)
    end

    # Tests for LhsFiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 4, 5, 2
        k, X, X′ = EQ(), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        x, x′ = randn(rng, N), randn(rng, N′)
        k′, kx = LhsFiniteCrossKernel(k, X), LhsFiniteCrossKernel(k, x)

        @test size(k′) == (N, Inf)
        @test size(k′, 1) == N
        @test size(k′, 2) == Inf
        @test isstationary(k′) == false
        @test eachindex(k′, 1) == 1:N
        @test pairwise(k′, eachindex(k′, 1), X′) ≈ pairwise(k, X, X′)
        @test pairwise(k′, 1:N-1, X′) ≈ pairwise(k, X[1:N-1], X′)

        cross_kernel_tests(k′, eachindex(k′, 1), X, X′)
        cross_kernel_tests(kx, eachindex(kx, 1), x, x′)
        cross_kernel_tests(k′, eachindex(k′, 1), BlockData([X]), BlockData([X′]))
        cross_kernel_tests(kx, eachindex(kx, 1), BlockData([x]), BlockData([x′]))

        show(IOBuffer(), k′)
    end

    # Tests for RhsFiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 4, 5, 2
        X, Y, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        x, y, x′ = randn(rng, N), randn(rng, N), randn(rng, N′)
        k = EQ()
        k′, kx = RhsFiniteCrossKernel(k, X′), RhsFiniteCrossKernel(k, x′)

        @test size(k′) == (Inf, N′)
        @test size(k′, 1) == Inf
        @test size(k′, 2) == N′
        @test isstationary(k′) == false
        @test eachindex(k′, 2) == 1:N′
        @test pairwise(k′, X, eachindex(k′, 2)) ≈ pairwise(k, X, X′)
        @test pairwise(k′, X, 1:N′-1) ≈ pairwise(k, X, X′[1:N′-1])

        d1, d2 = 1:N, eachindex(k′, 2)
        cross_kernel_tests(k′, X, d1, d2)
        cross_kernel_tests(kx, x, d1, d2)
        cross_kernel_tests(k′, BlockData([X]), BlockData([d1]), BlockData([d2]))
        cross_kernel_tests(kx, BlockData([x]), BlockData([d1]), BlockData([d2]))

        show(IOBuffer(), k′)
    end

    # Tests for FiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 7, 2
        x, x′ = randn(rng, N), randn(rng, N′)
        k, X, X′ = EQ(), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        k′, kx = FiniteCrossKernel(k, X, X′), FiniteCrossKernel(k, x, x′)

        @test size(k′) == (N, N′)
        @test size(k′, 1) == N
        @test size(k′, 2) == N′
        @test isstationary(k′) == false
        @test eachindex(k′, 1) == 1:N
        @test eachindex(k′, 2) == 1:N′
        @test AM(k′) == pairwise(k′, eachindex(k′, 1), eachindex(k′, 2))
        @test AM(k′) ≈ pairwise(k, X, X′)
        @test pairwise(k′, 1:N-1, 2:N) ≈ pairwise(k, X[1:N-1], X′[2:N])
        @test pairwise(k′, eachindex(k′, 1), 1:N-1) ≈ pairwise(k, X, X′[1:N-1])
        @test pairwise(k′, 2:N, eachindex(k′, 2)) ≈ pairwise(k, X[2:N], X′)

        cross_kernel_tests(k′, 1:N, 1:N, 1:N′)
        cross_kernel_tests(kx, 1:N, 1:N, 1:N′)
        cross_kernel_tests(k′, BlockData([1:N]), BlockData([1:N]), BlockData([1:N′]))
        cross_kernel_tests(kx, BlockData([1:N]), BlockData([1:N]), BlockData([1:N′]))

        # Recursion depth == 2.
        r, c = 1:N-1, 1:N′-2
        k2 = FiniteCrossKernel(k′, r, c)
        @test size(k2) == (length(r), length(c))
        @test AM(k2) == pairwise(k2, r, c)
        @test AM(k2) == pairwise(k′, r, c)
        @test pairwise(k2, 2:N-1, c) == pairwise(k′, 2:N-1, c)

        cross_kernel_tests(k2, r, r, c)
        cross_kernel_tests(k2, BlockData([r]), BlockData([r]), BlockData([c]))

        show(IOBuffer(), k′)
    end
end
