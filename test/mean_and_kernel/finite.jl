using Stheno: FiniteMean, OneMean, AM, AV, pairwise, FiniteKernel, FiniteCrossKernel
using Stheno: EQ, Exp, Linear, Noise, PerEQ

@testset "finite" begin

    # Tests for FiniteMean.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        μ, X, x = OneMean(), ColsAreObs(randn(rng, D, N)), randn(rng, N)
        μX, μx = FiniteMean(μ, X), FiniteMean(μ, x)

        # Recusion depth 1.
        for (μ′, X′) in zip([μX, μx], [X, x])
            @test map(μ′, :) == map(μ, X′)
        end

        # Ensure that inputs remain differentiable.
        adjoint_test(x->map(FiniteMean(OneMean(), x), :), randn(rng, N), x)
        adjoint_test(X->map(FiniteMean(OneMean(), ColsAreObs(X)), :), randn(rng, N), X.X)
    end

    # Tests for FiniteKernel.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        k, X, x = EQ(), ColsAreObs(randn(rng, D, N)), randn(rng, N)
        kX, kx = FiniteKernel(k, X), FiniteKernel(k, x)
        ȳ = randn(rng, N, N)

        for (k′, X′) in zip((kX, kx), (X, x))

            # Check for correctness relative to base kernel.
            @test pairwise(k′, :) == pairwise(k, X′)
            @test pairwise(k′, :) ≈ pairwise(k′, :, :)
        end

        # Ensure that inputs remain differentiable.
        adjoint_test(x->pw(FiniteKernel(EQ(), x), :), ȳ, x)
        adjoint_test(x->pw(FiniteKernel(EQ(), x), :, :), ȳ, x)
        adjoint_test(X->pw(FiniteKernel(EQ(), X), :), ȳ, X)
        adjoint_test(X->pw(FiniteKernel(EQ(), X), :, :), ȳ, X)
    end

    # Tests for FiniteCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 7, 2
        x, x′ = randn(rng, N), randn(rng, N′)
        k, X, X′ = EQ(), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        k′, kx = FiniteCrossKernel(k, X, X′), FiniteCrossKernel(k, x, x′)
        ȳ = randn(rng, N, N′)

        @test pairwise(kx, :, :) == pairwise(k, x, x′)
        adjoint_test(x->pw(FiniteCrossKernel(EQ(), x, x′), :, :), ȳ, x)
        adjoint_test(x′->pw(FiniteCrossKernel(EQ(), x, x′), :, :), ȳ, x′)

        @test pairwise(k′, :, :) == pairwise(k, X, X′)
        adjoint_test(X->pw(FiniteCrossKernel(EQ(), X, X′), :, :), ȳ, X)
        adjoint_test(X′->pw(FiniteCrossKernel(EQ(), X, X′), :, :), ȳ, X′)
    end

    # # Tests for LhsFiniteCrossKernel.
    # let
    #     rng, N, N′, D = MersenneTwister(123456), 4, 5, 2
    #     k, X, X′ = EQ(), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
    #     x, x′ = randn(rng, N), randn(rng, N′)
    #     k′, kx = LhsFiniteCrossKernel(k, X), LhsFiniteCrossKernel(k, x)
    #     ȳ = randn(rng, N, N′)

    #     @test pairwise(kx, :, x′) == pairwise(k, x, x′)
    #     adjoint_test(x->pw(LhsFiniteCrossKernel(EQ(), x), :, x′), ȳ, x)
    #     adjoint_test(x′->pw(LhsFiniteCrossKernel(EQ(), x), :, x′), ȳ, x′)

    #     @test pairwise(k′, :, X′) == pairwise(k, X, X′)
    #     adjoint_test(X->pw(LhsFiniteCrossKernel(EQ(), X), :, X′), ȳ, X)
    #     adjoint_test(X′->pw(LhsFiniteCrossKernel(EQ(), X), :, X′), ȳ, X′)
    # end

    # # Tests for RhsFiniteCrossKernel.
    # let
    #     rng, N, N′, D = MersenneTwister(123456), 4, 5, 2
    #     X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
    #     x, x′ = randn(rng, N), randn(rng, N′)
    #     ȳ = randn(rng, N, N′)
    #     k = EQ()
    #     k′, kx = RhsFiniteCrossKernel(k, X′), RhsFiniteCrossKernel(k, x′)

    #     @test pairwise(kx, x, :) == pairwise(k, x, x′)
    #     adjoint_test(x->pw(RhsFiniteCrossKernel(k, x′), x, :), ȳ, x)
    #     adjoint_test(x′->pw(RhsFiniteCrossKernel(k, x′), x, :), ȳ, x′)


    #     @test pairwise(k′, X, :) == pairwise(k, X, X′)
    #     adjoint_test(X->pw(RhsFiniteCrossKernel(k, X′), X, :), ȳ, X)
    #     adjoint_test(X′->pw(RhsFiniteCrossKernel(k, X′), X, :), ȳ, X′)
    # end
end
