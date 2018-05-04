using Stheno: CompositeMean, CompositeCrossKernel, CompositeKernel, LhsCross, RhsCross,
    OuterCross, OuterKernel

@testset "compose" begin

    # Test composite mean functionality.
    let
        rng, μ1, μ2 = MersenneTwister(123456), ConstantMean(1.0), ZeroMean{Float64}()
        N, D = 100, 2
        x, X = randn(rng, N), randn(rng, N, D)
        @test mean(CompositeMean(+, μ1, μ2), x) == ones(N)
        @test mean(CompositeMean(+, μ1, μ2), X) == ones(N)
        @test mean(CompositeMean(*, μ1, μ2), x) == zeros(N)
        @test mean(CompositeMean(*, μ1, μ2), X) == zeros(N)
    end

    # Test composite cov functionality.
    let
        rng, k, f = MersenneTwister(123456), EQ(), sin
        N, N′, D = 100, 105, 2
        X, X′ = randn(rng, N, D), randn(N′, D)
        @test xcov(CompositeKernel(f, k), X, X′) == map(f, xcov(k, X, X′))
        @test typeof(cov(CompositeKernel(*, k, k), X)) <: LazyPDMat
        @test cov(CompositeKernel(*, k, k), X) == map(*, cov(k, X), cov(k, X))
        @test xcov(CompositeKernel(*, k, k), X, X′) == map(*, xcov(k, X, X′), xcov(k, X, X′))
        _generic_kernel_tests(k, X, X′)
    end

    # Test composite xcov functionality.
    let
        rng, k, f = MersenneTwister(123456), EQ(), sin
        N, N′, D = 100, 105, 2
        X, X′ = randn(rng, N, D), randn(N′, D)
        @test xcov(CompositeCrossKernel(f, k), X, X′) == map(f, xcov(k, X, X′))
        @test xcov(CompositeCrossKernel(+, k, k), X, X′) == xcov(k, X, X′) + xcov(k, X, X′)
    end

    # Test LhsCross, RhsCross, OuterCross and OuterKernel functionality.
    let
        rng, k, N, N′, D = MersenneTwister(123456), EQ(), 5, 6, 2
        f, f′ = ConstantMean(1.0), ConstantMean(2.0)
        X, X′ = randn(rng, N, D), randn(rng, N′, D)
        @test xcov(LhsCross(f, k), X, X′) == Diagonal(mean(f, X)) * xcov(k, X, X′)
        @test xcov(RhsCross(k, f), X, X′) == xcov(k, X, X′) * Diagonal(mean(f, X′))
        @test xcov(OuterCross(f, k, f′), X, X′) ==
            Diagonal(mean(f, X)) * xcov(k, X, X′) * Diagonal(mean(f′, X′))
        @test xcov(OuterKernel(f, k), X, X′) ==
            Diagonal(mean(f, X)) * xcov(k, X, X′) * Diagonal(mean(f, X′))
        @test cov(OuterKernel(f, k), X) == Xt_A_X(cov(k, X), Diagonal(mean(f, X)))
        _generic_kernel_tests(OuterKernel(f, k), X, X′)
    end

    # Test mean function composition.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        X = randn(rng, N, D)
        μ, μ′ = ConstantMean(randn(rng)), CustomMean(x->randn(rng) .* sin.(x))

        # Test conversion and promotion.
        c = randn(rng)
        @test convert(MeanFunction, c) == ConstantMean(c)
        @test mean(convert(MeanFunction, sin), X) == mean(CustomMean(x->sin.(x)), X)

        # # Test addition.
        # @test μ + μ′ == CompositeMean(+, μ, μ′)
        # @test mean(μ + μ′)(X) == mean(μ)(X) + mean(μ′)(X)
        # @test μ + 5 == CompositeMean(+, μ, ConstantMean(5))
        # @test 2.34 + μ′ == CompositeMean(+, 2.34, μ′)

        # # Test multiplication.
        # @test μ * μ′ == CompositeMean(*, μ, μ′)
        # @test mean(μ * μ′)(X) == mean(μ)(X) .* mean(μ′)(X)
        # @test μ * 4.32 == CompositeMean(*, μ, ConstantMean(4.32))
        # @test 4.23 * μ′ == CompositeMean(*, ConstantMean(4.23), μ′)
    end

    # import Stheno: LhsOp, RhsOp
    # @test !isstationary(LhsOp{typeof(+), typeof(sin), EQ})
    # @test sin + EQ() == sin + EQ()
    # @test sin + EQ() != cos + EQ()
    # @test sin + EQ() != sin + RQ(1.0)
    # @test (sin + EQ())(5.0, 4.0) == sin(5.0) + EQ()(5.0, 4.0)
    # @test (cos * RQ(1.0))(3.3, 6.7) == cos(3.3) * RQ(1.0)(3.3, 6.7)
end
