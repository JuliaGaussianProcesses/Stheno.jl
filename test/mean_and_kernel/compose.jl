using Stheno: UnaryMean, BinaryMean, BinaryKernel, BinaryCrossKernel, ConstantMean,
    LhsCross, RhsCross, OuterCross, OuterKernel, map, pairwise

@testset "compose" begin

    # Test UnaryMean.
    let
        rng, N = MersenneTwister(123456), 100
        μ, f, x = CustomMean(sin), exp, randn(rng, N)
        ν = UnaryMean(f, μ)
        @test ν(x[1]) == exp(μ(x[1]))
        mean_function_tests(ν, x)
    end

    # Test BinaryMean.
    let
        rng, N = MersenneTwister(123456), 100
        μ1, μ2 = CustomMean(sin), CustomMean(cos)
        f, x = exp, randn(rng, N)
        ν1, ν2 = BinaryMean(+, μ1, μ2), BinaryMean(*, μ1, μ2)
        @test ν1(x[1]) == μ1(x[1]) .+ μ2(x[1])
        @test ν2(x[3]) == μ1(x[3]) .* μ2(x[3])

        mean_function_tests(ν1, x)
        mean_function_tests(ν2, x)
    end

    # # Test CompositeKernel and CompositeCrossKernel.
    # let
    #     rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
    #     x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
    #     X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

    #     # Define base kernels and do composition.
    #     k1, k2 = EQ(), Linear(randn(rng))
    #     ν1, ν2 = CompositeKernel(+, k1, k2), CompositeCrossKernel(*, k1, k2)
    #     @test ν1(X0[1], X1[2]) == k1(X0[1], X1[2]) + k2(X0[1], X1[2])
    #     @test ν2(X0[1], X1[2]) == k1(X0[1], X1[2]) .* k2(X0[1], X1[2])
    #     @test size(ν1, 1) == Inf && size(ν1, 2) == Inf
    #     @test size(ν2, 1) == Inf && size(ν2, 2) == Inf
    #     @test !isstationary(ν1) && !isstationary(ν2)
    #     @test isstationary(CompositeKernel(+, k1, k1)) == true

    #     xb0, xb1, xb2 = BlockData([x0, x1]), BlockData([x1, x0]), BlockData([x2, x2])
    #     XB0, XB1, XB2 = BlockData([X0, X1]), BlockData([X1, X0]), BlockData([X2, X2])

    #     cross_kernel_tests(ν2, x0, x1, x2)
    #     cross_kernel_tests(ν2, X0, X1, X2)
    #     cross_kernel_tests(ν2, xb0, xb1, xb2)
    #     cross_kernel_tests(ν2, XB0, XB1, XB2)

    #     kernel_tests(ν1, x0, x1, x2)
    #     kernel_tests(ν1, X0, X1, X2)
    #     kernel_tests(ν1, xb0, xb1, xb2)
    #     kernel_tests(ν1, XB0, XB1, XB2)
    # end

    # # Test LhsCross functionality.
    # let
    #     rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
    #     x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
    #     X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

    #     f, k = x->sum(abs2, x), EQ()
    #     ν = LhsCross(f, k)

    #     @test ν(X0[1], X0[2]) == f(X0[1]) * k(X0[1], X0[2])
    #     @test size(ν, 1) == Inf && size(ν, 2) == Inf

    #     xb0, xb1, xb2 = BlockData([x0, x1]), BlockData([x1, x0]), BlockData([x2, x2])
    #     XB0, XB1, XB2 = BlockData([X0, X1]), BlockData([X1, X0]), BlockData([X2, X2])
    #     cross_kernel_tests(ν, x0, x1, x2)
    #     cross_kernel_tests(ν, X0, X1, X2)
    #     cross_kernel_tests(ν, xb0, xb1, xb2)
    #     cross_kernel_tests(ν, XB0, XB1, XB2)
    # end

    # # Test RhsCross functionality.
    # let
    #     rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
    #     x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
    #     X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

    #     f, k = x->sum(abs2, x), EQ()
    #     ν = RhsCross(k, f)

    #     @test ν(X0[1], X0[2]) == k(X0[1], X0[2]) * f(X0[2])
    #     @test size(ν, 1) == Inf && size(ν, 2) == Inf

    #     xb0, xb1, xb2 = BlockData([x0, x1]), BlockData([x1, x0]), BlockData([x2, x2])
    #     XB0, XB1, XB2 = BlockData([X0, X1]), BlockData([X1, X0]), BlockData([X2, X2])
    #     cross_kernel_tests(ν, x0, x1, x2)
    #     cross_kernel_tests(ν, X0, X1, X2)
    #     cross_kernel_tests(ν, xb0, xb1, xb2)
    #     cross_kernel_tests(ν, XB0, XB1, XB2)
    # end

    # # Test OuterCross.
    # let
    #     rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
    #     x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
    #     X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

    #     f, k = x->sum(abs2, x), EQ()
    #     ν = OuterCross(f, k)

    #     @test ν(X0[1], X0[2]) == f(X0[1]) * k(X0[1], X0[2]) * f(X0[2])
    #     @test size(ν, 1) == Inf && size(ν, 2) == Inf

    #     xb0, xb1, xb2 = BlockData([x0, x1]), BlockData([x1, x0]), BlockData([x2, x2])
    #     XB0, XB1, XB2 = BlockData([X0, X1]), BlockData([X1, X0]), BlockData([X2, X2])
    #     cross_kernel_tests(ν, x0, x1, x2)
    #     cross_kernel_tests(ν, X0, X1, X2)
    #     cross_kernel_tests(ν, xb0, xb1, xb2)
    #     cross_kernel_tests(ν, XB0, XB1, XB2)
    # end

    # # Test OuterKernel.
    # let
    #     rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
    #     x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
    #     X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

    #     f, k = x->sum(abs2, x), EQ()
    #     ν = OuterKernel(f, k)

    #     @test ν(X0[1], X0[2]) == f(X0[1]) * k(X0[1], X0[2]) * f(X0[2])
    #     @test size(ν, 1) == Inf && size(ν, 2) == Inf

    #     xb0, xb1, xb2 = BlockData([x0, x1]), BlockData([x1, x0]), BlockData([x2, x2])
    #     XB0, XB1, XB2 = BlockData([X0, X1]), BlockData([X1, X0]), BlockData([X2, X2])
    #     kernel_tests(ν, x0, x1, x2)
    #     kernel_tests(ν, X0, X1, X2)
    #     cross_kernel_tests(ν, xb0, xb1, xb2)
    #     cross_kernel_tests(ν, XB0, XB1, XB2)
    # end

    # # Test mean function composition.
    # let
    #     rng, N, D = MersenneTwister(123456), 5, 2
    #     X, s = ColsAreObs(randn(rng, D, N)), randn(rng)
    #     μ, μ′ = ConstantMean(randn(rng)), CustomMean(x->sum(s .* x))

    #     # Test conversion and promotion.
    #     c = randn(rng)
    #     @test convert(MeanFunction, c) == ConstantMean(c)
    #     @test promote_rule(ConstantMean, Real) == MeanFunction
    #     @test promote(ConstantMean(c), c) == (ConstantMean(c), ConstantMean(c))

    #     # Test addition.
    #     @test μ + μ′ == CompositeMean(+, μ, μ′)
    #     @test map(μ + μ′, X) == map(μ, X) + map(μ′, X)
    #     @test map(μ + 5, X) == map(μ, X) .+ 5
    #     @test 2.34 + μ′ == CompositeMean(+, ConstantMean(2.34), μ′)

    #     # Test multiplication.
    #     @test μ * μ′ == CompositeMean(*, μ, μ′)
    #     @test map(μ * μ′, X) == map(μ, X) .* map(μ′, X)
    #     @test map(μ * 4.32, X) == 4.32 .* map(μ, X)
    #     @test 4.23 * μ′ == CompositeMean(*, ConstantMean(4.23), μ′)
    # end

    # # Test kernel composition.
    # let
    #     rng, N, D = MersenneTwister(123456), 5, 2
    #     X, s = ColsAreObs(randn(rng, N, D)), randn(rng)
    #     k, k′ = EQ(), Linear(1)

    #     # Test conversion and promotion.
    #     @test convert(Kernel, s) == ConstantKernel(s)
    #     @test promote_rule(typeof(k), typeof(s)) == Kernel
    #     @test promote(k, s) == (k, ConstantKernel(s))
    #     @test promote(s, k) == (ConstantKernel(s), k)

    #     # Test addition.
    #     @test k + k′ == CompositeKernel(+, k, k′)
    #     @test k + s == CompositeKernel(+, k, ConstantKernel(s))
    #     @test s + k′ == CompositeKernel(+, ConstantKernel(s), k′)

    #     # Test multiplication.
    #     @test k * k′ == CompositeKernel(*, k, k′)
    #     @test k′ * s == CompositeKernel(*, k′, ConstantKernel(s))
    #     @test s * k == CompositeKernel(*, ConstantKernel(s), k)
    # end
end
