using Stheno: CompositeMean, CompositeKernel, CompositeCrossKernel, ConstantMean,
    ZeroMean, LhsCross, RhsCross, OuterCross, OuterKernel, MeanFunction, Kernel

@testset "compose" begin

    # Test CompositeMean.
    let
        rng, N, D = MersenneTwister(123456), 100, 2
        μ1, μ2 = ConstantMean(1.0), ZeroMean{Float64}()
        x, X = RowVector(randn(rng, N)), randn(rng, D, N)
        ν1, ν2 = CompositeMean(+, μ1, μ2), CompositeMean(*, μ1, μ2)
        @test ν1(X[:, 1]) == μ1(X[:, 1]) + μ2(X[:, 1])
        @test ν2(X[:, 3]) == μ1(X[:, 3]) .* μ2(X[:, 3])
        @test length(ν1) == Inf && length(ν2) == Inf

        mean_function_tests(ν1, x)
        mean_function_tests(ν1, X)
        mean_function_tests(ν2, x)
        mean_function_tests(ν2, X)
    end

    # Test CompositeKernel and CompositeCrossKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)

        # Define base kernels and do composition.
        k1, k2 = EQ(), Linear(randn(rng))
        ν1, ν2 = CompositeKernel(+, k1, k2), CompositeCrossKernel(*, k1, k2)
        @test ν1(X0[:, 1], X1[:, 2]) == k1(X0[:, 1], X1[:, 2]) + k2(X0[:, 1], X1[:, 2])
        @test ν2(X0[:, 1], X1[:, 2]) == k1(X0[:, 1], X1[:, 2]) .* k2(X0[:, 1], X1[:, 2])
        @test size(ν1, 1) == Inf && size(ν1, 2) == Inf
        @test !isstationary(ν1) && !isstationary(ν2)
        @test isstationary(CompositeKernel(+, k1, k1)) == true

        cross_kernel_tests(ν2, x0, x1, x2)
        cross_kernel_tests(ν2, X0, X1, X2)
        kernel_tests(ν1, x0, x1, x2)
        kernel_tests(ν1, X0, X1, X2)
    end

    # Test LhsCross functionality.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)

        f, k = x->sum(abs2, x), EQ()
        ν = LhsCross(f, k)

        @test ν(X0[:, 1], X0[:, 2]) == f(X0[:, 1]) * k(X0[:, 1], X0[:, 2])
        @test size(ν, 1) == Inf && size(ν, 2) == Inf

        cross_kernel_tests(ν, x0, x1, x2)
        cross_kernel_tests(ν, X0, X1, X2)
    end

    # Test RhsCross functionality.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)

        f, k = x->sum(abs2, x), EQ()
        ν = RhsCross(k, f)

        @test ν(X0[:, 1], X0[:, 2]) == k(X0[:, 1], X0[:, 2]) * f(X0[:, 2])
        @test size(ν, 1) == Inf && size(ν, 2) == Inf

        cross_kernel_tests(ν, x0, x1, x2)
        cross_kernel_tests(ν, X0, X1, X2)
    end

    # Test OuterCross.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)

        f, k = x->sum(abs2, x), EQ()
        ν = OuterCross(f, k)

        @test ν(X0[:, 1], X0[:, 2]) == f(X0[:, 1]) * k(X0[:, 1], X0[:, 2]) * f(X0[:, 2])
        @test size(ν, 1) == Inf && size(ν, 2) == Inf

        cross_kernel_tests(ν, x0, x1, x2)
        cross_kernel_tests(ν, X0, X1, X2)
    end

    # Test OuterKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)

        f, k = x->sum(abs2, x), EQ()
        ν = OuterKernel(f, k)

        @test ν(X0[:, 1], X0[:, 2]) == f(X0[:, 1]) * k(X0[:, 1], X0[:, 2]) * f(X0[:, 2])
        @test size(ν, 1) == Inf && size(ν, 2) == Inf

        kernel_tests(ν, x0, x1, x2)
        kernel_tests(ν, X0, X1, X2)
    end

    # Test mean function composition.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        X, s = randn(rng, N, D), randn(rng)
        μ, μ′ = ConstantMean(randn(rng)), CustomMean(x->vec(s .* sum(sin.(x), 2)))

        # Test conversion and promotion.
        c = randn(rng)
        @test convert(MeanFunction, c) == ConstantMean(c)
        @test promote_rule(ConstantMean, Real) == MeanFunction
        @test promote(ConstantMean(c), c) == (ConstantMean(c), ConstantMean(c))

        # Test addition.
        @test μ + μ′ == CompositeMean(+, μ, μ′)
        @test map(μ + μ′, X) == map(μ, X) + map(μ′, X)
        @test μ + 5 == CompositeMean(+, μ, ConstantMean(5))
        @test 2.34 + μ′ == CompositeMean(+, ConstantMean(2.34), μ′)

        # Test multiplication.
        @test μ * μ′ == CompositeMean(*, μ, μ′)
        @test map(μ * μ′, X) == map(μ, X) .* map(μ′, X)
        @test μ * 4.32 == CompositeMean(*, μ, ConstantMean(4.32))
        @test 4.23 * μ′ == CompositeMean(*, ConstantMean(4.23), μ′)
    end

    # Test kernel composition.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        X, s = randn(rng, N, D), randn(rng)
        k, k′ = EQ(), Linear(1)

        # Test conversion and promotion.
        @test convert(Kernel, s) == ConstantKernel(s)
        @test promote_rule(typeof(k), typeof(s)) == Kernel
        @test promote(k, s) == (k, ConstantKernel(s))
        @test promote(s, k) == (ConstantKernel(s), k)

        # Test addition.
        @test k + k′ == CompositeKernel(+, k, k′)
        @test k + s == CompositeKernel(+, k, ConstantKernel(s))
        @test s + k′ == CompositeKernel(+, ConstantKernel(s), k′)

        # Test multiplication.
        @test k * k′ == CompositeKernel(*, k, k′)
        @test k′ * s == CompositeKernel(*, k′, ConstantKernel(s))
        @test s * k == CompositeKernel(*, ConstantKernel(s), k)
    end
end
