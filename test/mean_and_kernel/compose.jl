using Stheno: MapReduce, Const, Zero, LhsCross, RhsCross, Outer

@testset "compose" begin

    # Test unary MapReduce functionality.
    let
        rng, N, D, μ1, μ2 = MersenneTwister(123456), 100, 2, Const(1, 1.0), Zero{1, Float64}()
        x, X = RowVector(randn(rng, N)), randn(rng, D, N)
        ν1, ν2 = MapReduce(1, +, μ1, μ2), MapReduce(1, *, μ1, μ2)
        @test ν1(X[:, 1]) == μ1(X[:, 1]) + μ2(X[:, 1])
        @test ν2(X[:, 3]) == μ1(X[:, 3]) .* μ2(X[:, 3])

        unary_colwise_tests(ν1, x)
        unary_colwise_tests(ν1, X)
        unary_colwise_tests(ν2, x)
        unary_colwise_tests(ν2, X)
    end

    # Test binary MapReduce functionality.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x, X = randn(rng, D, N), RowVector(randn(rng, N))
        x′, X′ = randn(rng, D, N′), RowVector(randn(rng, N′))
        xr, Xr = randn(rng, D, N), RowVector(randn(rng, N))

        # Define base kernels and do composition.
        k1, k2 = EQ(), Linear(randn(rng))
        ν1, ν2 = MapReduce(2, +, k1, k2), MapReduce(2, *, k1, k2)
        @test ν1(X[:, 1], X′[:, 2]) == k1(X[:, 1], X′[:, 2]) + k2(X[:, 1], X′[:, 2])
        @test ν2(X[:, 1], X′[:, 2]) == k1(X[:, 1], X′[:, 2]) .* k2(X[:, 1], X′[:, 2])

        binary_colwise_tests(ν1, x, xr)
        binary_colwise_tests(ν1, X, Xr)
        binary_colwise_tests(ν2, x, xr)
        binary_colwise_tests(ν2, X, Xr)

        pairwise_tests(ν1, x, x′)
        pairwise_tests(ν1, X, X′)
        pairwise_tests(ν2, x, x′)
        pairwise_tests(ν2, X, X′)
    end

    # Test LhsCross functionality.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x, X = randn(rng, D, N), RowVector(randn(rng, N))
        x′, X′ = randn(rng, D, N′), RowVector(randn(rng, N′))
        xr, Xr = randn(rng, D, N), RowVector(randn(rng, N))

        f, k = x->sum(abs2, x), EQ()
        ν = LhsCross(f, k)

        @test ν(X[:, 1], X[:, 2]) == f(X[:, 1]) * k(X[:, 1], X[:, 2])

        binary_colwise_tests(ν, x, xr)
        binary_colwise_tests(ν, X, Xr)

        pairwise_tests(ν, x, x′)
        pairwise_tests(ν, X, X′)
    end

    # Test RhsCross functionality.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x, X = randn(rng, D, N), RowVector(randn(rng, N))
        x′, X′ = randn(rng, D, N′), RowVector(randn(rng, N′))
        xr, Xr = randn(rng, D, N), RowVector(randn(rng, N))

        f, k = x->sum(abs2, x), EQ()
        ν = RhsCross(k, f)

        @test ν(X[:, 1], X[:, 2]) == k(X[:, 1], X[:, 2]) * f(X[:, 2])

        binary_colwise_tests(ν, x, xr)
        binary_colwise_tests(ν, X, Xr)

        pairwise_tests(ν, x, x′)
        pairwise_tests(ν, X, X′)
    end

    # Test Outer.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x, X = randn(rng, D, N), RowVector(randn(rng, N))
        x′, X′ = randn(rng, D, N′), RowVector(randn(rng, N′))
        xr, Xr = randn(rng, D, N), RowVector(randn(rng, N))

        f, k = x->sum(abs2, x), EQ()
        ν = Outer(f, k)

        @test ν(X[:, 1], X[:, 2]) == f(X[:, 1]) * k(X[:, 1], X[:, 2]) * f(X[:, 2])

        binary_colwise_tests(ν, x, xr)
        binary_colwise_tests(ν, X, Xr)

        pairwise_tests(ν, x, x′)
        pairwise_tests(ν, X, X′)
    end

    # Test mean function composition.
    let
        rng, N, D = MersenneTwister(123456), 5, 2
        X, s = randn(rng, N, D), randn(rng)
        μ, μ′ = Const(1, randn(rng)), CustomMean(x->s * sum(sin, x))

        # Test conversion and promotion.
        c = randn(rng)
        @test convert(MeanFunction, c) == Const(1, c)
        @test promote(Const(1, c), c) == (Const(1, c), Const(1, c))

        # Test addition.
        @test μ + μ′ == MapReduce(1, +, μ, μ′)
        @test mean(μ + μ′, X) == mean(μ, X) + mean(μ′, X)
        @test μ + 5 == MapReduce(1, +, μ, Const(1, 5))
        @test 2.34 + μ′ == MapReduce(1, +, Const(1, 2.34), μ′)
        # @test mean(μ′ + sin, X) == mean(μ′, X) + CustomMean()

        # Test multiplication.
        @test μ * μ′ == MapReduce(1, *, μ, μ′)
        @test mean(μ * μ′, X) == mean(μ, X) .* mean(μ′, X)
        @test μ * 4.32 == MapReduce(1, *, μ, Const(1, 4.32))
        @test 4.23 * μ′ == MapReduce(1, *, Const(1, 4.23), μ′)
    end

    # # Test kernel composition.
    # let
    #     rng, N, D = MersenneTwister(123456), 5, 2
    #     X, s = randn(rng, N, D), randn(rng)
    #     k, k′ = EQ(), Linear(1)

    #     # Test conversion and promotion.
    #     @test convert(Kernel, s) == ConstantKernel(s)
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
