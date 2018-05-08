using Stheno: CustomMean, ZeroMean, ConstantMean, nobs, ndims

@testset "mean" begin

    # Test CustomMean function.
    let
        rng, N, D = MersenneTwister(123456), 11, 2
        X, x = randn(rng, D, N), randn(rng, N)
        foo = x->sum(abs, x)
        f = CustomMean(foo)
        @test f(X[:, 1]) == foo(X[:, 1])
        unary_obswise_tests(foo, X)

        @test isinf(length(f))
        @test size(f, 1) == Inf && size(f, 2) == 1
        @test size(f) == (Inf,)
    end

    # Test ZeroMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = randn(rng, D, P), randn(rng, P)
        f = ZeroMean{Float64}()

        @test f(randn(rng)) === zero(Float64)
        @test f == ZeroMean{Float32}()

        unary_obswise_tests(f, x)
        unary_obswise_tests(f, X)

        @test isinf(length(f))
        @test isinf(size(f, 1)) && size(f, 2) == 1
        @test size(f) == (Inf,)
    end

    # Test Const.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = randn(rng, D, P), randn(rng, P)
        c = randn(rng)
        f = ConstantMean(c)

        @test ConstantMean(1.0) == ConstantMean(1)
        @test ConstantMean(1.0) ≠ ConstantMean(2)
        @test f(randn(rng)) == c
        unary_obswise_tests(f, x)
        unary_obswise_tests(f, X)

        @test isinf(length(f))
        @test isinf(size(f, 1)) && size(f, 2) == 1
        @test size(f) == (Inf,)
    end
end
