@testset "mean" begin

    # Test CustomMean function.
    let
        rng, N, D = MersenneTwister(123456), 11, 2
        X, x = randn(rng, D, N), RowVector(randn(rng, N))
        foo = x->sum(abs, x)
        f = CustomMean(foo)
        @test f(X[:, 1]) == foo(X[:, 1])
        unary_colwise_tests(foo, X)

        @test isinf(length(f))
        @test size(f, 1) == Inf && size(f, 2) == 1
        @test size(f) == (Inf,)
    end
end
