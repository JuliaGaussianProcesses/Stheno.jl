@testset "generic" begin

    # Test unary_colwise.
    let
        rng, P, D = MersenneTwister(123456), 3, 4
        X, x = randn(rng, D, P), RowVector(randn(rng, P))
        foo = x->sum(abs2, x)

        @test unary_colwise_fallback(foo, X) == [foo(X[:, 1]), foo(X[:, 2]), foo(X[:, 3])]
        @test unary_colwise_fallback(foo, x) == [foo(x[1]), foo(x[2]), foo(x[3])]

        @test unary_colwise(foo, x) == unary_colwise_fallback(foo, x)
        @test unary_colwise(foo, X) == unary_colwise_fallback(foo, X)
    end

    # Test binary_colwise and pairwise.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, X′ = randn(rng, D, P), randn(rng, D, Q)
        x, x′ = RowVector(randn(rng, P)), RowVector(randn(rng, Q))
        foo = (x, x′)->sum(abs2, x - x′)
        Stheno.isstationary(::typeof(foo)) = true

        @test binary_colwise_fallback(foo, X, X) ==
            [foo(X[:, 1], X[:, 1]), foo(X[:, 2], X[:, 2]), foo(X[:, 3], X[:, 3])]
        @test binary_colwise_fallback(foo, x, x) ==
            [foo(x[1], x[1]), foo(x[2], x[2]), foo(x[3], x[3])]

        @test binary_colwise(foo, X′, X′) == binary_colwise_fallback(foo, X′, X′)
        @test binary_colwise(foo, x′, x′) == binary_colwise_fallback(foo, x′, x′)
        @test binary_colwise(foo, X) == binary_colwise(foo, X, X)
        @test binary_colwise(foo, x) == binary_colwise(foo, x, x)

        @test pairwise_fallback(foo, X, X′) ==
            [foo(X[:, 1], X′[:, 1]) foo(X[:, 1], X′[:, 2]);
             foo(X[:, 2], X′[:, 1]) foo(X[:, 2], X′[:, 2]);
             foo(X[:, 3], X′[:, 1]) foo(X[:, 3], X′[:, 2])]
        @test pairwise_fallback(foo, x, x′) ==
            [foo(x[1], x′[1]) foo(x[1], x′[2]);
             foo(x[2], x′[1]) foo(x[2], x′[2]);
             foo(x[3], x′[1]) foo(x[3], x′[2])]

        @test pairwise(foo, X, X′) == pairwise_fallback(foo, X, X′)
        @test pairwise(foo, x, x′) == pairwise_fallback(foo, x, x′)
        @test pairwise(foo, X) == pairwise(foo, X, X)
        @test pairwise(foo, x) == pairwise(foo, x, x)

        # Check that stationarity doesn't effect outcome.
        bar = (x, x′)->sum(abs2, x - x′)
        Stheno.isstationary(::typeof(bar)) = false
        @test binary_colwise(foo, x) == binary_colwise(bar, x)
        @test pairwise(foo, x) == pairwise(bar, x)
    end
end
