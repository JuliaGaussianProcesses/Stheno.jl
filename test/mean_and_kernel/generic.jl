using Stheno: nobs, ndims, getobs

@testset "generic" begin

    # Test utility.
    let
        x, X = randn(5), randn(5, 3)
        @test nobs(x) == 5 && ndims(x) == 1 && getobs(x, 1) == x[1]
        @test nobs(X) == 3 && ndims(X) == 5 && getobs(X, 1) == X[:, 1]
    end

    # Test unary_obswise.
    let
        rng, P, D = MersenneTwister(123456), 3, 4
        X, x = randn(rng, D, P), randn(rng, P)
        foo = x->sum(abs2, x)

        @test unary_obswise_fallback(foo, X) == [foo(X[:, 1]), foo(X[:, 2]), foo(X[:, 3])]
        @test unary_obswise_fallback(foo, x) == [foo(x[1]), foo(x[2]), foo(x[3])]

        @test unary_obswise(foo, x) == unary_obswise_fallback(foo, x)
        @test unary_obswise(foo, X) == unary_obswise_fallback(foo, X)
    end

    # Test binary_obswise and pairwise.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, X′ = randn(rng, D, P), randn(rng, D, Q)
        x, x′ = randn(rng, P), randn(rng, Q)
        foo = (x, x′)->sum(abs2, x - x′)
        Stheno.isstationary(::typeof(foo)) = true

        @test binary_obswise_fallback(foo, X, X) ==
            [foo(X[:, 1], X[:, 1]), foo(X[:, 2], X[:, 2]), foo(X[:, 3], X[:, 3])]
        @test binary_obswise_fallback(foo, x, x) ==
            [foo(x[1], x[1]), foo(x[2], x[2]), foo(x[3], x[3])]

        @test binary_obswise(foo, X′, X′) == binary_obswise_fallback(foo, X′, X′)
        @test binary_obswise(foo, x′, x′) == binary_obswise_fallback(foo, x′, x′)
        @test binary_obswise(foo, X) == binary_obswise(foo, X, X)
        @test binary_obswise(foo, x) == binary_obswise(foo, x, x)

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
        @test binary_obswise(foo, x) == binary_obswise(bar, x)
        @test pairwise(foo, x) == pairwise(bar, x)
    end
end
