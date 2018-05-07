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

    # Test Zero.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, X′ = randn(rng, D, P), randn(rng, D, Q), randn(rng, D, P)
        x, x′ = RowVector(randn(rng, P)), RowVector(randn(rng, Q))
        Xr, xr = randn(rng, D, P), RowVector(randn(rng, P))
        f1, f2 = Stheno.Zero{1, Float64}(), Stheno.Zero{2, Float64}()

        @test f1(randn(rng)) === zero(Float64)
        @test f2(randn(rng), randn(rng)) === zero(Float64)
        @test f1 == Stheno.Zero{1, Float32}()
        @test f1 ≠ Stheno.Zero{2, Float32}()

        unary_colwise_tests(f1, x)
        unary_colwise_tests(f1, X)
        binary_colwise_tests(f2, x, xr)
        binary_colwise_tests(f2, X, Xr)
        pairwise_tests(f2, x, x′)
        pairwise_tests(f2, X, X′)

        @test isinf(length(f1))
        @test isinf(size(f1, 1)) && size(f1, 2) == 1
        @test size(f1) == (Inf,)

        @test isinf(size(f2, 1)) && isinf(size(f2, 2))
        @test size(f2) == (Inf, Inf)
    end

    # Test Const.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, X′ = randn(rng, D, P), randn(rng, D, Q), randn(rng, D, P)
        x, x′ = RowVector(randn(rng, P)), RowVector(randn(rng, Q))
        Xr, xr = randn(rng, D, P), RowVector(randn(rng, P))
        c = randn(rng)
        f1, f2 = Stheno.Const(1, c), Stheno.Const(2, c)

        @test f1 == f1
        @test f2 == f2
        @test f1 ≠ f2
        @test f1(randn(rng)) == c
        @test f2(randn(rng), randn(rng)) == c
        unary_colwise_tests(f1, x)
        unary_colwise_tests(f1, X)
        binary_colwise_tests(f2, x, xr)
        binary_colwise_tests(f2, X, Xr)
        pairwise_tests(f2, x, x′)
        pairwise_tests(f2, X, X′)

        @test isinf(length(f1))
        @test isinf(size(f1, 1)) && size(f1, 2) == 1
        @test size(f1) == (Inf,)

        @test isinf(size(f2, 1)) && isinf(size(f2, 2))
        @test size(f2) == (Inf, Inf)
    end
end
