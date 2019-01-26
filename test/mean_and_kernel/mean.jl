using Stheno: CustomMean, ZeroMean, OneMean

@testset "mean" begin

    # Test CustomMean function.
    let
        rng, N, D = MersenneTwister(123456), 11, 2
        X, x = ColsAreObs(randn(rng, D, N)), randn(rng, N)
        foo_mean = x->sum(abs2, x)
        f = CustomMean(foo_mean)

        for x in [x, X]
            @test map(f, x) == map(foo_mean, x)
            differentiable_mean_function_tests(f, randn(rng, N), x)
        end
    end

    # Test ZeroMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = ColsAreObs(randn(rng, D, P)), randn(rng, P)
        f = ZeroMean{Float64}()

        for x in [x, X]
            @test map(f, x) == zeros(size(x))
            differentiable_mean_function_tests(f, randn(rng, P), x)
        end
    end

    # Test OneMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = ColsAreObs(randn(rng, D, P)), randn(rng, P)
        c = randn(rng)
        f = OneMean()

        for x in [x, X]
            @test map(f, x) == ones(size(x))
            differentiable_mean_function_tests(f, randn(rng, P), x)
        end
    end
end
