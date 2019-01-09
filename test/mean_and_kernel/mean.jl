using Stheno: CustomMean, ZeroMean, OneMean, EmpiricalMean

@testset "mean" begin

    # Test CustomMean function.
    let
        rng, N, D = MersenneTwister(123456), 11, 2
        X, x = ColsAreObs(randn(rng, D, N)), randn(rng, N)
        foo_mean = x->sum(abs2, x)
        f = CustomMean(foo_mean)

        @test f(X[1]) == foo_mean(X[1])

        mean_function_tests(f, x)
        differentiable_mean_function_tests(f, randn(rng, N), x)

        mean_function_tests(f, X)
        differentiable_mean_function_tests(f, randn(rng, N), X)
    end

    # Test ZeroMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = ColsAreObs(randn(rng, D, P)), randn(rng, P)
        f = ZeroMean{Float64}()

        @test f(randn(rng)) === zero(Float64)

        mean_function_tests(f, x)
        differentiable_mean_function_tests(f, randn(rng, P), x)

        mean_function_tests(f, X)
        differentiable_mean_function_tests(f, randn(rng, P), X)
    end

    # Test OneMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = ColsAreObs(randn(rng, D, P)), randn(rng, P)
        c = randn(rng)
        f = OneMean()

        @test f(randn(rng)) == 1

        mean_function_tests(f, x)
        differentiable_mean_function_tests(f, randn(rng, P), x)

        mean_function_tests(f, X)
        differentiable_mean_function_tests(f, randn(rng, P), X)
    end

    # Test EmpiricalMean.
    let
        rng, N, D = MersenneTwister(123456), 11, 2
        x = 1:N
        m = randn(rng, N)
        μ = EmpiricalMean(m)

        @test map(μ, :) == m
        @test AbstractVector(μ) == m
    end
end
