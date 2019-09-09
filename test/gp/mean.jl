using Stheno: CustomMean, ZeroMean, OneMean, ConstMean

@timedtestset "mean" begin
    @timedtestset "CustomMean" begin
        rng, N, D = MersenneTwister(123456), 11, 2
        x = randn(rng, N)
        foo_mean = x->sum(abs2, x)
        f = CustomMean(foo_mean)

        @test ew(f, x) == map(foo_mean, x)
        differentiable_mean_function_tests(f, randn(rng, N), x)
    end
    @timedtestset "ZeroMean" begin
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = ColsAreObs(randn(rng, D, P)), randn(rng, P)
        f = ZeroMean{Float64}()

        for x in [x, X]
            @test ew(f, x) == zeros(size(x))
            differentiable_mean_function_tests(f, randn(rng, P), x)
        end
    end
    @timedtestset "OneMean" begin
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = ColsAreObs(randn(rng, D, P)), randn(rng, P)
        f = OneMean()

        for x in [x, X]
            @test ew(f, x) == ones(size(x))
            differentiable_mean_function_tests(f, randn(rng, P), x)
        end
    end
    @timedtestset "ConstMean" begin
        rng, D, N = MersenneTwister(123456), 5, 3
        X, x, c = ColsAreObs(randn(rng, D, N)), randn(rng, N), randn(rng)
        m = ConstMean(c)

        for x in [x, X]
            @test ew(m, x) == fill(c, N)
            differentiable_mean_function_tests(m, randn(rng, N), x)
        end
    end
    @timedtestset "(is)zero" begin
        @test zero(ZeroMean()) == ZeroMean()
        @test zero(OneMean()) == ZeroMean()
        @test iszero(ZeroMean()) == true
        @test iszero(OneMean()) == false
    end
end
