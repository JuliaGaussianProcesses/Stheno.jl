using Stheno: ConstantMean, ITMean, ITKernel, map, pairwise

@testset "input_transform" begin

    # Test ITMean.
    let
        rng, N, D = MersenneTwister(123456), 10, 2
        μ, f, X = ConstantMean(randn(rng)), x->sum(abs2, x), ColsAreObs(randn(rng, D, N))
        μf = ITMean(μ, f)

        @test ITMean(μ, identity) == μ
        @test length(μf) == Inf
        @test μf(X[1]) == (μ ∘ f)(X[1])
        mean_function_tests(μf, X)
    end

    # Test ITKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), x->sum(abs2, x)
        X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        kf = ITKernel(k, f)

        @test ITKernel(k, identity) == k
        @test size(kf, 1) == Inf && size(kf, 2) == Inf

        @test kf(X0[1], X1[1]) == k(f(X0[1]), f(X1[1]))
        kernel_tests(kf, X0, X1, X2, 1e-6)
    end

    # Test convenience code.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        m, k = ConstantMean(randn(rng)), EQ()
        X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        m1, k1 = pick_dims(m, 1), pick_dims(k, 1)

        @test m1(X0[1]) == m(X0[1][1])
        @test k1(X0[1], X0[2]) == k(X0[1][1], X0[2][1])
        mean_function_tests(m1, X0)
        kernel_tests(k1, X0, X1, X2)

        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        mean_function_tests(periodic(m, 0.1), x0)
        kernel_tests(periodic(k, 0.1), x0, x1, x2)
    end
end
