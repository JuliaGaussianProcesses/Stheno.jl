using Stheno: ITMean, ITKernel, LhsITCross, RhsITCross, ITCross

@testset "input_transform" begin

    # Test ITMean.
    let
        rng, N, D = MersenneTwister(123456), 10, 2
        μ, f, x = OneMean(), abs2, randn(rng, N)
        μf = ITMean(μ, f)

        @test μf(x[1]) == (μ ∘ f)(x[1])
        mean_function_tests(μf, x)
    end

    # Test ITKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = ITKernel(k, f)

        @test kf(x0[1], x1[1]) == k(f(x0[1]), f(x1[1]))
        kernel_tests(kf, x0, x1, x2, 1e-6)
    end

    # Test LhsITCross.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = LhsITCross(k, f)

        @test kf(x0[1], x1[1]) == k(f(x0[1]), x1[1])
        cross_kernel_tests(kf, x0, x1, x2)
    end

    # Test RhsITCross
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = RhsITCross(k, f)

        @test kf(x0[1], x1[1]) == k(x0[1], f(x1[1]))
        cross_kernel_tests(kf, x0, x1, x2)
    end

    # Test ITCross
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f, f′ = EQ(), abs2, sin
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = ITCross(k, f, f′)

        @test kf(x0[1], x1[1]) == k(f(x0[1]), f′(x1[1]))
        cross_kernel_tests(kf, x0, x1, x2)
    end

    # # Test convenience code.
    # let
    #     rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
    #     m, k = ConstantMean(randn(rng)), EQ()
    #     X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
    #     m1, k1 = pick_dims(m, 1), pick_dims(k, 1)

    #     @test m1(X0[1]) == m(X0[1][1])
    #     @test k1(X0[1], X0[2]) == k(X0[1][1], X0[2][1])
    #     mean_function_tests(m1, X0)
    #     kernel_tests(k1, X0, X1, X2)

    #     x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
    #     mean_function_tests(periodic(m, 0.1), x0)
    #     kernel_tests(periodic(k, 0.1), x0, x1, x2)
    # end
end
