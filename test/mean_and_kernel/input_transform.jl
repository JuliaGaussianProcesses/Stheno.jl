using Stheno: ITMean, ITKernel, LhsITCross, RhsITCross, ITCross

@testset "input_transform" begin

    # Test ITMean.
    let
        rng, N, D = MersenneTwister(123456), 10, 2
        μ, f, x = OneMean(), abs2, randn(rng, N)
        μf = ITMean(μ, f)

        @test μf(x[1]) == (μ ∘ f)(x[1])
        differentiable_mean_function_tests(rng, μf, x)
    end

    # Test ITKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = ITKernel(k, f)

        @test kf(x0[1], x1[1]) == k(f(x0[1]), f(x1[1]))
        differentiable_kernel_tests(rng, kf, x0, x1, x2; rtol=1e-9, atol=1e-9)
    end

    # Test LhsITCross.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = LhsITCross(k, f)

        @test kf(x0[1], x1[1]) == k(f(x0[1]), x1[1])
        differentiable_cross_kernel_tests(rng, kf, x0, x1, x2)
    end

    # Test RhsITCross
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = RhsITCross(k, f)

        @test kf(x0[1], x1[1]) == k(x0[1], f(x1[1]))
        differentiable_cross_kernel_tests(rng, kf, x0, x1, x2)
    end

    # Test ITCross
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f, f′ = EQ(), abs2, sin
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = ITCross(k, f, f′)

        @test kf(x0[1], x1[1]) == k(f(x0[1]), f′(x1[1]))
        differentiable_cross_kernel_tests(rng, kf, x0, x1, x2)
    end
end
