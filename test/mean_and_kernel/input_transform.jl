using Stheno: ITMean, ITKernel, LhsITCross, RhsITCross, ITCross, OneMean
using Stheno: EQ, Exp, Linear, Noise, PerEQ

@testset "input_transform" begin

    # Test ITMean.
    let
        rng, N, D = MersenneTwister(123456), 10, 2
        μ, f, x = OneMean(), abs2, randn(rng, N)
        μf = ITMean(μ, f)

        @test ew(μf, x) == ew(μ, map(f, x))
        differentiable_mean_function_tests(rng, μf, x)
    end

    # Test ITKernel.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = ITKernel(k, f)

        @test ew(kf, x0, x1) == ew(k, map(f, x0), map(f, x1))
        differentiable_kernel_tests(rng, kf, x0, x1, x2; rtol=1e-9, atol=1e-9)
    end

    # Test LhsITCross.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = LhsITCross(k, f)

        @test ew(kf, x0, x1) == ew(k, map(f, x0), x1)
        differentiable_cross_kernel_tests(rng, kf, x0, x1, x2)
    end

    # Test RhsITCross
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f = EQ(), abs2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = RhsITCross(k, f)

        @test ew(kf, x0, x1) == ew(k, x0, map(f, x1))
        differentiable_cross_kernel_tests(rng, kf, x0, x1, x2)
    end

    # Test ITCross
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        k, f, f′ = EQ(), abs2, sin
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        kf = ITCross(k, f, f′)

        @test ew(kf, x0, x1) == ew(k, map(f, x0), map(f′, x1))
        differentiable_cross_kernel_tests(rng, kf, x0, x1, x2)
    end
end
