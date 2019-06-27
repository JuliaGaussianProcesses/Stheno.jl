using Stheno: UnaryMean, BinaryMean, BinaryKernel, BinaryCrossKernel, OneMean,
    LhsCross, RhsCross, OuterCross, OuterKernel, pairwise, CustomMean, ZeroMean,
    ZeroKernel, OneKernel, pw
using Stheno: EQ, Exp, Linear, Noise, PerEQ

@testset "compose" begin

    @testset "UnaryMean" begin
        rng, N = MersenneTwister(123456), 100
        μ, f, ȳ, x = CustomMean(sin), exp, randn(rng, N), randn(rng, N)
        ν = UnaryMean(f, μ)
        @test ew(ν, x) == map(exp, ew(μ, x))
        mean_function_tests(ν, x)

        differentiable_mean_function_tests(UnaryMean(exp, ZeroMean()), ȳ, x)
        differentiable_mean_function_tests(UnaryMean(exp, OneMean()), ȳ, x)
    end

    @testset "BinaryMean" begin
        rng, N = MersenneTwister(123456), 100
        μ1, μ2 = CustomMean(sin), CustomMean(cos)
        f, ȳ, x = exp, randn(rng, N), randn(rng, N)
        ν1, ν2 = BinaryMean(+, μ1, μ2), BinaryMean(*, μ1, μ2)

        let
            m1, m2 = ew(μ1, x), ew(μ2, x)
            @test ew(ν1, x) == m1 .+ m2
            @test ew(ν2, x) == m1 .* m2
        end

        differentiable_mean_function_tests(ν1, ȳ, x)
        differentiable_mean_function_tests(ν2, ȳ, x)

        differentiable_mean_function_tests(BinaryMean(+, ZeroMean(), OneMean()), ȳ, x)
        differentiable_mean_function_tests(BinaryMean(+, μ1, OneMean()), ȳ, x)
        differentiable_mean_function_tests(BinaryMean(*, ZeroMean(), μ2), ȳ, x)
    end

    @testset "BinaryKernel" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)

        k = EQ()
        ν = BinaryKernel(+, k, k)

        # Self-consistency testing.
        kernel_tests(ν, x0, x1, x2)

        # Absolute correctness testing.
        @test ew(ν, x0, x1) == ew(k, x0, x1) .+ ew(k, x0, x1)
        @test pairwise(ν, x0, x2) == pairwise(k, x0, x2) .+ pairwise(k, x0, x2)

        @test ew(ν, x0) == ew(k, x0) .+ ew(k, x0)
        @test pairwise(ν, x0) == pairwise(k, x0) .+ pairwise(k, x0)

        let
            k = BinaryKernel(+, ZeroKernel(), OneKernel())
            differentiable_kernel_tests(rng, k, x0, x1, x2)
        end
        differentiable_kernel_tests(rng, BinaryKernel(*, ZeroKernel(), EQ()), x0, x1, x2)
        differentiable_kernel_tests(rng, BinaryKernel(+, EQ(), OneKernel()), x0, x1, x2)

        # # Some compositions of stationary kernels are stationary.
        # stationary_kernel_tests(
        #     ν,
        #     range(-5.0, step=1, length=N),
        #     range(-4.0, step=1, length=N),
        #     range(-5.0, step=2, length=N),
        #     range(-3.0, step=1, length=N′),
        #     range(-2.0, step=2, length=N′),
        # )
    end

    @testset "BinaryCrossKernel" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)

        k = EQ()
        ν = BinaryCrossKernel(+, k, k)

        # Self-consistency testing.
        differentiable_cross_kernel_tests(rng, ν, x0, x1, x2)

        # Absolute correctness testing.
        @test ew(ν, x0, x1) == ew(k, x0, x1) .+ ew(k, x0, x1)
        @test pairwise(ν, x0, x2) == pairwise(k, x0, x2) .+ pairwise(k, x0, x2)

        @test ew(ν, x0) == ew(k, x0) .+ ew(k, x0)
        @test pairwise(ν, x0) == pairwise(k, x0) .+ pairwise(k, x0)

        let
            k = BinaryCrossKernel(+, ZeroKernel(), OneKernel())
            differentiable_cross_kernel_tests(rng, k, x0, x1, x2)
        end
        let
            k = BinaryCrossKernel(*, ZeroKernel(), EQ())
            differentiable_cross_kernel_tests(rng, k, x0, x1, x2)
        end
        let
            k = BinaryCrossKernel(+, EQ(), OneKernel())
            differentiable_cross_kernel_tests(rng, k, x0, x1, x2)
        end
    end

    @testset "LhsCross" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)

        f, k = CustomMean(abs2), EQ()
        ν = LhsCross(f, k)

        @test pw(ν, x0, x1) == ew(f, x0) .* pw(k, x0, x1)

        differentiable_cross_kernel_tests(rng, ν, x0, x1, x2)
    end

    @testset "RhsCross" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)

        k, f = EQ(), CustomMean(abs2)
        ν = RhsCross(k, f)

        @test pw(ν, x0, x1) == pw(k, x0, x1) .* ew(f, x1)'
        differentiable_cross_kernel_tests(rng, ν, x0, x1, x2)
    end

    @testset "OuterCross" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)

        k, f = EQ(), CustomMean(abs2)
        ν = OuterCross(f, k)

        @test pw(ν, x0, x1) == ew(f, x0) .* pw(k, x0, x1) .* ew(f, x1)'
        differentiable_cross_kernel_tests(rng, ν, x0, x1, x2)
    end

    @testset "OuterKernel" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)

        k, f = EQ(), CustomMean(abs2)
        ν = OuterKernel(f, k)

        @test pw(ν, x0, x1) == ew(f, x0) .* pw(k, x0, x1) .* ew(f, x1)'
        differentiable_kernel_tests(rng, ν, x0, x1, x2)
    end
end
