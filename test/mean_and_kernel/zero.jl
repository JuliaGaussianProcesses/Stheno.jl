using Stheno: FiniteZeroMean, FiniteZeroKernel, LhsFiniteZeroCrossKernel,
    RhsFiniteZeroCrossKernel, FiniteZeroCrossKernel
using Base: OneTo

@testset "zero" begin

    let
        μ1, μ2 = ZeroMean{Float64}(), FiniteZeroMean(OneTo(10))
        k1, k2 = ZeroKernel{Float64}(), FiniteZeroCrossKernel(OneTo(4), OneTo(5))
        k3, k4 = LhsFiniteZeroCrossKernel(OneTo(3)), RhsFiniteZeroCrossKernel(OneTo(5))
        k5, k6 = ZeroKernel{Float64}(), FiniteZeroKernel(OneTo(7))

        # Check addition of zero elements.
        for x in [μ1, μ2, k1, k2, k3, k4, k5, k6]
            @test zero(x) === x
            @test x + x === zero(x)
            @test x * x === x
        end

        # Check (element-wise) multiplication of zero elements.
        @test_throws AssertionError μ2 * FiniteZeroMean(randn(11))
        @test_throws AssertionError k2 * FiniteZeroCrossKernel(randn(5), randn(1))
        @test_throws AssertionError k3 * LhsFiniteZeroCrossKernel(randn(4))
        @test_throws AssertionError k4 * RhsFiniteZeroCrossKernel(randn(2))

        @show @which k6 * FiniteZeroKernel(randn(8))
        @test_throws AssertionError k6 * FiniteZeroKernel(randn(8))

        @test μ1 * ConstantMean(3.0) === μ1
        @test ConstantMean(2.0) * μ1 === μ1

        @test μ2 * FiniteMean(μ1, randn(10)) === μ2
        @test FiniteMean(μ1, randn(10)) * μ2 === μ2

        @test k1 * EQ() === k1
        @test EQ() * k1 === k1

    end


end
