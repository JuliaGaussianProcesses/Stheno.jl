using Random
using Stheno: FiniteMean, FiniteKernel, FiniteCrossKernel, LhsFiniteCrossKernel,
    RhsFiniteCrossKernel, ZeroMean, OneMean, ZeroKernel, OneKernel

@testset "algebra" begin

    let
        rng = MersenneTwister(123456)
        x, α = randn(rng), randn(rng)

        # MeanFunction addition.
        @test (α + CustomMean(sin))(x) == α + CustomMean(sin)(x)
        @test (CustomMean(cos) + α)(x) == CustomMean(cos)(x) + α
        @test (CustomMean(sin) + CustomMean(cos))(x) == sin(x) + cos(x)

        # Special cases of addition.
        @test ZeroMean() + ZeroMean() === ZeroMean()
        @test ZeroMean() + OneMean() === OneMean()
        @test OneMean() + ZeroMean() === OneMean()

        # MeanFunction multiplication.
        @test (α * CustomMean(sin))(x) == α * CustomMean(sin)(x)
        @test (CustomMean(cos) * α)(x) == CustomMean(cos)(x) * α
        @test (CustomMean(sin) * CustomMean(cos))(x) == sin(x) * cos(x)

        # Special cases of multiplication.
        @test ZeroMean() * ZeroMean() === ZeroMean()
        @test ZeroMean() * CustomMean(sin) === ZeroMean()
        @test CustomMean(cos) * ZeroMean() === ZeroMean()
    end

    let
        rng = MersenneTwister(123456)
        α, x, x′ = randn(rng), randn(rng), randn(rng)

        # Kernel addition.
        @test (α + EQ())(x, x′) == α + EQ()(x, x′)
        @test (EQ() + α)(x, x′) == EQ()(x, x′) + α
        @test (EQ() + OneKernel())(x, x′) == EQ()(x, x′) + OneKernel()(x, x′)

        # Adding zero to kernels.
        @test ZeroKernel() + ZeroKernel() === ZeroKernel()
        @test ZeroKernel() + EQ() === EQ()
        @test EQ() + ZeroKernel() === EQ()

        # Multiplying kernels by constants.
        @test (α * EQ())(x, x′) == α * EQ()(x, x′)
        @test (EQ() * α)(x, x′) == EQ()(x, x′) * α
        @test (EQ() * Linear())(x, x′) == EQ()(x, x′) * Linear()(x, x′)

        # Multiplying kernels by zero.
        @test ZeroKernel() * ZeroKernel() === ZeroKernel()
        @test ZeroKernel() * EQ() === ZeroKernel()
        @test EQ() * ZeroKernel() === ZeroKernel()

        # Multiplying kernels by one.
        @test OneKernel() * OneKernel() === OneKernel()
        @test OneKernel() * EQ() === EQ()
        @test EQ() * OneKernel() === EQ()
    end

end

# using Stheno: FiniteZeroMean, FiniteZeroKernel, LhsFiniteZeroCrossKernel,
#     RhsFiniteZeroCrossKernel, FiniteZeroCrossKernel
# using Base: OneTo

# @testset "zero" begin

#     let
#         μ1, μ2 = ZeroMean{Float64}(), FiniteZeroMean(OneTo(10))
#         μ3, μ4 = BlockMean([μ1, μ1]), BlockMean([μ2, μ2])
#         k1, k2 = ZeroKernel{Float64}(), FiniteZeroCrossKernel(OneTo(4), OneTo(5))
#         k3, k4 = LhsFiniteZeroCrossKernel(OneTo(3)), RhsFiniteZeroCrossKernel(OneTo(5))
#         k5, k6 = ZeroKernel{Float64}(), FiniteZeroKernel(OneTo(7))

#         k7, k8 = BlockCrossKernel([k1 k1; k1 k1]), BlockCrossKernel([k2 k2; k2 k2])
#         k9, k10 = BlockCrossKernel([k3 k3; k3 k3]), BlockCrossKernel([k4 k4; k4 k4])

#         k11 = BlockKernel([k5, k5], [k5 k5; k5 k5])
#         k12 = BlockKernel([k6, k6], [k6 k6; k6 k6])

#         # Check addition of zero elements.
#         for x in [μ1, μ2, μ3, μ4, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12]
#             @test zero(x) == x
#             @test x + x == zero(x)
#             @test x * x == x
#         end

#         # Check (element-wise) multiplication of zero elements.
#         @test_throws AssertionError μ2 * FiniteZeroMean(randn(11))
#         @test_throws AssertionError k2 * FiniteZeroCrossKernel(randn(5), randn(1))
#         @test_throws AssertionError k3 * LhsFiniteZeroCrossKernel(randn(4))
#         @test_throws AssertionError k4 * RhsFiniteZeroCrossKernel(randn(2))
#         @test_throws AssertionError k6 * FiniteZeroKernel(randn(8))

#         @test μ1 * ConstantMean(3.0) === μ1
#         @test ConstantMean(2.0) * μ1 === μ1

#         @test μ2 * FiniteMean(μ1, randn(10)) === μ2
#         @test FiniteMean(μ1, randn(10)) * μ2 === μ2

#         @test k1 * EQ() === k1
#         @test EQ() * k1 === k1
#     end
# end
