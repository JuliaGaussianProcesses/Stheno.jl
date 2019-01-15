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

        # Sum of `ConstKernel`s isa ConstKernel
        @test ConstKernel(5) + ConstKernel(4) isa ConstKernel
        @test (ConstKernel(5) + ConstKernel(4))(x, x′) ==
            ConstKernel(5)(x, x′) + ConstKernel(4)(x, x′)

        # Multiplying kernels by zero.
        @test ZeroKernel() * ZeroKernel() === ZeroKernel()
        @test ZeroKernel() * EQ() === ZeroKernel()
        @test EQ() * ZeroKernel() === ZeroKernel()

        # Multiplying kernels by one.
        @test OneKernel() * OneKernel() === OneKernel()
        @test OneKernel() * EQ() === EQ()
        @test EQ() * OneKernel() === EQ()

        # Product of `ConstKernel`s isa ConstKernel
        @test ConstKernel(5) * ConstKernel(4) isa ConstKernel
        @test (ConstKernel(5) * ConstKernel(4))(x, x′) ==
            ConstKernel(5)(x, x′) * ConstKernel(4)(x, x′)
    end
end
