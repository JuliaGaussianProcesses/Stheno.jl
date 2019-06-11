using Random
using Stheno: ZeroMean, OneMean, ZeroKernel, OneKernel, CustomMean, pw, ConstKernel
using Stheno: EQ, Exp, Linear, Noise, PerEQ

@testset "algebra" begin

    @testset "MeanFunction" begin
        rng = MersenneTwister(123456)
        x, α = randn(rng, 3), randn(rng)

        # MeanFunction addition.
        @test ew(α + CustomMean(sin), x) == α .+ ew(CustomMean(sin), x)
        @test ew(CustomMean(cos) + α, x) == ew(CustomMean(cos), x) .+ α
        @test ew(CustomMean(sin) + CustomMean(cos), x) == map(sin, x) + map(cos, x)

        # Special cases of addition.
        @test ZeroMean() + ZeroMean() === ZeroMean()
        @test ZeroMean() + OneMean() === OneMean()
        @test OneMean() + ZeroMean() === OneMean()

        # MeanFunction multiplication.
        @test ew(α * CustomMean(sin), x) == α .* map(sin, x)
        @test ew(CustomMean(cos) * α, x) == map(cos, x) .* α
        @test ew(CustomMean(sin) * CustomMean(cos), x) == sin.(x) .* cos.(x)

        # Special cases of multiplication.
        @test ZeroMean() * ZeroMean() === ZeroMean()
        @test ZeroMean() * CustomMean(sin) === ZeroMean()
        @test CustomMean(cos) * ZeroMean() === ZeroMean()
    end

    @testset "Kernel" begin
        rng, N, N′ = MersenneTwister(123456), 10, 11
        α, x, x′ = randn(rng), randn(rng, N), randn(rng, N′)

        # Kernel addition.
        @test pw(α + EQ(), x, x′) == α .+ pw(EQ(), x, x′)
        @test pw(EQ() + α, x, x′) == pw(EQ(), x, x′) .+ α
        @test pw(EQ() + OneKernel(), x, x′) == pw(EQ(), x, x′) .+ pw(OneKernel(), x, x′)

        # Adding zero to kernels.
        @test ZeroKernel() + ZeroKernel() === ZeroKernel()
        @test ZeroKernel() + EQ() === EQ()
        @test EQ() + ZeroKernel() === EQ()

        # Multiplying kernels by constants.
        @test pw(α * EQ(), x, x′) == α .* pw(EQ(), x, x′)
        @test pw(EQ() * α, x, x′) == pw(EQ(), x, x′) .* α
        @test pw(EQ() * Linear(), x, x′) == pw(EQ(), x, x′) .* pw(Linear(), x, x′)

        # Sum of `ConstKernel`s isa ConstKernel
        @test ConstKernel(5) + ConstKernel(4) isa ConstKernel
        @test pw(ConstKernel(5) + ConstKernel(4), x, x′) ==
            pw(ConstKernel(5), x, x′) .+ pw(ConstKernel(4), x, x′)

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
        @test pw(ConstKernel(5) * ConstKernel(4), x, x′) ==
            pw(ConstKernel(5), x, x′) .* pw(ConstKernel(4), x, x′)
    end
end
