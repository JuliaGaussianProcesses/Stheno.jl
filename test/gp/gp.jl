using Stheno: OuterKernel, BinaryKernel, GPC, ZeroMean, OneMean, ZeroKernel
using Stheno: EQ, Exp, Linear, Noise, PerEQ

@testset "gp" begin

    # Check various ways to construct a GP do what you would expect.
    @testset "sugar" begin
        m = 5.1
        @test mean(GP(EQ(), GPC())) isa ZeroMean
        @test mean(GP(zero(m), EQ(), GPC())) isa ZeroMean
        @test mean(GP(one(m), EQ(), GPC())) isa OneMean

        x = randn(10)
        @test ew(mean(GP(sin, EQ(), GPC())), x) == sin.(x)
    end

    # Test the creation of indepenent GPs.
    @testset "independent GPs" begin
        rng = MersenneTwister(123456)

        # Specification for two independent GPs.
        gpc = GPC()
        μ1, μ2 = OneMean(), ZeroMean()
        k1, k2 = EQ(), Linear()
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)

        @test mean(f1) == μ1
        @test mean(f2) == μ2

        @test kernel(f1) == k1
        @test kernel(f2) == k2

        @test kernel(f1, f1) == k1
        @test kernel(f1, f2) == ZeroKernel()
        @test kernel(f2, f1) == ZeroKernel()
        @test kernel(f2, f2) == k2
    end
end
