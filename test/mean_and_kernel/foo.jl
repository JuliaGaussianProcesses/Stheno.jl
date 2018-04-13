# This file is named as such because, for some reason, Julia is unable to load a file
# called compose.jl or composite.jl

@testset "compose" begin

    # Check that basic GP-independent functionality works.
    using Stheno: CompositeMean, CompositeKernel, CompositeCrossKernel
    let
        @test CompositeMean(sin, 5.0).f == sin
        @test CompositeMean(sin, 5.0).x == (5.0,)
        @test CompositeMean(sin, 3, 2).x == (3, 2)
        @test typeof(CompositeMean(cos, 5)) == CompositeMean{typeof(cos), 1}
        @test typeof(CompositeMean(sin, 5, 4)) == CompositeMean{typeof(sin), 2}
        @test CompositeMean(sin, cos)(5.0) == sin(cos(5.0))
        @test CompositeMean(+, sin, cos)(4.0) == sin(4.0) + cos(4.0)
    end

    # Check that mean-function functionality works.
    let
        rng, μ1, μ2 = MersenneTwister(123456), ConstantMean(1.0), ZeroMean{Float64}()
        N, D = 100, 2
        x, X = randn(rng, N), randn(rng, N, D)
        @test mean(CompositeMean(+, μ1, μ2), x) == ones(N)
        @test mean(CompositeMean(+, μ1, μ2), X) == ones(N)
        @test mean(CompositeMean(*, μ1, μ2), x) == zeros(N)
        @test mean(CompositeMean(*, μ1, μ2), X) == zeros(N)
    end

    # Check that cov functionality works.
    let
        rng, k, f = MersenneTwister(123456), EQ(), CustomMean(sin)
        N, D = 100, 2
        x, X = randn(rng, N), randn(rng, N, D)
        @test cov(CompositeKernel((f, k, f′)->f * k * f′, f, k, Adjoint(f)), X) ==
            mean(f, X) .* cov(EQ(), X) .* mean(f, X)'
    end

    # # Tests for UnaryComposite kernel.
    # using Stheno: UnaryComposite
    # @test isstationary(UnaryComposite(x->5x, EQ()))
    # @test !isstationary(UnaryComposite(x->5x, Linear(1)))     

    # # Tests for BinaryComposite kernel.
    # using Stheno: BinaryComposite
    # @test isstationary(BinaryComposite(+, EQ(), EQ()))
    # @test !isstationary(BinaryComposite(+, EQ(), Linear(1)))

    # # Test addition.
    # @test isstationary(EQ() + EQ())
    # @test !isstationary(EQ() + Linear(5))
    # @test isstationary(5 * EQ())
    # @test (EQ() + EQ())(5.0, 4.0) == EQ()(5.0, 4.0) + EQ()(5.0, 4.0)
    # @test (EQ() + 5.0)(3.0, 3.5) == EQ()(3.0, 3.5) + 5.0
    # @test (5.0 + EQ())(3.0, 3.5) == 5.0 + EQ()(3.0, 3.5)

    # # Test multiplication.
    # @test isstationary(EQ() * EQ())
    # @test isstationary(EQ() * 5.0)
    # @test isstationary(5.0 * EQ())
    # @test !isstationary(EQ() * Linear(5.0))
    # @test (EQ() * EQ())(5.0, 4.0) == EQ()(5.0, 4.0) * EQ()(5.0, 4.0)
    # @test (EQ() * 5.0)(3.0, 3.5) == EQ()(3.0, 3.5) * 5.0

    # import Stheno: LhsOp, RhsOp
    # @test !isstationary(LhsOp{typeof(+), typeof(sin), EQ})
    # @test sin + EQ() == sin + EQ()
    # @test sin + EQ() != cos + EQ()
    # @test sin + EQ() != sin + RQ(1.0)
    # @test (sin + EQ())(5.0, 4.0) == sin(5.0) + EQ()(5.0, 4.0)
    # @test (cos * RQ(1.0))(3.3, 6.7) == cos(3.3) * RQ(1.0)(3.3, 6.7)

    # if check_mem

    #     # Performance checks: +
    #     @test memory(@benchmark EQ() + EQ() seconds=0.1) == 0
    #     @test memory(@benchmark $(EQ() + EQ())(1.0, 0.0) seconds=0.1) == 0
    #     @test memory(@benchmark EQ() + RQ(1.0) seconds=0.1) == 0
    #     @test memory(@benchmark $(EQ() + RQ(1.0))(1.0, 0.0) seconds=0.1) == 0

    #     # Peformance checks: *
    #     @test memory(@benchmark EQ() * EQ() seconds=0.1) == 0
    #     @test memory(@benchmark $(EQ() * EQ())(1.0, 0.0) seconds=0.1) == 0
    #     @test memory(@benchmark EQ() * RQ(1.0) seconds=0.1) == 0
    #     @test memory(@benchmark $(EQ() * RQ(1.0))(1.0, 0.0) seconds=0.1) == 0
    # end
end
