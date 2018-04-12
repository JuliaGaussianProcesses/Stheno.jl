# This file is named as such because, for some reason, Julia is unable to load a file
# called compose.jl or composite.jl

@testset "compose" begin

    # Tests for UnaryComposite kernel.
    using Stheno: UnaryComposite
    @test isstationary(UnaryComposite(x->5x, EQ()))
    @test !isstationary(UnaryComposite(x->5x, Linear(1)))     

    # Tests for BinaryComposite kernel.
    using Stheno: BinaryComposite
    @test isstationary(BinaryComposite(+, EQ(), EQ()))
    @test !isstationary(BinaryComposite(+, EQ(), Linear(1)))

    # Test addition.
    @test isstationary(EQ() + EQ())
    @test !isstationary(EQ() + Linear(5))
    @test isstationary(5 * EQ())
    @test (EQ() + EQ())(5.0, 4.0) == EQ()(5.0, 4.0) + EQ()(5.0, 4.0)
    @test (EQ() + 5.0)(3.0, 3.5) == EQ()(3.0, 3.5) + 5.0
    @test (5.0 + EQ())(3.0, 3.5) == 5.0 + EQ()(3.0, 3.5)

    # Test multiplication.
    @test isstationary(EQ() * EQ())
    @test isstationary(EQ() * 5.0)
    @test isstationary(5.0 * EQ())
    @test !isstationary(EQ() * Linear(5.0))
    @test (EQ() * EQ())(5.0, 4.0) == EQ()(5.0, 4.0) * EQ()(5.0, 4.0)
    @test (EQ() * 5.0)(3.0, 3.5) == EQ()(3.0, 3.5) * 5.0

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
