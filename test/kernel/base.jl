@testset "kernel" begin

    # Check that the type system works as expected.
    import Stheno: Stationary, NonStationary
    @test issubtype(Kernel{Stationary}, Kernel)
    @test issubtype(Kernel{NonStationary}, Kernel)
    @test !issubtype(Kernel{Stationary}, Kernel{NonStationary})
    @test !issubtype(Kernel{NonStationary}, Kernel{Stationary})

    # Tests for Constant <: Kernel{Stationary}.
    import Stheno.Constant
    rng = MersenneTwister(123456)
    @test Constant(5.0).value == 5.0
    @test Constant(4.9)(randn(rng), randn(rng)) == 4.9
    @test Constant(1.0) == Constant(1.0)
    @test Constant(1.0) != 1.0

    # Tests for Exponentiated Quadratic (EQ) kernel.
    @test EQ()(5.0, 5.0) == 1
    @test EQ()(5.0, 100.0) ≈ 0
    @test EQ() == EQ()
    @test EQ() != Constant(1.0)
    @test EQ() != 1.0

    # Tests for Rational Quadratic (RQ) kernel.
    @test RQ(1.0)(1.0, 1.0) == 1
    @test RQ(100.0)(1.0, 1000.0) ≈ 0
    @test RQ(1.0) == RQ(1.0)
    @test RQ(1.0) == RQ(1)
    @test RQ(1.0) != RQ(5.0)
    @test RQ(1000.0) != EQ()

    # Tests for Linear kernel.
    @test Linear(0.0)(1.0, 1.0) == 1
    @test Linear(1.0)(1.0, 1.0) == 0
    @test Linear(0.0)(0.0, 0.0) == 0
    @test Linear(0.0)(5.0, 4.0) ≈ 20
    @test Linear(2.0)(5.0, 4.0) ≈ 6

    # Performance checks: Constant.
    @test memory(@benchmark Constant(1.0) seconds=0.1) == 0
    @test memory(@benchmark $(Constant(1.0))(1.0, 0.0) seconds=0.1) == 0

    # Performance checks: EQ.
    @test memory(@benchmark EQ() seconds=0.1) == 0
    @test memory(@benchmark $(EQ())(1.0, 0.0) seconds=0.1) == 0

    # Performance checks: RQ.
    @test memory(@benchmark RQ(1.0) seconds=0.1) == 0
    @test memory(@benchmark $(RQ(1.0))(1.0, 0.0) seconds=0.1) == 0

    # Performance checks: Linear.
    @test memory(@benchmark Linear(1.0) seconds=0.1) == 0
    @test memory(@benchmark $(Linear(1.0))(1.0, 0.0) seconds=0.1) == 0
end
