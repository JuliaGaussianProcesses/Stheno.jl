@testset "compose" begin

    # Tests for Composite kernel.
    import Stheno.Composite
    @test isstationary(Composite{typeof(+), Tuple{EQ, Noise}})
    @test isstationary(Composite{typeof(+), Tuple{EQ, RQ, Noise}})
    @test !isstationary(Composite{typeof(+), Tuple{Wiener, EQ}})
    @test Composite{typeof(+), Tuple{EQ, EQ}}((EQ(), EQ())) ==
          Composite{typeof(+), Tuple{EQ, EQ}}((EQ(), EQ()))
    @test Composite{typeof(+), Tuple{EQ, EQ}}((EQ(), EQ())) !=
          Composite{typeof(+), Tuple{EQ, RQ}}((EQ(), RQ(1.0)))

    # Test addition.
    @test isstationary(EQ() + EQ())
    @test isstationary(typeof(EQ() + EQ()))
    @test isstationary(EQ() + RQ(1.0))
    @test isstationary(EQ() + 5.0)
    @test !isstationary(EQ() + Linear(5))
    @test (EQ() + EQ())(5.0, 4.0) == EQ()(5.0, 4.0) + EQ()(5.0, 4.0)
    @test (EQ() + 5.0)(3.0, 3.5) == EQ()(3.0, 3.5) + 5.0
    @test (RQ(3.3) + EQ())(1.2, 1.1) == RQ(3.3)(1.2, 1.1) + EQ()(1.2, 1.1)
    @test (RQ(1.0) + EQ() + 5.0)(3.4, 2.1) == RQ(1.0)(3.4, 2.1) + EQ()(3.4, 2.1) + 5.0

    # Test multiplication.
    @test typeof(EQ() * EQ()) <: Composite
    @test isstationary(typeof(EQ() * EQ()))
    @test isstationary(typeof(EQ() * RQ(1.0)))
    @test isstationary(typeof(EQ() * 5.0))
    @test !isstationary(typeof(EQ() * Linear(5.0)))
    @test (EQ() * EQ())(5.0, 4.0) == EQ()(5.0, 4.0) * EQ()(5.0, 4.0)
    @test (EQ() * 5.0)(3.0, 3.5) == EQ()(3.0, 3.5) * 5.0
    @test (RQ(3.3) * EQ())(1.2, 1.1) == RQ(3.3)(1.2, 1.1) * EQ()(1.2, 1.1)
    @test (RQ(1.0) * EQ() * 5.0)(3.4, 2.1) == RQ(1.0)(3.4, 2.1) * EQ()(3.4, 2.1) * 5.0

    import Stheno: LhsOp, RhsOp
    @test !isstationary(LhsOp{typeof(+), typeof(sin), EQ})
    @test sin + EQ() == sin + EQ()
    @test sin + EQ() != cos + EQ()
    @test sin + EQ() != sin + RQ(1.0)
    @test (sin + EQ())(5.0, 4.0) == sin(5.0) + EQ()(5.0, 4.0)
    @test (cos * RQ(1.0))(3.3, 6.7) == cos(3.3) * RQ(1.0)(3.3, 6.7)

    if check_mem

        # Performance checks: +
        @test memory(@benchmark EQ() + EQ() seconds=0.1) == 0
        @test memory(@benchmark $(EQ() + EQ())(1.0, 0.0) seconds=0.1) == 0
        @test memory(@benchmark EQ() + RQ(1.0) seconds=0.1) == 0
        @test memory(@benchmark $(EQ() + RQ(1.0))(1.0, 0.0) seconds=0.1) == 0

        # Peformance checks: *
        @test memory(@benchmark EQ() * EQ() seconds=0.1) == 0
        @test memory(@benchmark $(EQ() * EQ())(1.0, 0.0) seconds=0.1) == 0
        @test memory(@benchmark EQ() * RQ(1.0) seconds=0.1) == 0
        @test memory(@benchmark $(EQ() * RQ(1.0))(1.0, 0.0) seconds=0.1) == 0
    end
end
