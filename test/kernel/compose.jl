@testset "compose" begin

    # Tests for Composite kernel.
    import Stheno.Composite
    @test Composite{Stationary, typeof(+), Tuple{EQ, EQ}} <: Kernel{Stationary}
    @test Composite{NonStationary, typeof(+), Tuple{EQ, EQ}} <: Kernel{NonStationary}
    @test Composite{Stationary, typeof(+), Tuple{EQ, EQ}}((EQ(), EQ())) ==
          Composite{Stationary, typeof(+), Tuple{EQ, EQ}}((EQ(), EQ()))
    @test Composite{Stationary, typeof(+), Tuple{EQ, EQ}}((EQ(), EQ())) !=
          Composite{Stationary, typeof(+), Tuple{EQ, RQ}}((EQ(), RQ(1.0)))

    # Test addition.
    @test typeof(EQ() + EQ()) <: Composite
    @test typeof(EQ() + EQ()) <: Kernel{Stationary}
    @test typeof(EQ() + RQ(1.0)) <: Kernel{Stationary}
    @test typeof(EQ() + 5.0) <: Kernel{Stationary}
    @test typeof(EQ() + Linear(5.0)) <: Kernel{NonStationary}
    @test (EQ() + EQ())(5.0, 4.0) == EQ()(5.0, 4.0) + EQ()(5.0, 4.0)
    @test (EQ() + 5.0)(3.0, 3.5) == EQ()(3.0, 3.5) + 5.0
    @test (RQ(3.3) + EQ())(1.2, 1.1) == RQ(3.3)(1.2, 1.1) + EQ()(1.2, 1.1)
    @test (RQ(1.0) + EQ() + 5.0)(3.4, 2.1) == RQ(1.0)(3.4, 2.1) + EQ()(3.4, 2.1) + 5.0

    # Test multiplication.
    @test typeof(EQ() * EQ()) <: Composite
    @test typeof(EQ() * EQ()) <: Kernel{Stationary}
    @test typeof(EQ() * RQ(1.0)) <: Kernel{Stationary}
    @test typeof(EQ() * 5.0) <: Kernel{Stationary}
    @test typeof(EQ() * Linear(5.0)) <: Kernel{NonStationary}
    @test (EQ() * EQ())(5.0, 4.0) == EQ()(5.0, 4.0) * EQ()(5.0, 4.0)
    @test (EQ() * 5.0)(3.0, 3.5) == EQ()(3.0, 3.5) * 5.0
    @test (RQ(3.3) * EQ())(1.2, 1.1) == RQ(3.3)(1.2, 1.1) * EQ()(1.2, 1.1)
    @test (RQ(1.0) * EQ() * 5.0)(3.4, 2.1) == RQ(1.0)(3.4, 2.1) * EQ()(3.4, 2.1) * 5.0

    # Performance checks: +
    @test memory(@benchmark EQ() + EQ() seconds=0.1) == 0
    @test memory(@benchmark $(EQ() + EQ())(1.0, 0.0) seconds=0.1) == 0
    @test memory(@benchmark EQ() + RQ(1.0) seconds=0.1) == 0
    @test memory(@benchmark $(EQ() + RQ(1.0))(1.0, 0.0) seconds=0.1) == 0

    # Peformance checks: *
    @test memory(@benchmark EQ() * EQ() seconds=0.1) == 0
    @test memory(@benchmark $(EQ() * EQ())(1.0, 0.0) seconds=0.1) == 0
end
