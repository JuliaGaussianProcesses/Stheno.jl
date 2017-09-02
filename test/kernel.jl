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
end
