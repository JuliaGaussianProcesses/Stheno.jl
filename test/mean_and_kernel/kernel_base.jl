@testset "kernel" begin

    # Check that the type system works as expected.
    import Stheno: Stationary, NonStationary
    @test !(Kernel{Stationary} <: Kernel)
    @test !(Kernel{NonStationary} <: Kernel)
    @test !(Kernel{Stationary} <: Kernel{NonStationary})
    @test !(Kernel{NonStationary} <: Kernel{Stationary})

    import Stheno.Zero
    let rng = MersenneTwister(123456)
        @test Zero()(randn(rng), 4.0) == 0.0
        @test Zero() == Zero()
    end

    # Tests for Constant <: Kernel{Stationary}.
    import Stheno.Constant
    let rng = MersenneTwister(123456)
        @test Constant(5.0).value == 5.0
        @test Constant(4.9)(randn(rng), randn(rng)) == 4.9
        @test Constant(1.0) == Constant(1.0)
        @test Constant(1.0) != 1.0
    end

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
    @test Linear(2.0) == Linear(2.0)
    @test Linear(1.0) != Linear(2.0)

    # Tests for Polynomial kernel.
    @test Poly(2, -1.0)(1.0, 1.0) == 0.0
    @test Poly(5, -1.0)(1.0, 1.0) == 0.0
    @test Poly(5, 0.0)(1.0, 1.0) == 1.0
    @test Poly(5, 0.0) == Poly(5, 0.0)
    @test Poly(2, 1.0) != Poly(5, 1.0)

    # Tests for Noise kernel.
    @test Noise <: Kernel{Stationary}
    @test !(Noise <: Kernel{NonStationary})
    @test Noise()(1.0, 1.0) == 1.0
    @test Noise()(0.0, 1e-9) == 0.0
    @test Noise() == Noise()
    @test Noise() != RQ(1.0)

    # Tests for Wiener kernel.
    @test Wiener <: Kernel{NonStationary}
    @test !(Wiener <: Kernel{Stationary})
    @test Wiener()(1.0, 1.0) == 1.0
    @test Wiener()(1.0, 1.5) == 1.0
    @test Wiener()(1.5, 1.0) == 1.0
    @test Wiener() == Wiener()
    @test Wiener() != Noise()

    # Tests for WienerVelocity.
    @test WienerVelocity <: Kernel{NonStationary}
    @test !(WienerVelocity <: Kernel{Stationary})
    @test WienerVelocity()(1.0, 1.0) == 1 / 3
    @test WienerVelocity() == WienerVelocity()
    @test WienerVelocity() != Wiener()
    @test WienerVelocity() != Noise()

    # Tests for Exponential.
    @test Exponential <: Kernel{Stationary}
    @test !(Exponential <: Kernel{NonStationary})
    @test Exponential() == Exponential()
    @test Exponential()(5.0, 5.0) == 1.0
    @test Exponential() != EQ()
    @test Exponential() != Noise()

    if check_mem

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

        # Performance checks: Linear.
        @test memory(@benchmark Poly(2, 1.0) seconds=0.1) == 0
        @test memory(@benchmark $(Poly(2, 1.0))(1.0, 0.0) seconds=0.1) == 0

        # Performance checks: White noise.
        @test memory(@benchmark Noise() seconds=0.1) == 0
        @test memory(@benchmark $(Noise())(1.0, 0.0) seconds=0.1) == 0

        # Performance checks: Wiener process.
        @test memory(@benchmark Wiener() seconds=0.1) == 0
        @test memory(@benchmark $(Wiener())(1.0, 0.0) seconds=0.1) == 0

        # Performance checks: Wiener process.
        @test memory(@benchmark WienerVelocity() seconds=0.1) == 0
        @test memory(@benchmark $(WienerVelocity())(1.0, 0.0) seconds=0.1) == 0

        # Performance checks: Exponential
        @test memory(@benchmark Exponential() seconds=0.1) == 0
        @test memory(@benchmark $(Exponential())(1.0, 0.0) seconds=0.1) == 0
    end
end
