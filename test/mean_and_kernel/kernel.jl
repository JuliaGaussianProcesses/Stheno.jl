using Stheno: ZeroKernel, ConstantKernel

@testset "kernel" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)

        # Tests for ZeroKernel.
        k_zero = ZeroKernel{Float64}()
        @test isstationary(k_zero)
        @test k_zero(0, 0) === zero(Float64)
        @test size(k_zero, 1) == Inf && size(k_zero, 2) == Inf
        @test size(k_zero) == (Inf, Inf)
        kernel_tests(k_zero, x0, x1, x2)
        kernel_tests(k_zero, X0, X1, X2)

        # Tests for ConstantKernel.
        k_const = ConstantKernel(randn(rng))
        @test isstationary(k_const)
        @test size(k_const, 1) == Inf && size(k_const, 2) == Inf
        @test size(k_const) == (Inf, Inf)
        kernel_tests(k_const, x0, x1, x2)
        kernel_tests(k_const, X0, X1, X2)

        # Tests for EQ.
        @test isstationary(EQ())
        @test size(EQ(), 1) == Inf && size(EQ(), 2) == Inf
        @test size(EQ()) == (Inf, Inf)
        kernel_tests(EQ(), x0, x1, x2)
        kernel_tests(EQ(), X0, X1, X2)

        # Tests for Linear.
        @test !isstationary(Linear)
        a, b = Linear(randn(rng)), Linear(randn(rng))
        @test a == a && a ≠ b
        kernel_tests(a, x0, x1, x2)
        kernel_tests(a, X0, X1, X2)

        # Tests for Noise
        @test isstationary(Noise(randn(rng)))
        @test Noise(5.0) == Noise(5)
        kernel_tests(Noise(5), x0, x1, x2)
        kernel_tests(Noise(5), X0, X1, X2)
    end

    # # Tests for Rational Quadratic (RQ) kernel.
    # @test isstationary(RQ)
    # @test RQ(1.0)(1.0, 1.0) == 1
    # @test RQ(100.0)(1.0, 1000.0) ≈ 0
    # @test RQ(1.0) == RQ(1.0)
    # @test RQ(1.0) == RQ(1)
    # @test RQ(1.0) != RQ(5.0)
    # @test RQ(1000.0) != EQ()

    # # Tests for Linear kernel.
    # @test !isstationary(Linear)
    # @test Linear(0.0)(1.0, 1.0) == 1
    # @test Linear(1.0)(1.0, 1.0) == 0
    # @test Linear(0.0)(0.0, 0.0) == 0
    # @test Linear(0.0)(5.0, 4.0) ≈ 20
    # @test Linear(2.0)(5.0, 4.0) ≈ 6
    # @test Linear(2.0) == Linear(2.0)
    # @test Linear(1.0) != Linear(2.0)

    # # Tests for Polynomial kernel.
    # @test !isstationary(Poly)
    # @test Poly(2, -1.0)(1.0, 1.0) == 0.0
    # @test Poly(5, -1.0)(1.0, 1.0) == 0.0
    # @test Poly(5, 0.0)(1.0, 1.0) == 1.0
    # @test Poly(5, 0.0) == Poly(5, 0.0)
    # @test Poly(2, 1.0) != Poly(5, 1.0)

    # # Tests for Noise kernel.
    # @test isstationary(Noise)
    # @test Noise()(1.0, 1.0) == 1.0
    # @test Noise()(0.0, 1e-9) == 0.0
    # @test Noise() == Noise()
    # @test Noise() != RQ(1.0)

    # # Tests for Wiener kernel.
    # @test !isstationary(Wiener)
    # @test Wiener()(1.0, 1.0) == 1.0
    # @test Wiener()(1.0, 1.5) == 1.0
    # @test Wiener()(1.5, 1.0) == 1.0
    # @test Wiener() == Wiener()
    # @test Wiener() != Noise()

    # # Tests for WienerVelocity.
    # @test !isstationary(WienerVelocity)
    # @test WienerVelocity()(1.0, 1.0) == 1 / 3
    # @test WienerVelocity() == WienerVelocity()
    # @test WienerVelocity() != Wiener()
    # @test WienerVelocity() != Noise()

    # # Tests for Exponential.
    # @test isstationary(Exponential)
    # @test Exponential() == Exponential()
    # @test Exponential()(5.0, 5.0) == 1.0
    # @test Exponential() != EQ()
    # @test Exponential() != Noise()

    # if check_mem

    #     # Performance checks: Constant.
    #     @test memory(@benchmark Constant(1.0) seconds=0.1) == 0
    #     @test memory(@benchmark $(Constant(1.0))(1.0, 0.0) seconds=0.1) == 0

    #     # Performance checks: EQ.
    #     @test memory(@benchmark EQ() seconds=0.1) == 0
    #     @test memory(@benchmark $(EQ())(1.0, 0.0) seconds=0.1) == 0

    #     # Performance checks: RQ.
    #     @test memory(@benchmark RQ(1.0) seconds=0.1) == 0
    #     @test memory(@benchmark $(RQ(1.0))(1.0, 0.0) seconds=0.1) == 0

    #     # Performance checks: Linear.
    #     @test memory(@benchmark Linear(1.0) seconds=0.1) == 0
    #     @test memory(@benchmark $(Linear(1.0))(1.0, 0.0) seconds=0.1) == 0

    #     # Performance checks: Linear.
    #     @test memory(@benchmark Poly(2, 1.0) seconds=0.1) == 0
    #     @test memory(@benchmark $(Poly(2, 1.0))(1.0, 0.0) seconds=0.1) == 0

    #     # Performance checks: White noise.
    #     @test memory(@benchmark Noise() seconds=0.1) == 0
    #     @test memory(@benchmark $(Noise())(1.0, 0.0) seconds=0.1) == 0

    #     # Performance checks: Wiener process.
    #     @test memory(@benchmark Wiener() seconds=0.1) == 0
    #     @test memory(@benchmark $(Wiener())(1.0, 0.0) seconds=0.1) == 0

    #     # Performance checks: Wiener process.
    #     @test memory(@benchmark WienerVelocity() seconds=0.1) == 0
    #     @test memory(@benchmark $(WienerVelocity())(1.0, 0.0) seconds=0.1) == 0

    #     # Performance checks: Exponential
    #     @test memory(@benchmark Exponential() seconds=0.1) == 0
    #     @test memory(@benchmark $(Exponential())(1.0, 0.0) seconds=0.1) == 0
    # end
end
