@testset "kernel" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x, X = randn(rng, D, N), RowVector(randn(rng, N))
        x′, X′ = randn(rng, D, N′), RowVector(randn(rng, N′))
        xr, Xr = randn(rng, D, N), RowVector(randn(rng, N))

        # Tests for ZeroKernel.
        k_zero = ZeroKernel{Float64}()
        @test isstationary(k_zero)
        @test k_zero(0, 0) === zero(Float64)
        binary_colwise_tests(k_zero, x, xr)
        binary_colwise_tests(k_zero, X, Xr)
        pairwise_tests(k_zero, x, x′)
        pairwise_tests(k_zero, X, X′)
        @test size(k_zero, 1) == Inf && size(k_zero, 2) == Inf
        @test size(k_zero) == (Inf, Inf)

        # Tests for ConstantKernel.
        k_const = ConstantKernel(randn(rng))
        @test isstationary(k_const)
        binary_colwise_tests(k_const, x, xr)
        binary_colwise_tests(k_const, X, Xr)
        pairwise_tests(k_const, x, x′)
        pairwise_tests(k_const, X, X′)
        @test size(k_const, 1) == Inf && size(k_const, 2) == Inf
        @test size(k_const) == (Inf, Inf)

        # Tests for EQ.
        @test isstationary(EQ())
        binary_colwise_tests(EQ(), x, xr)
        binary_colwise_tests(EQ(), X, Xr)
        pairwise_tests(EQ(), x, x′)
        pairwise_tests(EQ(), X, X′)
        @test size(EQ(), 1) == Inf && size(EQ(), 2) == Inf
        @test size(EQ()) == (Inf, Inf)

        # Tests for Linear.
        @test !isstationary(Linear)
        a, b = Linear(randn(rng)), Linear(randn(rng))
        @test a == a && a ≠ b
        binary_colwise_tests(Linear(randn(rng)), x, xr)
        binary_colwise_tests(Linear(randn(rng)), X, Xr)
        pairwise_tests(Linear(randn(rng)), x, x′)
        pairwise_tests(Linear(randn(rng)), X, X′)

        # Tests for Noise
        @test isstationary(Noise(randn(rng)))
        @test Noise(5.0) == Noise(5)
        binary_colwise_tests(Noise(randn(rng)), x, xr)
        binary_colwise_tests(Noise(randn(rng)), X, Xr)
        pairwise_tests(Noise(randn(rng)), x, x′)
        pairwise_tests(Noise(randn(rng)), X, X′)
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
