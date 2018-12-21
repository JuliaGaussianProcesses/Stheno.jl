using Stheno: CrossKernel, ZeroKernel, ConstantKernel, pairwise, EmpiricalKernel

@testset "kernel" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

        # Tests for ZeroKernel.
        let
            k = ZeroKernel{Float64}()
            @test k(0, 0) === zero(Float64)
            @test map(k, x0) isa Zeros
            kernel_tests(k, x0, x1, x2)
            kernel_tests(k, X0, X1, X2)
        end

        # Tests for ConstantKernel.
        let
            k = ConstantKernel(randn(rng))
            kernel_tests(k, x0, x1, x2)
            kernel_tests(k, X0, X1, X2)
        end

        # Tests for EQ.
        kernel_tests(EQ(), x0, x1, x2)
        kernel_tests(EQ(), X0, X1, X2)

        # Tests for PerEQ.
        let
            @test map(PerEQ(1), x0) isa Ones
            kernel_tests(PerEQ(1), x0, x1, x2)
            kernel_tests(PerEQ(1f0), x0, x1, x2)
        end

        # # Tests for Exponential.
        # @test isstationary(EQ())
        # @test size(Exponential(), 1) == Inf && size(Exponential(), 2) == Inf
        # @test size(Exponential()) == (Inf, Inf)
        # kernel_tests(Exponential(), x0, x1, x2)

        # # Tests for Linear.
        # @test !isstationary(Linear)
        # a, b = Linear(randn(rng)), Linear(randn(rng))
        # @test a == a && a ≠ b
        # kernel_tests(a, x0, x1, x2)
        # kernel_tests(a, X0, X1, X2)

        # # Tests for Noise
        # @test isstationary(Noise(randn(rng)))
        # @test Noise(5.0) == Noise(5)
        # kernel_tests(Noise(5.0), x0, x1, x2)
        # kernel_tests(Noise(5.0), X0, X1, X2)
        # # kernel_tests(Noise(5.0), XB0, XB1, XB2)

        # # Tests for EmpiricalKernel.
        # A_ = randn(rng, N, N)
        # A = LazyPDMat(A_ * A_' + 1e-6I)
        # k = EmpiricalKernel(A)
        # Ds0, Ds1, Ds2 = 1:N, 1:N, 1:N-1
        # kernel_tests(k, Ds0, Ds1, Ds2)
        # @test size(k, 1) == N && size(k, 2) == N

        # @test map(k, :) == diag(A)
        # @test map(k, :, :) == diag(A)
        # @test pairwise(k, :) == A
        # @test pairwise(k, :, :) == A

        # @test_throws AssertionError AbstractMatrix(EQ())
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
end
