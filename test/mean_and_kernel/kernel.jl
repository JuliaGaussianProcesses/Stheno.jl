using Stheno: CrossKernel, ZeroKernel, OneKernel, pairwise, EmpiricalKernel

@testset "kernel" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        x0_r, x1_r = range(-5.0, step=1, length=N), range(-4.0, step=1, length=N)
        x2_r, x3_r = range(-5.0, step=2, length=N), range(-3.0, step=1, length=N′)
        x4_r = range(-2.0, step=2, length=N′)
        # X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

        # Tests for ZeroKernel.
        let
            k = ZeroKernel{Float64}()
            @test k(0, 0) === zero(Float64)
            @test map(k, x0) isa Zeros
            kernel_tests(k, x0, x1, x2)
            # kernel_tests(k, X0, X1, X2)
        end

        # Tests for OneKernel.
        kernel_tests(OneKernel(), x0, x1, x2)
        # kernel_tests(k, X0, X1, X2)

        # Tests for ConstKernel.
        @test ConstKernel(5.0)(randn(rng), randn(rng)) == 5.0
        kernel_tests(ConstKernel(5.0), x0, x1, x2)

        # Tests for EQ.
        @test map(EQ(), x0) isa Ones
        kernel_tests(EQ(), x0, x1, x2)
        stationary_kernel_tests(EQ(), x0_r, x1_r, x2_r, x3_r, x4_r)
        # kernel_tests(EQ(), X0, X1, X2)

        # Tests for PerEQ.
        @test map(PerEQ(), x0) isa Ones
        kernel_tests(PerEQ(), x0, x1, x2)
        stationary_kernel_tests(PerEQ(), x0_r, x1_r, x2_r, x3_r, x4_r)

        # Tests for Exponential.
        @test map(Exp(), x0) isa Ones
        kernel_tests(Exp(), x0 .+ 1, x1, x2)
        stationary_kernel_tests(Exp(), x0_r, x1_r, x2_r, x3_r, x4_r)

        # Tests for Linear.
        kernel_tests(Linear(), x0, x1, x2)
        # kernel_tests(a, X0, X1, X2)

        # Tests for Noise. It doesn't follow the usual kernel consistency criteria.
        @test pairwise(Noise(), x0, x0) == zeros(length(x0), length(x0))
        @test pairwise(Noise(), x0) == Diagonal(ones(length(x0)))

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
end
