using Stheno: CrossKernel, ZeroKernel, ConstantKernel, pairwise, EmpiricalKernel,
    _pairwise_fallback

struct FooKernel <: CrossKernel end
(::FooKernel)(x, x′) = sum(abs2, x - x′)

@testset "kernel" begin

    # Test pairwise.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, X′ = ColsAreObs(randn(rng, D, P)), ColsAreObs(randn(rng, D, Q))
        x, x′ = randn(rng, P), randn(rng, Q)
        foo = FooKernel()

        @test _pairwise_fallback(foo, X, X′) ==
            [foo(X[1], X′[1]) foo(X[1], X′[2]);
             foo(X[2], X′[1]) foo(X[2], X′[2]);
             foo(X[3], X′[1]) foo(X[3], X′[2])]
        @test _pairwise_fallback(foo, x, x′) ==
            [foo(x[1], x′[1]) foo(x[1], x′[2]);
             foo(x[2], x′[1]) foo(x[2], x′[2]);
             foo(x[3], x′[1]) foo(x[3], x′[2])]

        @test pairwise(foo, X, X′) == _pairwise_fallback(foo, X, X′)
        @test pairwise(foo, x, x′) == _pairwise_fallback(foo, x, x′)
        @test pairwise(foo, X) == pairwise(foo, X, X)
        @test pairwise(foo, x) == pairwise(foo, x, x)

        XB = BlockData([X, X′])
        @test pairwise(foo, XB, XB) ==
            vcat(hcat(pairwise(foo, X, X), pairwise(foo, X, X′)),
                 hcat(pairwise(foo, X′, X), pairwise(foo, X′, X′)))
    end

    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        XB0, XB1, XB2 = BlockData([x0, X0]), BlockData([x1, X1]), BlockData([x2, X2])

        # Tests for ZeroKernel.
        k_zero = ZeroKernel{Float64}()
        @test isstationary(k_zero)
        @test k_zero(0, 0) === zero(Float64)
        @test size(k_zero, 1) == Inf && size(k_zero, 2) == Inf
        @test size(k_zero) == (Inf, Inf)
        @test ZeroKernel{Float64}() == ZeroKernel{Float32}()
        kernel_tests(k_zero, x0, x1, x2)
        kernel_tests(k_zero, X0, X1, X2)
        kernel_tests(k_zero, XB0, XB1, XB2)

        @test ZeroKernel{Float64}() + ZeroKernel{Float64}() == ZeroKernel{Float64}()
        @test ZeroKernel{Float64}() * ZeroKernel{Float64}() === ZeroKernel{Float64}()
        @test ZeroKernel{Float64}() * EQ() == ZeroKernel{Float64}()

        # Tests for ConstantKernel.
        k_const = ConstantKernel(randn(rng))
        @test isstationary(k_const)
        @test k_const == k_const
        @test size(k_const, 1) == Inf && size(k_const, 2) == Inf
        @test size(k_const) == (Inf, Inf)
        kernel_tests(k_const, x0, x1, x2)
        kernel_tests(k_const, X0, X1, X2)
        kernel_tests(k_const, XB0, XB1, XB2)

        zro = ZeroKernel{Float64}()
        @test k_const + k_const isa ConstantKernel
        @test k_const * k_const == ConstantKernel(k_const.c^2)
        @test zro + k_const === k_const
        @test k_const + zro === k_const
        @test zro * k_const === zro
        @test k_const * zro === zro

        # Tests for EQ.
        @test isstationary(EQ())
        @test size(EQ(), 1) == Inf && size(EQ(), 2) == Inf
        @test size(EQ()) == (Inf, Inf)
        kernel_tests(EQ(), x0, x1, x2)
        kernel_tests(EQ(), X0, X1, X2)
        kernel_tests(EQ(), BlockData([x0, x0]), BlockData([x1, x1]), BlockData([x2, x2]))
        kernel_tests(EQ(), BlockData([X0, X0]), BlockData([X1, X1]), BlockData([X2, X2]))

        # Tests for Exponential.
        @test isstationary(EQ())
        @test size(Exponential(), 1) == Inf && size(Exponential(), 2) == Inf
        @test size(Exponential()) == (Inf, Inf)
        kernel_tests(Exponential(), x0, x1, x2)
        kernel_tests(EQ(), BlockData([x0, x0]), BlockData([x1, x1]), BlockData([x2, x2]))

        # Tests for Linear.
        @test !isstationary(Linear)
        a, b = Linear(randn(rng)), Linear(randn(rng))
        @test a == a && a ≠ b
        kernel_tests(a, x0, x1, x2)
        kernel_tests(a, X0, X1, X2)
        kernel_tests(a, BlockData([x0, x0]), BlockData([x1, x1]), BlockData([x2, x2]))
        kernel_tests(a, BlockData([X0, X0]), BlockData([X1, X1]), BlockData([X2, X2]))

        # Tests for Noise
        @test isstationary(Noise(randn(rng)))
        @test Noise(5.0) == Noise(5)
        kernel_tests(Noise(5.0), x0, x1, x2)
        kernel_tests(Noise(5.0), X0, X1, X2)
        # kernel_tests(Noise(5.0), XB0, XB1, XB2)

        # Tests for EmpiricalKernel.
        A_ = randn(rng, N, N)
        A = LazyPDMat(A_ * A_' + 1e-6I)
        k = EmpiricalKernel(A)
        Ds0, Ds1, Ds2 = 1:N, 1:N, 1:N-1
        kernel_tests(k, Ds0, Ds1, Ds2)
        kernel_tests(k, BlockData([Ds0, Ds0]), BlockData([Ds1, Ds1]), BlockData([Ds2, Ds2]))
        @test size(k, 1) == N && size(k, 2) == N

        @test map(k, :) == diag(A)
        @test map(k, :, :) == diag(A)
        @test pairwise(k, :) == A
        @test pairwise(k, :, :) == A

        @test_throws AssertionError AbstractMatrix(EQ())
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
