using Stheno: CrossKernel, ZeroKernel, OneKernel, ConstKernel, CustomMean, pw
using Stheno: EQ, Exp, Linear, Noise, PerEQ, Matern32, Matern52, RQ
using LinearAlgebra

@testset "kernel" begin

    @testset "base kernels" begin
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0 = collect(range(-2.0, 2.0; length=N)) .+ 1e-3 .* randn(rng, N)
        x1 = collect(range(-1.7, 2.3; length=N)) .+ 1e-3 .* randn(rng, N)
        x2 = collect(range(-1.7, 3.3; length=N′)) .+ 1e-3 .* randn(rng, N′)
        x0_r, x1_r = range(-5.0, step=1, length=N), range(-4.0, step=1, length=N)
        x2_r, x3_r = range(-5.0, step=2, length=N), range(-3.0, step=1, length=N′)
        x4_r = range(-2.0, step=2, length=N′)

        X0 = ColsAreObs(randn(rng, D, N))
        X1 = ColsAreObs(randn(rng, D, N))
        X2 = ColsAreObs(randn(rng, D, N′))

        ȳ, Ȳ, Ȳ_sq = randn(rng, N), randn(rng, N, N′), randn(rng, N, N)

        @testset "ZeroKernel" begin
            differentiable_kernel_tests(ZeroKernel(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(ZeroKernel(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @testset "OneKernel" begin
            differentiable_kernel_tests(OneKernel(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(OneKernel(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @testset "ConstKernel" begin
            @test ew(ConstKernel(5), x0) == 5 .* ones(length(x0))
            differentiable_kernel_tests(ConstKernel(5.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(ConstKernel(5.0), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @testset "EQ" begin
            differentiable_kernel_tests(EQ(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(EQ(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @testset "PerEQ" begin
            differentiable_kernel_tests(PerEQ(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2; atol=1e-6)
        end

        @testset "Exp" begin
            differentiable_kernel_tests(Exp(), ȳ, Ȳ, Ȳ_sq, x0 .+ 1, x1, x2)
            differentiable_kernel_tests(Exp(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @testset "Matern32" begin
            differentiable_kernel_tests(Matern32(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(Matern32(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @testset "Matern52" begin
            differentiable_kernel_tests(Matern52(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(Matern52(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @testset "RQ" begin
            @testset "α=1.0" begin
                differentiable_kernel_tests(RQ(1.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
                differentiable_kernel_tests(RQ(1.0), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            end
            @testset "α=1.5" begin
                differentiable_kernel_tests(RQ(1.5), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
                differentiable_kernel_tests(RQ(1.5), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            end
            @testset "α=100.0" begin
                differentiable_kernel_tests(RQ(100.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
                differentiable_kernel_tests(RQ(100.0), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            end
            @testset "single-input" begin
                adjoint_test((α, x, x′)->ew(RQ(α), x, x′), ȳ, 1.5, x0, x1; print_results=true)
                adjoint_test((α, x, x′)->pw(RQ(α), x, x′), Ȳ, 1.5, x0, x2)
                adjoint_test((α, x)->ew(RQ(α), x), ȳ, 1.5, x0)
                adjoint_test((α, x)->pw(RQ(α), x), Ȳ_sq, 1.5, x0)
            end
            @testset "multi-input" begin
                adjoint_test((α, x, x′)->ew(RQ(α), x, x′), ȳ, 1.5, X0, X1)
                adjoint_test((α, x, x′)->pw(RQ(α), x, x′), Ȳ, 1.5, X0, X2)
                adjoint_test((α, x)->ew(RQ(α), x), ȳ, 1.5, X0)
                adjoint_test((α, x)->pw(RQ(α), x), Ȳ_sq, 1.5, X0)
            end
        end

        @testset "Linear" begin
            differentiable_kernel_tests(Linear(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(Linear(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @testset "Noise" begin
            @test pw(Noise(), x0, x0) == zeros(length(x0), length(x0))
            @test pw(Noise(), x0) == Diagonal(ones(length(x0)))
        end
    end

    @testset "(is)zero" begin
        @test zero(ZeroKernel()) == ZeroKernel()
        @test zero(EQ()) == ZeroKernel()
        @test iszero(ZeroKernel()) == true
        @test iszero(EQ()) == false
    end

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
