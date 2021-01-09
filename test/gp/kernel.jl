using Stheno: ZeroKernel, OneKernel, ConstKernel, CustomMean, pw, Stretched, Scaled
using Stheno: EQ, Exp, Linear, Noise, PerEQ, Matern32, Matern52, RQ, Cosine, Sum, Product, scale, stretch,
    Poly, GammaExp, Wiener, WienerVelocity, Precomputed
using Stheno: LinearLayer, Primitive, product, NeuralKernelNetwork

@timedtestset "kernel" begin

    @timedtestset "base kernels" begin
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0 = collect(range(-2.0, 2.0; length=N)) .+ 1e-3 .* randn(rng, N)
        x1 = collect(range(-1.7, 2.3; length=N)) .+ 1e-3 .* randn(rng, N)
        x2 = collect(range(-1.7, 3.3; length=N′)) .+ 1e-3 .* randn(rng, N′)
        x0_r, x1_r = range(-5.0, step=1, length=N), range(-4.0, step=1, length=N)
        x2_r, x3_r = range(-5.0, step=2, length=N), range(-3.0, step=1, length=N′)
        x4_r = range(-2.0, step=2, length=N′)

        X0 = ColVecs(randn(rng, D, N))
        X1 = ColVecs(randn(rng, D, N))
        X2 = ColVecs(randn(rng, D, N′))

        ȳ, Ȳ, Ȳ_sq = randn(rng, N), randn(rng, N, N′), randn(rng, N, N)

        @timedtestset "ZeroKernel" begin
            differentiable_kernel_tests(ZeroKernel(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(ZeroKernel(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @timedtestset "OneKernel" begin
            differentiable_kernel_tests(OneKernel(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(OneKernel(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @timedtestset "ConstKernel" begin
            @test ew(ConstKernel(5), x0) == 5 .* ones(length(x0))
            differentiable_kernel_tests(ConstKernel(5.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(ConstKernel(5.0), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @timedtestset "EQ" begin
            differentiable_kernel_tests(EQ(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(EQ(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @timedtestset "PerEQ" begin
            differentiable_kernel_tests(PerEQ(1.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2; atol=1e-6)
        end

        @timedtestset "Exp" begin
            differentiable_kernel_tests(Exp(), ȳ, Ȳ, Ȳ_sq, x0 .+ 1, x1, x2)
            differentiable_kernel_tests(Exp(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @timedtestset "Matern32" begin
            differentiable_kernel_tests(Matern32(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(Matern32(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @timedtestset "Matern52" begin
            differentiable_kernel_tests(Matern52(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(Matern52(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @timedtestset "RQ" begin
            @timedtestset "α=1.0" begin
                differentiable_kernel_tests(RQ(1.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
                differentiable_kernel_tests(RQ(1.0), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            end
            @timedtestset "α=1.5" begin
                differentiable_kernel_tests(RQ(1.5), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
                differentiable_kernel_tests(RQ(1.5), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            end
            @timedtestset "α=100.0" begin
                differentiable_kernel_tests(RQ(100.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
                differentiable_kernel_tests(RQ(100.0), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            end
            @timedtestset "single-input" begin
                adjoint_test((α, x, x′)->ew(RQ(α), x, x′), ȳ, 1.5, x0, x1)
                adjoint_test((α, x, x′)->pw(RQ(α), x, x′), Ȳ, 1.5, x0, x2)
                adjoint_test((α, x)->ew(RQ(α), x), ȳ, 1.5, x0)
                adjoint_test((α, x)->pw(RQ(α), x), Ȳ_sq, 1.5, x0)
            end
            @timedtestset "multi-input" begin
                adjoint_test((α, x, x′)->ew(RQ(α), x, x′), ȳ, 1.5, X0, X1)
                adjoint_test((α, x, x′)->pw(RQ(α), x, x′), Ȳ, 1.5, X0, X2)
                adjoint_test((α, x)->ew(RQ(α), x), ȳ, 1.5, X0)
                adjoint_test((α, x)->pw(RQ(α), x), Ȳ_sq, 1.5, X0)
            end
        end

        @timedtestset "Cosine" begin
            @timedtestset "p=1.0" begin
                differentiable_kernel_tests(Cosine(1.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2; atol=1e-8)        
            end
            @timedtestset "p=1.5" begin
                differentiable_kernel_tests(Cosine(1.5), ȳ, Ȳ, Ȳ_sq, x0, x1, x2; atol=1e-8)        
            end
            @timedtestset "p=100.0" begin
                differentiable_kernel_tests(Cosine(100.0), ȳ, Ȳ, Ȳ_sq, x0, x1, x2; atol=1e-8)        
            end
            @timedtestset "single-input" begin
                adjoint_test((p, x, x′)->ew(Cosine(p), x, x′), ȳ, 1.5, x0, x1)
                adjoint_test((p, x, x′)->pw(Cosine(p), x, x′), Ȳ, 1.5, x0, x2)
                adjoint_test((p, x)->ew(Cosine(p), x), ȳ, 1.5, x0)
                adjoint_test((p, x)->pw(Cosine(p), x), Ȳ_sq, 1.5, x0)
            end
        end

        @timedtestset "Linear" begin
            differentiable_kernel_tests(Linear(), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(Linear(), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
        end

        @timedtestset "Poly" begin
            @test pw(Poly(1, 0.0), x0) ≈ pw(Linear(), x0)
            @testset "Poly{$p}" for p in [1, 2, 3]
                differentiable_kernel_tests(Poly(p, 0.5), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
                differentiable_kernel_tests(Poly(p, 0.5), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            end
        end

        @timedtestset "GammaExp" begin
            @test pw(GammaExp(2.0), x0) ≈ pw(stretch(EQ(), sqrt(2)), x0)
            @testset "γ=$γ" for γ in [1.0, 1.5, 2.0]
                differentiable_kernel_tests(GammaExp(γ), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
                differentiable_kernel_tests(GammaExp(γ), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            end
        end

        @timedtestset "Wiener" begin
            x0 = collect(range(0.1, 2.0; length=N)) .+ 1e-3 .* randn(rng, N)
            x1 = collect(range(0.1, 2.3; length=N)) .+ 1e-3 .* randn(rng, N)
            x2 = collect(range(0.1, 3.3; length=N′)) .+ 1e-3 .* randn(rng, N′)
            kernel_tests(Wiener(), x0, x1, x2)
        end

        @timedtestset "WienerVelocity" begin
            x0 = collect(range(0.1, 2.0; length=N)) .+ 1e-3 .* randn(rng, N)
            x1 = collect(range(0.1, 2.3; length=N)) .+ 1e-3 .* randn(rng, N)
            x2 = collect(range(0.1, 3.3; length=N′)) .+ 1e-3 .* randn(rng, N′)
            kernel_tests(WienerVelocity(), x0, x1, x2)
        end

        @timedtestset "Noise" begin
            @test ew(Noise(), x0, x0) == zeros(length(x0))
            @test pw(Noise(), x0, x0) == zeros(length(x0), length(x0))
            @test ew(Noise(), x0) == ones(length(x0))
            @test pw(Noise(), x0) == Diagonal(ones(length(x0)))
        end

        @timedtestset "precomputed" begin
            rng = MersenneTwister(123456)
            x = ColVecs(randn(rng, N, N))
            K = pw(Linear(), x)
            k = precomputed(K)
            @test pw(k, 1:N) == pw(Linear(), x)
            @test ew(k, 1:N) ≈ ew(Linear(), x)
            kernel_tests(k, 1:N-1, 2:N, 1:N)
        end

        @timedtestset "Sum" begin
            differentiable_kernel_tests(Sum(EQ(), Exp()), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(Sum(EQ(), Exp()), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            @test EQ() + Exp() isa Sum
        end

        @timedtestset "Product" begin
            differentiable_kernel_tests(Product(EQ(), Exp()), ȳ, Ȳ, Ȳ_sq, x0, x1, x2; atol=1e-6)
            differentiable_kernel_tests(Product(EQ(), Exp()), ȳ, Ȳ, Ȳ_sq, X0, X1, X2; atol=1e-6)
            @test EQ() * Exp() isa Product
        end

        @timedtestset "Scaled" begin
            differentiable_kernel_tests(scale(EQ(), 0.5), ȳ, Ȳ, Ȳ_sq, x0, x1, x2)
            differentiable_kernel_tests(scale(EQ(), 0.5), ȳ, Ȳ, Ȳ_sq, X0, X1, X2)
            adjoint_test(σ²->pw(scale(EQ(), σ²), X0), Ȳ_sq, 0.5)
            @test 0.5 * EQ() isa Scaled
            @test EQ() * 0.5 isa Scaled
        end

        @timedtestset "Stretched" begin
            @timedtestset "Scalar a" begin
                k = stretch(EQ(), 0.1)
                differentiable_kernel_tests(k, ȳ, Ȳ, Ȳ_sq, x0, x1, x2; atol=1e-7, rtol=1e-7)
                differentiable_kernel_tests(k, ȳ, Ȳ, Ȳ_sq, X0, X1, X2; atol=1e-7, rtol=1e-7)
            end
            @timedtestset "Vector a" begin
                k = stretch(EQ(), randn(rng, D))
                differentiable_kernel_tests(k, ȳ, Ȳ, Ȳ_sq, X0, X1, X2; atol=1e-7, rtol=1e-7)
            end
            @timedtestset "Matrix a" begin
                k = stretch(EQ(), randn(rng, D, D))
                differentiable_kernel_tests(k, ȳ, Ȳ, Ȳ_sq, X0, X1, X2; atol=1e-7, rtol=1e-7)
            end
        end

        @timedtestset "kernel" begin
            x = randn(11)
            @test pw(EQ(), x) == pw(kernel(EQ()), x)
            @test pw(EQ(), x) == pw(kernel(EQ(); l=1.0), x)
            @test pw(EQ(), x) == pw(kernel(EQ(); s=1.0), x)
            @test pw(EQ(), x) == pw(kernel(EQ(); l=1.0, s=1.0), x)
            @test pw(stretch(EQ(), 1 / 1.1), x) == pw(kernel(EQ(); l=1.1), x)
            @test pw(0.9 * EQ(), x) == pw(kernel(EQ(); s=0.9), x)
            @test pw(0.9 * stretch(EQ(), 1 / 1.1), x) == pw(kernel(EQ(), l=1.1, s=0.9), x)
        end
    end

    @timedtestset "(is)zero" begin
        @test zero(ZeroKernel()) == ZeroKernel()
        @test zero(EQ()) == ZeroKernel()
        @test iszero(ZeroKernel()) == true
        @test iszero(EQ()) == false
    end
end
