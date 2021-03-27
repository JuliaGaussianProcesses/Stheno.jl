using Stheno: GPC, ZeroMean, ConstMean, CustomMean, ZeroKernel

@timedtestset "gp" begin

    # Ensure that basic functionality works as expected.
    @timedtestset "GP" begin
        rng, gpc, N, N′ = MersenneTwister(123456), GPC(), 5, 6
        m = CustomMean(sin)
        k = SqExponentialKernel()
        f = wrap(GP(m, k), gpc)
        x = collect(range(-1.0, 1.0; length=N))
        x′ = collect(range(-1.0, 1.0; length=N′))

        @test mean_vector(f, x) == elementwise(m, x)
        @test cov(f, x) == kernelmatrix(k, x)
        @test cov_diag(f, x) == diag(cov(f, x))
        @test cov(f, x, x) == kernelmatrix(k, x, x)
        @test cov(f, x, x′) == kernelmatrix(k, x, x′)
        @test cov(f, x, x′) ≈ cov(f, x′, x)'
    end

    # Check that mean-function specialisations work as expected.
    @timedtestset "sugar" begin
        @test GP(5, SqExponentialKernel()).m isa ConstMean
        @test GP(SqExponentialKernel()).m isa ZeroMean
    end

    # Check that GP(kernel, mean, gpc) always works.
    @timedtestset "reversed construction" begin
        k = SqExponentialKernel()
        @test GP(k, CustomMean(sin)).m isa CustomMean{typeof(sin)}
        @test GP(k, CustomMean(sin)).k isa SqExponentialKernel
        @test GP(k, 5.0).m isa ConstMean
        @test GP(k, 5.0).k isa SqExponentialKernel
    end

    # Test the creation of indepenent GPs.
    @timedtestset "independent GPs" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x, x′ = randn(rng, N), randn(rng, N′)

        # Specification of two independent GPs
        gpc = GPC()
        m1, m2 = ZeroMean(), ConstMean(5)
        k1, k2 = SqExponentialKernel(), SqExponentialKernel()
        f1, f2 = wrap(GP(m1, k1), gpc), wrap(GP(m2, k2), gpc)
        @show typeof(f1)

        @test mean_vector(f1, x) == elementwise(m1, x)
        @test mean_vector(f2, x) == elementwise(m2, x)

        @test cov(f1, f2, x, x′) == zeros(N, N′)
        @test cov_diag(f1, x) == ones(N)

        @test cov(f1, f1, x′, x) ≈ cov(f1, f1, x, x′)'
    end
end
