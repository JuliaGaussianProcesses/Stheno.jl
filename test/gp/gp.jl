using Stheno: GPC

@timedtestset "gp" begin

    # Ensure that basic functionality works as expected.
    @timedtestset "GP" begin
        rng, gpc, N, N′ = MersenneTwister(123456), GPC(), 5, 6
        m = AbstractGPs.CustomMean(sin)
        k = SqExponentialKernel()
        f = wrap(GP(m, k), gpc)
        x = collect(range(-1.0, 1.0; length=N))
        x′ = collect(range(-1.0, 1.0; length=N′))

        @test mean(f, x) == AbstractGPs._map(m, x)
        @test cov(f, x) == kernelmatrix(k, x)
        @test cov_diag(f, x) == diag(cov(f, x))
        @test cov(f, x, x) == kernelmatrix(k, x, x)
        @test cov(f, x, x′) == kernelmatrix(k, x, x′)
        @test cov(f, x, x′) ≈ cov(f, x′, x)'
    end

    # Test the creation of indepenent GPs.
    @timedtestset "independent GPs" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x, x′ = randn(rng, N), randn(rng, N′)

        # Specification of two independent GPs
        gpc = GPC()
        m1, m2 = AbstractGPs.ZeroMean(), AbstractGPs.ConstMean(5)
        k1, k2 = SqExponentialKernel(), SqExponentialKernel()
        f1, f2 = wrap(GP(m1, k1), gpc), wrap(GP(m2, k2), gpc)

        @test mean(f1, x) == AbstractGPs._map(m1, x)
        @test mean(f2, x) == AbstractGPs._map(m2, x)

        @test cov(f1, f2, x, x′) == zeros(N, N′)
        @test cov_diag(f1, x) == ones(N)

        @test cov(f1, f1, x′, x) ≈ cov(f1, f1, x, x′)'
    end
end
