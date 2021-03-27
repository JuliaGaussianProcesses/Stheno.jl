@timedtestset "indexing" begin
    @timedtestset "no noise" begin
        f, x = GP(SqExponentialKernel(), GPC()), randn(17)
        fx = f(x)
        @test mean(fx) == mean_vector(f, x)
        @test cov(fx) == cov(f, x)
    end
    @timedtestset "(Symmetric) Matrix-valued noise" begin
        rng, N = MersenneTwister(123456), 13
        f, x, A = GP(SqExponentialKernel(), GPC()), randn(rng, N), randn(rng, N, N)
        C = Symmetric(A * A' + I)
        fx = f(x, C)
        @test mean(fx) == mean_vector(f, x)
        @test cov(fx) == cov(f, x) + C
    end
    @timedtestset "Vector-valued noise" begin
        rng, N = MersenneTwister(123456), 11
        f, x, a = GP(SqExponentialKernel(), GPC()), randn(rng, N), exp.(randn(rng, N))
        fx = f(x, a)
        @test mean(fx) == mean_vector(f, x)
        @test cov(fx) == cov(f, x) + Diagonal(a)
    end
    @timedtestset "Real-valued noise" begin
        rng, N = MersenneTwister(123456), 13
        f, x, σ² = GP(SqExponentialKernel(), GPC()), randn(rng, N), exp(randn(rng))
        fx = f(x, σ²)
        @test mean(fx) == mean_vector(f, x)
        @test cov(fx) == cov(f, x) + Diagonal(fill(σ², N))
    end
end
