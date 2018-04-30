let
    rng, N, N′, D = MersenneTwister(123456), 5, 6,  2
    X, X′ = randn(rng, N, D), randn(rng, N′, D)
    y = randn(rng, N)

    # Test mechanics for finite conditioned process with single conditioning.
    f = GP(ConstantMean(1), EQ(), GPC())
    f′ = f | (f(X) ← y)
    @test length(f′) == Inf
    @test length(rand(rng, f′, X)) == N
    @test maximum(rand(rng, f′, X) - y) < 1e-5
    @test mean(f′, X) ≈ y
    @test all(abs.(Matrix(cov(kernel(f′), X))) .< 1e-9)
end
