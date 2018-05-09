let
    rng, N, N′, D = MersenneTwister(123456), 5, 6,  2
    X, X′ = randn(rng, D, N), randn(rng, D, N′)
    y = randn(rng, N)

    # Test mechanics for finite conditioned process with single conditioning.
    f = GP(ConstantMean(1), EQ(), GPC())
    f′ = f | (f(X) ← y)
    @test length(f′) == Inf
    @test length(rand(rng, f′, X)) == N
    @test maximum(rand(rng, f′, X) - y) < 1e-3
    @test mean(f′, X) ≈ y
    @test all(abs.(Matrix(cov(kernel(f′), X))) .< 1e-8)
end
