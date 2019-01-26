# Test Titsias implementation by checking that it (approximately) recovers exact inference
# when M = N and Z = X.
@testset "approximate conditioning" begin
    rng, N, N′, D, σ² = MersenneTwister(123456), 2, 3, 5, 1e-1
    X_, X′_ = randn(rng, D, N), randn(rng, D, N′)
    X, X′, Z = ColsAreObs(X_), ColsAreObs(X′_), ColsAreObs(randn(rng, D, N + N′))
    μ, k, XX′ = ConstantMean(1.0), EQ(), ColsAreObs(hcat(X_, X′_))

    # Construct toy problem.
    gpc = GPC()
    f = GP(μ, k, gpc)
    y = f + GP(Noise(σ²), gpc)
    ŷ = rand(rng, y(XX′))

    # Compute exact posterior.
    f′XX′ = f(XX′) | (y(XX′)←ŷ)

    # Compute approximate posterior suff. stats.
    μᵤ, Σᵤᵤ = Stheno.optimal_q(f(XX′), ŷ, f(XX′), sqrt(σ²))

    # Check that exact and approx. posteriors are close in this case.
    @test isapprox(μᵤ, mean(f′XX′); rtol=1e-4)
    @test isapprox(Σᵤᵤ, cov(f′XX′); rtol=1e-4)

    # Compute conditioner and exact posterior compute at test points.
    conditioner = Stheno.Titsias(f(XX′), μᵤ, Σᵤᵤ)
    f′Z = f(Z) | (y(XX′)←ŷ)
    f′Z_approx = f(Z) | conditioner

    # Check that exact and approximate posteriors match up.
    @test isapprox(mean(f′Z), mean(f′Z_approx); rtol=1e-4)
    @test isapprox(cov(f′Z), cov(f′Z_approx); rtol=1e-4)


    # Check that Titsias with BlockGP works the same as Titsias with regular GP.
    ŷX, ŷX′ = ŷ[1:N], ŷ[N+1:end]

    fb, ŷb = BlockGP([f(X), f(X′)]), BlockVector([ŷX, ŷX′])
    μb, Σb = Stheno.optimal_q(fb, ŷb, fb, sqrt(σ²))

    @test μb isa BlockVector
    @test Σb isa LazyPDMat
    @test Stheno.unbox(Σb) isa Symmetric
    @test Stheno.unbox(Stheno.unbox(Σb)) isa AbstractBlockMatrix
    @test μb ≈ μᵤ
    @test Σb ≈ Σᵤᵤ

    # Test that conditioning is indifferent to choice of Blocks.
    conditioner_blocked = Stheno.Titsias(fb, μb, Σb)
    f′Zb = f(BlockData([Z])) | conditioner_blocked

    @test isapprox(mean(f′Z), mean(f′Zb); rtol=1e-4)
    @test isapprox(cov(f′Z), cov(f′Zb); rtol=1e-4)
end
