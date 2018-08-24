using Stheno: get_f, get_y, Observation, merge
using BlockArrays

@testset "conditioning" begin

# Test Observation functionality.
let
    rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
    X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
    y, y′ = randn(rng, N), randn(rng, N′)
    f = GP(ConstantMean(1), EQ(), GPC())

    fX, fX′ = f(X), f(X′)
    c1, c2 = fX←y, fX′←y′
    @test Observation(fX, y) == c1
    @test get_f(c1) === fX && get_f(c2) === fX′
    @test get_y(c1) === y && get_y(c2) === y′

    c = merge((c1, c2))
    @test get_y(c) == BlockVector([y, y′])
    @test get_f(c).fs == [fX, fX′]
end

let
    rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
    X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
    y = randn(rng, N)

    # Test mechanics for finite conditioned process with single conditioning.
    f = GP(ConstantMean(1), EQ(), GPC())
    f′ = f | (f(X) ← y)
    @test length(f′) == Inf
    @test length(rand(rng, f′(X))) == N
    @test maximum(abs.(rand(rng, f′(X)) - y)) < 1e-3
    @test maximum(abs.(mean_vec(f′(X)) - y)) < 1e-3
    @test all(abs.(cov(f′(X))) .< 1e-6)
end

# Some tests for self-consistency in the posterior when singly and multiply conditioning.
let
    rng, N, N′, D = MersenneTwister(123456), 1, 2, 2
    X_, X′_ = randn(rng, D, N), randn(rng, D, N′)
    X, X′, Z = ColsAreObs(X_), ColsAreObs(X′_), ColsAreObs(randn(rng, D, N + N′))
    μ, k, XX′ = ConstantMean(1.0), EQ(), ColsAreObs(hcat(X_, X′_))
    f = GP(μ, k, GPC())
    y = rand(rng, f(XX′))

    # Construct full posterior
    f′X = f(X) | (f(XX′) ← y)
    f′X′ = f(X′) | (f(XX′) ← y)
    f′XX′ = f(XX′) | (f(XX′) ← y)
    f′ = f | (f(XX′) ← y)

    # Check that each method is self-consistent.
    Σ′XX′ = cov(f′XX′)
    @test Σ′XX′ ≈ cov(f′(XX′))
    @test Σ′XX′[1:N, 1:N] ≈ cov(f′X)[1:N, 1:N]
    @test Σ′XX′[N+1:N+N′, N+1:N+N′] ≈ cov(f′X′)[1:N′, 1:N′]
    # @test maximum(abs.(Σ′XX′[1:N, N+1:N+N′] - xcov(f′X, f′X′, 1:N, 1:N′))) < 1e-12
    # @test maximum(abs.(Σ′XX′[N+1:N+N′, 1:N] - xcov(f′X′, f′X, 1:N′, 1:N))) < 1e-12

    # Test that conditioning works for BlockGPs.
    fb, Xb = BlockGP([f, f]), BlockData([X, X′])
    Zb = BlockData([Z, Z])
    fb′ = fb | (fb(Xb)←BlockArray(y, [N, N′]))
    @test mean_vec(fb′(Zb)) ≈ mean_vec(f′(Zb))
    @test maximum(abs.(cov(fb′(Zb)) - cov(f′(Zb)))) < 1e-6

    f′b = f | (fb(Xb)←BlockArray(y, [N, N′]))
    @test mean_vec(f′b(X)) ≈ mean_vec(f′X)
    @test maximum(abs.(cov(f′b(Zb)) - cov(f′(Zb)))) < 1e-6

    # Test sugar for multiple-conditioning.
    @test mean_vec(fb′(Zb)) ≈ mean_vec((f | (f(X)←y[1:N], f(X′)←y[N+1:end]))(Zb))
    @test maximum(abs.(cov(fb′(Zb)) - cov((f | (f(X)←y[1:N], f(X′)←y[N+1:end]))(Zb)))) < 1e-6

    yX = rand(rng, f(X))
    f′X, f′X′ = (f(X), f(X′)) | (f(X)←yX)
    f′X1, f′X′1 = f(X) | (f(X)←yX), f(X′) | (f(X)←yX)
    @test mean_vec(f′X) ≈ mean_vec(f′X1)
    @test mean_vec(f′X′) ≈ mean_vec(f′X′1)
    @test maximum(abs.(cov(f′X) - cov(f′X1))) < 1e-6
    @test maximum(abs.(cov(f′X′) - cov(f′X′1))) < 1e-6

    yX, yX′ = y[1:N], y[N+1:end]
    f′X, f′X′ = (f(X), f(X′)) | (f(X)←yX, f(X′)←yX′)
    f′X1, f′X′1 = f(X) | (f(X)←yX, f(X′)←yX′), f(X′) | (f(X)←yX, f(X′)←yX′)
    @test mean_vec(f′X) ≈ mean_vec(f′X1)
    @test mean_vec(f′X′) ≈ mean_vec(f′X′1)
    @test maximum(abs.(cov(f′X) - cov(f′X1))) < 1e-6
    @test maximum(abs.(cov(f′X′) - cov(f′X′1))) < 1e-6
end

# Test Titsias implementation by checking that it (approximately) recovers exact inference
# when M = N and Z = X.
let
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
    @test isapprox(μᵤ, mean_vec(f′XX′); rtol=1e-4)
    @test isapprox(Σᵤᵤ, cov(f′XX′); rtol=1e-4)

    # Compute conditioner and exact posterior compute at test points.
    conditioner = Stheno.Titsias(f(XX′), μᵤ, Σᵤᵤ)
    f′Z = f(Z) | (y(XX′)←ŷ)
    f′Z_approx = f(Z) | conditioner

    # Check that exact and approximate posteriors match up.
    @test isapprox(mean_vec(f′Z), mean_vec(f′Z_approx); rtol=1e-4)
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

    @test isapprox(mean_vec(f′Z), mean_vec(f′Zb); rtol=1e-4)
    @test isapprox(cov(f′Z), cov(f′Zb); rtol=1e-4)
end

end
