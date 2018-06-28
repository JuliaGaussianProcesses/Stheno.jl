@testset "conditioning" begin

let
    rng, N, N′, D = MersenneTwister(123456), 5, 6,  2
    X, X′ = MatData(randn(rng, D, N)), MatData(randn(rng, D, N′))
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

# Some tests for self-consistency in the posterior when doing single-conditioning.
let
    rng, N, N′, D = MersenneTwister(123456), 2, 3, 2
    X_, X′_ = randn(rng, D, N), randn(rng, D, N′)
    X, X′, Z = MatData(X_), MatData(X′_), MatData(randn(rng, D, N + N′))
    μ, k, XX′ = ConstantMean(1.0), EQ(), MatData(hcat(X_, X′_))
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

    # Test that conditioning works for JointGPs.
    fb, Xb = JointGP([f, f]), BlockData([X, X′])
    Zb = BlockData([Z, Z])
    fb′ = fb | (fb(Xb)←BlockArray(y, [N, N′]))
    @test mean_vec(fb′(Zb)) ≈ mean_vec(f′(Zb))
    @test cov(fb′(Zb)) ≈ cov(f′(Zb))

    # f′b = f | (fb(Xb)←BlockArray(y, [N, N′]))
    # @test mean_vec(f′b(X)) ≈ mean_vec(f′X)
end

end

# @testset "approximate conditioning" begin
    
# end
