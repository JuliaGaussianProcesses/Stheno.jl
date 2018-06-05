@testset "conditioning" begin

let
    rng, N, N′, D = MersenneTwister(123456), 5, 6,  2
    X, X′ = randn(rng, D, N), randn(rng, D, N′)
    y = randn(rng, N)

    # Test mechanics for finite conditioned process with single conditioning.
    f = GP(ConstantMean(1), EQ(), GPC())
    f′ = f | (f(X) ← y)
    @test length(f′) == Inf
    @test length(rand(rng, f′, X)) == N
    @test maximum(abs.(rand(rng, f′, X) - y)) < 1e-3
    @test maximum(abs.(mean(f′, X) - y)) < 1e-3
    @test all(abs.(Matrix(cov(kernel(f′), X))) .< 1e-6)
end

# Some tests for self-consistency in the posterior when doing single-conditioning.
let
    rng, N, N′, D = MersenneTwister(123456), 100, 50, 2
    X, X′ = randn(rng, D, N), randn(rng, D, N′)
    μ, k, XX′ = ConstantMean(1.0), EQ(), hcat(X, X′)
    f = GP(μ, k, GPC())
    y = rand(rng, f, XX′)

    # Construct full posterior
    f′X = f(X) | (f(XX′) ← y)
    f′X′ = f(X′) | (f(XX′) ← y)
    f′XX′ = f(XX′) | (f(XX′) ← y)
    f′ = f | (f(XX′) ← y)

    # Check that each method is self-consistent.
    Σ′XX′ = cov(f′XX′, 1:N+N′)
    @test Σ′XX′ ≈ cov(f′, XX′)
    @test Σ′XX′[1:N, 1:N] ≈ cov(f′X, 1:N)
    @test Σ′XX′[N+1:N+N′, N+1:N+N′] ≈ cov(f′X′, 1:N′)
    # @test maximum(abs.(Σ′XX′[1:N, N+1:N+N′] - xcov(f′X, f′X′, 1:N, 1:N′))) < 1e-12
    # @test maximum(abs.(Σ′XX′[N+1:N+N′, 1:N] - xcov(f′X′, f′X, 1:N′, 1:N))) < 1e-12
end

end

@testset "approximate conditioning" begin
    
end
