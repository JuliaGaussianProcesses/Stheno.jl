@testset "normal" begin

    # Test a generic toy problem.
    let
        rng = MersenneTwister(123456)
        N, S = 5, 100000
        μ, Σ = randn(rng, N), cov(EQ(), randn(rng, N))
        d = Normal(μ, Σ)

        @test mean(d) == μ
        @test cov(d) == Σ
        @test dims(d) == N

        x̂ = sample(rng, d, S)
        @test size(x̂) == (N, S)
        @test maximum(abs.(mean(x̂, 2) - mean(d))) < 1e-2
        @test maximum(abs.(cov(x̂, 2) - full(Σ))) < 1e-2
    end

    # Test a constant-mean toy problem.
    let
        rng = MersenneTwister(123456)
        N, S = 5, 100000
        μ, Σ = randn(rng), cov(EQ(), randn(rng, N))
        d = Normal(μ, Σ)

        @test mean(d) == μ
        @test cov(d) == Σ
        @test dims(d) == N

        x̂ = sample(rng, d, S)
        @test size(x̂) == (N, S)
        @test maximum(abs.(mean(x̂, 2) - mean(d))) < 1e-2
        @test maximum(abs.(cov(x̂, 2) - full(Σ))) < 1e-2
    end
end
