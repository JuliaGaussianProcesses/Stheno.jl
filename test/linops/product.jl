@testset "product" begin

    # Test the multiplication of a GP by a constant.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        X, X′ = randn(rng, N, D), randn(rng, N′, D)
        g1, c, c′ = GP(ConstantMean(1.0), EQ(), GPC()), -4.3, 2.1
        g2, g2′ = c * g1, g1 * c′
        g3, g3′ = c * g2, g2′ * c′
        g4, g4′ = c * g3, g3′ * c′

        @test mean(g2, X) == c .* mean(g1, X)
        @test mean(g3, X) == c .* mean(g2, X)
        @test mean(g4, X) == c .* mean(g3, X)
        @test mean(g2′, X) == mean(g1, X) .* c′
        @test mean(g3′, X) == mean(g2′, X) .* c′
        @test mean(g4′, X) == mean(g3′, X) .* c′

        @test cov(g2, X) ≈ c^2 .* cov(g1, X)
        @test cov(g3, X) ≈ c^2 .* cov(g2, X)
        @test cov(g4, X) ≈ c^2 .* cov(g3, X)
        @test cov(g2′, X) ≈ c′^2 .* cov(g1, X)
        @test cov(g3′, X) ≈ c′^2 .* cov(g2′, X)
        @test cov(g4′, X) ≈ c′^2 .* cov(g3′, X)

        @test xcov(g2, g1, X, X′) ≈ c .* xcov(g1, X, X′)
        @test xcov(g3, g1, X, X′) ≈ c^2 .* xcov(g1, X, X′)
        @test xcov(g4, g1, X, X′) ≈ c^3 .* xcov(g1, X, X′)
        @test xcov(g2′, g1, X, X′) ≈ c′ .* xcov(g1, X, X′)
        @test xcov(g3′, g1, X, X′) ≈ c′^2 .* xcov(g1, X, X′)
        @test xcov(g4′, g1, X, X′) ≈ c′^3 .* xcov(g1, X, X′)

        @test xcov(g2, g2′, X, X′) ≈ (c * c′) .* xcov(g1, X, X′)
        @test xcov(g3, g3′, X, X′) ≈ (c * c′)^2 .* xcov(g1, X, X′)
        @test xcov(g4, g4′, X, X′) ≈ (c * c′)^3 .* xcov(g1, X, X′)

        @test xcov(g2, g3′, X, X′) ≈ (c * c′^2) .* xcov(g1, X, X′)
        @test xcov(g3, g2′, X, X′) ≈ (c^2 * c′) .* xcov(g1, X, X′)
        @test xcov(g2, g4′, X, X′) ≈ (c * c′^3) .* xcov(g1, X, X′)
        @test xcov(g4, g2′, X, X′) ≈ (c^3 * c′) .* xcov(g1, X, X′)
    end
end
