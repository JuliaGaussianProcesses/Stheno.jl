@testset "product" begin

    # Test the multiplication of a GP by a constant.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        X, X′ = randn(rng, D, N), randn(rng, D, N′)
        g1, c, c′ = GP(ConstantMean(1.0), EQ(), GPC()), -4.3, 2.1
        g2, g2′ = c * g1, g1 * c′
        g3, g3′ = c * g2, g2′ * c′
        g4, g4′ = c * g3, g3′ * c′

        @test mean_vec(g2(X)) == c .* mean_vec(g1(X))
        @test mean_vec(g3(X)) == c .* mean_vec(g2(X))
        @test mean_vec(g4(X)) == c .* mean_vec(g3(X))
        @test mean_vec(g2′(X)) == mean_vec(g1(X)) .* c′
        @test mean_vec(g3′(X)) == mean_vec(g2′(X)) .* c′
        @test mean_vec(g4′(X)) == mean_vec(g3′(X)) .* c′

        @test cov(g2(X)) ≈ c^2 .* cov(g1(X))
        @test cov(g3(X)) ≈ c^2 .* cov(g2(X))
        @test cov(g4(X)) ≈ c^2 .* cov(g3(X))
        @test cov(g2′(X)) ≈ c′^2 .* cov(g1(X))
        @test cov(g3′(X)) ≈ c′^2 .* cov(g2′(X))
        @test cov(g4′(X)) ≈ c′^2 .* cov(g3′(X))

        @test xcov(g2(X), g1(X′)) ≈ c .* xcov(g1(X), g1(X′))
        @test xcov(g3(X), g1(X′)) ≈ c^2 .* xcov(g1(X), g1(X′))
        @test xcov(g4(X), g1(X′)) ≈ c^3 .* xcov(g1(X), g1(X′))
        @test xcov(g2′(X), g1(X′)) ≈ c′ .* xcov(g1(X), g1(X′))
        @test xcov(g3′(X), g1(X′)) ≈ c′^2 .* xcov(g1(X), g1(X′))
        @test xcov(g4′(X), g1(X′)) ≈ c′^3 .* xcov(g1(X), g1(X′))

        @test xcov(g2(X), g2′(X′)) ≈ (c * c′) .* xcov(g1(X), g1(X′))
        @test xcov(g3(X), g3′(X′)) ≈ (c * c′)^2 .* xcov(g1(X), g1(X′))
        @test xcov(g4(X), g4′(X′)) ≈ (c * c′)^3 .* xcov(g1(X), g1(X′))

        @test xcov(g2(X), g3′(X′)) ≈ (c * c′^2) .* xcov(g1(X), g1(X′))
        @test xcov(g3(X), g2′(X′)) ≈ (c^2 * c′) .* xcov(g1(X), g1(X′))
        @test xcov(g2(X), g4′(X′)) ≈ (c * c′^3) .* xcov(g1(X), g1(X′))
        @test xcov(g4(X), g2′(X′)) ≈ (c^3 * c′) .* xcov(g1(X), g1(X′))
    end
end
