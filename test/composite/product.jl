using Stheno: GPC, EQ

@testset "product" begin
    @testset "GP mul errors" begin
        gpc = GPC()
        f1, f2 = GP(EQ(), gpc), GP(EQ(), gpc)
        @test_throws ArgumentError f1 * f2
    end
    @testset "multiply by constant" begin
        rng, N, N′, D = MersenneTwister(123456), 3, 5, 2
        X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        g1, c, c′ = GP(1, EQ(), GPC()), -4.3, 2.1
        g2, g2′ = c * g1, g1 * c′
        g3, g3′ = c * g2, g2′ * c′
        g4, g4′ = c * g3, g3′ * c′

        @test mean(g2(X)) == c .* mean(g1(X))
        @test mean(g3(X)) == c .* mean(g2(X))
        @test mean(g4(X)) == c .* mean(g3(X))
        @test mean(g2′(X)) == mean(g1(X)) .* c′
        @test mean(g3′(X)) == mean(g2′(X)) .* c′
        @test mean(g4′(X)) == mean(g3′(X)) .* c′

        @test cov(g2(X)) ≈ c^2 .* cov(g1(X))
        @test cov(g3(X)) ≈ c^2 .* cov(g2(X))
        @test cov(g4(X)) ≈ c^2 .* cov(g3(X))
        @test cov(g2′(X)) ≈ c′^2 .* cov(g1(X))
        @test cov(g3′(X)) ≈ c′^2 .* cov(g2′(X))
        @test cov(g4′(X)) ≈ c′^2 .* cov(g3′(X))

        @test cov(g2(X), g1(X′)) ≈ c .* cov(g1(X), g1(X′))
        @test cov(g3(X), g1(X′)) ≈ c^2 .* cov(g1(X), g1(X′))
        @test cov(g4(X), g1(X′)) ≈ c^3 .* cov(g1(X), g1(X′))
        @test cov(g2′(X), g1(X′)) ≈ c′ .* cov(g1(X), g1(X′))
        @test cov(g3′(X), g1(X′)) ≈ c′^2 .* cov(g1(X), g1(X′))
        @test cov(g4′(X), g1(X′)) ≈ c′^3 .* cov(g1(X), g1(X′))

        @test cov(g2(X), g2′(X′)) ≈ (c * c′) .* cov(g1(X), g1(X′))
        @test cov(g3(X), g3′(X′)) ≈ (c * c′)^2 .* cov(g1(X), g1(X′))
        @test cov(g4(X), g4′(X′)) ≈ (c * c′)^3 .* cov(g1(X), g1(X′))

        @test cov(g2(X), g3′(X′)) ≈ (c * c′^2) .* cov(g1(X), g1(X′))
        @test cov(g3(X), g2′(X′)) ≈ (c^2 * c′) .* cov(g1(X), g1(X′))
        @test cov(g2(X), g4′(X′)) ≈ (c * c′^3) .* cov(g1(X), g1(X′))
        @test cov(g4(X), g2′(X′)) ≈ (c^3 * c′) .* cov(g1(X), g1(X′))

        @testset "Consistency Tests" begin
            rng, P, Q = MersenneTwister(123456), 3, 5
            x0 = collect(range(-1.0, 1.0; length=P))
            x1 = collect(range(-0.5, 1.5; length=Q))
            x2, x3 = randn(rng, Q), randn(rng, P)

            gpc = GPC()
            f1 = GP(cos, EQ(), gpc)
            f2 = 5 * f1
            abstractgp_interface_tests(f2, f1, x0, x1, x2, x3)
        end
        @testset "Diff Tests" begin
            standard_1D_tests(
                MersenneTwister(123456),
                Dict(:σ=>2.3),
                θ->begin
                    f = GP(0.5, EQ(), GPC())
                    return θ[:σ] * f, f
                end,
                X, X′,
            )
        end
    end
    @testset "multiply by function" begin
        rng, N, N′, D = MersenneTwister(123456), 3, 5, 2
        X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        g1, f, f′ = GP(1, EQ(), GPC()), x->sum(sin, x), x->sum(cos, x)
        g2, g2′ = f * g1, g1 * f′
        g3, g3′ = f * g2, g2′ * f′
        g4, g4′ = f * g3, g3′ * f′

        @test mean(g2(X)) == map(f, X) .* mean(g1(X))
        @test mean(g3(X)) == map(f, X) .* mean(g2(X))
        @test mean(g4(X)) == map(f, X) .* mean(g3(X))
        @test mean(g2′(X)) == mean(g1(X)) .* map(f′, X)
        @test mean(g3′(X)) == mean(g2′(X)) .* map(f′, X)
        @test mean(g4′(X)) == mean(g3′(X)) .* map(f′, X)

        fX, f′X = map(f, X), map(f′, X)
        @test cov(g2(X)) ≈ fX .* cov(g1(X)) .* fX'
        @test cov(g3(X)) ≈ fX .* cov(g2(X)) .* fX'
        @test cov(g4(X)) ≈ fX .* cov(g3(X)) .* fX'
        @test cov(g2′(X)) ≈ f′X .* cov(g1(X)) .* f′X'
        @test cov(g3′(X)) ≈ f′X .* cov(g2′(X)) .* f′X'
        @test cov(g4′(X)) ≈ f′X .* cov(g3′(X)) .* f′X'

        fX′, f′X′ = map(f, X′), map(f′, X′)
        @test cov(g2(X), g1(X′)) ≈ fX .* cov(g1(X), g1(X′))
        @test cov(g3(X), g1(X′)) ≈ fX.^2 .* cov(g1(X), g1(X′))
        @test cov(g4(X), g1(X′)) ≈ fX.^3 .* cov(g1(X), g1(X′))
        @test cov(g2′(X), g1(X′)) ≈ f′X .* cov(g1(X), g1(X′))
        @test cov(g3′(X), g1(X′)) ≈ (f′X.^2) .* cov(g1(X), g1(X′))
        @test cov(g4′(X), g1(X′)) ≈ (f′X.^3) .* cov(g1(X), g1(X′))

        @test cov(g1(X′), g2(X)) == cov(g2(X), g1(X′))'
        @test cov(g1(X′), g3(X)) == cov(g3(X), g1(X′))'
        @test cov(g1(X′), g4(X)) == cov(g4(X), g1(X′))'

        @test cov(g2(X), g2′(X′)) ≈ fX .* cov(g1(X), g1(X′)) .* f′X′'
        @test cov(g3(X), g3′(X′)) ≈ fX.^2 .* cov(g1(X), g1(X′)) .* (f′X′.^2)'
        @test cov(g4(X), g4′(X′)) ≈ fX.^3 .* cov(g1(X), g1(X′)) .* (f′X′.^3)'

        @test cov(g2(X), g3′(X′)) ≈ fX .* cov(g1(X), g1(X′)) .* (f′X′').^2
        @test cov(g3(X), g2′(X′)) ≈ fX.^2 .* cov(g1(X), g1(X′)) .* (f′X′')
        @test cov(g2(X), g4′(X′)) ≈ fX .* cov(g1(X), g1(X′)) .* (f′X′').^3
        @test cov(g4(X), g2′(X′)) ≈ fX.^3 .* cov(g1(X), g1(X′)) .* (f′X′')

        @testset "Consistency Tests" begin
            rng, P, Q = MersenneTwister(123456), 3, 5
            x0 = collect(range(-1.0, 1.0; length=P))
            x1 = collect(range(-0.5, 1.5; length=Q))
            x2, x3 = randn(rng, Q), randn(rng, P)

            gpc = GPC()
            f1 = GP(cos, EQ(), gpc)
            f2 = sin * f1
            abstractgp_interface_tests(f2, f1, x0, x1, x2, x3)
        end
        @testset "Diff Tests" begin
            standard_1D_tests(
                MersenneTwister(123456),
                Dict(:σ=>2.3, :c=>1.3),
                θ->begin
                    f = GP(θ[:c], EQ(), GPC())
                    return (x->θ[:σ] * x) * f, f
                end,
                collect(range(-2.0, 2.0; length=N)),
                collect(range(-1.5, 2.5; length=N′)),
            )
        end
    end
end
