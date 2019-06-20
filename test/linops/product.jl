using Stheno: GPC

@testset "product" begin

    @testset "multiply by constant" begin
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        g1, c, c′ = GP(1, eq(), GPC()), -4.3, 2.1
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

        @testset "Standardised Tests" begin
            function foo(θ)
                f = GP(sin, eq(l=θ[:l]), GPC())
                return θ[:σ] * f, f
            end
            x, A = collect(range(-5.0, 5.0; length=N)), randn(rng, N, N)
            z, B = collect(range(-5.0, 5.0; length=N′)), randn(rng, N′, N′)
            θ = Dict(:l=>0.5, :σ=>2.3)
            y = rand(rng, first(foo(θ))(x, _to_psd(A)))
            check_consistency(rng, θ, foo, x, y, A, _to_psd, z, B)
        end
    end
    @testset "multiply by function" begin
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        g1, f, f′ = GP(1, eq(), GPC()), x->sum(sin, x), x->sum(cos, x)
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
    end
end
