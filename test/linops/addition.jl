using Stheno: GPC

@testset "addition" begin
    @testset "Correlated GPs" begin
        rng, N, N′, D, gpc = MersenneTwister(123456), 5, 6, 2, GPC()
        X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        f1, f2 = GP(1, eq(), gpc), GP(2, linear(), gpc)
        f3 = f1 + f2
        f4 = f1 + f3
        f5 = f3 + f4

        for (n, (fp, fa, fb)) in enumerate([(f3, f1, f2), (f4, f1, f3), (f5, f3, f4)])
            Σp = cov(fa(X)) + cov(fb(X)) + cov(fa(X), fb(X)) + cov(fb(X), fa(X))
            ΣpXX′ = cov(fa(X), fa(X′)) + cov(fb(X), fb(X′)) + cov(fa(X), fb(X′)) +
                cov(fa(X), fb(X′))
            @test mean(fp(X)) ≈ mean(fa(X)) + mean(fb(X))
            @test cov(fp(X)) ≈ Σp
            @test cov(fp(X), fp(X′)) ≈ ΣpXX′
            @test cov(fp(X′), fp(X)) ≈ transpose(ΣpXX′)                
            @test cov(fp(X), fa(X′)) ≈ cov(fa(X), fa(X′)) + cov(fb(X), fa(X′))
            @test cov(fp(X′), fa(X)) ≈ cov(fa(X′), fa(X)) + cov(fb(X′), fa(X))
            @test cov(fa(X), fp(X′)) ≈ cov(fb(X), fa(X′)) + cov(fa(X), fa(X′))
            @test cov(fa(X′), fp(X)) ≈ cov(fb(X′), fa(X)) + cov(fa(X′), fa(X))
        end
    end
    @testset "Verify mean / kernel numerically" begin
        rng, N, D = MersenneTwister(123456), 5, 6, 2
        X = ColsAreObs(randn(rng, D, N))
        c, f = randn(rng), GP(5, eq(), GPC())

        @test mean((f + c)(X)) == mean(f(X)) .+ c
        @test mean((f + c)(X)) == c .+ mean(f(X))
        @test cov((f + c)(X)) == cov(f(X))
        @test cov((c + f)(X)) == cov(f(X))

        @test mean((f - c)(X)) == mean(f(X)) .- c
        @test mean((c - f)(X)) == c .- mean(f(X))
        @test cov((f - c)(X)) == cov(f(X))
        @test cov((c - f)(X)) == cov(f(X))

        x = randn(rng, N + D)
        @test mean((f + sin)(x)) == mean(f(x)) + map(sin, x)
        @test mean((sin + f)(x)) == map(sin, x) + mean(f(x))
        @test cov((f + sin)(x)) == cov(f(x))
        @test cov((sin + f)(x)) == cov(f(x))
    end
    @testset "Standardised Tests (independent sum)" begin
        standard_1D_tests(
            MersenneTwister(123456),
            Dict(:l1=>0.5, :l2=>2.3),
            θ->begin
                gpc = GPC()
                f1 = GP(sin, eq(l=θ[:l1]), gpc)
                f2 = GP(cos, eq(l=θ[:l2]), gpc)
                f3 = f1 + f2
                return f3, f3
            end,
            13, 11,
        )
    end
    @testset "Standardised Tests (correlated sum)" begin
        standard_1D_tests(
            MersenneTwister(123456),
            Dict(:l1=>0.5, :l2=>2.3),
            θ->begin
                gpc = GPC()
                f1 = GP(sin, eq(l=θ[:l1]), gpc)
                f2 = GP(cos, eq(l=θ[:l2]), gpc)
                f3 = f1 + f2
                f4 = f1 + f3
                f5 = f3 + f4
                return f5, f5
            end,
            13, 11,
        )
    end
end
