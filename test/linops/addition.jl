using Stheno: GPC

@testset "addition" begin

    @testset "correlated GPs" begin
        rng, N, N′, D, gpc = MersenneTwister(123456), 5, 6, 2, GPC()
        X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        f1, f2 = GP(1, eq(), gpc), GP(2, Linear(), gpc)
        f3 = f1 + f2
        f4 = f1 + f3
        f5 = f3 + f4

        for (n, (fp, fa, fb)) in enumerate([(f3, f1, f2), (f4, f1, f3), (f5, f3, f4)])
            Σp = cov(fa(X)) + cov(fb(X)) + cov(fa(X), fb(X)) + cov(fb(X), fa(X))
            ΣpXX′ = cov(fa(X), fa(X′)) + cov(fb(X), fb(X′)) + cov(fa(X), fb(X′)) +
                cov(fa(X), fb(X′))
            # @test mean(fp) == Stheno.CompositeMean(+, mean(fa), mean(fb))
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

    # let
    #     rng, N, N′, D, gpc = MersenneTwister(123456), 5, 6, 2, GPC()
    #     X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
    #     μ1, μ2, k1, k2 = ConstantMean(1.0), ConstantMean(2.0), eq(), Linear(1.0)

    #     # Addition of BlockGPs.
    #     f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
    #     fj1, fj2 = BlockGP([f1, f2]), BlockGP([f2, f1])
    #     @test mean((fj1 + fj2)(BlockData([X, X′]))) ==
    #         BlockVector([mean((f1 + f2)(X)), mean((f2 + f1)(X′))])

    #     # Addition of a BlockGP and a GP.
    #     @test mean((fj1 + f1)(BlockData([X, X′]))) ==
    #         BlockData([mean((f1 + f1)(X)), mean((f2 + f1)(X′))])

    #     # Addition of a GP and a BlockGP.
    #     @test mean((f2 + fj2)(BlockData([X, X′]))) ==
    #         BlockData([mean((f2 + f2)(X)), mean((f2 + f1)(X′))])
    # end

    @testset "verify mean / kernel numerically" begin
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
end
