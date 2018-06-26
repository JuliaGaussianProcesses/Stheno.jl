@testset "addition" begin
    let
        rng, N, N′, D, gpc = MersenneTwister(123456), 5, 6, 2, GPC()
        X, X′ = DataSet(randn(rng, D, N)), DataSet(randn(rng, D, N′))
        μ1, μ2, k1, k2 = ConstantMean(1.0), ConstantMean(2.0), EQ(), Linear(1.0)
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
        f3 = f1 + f2
        f4 = f1 + f3
        f5 = f3 + f4

        for (n, (fp, fa, fb)) in enumerate([(f3, f1, f2), (f4, f1, f3), (f5, f3, f4)])
            Σp = cov(fa(X)) + cov(fb(X)) + xcov(fa(X), fb(X)) + xcov(fb(X), fa(X))
            ΣpXX′ = xcov(fa(X), fa(X′)) + xcov(fb(X), fb(X′)) + xcov(fa(X), fb(X′)) +
                xcov(fa(X), fb(X′))
            @test mean(fp) == Stheno.CompositeMean(+, mean(fa), mean(fb))
            @test mean_vec(fp(X)) ≈ mean_vec(fa(X)) + mean_vec(fb(X))
            @test cov(fp(X)) ≈ Σp
            @test xcov(fp(X), fp(X′)) ≈ ΣpXX′
            @test xcov(fp(X′), fp(X)) ≈ transpose(ΣpXX′)                
            @test xcov(fp(X), fa(X′)) ≈ xcov(fa(X), fa(X′)) + xcov(fb(X), fa(X′))
            @test xcov(fp(X′), fa(X)) ≈ xcov(fa(X′), fa(X)) + xcov(fb(X′), fa(X))
            @test xcov(fa(X), fp(X′)) ≈ xcov(fb(X), fa(X′)) + xcov(fa(X), fa(X′))
            @test xcov(fa(X′), fp(X)) ≈ xcov(fb(X′), fa(X)) + xcov(fa(X′), fa(X))
        end
    end

    let
        rng, N, N′, D, gpc = MersenneTwister(123456), 5, 6, 2, GPC()
        X, X′ = DataSet(randn(rng, D, N)), DataSet(randn(rng, D, N′))
        μ1, μ2, k1, k2 = ConstantMean(1.0), ConstantMean(2.0), EQ(), Linear(1.0)
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
        fj1, fj2 = JointGP([f1, f2]), JointGP([f2, f1])
        @test mean_vec((fj1 + fj2)(BlockData([X, X′]))) ==
            BlockVector([mean_vec((f1 + f2)(X)), mean_vec((f2 + f1)(X′))])
    end
end
