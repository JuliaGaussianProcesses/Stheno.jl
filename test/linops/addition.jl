@testset "addition" begin
    let
        rng, N, N′, D, gpc = MersenneTwister(123456), 5, 6, 2, GPC()
        X, X′ = randn(rng, D, N), randn(rng, D, N′)
        μ1, μ2, k1, k2 = ConstantMean(1.0), ConstantMean(2.0), EQ(), Linear(1.0)
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
        f3 = f1 + f2
        f4 = f1 + f3
        f5 = f3 + f4

        for (n, (fp, fa, fb)) in enumerate([(f3, f1, f2), (f4, f1, f3), (f5, f3, f4)])
            Σp = cov(fa, X) + cov(fb, X) + xcov(fa, fb, X) + xcov(fb, fa, X)
            ΣpXX′ = xcov(fa, X, X′) + xcov(fb, X, X′) + xcov(fa, fb, X, X′) +
                xcov(fa, fb, X, X′)
            @test mean(fp) == Stheno.CompositeMean(+, mean(fa), mean(fb))
            @test mean(fp, X) ≈ mean(fa, X) + mean(fb, X)
            @test cov(fp, X) ≈ Σp
            @test xcov(fp, X, X′) ≈ ΣpXX′
            @test xcov(fp, X′, X) ≈ transpose(ΣpXX′)                
            @test xcov(fp, fa, X, X′) ≈ xcov(fa, X, X′) + xcov(fb, fa, X, X′)
            @test xcov(fp, fa, X′, X) ≈ xcov(fa, X′, X) + xcov(fb, fa, X′, X)
            @test xcov(fa, fp, X, X′) ≈ xcov(fb, fa, X, X′) + xcov(fa, X, X′)
            @test xcov(fa, fp, X′, X) ≈ xcov(fb, fa, X′, X) + xcov(fa, X′, X)
        end
    end
end
