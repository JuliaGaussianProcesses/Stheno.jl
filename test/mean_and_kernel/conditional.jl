@testset "posterior" begin

    using Stheno: CondCache, ConditionalMean, ConditionalKernel, ConditionalCrossKernel
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        X, X′ = randn(rng, N, D), randn(rng, N′, D)
        μf, μg = ConstantMean(1.0), ZeroMean{Float64}()
        kff, kgg, kfg = EQ(), EQ(), ZeroKernel{Float64}()

        #= Test that observing the mean function shrinks the posterior covariance
           appropriately, but leaves the posterior mean at the prior mean (modulo noise) =# 
        cache = CondCache(FiniteKernel(kff, X), FiniteMean(μf, X), ones(Float64, N))
        kffX = LhsFiniteCrossKernel(kff, X)
        @test cache.α ≈ zeros(Float64, N)

        # Means at data and away from data.
        @test mean(ConditionalMean(
            cache,
            FiniteMean(μf, X),
            FiniteCrossKernel(kffX, X)), # == FiniteCrossKernel(kff, X, X)
        ) ≈ mean(FiniteMean(μf, X))
        @test mean(ConditionalMean(cache, μf, kffX), X) ≈ mean(μf, X)
        @test mean(ConditionalMean(
            cache,
            FiniteMean(μf, X′),
            FiniteCrossKernel(kffX, X′), # == FiniteCrossKernel(kff, X, X′)
        )) ≈ mean(FiniteMean(μf, X′))
        @test mean(ConditionalMean(cache, μf, kffX), X′) ≈ mean(μf, X′)

        # Covariance at data should be (close to) zero.
        @test all(abs.(Matrix(cov(ConditionalKernel(
            cache,
            FiniteCrossKernel(kffX, X), # == FiniteCrossKernel(kff, X, X)
            FiniteKernel(kff, X),
        )))) .< 1e-10)

        # Posterior covariance of indep. process should be same as prior.
        kfgX = LhsFiniteCrossKernel(kfg, X)
        @test mean(ConditionalMean(
            cache,
            FiniteMean(μg, X),
            FiniteCrossKernel(kfgX, X),
        )) ≈ mean(FiniteMean(μg, X))
        @test mean(ConditionalMean(cache, μg, kfgX), X) ≈ mean(μg, X)
        @test mean(ConditionalMean(
            cache,
            FiniteMean(μg, X′),
            FiniteCrossKernel(kfgX, X′),
        )) ≈ mean(FiniteMean(μg, X′))
        @test mean(ConditionalMean(cache, μg, kfgX), X′) ≈ mean(μg, X′)
        @test cov(ConditionalKernel(
            cache,
            FiniteCrossKernel(kfgX, X),
            FiniteKernel(kgg, X),
        )) ≈ cov(FiniteKernel(kgg, X))
        @test cov(ConditionalKernel(cache, kfgX, kgg), X) ≈ cov(kgg, X)
        @test cov(ConditionalKernel(
            cache,
            FiniteCrossKernel(kfgX, X′),
            FiniteKernel(kgg, X′),
        )) ≈ cov(FiniteKernel(kgg, X′))
        @test cov(ConditionalKernel(cache, kfgX, kgg), X′) ≈ cov(kgg, X′)
        @test xcov(ConditionalKernel(cache, kfgX, kgg), X, X′) ≈ xcov(kgg, X, X′)
        @test xcov(ConditionalKernel(cache, kfgX, kgg), X′, X) ≈ xcov(kgg, X′, X)

        # Should be able to get similar results using a ConditionalCrossKernel.
        @test xcov(ConditionalCrossKernel(cache, kffX, kffX, kff), X, X′) ≈
            xcov(ConditionalCrossKernel(
                cache,
                FiniteCrossKernel(kffX, X),
                FiniteCrossKernel(kffX, X′),
                FiniteCrossKernel(kff, X, X′),
            ))

        # Posterior xcov between two independent processes, conditioned on observations of
        # one of them should be independent.
        @test xcov(ConditionalCrossKernel(
            cache,
            FiniteCrossKernel(kffX, X),
            FiniteCrossKernel(kfgX, X),
            FiniteCrossKernel(kfg, X, X),
        )) ≈ xcov(FiniteCrossKernel(kfg, X, X))
        @test xcov(ConditionalCrossKernel(cache, kffX, kfgX, kfg), X, X) ≈ xcov(kfg, X, X)
        @test xcov(ConditionalCrossKernel(
            cache,
            FiniteCrossKernel(kffX, X),
            FiniteCrossKernel(kfgX, X′),
            FiniteCrossKernel(kfg, X, X′),
        )) ≈ xcov(FiniteCrossKernel(kfg, X, X′))
        @test xcov(ConditionalCrossKernel(cache, kffX, kfgX, kfg), X, X′) ≈ xcov(kfg, X, X′)
        @test xcov(ConditionalCrossKernel(
            cache,
            FiniteCrossKernel(kfgX, X′),
            FiniteCrossKernel(kfgX, X′),
            FiniteKernel(kgg, X′),
        )) ≈ xcov(FiniteKernel(kgg, X′))
        @test xcov(ConditionalCrossKernel(cache, kfgX, kfgX, kgg), X′, X′) ≈ xcov(kgg, X′, X′)
    end
end
