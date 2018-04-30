using Stheno: CondCache, ConditionalMean, ConditionalKernel, ConditionalCrossKernel

@testset "conditional" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        X, X′ = randn(rng, N, D), randn(rng, N′, D)
        μf, μg = ConstantMean(1.0), ZeroMean{Float64}()
        kff, kgg, kfg = EQ(), EQ(), ZeroKernel{Float64}()

        #= Test that observing the mean function shrinks the posterior covariance
           appropriately, but leaves the posterior mean at the prior mean (modulo noise) =# 
        cache = CondCache(kff, μf, X, ones(Float64, N))
        @test cache.α ≈ zeros(Float64, N)

        # Means at data and away from data.
        @test mean(ConditionalMean(cache, μf, kff), X) ≈ mean(μf, X)
        @test mean(ConditionalMean(cache, μf, kff), X′) ≈ mean(μf, X′)

        # Covariance at data should be (close to) zero.
        @test all(abs.(Matrix(cov(ConditionalKernel(cache, kff, kff), X))) .< 1e-10)

        # Posterior covariance of indep. process should be same as prior.
        @test mean(ConditionalMean(cache, μg, kfg), X) ≈ mean(μg, X)
        @test mean(ConditionalMean(cache, μg, kfg), X′) ≈ mean(μg, X′)
        @test cov(ConditionalKernel(cache, kfg, kgg), X) ≈ cov(kgg, X)
        @test cov(ConditionalKernel(cache, kfg, kgg), X′) ≈ cov(kgg, X′)
        @test xcov(ConditionalKernel(cache, kfg, kgg), X, X′) ≈ xcov(kgg, X, X′)
        @test xcov(ConditionalKernel(cache, kfg, kgg), X′, X) ≈ xcov(kgg, X′, X)

        # Should be able to get similar results using a ConditionalCrossKernel.
        @test xcov(ConditionalCrossKernel(cache, kff, kff, kff), X, X) ≈
            Matrix(cov(ConditionalKernel(cache, kff, kff), X))

        # Posterior xcov between two independent processes, conditioned on observations of
        # one of them should be independent.
        @test xcov(ConditionalCrossKernel(cache, kff, kfg, kfg), X, X) ≈ xcov(kfg, X, X)
        @test xcov(ConditionalCrossKernel(cache, kff, kfg, kfg), X, X′) ≈ xcov(kfg, X, X′)
        @test xcov(ConditionalCrossKernel(cache, kfg, kfg, kgg), X′, X′) ≈ xcov(kgg, X′, X′)

        _generic_kernel_tests(ConditionalKernel(cache, kfg, kgg), X, X′)
    end
end
