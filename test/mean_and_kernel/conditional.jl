using Stheno: CondCache, ConditionalMean, ConditionalKernel, ConditionalCrossKernel,
    ConstantMean, ZeroMean, ZeroKernel, ConstantKernel, pairwise, chol

@testset "conditional" begin

    # Tests for conditioning with independent processes.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 13, 2
        X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        μf, μg, μh = ConstantMean(randn(rng)), ZeroMean{Float64}(), ConstantMean(randn(rng))
        kff, kgg, khh = EQ(), EQ() + Noise(1e-6), Noise(1e-6)
        kfg, kfh, kgh = EQ(), ZeroKernel{Float64}(), ZeroKernel{Float64}()

        # Run all tests for both scalar and vector input domains.
        for (X0, X1, X2) in [(X0, X1, X2), (x0, x1, x2)]

            # Construct conditioned objects.
            y = map(μf, X0) + chol(LazyPDMat(pairwise(kff, X0)))' * randn(rng, N)
            cache = CondCache(kff, μf, X0, y)
            kffX = LhsFiniteCrossKernel(kff, X0)
            kfgX = LhsFiniteCrossKernel(kfg, X0)
            kfhX = LhsFiniteCrossKernel(kfh, X0)
            μ′f = ConditionalMean(cache, μf, kffX)
            μ′g = ConditionalMean(cache, μg, kfgX)
            μ′h = ConditionalMean(cache, μh, kfhX)
            k′ff = ConditionalKernel(cache, kffX, kff)
            k′gg = ConditionalKernel(cache, kfgX, kgg)
            k′hh = ConditionalKernel(cache, kfhX, khh)
            k′fg = ConditionalCrossKernel(cache, kffX, kfgX, kfg)
            k′fh = ConditionalCrossKernel(cache, kffX, kfhX, kfh)
            k′gh = ConditionalCrossKernel(cache, kfgX, kfhX, kgh)

            # Run standard consistency tests on all of the above.
            for (n, μ′) in enumerate([μ′f, μ′g, μ′h])
                mean_function_tests(μ′, X0)
                mean_function_tests(μ′, X2)
                mean_function_tests(μ′, BlockData([X0, X2]))
            end
            for (n, k′) in enumerate([k′ff, k′gg, k′hh])
                kernel_tests(k′, X0, X1, X2, 1e-6)
                kernel_tests(k′, BlockData([X0]), BlockData([X1]), BlockData([X2]), 1e-6)
                kernel_tests(k′, BlockData([X0]), BlockData([X1]), BlockData([X2, X1]), 1e-6)
            end
            for k′ in [k′fg, k′fh, k′gh]
                cross_kernel_tests(k′, X0, X1, X2)
                cross_kernel_tests(k′, BlockData([X0]), BlockData([X1]), BlockData([X2]))
                cross_kernel_tests(k′, BlockData([X0]), BlockData([X1]), BlockData([X2, X1]))
            end

            # Test that observing the mean function shrinks the posterior covariance
            # appropriately, but leaves the posterior mean at the prior mean (modulo noise).
            cache = CondCache(kff, μf, X0, map(μf, X0))
            @test cache.α ≈ zeros(Float64, N)

            # Posterior covariance at the data should be _exactly_ zero.
            @test maximum(abs.(pairwise(k′ff, X0))) < 1e-6

            # Posterior for indep. process should be _exactly_ the same as the prior.
            @test map(μ′h, X0) == map(μh, X0)
            @test map(μ′h, X1) == map(μh, X1)
            @test map(μ′h, X2) == map(μh, X2)
            @test map(k′hh, X0) == map(khh, X0)
            @test map(k′hh, X1) == map(khh, X1)
            @test map(k′hh, X2) == map(khh, X2)
            @test map(k′hh, X0, X1) == map(khh, X0, X1)
            @test pairwise(k′hh, X0) == pairwise(khh, X0)
            @test pairwise(k′hh, X0, X2) == pairwise(khh, X0, X2)
            @test pairwise(k′fh, X0, X2) == pairwise(kfh, X0, X2)
            @test pairwise(k′fh, X1, X2) == pairwise(kfh, X0, X2)

            # Ensure that printing doesn't error.
            show(IOBuffer(), μ′f)
            show(IOBuffer(), k′ff)
            show(IOBuffer(), k′fg)

            # Ensure lengths are propagated correctly.
            @test length(μ′f) == Inf
            @test size(k′ff, 1) == Inf && size(k′ff, 2) == Inf
            @test size(k′fg, 1) == Inf && size(k′fg, 2) == Inf
        end
    end


    #     # Should be able to get similar results using a ConditionalCrossKernel.
    #     @test xcov(ConditionalCrossKernel(cache, kff, kff, kff), X, X) ≈
    #         Matrix(cov(ConditionalKernel(cache, kff, kff), X))
end
