using Stheno: CondCache, ConditionalMean, ConditionalKernel, ConditionalCrossKernel,
    ConstantMean, ZeroMean, ZeroKernel, ConstantKernel

@testset "conditional" begin

    # Tests for conditioning with independent processes.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 13, 2
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        μf, μg, μh = ConstantMean(randn(rng)), ZeroMean{Float64}(), ConstantMean(randn(rng))
        kff, kgg, khh = EQ(), EQ() + Noise(1e-6), Noise(1e-6)
        kfg, kfh, kgh = EQ(), ZeroKernel{Float64}(), ZeroKernel{Float64}()

        # Run all tests for both scalar and vector input domains.
        for (X0, X1, X2) in [(X0, X1, X2), (x0, x1, x2)]

            # Construct conditioned objects.
            y = unary_obswise(μf, X0) + chol(LazyPDMat(pairwise(kff, X0)))' * randn(rng, N)
            cache = CondCache(kff, μf, X0, y)
            μ′f = ConditionalMean(cache, μf, kff)
            μ′g = ConditionalMean(cache, μg, kfg)
            μ′h = ConditionalMean(cache, μh, kfh)
            k′ff = ConditionalKernel(cache, kff, kff)
            k′gg = ConditionalKernel(cache, kfg, kgg)
            k′hh = ConditionalKernel(cache, kfh, khh)
            k′fg = ConditionalCrossKernel(cache, kff, kfg, kfg)
            k′fh = ConditionalCrossKernel(cache, kff, kfh, kfh)
            k′gh = ConditionalCrossKernel(cache, kfg, kfh, kgh)

            # Run standard consistency tests on all of the above.
            for (n, μ′) in enumerate([μ′f, μ′g, μ′h])
                mean_function_tests(μ′, X0)
                mean_function_tests(μ′, X2)
            end
            for k′ in [k′ff, k′gg, k′hh]
                kernel_tests(k′, X0, X1, X2)
            end
            for k′ in [k′fg, k′fh, k′gh]
                cross_kernel_tests(k′, X0, X1, X2)
            end

            # Test that observing the mean function shrinks the posterior covariance
            # appropriately, but leaves the posterior mean at the prior mean (modulo noise).
            cache = CondCache(kff, μf, X0, unary_obswise(μf, X0))
            @test cache.α ≈ zeros(Float64, N)

            # Posterior covariance at the data should be _exactly_ zero.
            @test all(pairwise(k′ff, X0) .== 0)

            # Posterior for indep. process should be _exactly_ the same as the prior.
            @test unary_obswise(μ′h, X0) == unary_obswise(μh, X0)
            @test unary_obswise(μ′h, X1) == unary_obswise(μh, X1)
            @test unary_obswise(μ′h, X2) == unary_obswise(μh, X2)
            @test binary_obswise(k′hh, X0) == binary_obswise(khh, X0)
            @test binary_obswise(k′hh, X1) == binary_obswise(khh, X1)
            @test binary_obswise(k′hh, X2) == binary_obswise(khh, X2)
            @test binary_obswise(k′hh, X0, X1) == binary_obswise(khh, X0, X1)
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
