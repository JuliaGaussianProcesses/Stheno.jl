using Stheno: CondCache, CondMean, CondKernel, CondCrossKernel, CustomMean,
    OneMean, ZeroMean, ZeroKernel, OneKernel, pairwise, CondCache, OuterKernel, BinaryKernel
using LinearAlgebra: cholesky

@testset "conditional" begin

    # Tests for conditioning with independent processes.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 13, 8

        # The numerical stability of the result depends _strongly_ on how the data are
        # spaced out - if you sample them randomly and happen to wind up with a couple of
        # points really close to one another, then you get an ill-conditioned covariance
        # matrix and loss of precision.
        x0 = collect(range(-3.0, stop=3.0, length=N))
        x1 = collect(range(-3.5, stop=2.5, length=N))
        x2 = collect(range(-2.5, stop=3.5, length=N′))
        μf, μg, μh = OneMean(), ZeroMean(), OneMean()
        kff = EQ()
        kgg = BinaryKernel(+, EQ(), EQ())
        khh = OuterKernel(CustomMean(x->1e-1 * sum(x)), EQ())
        kfg, kfh, kgh = EQ(), ZeroKernel(), ZeroKernel()

        # # Points in high dimensions probably won't be very close together.
        # X0 = ColsAreObs(randn(rng, D, N))
        # X1 = ColsAreObs(randn(rng, D, N))
        # X2 = ColsAreObs(randn(rng, D, N′))

        # Run all tests for both scalar and vector input domains.
        for (X0, X1, X2) in [(x0, x1, x2)]

            # Construct conditioned objects.
            y = map(μf, X0) + cholesky(pairwise(kff, X0)).U' * randn(rng, N)
            cache = CondCache(kff, μf, X0, y)
            μ′f = CondMean(cache, μf, kff)
            μ′g = CondMean(cache, μg, kfg)
            μ′h = CondMean(cache, μh, kfh)
            k′ff = CondKernel(cache, kff, kff)
            k′gg = CondKernel(cache, kfg, kgg)
            k′hh = CondKernel(cache, kfh, khh)
            k′fg = CondCrossKernel(cache, kff, kfg, kfg)
            k′fh = CondCrossKernel(cache, kff, kfh, kfh)
            k′gh = CondCrossKernel(cache, kfg, kfh, kgh)

            # Run standard consistency tests on all of the above.
            for (n, μ′) in enumerate([μ′f, μ′g, μ′h])
                differentiable_mean_function_tests(rng, μ′, X0)
                differentiable_mean_function_tests(rng, μ′, X2)
            end
            for (n, k′) in enumerate([k′ff, k′gg, k′hh])
                differentiable_kernel_tests(rng, k′, X0, X1, X2)
            end
            for k′ in [k′fg, k′fh, k′gh]
                differentiable_cross_kernel_tests(rng, k′, X0, X1, X2)
            end

            # Test that observing the mean function shrinks the posterior covariance
            # appropriately, but leaves the posterior mean at the prior mean (modulo noise).
            cache = CondCache(kff, μf, X0, map(μf, X0))
            @test cache.α ≈ zeros(N)

            # Posterior covariance at the data should be fairly close to zero.
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
        end
    end
end
