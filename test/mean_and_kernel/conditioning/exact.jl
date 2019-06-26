using Stheno: CondCache, CondMean, CondKernel, CondCrossKernel, CustomMean, pw,
    OneMean, ZeroMean, ZeroKernel, OneKernel, pairwise, CondCache, OuterKernel, BinaryKernel
using Stheno: EQ, Exp, Linear, Noise, PerEQ
using LinearAlgebra: cholesky

@testset "exact" begin

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

    # Run all tests for both scalar and vector input domains.
    for (X0, X1, X2) in [(x0, x1, x2)]

        # Construct conditioned objects.
        y = ew(μf, X0) + cholesky(pw(kff, X0)).U' * randn(rng, N)
        cache = CondCache(kff, μf, X0, y, zeros(length(y), length(y)))
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
        cache = CondCache(kff, μf, X0, ew(μf, X0), zeros(length(y), length(y)))
        @test cache.α ≈ zeros(N)

        # Posterior covariance at the data should be fairly close to zero.
        @test maximum(abs.(pw(k′ff, X0))) < 1e-6

        # Posterior for indep. process should be _exactly_ the same as the prior.
        @test ew(μ′h, X0) == ew(μh, X0)
        @test ew(μ′h, X1) == ew(μh, X1)
        @test ew(μ′h, X2) == ew(μh, X2)
        @test ew(k′hh, X0) == ew(khh, X0)
        @test ew(k′hh, X1) == ew(khh, X1)
        @test ew(k′hh, X2) == ew(khh, X2)
        @test ew(k′hh, X0, X1) == ew(khh, X0, X1)
        @test pw(k′hh, X0) == pw(khh, X0)
        @test pw(k′hh, X0, X2) == pw(khh, X0, X2)
        @test pw(k′fh, X0, X2) == pw(kfh, X0, X2)
        @test pw(k′fh, X1, X2) == pw(kfh, X0, X2)
    end
end
